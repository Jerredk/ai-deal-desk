"""Deterministic deal-desk tools and dual Anthropic ACI schema registries."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.deal_economics import (
    HIGH_RISK_LEGAL_FLAGS,
    MARGIN_FLOOR_PCT,
    VALID_PAYMENT_TERMS,
    VALID_SEGMENTS,
    build_triggered_flags,
    compute_gross_margin_components,
    determine_approval_tier,
    normalize_discount_pct,
)
from src.hybrid_rag import bm25_only_search, hybrid_rrf_search

DEFAULT_DB_PATH = "db/deal_desk.db"
CANONICAL_LEGAL_FLAGS = sorted(HIGH_RISK_LEGAL_FLAGS) + [
    "MARGIN_BELOW_FLOOR",
    "NET_90_WORKING_CAPITAL",
    "PRICE_CAP_AUTO_RENEWAL",
]


# ---------------------------------------------------------------------------
# Pydantic input models
# ---------------------------------------------------------------------------
class EvaluateDealEconomicsInput(BaseModel):
    annual_contract_value: float = Field(..., gt=0)
    term_months: int = Field(..., gt=0)
    discount_pct: float
    payment_terms: str
    custom_onboarding_hours: float = Field(0.0, ge=0)
    proposed_legal_flags: list[str] | None = None

    @field_validator("payment_terms")
    @classmethod
    def _validate_payment_terms(cls, v: str) -> str:
        if v not in VALID_PAYMENT_TERMS:
            raise ValueError(
                f"payment_terms must be one of {sorted(VALID_PAYMENT_TERMS)}; got {v!r}"
            )
        return v


class QueryHistoricalBenchmarksInput(BaseModel):
    segment: str
    discount_pct: float
    db_path: str = DEFAULT_DB_PATH

    @field_validator("segment")
    @classmethod
    def _validate_segment(cls, v: str) -> str:
        if v not in VALID_SEGMENTS:
            raise ValueError(
                f"segment must be one of {sorted(VALID_SEGMENTS)}; got {v!r}"
            )
        return v


class SearchPoliciesInput(BaseModel):
    query: str = Field(..., min_length=1)
    source_type: Literal["ALL", "DEAL_DESK_POLICY", "CUAD_CLAUSE"] = "ALL"
    top_k: int = Field(4, ge=1, le=16)
    mode: Literal["hybrid", "bm25"] = "hybrid"
    db_path: str = DEFAULT_DB_PATH


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def evaluate_deal_economics(
    annual_contract_value: float,
    term_months: int,
    discount_pct: float,
    payment_terms: str,
    custom_onboarding_hours: float = 0.0,
    proposed_legal_flags: list[str] | None = None,
) -> dict:
    """Compute margin components, approval tier, and triggered risk flags."""
    payload = EvaluateDealEconomicsInput(
        annual_contract_value=annual_contract_value,
        term_months=term_months,
        discount_pct=discount_pct,
        payment_terms=payment_terms,
        custom_onboarding_hours=custom_onboarding_hours,
        proposed_legal_flags=proposed_legal_flags,
    )
    normalized_discount, normalized_note = normalize_discount_pct(payload.discount_pct)
    comps = compute_gross_margin_components(
        payload.annual_contract_value,
        normalized_discount,
        payload.payment_terms,
        payload.custom_onboarding_hours,
    )
    flags = build_triggered_flags(
        comps["gross_margin_pct"],
        payload.payment_terms,
        payload.proposed_legal_flags,
    )
    # Tier routing uses high-risk legal flags + economics (same as seed ground truth)
    legal_for_tier = [
        f for f in (payload.proposed_legal_flags or []) if f in HIGH_RISK_LEGAL_FLAGS
    ]
    tier = determine_approval_tier(
        normalized_discount,
        comps["gross_margin_pct"],
        payload.payment_terms,
        legal_for_tier,
    )
    margin_buffer_bps = round((comps["gross_margin_pct"] - MARGIN_FLOOR_PCT) * 100.0, 2)

    result: dict[str, Any] = {
        "list_arr": round(comps["list_arr"], 6),
        "annual_contract_value": payload.annual_contract_value,
        "term_months": payload.term_months,
        "discount_pct": normalized_discount,
        "payment_terms": payload.payment_terms,
        "base_cogs": round(comps["base_cogs"], 6),
        "services_cogs": round(comps["services_cogs"], 6),
        "working_capital_penalty": round(comps["working_capital_penalty"], 6),
        "gross_margin_pct": comps["gross_margin_pct"],
        "margin_floor_pct": MARGIN_FLOOR_PCT,
        "margin_buffer_bps": margin_buffer_bps,
        "required_approval_tier": tier,
        "triggered_flags": flags,
    }
    if normalized_note:
        result["normalized_discount_note"] = normalized_note
    return result


def _discount_band(discount_pct: float) -> tuple[str, str]:
    """Return (band_label, SQL WHERE fragment placeholders handled by caller)."""
    d, _ = normalize_discount_pct(discount_pct)
    if d <= 0.15:
        return "0-15%", "discount_pct <= 0.15"
    if d <= 0.25:
        return "15-25%", "discount_pct > 0.15 AND discount_pct <= 0.25"
    return ">25%", "discount_pct > 0.25"


def query_historical_benchmarks(
    segment: str,
    discount_pct: float,
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    """Aggregate win-rate / margin / cycle benchmarks from historical_deals."""
    payload = QueryHistoricalBenchmarksInput(
        segment=segment,
        discount_pct=discount_pct,
        db_path=db_path,
    )
    band_label, band_sql = _discount_band(payload.discount_pct)
    conn = sqlite3.connect(payload.db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"""
        SELECT outcome, gross_margin_pct, sales_cycle_days, has_nonstandard_legal
        FROM historical_deals
        WHERE segment = ? AND ({band_sql})
        """,
        (payload.segment,),
    ).fetchall()
    conn.close()

    sample_size = len(rows)
    if sample_size == 0:
        return {
            "segment": payload.segment,
            "discount_band": band_label,
            "sample_size": 0,
            "win_rate_pct": 0.0,
            "avg_gross_margin_pct": 0.0,
            "avg_cycle_days_standard": 0.0,
            "avg_cycle_days_with_legal_redlines": 0.0,
        }

    wins = sum(1 for r in rows if r["outcome"] == "Closed-Won")
    standard = [r for r in rows if int(r["has_nonstandard_legal"]) == 0]
    legal = [r for r in rows if int(r["has_nonstandard_legal"]) == 1]

    def _avg_cycle(subset: list[sqlite3.Row]) -> float:
        if not subset:
            return 0.0
        return round(sum(r["sales_cycle_days"] for r in subset) / len(subset), 2)

    return {
        "segment": payload.segment,
        "discount_band": band_label,
        "sample_size": sample_size,
        "win_rate_pct": round(100.0 * wins / sample_size, 2),
        "avg_gross_margin_pct": round(
            sum(r["gross_margin_pct"] for r in rows) / sample_size, 2
        ),
        "avg_cycle_days_standard": _avg_cycle(standard),
        "avg_cycle_days_with_legal_redlines": _avg_cycle(legal),
    }


def search_deal_desk_policies_and_clauses(
    query: str,
    source_type: str = "ALL",
    top_k: int = 4,
    mode: str = "hybrid",
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    """Dispatch to BM25 or Hybrid RRF retrieval over Deal-Desk policies / CUAD clauses."""
    payload = SearchPoliciesInput(
        query=query,
        source_type=source_type,  # type: ignore[arg-type]
        top_k=top_k,
        mode=mode,  # type: ignore[arg-type]
        db_path=db_path,
    )
    st = None if payload.source_type == "ALL" else payload.source_type
    if payload.mode == "bm25":
        results = bm25_only_search(
            payload.query, top_k=payload.top_k, source_type=st, db_path=payload.db_path
        )
    else:
        results = hybrid_rrf_search(
            payload.query, top_k=payload.top_k, source_type=st, db_path=payload.db_path
        )
    return {
        "query": payload.query,
        "source_type": payload.source_type,
        "mode": payload.mode,
        "top_k": payload.top_k,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Anthropic tool-definition registries
# ---------------------------------------------------------------------------
V1_BASIC_TOOLS: list[dict[str, Any]] = [
    {
        "name": "evaluate_deal_economics",
        "description": "Evaluate deal economics and approval tier.",
        "input_schema": {
            "type": "object",
            "properties": {
                "annual_contract_value": {"type": "number"},
                "term_months": {"type": "integer"},
                "discount_pct": {"type": "number"},
                "payment_terms": {"type": "string"},
                "custom_onboarding_hours": {"type": "number"},
                "proposed_legal_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "annual_contract_value",
                "term_months",
                "discount_pct",
                "payment_terms",
            ],
        },
    },
    {
        "name": "query_historical_benchmarks",
        "description": "Query historical deal benchmarks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "segment": {"type": "string"},
                "discount_pct": {"type": "number"},
            },
            "required": ["segment", "discount_pct"],
        },
    },
    {
        "name": "search_deal_desk_policies_and_clauses",
        "description": "Search policies and contract clauses.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "source_type": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
]


V2_POKAYOKE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "evaluate_deal_economics",
        "description": (
            "ALWAYS call this first for any deal-desk request. Computes list ARR, "
            "base hosting/support COGS (18% of list ARR), services COGS ($150/hr "
            "unbilled onboarding), Net-60/90 working-capital penalties, effective "
            "gross margin %, margin buffer vs the 72.0% hard floor, required approval "
            "tier (Auto-Approve / VP-Sales / CFO-And-Legal), and triggered risk flags. "
            "Pass discount_pct as a DECIMAL fraction (0.20 for 20%, NOT 20). "
            "Pass any detected high-risk legal flag codes in proposed_legal_flags."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "annual_contract_value": {
                    "type": "number",
                    "description": "Sold ACV in USD (invoice amount after discount).",
                },
                "term_months": {
                    "type": "integer",
                    "description": "Contract term length in months (e.g., 12, 24, 36).",
                },
                "discount_pct": {
                    "type": "number",
                    "description": (
                        "Discount as a DECIMAL fraction of list ARR. "
                        "Example: 20% off → 0.20. Values > 1.0 are auto-normalized."
                    ),
                },
                "payment_terms": {
                    "type": "string",
                    "enum": ["Net-30", "Net-60", "Net-90"],
                    "description": "Customer payment terms. Net-90 always escalates to CFO-And-Legal.",
                },
                "custom_onboarding_hours": {
                    "type": "number",
                    "description": (
                        "Unbilled/discounted implementation hours. Costed at $150/hr "
                        "against deal gross margin. Default 0."
                    ),
                },
                "proposed_legal_flags": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": CANONICAL_LEGAL_FLAGS,
                    },
                    "description": (
                        "Canonical legal/commercial risk flag codes detected in the "
                        "redline text, e.g. MFN_CLAUSE, UNCAPPED_LIABILITY, "
                        "TERMINATION_CONVENIENCE_30D, NON_STANDARD_AUDIT."
                    ),
                },
            },
            "required": [
                "annual_contract_value",
                "term_months",
                "discount_pct",
                "payment_terms",
            ],
        },
    },
    {
        "name": "query_historical_benchmarks",
        "description": (
            "Call after evaluate_deal_economics when you need comparable win-rate, "
            "margin, and sales-cycle context for the same segment and discount band "
            "(0-15%, 15-25%, or >25%). Useful evidence for VP/CFO escalation memos."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "segment": {
                    "type": "string",
                    "enum": ["Mid-Market", "Enterprise"],
                    "description": "Customer segment for historical filtering.",
                },
                "discount_pct": {
                    "type": "number",
                    "description": (
                        "Deal discount as DECIMAL fraction (0.20 for 20%) used to "
                        "select the discount band."
                    ),
                },
            },
            "required": ["segment", "discount_pct"],
        },
    },
    {
        "name": "search_deal_desk_policies_and_clauses",
        "description": (
            "Call when you need Deal-Desk policy language (discount bands, 72% margin "
            "floor, working-capital rules, escalation matrix) or to classify commercial "
            "contract redlines against CUAD-style clauses (MFN, uncapped liability, "
            "30-day termination for convenience, on-site audit). Use source_type to "
            "narrow the corpus; default mode is hybrid RRF retrieval."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query or redline excerpt.",
                },
                "source_type": {
                    "type": "string",
                    "enum": ["ALL", "DEAL_DESK_POLICY", "CUAD_CLAUSE"],
                    "description": "Corpus filter. ALL searches both policies and clauses.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to return (1-16). Default 4.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["hybrid", "bm25"],
                    "description": (
                        "Retrieval mode. Use 'hybrid' for v2/v3 (BM25+dense RRF); "
                        "'bm25' for v1 lexical-only baseline."
                    ),
                },
            },
            "required": ["query"],
        },
    },
]


TOOL_DISPATCH = {
    "evaluate_deal_economics": evaluate_deal_economics,
    "query_historical_benchmarks": query_historical_benchmarks,
    "search_deal_desk_policies_and_clauses": search_deal_desk_policies_and_clauses,
}


def execute_tool(
    tool_name: str,
    tool_input: dict,
    db_path: str = DEFAULT_DB_PATH,
    rag_mode: str = "hybrid",
) -> str:
    """Execute a tool by name; return JSON string (including structured errors)."""
    try:
        if tool_name not in TOOL_DISPATCH:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        args = dict(tool_input)
        if tool_name == "query_historical_benchmarks":
            args.setdefault("db_path", db_path)
        elif tool_name == "search_deal_desk_policies_and_clauses":
            args.setdefault("db_path", db_path)
            if "mode" not in args:
                args["mode"] = rag_mode

        result = TOOL_DISPATCH[tool_name](**args)
        return json.dumps(result)
    except (ValidationError, ValueError, TypeError, FileNotFoundError, KeyError) as exc:
        return json.dumps({"error": str(exc)})
