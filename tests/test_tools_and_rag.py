"""Increment 2 verification: cost ledger, hybrid RAG, and deterministic tools."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from db.seed_db import seed_db
from src.cost_ledger import (
    ANTHROPIC_PRICING_USD_PER_1M,
    CostLedger,
    calculate_call_cost_usd,
)
from src.hybrid_rag import (
    bm25_only_search,
    ensure_embeddings,
    hybrid_rrf_search,
    local_embed_text,
)
from src.tools import (
    V1_BASIC_TOOLS,
    V2_POKAYOKE_TOOLS,
    evaluate_deal_economics,
    execute_tool,
    query_historical_benchmarks,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DB = PROJECT_ROOT / "db" / "test_inc2_deal_desk.db"


@pytest.fixture(scope="module")
def seeded_db_path():
    if TEST_DB.exists():
        TEST_DB.unlink()
    seed_db(str(TEST_DB))
    ensure_embeddings(str(TEST_DB))
    yield str(TEST_DB)
    if TEST_DB.exists():
        TEST_DB.unlink()


# ---------------------------------------------------------------------------
# Cost ledger
# ---------------------------------------------------------------------------
def test_calculate_call_cost_exact_usd_and_cache_multipliers():
    # Sonnet 4.6: cache_write = 1.25× input, cache_read = 0.10× input
    rates = ANTHROPIC_PRICING_USD_PER_1M["claude-sonnet-4-6"]
    assert rates["cache_write"] / rates["input"] == pytest.approx(1.25)
    assert rates["cache_read"] / rates["input"] == pytest.approx(0.10)

    # Sonnet 5.5: cache_read = 0.05× input
    rates55 = ANTHROPIC_PRICING_USD_PER_1M["claude-sonnet-5-5"]
    assert rates55["cache_write"] / rates55["input"] == pytest.approx(1.25)
    assert rates55["cache_read"] / rates55["input"] == pytest.approx(0.05)

    # Manual exact cost: 1M in / 1M out / 1M cache_write / 1M cache_read on sonnet-4-6
    cost = calculate_call_cost_usd(
        "claude-sonnet-4-6",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_creation_tokens=1_000_000,
        cache_read_tokens=1_000_000,
    )
    assert cost == pytest.approx(3.0 + 15.0 + 3.75 + 0.30)
    assert cost == round(cost, 6)

    # Alias pricing parity
    assert calculate_call_cost_usd("claude-haiku-4-5", 1000, 500) == calculate_call_cost_usd(
        "claude-haiku-4-5-20251001", 1000, 500
    )
    assert calculate_call_cost_usd("claude-opus-4-5", 1000, 500) == calculate_call_cost_usd(
        "claude-opus-4-6", 1000, 500
    )

    with pytest.raises(ValueError, match="Unknown model"):
        calculate_call_cost_usd("claude-unknown-99", 10, 10)


def test_cost_ledger_persists_and_summarizes_percentiles(tmp_path):
    db_path = str(tmp_path / "ledger.db")
    ledger = CostLedger(db_path)

    # 5 runs with increasing cost/latency for stable percentiles
    for i, (inp, out, lat) in enumerate(
        [
            (10_000, 2_000, 100.0),
            (20_000, 4_000, 200.0),
            (30_000, 6_000, 300.0),
            (40_000, 8_000, 400.0),
            (50_000, 10_000, 500.0),
        ],
        start=1,
    ):
        run_id = f"run-{i:02d}"
        recorded = ledger.record_call(
            run_id=run_id,
            experiment_version="v2-test",
            case_id="GE-001",
            step_name="agent_turn",
            model="claude-sonnet-4-6",
            input_tokens=inp,
            output_tokens=out,
            cache_creation_tokens=1_000 if i == 1 else 0,
            cache_read_tokens=5_000 if i > 1 else 0,
            latency_ms=lat,
        )
        assert "cost_usd" in recorded
        assert recorded["cost_usd"] == calculate_call_cost_usd(
            "claude-sonnet-4-6",
            inp,
            out,
            recorded["cache_creation_tokens"],
            recorded["cache_read_tokens"],
        )
        assert "T" in recorded["created_at"]  # ISO-8601 UTC-ish

        summary = ledger.get_run_summary(run_id)
        assert summary["calls"] == 1
        assert summary["total_cost_usd"] == recorded["cost_usd"]

    exp = ledger.get_experiment_summary("v2-test")
    assert exp["run_count"] == 5
    assert exp["p10_cost_usd"] <= exp["p50_cost_usd"] <= exp["p90_cost_usd"]
    assert exp["p10_latency_ms"] <= exp["p50_latency_ms"] <= exp["p90_latency_ms"]
    assert exp["mean_cost_usd"] > 0
    assert exp["avg_cache_hit_rate_pct"] >= 0

    # Persist check
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM cost_ledger").fetchone()[0]
    conn.close()
    assert count == 5


# ---------------------------------------------------------------------------
# evaluate_deal_economics ↔ golden ground truth
# ---------------------------------------------------------------------------
def test_evaluate_deal_economics_matches_all_20_golden_cases(seeded_db_path):
    conn = sqlite3.connect(seeded_db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM golden_eval_cases ORDER BY case_id"
    ).fetchall()
    conn.close()
    assert len(rows) == 20

    matches = 0
    for row in rows:
        payload = json.loads(row["deal_payload_json"])
        risk_flags = json.loads(row["expected_risk_flags_json"])
        result = evaluate_deal_economics(
            annual_contract_value=payload["annual_contract_value"],
            term_months=payload["term_months"],
            discount_pct=payload["discount_pct"],
            payment_terms=payload["payment_terms"],
            custom_onboarding_hours=payload["services_hours"],
            proposed_legal_flags=risk_flags,
        )
        assert result["gross_margin_pct"] == pytest.approx(
            row["expected_gross_margin_pct"]
        )
        assert result["required_approval_lane"] == row["expected_approval_lane"]
        assert result["margin_floor_pct"] == 72.0
        matches += 1
    assert matches == 20


def test_evaluate_deal_economics_normalizes_percentage_discount():
    as_decimal = evaluate_deal_economics(
        annual_contract_value=100_000,
        term_months=12,
        discount_pct=0.20,
        payment_terms="Net-30",
        custom_onboarding_hours=10,
    )
    as_percent = evaluate_deal_economics(
        annual_contract_value=100_000,
        term_months=12,
        discount_pct=20.0,
        payment_terms="Net-30",
        custom_onboarding_hours=10,
    )
    assert as_percent["discount_pct"] == pytest.approx(0.20)
    assert "normalized_discount_note" in as_percent
    assert as_percent["gross_margin_pct"] == as_decimal["gross_margin_pct"]
    assert as_percent["required_approval_lane"] == as_decimal["required_approval_lane"]


# ---------------------------------------------------------------------------
# Historical benchmarks
# ---------------------------------------------------------------------------
def test_query_historical_benchmarks_both_segments(seeded_db_path):
    for segment in ("Mid-Market", "Enterprise"):
        for discount in (0.10, 0.20, 0.30):
            result = query_historical_benchmarks(
                segment=segment,
                discount_pct=discount,
                db_path=seeded_db_path,
            )
            assert result["segment"] == segment
            assert result["sample_size"] > 0
            assert 0.0 <= result["win_rate_pct"] <= 100.0
            assert result["avg_gross_margin_pct"] != 0.0 or result["sample_size"] > 0
            assert result["discount_band"] in {"0-15%", "15-25%", ">25%"}


# ---------------------------------------------------------------------------
# Hybrid RAG
# ---------------------------------------------------------------------------
def test_local_embed_query_matches_stored_vectors(seeded_db_path):
    conn = sqlite3.connect(seeded_db_path)
    row = conn.execute(
        "SELECT content, embedding_json FROM rag_chunks WHERE chunk_id = 'CUAD-001'"
    ).fetchone()
    conn.close()
    stored = json.loads(row[1])
    recomputed = local_embed_text(row[0])
    assert stored == recomputed


def test_bm25_and_hybrid_retrieve_mfc_exact_and_paraphrase(seeded_db_path):
    exact = "Most Favored Customer"
    paraphrase = "customer demanding automatic lowest price parity"

    bm25_exact = bm25_only_search(exact, top_k=4, db_path=seeded_db_path)
    hybrid_exact = hybrid_rrf_search(exact, top_k=4, db_path=seeded_db_path)
    assert bm25_exact and hybrid_exact
    assert any(r["chunk_id"] == "CUAD-001" for r in bm25_exact)
    assert any(r["chunk_id"] == "CUAD-001" for r in hybrid_exact)
    for r in bm25_exact + hybrid_exact:
        assert {"chunk_id", "source_type", "category", "title", "content", "score"} <= set(
            r
        )

    bm25_para = bm25_only_search(paraphrase, top_k=4, db_path=seeded_db_path)
    hybrid_para = hybrid_rrf_search(paraphrase, top_k=4, db_path=seeded_db_path)
    assert bm25_para and hybrid_para

    def _mfc_relevant(results: list[dict]) -> bool:
        for r in results:
            blob = f"{r['title']} {r['content']} {r['category']}".lower()
            if (
                r["chunk_id"] == "CUAD-001"
                or "favored" in blob
                or "mfc" in blob
                or "pricing parity" in blob
                or "more favorable" in blob
            ):
                return True
        return False

    assert _mfc_relevant(hybrid_para), f"hybrid paraphrase miss: {hybrid_para}"
    # BM25 or hybrid should surface policy/clause content for pricing parity semantics
    assert _mfc_relevant(bm25_para) or _mfc_relevant(hybrid_para)


def test_search_filters_by_source_type(seeded_db_path):
    policy = hybrid_rrf_search(
        "gross margin hard floor 72",
        top_k=3,
        source_type="DEAL_DESK_POLICY",
        db_path=seeded_db_path,
    )
    assert policy
    assert all(r["source_type"] == "DEAL_DESK_POLICY" for r in policy)

    clauses = bm25_only_search(
        "uncapped liability indemnification",
        top_k=3,
        source_type="CUAD_CLAUSE",
        db_path=seeded_db_path,
    )
    assert clauses
    assert all(r["source_type"] == "CUAD_CLAUSE" for r in clauses)


# ---------------------------------------------------------------------------
# execute_tool error handling + schema registries
# ---------------------------------------------------------------------------
def test_execute_tool_returns_structured_errors(seeded_db_path):
    bad_payment = execute_tool(
        "evaluate_deal_economics",
        {
            "annual_contract_value": 100000,
            "term_months": 12,
            "discount_pct": 0.1,
            "payment_terms": "Net-45",
        },
        db_path=seeded_db_path,
    )
    payload = json.loads(bad_payment)
    assert "error" in payload
    assert "Net-45" in payload["error"] or "payment_terms" in payload["error"]

    unknown = json.loads(execute_tool("not_a_real_tool", {}, db_path=seeded_db_path))
    assert "error" in unknown

    bad_segment = json.loads(
        execute_tool(
            "query_historical_benchmarks",
            {"segment": "SMB", "discount_pct": 0.1},
            db_path=seeded_db_path,
        )
    )
    assert "error" in bad_segment

    # Valid path still works
    ok = json.loads(
        execute_tool(
            "evaluate_deal_economics",
            {
                "annual_contract_value": 95000,
                "term_months": 12,
                "discount_pct": 0.10,
                "payment_terms": "Net-30",
                "custom_onboarding_hours": 10,
            },
            db_path=seeded_db_path,
        )
    )
    assert "error" not in ok
    assert ok["required_approval_lane"] == "Auto-Approve"


def test_dual_aci_schema_registries():
    assert len(V1_BASIC_TOOLS) == 3
    assert len(V2_POKAYOKE_TOOLS) == 3
    v1_names = {t["name"] for t in V1_BASIC_TOOLS}
    v2_names = {t["name"] for t in V2_POKAYOKE_TOOLS}
    assert v1_names == v2_names

    v2_econ = next(t for t in V2_POKAYOKE_TOOLS if t["name"] == "evaluate_deal_economics")
    payment_enum = v2_econ["input_schema"]["properties"]["payment_terms"]["enum"]
    assert payment_enum == ["Net-30", "Net-60", "Net-90"]
    assert "0.20" in v2_econ["description"] or "DECIMAL" in v2_econ["description"]

    v1_econ = next(t for t in V1_BASIC_TOOLS if t["name"] == "evaluate_deal_economics")
    assert "enum" not in v1_econ["input_schema"]["properties"]["payment_terms"]
