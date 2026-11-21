"""Deterministic financial rules and seed data for the Deal-Desk Agent."""

from __future__ import annotations

import json
import logging
import random
import sqlite3
import sys
from pathlib import Path

# Allow `python db/seed_db.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from db.schema import init_db
from src.deal_economics import (
    BASE_COGS_PCT,
    HIGH_RISK_LEGAL_FLAGS,
    SERVICES_HOURLY_RATE,
    WORKING_CAPITAL_PENALTY,
    compute_gross_margin_pct,
    determine_approval_lane,
)
from src.hybrid_rag import LOCAL_EMBEDDING_DIM, local_embed_text

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Re-export SSOT symbols for Increment 1 tests / callers
__all__ = [
    "BASE_COGS_PCT",
    "SERVICES_HOURLY_RATE",
    "WORKING_CAPITAL_PENALTY",
    "HIGH_RISK_LEGAL_FLAGS",
    "DEFAULT_DB_PATH",
    "compute_gross_margin_pct",
    "determine_approval_lane",
    "seed_db",
]

DEFAULT_DB_PATH = "db/deal_desk.db"


# ---------------------------------------------------------------------------
# Historical deals (250 deterministic records)
# ---------------------------------------------------------------------------
ACCOUNT_PREFIXES = [
    "Apex", "Nimbus", "Vertex", "Cascade", "Meridian", "Pinnacle", "Horizon",
    "Summit", "Atlas", "Beacon", "Cipher", "Delta", "Echo", "Forge", "Granite",
    "Harbor", "Ion", "Jade", "Keystone", "Lumen", "Monarch", "Nova", "Orbit",
    "Prism", "Quartz",
]
ACCOUNT_SUFFIXES = [
    "Systems", "Dynamics", "Analytics", "Labs", "Partners", "Group", "Corp",
    "Solutions", "Networks", "Industries",
]


def _account_name(rng: random.Random, idx: int) -> str:
    return (
        f"{ACCOUNT_PREFIXES[idx % len(ACCOUNT_PREFIXES)]} "
        f"{ACCOUNT_SUFFIXES[idx % len(ACCOUNT_SUFFIXES)]} "
        f"{1000 + idx}"
    )


def generate_historical_deals(n: int = 250) -> list[dict]:
    """Generate n deterministic historical deals with realistic win-rate decay."""
    rng = random.Random(42)
    deals: list[dict] = []

    for i in range(n):
        deal_id = f"HD-{i + 1:04d}"
        segment = "Enterprise" if rng.random() < 0.45 else "Mid-Market"

        if segment == "Enterprise":
            acv = round(rng.uniform(150_000, 900_000), 2)
            term_months = rng.choice([12, 24, 36])
        else:
            acv = round(rng.uniform(40_000, 180_000), 2)
            term_months = rng.choice([12, 24])

        # Discount distribution skewed toward moderate discounts
        discount_roll = rng.random()
        if discount_roll < 0.35:
            discount_pct = round(rng.uniform(0.05, 0.15), 4)
        elif discount_roll < 0.70:
            discount_pct = round(rng.uniform(0.15, 0.25), 4)
        elif discount_roll < 0.90:
            discount_pct = round(rng.uniform(0.25, 0.35), 4)
        else:
            discount_pct = round(rng.uniform(0.35, 0.45), 4)

        payment_terms = rng.choices(
            ["Net-30", "Net-60", "Net-90"],
            weights=[0.55, 0.30, 0.15],
        )[0]

        onboarding_hours = round(rng.uniform(8, 120), 1)
        custom_onboarding_cost = round(onboarding_hours * SERVICES_HOURLY_RATE, 2)
        has_nonstandard_legal = 1 if rng.random() < 0.22 else 0

        gross_margin_pct = compute_gross_margin_pct(
            acv, discount_pct, payment_terms, onboarding_hours
        )

        # Win-rate decays with extreme discounts; legal redlines lengthen cycle
        base_win_prob = 0.72
        if discount_pct > 0.35:
            base_win_prob -= 0.28
        elif discount_pct > 0.25:
            base_win_prob -= 0.15
        elif discount_pct > 0.20:
            base_win_prob -= 0.06

        if payment_terms == "Net-90":
            base_win_prob -= 0.05
        if has_nonstandard_legal:
            base_win_prob -= 0.08
        if gross_margin_pct < 72.0:
            base_win_prob -= 0.10

        base_win_prob = max(0.12, min(0.88, base_win_prob))
        outcome = "Closed-Won" if rng.random() < base_win_prob else "Closed-Lost"

        cycle_base = 45 if segment == "Mid-Market" else 75
        cycle_days = cycle_base + int(discount_pct * 80)
        if has_nonstandard_legal:
            cycle_days += rng.randint(25, 55)
        if payment_terms == "Net-90":
            cycle_days += rng.randint(5, 20)
        cycle_days += rng.randint(-8, 12)
        cycle_days = max(21, cycle_days)

        deals.append(
            {
                "deal_id": deal_id,
                "account_name": _account_name(rng, i),
                "segment": segment,
                "annual_contract_value": acv,
                "term_months": term_months,
                "discount_pct": discount_pct,
                "payment_terms": payment_terms,
                "custom_onboarding_cost": custom_onboarding_cost,
                "gross_margin_pct": gross_margin_pct,
                "has_nonstandard_legal": has_nonstandard_legal,
                "outcome": outcome,
                "sales_cycle_days": cycle_days,
            }
        )

    return deals


# ---------------------------------------------------------------------------
# RAG chunks — 6 Deal-Desk policies + 10 CUAD-style clauses
# ---------------------------------------------------------------------------
def build_rag_chunks() -> list[dict]:
    """Return 16 structured policy + CUAD clause chunks."""
    policies = [
        {
            "chunk_id": "POL-001",
            "source_type": "DEAL_DESK_POLICY",
            "category": "discount_bands",
            "title": "Standard Discount Authority Bands",
            "content": (
                "Deal Desk Discount Authority Policy v3.2. AE-level auto-approval is "
                "permitted for discounts up to and including 15% of list ARR when all "
                "other commercial terms are standard (Net-30, no high-risk legal "
                "redlines, and projected gross margin >= 75%). Discounts above 15% and "
                "up to 25% require VP Sales countersignature. Discounts exceeding 25% "
                "of list ARR must be escalated to CFO and Legal for joint approval. "
                "Discount is measured against current published list ARR for the SKU "
                "bundle; multi-year prepay does not expand AE discount authority."
            ),
        },
        {
            "chunk_id": "POL-002",
            "source_type": "DEAL_DESK_POLICY",
            "category": "margin_floor",
            "title": "Gross Margin Hard Floor — 72.0%",
            "content": (
                "All SaaS deals must maintain a projected gross margin of at least "
                "72.0% after adjusting for hosting/support COGS (18% of list ARR), "
                "unbilled custom onboarding services, and working-capital penalties "
                "for extended payment terms. Deals projecting gross margin below "
                "72.0% are blocked from AE or VP auto-paths and require CFO-And-Legal "
                "review regardless of nominal discount percentage. The 72.0% floor is "
                "a hard control; margin recovery via future upsell may not be used to "
                "justify an under-floor close."
            ),
        },
        {
            "chunk_id": "POL-003",
            "source_type": "DEAL_DESK_POLICY",
            "category": "working_capital",
            "title": "Working-Capital Adjustments for Net-60 and Net-90",
            "content": (
                "Standard payment terms are Net-30 with zero working-capital penalty. "
                "Net-60 terms apply a 1.5% of ACV working-capital penalty when computing "
                "effective gross margin. Net-90 terms apply a 3.0% of ACV penalty and "
                "always require CFO-And-Legal escalation. Extended payment terms requested "
                "by Enterprise procurement must be modeled before approval; AEs may not "
                "trade payment-term concessions for a lower face discount without "
                "recomputing margin."
            ),
        },
        {
            "chunk_id": "POL-004",
            "source_type": "DEAL_DESK_POLICY",
            "category": "services_cogs",
            "title": "Custom Onboarding Services COGS",
            "content": (
                "Custom implementation and onboarding hours that are discounted or "
                "unbilled are booked as services COGS at $150 per hour against the deal "
                "P&L. Services COGS reduce effective gross margin and must be included "
                "in Deal Desk margin calculations before approval routing. Bundled "
                "premium onboarding (>40 hours) on Mid-Market deals frequently creates "
                "margin traps where a seemingly acceptable <=20% discount still lands "
                "below the 72% floor when combined with Net-90 terms."
            ),
        },
        {
            "chunk_id": "POL-005",
            "source_type": "DEAL_DESK_POLICY",
            "category": "escalation_matrix",
            "title": "Approval Escalation Matrix",
            "content": (
                "Auto-Approve: discount <= 15%, gross margin >= 75%, Net-30 only, and "
                "zero high-risk legal flags. VP-Sales: discount between 15% and 25% "
                "(exclusive of lower bound, inclusive of upper), gross margin >= 72%, "
                "payment terms Net-30 or Net-60, and zero high-risk legal flags. "
                "CFO-And-Legal: any discount > 25%, any gross margin < 72%, any Net-90 "
                "terms, or any high-risk legal flag (MFC_CLAUSE, UNCAPPED_LIABILITY, "
                "TERMINATION_CONVENIENCE_30D, NON_STANDARD_AUDIT). Mixed conditions that "
                "do not cleanly fit Auto or VP default to CFO-And-Legal."
            ),
        },
        {
            "chunk_id": "POL-006",
            "source_type": "DEAL_DESK_POLICY",
            "category": "baseline_cogs",
            "title": "Baseline SaaS Hosting and Support COGS",
            "content": (
                "Baseline SaaS hosting and customer-support COGS are modeled as 18% of "
                "list ARR. At zero discount this yields an 82% gross margin before "
                "services and working-capital adjustments. List ARR is recovered from "
                "sold ACV via list_arr = annual_contract_value / (1 - discount_pct). "
                "Base COGS therefore rise as discount increases because hosting cost is "
                "tied to list capacity, not discounted invoice amount."
            ),
        },
    ]

    # 5 high-risk CUAD-modeled clauses + 5 standard/acceptable baseline clauses
    clauses = [
        {
            "chunk_id": "CUAD-001",
            "source_type": "CUAD_CLAUSE",
            "category": "MFC_CLAUSE",
            "title": "Most Favored Customer Pricing Parity",
            "content": (
                "Most Favored Customer. If during the Term Vendor offers to any "
                "similarly situated customer pricing, discounts, or commercial terms "
                "more favorable than those provided to Customer under this Agreement, "
                "Vendor shall promptly offer Customer the same pricing and terms, and "
                "Customer's fees shall be retroactively adjusted. Customer may audit "
                "Vendor's pricing records annually to verify compliance. FLAG: MFC_CLAUSE "
                "(high-risk)."
            ),
        },
        {
            "chunk_id": "CUAD-002",
            "source_type": "CUAD_CLAUSE",
            "category": "UNCAPPED_LIABILITY",
            "title": "Uncapped Indemnification and Liability",
            "content": (
                "Indemnification and Limitation of Liability. Notwithstanding any other "
                "provision, Vendor's aggregate liability arising out of or related to "
                "this Agreement shall be unlimited with respect to (a) indemnification "
                "obligations for intellectual property infringement, (b) breaches of "
                "confidentiality, and (c) any claim alleging gross negligence or willful "
                "misconduct. No cap, basket, or exclusion shall apply. FLAG: "
                "UNCAPPED_LIABILITY (high-risk)."
            ),
        },
        {
            "chunk_id": "CUAD-003",
            "source_type": "CUAD_CLAUSE",
            "category": "TERMINATION_CONVENIENCE_30D",
            "title": "30-Day Termination for Convenience",
            "content": (
                "Termination for Convenience. Customer may terminate this Agreement or "
                "any Order Form for any reason or no reason upon thirty (30) days' prior "
                "written notice to Vendor. Upon such termination, Vendor shall refund "
                "any prepaid fees on a pro-rata basis for the unused portion of the "
                "Term within fifteen (15) days. No early-termination fee shall apply. "
                "FLAG: TERMINATION_CONVENIENCE_30D (high-risk)."
            ),
        },
        {
            "chunk_id": "CUAD-004",
            "source_type": "CUAD_CLAUSE",
            "category": "NON_STANDARD_AUDIT",
            "title": "On-Site Customer Audit Rights",
            "content": (
                "Audit Rights. Upon ten (10) business days' notice, Customer and its "
                "designated third-party auditors may conduct an on-site audit of "
                "Vendor's facilities, systems, personnel, subprocessors, and books and "
                "records related to the Services, during normal business hours, up to "
                "four (4) times per calendar year, at Vendor's sole expense if any "
                "material non-compliance is found. FLAG: NON_STANDARD_AUDIT (high-risk)."
            ),
        },
        {
            "chunk_id": "CUAD-005",
            "source_type": "CUAD_CLAUSE",
            "category": "PRICE_CAP_AUTO_RENEWAL",
            "title": "Auto-Renewal with Aggressive Price Cap",
            "content": (
                "Auto-Renewal and Price Cap. This Agreement automatically renews for "
                "successive twelve (12) month periods unless either party provides "
                "notice of non-renewal at least ninety (90) days before the end of the "
                "then-current Term. Any price increase upon renewal shall not exceed "
                "two percent (2%) annually, and Vendor waives the right to discontinue "
                "SKUs included in Customer's then-current Order Form. Related commercial "
                "risk flag context: PRICE_CAP_AUTO_RENEWAL (monitor; escalate if "
                "combined with other high-risk redlines)."
            ),
        },
        # Standard / acceptable baseline clauses (retrieval must discriminate)
        {
            "chunk_id": "CUAD-006",
            "source_type": "CUAD_CLAUSE",
            "category": "STANDARD_LIABILITY_CAP",
            "title": "Standard Liability Cap (12 Months Fees)",
            "content": (
                "Limitation of Liability. Except for indemnification obligations and "
                "breaches of confidentiality, each party's aggregate liability under "
                "this Agreement shall not exceed the total fees paid by Customer to "
                "Vendor during the twelve (12) months preceding the claim. Neither "
                "party shall be liable for indirect, incidental, special, or "
                "consequential damages. This is a standard acceptable commercial "
                "baseline clause."
            ),
        },
        {
            "chunk_id": "CUAD-007",
            "source_type": "CUAD_CLAUSE",
            "category": "STANDARD_PAYMENT_NET30",
            "title": "Standard Net-30 Payment Terms",
            "content": (
                "Payment Terms. Customer shall pay all undisputed invoices within "
                "thirty (30) days of the invoice date (Net-30). Late payments accrue "
                "interest at the lesser of 1.5% per month or the maximum rate permitted "
                "by law. This is a standard acceptable commercial baseline clause "
                "aligned with Deal Desk Auto-Approve payment policy."
            ),
        },
        {
            "chunk_id": "CUAD-008",
            "source_type": "CUAD_CLAUSE",
            "category": "STANDARD_TERMINATION_CAUSE",
            "title": "Termination for Cause Only",
            "content": (
                "Termination for Cause. Either party may terminate this Agreement upon "
                "written notice if the other party materially breaches and fails to "
                "cure within thirty (30) days after receiving notice of such breach. "
                "Termination for convenience is not granted. This is a standard "
                "acceptable commercial baseline clause."
            ),
        },
        {
            "chunk_id": "CUAD-009",
            "source_type": "CUAD_CLAUSE",
            "category": "STANDARD_CONFIDENTIALITY",
            "title": "Mutual Confidentiality (Standard)",
            "content": (
                "Confidentiality. Each party agrees to protect the other party's "
                "Confidential Information with at least the same degree of care it "
                "uses to protect its own, and in no event less than reasonable care, "
                "for a period of three (3) years after disclosure (or perpetually for "
                "trade secrets). This is a standard acceptable commercial baseline "
                "clause."
            ),
        },
        {
            "chunk_id": "CUAD-010",
            "source_type": "CUAD_CLAUSE",
            "category": "STANDARD_DATA_PROTECTION",
            "title": "Standard Data Protection Addendum Reference",
            "content": (
                "Data Protection. Vendor shall process Customer Personal Data solely "
                "in accordance with the Data Processing Addendum attached as Exhibit B "
                "and applicable data-protection laws. Customer remains the controller; "
                "Vendor acts as processor. Subprocessors are listed at Vendor's trust "
                "portal and may be updated with thirty (30) days' notice. This is a "
                "standard acceptable commercial baseline clause."
            ),
        },
    ]

    return policies + clauses


def embed_chunks_locally(chunks: list[dict]) -> list[dict]:
    """Compute and store local dense embeddings via shared hybrid_rag embedder."""
    for chunk in chunks:
        chunk["embedding_json"] = json.dumps(local_embed_text(chunk["content"]))
    logger.info(
        "Stored local dense embeddings (dim=%d) for %d chunks in SQLite.",
        LOCAL_EMBEDDING_DIM,
        len(chunks),
    )
    return chunks


# ---------------------------------------------------------------------------
# Golden evaluation cases (20)
# ---------------------------------------------------------------------------
def _payload(
    account: str,
    segment: str,
    acv: float,
    term_months: int,
    discount_pct: float,
    payment_terms: str,
    services_hours: float,
    redline_text: str,
) -> dict:
    return {
        "account_name": account,
        "segment": segment,
        "annual_contract_value": acv,
        "term_months": term_months,
        "discount_pct": discount_pct,
        "payment_terms": payment_terms,
        "services_hours": services_hours,
        "proposed_contract_redline_text": redline_text,
    }


def build_golden_eval_cases() -> list[dict]:
    """20 diverse scenarios with mathematically consistent ground truth."""
    cases_spec: list[dict] = [
        # ----- 5 standard Auto-Approve -----
        {
            "case_id": "GE-001",
            "scenario_name": "Clean Mid-Market Net-30 low discount",
            "difficulty": "standard",
            "account": "BrightPath Analytics",
            "segment": "Mid-Market",
            "acv": 95_000.0,
            "term_months": 12,
            "discount_pct": 0.10,
            "payment_terms": "Net-30",
            "services_hours": 10.0,
            "redline_text": (
                "Customer accepts Vendor's standard MSA. Limitation of liability capped "
                "at twelve months of fees. Payment terms Net-30. Termination for cause only."
            ),
            "risk_flags": [],
        },
        {
            "case_id": "GE-002",
            "scenario_name": "Enterprise standard terms 12% discount",
            "difficulty": "standard",
            "account": "Northwind Enterprise Corp",
            "segment": "Enterprise",
            "acv": 320_000.0,
            "term_months": 24,
            "discount_pct": 0.12,
            "payment_terms": "Net-30",
            "services_hours": 16.0,
            "redline_text": (
                "Standard Order Form. Mutual confidentiality. Data Processing Addendum "
                "Exhibit B applies. No MFC, uncapped liability, or convenience termination."
            ),
            "risk_flags": [],
        },
        {
            "case_id": "GE-003",
            "scenario_name": "Mid-Market near-band Auto-Approve",
            "difficulty": "standard",
            "account": "Cedarlane Partners",
            "segment": "Mid-Market",
            "acv": 72_000.0,
            "term_months": 12,
            "discount_pct": 0.15,
            "payment_terms": "Net-30",
            "services_hours": 8.0,
            "redline_text": (
                "Paper process uses Vendor paper. Standard liability cap and Net-30. "
                "No non-standard audit rights requested."
            ),
            "risk_flags": [],
        },
        {
            "case_id": "GE-004",
            "scenario_name": "Enterprise low discount high ACV",
            "difficulty": "standard",
            "account": "Helios Manufacturing Group",
            "segment": "Enterprise",
            "acv": 480_000.0,
            "term_months": 36,
            "discount_pct": 0.08,
            "payment_terms": "Net-30",
            "services_hours": 20.0,
            "redline_text": (
                "Customer legal reviewed and accepted standard MSA without redlines. "
                "Auto-renewal at list with standard notice period."
            ),
            "risk_flags": [],
        },
        {
            "case_id": "GE-005",
            "scenario_name": "Mid-Market 5% discount clean close",
            "difficulty": "standard",
            "account": "Riverstone Labs",
            "segment": "Mid-Market",
            "acv": 55_000.0,
            "term_months": 12,
            "discount_pct": 0.05,
            "payment_terms": "Net-30",
            "services_hours": 12.0,
            "redline_text": (
                "No material redlines. Standard confidentiality and Net-30 payment schedule."
            ),
            "risk_flags": [],
        },
        # ----- 5 VP-Sales (discount 15–25% and/or Net-60) -----
        {
            "case_id": "GE-006",
            "scenario_name": "VP path 18% discount Net-30",
            "difficulty": "standard",
            "account": "Summit Dynamics",
            "segment": "Mid-Market",
            "acv": 110_000.0,
            "term_months": 24,
            "discount_pct": 0.18,
            "payment_terms": "Net-30",
            "services_hours": 14.0,
            "redline_text": (
                "Competitive pressure justified 18% off list. Standard liability language. "
                "No high-risk CUAD categories present."
            ),
            "risk_flags": [],
        },
        {
            "case_id": "GE-007",
            "scenario_name": "VP path 22% discount Net-60",
            "difficulty": "standard",
            "account": "Orion Global Services",
            "segment": "Enterprise",
            "acv": 260_000.0,
            "term_months": 24,
            "discount_pct": 0.22,
            "payment_terms": "Net-60",
            "services_hours": 18.0,
            "redline_text": (
                "Procurement requires Net-60. Discount 22%. Legal accepted standard "
                "liability cap and termination-for-cause."
            ),
            "risk_flags": [],
        },
        {
            "case_id": "GE-008",
            "scenario_name": "VP path 20% discount Net-30 Enterprise",
            "difficulty": "standard",
            "account": "Cobalt Financial Systems",
            "segment": "Enterprise",
            "acv": 390_000.0,
            "term_months": 36,
            "discount_pct": 0.20,
            "payment_terms": "Net-30",
            "services_hours": 24.0,
            "redline_text": (
                "Multi-year Enterprise with 20% discount. Paper is Vendor MSA with "
                "standard DPA. No MFC or uncapped indemnity."
            ),
            "risk_flags": [],
        },
        {
            "case_id": "GE-009",
            "scenario_name": "VP path 16% discount Net-60 Mid-Market",
            "difficulty": "standard",
            "account": "Fieldstone Networks",
            "segment": "Mid-Market",
            "acv": 88_000.0,
            "term_months": 12,
            "discount_pct": 0.16,
            "payment_terms": "Net-60",
            "services_hours": 10.0,
            "redline_text": (
                "Net-60 requested by AP. Discount lightly above AE band at 16%. "
                "No non-standard audit or convenience termination."
            ),
            "risk_flags": [],
        },
        {
            "case_id": "GE-010",
            "scenario_name": "VP path 25% discount Net-30 ceiling",
            "difficulty": "standard",
            "account": "Atlas Retail Platforms",
            "segment": "Enterprise",
            "acv": 210_000.0,
            "term_months": 24,
            "discount_pct": 0.25,
            "payment_terms": "Net-30",
            "services_hours": 15.0,
            "redline_text": (
                "At VP discount ceiling of 25%. Margin still above floor after light "
                "onboarding. Standard commercial terms only."
            ),
            "risk_flags": [],
        },
        # ----- 5 edge_margin traps: discount <= 20% but margin < 72% -----
        {
            "case_id": "GE-011",
            "scenario_name": "Margin trap: 18% discount + heavy services + Net-90",
            "difficulty": "edge_margin",
            "account": "Pinecrest Healthcare",
            "segment": "Mid-Market",
            "acv": 80_000.0,
            "term_months": 12,
            "discount_pct": 0.18,
            "payment_terms": "Net-90",
            "services_hours": 80.0,
            "redline_text": (
                "AE proposes 18% discount which looks within VP band, but includes 80 "
                "unbilled onboarding hours and Net-90 AP terms. Standard MSA otherwise."
            ),
            "risk_flags": ["MARGIN_BELOW_FLOOR", "NET_90_WORKING_CAPITAL"],
        },
        {
            "case_id": "GE-012",
            "scenario_name": "Margin trap: 15% discount + services + Net-90",
            "difficulty": "edge_margin",
            "account": "Lakeside Education Board",
            "segment": "Mid-Market",
            "acv": 60_000.0,
            "term_months": 12,
            "discount_pct": 0.15,
            "payment_terms": "Net-90",
            "services_hours": 70.0,
            "redline_text": (
                "Nominal 15% discount appears Auto-eligible, but Net-90 plus 70 hours "
                "of complimentary implementation destroy margin."
            ),
            "risk_flags": ["MARGIN_BELOW_FLOOR", "NET_90_WORKING_CAPITAL"],
        },
        {
            "case_id": "GE-013",
            "scenario_name": "Margin trap: 20% discount Enterprise services Net-90",
            "difficulty": "edge_margin",
            "account": "Ironclad Logistics Inc",
            "segment": "Enterprise",
            "acv": 150_000.0,
            "term_months": 24,
            "discount_pct": 0.20,
            "payment_terms": "Net-90",
            "services_hours": 120.0,
            "redline_text": (
                "20% discount looks commercially fine on face. Enterprise asks for "
                "white-glove onboarding (120 hours unbilled) and Net-90."
            ),
            "risk_flags": ["MARGIN_BELOW_FLOOR", "NET_90_WORKING_CAPITAL"],
        },
        {
            "case_id": "GE-014",
            "scenario_name": "Margin trap: 12% discount Mid-Market Net-90 services",
            "difficulty": "edge_margin",
            "account": "Harborview Credit Union",
            "segment": "Mid-Market",
            "acv": 70_000.0,
            "term_months": 12,
            "discount_pct": 0.12,
            "payment_terms": "Net-90",
            "services_hours": 90.0,
            "redline_text": (
                "Low 12% discount masks a margin breach once 90 hours of custom "
                "onboarding and Net-90 working-capital drag are applied."
            ),
            "risk_flags": ["MARGIN_BELOW_FLOOR", "NET_90_WORKING_CAPITAL"],
        },
        {
            "case_id": "GE-015",
            "scenario_name": "Margin trap: 19% discount + Net-90 onboarding bundle",
            "difficulty": "edge_margin",
            "account": "Silverline Media Co",
            "segment": "Mid-Market",
            "acv": 100_000.0,
            "term_months": 12,
            "discount_pct": 0.19,
            "payment_terms": "Net-90",
            "services_hours": 100.0,
            "redline_text": (
                "Discount 19% (<=20%) with bundled onboarding and Net-90. AE believes "
                "deal is VP-routable; margin math says otherwise."
            ),
            "risk_flags": ["MARGIN_BELOW_FLOOR", "NET_90_WORKING_CAPITAL"],
        },
        # ----- 5 compound legal / CUAD redline traps -----
        {
            "case_id": "GE-016",
            "scenario_name": "Compound: MFC clause on otherwise clean deal",
            "difficulty": "compound_trap",
            "account": "GlobalTech Procurement LLC",
            "segment": "Enterprise",
            "acv": 275_000.0,
            "term_months": 24,
            "discount_pct": 0.10,
            "payment_terms": "Net-30",
            "services_hours": 20.0,
            "redline_text": (
                "Customer inserts Most Favored Customer pricing parity: if Vendor offers "
                "any similarly situated customer more favorable pricing, Customer "
                "receives the same terms retroactively and may audit Vendor pricing "
                "records annually. FLAG MFC_CLAUSE."
            ),
            "risk_flags": ["MFC_CLAUSE"],
        },
        {
            "case_id": "GE-017",
            "scenario_name": "Compound: uncapped liability + mild discount",
            "difficulty": "edge_legal",
            "account": "Beacon Insurance Holdings",
            "segment": "Enterprise",
            "acv": 340_000.0,
            "term_months": 36,
            "discount_pct": 0.14,
            "payment_terms": "Net-30",
            "services_hours": 22.0,
            "redline_text": (
                "Indemnification and liability shall be uncapped for IP infringement, "
                "confidentiality breaches, and alleged gross negligence. No aggregate "
                "cap applies. FLAG UNCAPPED_LIABILITY."
            ),
            "risk_flags": ["UNCAPPED_LIABILITY"],
        },
        {
            "case_id": "GE-018",
            "scenario_name": "Compound: 30-day termination for convenience",
            "difficulty": "edge_legal",
            "account": "Cascade Municipal IT",
            "segment": "Mid-Market",
            "acv": 120_000.0,
            "term_months": 12,
            "discount_pct": 0.11,
            "payment_terms": "Net-30",
            "services_hours": 16.0,
            "redline_text": (
                "Customer may terminate for convenience on thirty (30) days' notice "
                "with pro-rata refund of prepaid fees and no early-termination fee. "
                "FLAG TERMINATION_CONVENIENCE_30D."
            ),
            "risk_flags": ["TERMINATION_CONVENIENCE_30D"],
        },
        {
            "case_id": "GE-019",
            "scenario_name": "Compound: on-site audit rights + Net-60",
            "difficulty": "compound_trap",
            "account": "Keystone Federal Integrators",
            "segment": "Enterprise",
            "acv": 410_000.0,
            "term_months": 24,
            "discount_pct": 0.17,
            "payment_terms": "Net-60",
            "services_hours": 30.0,
            "redline_text": (
                "Customer and third-party auditors may conduct on-site audits of "
                "Vendor facilities and subprocessors up to four times per year at "
                "Vendor expense if non-compliance found. FLAG NON_STANDARD_AUDIT. "
                "Also requests Net-60."
            ),
            "risk_flags": ["NON_STANDARD_AUDIT"],
        },
        {
            "case_id": "GE-020",
            "scenario_name": "Compound: MFC + uncapped liability + Net-90 margin stress",
            "difficulty": "compound_trap",
            "account": "Meridian Sovereign Wealth Ops",
            "segment": "Enterprise",
            "acv": 500_000.0,
            "term_months": 36,
            "discount_pct": 0.20,
            "payment_terms": "Net-90",
            "services_hours": 100.0,
            "redline_text": (
                "Stack of redlines: Most Favored Customer pricing parity; uncapped "
                "indemnification/liability; Net-90 payment; plus 100 hours unbilled "
                "onboarding. FLAGS: MFC_CLAUSE, UNCAPPED_LIABILITY, "
                "NET_90_WORKING_CAPITAL, MARGIN_BELOW_FLOOR risk."
            ),
            "risk_flags": [
                "MFC_CLAUSE",
                "UNCAPPED_LIABILITY",
                "NET_90_WORKING_CAPITAL",
                "MARGIN_BELOW_FLOOR",
            ],
        },
    ]

    cases: list[dict] = []
    for spec in cases_spec:
        margin = compute_gross_margin_pct(
            spec["acv"],
            spec["discount_pct"],
            spec["payment_terms"],
            spec["services_hours"],
        )
        lane = determine_approval_lane(
            spec["discount_pct"],
            margin,
            spec["payment_terms"],
            spec["risk_flags"],
        )
        payload = _payload(
            spec["account"],
            spec["segment"],
            spec["acv"],
            spec["term_months"],
            spec["discount_pct"],
            spec["payment_terms"],
            spec["services_hours"],
            spec["redline_text"],
        )
        cases.append(
            {
                "case_id": spec["case_id"],
                "scenario_name": spec["scenario_name"],
                "difficulty": spec["difficulty"],
                "deal_payload_json": json.dumps(payload),
                "expected_gross_margin_pct": margin,
                "expected_approval_lane": lane,
                "expected_risk_flags_json": json.dumps(spec["risk_flags"]),
            }
        )

    return cases


# ---------------------------------------------------------------------------
# Persist helpers
# ---------------------------------------------------------------------------
def clear_tables(conn: sqlite3.Connection) -> None:
    for table in (
        "historical_deals",
        "golden_eval_cases",
        "rag_chunks",
        "cost_ledger",
        "eval_results",
    ):
        conn.execute(f"DELETE FROM {table}")
    conn.commit()


def insert_historical_deals(conn: sqlite3.Connection, deals: list[dict]) -> None:
    conn.executemany(
        """
        INSERT INTO historical_deals (
            deal_id, account_name, segment, annual_contract_value, term_months,
            discount_pct, payment_terms, custom_onboarding_cost, gross_margin_pct,
            has_nonstandard_legal, outcome, sales_cycle_days
        ) VALUES (
            :deal_id, :account_name, :segment, :annual_contract_value, :term_months,
            :discount_pct, :payment_terms, :custom_onboarding_cost, :gross_margin_pct,
            :has_nonstandard_legal, :outcome, :sales_cycle_days
        )
        """,
        deals,
    )


def insert_rag_chunks(conn: sqlite3.Connection, chunks: list[dict]) -> None:
    conn.executemany(
        """
        INSERT INTO rag_chunks (
            chunk_id, source_type, category, title, content, embedding_json
        ) VALUES (
            :chunk_id, :source_type, :category, :title, :content, :embedding_json
        )
        """,
        chunks,
    )


def insert_golden_cases(conn: sqlite3.Connection, cases: list[dict]) -> None:
    conn.executemany(
        """
        INSERT INTO golden_eval_cases (
            case_id, scenario_name, difficulty, deal_payload_json,
            expected_gross_margin_pct, expected_approval_lane, expected_risk_flags_json
        ) VALUES (
            :case_id, :scenario_name, :difficulty, :deal_payload_json,
            :expected_gross_margin_pct, :expected_approval_lane, :expected_risk_flags_json
        )
        """,
        cases,
    )


def seed_db(db_path: str = DEFAULT_DB_PATH) -> dict[str, int]:
    """Initialize schema and load all deterministic seed data."""
    conn = init_db(db_path)
    clear_tables(conn)

    deals = generate_historical_deals(250)
    chunks = embed_chunks_locally(build_rag_chunks())
    cases = build_golden_eval_cases()

    insert_historical_deals(conn, deals)
    insert_rag_chunks(conn, chunks)
    insert_golden_cases(conn, cases)
    conn.commit()

    counts = {
        "historical_deals": conn.execute("SELECT COUNT(*) FROM historical_deals").fetchone()[0],
        "golden_eval_cases": conn.execute("SELECT COUNT(*) FROM golden_eval_cases").fetchone()[0],
        "rag_chunks": conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0],
        "cost_ledger": conn.execute("SELECT COUNT(*) FROM cost_ledger").fetchone()[0],
    }

    # Sanity log for margin-trap cases
    traps = [
        c
        for c in cases
        if c["case_id"] in {"GE-011", "GE-012", "GE-013", "GE-014", "GE-015"}
    ]
    for t in traps:
        payload = json.loads(t["deal_payload_json"])
        logger.info(
            "%s discount=%.0f%% margin=%.2f%% lane=%s",
            t["case_id"],
            payload["discount_pct"] * 100,
            t["expected_gross_margin_pct"],
            t["expected_approval_lane"],
        )

    conn.close()
    logger.info(
        "Seed complete at %s — historical=%d golden=%d rag=%d ledger=%d",
        db_path,
        counts["historical_deals"],
        counts["golden_eval_cases"],
        counts["rag_chunks"],
        counts["cost_ledger"],
    )
    return counts


if __name__ == "__main__":
    seed_db()
