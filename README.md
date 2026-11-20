# enterprise-deal-desk

Raw-API Tool-Using Deal-Desk & Margin-Guardrail Agent with Per-Run Cost Instrumentation and Automated Evals

## 1. Abstract

Enterprise B2B SaaS companies in the $25M–$150M ARR band routinely lose margin—and deal velocity—when Account Executives negotiate non-standard discounts, Net-60/90 payment terms, unbilled onboarding hours, and legal redlines without a single system of record that recomputes economics and routes approvals. Those factors compound silently: a “safe-looking” ≤20% discount can still breach a 72% gross-margin floor once services COGS and working-capital penalties are applied, while high-risk contract language (MFN, uncapped liability, convenience termination, aggressive audit rights) escapes deal-desk review until late in the cycle.

This repository implements a framework-free Python + SQLite tool-using agent on the Anthropic Messages API. Deterministic unit-economics tools compute list ARR, hosting/support COGS, services COGS, and working-capital adjustments; Hybrid RAG (BM25 + dense retrieval) grounds policy and commercial-clause decisions; an Evaluator-Optimizer critique loop sharpens approval routing; and every run writes token/USD/latency rows to a cost ledger for apples-to-apples experiment comparison.

## 2. Implementation & How to Run

**Architecture (Increment 1 foundation):**
- `db/` — SQLite warehouse (`schema.py`) and deterministic seed script (`seed_db.py`) for historical deals, golden eval cases, RAG chunks, and the cost ledger.
- `src/` — reserved for deterministic tools, Hybrid RAG, cost instrumentation, and the raw Anthropic agent loop (Increments 2+).
- `tests/` — pytest verification of schema, seed integrity, and margin/approval ground truth.
- `outputs/` — run artifacts (kept via `.gitkeep`).

**Quickstart:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set ANTHROPIC_API_KEY=
python db/seed_db.py
pytest -v
```

## 3. Test Setup

**Unstructured Retrieval Corpus.** Six B2B SaaS Deal-Desk policy documents covering discount authority bands, the 72.0% hard margin floor (with 82% baseline margin at 0% discount / 18% list-ARR COGS), working-capital penalties for Net-60/90, onboarding services COGS, and the escalation matrix—plus ten commercial contract clauses modeled on CUAD categories (Most Favored Nation pricing parity, uncapped liability, 30-day termination for convenience, on-site audit rights, auto-renewal/price-cap, and five standard/acceptable baseline clauses so retrieval must discriminate). Dense embeddings for Hybrid RAG are computed locally and stored in SQLite (`rag_chunks.embedding_json`); no third-party embedding API is required.

**Structured Warehouse & Golden Eval Suite.** Two hundred fifty seeded historical B2B SaaS deals (`random.seed(42)`) with realistic win-rate decay at extreme discounts and longer cycles when legal redlines are present. Twenty adversarial golden cases: 5 Auto-Approve, 5 VP-Sales, 10 CFO-And-Legal—including six nominal-discount margin traps where `discount <= 20%` still breaches the 72.0% floor after services COGS and Net-90 working-capital drag. Every golden `expected_gross_margin_pct` and `expected_approval_tier` is computed from the same Python formulas used at runtime.

**Planned Progression.** Cross-version evaluation of `v1` (baseline prompting) vs. `v2` (poka-yoked tools + Hybrid RAG + prompt caching) vs. `v3` (Evaluator-Optimizer loop), scored with deterministic assertions plus LLM-as-a-Judge grading on an independent Anthropic model (separate from the deal-desk agent).

## 4. Test Results

**Status: In Progress (Increment 1 Verified; Full Agent Benchmarks Pending)**

Increment 1 validation: **6/6** pytest checks passing. Seeded warehouse: **250** historical deals, **16** RAG chunks (6 policy + 10 CUAD), **20** formula-verified golden cases.

| Experiment Version | Approval Tier Accuracy | Margin Math Accuracy | Redline Recall | Judge Score (1-5) | p50 Cost/Run (USD) | p50 Latency (s) |
|---|---|---|---|---|---|---|
| v1 Baseline | TBD (Increments 3–5) | TBD (Increments 3–5) | TBD (Increments 3–5) | TBD (Increments 3–5) | TBD (Increments 3–5) | TBD (Increments 3–5) |
| v2 Tools + Hybrid RAG + Caching | TBD (Increments 3–5) | TBD (Increments 3–5) | TBD (Increments 3–5) | TBD (Increments 3–5) | TBD (Increments 3–5) | TBD (Increments 3–5) |
| v3 Full Evaluator-Optimizer | TBD (Increments 3–5) | TBD (Increments 3–5) | TBD (Increments 3–5) | TBD (Increments 3–5) | TBD (Increments 3–5) | TBD (Increments 3–5) |

## 5. Conclusion

**Status: Preliminary (Increment 1 Foundation Complete)**

The core architectural finding from Increment 1 is that isolating SaaS margin math and approval-tier routing in deterministic Python/SQLite eliminates arithmetic drift and establishes exact ground truth *before* introducing LLM reasoning. Final conclusions on cost-quality trade-offs and prompt-caching ROI will be added upon completion of the 60-run evaluation suite.
