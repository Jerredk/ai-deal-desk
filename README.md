# enterprise-deal-desk

Raw-API Tool-Using Deal-Desk & Margin-Guardrail Agent with Per-Run Cost Instrumentation and Automated Evals

## 1. Abstract

Enterprise B2B SaaS companies in the $25M–$150M ARR band routinely lose margin—and deal velocity—when Account Executives negotiate non-standard discounts, Net-60/90 payment terms, unbilled onboarding hours, and legal redlines without a single system of record that recomputes economics and routes approvals. Those factors compound silently: a “safe-looking” ≤20% discount can still breach a 72% gross-margin floor once services COGS and working-capital penalties are applied, while high-risk contract language (MFN, uncapped liability, convenience termination, aggressive audit rights) escapes deal-desk review until late in the cycle.

This repository implements a framework-free Python + SQLite tool-using agent on the Anthropic Messages API. Deterministic unit-economics tools compute list ARR, hosting/support COGS, services COGS, and working-capital adjustments; Hybrid RAG (BM25 + dense retrieval) grounds policy and commercial-clause decisions; an Evaluator-Optimizer critique loop sharpens approval routing; and every run writes token/USD/latency rows to a cost ledger for apples-to-apples experiment comparison.

## 2. Implementation & How to Run

**Architecture:**
- `db/` — SQLite warehouse (`schema.py`) and deterministic seed script (`seed_db.py`) for historical deals, golden eval cases, RAG chunks, cost ledger, and resumable `eval_results`.
- `src/` — deterministic tools, Hybrid RAG, Anthropic cost ledger, versioned prompts, raw Messages API agent (`v1`/`v2`/`v3`), and the eval harness + report exporter.
- `tests/` — pytest suite for seed integrity, tools/RAG, agent loop (mocked), and offline eval grading.
- `outputs/` — benchmark CSV/JSON artifacts from the 60-run evaluation.

**Quickstart:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set ANTHROPIC_API_KEY=
python db/seed_db.py
pytest -v
python -m src.agent --case-id GE-015 --version v3
python -m src.evals --versions v1 v2 v3 --workers 3
```

## 3. Test Setup

**Unstructured Retrieval Corpus.** Six B2B SaaS Deal-Desk policy documents covering discount authority bands, the 72.0% hard margin floor (with 82% baseline margin at 0% discount / 18% list-ARR COGS), working-capital penalties for Net-60/90, onboarding services COGS, and the escalation matrix—plus ten commercial contract clauses modeled on CUAD categories (Most Favored Nation pricing parity, uncapped liability, 30-day termination for convenience, on-site audit rights, auto-renewal/price-cap, and five standard/acceptable baseline clauses so retrieval must discriminate). Dense embeddings for Hybrid RAG are computed locally and stored in SQLite (`rag_chunks.embedding_json`); no third-party embedding API is required.

**Structured Warehouse & Golden Eval Suite.** Two hundred fifty seeded historical B2B SaaS deals (`random.seed(42)`) with realistic win-rate decay at extreme discounts and longer cycles when legal redlines are present. Twenty adversarial golden cases: 5 Auto-Approve, 5 VP-Sales, 10 CFO-And-Legal—including six nominal-discount margin traps where `discount <= 20%` still breaches the 72.0% floor after services COGS and Net-90 working-capital drag. Every golden `expected_gross_margin_pct` and `expected_approval_tier` is computed from the same Python formulas used at runtime.

**Evaluation Protocol.** Cross-version runs of `v1` (baseline prompting + BM25 + basic tools) vs. `v2` (poka-yoked ACI tools + Hybrid RAG + prompt caching) vs. `v3` (Evaluator-Optimizer critic loop). Scoring combines deterministic assertions (tier exact match, margin ±0.05, flag precision/recall/F1) with LLM-as-a-Judge grading on an independent Anthropic model (`claude-opus-4-6`) separate from the deal-desk agent (`claude-sonnet-4-6`). Full suite: **60 runs** (20 cases × 3 versions).

## 4. Test Results

**Status: Complete (60/60 runs; Anthropic list rates pinned as of 2026-11-21)**

Unit tests: **31/31** passing. Seeded warehouse: **250** historical deals, **16** RAG chunks, **20** formula-verified golden cases. Empirical agent benchmark below.

### Overall (`outputs/benchmark_summary.csv`)

| Experiment Version | Cases | Det. Pass % | Tier Acc. % | Margin Acc. % | Flag Recall | Flag Precision | Flag F1 | Mean Judge (1–5) | Rev. Trigger % | Mean Cost (USD) | p10 / p50 / p90 Cost | Avg Cache Hit % | p50 / p90 Latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| v1 Baseline | 20 | 85.0 | 100.0 | 100.0 | 0.850 | 0.708 | 0.757 | 4.99 | 0.0 | 0.0967 | 0.0578 / 0.1029 / 0.1202 | 0.0 | 73.1 / 86.0 |
| v2 Tools + Hybrid RAG + Caching | 20 | 100.0 | 100.0 | 100.0 | 1.000 | 1.000 | 1.000 | 4.97 | 0.0 | 0.0606 | 0.0435 / 0.0514 / 0.1009 | 57.0 | 47.0 / 76.8 |
| v3 Full Evaluator-Optimizer | 20 | 100.0 | 100.0 | 100.0 | 1.000 | 0.975 | 0.983 | 4.99 | 5.0 | 0.0813 | 0.0512 / 0.0640 / 0.1179 | 49.4 | 51.4 / 88.3 |

Judge dimension means (all versions ≈5.0 commercial / policy / clarity; actionability 4.90–4.95).

### Difficulty breakdown (`outputs/difficulty_breakdown.csv`)

| Version | Difficulty | Cases | Det. Pass % | Mean Judge | p50 Cost (USD) |
|---|---|---:|---:|---:|---:|
| v1 | standard | 10 | 90.0 | 4.97 | 0.0794 |
| v1 | edge_margin | 5 | 100.0 | 5.00 | 0.1090 |
| v1 | edge_legal | 2 | 50.0 | 5.00 | 0.0942 |
| v1 | compound_trap | 3 | 66.7 | 5.00 | 0.1156 |
| v2 | standard | 10 | 100.0 | 4.95 | 0.0499 |
| v2 | edge_margin | 5 | 100.0 | 5.00 | 0.0998 |
| v2 | edge_legal | 2 | 100.0 | 5.00 | 0.0491 |
| v2 | compound_trap | 3 | 100.0 | 5.00 | 0.0546 |
| v3 | standard | 10 | 100.0 | 4.97 | 0.0593 |
| v3 | edge_margin | 5 | 100.0 | 5.00 | 0.1157 |
| v3 | edge_legal | 2 | 100.0 | 5.00 | 0.0573 |
| v3 | compound_trap | 3 | 100.0 | 5.00 | 0.0778 |

**Case-level notes.** v1 failed deterministic pass on GE-009 / GE-018 / GE-019 via non-canonical or prose-annotated flag strings (tier + margin were already exact). v2 and v3 rescued all three with perfect canonical recall. v3’s critic triggered a revision on **1/20** runs (a `compound_trap` case; 33% of that difficulty bucket).

## 5. Conclusion

**Status: Complete (empirical 60-run suite)**

Isolating SaaS margin math and approval-tier routing in deterministic Python/SQLite eliminated arithmetic drift: **tier and margin accuracy were 100% across all three versions**. Quality lift was concentrated in **canonical redline flag discipline**:

- **v1 → v2:** deterministic pass **85% → 100%**; flag recall **0.85 → 1.00**; F1 **0.76 → 1.00**. Mean agent cost fell **~37%** ($0.097 → $0.061) with **~57% average cache hit rate** (v1: 0%) and p50 latency **73s → 47s**. Opus judge scores stayed ~5.0—v1 memos already read well; v2 mainly fixed machine-checkable taxonomy.
- **v2 → v3:** quality stayed at **100%** deterministic pass / full recall. Critic revisions fired rarely (**5%** overall; only on a compound trap). Marginal economics: mean cost **$0.061 → $0.081** (~**+34%** vs v2) and p50 latency **47s → 51s**, with cache hit rate still healthy (~49%).

**Practical takeaway.** For standard, margin-trap, and most legal deals, **v2 (poka-yoked ACI tools + Hybrid RAG + prompt caching) is the default production posture**—it closes the flag-recall gap and pays for itself on cost/latency. Reserve **v3’s Evaluator-Optimizer** for high-stakes compound redline stacks where an independent critic pass is worth the ~$0.02–0.03 median premium; it is not required to hit the golden-suite deterministic bar once v2 tooling is in place.
