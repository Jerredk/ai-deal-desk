"""Build a self-contained, reproducible HTML evaluation dashboard (Step 6).

Consumes existing artifacts only — no live LLM calls, no changes to agent/evals.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
DB_PATH = ROOT / "db" / "deal_desk.db"
RESULTS = ROOT / "results"
RESULTS_DATA = RESULTS / "data"

AGENT_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "claude-opus-4-6"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _git_meta() -> dict[str, str]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    return {
        "commit_sha": run(["git", "rev-parse", "--short", "HEAD"]),
        "commit_full": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "branch", "--show-current"]),
        "commit_timestamp": run(["git", "log", "-1", "--format=%cI"]),
    }


def _run_pytest() -> dict[str, Any]:
    proc = subprocess.run(
        ["pytest", "-v", "--tb=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    tests: list[dict[str, str]] = []
    for line in out.splitlines():
        m = re.search(r"^(tests/\S+)\s+(PASSED|FAILED|SKIPPED|ERROR)", line)
        if m:
            tests.append({"name": m.group(1), "outcome": m.group(2).title()})
    summary = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "duration_s": None}
    m = re.search(
        r"=+\s*(?:(\d+)\s+passed)?(?:,\s*)?(?:(\d+)\s+failed)?(?:,\s*)?(?:(\d+)\s+skipped)?"
        r".*?in\s+([\d.]+)s",
        out,
    )
    # Prefer pytest's classic footer: "31 passed in 1.29s"
    m2 = re.search(r"(\d+)\s+passed(?:,\s*(\d+)\s+failed)?(?:,\s*(\d+)\s+skipped)?\s+in\s+([\d.]+)s", out)
    if m2:
        summary["passed"] = int(m2.group(1))
        summary["failed"] = int(m2.group(2) or 0)
        summary["skipped"] = int(m2.group(3) or 0)
        summary["duration_s"] = float(m2.group(4))
    elif tests:
        summary["passed"] = sum(1 for t in tests if t["outcome"] == "Passed")
        summary["failed"] = sum(1 for t in tests if t["outcome"] in {"Failed", "Error"})
        summary["skipped"] = sum(1 for t in tests if t["outcome"] == "Skipped")
    return {
        "exit_code": proc.returncode,
        "summary": summary,
        "tests": tests,
        "raw_tail": "\n".join(out.strip().splitlines()[-8:]),
    }


def _classify_run(row: dict[str, Any]) -> str:
    if int(row.get("deterministic_pass") or 0) == 1:
        return "Passed"
    tier_ok = int(row.get("tier_exact_match") or 0) == 1
    margin_ok = int(row.get("margin_exact_match") or 0) == 1
    recall = float(row.get("flag_recall") or 0.0)
    if tier_ok and margin_ok and recall < 1.0:
        return "Partially Passed"
    return "Failed"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _assemble_dataset(pytest_report: dict[str, Any], git_meta: dict[str, str]) -> dict[str, Any]:
    summary = _read_json(OUTPUTS / "benchmark_summary.json")
    summary_csv = _read_csv(OUTPUTS / "benchmark_summary.csv")
    difficulty = _read_csv(OUTPUTS / "difficulty_breakdown.csv")
    case_csv = _read_csv(OUTPUTS / "case_level_evals.csv")

    conn = _connect()
    golden = {
        r["case_id"]: dict(r)
        for r in conn.execute("SELECT * FROM golden_eval_cases ORDER BY case_id")
    }
    eval_rows = [dict(r) for r in conn.execute("SELECT * FROM eval_results ORDER BY experiment_version, case_id")]

    # Per-version token aggregates (agent steps only; exclude llm_judge)
    token_by_version: dict[str, dict[str, float]] = {}
    for ver in ("v1", "v2", "v3"):
        row = conn.execute(
            """
            SELECT
              COALESCE(SUM(input_tokens), 0),
              COALESCE(SUM(output_tokens), 0),
              COALESCE(SUM(cache_creation_tokens), 0),
              COALESCE(SUM(cache_read_tokens), 0),
              COALESCE(SUM(latency_ms), 0),
              COUNT(*)
            FROM cost_ledger
            WHERE experiment_version = ? AND step_name != 'llm_judge'
              AND run_id IN (
                SELECT run_id FROM eval_results WHERE experiment_version = ?
              )
            """,
            (ver, ver),
        ).fetchone()
        token_by_version[ver] = {
            "input_tokens": int(row[0]),
            "output_tokens": int(row[1]),
            "cache_creation_tokens": int(row[2]),
            "cache_read_tokens": int(row[3]),
            "total_latency_ms": float(row[4]),
            "agent_api_calls": int(row[5]),
        }

    # Per-run cost ledger breakdown
    cost_by_run: dict[str, list[dict[str, Any]]] = {}
    for r in conn.execute(
        """
        SELECT run_id, step_name, model, input_tokens, output_tokens,
               cache_creation_tokens, cache_read_tokens, cost_usd, latency_ms, created_at
        FROM cost_ledger
        ORDER BY id
        """
    ):
        cost_by_run.setdefault(r["run_id"], []).append(dict(r))

    # Revision / critic step stats
    critic_stats = {}
    for ver in ("v1", "v2", "v3"):
        rev = conn.execute(
            "SELECT SUM(revision_applied), COUNT(*) FROM eval_results WHERE experiment_version=?",
            (ver,),
        ).fetchone()
        critic_calls = conn.execute(
            """
            SELECT COUNT(*) FROM cost_ledger
            WHERE experiment_version=? AND step_name='critic'
            """,
            (ver,),
        ).fetchone()[0]
        revision_steps = conn.execute(
            """
            SELECT COUNT(*) FROM cost_ledger
            WHERE experiment_version=? AND step_name='producer_revision'
            """,
            (ver,),
        ).fetchone()[0]
        critic_stats[ver] = {
            "revision_applied_count": int(rev[0] or 0),
            "cases": int(rev[1] or 0),
            "critic_api_calls": int(critic_calls),
            "producer_revision_api_calls": int(revision_steps),
            "producer_revision_skipped": int((rev[1] or 0) - (rev[0] or 0))
            if ver == "v3"
            else int(rev[1] or 0),
        }

    timestamps = conn.execute(
        "SELECT MIN(created_at), MAX(created_at) FROM eval_results"
    ).fetchone()
    conn.close()

    cases: list[dict[str, Any]] = []
    status_counts = {"Passed": 0, "Partially Passed": 0, "Failed": 0, "Skipped": 0}
    for er in eval_rows:
        case_id = er["case_id"]
        g = golden.get(case_id, {})
        payload = {}
        if g.get("deal_payload_json"):
            payload = json.loads(g["deal_payload_json"])
        memo = json.loads(er["memo_json"]) if er.get("memo_json") else {}
        expected_flags = json.loads(g.get("expected_risk_flags_json") or "[]")
        status = _classify_run(er)
        status_counts[status] += 1
        steps = cost_by_run.get(er["run_id"], [])
        agent_steps = [s for s in steps if s["step_name"] != "llm_judge"]
        judge_steps = [s for s in steps if s["step_name"] == "llm_judge"]
        cases.append(
            {
                "eval_id": er["eval_id"],
                "run_id": er["run_id"],
                "experiment_version": er["experiment_version"],
                "case_id": case_id,
                "account_name": payload.get("account_name")
                or memo.get("account_name")
                or "",
                "difficulty": er["difficulty"],
                "status": status,
                "tier_exact_match": int(er["tier_exact_match"]),
                "margin_exact_match": int(er["margin_exact_match"]),
                "flag_precision": float(er["flag_precision"]),
                "flag_recall": float(er["flag_recall"]),
                "flag_f1": float(er["flag_f1"]),
                "deterministic_pass": int(er["deterministic_pass"]),
                "expected_approval_tier": g.get("expected_approval_tier"),
                "expected_gross_margin_pct": g.get("expected_gross_margin_pct"),
                "expected_risk_flags": expected_flags,
                "actual_approval_tier": memo.get("required_approval_tier"),
                "actual_gross_margin_pct": memo.get("calculated_gross_margin_pct"),
                "actual_risk_flags": memo.get("detected_risk_flags") or [],
                "judge_model": er["judge_model"],
                "judge_commercial_accuracy": int(er["judge_commercial_accuracy"]),
                "judge_policy_grounding": int(er["judge_policy_grounding"]),
                "judge_executive_clarity": int(er["judge_executive_clarity"]),
                "judge_actionability": int(er["judge_actionability"]),
                "judge_overall_score": float(er["judge_overall_score"]),
                "judge_rationale": er["judge_rationale"],
                "revision_applied": int(er["revision_applied"]),
                "total_cost_usd": float(er["total_cost_usd"]),
                "total_latency_ms": float(er["total_latency_ms"]),
                "cache_hit_rate_pct": float(er["cache_hit_rate_pct"]),
                "created_at": er["created_at"],
                "deal_payload": payload,
                "memo": memo,
                "agent_token_totals": {
                    "input_tokens": sum(int(s["input_tokens"]) for s in agent_steps),
                    "output_tokens": sum(int(s["output_tokens"]) for s in agent_steps),
                    "cache_creation_tokens": sum(
                        int(s["cache_creation_tokens"]) for s in agent_steps
                    ),
                    "cache_read_tokens": sum(
                        int(s["cache_read_tokens"]) for s in agent_steps
                    ),
                    "cost_usd": round(sum(float(s["cost_usd"]) for s in agent_steps), 6),
                },
                "judge_token_totals": {
                    "input_tokens": sum(int(s["input_tokens"]) for s in judge_steps),
                    "output_tokens": sum(int(s["output_tokens"]) for s in judge_steps),
                    "cost_usd": round(sum(float(s["cost_usd"]) for s in judge_steps), 6),
                },
                "step_trace": steps,
            }
        )

    # KPI helpers from summary JSON
    by_ver = {r["experiment_version"]: r for r in summary}
    v1, v2, v3 = by_ver["v1"], by_ver["v2"], by_ver["v3"]
    p50_cost_reduction_pct = round(
        100.0 * (v1["p50_cost_usd"] - v2["p50_cost_usd"]) / v1["p50_cost_usd"], 1
    )
    mean_cost_reduction_pct = round(
        100.0 * (v1["mean_cost_usd"] - v2["mean_cost_usd"]) / v1["mean_cost_usd"], 1
    )

    unit = pytest_report["summary"]
    unit_status = {
        "Passed": unit["passed"],
        "Failed": unit["failed"],
        "Skipped": unit["skipped"],
        "Partially Passed": 0,
    }

    return {
        "meta": {
            "repo": "enterprise-deal-desk",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git": git_meta,
            "models": {"agent": AGENT_MODEL, "judge": JUDGE_MODEL},
            "eval_window": {
                "min_created_at": timestamps[0],
                "max_created_at": timestamps[1],
            },
            "sources": {
                "benchmark_summary_json": "outputs/benchmark_summary.json",
                "benchmark_summary_csv": "outputs/benchmark_summary.csv",
                "difficulty_breakdown_csv": "outputs/difficulty_breakdown.csv",
                "case_level_evals_csv": "outputs/case_level_evals.csv",
                "database": "db/deal_desk.db",
            },
        },
        "kpis": {
            "unit_tests_passed": f"{unit['passed']}/{unit['passed'] + unit['failed'] + unit['skipped']} Passed",
            "unit_tests_raw": unit,
            "benchmark_runs": f"{len(cases)}/60",
            "v1_to_v2_pass_rate": f"{v1['deterministic_pass_rate_pct']}% → {v2['deterministic_pass_rate_pct']}%",
            "v1_to_v2_flag_f1": f"{v1['mean_flag_f1']:.4f} → {v2['mean_flag_f1']:.4f}",
            "v1_to_v2_p50_cost_reduction_pct": p50_cost_reduction_pct,
            "v1_p50_cost_usd": v1["p50_cost_usd"],
            "v2_p50_cost_usd": v2["p50_cost_usd"],
            "v2_cache_hit_rate_pct": v2["avg_cache_hit_rate_pct"],
            "v1_to_v2_mean_cost_reduction_pct": mean_cost_reduction_pct,
        },
        "status_breakdown": {
            "unit_tests": unit_status,
            "benchmark_runs": status_counts,
            "v3_producer_revision_skipped": critic_stats["v3"]["producer_revision_skipped"],
            "partial_pass_cases": [
                c["case_id"]
                for c in cases
                if c["experiment_version"] == "v1" and c["status"] == "Partially Passed"
            ],
        },
        "benchmark_summary": summary,
        "benchmark_summary_csv": summary_csv,
        "difficulty_breakdown": difficulty,
        "token_by_version": token_by_version,
        "critic_stats": critic_stats,
        "cases": cases,
        "pytest": pytest_report,
        "notable": {
            "v1_partial_pass_cases": ["GE-009", "GE-018", "GE-019"],
            "v3_revision_case": "GE-016",
            "v3_revision_rate_pct": v3["revision_trigger_rate_pct"],
            "v3_flag_precision": v3["mean_flag_precision"],
            "max_tokens_fix": "v3/GE-020 required max_tokens 2048 → 4096 after JSON truncation",
        },
    }


def _write_data_files(
    dataset: dict[str, Any],
    pytest_report: dict[str, Any],
    data_dir: Path,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "benchmark_summary.json",
        "benchmark_summary.csv",
        "difficulty_breakdown.csv",
        "case_level_evals.csv",
    ):
        shutil.copy2(OUTPUTS / name, data_dir / name)

    (data_dir / "unit_test_results.json").write_text(
        json.dumps(pytest_report, indent=2), encoding="utf-8"
    )
    manifest = {
        "generated_at": dataset["meta"]["generated_at"],
        "git": dataset["meta"]["git"],
        "models": dataset["meta"]["models"],
        "artifact_files": [
            "benchmark_summary.json",
            "benchmark_summary.csv",
            "difficulty_breakdown.csv",
            "case_level_evals.csv",
            "unit_test_results.json",
            "traceability_manifest.json",
            "dashboard_dataset.json",
        ],
        "source_mapping": dataset["meta"]["sources"],
        "reproduction": "python -m src.build_report",
        "case_count": len(dataset["cases"]),
        "unit_tests": dataset["pytest"]["summary"],
    }
    (data_dir / "traceability_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (data_dir / "dashboard_dataset.json").write_text(
        json.dumps(dataset, indent=2), encoding="utf-8"
    )


def _html_document(dataset: dict[str, Any]) -> str:
    payload = json.dumps(dataset, ensure_ascii=False)
    # Prevent </script> breakouts in embedded JSON
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e")
    return HTML_TEMPLATE.replace("@@EVAL_DATA@@", payload)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>enterprise-deal-desk — Evaluation Dashboard</title>
<style>
:root {
  --bg: #f4f1ea;
  --ink: #1c1917;
  --muted: #57534e;
  --card: #fffcf7;
  --line: #d6d3d1;
  --accent: #0f766e;
  --accent2: #b45309;
  --pass: #15803d;
  --partial: #a16207;
  --fail: #b91c1c;
  --skip: #57534e;
  --v1: #78716c;
  --v2: #0f766e;
  --v3: #1d4ed8;
  --shadow: 0 1px 2px rgba(28,25,22,.06), 0 8px 24px rgba(28,25,22,.06);
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --sans: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  --ui: "Avenir Next", "Segoe UI", system-ui, sans-serif;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; color: var(--ink); background:
    radial-gradient(1200px 600px at 10% -10%, #efe6d6 0%, transparent 55%),
    radial-gradient(900px 500px at 100% 0%, #d9ece8 0%, transparent 50%),
    var(--bg);
  font-family: var(--ui); line-height: 1.45;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 0 1.25rem 4rem; }
nav.top {
  position: sticky; top: 0; z-index: 40; backdrop-filter: blur(10px);
  background: rgba(244,241,234,.92); border-bottom: 1px solid var(--line);
}
nav.top .inner { display:flex; flex-wrap:wrap; gap:.75rem 1rem; align-items:center; justify-content:space-between; padding:.75rem 1.25rem; max-width:1180px; margin:0 auto; }
nav.top .brand { font-weight:700; letter-spacing:.01em; }
nav.top .links { display:flex; flex-wrap:wrap; gap:.65rem .9rem; font-size:.9rem; }
header.hero { padding: 2rem 0 1rem; }
header.hero h1 { font-family: var(--sans); font-size: clamp(1.8rem, 3vw, 2.6rem); margin:0 0 .4rem; line-height:1.15; }
header.hero .sub { color: var(--muted); max-width: 62ch; }
.meta-row { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1rem; }
.pill { display:inline-flex; align-items:center; gap:.35rem; border:1px solid var(--line); background:var(--card); padding:.25rem .6rem; border-radius:999px; font-size:.8rem; color:var(--muted); font-family:var(--mono); }
section { margin: 2.5rem 0; scroll-margin-top: 4.5rem; }
section h2 { font-family: var(--sans); font-size:1.65rem; margin:0 0 .75rem; border-bottom:1px solid var(--line); padding-bottom:.4rem; }
section h3 { margin:1.4rem 0 .6rem; font-size:1.05rem; }
.grid { display:grid; gap:1rem; }
.grid.kpis { grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
.card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; box-shadow:var(--shadow); }
.card .label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; }
.card .value { font-size:1.25rem; font-weight:700; margin-top:.25rem; }
.card .hint { color:var(--muted); font-size:.8rem; margin-top:.2rem; }
.badge { display:inline-block; padding:.15rem .5rem; border-radius:999px; font-size:.75rem; font-weight:600; }
.badge.Passed { background:#dcfce7; color:var(--pass); }
.badge.Partially\ Passed, .badge.partial { background:#fef3c7; color:var(--partial); }
.badge.Failed { background:#fee2e2; color:var(--fail); }
.badge.Skipped { background:#e7e5e4; color:var(--skip); }
table { width:100%; border-collapse:collapse; font-size:.9rem; background:var(--card); border:1px solid var(--line); border-radius:12px; overflow:hidden; }
th, td { padding:.55rem .65rem; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
th { background:#ebe6dc; font-size:.78rem; text-transform:uppercase; letter-spacing:.03em; color:var(--muted); }
tr:last-child td { border-bottom:none; }
.chart-card { padding:1rem; }
.chart-card svg { width:100%; height:auto; display:block; }
.muted { color:var(--muted); }
.mono { font-family:var(--mono); font-size:.82rem; }
.controls { display:flex; flex-wrap:wrap; gap:.6rem; margin: .8rem 0 1rem; }
.controls input, .controls select {
  border:1px solid var(--line); background:#fff; border-radius:8px; padding:.45rem .6rem; font:inherit;
}
.drawer {
  display:none; margin-top:.6rem; padding:1rem; border:1px dashed var(--line); border-radius:12px; background:#fffef9;
}
.drawer.open { display:block; }
pre.json {
  white-space:pre-wrap; word-break:break-word; background:#1c1917; color:#f5f5f4;
  padding:1rem; border-radius:10px; font-size:.75rem; max-height:320px; overflow:auto;
}
.two-col { display:grid; grid-template-columns:1.2fr .8fr; gap:1rem; }
@media (max-width: 900px) { .two-col { grid-template-columns:1fr; } }
.list-tight li { margin:.25rem 0; }
footer { margin-top:3rem; padding-top:1rem; border-top:1px solid var(--line); color:var(--muted); font-size:.85rem; }
.clickable { cursor:pointer; }
.clickable:hover { background:#f7f3ea; }
</style>
</head>
<body>
<nav class="top">
  <div class="inner">
    <div class="brand">enterprise-deal-desk</div>
    <div class="links">
      <a href="#overview">Overview</a>
      <a href="#iterations">Iterations</a>
      <a href="#economics">Economics</a>
      <a href="#difficulty">Difficulty</a>
      <a href="#walkthrough">Walkthrough</a>
      <a href="#explorer">Explorer</a>
      <a href="#traceability">Traceability</a>
    </div>
  </div>
</nav>

<div class="wrap">
<header class="hero">
  <h1>Evaluation Dashboard</h1>
  <p class="sub">Self-contained report of the framework-free Anthropic Deal-Desk agent and the 60-run <code>v1</code>/<code>v2</code>/<code>v3</code> empirical evaluation. Opens via <code>file://</code> with zero network dependencies.</p>
  <div class="meta-row" id="meta-pills"></div>
</header>

<section id="overview">
  <h2>1. Executive Overview</h2>
  <div class="card" style="margin-bottom:1rem;">
    <p><strong>Architecture implemented:</strong> a framework-free Anthropic Messages API tool-use loop; deterministic Python/SQLite SaaS unit-economics (18% list-ARR COGS, $150/hr services, Net-60/90 working-capital penalties, 72% hard floor); local 384-d deterministic dense embeddings (hashing bag-of-words) + BM25 Hybrid RAG with Reciprocal Rank Fusion over Deal-Desk policies &amp; CUAD-style clauses; Anthropic prompt caching on system + tools; Evaluator-Optimizer critic loop (<code>v3</code>); and cross-model LLM-as-a-Judge on independent <code>claude-opus-4-6</code> (agent: <code>claude-sonnet-4-6</code>).</p>
  </div>
  <div class="grid kpis" id="kpi-cards"></div>
  <h3>Status breakdown</h3>
  <div class="two-col">
    <div class="card" id="status-tables"></div>
    <div class="card">
      <div class="label">Interpretation</div>
      <ul class="list-tight">
        <li><span class="badge Passed">Passed</span> — <code>deterministic_pass == 1</code> (tier + margin + full flag recall).</li>
        <li><span class="badge partial">Partially Passed</span> — tier &amp; margin exact, but <code>flag_recall &lt; 1</code> (v1 GE-009 / GE-018 / GE-019).</li>
        <li><span class="badge Failed">Failed</span> — tier or margin miss.</li>
        <li><span class="badge Skipped">Skipped</span> — unit-test skips; v3 <code>producer_revision</code> skipped when critic approved first draft.</li>
      </ul>
    </div>
  </div>
</section>

<section id="iterations">
  <h2>2. Iteration Progression</h2>
  <div class="card" style="overflow:auto; margin-bottom:1rem;" id="summary-table"></div>
  <div class="card chart-card"><h3>Chart 1 — Quality progression</h3><div id="chart-quality"></div></div>
  <div class="card chart-card" style="margin-top:1rem;"><h3>Chart 2 — Opus judge rubric (1–5)</h3><div id="chart-rubric"></div></div>
</section>

<section id="economics">
  <h2>3. Cost, Token &amp; Caching Economics</h2>
  <p class="muted">Agent unit economics exclude <code>llm_judge</code> ledger rows. Prompt caching + tighter ACI tooling made <strong>v2 ~37% cheaper on mean cost</strong> and <strong>~50% cheaper at p50</strong> vs v1; v3 adds critic overhead.</p>
  <div class="card chart-card"><div id="chart-cost"></div></div>
  <div class="card" style="margin-top:1rem; overflow:auto;" id="token-table"></div>
</section>

<section id="difficulty">
  <h2>4. Scenario Difficulty &amp; Failure Analysis</h2>
  <div class="card chart-card"><div id="chart-difficulty"></div></div>
  <div class="card" style="margin-top:1rem; overflow:auto;" id="difficulty-table"></div>
  <div class="card" style="margin-top:1rem;" id="failure-notes"></div>
</section>

<section id="walkthrough">
  <h2>5. Engineering &amp; Evaluation Walkthrough</h2>
  <div class="card">
    <ol>
      <li><strong>Increment 1 — Deterministic foundation.</strong> Isolated SaaS margin math and approval tiers in Python/SQLite so golden cases are mathematically exact before any LLM reasoning.</li>
      <li><strong>Increment 2 — ACI tools + Hybrid RAG + cost ledger.</strong> Poka-yoked schemas force decimal discounts, payment-term enums, and canonical legal flags; BM25 + dense cosine RRF retrieves policy/CUAD chunks; pinned Anthropic rate card (as of 2026-11-21) logs USD/latency.</li>
      <li><strong>Increment 3 — Raw Messages agent.</strong> Versioned prompts, prompt caching, and an optional Evaluator-Optimizer critic produce a structured <code>DealDeskExecutiveMemo</code>.</li>
      <li><strong>Increment 4 — Eval harness.</strong> Deterministic graders + independent Opus judge with resumable <code>eval_results</code> persistence.</li>
      <li><strong>Increment 5 — 60-run empirical suite.</strong> v1→v2 closed the flag-taxonomy gap (85%→100% pass) while cutting p50 cost ~50% via caching. v3 kept 100% pass but only revised 5% of cases (GE-016 compound trap), so <strong>v2 is the Pareto-optimal default</strong>; v3 is reserved for high-stakes compound stacks.</li>
      <li><strong>Operational fix.</strong> v3/GE-020 truncated JSON at <code>max_tokens=2048</code>; raising to 4096 completed the suite.</li>
      <li><strong>Limitations.</strong> Local hashing embeddings (not neural); seeded SQLite warehouse; single-revision critic cap; judge spend excluded from agent unit economics by design.</li>
    </ol>
  </div>
</section>

<section id="explorer">
  <h2>6. Interactive Case-Level Explorer</h2>
  <div class="controls">
    <select id="f-version"><option value="All">All versions</option><option>v1</option><option>v2</option><option>v3</option></select>
    <select id="f-diff"><option value="All">All difficulties</option><option>standard</option><option>edge_margin</option><option>edge_legal</option><option>compound_trap</option></select>
    <select id="f-status"><option value="All">All statuses</option><option>Passed</option><option>Partially Passed</option><option>Failed</option></select>
    <input id="f-search" type="search" placeholder="Search case id or account…" style="min-width:220px;" />
  </div>
  <div class="card" style="overflow:auto;">
    <table>
      <thead>
        <tr>
          <th>Version</th><th>Case</th><th>Account</th><th>Difficulty</th><th>Status</th>
          <th>Tier</th><th>Margin</th><th>Flag F1</th><th>Judge</th><th>Cost</th><th>Rev</th>
        </tr>
      </thead>
      <tbody id="case-tbody"></tbody>
    </table>
  </div>
  <div id="case-drawer" class="drawer"></div>
</section>

<section id="traceability">
  <h2>7. Traceability &amp; Raw Artifacts</h2>
  <div class="card" id="trace-box"></div>
</section>

<footer>
  Generated by <code>python -m src.build_report</code>. Self-contained HTML — no CDNs, no localhost server required.
</footer>
</div>

<script id="eval-data" type="application/json">@@EVAL_DATA@@</script>
<script>
(function() {
  const DATA = JSON.parse(document.getElementById('eval-data').textContent);
  const VERSIONS = ['v1','v2','v3'];
  const COLORS = {v1:'#78716c', v2:'#0f766e', v3:'#1d4ed8'};

  function el(tag, attrs, children) {
    const n = document.createElement(tag);
    if (attrs) Object.entries(attrs).forEach(([k,v]) => {
      if (k === 'className') n.className = v;
      else if (k === 'text') n.textContent = v;
      else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2).toLowerCase(), v);
      else n.setAttribute(k, v);
    });
    (children || []).forEach(c => n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c));
    return n;
  }

  // Meta pills
  const meta = DATA.meta;
  const pills = document.getElementById('meta-pills');
  [
    'repo: ' + meta.repo,
    'commit: ' + (meta.git.commit_sha || 'n/a'),
    'branch: ' + (meta.git.branch || 'n/a'),
    'generated: ' + meta.generated_at,
    'agent: ' + meta.models.agent,
    'judge: ' + meta.models.judge
  ].forEach(t => pills.appendChild(el('span', {className:'pill', text:t})));

  // KPIs
  const k = DATA.kpis;
  const kpiRoot = document.getElementById('kpi-cards');
  [
    ['Unit Tests', k.unit_tests_passed, 'pytest -v (read-only capture)'],
    ['Benchmark Runs', k.benchmark_runs, '20 cases × 3 versions'],
    ['v1 → v2 Pass Rate', k.v1_to_v2_pass_rate, 'Deterministic pass'],
    ['v1 → v2 Legal Flag F1', k.v1_to_v2_flag_f1, 'Canonical redline taxonomy'],
    ['v1 → v2 p50 Cost', '-' + k.v1_to_v2_p50_cost_reduction_pct + '%', '$' + k.v1_p50_cost_usd.toFixed(4) + ' → $' + k.v2_p50_cost_usd.toFixed(4)],
    ['v2 Cache Hit Rate', k.v2_cache_hit_rate_pct.toFixed(1) + '%', 'Prompt caching savings']
  ].forEach(([label, value, hint]) => {
    kpiRoot.appendChild(el('div', {className:'card'}, [
      el('div', {className:'label', text:label}),
      el('div', {className:'value', text:String(value)}),
      el('div', {className:'hint', text:hint})
    ]));
  });

  // Status tables
  function statusTable(title, counts) {
    const rows = Object.entries(counts).map(([s,n]) =>
      el('tr', {}, [el('td', {}, [el('span', {className:'badge ' + (s === 'Partially Passed' ? 'partial' : s), text:s})]), el('td', {text:String(n)})])
    );
    return el('div', {}, [
      el('div', {className:'label', text:title}),
      el('table', {}, [el('thead', {}, [el('tr', {}, [el('th',{text:'Status'}), el('th',{text:'Count'})])]), el('tbody', {}, rows)])
    ]);
  }
  const st = document.getElementById('status-tables');
  st.appendChild(statusTable('Unit tests', DATA.status_breakdown.unit_tests));
  st.appendChild(el('div', {style:'height:0.8rem'}));
  st.appendChild(statusTable('Benchmark runs (60)', DATA.status_breakdown.benchmark_runs));
  st.appendChild(el('p', {className:'muted mono', text:'v3 producer_revision skipped (critic approved first pass): ' + DATA.status_breakdown.v3_producer_revision_skipped}));

  // Summary comparison table
  const sumRoot = document.getElementById('summary-table');
  const fields = [
    ['experiment_version','Version'],
    ['cases_evaluated','Cases'],
    ['deterministic_pass_rate_pct','Det. Pass %'],
    ['tier_accuracy_pct','Tier %'],
    ['margin_accuracy_pct','Margin %'],
    ['mean_flag_recall','Flag Recall'],
    ['mean_flag_precision','Flag Prec.'],
    ['mean_flag_f1','Flag F1'],
    ['mean_judge_score','Judge'],
    ['revision_trigger_rate_pct','Rev %'],
    ['mean_cost_usd','Mean $'],
    ['p50_cost_usd','p50 $'],
    ['avg_cache_hit_rate_pct','Cache %'],
    ['p50_latency_s','p50 Lat (s)']
  ];
  sumRoot.appendChild(el('table', {}, [
    el('thead', {}, [el('tr', {}, fields.map(([,h]) => el('th',{text:h})))]),
    el('tbody', {}, DATA.benchmark_summary.map(row => el('tr', {}, fields.map(([k]) => {
      let v = row[k];
      if (typeof v === 'number' && k.includes('cost')) v = v.toFixed(4);
      else if (typeof v === 'number' && (k.includes('flag') || k.includes('cache') || k.includes('latency') || k.includes('judge'))) v = Number(v).toFixed(3);
      return el('td', {text:String(v)});
    }))))
  ]));

  // SVG helpers
  function svgEl(name, attrs) {
    const n = document.createElementNS('http://www.w3.org/2000/svg', name);
    Object.entries(attrs || {}).forEach(([k,v]) => n.setAttribute(k, v));
    return n;
  }
  function groupedBars(containerId, seriesDefs, ymax, yLabel) {
    // seriesDefs: [{key,label, getter(row)->number}]
    const root = document.getElementById(containerId);
    root.innerHTML = '';
    const W=860, H=340, padL=58, padR=20, padT=24, padB=70;
    const svg = svgEl('svg', {viewBox:`0 0 ${W} ${H}`, role:'img', 'aria-label':yLabel});
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const groups = VERSIONS;
    const gW = plotW / groups.length;
    const barW = Math.min(18, (gW - 20) / seriesDefs.length);

    // axes
    svg.appendChild(svgEl('line', {x1:padL,y1:padT,x2:padL,y2:padT+plotH, stroke:'#a8a29e', 'stroke-width':'1'}));
    svg.appendChild(svgEl('line', {x1:padL,y1:padT+plotH,x2:padL+plotW,y2:padT+plotH, stroke:'#a8a29e', 'stroke-width':'1'}));
    for (let i=0;i<=5;i++){
      const y = padT + plotH - (i/5)*plotH;
      const val = (ymax * i/5);
      svg.appendChild(svgEl('line', {x1:padL,y1:y,x2:padL+plotW,y2:y, stroke:'#e7e5e4', 'stroke-width':'1'}));
      const t = svgEl('text', {x:padL-8, y:y+4, 'text-anchor':'end', 'font-size':'11', fill:'#57534e'});
      t.textContent = String(Math.round(val*10)/10);
      svg.appendChild(t);
    }
    const ytitle = svgEl('text', {x:16, y:16, 'font-size':'12', fill:'#57534e'});
    ytitle.textContent = yLabel;
    svg.appendChild(ytitle);

    const by = Object.fromEntries(DATA.benchmark_summary.map(r => [r.experiment_version, r]));
    groups.forEach((ver, gi) => {
      const gx = padL + gi*gW + 16;
      seriesDefs.forEach((s, si) => {
        const val = s.get(by[ver]);
        const h = (val / ymax) * plotH;
        const x = gx + si*(barW+3);
        const y = padT + plotH - h;
        svg.appendChild(svgEl('rect', {x, y, width:barW, height:Math.max(0,h), fill:s.color || COLORS[ver], rx:'2'}));
        const lab = svgEl('text', {x:x+barW/2, y:y-4, 'text-anchor':'middle', 'font-size':'9', fill:'#44403c'});
        lab.textContent = (Math.round(val*10)/10).toString();
        svg.appendChild(lab);
      });
      const tl = svgEl('text', {x:gx + (seriesDefs.length*(barW+3))/2, y:padT+plotH+18, 'text-anchor':'middle', 'font-size':'13', 'font-weight':'700', fill:COLORS[ver]});
      tl.textContent = ver;
      svg.appendChild(tl);
    });

    // legend
    seriesDefs.forEach((s, i) => {
      const x = padL + i*140;
      svg.appendChild(svgEl('rect', {x, y:H-28, width:12, height:12, fill:s.color, rx:'2'}));
      const t = svgEl('text', {x:x+18, y:H-18, 'font-size':'11', fill:'#44403c'});
      t.textContent = s.label;
      svg.appendChild(t);
    });
    root.appendChild(svg);
  }

  groupedBars('chart-quality', [
    {label:'Det. Pass %', color:'#0f766e', get:r => r.deterministic_pass_rate_pct},
    {label:'Tier Acc. %', color:'#1d4ed8', get:r => r.tier_accuracy_pct},
    {label:'Margin Acc. %', color:'#b45309', get:r => r.margin_accuracy_pct},
    {label:'Flag Recall %', color:'#7c3aed', get:r => r.mean_flag_recall*100},
    {label:'Flag F1 %', color:'#be123c', get:r => r.mean_flag_f1*100},
    {label:'Judge % of 5', color:'#334155', get:r => (r.mean_judge_score/5)*100}
  ], 100, 'Percent');

  groupedBars('chart-rubric', [
    {label:'Commercial', color:'#0f766e', get:r => r.judge_commercial_accuracy},
    {label:'Policy', color:'#1d4ed8', get:r => r.judge_policy_grounding},
    {label:'Clarity', color:'#b45309', get:r => r.judge_executive_clarity},
    {label:'Actionability', color:'#7c3aed', get:r => r.judge_actionability}
  ], 5, 'Score (1–5)');

  // Cost chart: p10/p50/p90 + stacked token mix as second panel
  (function(){
    const root = document.getElementById('chart-cost');
    const W=900, H=360, padL=60, padR=20, padT=30, padB=80;
    const svg = svgEl('svg', {viewBox:`0 0 ${W} ${H}`});
    const plotW=W-padL-padR, plotH=H-padT-padB;
    const by = Object.fromEntries(DATA.benchmark_summary.map(r => [r.experiment_version, r]));
    const ymax = Math.max(...VERSIONS.map(v => by[v].p90_cost_usd)) * 1.15;
    svg.appendChild(svgEl('line', {x1:padL,y1:padT+plotH,x2:padL+plotW,y2:padT+plotH,stroke:'#a8a29e'}));
    for (let i=0;i<=4;i++){
      const y=padT+plotH-(i/4)*plotH;
      svg.appendChild(svgEl('line',{x1:padL,y1:y,x2:padL+plotW,y2:y,stroke:'#e7e5e4'}));
      const t=svgEl('text',{x:padL-8,y:y+4,'text-anchor':'end','font-size':'11',fill:'#57534e'});
      t.textContent = '$'+(ymax*i/4).toFixed(3);
      svg.appendChild(t);
    }
    const title=svgEl('text',{x:padL,y:18,'font-size':'13',fill:'#44403c','font-weight':'700'});
    title.textContent='USD cost percentiles per run (agent only)';
    svg.appendChild(title);
    const metrics=[['p10_cost_usd','#a8a29e'],['p50_cost_usd','#0f766e'],['p90_cost_usd','#b45309']];
    const gW=plotW/3;
    VERSIONS.forEach((ver,gi)=>{
      const gx=padL+gi*gW+24;
      metrics.forEach(([key,color],si)=>{
        const val=by[ver][key];
        const h=(val/ymax)*plotH;
        const x=gx+si*28;
        const y=padT+plotH-h;
        svg.appendChild(svgEl('rect',{x,y,width:22,height:h,fill:color,rx:'2'}));
        const lab=svgEl('text',{x:x+11,y:y-4,'text-anchor':'middle','font-size':'9',fill:'#444'});
        lab.textContent=val.toFixed(3);
        svg.appendChild(lab);
      });
      const tl=svgEl('text',{x:gx+40,y:padT+plotH+20,'text-anchor':'middle','font-size':'13','font-weight':'700',fill:COLORS[ver]});
      tl.textContent=ver;
      svg.appendChild(tl);
    });
    [['p10','#a8a29e'],['p50','#0f766e'],['p90','#b45309']].forEach(([lab,col],i)=>{
      const x=padL+i*90;
      svg.appendChild(svgEl('rect',{x,y:H-28,width:12,height:12,fill:col,rx:'2'}));
      const t=svgEl('text',{x:x+18,y:H-18,'font-size':'11',fill:'#444'}); t.textContent=lab; svg.appendChild(t);
    });
    const note=svgEl('text',{x:padL+300,y:H-18,'font-size':'11',fill:'#57534e'});
    note.textContent=`v1→v2 mean −${DATA.kpis.v1_to_v2_mean_cost_reduction_pct}% · p50 −${DATA.kpis.v1_to_v2_p50_cost_reduction_pct}% · v3 critic marginal overhead`;
    svg.appendChild(note);
    root.appendChild(svg);

    // token mix table
    const tt=document.getElementById('token-table');
    const tb=el('tbody', {}, VERSIONS.map(ver => {
      const t=DATA.token_by_version[ver];
      const total=t.input_tokens+t.output_tokens+t.cache_creation_tokens+t.cache_read_tokens;
      return el('tr', {}, [
        el('td',{text:ver}),
        el('td',{text:String(t.input_tokens)}),
        el('td',{text:String(t.cache_creation_tokens)}),
        el('td',{text:String(t.cache_read_tokens)}),
        el('td',{text:String(t.output_tokens)}),
        el('td',{text:String(total)}),
        el('td',{text:String(t.agent_api_calls)}),
        el('td',{text:(DATA.critic_stats[ver].revision_applied_count + '/' + DATA.critic_stats[ver].cases)})
      ]);
    }));
    tt.appendChild(el('table', {}, [
      el('thead', {}, [el('tr', {}, ['Version','Uncached Input Tok','Cache Creation','Cache Read','Output Tok','Total Tok','Agent API Calls','Revisions Applied'].map(h=>el('th',{text:h})))]),
      tb
    ]));
  })();

  // Difficulty chart + table
  (function(){
    const root=document.getElementById('chart-difficulty');
    const diffs=['standard','edge_margin','edge_legal','compound_trap'];
    const W=900,H=320,padL=60,padR=20,padT=28,padB=70;
    const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`});
    const plotW=W-padL-padR, plotH=H-padT-padB;
    const rows=DATA.difficulty_breakdown;
    const key=(v,d)=>rows.find(r=>r.experiment_version===v && r.difficulty===d);
    svg.appendChild(svgEl('line',{x1:padL,y1:padT+plotH,x2:padL+plotW,y2:padT+plotH,stroke:'#a8a29e'}));
    for(let i=0;i<=4;i++){
      const y=padT+plotH-(i/4)*plotH;
      svg.appendChild(svgEl('line',{x1:padL,y1:y,x2:padL+plotW,y2:y,stroke:'#e7e5e4'}));
      const t=svgEl('text',{x:padL-8,y:y+4,'text-anchor':'end','font-size':'11',fill:'#57534e'});
      t.textContent=String(i*25); svg.appendChild(t);
    }
    const title=svgEl('text',{x:padL,y:18,'font-size':'13','font-weight':'700',fill:'#444'});
    title.textContent='Deterministic pass rate % by difficulty'; svg.appendChild(title);
    const gW=plotW/diffs.length;
    diffs.forEach((d,di)=>{
      VERSIONS.forEach((v,vi)=>{
        const r=key(v,d); const val=r?Number(r.deterministic_pass_rate_pct):0;
        const h=(val/100)*plotH; const x=padL+di*gW+18+vi*22; const y=padT+plotH-h;
        svg.appendChild(svgEl('rect',{x,y,width:18,height:h,fill:COLORS[v],rx:'2'}));
        const lab=svgEl('text',{x:x+9,y:y-3,'text-anchor':'middle','font-size':'9',fill:'#444'});
        lab.textContent=String(val); svg.appendChild(lab);
      });
      const tl=svgEl('text',{x:padL+di*gW+gW/2,y:padT+plotH+22,'text-anchor':'middle','font-size':'11',fill:'#444'});
      tl.textContent=d; svg.appendChild(tl);
    });
    VERSIONS.forEach((v,i)=>{
      const x=padL+i*70; svg.appendChild(svgEl('rect',{x,y:H-24,width:12,height:12,fill:COLORS[v],rx:'2'}));
      const t=svgEl('text',{x:x+18,y:H-14,'font-size':'11',fill:'#444'}); t.textContent=v; svg.appendChild(t);
    });
    root.appendChild(svg);

    const dt=document.getElementById('difficulty-table');
    dt.appendChild(el('table', {}, [
      el('thead', {}, [el('tr', {}, ['Version','Difficulty','Cases','Det. Pass %','Mean Judge','p50 Cost $'].map(h=>el('th',{text:h})))]),
      el('tbody', {}, rows.map(r => el('tr', {}, [
        el('td',{text:r.experiment_version}), el('td',{text:r.difficulty}), el('td',{text:r.cases}),
        el('td',{text:r.deterministic_pass_rate_pct}), el('td',{text:r.mean_judge_score}),
        el('td',{text:Number(r.p50_cost_usd).toFixed(4)})
      ])))
    ]));

    document.getElementById('failure-notes').innerHTML =
      '<p><strong>v1 drops:</strong> <code>edge_legal</code> 50% and <code>compound_trap</code> 66.7% pass — GE-009 / GE-018 / GE-019 used non-canonical or prose-annotated flag strings while tier+margin stayed exact (Partially Passed).</p>' +
      '<p><strong>v2:</strong> 100% pass across every difficulty bucket; legal redline F1 = 1.000.</p>' +
      '<p><strong>v3:</strong> Critic revision on <code>' + DATA.notable.v3_revision_case + '</code> (compound_trap); overall revision rate ' + DATA.notable.v3_revision_rate_pct + '%. Minor false-positive precision dip to ' + DATA.notable.v3_flag_precision + '. ' + DATA.notable.max_tokens_fix + '.</p>';
  })();

  // Explorer
  const tbody = document.getElementById('case-tbody');
  const drawer = document.getElementById('case-drawer');
  function renderCases() {
    const ver = document.getElementById('f-version').value;
    const diff = document.getElementById('f-diff').value;
    const status = document.getElementById('f-status').value;
    const q = (document.getElementById('f-search').value || '').toLowerCase();
    tbody.innerHTML = '';
    DATA.cases.filter(c => {
      if (ver !== 'All' && c.experiment_version !== ver) return false;
      if (diff !== 'All' && c.difficulty !== diff) return false;
      if (status !== 'All' && c.status !== status) return false;
      if (q && !(c.case_id.toLowerCase().includes(q) || (c.account_name||'').toLowerCase().includes(q))) return false;
      return true;
    }).forEach(c => {
      const tr = el('tr', {className:'clickable'}, [
        el('td',{text:c.experiment_version}),
        el('td',{className:'mono', text:c.case_id}),
        el('td',{text:c.account_name || '—'}),
        el('td',{text:c.difficulty}),
        el('td', {}, [el('span',{className:'badge ' + (c.status==='Partially Passed'?'partial':c.status), text:c.status})]),
        el('td',{text:c.actual_approval_tier || '—'}),
        el('td',{text:String(c.actual_gross_margin_pct)}),
        el('td',{text:Number(c.flag_f1).toFixed(3)}),
        el('td',{text:String(c.judge_overall_score)}),
        el('td',{text:'$'+Number(c.total_cost_usd).toFixed(4)}),
        el('td',{text:c.revision_applied ? 'Yes' : 'No'})
      ]);
      tr.addEventListener('click', () => showCase(c));
      tbody.appendChild(tr);
    });
  }
  function showCase(c) {
    drawer.className = 'drawer open';
    drawer.innerHTML = '';
    drawer.appendChild(el('h3',{text:`${c.experiment_version} / ${c.case_id} — ${c.account_name}`}));
    drawer.appendChild(el('p',{},[
      el('span',{className:'badge ' + (c.status==='Partially Passed'?'partial':c.status), text:c.status}),
      document.createTextNode(' · difficulty ' + c.difficulty + ' · run_id ' + c.run_id)
    ]));
    const grid = el('div',{className:'two-col'});
    const left = el('div',{});
    left.appendChild(el('h3',{text:'Expected vs Actual'}));
    left.appendChild(el('table',{},[
      el('tbody',{},[
        el('tr',{},[el('th',{text:'Field'}),el('th',{text:'Expected'}),el('th',{text:'Actual'})]),
        el('tr',{},[el('td',{text:'Approval tier'}),el('td',{text:String(c.expected_approval_tier)}),el('td',{text:String(c.actual_approval_tier)})]),
        el('tr',{},[el('td',{text:'Gross margin %'}),el('td',{text:String(c.expected_gross_margin_pct)}),el('td',{text:String(c.actual_gross_margin_pct)})]),
        el('tr',{},[el('td',{text:'Risk flags'}),el('td',{className:'mono',text:JSON.stringify(c.expected_risk_flags)}),el('td',{className:'mono',text:JSON.stringify(c.actual_risk_flags)})]),
      ])
    ]));
    left.appendChild(el('h3',{text:'Opus Judge'}));
    left.appendChild(el('p',{text:`Overall ${c.judge_overall_score} · C ${c.judge_commercial_accuracy} / P ${c.judge_policy_grounding} / E ${c.judge_executive_clarity} / A ${c.judge_actionability}`}));
    left.appendChild(el('p',{className:'muted', text:c.judge_rationale || ''}));
    left.appendChild(el('h3',{text:'Tokens & Cost (agent)'}));
    left.appendChild(el('pre',{className:'json', text:JSON.stringify({agent:c.agent_token_totals, judge:c.judge_token_totals, cache_hit_rate_pct:c.cache_hit_rate_pct, total_latency_ms:c.total_latency_ms}, null, 2)}));
    const right = el('div',{});
    right.appendChild(el('h3',{text:'Deal payload'}));
    right.appendChild(el('pre',{className:'json', text:JSON.stringify(c.deal_payload, null, 2)}));
    right.appendChild(el('h3',{text:'Executive memo'}));
    right.appendChild(el('pre',{className:'json', text:JSON.stringify(c.memo, null, 2)}));
    grid.appendChild(left); grid.appendChild(right);
    drawer.appendChild(grid);
    drawer.scrollIntoView({behavior:'smooth', block:'nearest'});
  }
  ['f-version','f-diff','f-status','f-search'].forEach(id => document.getElementById(id).addEventListener('input', renderCases));
  renderCases();

  // Traceability
  const trace = document.getElementById('trace-box');
  trace.innerHTML =
    `<p><strong>Git:</strong> <code>${meta.git.commit_sha}</code> on <code>${meta.git.branch}</code> (${meta.git.commit_timestamp || 'n/a'})</p>` +
    `<p><strong>Reproduction:</strong> <code>python -m src.build_report</code></p>` +
    `<p><strong>Source artifacts:</strong></p><ul>` +
    Object.entries(meta.sources).map(([k,v]) => `<li>${k}: <code>${v}</code></li>`).join('') +
    `</ul><p><strong>Copied for inspection:</strong> <code>results/data/benchmark_summary.json</code>, <code>benchmark_summary.csv</code>, <code>difficulty_breakdown.csv</code>, <code>case_level_evals.csv</code>, <code>unit_test_results.json</code>, <code>traceability_manifest.json</code>, <code>dashboard_dataset.json</code>.</p>` +
    `<p class="muted">Charts and tables above are computed only from the embedded dataset (no network fetches).</p>`;
})();
</script>
</body>
</html>
"""


def build(output_dir: Path | None = None) -> dict[str, Any]:
    out_root = Path(output_dir) if output_dir else (ROOT / "results")
    data_dir = out_root / "data"

    if not (OUTPUTS / "benchmark_summary.json").exists():
        raise FileNotFoundError(
            "Missing outputs/benchmark_summary.json — run the Increment 5 benchmark first."
        )
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing database at {DB_PATH}")

    git_meta = _git_meta()
    pytest_report = _run_pytest()
    dataset = _assemble_dataset(pytest_report, git_meta)
    out_root.mkdir(parents=True, exist_ok=True)
    _write_data_files(dataset, pytest_report, data_dir)

    index_path = out_root / "index.html"
    index_path.write_text(_html_document(dataset), encoding="utf-8")
    return {
        "index": index_path,
        "data_dir": data_dir,
        "cases": len(dataset["cases"]),
        "unit_passed": pytest_report["summary"]["passed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build self-contained HTML eval dashboard")
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    info = build(Path(args.output_dir))
    print(f"Wrote {info['index']}")
    print(f"Wrote data dir {info['data_dir']}")
    print(f"Embedded cases: {info['cases']}")
    print(f"Unit tests captured: {info['unit_passed']} passed")


if __name__ == "__main__":
    main()
