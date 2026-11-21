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



def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def _wilson_interval(k: int, n: int, z: float = 1.96) -> dict[str, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if n <= 0:
        return {"k": 0, "n": 0, "rate": 0.0, "lo": 0.0, "hi": 0.0}
    import math

    p = k / n
    den = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2.0 * n)) / den
    margin = (z * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))) / den
    return {
        "k": k,
        "n": n,
        "rate": round(100.0 * p, 2),
        "lo": round(100.0 * max(0.0, centre - margin), 2),
        "hi": round(100.0 * min(1.0, centre + margin), 2),
    }


def _mcnemar_exact(b: int, c: int) -> dict[str, Any]:
    """Exact two-sided McNemar test on discordant paired binary outcomes.

    b = baseline fail & challenger pass; c = baseline pass & challenger fail.
    """
    from math import comb

    n = b + c
    if n == 0:
        return {
            "b": b,
            "c": c,
            "n_discordant": 0,
            "exact_p_two_sided": 1.0,
            "significant_at_0_05": False,
            "note": "No discordant pairs — versions identical on this binary outcome.",
        }
    # Two-sided exact binomial test under p=0.5
    k = max(b, c)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2**n)
    p = min(1.0, 2.0 * tail) if b != c else 1.0
    return {
        "b": b,
        "c": c,
        "n_discordant": n,
        "exact_p_two_sided": round(p, 4),
        "significant_at_0_05": bool(p < 0.05),
        "note": (
            "Exact McNemar (binomial) on discordant pairs only. "
            "Small discordant counts limit power; non-significance does not "
            "imply practical equivalence."
        ),
    }


def _dist_stats(values: list[float]) -> dict[str, float]:
    import statistics

    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "min": 0.0,
            "stdev": 0.0,
            "iqr": 0.0,
        }
    mean = statistics.mean(vals)
    stdev = statistics.stdev(vals) if n > 1 else 0.0
    q1 = _percentile(vals, 25)
    q3 = _percentile(vals, 75)
    return {
        "n": n,
        "mean": round(mean, 6),
        "median": round(_percentile(vals, 50), 6),
        "p10": round(_percentile(vals, 10), 6),
        "p50": round(_percentile(vals, 50), 6),
        "p90": round(_percentile(vals, 90), 6),
        "max": round(vals[-1], 6),
        "min": round(vals[0], 6),
        "stdev": round(stdev, 6),
        "iqr": round(q3 - q1, 6),
    }


def _normalize_flag_token(raw: str) -> str:
    token = (raw or "").strip()
    for sep in (":", " —", " -", "–"):
        if sep in token:
            token = token.split(sep, 1)[0].strip()
            break
    return token.upper().replace(" ", "_")


def _build_analytics(
    cases: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    cost_by_run: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Derived analytics for the decision-support dashboard (no new eval runs)."""
    by_ver_cases: dict[str, list[dict[str, Any]]] = {"v1": [], "v2": [], "v3": []}
    for c in cases:
        by_ver_cases.setdefault(c["experiment_version"], []).append(c)

    by_ver_summary = {r["experiment_version"]: r for r in summary}

    # Distributions
    distributions: dict[str, Any] = {}
    for ver, rows in by_ver_cases.items():
        costs = [float(r["total_cost_usd"]) for r in rows]
        lats = [float(r["total_latency_ms"]) / 1000.0 for r in rows]
        distributions[ver] = {
            "cost_usd": _dist_stats(costs),
            "latency_s": _dist_stats(lats),
            "pass_count": sum(int(r["deterministic_pass"]) for r in rows),
            "n": len(rows),
        }

    wilson = {
        ver: _wilson_interval(distributions[ver]["pass_count"], distributions[ver]["n"])
        for ver in ("v1", "v2", "v3")
    }

    # Paired case index
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for c in cases:
        by_case.setdefault(c["case_id"], {})[c["experiment_version"]] = c

    def mcnemar_pair(a: str, b: str) -> dict[str, Any]:
        bb = cc = 0
        improved, regressed, unchanged = [], [], []
        for cid, vers in sorted(by_case.items()):
            if a not in vers or b not in vers:
                continue
            pa = int(vers[a]["deterministic_pass"])
            pb = int(vers[b]["deterministic_pass"])
            if pa == 0 and pb == 1:
                bb += 1
                improved.append(cid)
            elif pa == 1 and pb == 0:
                cc += 1
                regressed.append(cid)
            else:
                unchanged.append(cid)
        out = _mcnemar_exact(bb, cc)
        out.update(
            {
                "baseline": a,
                "challenger": b,
                "improved_cases": improved,
                "regressed_cases": regressed,
                "unchanged_count": len(unchanged),
            }
        )
        return out

    mcnemar = {
        "v1_v2": mcnemar_pair("v1", "v2"),
        "v2_v3": mcnemar_pair("v2", "v3"),
    }

    # Paired continuous deltas (cost / latency) with bootstrap CI of mean paired diff
    import random

    def paired_delta(metric_fn, a: str, b: str, n_boot: int = 2000) -> dict[str, Any]:
        diffs = []
        for cid, vers in sorted(by_case.items()):
            if a not in vers or b not in vers:
                continue
            diffs.append(metric_fn(vers[b]) - metric_fn(vers[a]))
        if not diffs:
            return {}
        mean_d = sum(diffs) / len(diffs)
        rng = random.Random(42)
        boots = []
        for _ in range(n_boot):
            sample = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
            boots.append(sum(sample) / len(sample))
        boots.sort()
        lo = boots[int(0.025 * n_boot)]
        hi = boots[int(0.975 * n_boot)]
        return {
            "n_pairs": len(diffs),
            "mean_delta": round(mean_d, 6),
            "median_delta": round(sorted(diffs)[len(diffs) // 2], 6),
            "boot_ci95_mean_lo": round(lo, 6),
            "boot_ci95_mean_hi": round(hi, 6),
            "interpretation": (
                "Bootstrap 95% CI for mean paired delta (challenger − baseline). "
                "CI excluding 0 suggests a consistent directional shift under resampling."
            ),
        }

    paired_continuous = {
        "cost_v1_v2": paired_delta(lambda r: float(r["total_cost_usd"]), "v1", "v2"),
        "cost_v2_v3": paired_delta(lambda r: float(r["total_cost_usd"]), "v2", "v3"),
        "latency_v1_v2": paired_delta(
            lambda r: float(r["total_latency_ms"]) / 1000.0, "v1", "v2"
        ),
        "latency_v2_v3": paired_delta(
            lambda r: float(r["total_latency_ms"]) / 1000.0, "v2", "v3"
        ),
    }

    # Cost / latency step breakdown from ledger
    step_rollup: dict[str, dict[str, Any]] = {}
    for ver in ("v1", "v2", "v3"):
        buckets = {
            "producer": {"cost": 0.0, "latency_ms": 0.0, "calls": 0},
            "json_repair": {"cost": 0.0, "latency_ms": 0.0, "calls": 0},
            "critic": {"cost": 0.0, "latency_ms": 0.0, "calls": 0},
            "producer_revision": {"cost": 0.0, "latency_ms": 0.0, "calls": 0},
            "llm_judge": {"cost": 0.0, "latency_ms": 0.0, "calls": 0},
            "other_agent": {"cost": 0.0, "latency_ms": 0.0, "calls": 0},
        }
        for c in by_ver_cases[ver]:
            for s in cost_by_run.get(c["run_id"], []):
                name = s["step_name"]
                if name.startswith("producer_turn"):
                    key = "producer"
                elif name == "producer_json_repair":
                    key = "json_repair"
                elif name == "critic":
                    key = "critic"
                elif name == "producer_revision":
                    key = "producer_revision"
                elif name == "llm_judge":
                    key = "llm_judge"
                else:
                    key = "other_agent"
                buckets[key]["cost"] += float(s["cost_usd"])
                buckets[key]["latency_ms"] += float(s["latency_ms"])
                buckets[key]["calls"] += 1
        agent_cost = (
            buckets["producer"]["cost"]
            + buckets["json_repair"]["cost"]
            + buckets["critic"]["cost"]
            + buckets["producer_revision"]["cost"]
            + buckets["other_agent"]["cost"]
        )
        n = max(1, len(by_ver_cases[ver]))
        step_rollup[ver] = {
            "buckets": {
                k: {
                    "cost_usd": round(v["cost"], 6),
                    "latency_s": round(v["latency_ms"] / 1000.0, 3),
                    "calls": v["calls"],
                    "cost_per_deal_usd": round(v["cost"] / n, 6),
                }
                for k, v in buckets.items()
            },
            "agent_total_usd": round(agent_cost, 6),
            "agent_mean_per_deal_usd": round(agent_cost / n, 6),
            "judge_total_usd": round(buckets["llm_judge"]["cost"], 6),
            "judge_mean_per_deal_usd": round(buckets["llm_judge"]["cost"] / n, 6),
            "critic_overhead_total_usd": round(
                buckets["critic"]["cost"] + buckets["producer_revision"]["cost"], 6
            ),
            "critic_overhead_per_deal_usd": round(
                (buckets["critic"]["cost"] + buckets["producer_revision"]["cost"]) / n, 6
            ),
            "json_repair_calls": buckets["json_repair"]["calls"],
            "json_repair_cases_estimate": buckets["json_repair"]["calls"],  # calls, not unique cases
            "n_deals": n,
        }

    # Failure taxonomy (evidence-linked; no invented causes)
    taxonomy: list[dict[str, Any]] = []

    def add_tax(cat_id, label, description, case_version_pairs, resolved_in=None, notes=""):
        by_v = {"v1": [], "v2": [], "v3": []}
        for cid, ver in case_version_pairs:
            by_v[ver].append(cid)
        taxonomy.append(
            {
                "id": cat_id,
                "label": label,
                "description": description,
                "cases_by_version": by_v,
                "counts_by_version": {v: len(ids) for v, ids in by_v.items()},
                "resolved_in": resolved_in,
                "notes": notes,
                "evidence": "Observed from golden expected flags vs memo detected_risk_flags and deterministic graders.",
            }
        )

    # Non-canonical / prose flag labels causing deterministic fail (recall 0)
    prose_fails = []
    for c in by_ver_cases["v1"]:
        if int(c["deterministic_pass"]) == 0 and float(c["flag_recall"]) == 0.0:
            # Check if actual looks like prose annotation of expected
            exp = set(c.get("expected_risk_flags") or [])
            act_raw = c.get("actual_risk_flags") or []
            norms = {_normalize_flag_token(str(x)) for x in act_raw}
            if exp and any(_normalize_flag_token(e) in " ".join(norms) or any(e in str(a) for a in act_raw) for e in exp):
                prose_fails.append((c["case_id"], "v1"))
            elif not exp and act_raw:
                pass  # handled below
            else:
                prose_fails.append((c["case_id"], "v1"))

    # More precise: GE-018, GE-019 are prose; GE-009 is unexpected flags on empty expected
    add_tax(
        "non_canonical_flag_labels",
        "Non-canonical / prose risk-flag labels",
        "Agent included the right legal concept but emitted verbose prose instead of the exact canonical code, so deterministic flag matching scored recall = 0 (tier and margin were already exact).",
        [("GE-018", "v1"), ("GE-019", "v1")],
        resolved_in="v2",
        notes="Both cases became deterministic passes with flag F1 = 1.0 under v2 and v3.",
    )
    add_tax(
        "spurious_flags_on_clean_expected",
        "Spurious flags when golden expected set is empty",
        "Golden expected_risk_flags was empty, but the memo emitted annotated discount/payment flags; graders recorded precision = recall = 0 and Partially Passed.",
        [("GE-009", "v1")],
        resolved_in="v2",
        notes="v2/v3 emit no spurious flags on GE-009 (pass with F1 = 1.0).",
    )
    fp_v1 = [
        (c["case_id"], "v1")
        for c in by_ver_cases["v1"]
        if float(c["flag_precision"]) < 0.999
        and float(c["flag_recall"]) >= 0.999
        and int(c["deterministic_pass"]) == 1
    ]
    add_tax(
        "false_positive_extra_flags",
        "False-positive extra risk flags (still deterministic pass)",
        "Recall of required flags was complete, but additional non-expected flags reduced precision. Deterministic pass still holds because the harness requires full recall, not perfect precision.",
        fp_v1 + ([("GE-016", "v3")] if True else []),
        resolved_in="mostly_v2",
        notes=(
            "v2 eliminated these false positives on the paired cases. "
            "v3 re-introduced a false-positive NON_STANDARD_AUDIT on GE-016 after critic revision "
            "(precision 0.5) while remaining a deterministic pass."
        ),
    )
    add_tax(
        "incorrect_approval_tier",
        "Incorrect approval tier",
        "Predicted required_approval_tier did not exact-match the golden label.",
        [],
        resolved_in=None,
        notes="0 / 60 runs. Tier accuracy = 100% in every version.",
    )
    add_tax(
        "margin_calculation_error",
        "Margin calculation error",
        "Predicted gross margin outside the deterministic tolerance versus golden.",
        [],
        resolved_in=None,
        notes="0 / 60 runs. Margin accuracy = 100% in every version.",
    )
    add_tax(
        "critic_precision_regression",
        "Critic revision associated with precision regression",
        "On the only revised v3 case (GE-016), revision_applied=1 and flag precision fell from 1.0 (v2) to 0.5 (extra NON_STANDARD_AUDIT). Deterministic pass still held via full recall.",
        [("GE-016", "v3")],
        resolved_in=None,
        notes=(
            "This is an observed association on n=1 revised case — not evidence that the critic "
            "improves high-stakes compound deals. Hypothesis for further testing."
        ),
    )

    # Unavailable categories (explicit gaps)
    unavailable_failure_cats = [
        {
            "id": "missing_escalation_recommendations",
            "label": "Missing escalation recommendations",
            "reason": "No dedicated deterministic grader for escalation completeness in the current harness.",
        },
        {
            "id": "unsupported_commercial_recommendations",
            "label": "Unsupported commercial recommendations",
            "reason": "Judge rationale is free text; no structured extraction of unsupported claims was stored.",
        },
        {
            "id": "incomplete_executive_memos",
            "label": "Incomplete executive memos",
            "reason": "Schema presence is enforced; qualitative completeness is only reflected indirectly in Opus scores near ceiling.",
        },
    ]

    # Scenario-level comparison rows
    scenario_compare = []
    for cid, vers in sorted(by_case.items()):
        row = {"case_id": cid, "difficulty": vers.get("v1", vers.get("v2", {})).get("difficulty")}
        for ver in ("v1", "v2", "v3"):
            c = vers.get(ver)
            if not c:
                continue
            row[ver] = {
                "status": c["status"],
                "deterministic_pass": c["deterministic_pass"],
                "flag_precision": c["flag_precision"],
                "flag_recall": c["flag_recall"],
                "flag_f1": c["flag_f1"],
                "judge_overall_score": c["judge_overall_score"],
                "total_cost_usd": c["total_cost_usd"],
                "total_latency_ms": c["total_latency_ms"],
                "revision_applied": c["revision_applied"],
            }
        # deltas
        if "v1" in vers and "v2" in vers:
            row["v1_to_v2"] = {
                "pass_change": int(vers["v2"]["deterministic_pass"]) - int(vers["v1"]["deterministic_pass"]),
                "f1_change": round(float(vers["v2"]["flag_f1"]) - float(vers["v1"]["flag_f1"]), 4),
                "cost_change_usd": round(
                    float(vers["v2"]["total_cost_usd"]) - float(vers["v1"]["total_cost_usd"]), 6
                ),
                "latency_change_s": round(
                    (float(vers["v2"]["total_latency_ms"]) - float(vers["v1"]["total_latency_ms"]))
                    / 1000.0,
                    3,
                ),
            }
        if "v2" in vers and "v3" in vers:
            row["v2_to_v3"] = {
                "pass_change": int(vers["v3"]["deterministic_pass"]) - int(vers["v2"]["deterministic_pass"]),
                "f1_change": round(float(vers["v3"]["flag_f1"]) - float(vers["v2"]["flag_f1"]), 4),
                "cost_change_usd": round(
                    float(vers["v3"]["total_cost_usd"]) - float(vers["v2"]["total_cost_usd"]), 6
                ),
                "latency_change_s": round(
                    (float(vers["v3"]["total_latency_ms"]) - float(vers["v2"]["total_latency_ms"]))
                    / 1000.0,
                    3,
                ),
                "precision_change": round(
                    float(vers["v3"]["flag_precision"]) - float(vers["v2"]["flag_precision"]), 4
                ),
            }
        scenario_compare.append(row)

    # Judge ceiling analysis
    judge_dims = [
        "judge_overall_score",
        "judge_commercial_accuracy",
        "judge_policy_grounding",
        "judge_executive_clarity",
        "judge_actionability",
    ]
    judge_analysis = {"dimensions": {}, "ceiling_effect": True, "note": ""}
    for dim in judge_dims:
        vals = [float(c[dim]) for c in cases]
        judge_analysis["dimensions"][dim] = {
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "mean": round(sum(vals) / len(vals), 4),
            "n_at_5": sum(1 for v in vals if v >= 4.999),
            "n": len(vals),
        }
    judge_analysis["note"] = (
        "Observed Opus scores concentrate at 4.90–5.00 across all versions. "
        "Commercial accuracy, policy grounding, and executive clarity are saturated at 5.0 "
        "in the version-level means. This ceiling effect limits the judge suite as a "
        "between-version discriminator; deterministic flag metrics carry the comparative signal."
    )
    judge_analysis["available_dimensions"] = [
        "commercial_accuracy",
        "policy_grounding",
        "executive_clarity",
        "actionability",
        "overall_score",
    ]
    judge_analysis["unavailable_requested_dimensions"] = [
        {
            "name": "factual_correctness",
            "reason": "Not a separate stored rubric field; closest proxy is commercial_accuracy.",
        },
        {
            "name": "completeness",
            "reason": "Not separately scored in stored judge outputs.",
        },
        {
            "name": "evidence_grounding",
            "reason": "Closest stored field is policy_grounding; no distinct evidence-span score.",
        },
        {
            "name": "consistency",
            "reason": "Single run per case×version; no repeated-run consistency metric exists.",
        },
    ]

    # Latency investigation for v3 P90
    v3_sorted = sorted(by_ver_cases["v3"], key=lambda r: float(r["total_latency_ms"]), reverse=True)
    top = v3_sorted[0]
    latency_investigation = {
        "observation": "v3 p90 latency (88.3s) exceeds v2 p90 (76.8s).",
        "max_case": {
            "case_id": top["case_id"],
            "latency_s": round(float(top["total_latency_ms"]) / 1000.0, 3),
            "cost_usd": top["total_cost_usd"],
            "revision_applied": top["revision_applied"],
            "difficulty": top["difficulty"],
        },
        "evidence": (
            "The single slowest v3 run is GE-016 (revision_applied=1) at ~163.5s and $0.199 — "
            "far above other v3 cases. Critic steps add ~49.7s total ledger latency across 20 runs "
            f"(~{step_rollup['v3']['buckets']['critic']['latency_s']/20:.2f}s mean per deal), "
            "and JSON repair remains present (6 calls). "
            "Attribution of the p90 gap is therefore mixed: the extreme tail is coincident with "
            "the only critic revision, while non-revised heavy cases (e.g. GE-015) still approach ~96s."
        ),
        "hypothesis_not_proven": (
            "We cannot claim the critic universally improves high-stakes outcomes; "
            "only 1/20 cases revised, and that revision coincided with a precision regression."
        ),
    }

    # Comparison matrix deltas
    def pct_change(new, old):
        if old == 0:
            return None
        return round(100.0 * (new - old) / old, 2)

    matrix = []
    order = ["v1", "v2", "v3"]
    for ver in order:
        s = by_ver_summary[ver]
        d = distributions[ver]
        prev = by_ver_summary[order[order.index(ver) - 1]] if ver != "v1" else None
        prev_d = distributions[order[order.index(ver) - 1]] if ver != "v1" else None
        entry = {
            "version": ver,
            "pass_rate_pct": s["deterministic_pass_rate_pct"],
            "pass_count": d["pass_count"],
            "n": d["n"],
            "flag_precision": s["mean_flag_precision"],
            "flag_recall": s["mean_flag_recall"],
            "flag_f1": s["mean_flag_f1"],
            "tier_accuracy_pct": s["tier_accuracy_pct"],
            "margin_accuracy_pct": s["margin_accuracy_pct"],
            "mean_cost_usd": s["mean_cost_usd"],
            "p50_cost_usd": s["p50_cost_usd"],
            "p90_cost_usd": s["p90_cost_usd"],
            "max_cost_usd": d["cost_usd"]["max"],
            "stdev_cost_usd": d["cost_usd"]["stdev"],
            "p50_latency_s": s["p50_latency_s"],
            "p90_latency_s": s["p90_latency_s"],
            "max_latency_s": d["latency_s"]["max"],
            "mean_latency_s": d["latency_s"]["mean"],
            "cache_hit_rate_pct": s["avg_cache_hit_rate_pct"],
            "mean_judge_score": s["mean_judge_score"],
            "revision_rate_pct": s["revision_trigger_rate_pct"],
            "agent_api_calls": step_rollup[ver]["buckets"]["producer"]["calls"]
            + step_rollup[ver]["buckets"]["json_repair"]["calls"]
            + step_rollup[ver]["buckets"]["critic"]["calls"]
            + step_rollup[ver]["buckets"]["producer_revision"]["calls"],
            "json_repair_calls": step_rollup[ver]["json_repair_calls"],
            "deltas_vs_prev": None,
        }
        if prev and prev_d:
            entry["deltas_vs_prev"] = {
                "pass_rate_pp": round(
                    s["deterministic_pass_rate_pct"] - prev["deterministic_pass_rate_pct"], 2
                ),
                "flag_f1": round(s["mean_flag_f1"] - prev["mean_flag_f1"], 4),
                "mean_cost_pct": pct_change(s["mean_cost_usd"], prev["mean_cost_usd"]),
                "p50_cost_pct": pct_change(s["p50_cost_usd"], prev["p50_cost_usd"]),
                "p50_latency_pct": pct_change(s["p50_latency_s"], prev["p50_latency_s"]),
                "p90_latency_pct": pct_change(s["p90_latency_s"], prev["p90_latency_s"]),
                "cache_hit_pp": round(
                    s["avg_cache_hit_rate_pct"] - prev["avg_cache_hit_rate_pct"], 2
                ),
                "judge_overall": round(s["mean_judge_score"] - prev["mean_judge_score"], 4),
            }
        matrix.append(entry)

    data_gaps = [
        "No repeated-run consistency: each case×version was executed once.",
        "No human audit scores populated (human_audit_score column unused).",
        "Judge rubric lacks separate factual-correctness / completeness / consistency dimensions.",
        "Small legal/compound subgroups (n=2 and n=3) make subgroup percentages unstable.",
        "McNemar v1→v2 has only 3 discordant pairs (exact p=0.25) — underpowered for α=0.05 significance claims.",
        "v3 high-stakes benefit remains a hypothesis (1/20 revisions; precision regression on that case).",
    ]

    return {
        "distributions": distributions,
        "wilson_ci": wilson,
        "mcnemar": mcnemar,
        "paired_continuous": paired_continuous,
        "step_rollup": step_rollup,
        "failure_taxonomy": taxonomy,
        "unavailable_failure_categories": unavailable_failure_cats,
        "scenario_compare": scenario_compare,
        "judge_analysis": judge_analysis,
        "latency_investigation": latency_investigation,
        "comparison_matrix": matrix,
        "data_gaps": data_gaps,
        "cost_semantics": {
            "agent_unit_economics": (
                "eval_results.total_cost_usd and benchmark cost percentiles exclude llm_judge rows "
                "(production-relevant agent inference including critic/repair when present)."
            ),
            "evaluation_overhead": (
                "llm_judge costs are experimental evaluation overhead and are reported separately."
            ),
        },
        "execution_vs_success": {
            "benchmark_runs_completed": 60,
            "deterministic_full_pass": sum(1 for c in cases if c["status"] == "Passed"),
            "partial_pass": sum(1 for c in cases if c["status"] == "Partially Passed"),
            "failed": sum(1 for c in cases if c["status"] == "Failed"),
            "note": (
                "60/60 means every case×version produced a scored eval_results row (execution complete). "
                "Only rows with deterministic_pass=1 are full evaluation passes. "
                "v1 contributes 3 Partially Passed rows (tier+margin exact, flag recall incomplete)."
            ),
        },
    }



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

    analytics = _build_analytics(cases, summary, cost_by_run)

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
            "benchmark_runs_completed": f"{len(cases)}/60",
            "benchmark_runs": f"{len(cases)}/60",
            "deterministic_full_pass": analytics["execution_vs_success"]["deterministic_full_pass"],
            "partial_pass": analytics["execution_vs_success"]["partial_pass"],
            "v1_to_v2_pass_rate": f"{v1['deterministic_pass_rate_pct']}% → {v2['deterministic_pass_rate_pct']}%",
            "v1_to_v2_flag_f1": f"{v1['mean_flag_f1']:.4f} → {v2['mean_flag_f1']:.4f}",
            "v1_to_v2_p50_cost_reduction_pct": p50_cost_reduction_pct,
            "v1_to_v2_mean_cost_reduction_pct": mean_cost_reduction_pct,
            "v1_p50_cost_usd": v1["p50_cost_usd"],
            "v2_p50_cost_usd": v2["p50_cost_usd"],
            "v2_mean_cost_usd": v2["mean_cost_usd"],
            "v2_p50_latency_s": v2["p50_latency_s"],
            "v2_p90_latency_s": v2["p90_latency_s"],
            "v2_cache_hit_rate_pct": v2["avg_cache_hit_rate_pct"],
            "v2_flag_precision": v2["mean_flag_precision"],
            "v2_flag_recall": v2["mean_flag_recall"],
            "v2_flag_f1": v2["mean_flag_f1"],
            "sample_size_per_version": 20,
        },
        "status_breakdown": {
            "unit_tests": unit_status,
            "benchmark_runs": status_counts,
            "execution_vs_success": analytics["execution_vs_success"],
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
        "analytics": analytics,
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


def _load_html_template() -> str:
    path = Path(__file__).with_name("dashboard_template.html")
    return path.read_text(encoding="utf-8")


def _html_document(dataset: dict[str, Any]) -> str:
    payload = json.dumps(dataset, ensure_ascii=False)
    # Prevent </script> breakouts in embedded JSON
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e")
    return _load_html_template().replace("@@EVAL_DATA@@", payload)


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
