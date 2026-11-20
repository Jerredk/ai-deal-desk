"""Deterministic + cross-model LLM-as-a-Judge evaluation harness."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import anthropic
import numpy as np
from dotenv import load_dotenv

from db.schema import init_db
from src.agent import DealDeskAgent, _parse_json_response, _usage_tokens
from src.cost_ledger import CostLedger

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "db/deal_desk.db"
DEFAULT_JUDGE_MODEL = "claude-opus-4-6"
DEFAULT_AGENT_MODEL = "claude-sonnet-4-6"
MARGIN_TOLERANCE = 0.05

JUDGE_SYSTEM_PROMPT = """
You are an independent Deal-Desk LLM-as-a-Judge. Score the agent's executive memo
against ground-truth economics and the raw deal request. Be strict and numeric.

Return ONLY valid JSON (no markdown) with this schema:
{
  "commercial_accuracy": <int 1-5>,
  "policy_grounding": <int 1-5>,
  "executive_clarity": <int 1-5>,
  "actionability": <int 1-5>,
  "rationale": "<exactly two concise sentences>"
}

Rubric:
1) commercial_accuracy — fidelity to unit-economics numbers, working-capital impact,
   and approval routing vs ground truth.
2) policy_grounding — accurate identification/explanation of Deal-Desk policy
   thresholds and CUAD contract risks present in the deal.
3) executive_clarity — concise CRO/CFO-ready synthesis without filler or contradictions.
4) actionability — concrete, commercially viable redline fallbacks and close conditions.
""".strip()

_WRITE_LOCK = threading.Lock()


def _connect(db_path: str, timeout: float = 30.0) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=timeout, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _flag_metrics(predicted: set[str], expected: set[str]) -> dict[str, float]:
    if not predicted and not expected:
        return {"flag_precision": 1.0, "flag_recall": 1.0, "flag_f1": 1.0}
    tp = len(predicted & expected)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(expected) if expected else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "flag_precision": round(precision, 4),
        "flag_recall": round(recall, 4),
        "flag_f1": round(f1, 4),
    }


def grade_deterministic(case_row: dict, memo: dict) -> dict:
    """Grade memo against golden-case ground truth with exact/tolerance checks."""
    expected_flags = set(json.loads(case_row["expected_risk_flags_json"]))
    predicted_flags = set(memo.get("detected_risk_flags") or [])

    tier_exact_match = int(
        memo.get("required_approval_tier") == case_row["expected_approval_tier"]
    )
    margin_delta = abs(
        float(memo.get("calculated_gross_margin_pct", 0.0))
        - float(case_row["expected_gross_margin_pct"])
    )
    margin_exact_match = int(margin_delta <= MARGIN_TOLERANCE)
    metrics = _flag_metrics(predicted_flags, expected_flags)
    deterministic_pass = int(
        tier_exact_match == 1
        and margin_exact_match == 1
        and metrics["flag_recall"] == 1.0
    )
    return {
        "tier_exact_match": tier_exact_match,
        "margin_exact_match": margin_exact_match,
        "flag_precision": metrics["flag_precision"],
        "flag_recall": metrics["flag_recall"],
        "flag_f1": metrics["flag_f1"],
        "deterministic_pass": deterministic_pass,
        "predicted_flags": sorted(predicted_flags),
        "expected_flags": sorted(expected_flags),
    }


def grade_with_llm_judge(
    case_row: dict,
    memo: dict,
    *,
    run_id: str,
    experiment_version: str,
    db_path: str = DEFAULT_DB_PATH,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    client: anthropic.Anthropic | None = None,
    ledger: CostLedger | None = None,
) -> dict:
    """Score memo with an independent Anthropic judge; log cost as llm_judge."""
    client = client or anthropic.Anthropic()
    ledger = ledger or CostLedger(db_path)

    payload = case_row.get("deal_payload_json")
    if isinstance(payload, str):
        payload_obj = json.loads(payload)
    else:
        payload_obj = payload or {}

    judge_user = {
        "deal_payload": payload_obj,
        "ground_truth": {
            "expected_gross_margin_pct": case_row["expected_gross_margin_pct"],
            "expected_approval_tier": case_row["expected_approval_tier"],
            "expected_risk_flags": json.loads(case_row["expected_risk_flags_json"]),
        },
        "agent_memo": memo,
    }
    messages = [
        {
            "role": "user",
            "content": (
                "Score the agent memo using the rubric. Return JSON only.\n\n"
                f"{json.dumps(judge_user, indent=2)}"
            ),
        }
    ]

    t0 = time.perf_counter()
    response = client.messages.create(
        model=judge_model,
        max_tokens=1024,
        system=JUDGE_SYSTEM_PROMPT,
        messages=messages,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    tokens = _usage_tokens(response.usage)
    ledger.record_call(
        run_id=run_id,
        experiment_version=experiment_version,
        case_id=str(case_row.get("case_id", "UNKNOWN")),
        step_name="llm_judge",
        model=judge_model,
        input_tokens=tokens["input_tokens"],
        output_tokens=tokens["output_tokens"],
        cache_creation_tokens=tokens["cache_creation_tokens"],
        cache_read_tokens=tokens["cache_read_tokens"],
        latency_ms=latency_ms,
    )

    raw_text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            raw_text += getattr(block, "text", "") or ""
        elif isinstance(block, dict) and block.get("type") == "text":
            raw_text += block.get("text", "")
    scored = _parse_json_response(raw_text)

    dims = [
        int(scored["commercial_accuracy"]),
        int(scored["policy_grounding"]),
        int(scored["executive_clarity"]),
        int(scored["actionability"]),
    ]
    for name, val in zip(
        [
            "commercial_accuracy",
            "policy_grounding",
            "executive_clarity",
            "actionability",
        ],
        dims,
    ):
        if val < 1 or val > 5:
            raise ValueError(f"Judge score {name}={val} out of 1-5 range")

    overall = round(float(np.mean(dims)), 2)
    rationale = str(scored.get("rationale") or scored.get("judge_rationale") or "").strip()
    return {
        "judge_model": judge_model,
        "judge_commercial_accuracy": dims[0],
        "judge_policy_grounding": dims[1],
        "judge_executive_clarity": dims[2],
        "judge_actionability": dims[3],
        "judge_overall_score": overall,
        "judge_rationale": rationale,
    }


def _already_evaluated(
    conn: sqlite3.Connection, experiment_version: str, case_id: str
) -> bool:
    row = conn.execute(
        """
        SELECT 1 FROM eval_results
        WHERE experiment_version = ? AND case_id = ?
        LIMIT 1
        """,
        (experiment_version, case_id),
    ).fetchone()
    return row is not None


def _persist_eval_result(db_path: str, row: dict) -> None:
    with _WRITE_LOCK:
        conn = _connect(db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO eval_results (
                eval_id, run_id, experiment_version, case_id, difficulty,
                tier_exact_match, margin_exact_match, flag_precision, flag_recall,
                flag_f1, deterministic_pass, judge_model, judge_commercial_accuracy,
                judge_policy_grounding, judge_executive_clarity, judge_actionability,
                judge_overall_score, judge_rationale, revision_applied, total_cost_usd,
                total_latency_ms, cache_hit_rate_pct, memo_json, created_at
            ) VALUES (
                :eval_id, :run_id, :experiment_version, :case_id, :difficulty,
                :tier_exact_match, :margin_exact_match, :flag_precision, :flag_recall,
                :flag_f1, :deterministic_pass, :judge_model, :judge_commercial_accuracy,
                :judge_policy_grounding, :judge_executive_clarity, :judge_actionability,
                :judge_overall_score, :judge_rationale, :revision_applied, :total_cost_usd,
                :total_latency_ms, :cache_hit_rate_pct, :memo_json, :created_at
            )
            """,
            row,
        )
        conn.commit()
        conn.close()


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (anthropic.RateLimitError, anthropic.APITimeoutError)):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in {408, 409, 429, 500, 502, 503, 529}
    return False


def _with_retries(fn, *, max_attempts: int = 5, base_delay: float = 1.0):
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_retryable(exc) or attempt == max_attempts - 1:
                raise
            delay = base_delay * (2**attempt)
            logger.warning(
                "Retryable API error (%s); sleeping %.1fs (attempt %d/%d)",
                exc,
                delay,
                attempt + 1,
                max_attempts,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _evaluate_one_case(
    case_row: dict,
    experiment_version: str,
    *,
    db_path: str,
    agent_model: str,
    judge_model: str,
    agent_client: anthropic.Anthropic | None,
    judge_client: anthropic.Anthropic | None,
) -> dict:
    agent = DealDeskAgent(
        model=agent_model,
        db_path=db_path,
        client=agent_client,
    )
    result = _with_retries(
        lambda: agent.run_case(case_row, experiment_version=experiment_version)  # type: ignore[arg-type]
    )
    memo = result["memo"]
    det = grade_deterministic(case_row, memo)
    judge = _with_retries(
        lambda: grade_with_llm_judge(
            case_row,
            memo,
            run_id=result["run_id"],
            experiment_version=experiment_version,
            db_path=db_path,
            judge_model=judge_model,
            client=judge_client,
            ledger=agent.ledger,
        )
    )
    # Agent-only economics (llm_judge excluded by CostLedger.get_run_summary)
    cost = agent.ledger.get_run_summary(result["run_id"])
    eval_row = {
        "eval_id": f"eval_{experiment_version}_{case_row['case_id']}_{uuid.uuid4().hex[:8]}",
        "run_id": result["run_id"],
        "experiment_version": experiment_version,
        "case_id": case_row["case_id"],
        "difficulty": case_row["difficulty"],
        "tier_exact_match": det["tier_exact_match"],
        "margin_exact_match": det["margin_exact_match"],
        "flag_precision": det["flag_precision"],
        "flag_recall": det["flag_recall"],
        "flag_f1": det["flag_f1"],
        "deterministic_pass": det["deterministic_pass"],
        "judge_model": judge["judge_model"],
        "judge_commercial_accuracy": judge["judge_commercial_accuracy"],
        "judge_policy_grounding": judge["judge_policy_grounding"],
        "judge_executive_clarity": judge["judge_executive_clarity"],
        "judge_actionability": judge["judge_actionability"],
        "judge_overall_score": judge["judge_overall_score"],
        "judge_rationale": judge["judge_rationale"],
        "revision_applied": int(bool(result.get("revision_applied"))),
        "total_cost_usd": cost["total_cost_usd"],
        "total_latency_ms": cost["total_latency_ms"],
        "cache_hit_rate_pct": cost["cache_hit_rate_pct"],
        "memo_json": json.dumps(memo),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _persist_eval_result(db_path, eval_row)
    return eval_row


def _load_cases(db_path: str, limit: int | None = None) -> list[dict]:
    conn = _connect(db_path)
    sql = "SELECT * FROM golden_eval_cases ORDER BY case_id"
    if limit is not None:
        rows = conn.execute(sql + " LIMIT ?", (limit,)).fetchall()
    else:
        rows = conn.execute(sql).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def run_benchmark(
    versions: Sequence[str] = ("v1", "v2", "v3"),
    limit: int | None = None,
    workers: int = 3,
    resume: bool = True,
    db_path: str = DEFAULT_DB_PATH,
    agent_model: str = DEFAULT_AGENT_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    agent_client: anthropic.Anthropic | None = None,
    judge_client: anthropic.Anthropic | None = None,
) -> dict[str, Any]:
    """Run v1→v2→v3 benchmark with optional resume and parallel case execution."""
    init_db(db_path)
    cases = _load_cases(db_path, limit=limit)
    completed: list[dict] = []
    skipped = 0

    for version in versions:
        conn = _connect(db_path)
        todo: list[dict] = []
        for case in cases:
            if resume and _already_evaluated(conn, version, case["case_id"]):
                skipped += 1
                continue
            todo.append(case)
        conn.close()

        logger.info(
            "Benchmark %s: %d cases to run (%d skipped via resume)",
            version,
            len(todo),
            skipped if resume else 0,
        )
        if not todo:
            continue

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(
                    _evaluate_one_case,
                    case,
                    version,
                    db_path=db_path,
                    agent_model=agent_model,
                    judge_model=judge_model,
                    agent_client=agent_client,
                    judge_client=judge_client,
                ): case["case_id"]
                for case in todo
            }
            for fut in as_completed(futures):
                case_id = futures[fut]
                try:
                    row = fut.result()
                    completed.append(row)
                    logger.info(
                        "Completed %s/%s det_pass=%s judge=%.2f cost=$%.4f",
                        version,
                        case_id,
                        row["deterministic_pass"],
                        row["judge_overall_score"],
                        row["total_cost_usd"],
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Failed %s/%s", version, case_id)
                    raise

    return {
        "completed": len(completed),
        "skipped": skipped,
        "versions": list(versions),
        "rows": completed,
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _load_eval_rows(db_path: str) -> list[dict]:
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT * FROM eval_results ORDER BY experiment_version, case_id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _summarize_version(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {
            "cases_evaluated": 0,
            "deterministic_pass_rate_pct": 0.0,
            "tier_accuracy_pct": 0.0,
            "margin_accuracy_pct": 0.0,
            "mean_flag_precision": 0.0,
            "mean_flag_recall": 0.0,
            "mean_flag_f1": 0.0,
            "mean_judge_score": 0.0,
            "judge_commercial_accuracy": 0.0,
            "judge_policy_grounding": 0.0,
            "judge_executive_clarity": 0.0,
            "judge_actionability": 0.0,
            "revision_trigger_rate_pct": 0.0,
            "mean_cost_usd": 0.0,
            "p10_cost_usd": 0.0,
            "p50_cost_usd": 0.0,
            "p90_cost_usd": 0.0,
            "avg_cache_hit_rate_pct": 0.0,
            "p50_latency_s": 0.0,
            "p90_latency_s": 0.0,
        }

    costs = [float(r["total_cost_usd"]) for r in rows]
    lat_s = [float(r["total_latency_ms"]) / 1000.0 for r in rows]
    return {
        "cases_evaluated": n,
        "deterministic_pass_rate_pct": round(
            100.0 * sum(int(r["deterministic_pass"]) for r in rows) / n, 2
        ),
        "tier_accuracy_pct": round(
            100.0 * sum(int(r["tier_exact_match"]) for r in rows) / n, 2
        ),
        "margin_accuracy_pct": round(
            100.0 * sum(int(r["margin_exact_match"]) for r in rows) / n, 2
        ),
        "mean_flag_precision": round(_mean(float(r["flag_precision"]) for r in rows), 4),
        "mean_flag_recall": round(_mean(float(r["flag_recall"]) for r in rows), 4),
        "mean_flag_f1": round(_mean(float(r["flag_f1"]) for r in rows), 4),
        "mean_judge_score": round(
            _mean(float(r["judge_overall_score"]) for r in rows), 2
        ),
        "judge_commercial_accuracy": round(
            _mean(float(r["judge_commercial_accuracy"]) for r in rows), 2
        ),
        "judge_policy_grounding": round(
            _mean(float(r["judge_policy_grounding"]) for r in rows), 2
        ),
        "judge_executive_clarity": round(
            _mean(float(r["judge_executive_clarity"]) for r in rows), 2
        ),
        "judge_actionability": round(
            _mean(float(r["judge_actionability"]) for r in rows), 2
        ),
        "revision_trigger_rate_pct": round(
            100.0 * sum(int(r["revision_applied"]) for r in rows) / n, 2
        ),
        "mean_cost_usd": round(_mean(costs), 6),
        "p10_cost_usd": round(_percentile(costs, 10), 6),
        "p50_cost_usd": round(_percentile(costs, 50), 6),
        "p90_cost_usd": round(_percentile(costs, 90), 6),
        "avg_cache_hit_rate_pct": round(
            _mean(float(r["cache_hit_rate_pct"]) for r in rows), 4
        ),
        "p50_latency_s": round(_percentile(lat_s, 50), 3),
        "p90_latency_s": round(_percentile(lat_s, 90), 3),
    }


def export_benchmark_reports(
    db_path: str = DEFAULT_DB_PATH,
    output_dir: str = "outputs",
) -> dict[str, Path]:
    """Write article-ready CSV/JSON summaries from eval_results."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = _load_eval_rows(db_path)

    by_version: dict[str, list[dict]] = {"v1": [], "v2": [], "v3": []}
    for r in rows:
        by_version.setdefault(r["experiment_version"], []).append(r)

    summary_rows = []
    for version in ("v1", "v2", "v3"):
        summary = _summarize_version(by_version.get(version, []))
        summary["experiment_version"] = version
        summary_rows.append(summary)

    summary_csv = out / "benchmark_summary.csv"
    summary_json = out / "benchmark_summary.json"
    fieldnames = [
        "experiment_version",
        "cases_evaluated",
        "deterministic_pass_rate_pct",
        "tier_accuracy_pct",
        "margin_accuracy_pct",
        "mean_flag_precision",
        "mean_flag_recall",
        "mean_flag_f1",
        "mean_judge_score",
        "judge_commercial_accuracy",
        "judge_policy_grounding",
        "judge_executive_clarity",
        "judge_actionability",
        "revision_trigger_rate_pct",
        "mean_cost_usd",
        "p10_cost_usd",
        "p50_cost_usd",
        "p90_cost_usd",
        "avg_cache_hit_rate_pct",
        "p50_latency_s",
        "p90_latency_s",
    ]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    summary_json.write_text(json.dumps(summary_rows, indent=2))

    # Difficulty breakdown
    difficulties = ["standard", "edge_margin", "edge_legal", "compound_trap"]
    diff_rows = []
    for version in ("v1", "v2", "v3"):
        for diff in difficulties:
            subset = [
                r
                for r in by_version.get(version, [])
                if r["difficulty"] == diff
            ]
            n = len(subset)
            if n == 0:
                diff_rows.append(
                    {
                        "experiment_version": version,
                        "difficulty": diff,
                        "cases": 0,
                        "deterministic_pass_rate_pct": 0.0,
                        "mean_judge_score": 0.0,
                        "p50_cost_usd": 0.0,
                    }
                )
                continue
            costs = [float(r["total_cost_usd"]) for r in subset]
            diff_rows.append(
                {
                    "experiment_version": version,
                    "difficulty": diff,
                    "cases": n,
                    "deterministic_pass_rate_pct": round(
                        100.0 * sum(int(r["deterministic_pass"]) for r in subset) / n,
                        2,
                    ),
                    "mean_judge_score": round(
                        _mean(float(r["judge_overall_score"]) for r in subset), 2
                    ),
                    "p50_cost_usd": round(_percentile(costs, 50), 6),
                }
            )

    diff_csv = out / "difficulty_breakdown.csv"
    with diff_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment_version",
                "difficulty",
                "cases",
                "deterministic_pass_rate_pct",
                "mean_judge_score",
                "p50_cost_usd",
            ],
        )
        writer.writeheader()
        writer.writerows(diff_rows)

    case_csv = out / "case_level_evals.csv"
    case_fields = [
        "eval_id",
        "run_id",
        "experiment_version",
        "case_id",
        "difficulty",
        "tier_exact_match",
        "margin_exact_match",
        "flag_precision",
        "flag_recall",
        "flag_f1",
        "deterministic_pass",
        "judge_model",
        "judge_commercial_accuracy",
        "judge_policy_grounding",
        "judge_executive_clarity",
        "judge_actionability",
        "judge_overall_score",
        "judge_rationale",
        "revision_applied",
        "total_cost_usd",
        "total_latency_ms",
        "cache_hit_rate_pct",
        "human_audit_score",
        "created_at",
    ]
    with case_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=case_fields)
        writer.writeheader()
        for r in rows:
            out_row = {k: r.get(k) for k in case_fields if k != "human_audit_score"}
            out_row["human_audit_score"] = ""
            writer.writerow(out_row)

    return {
        "benchmark_summary_csv": summary_csv,
        "benchmark_summary_json": summary_json,
        "difficulty_breakdown_csv": diff_csv,
        "case_level_evals_csv": case_csv,
    }


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Deal-Desk evaluation harness")
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["v1", "v2", "v3"],
        choices=["v1", "v2", "v3"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    args = parser.parse_args()

    init_db(args.db_path)

    if not args.export_only:
        result = run_benchmark(
            versions=tuple(args.versions),
            limit=args.limit,
            workers=args.workers,
            resume=not args.no_resume,
            db_path=args.db_path,
            agent_model=args.agent_model,
            judge_model=args.judge_model,
        )
        print(
            f"Benchmark complete: completed={result['completed']} "
            f"skipped={result['skipped']} versions={result['versions']}"
        )

    paths = export_benchmark_reports(args.db_path, args.output_dir)
    print("Wrote reports:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
