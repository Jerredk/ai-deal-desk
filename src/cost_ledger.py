"""Anthropic model pricing and per-run cost/latency ledger."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from db.schema import init_db

# Pinned Anthropic list rates (USD per 1M tokens), as of 2026-11-21.
# Source: Anthropic public pricing; not fetched live (no pricing API).
# cache_write ≈ 1.25× input; cache_read ≈ 0.10× or 0.05× input by model family.
# Update this table (and the date) when rates change; keep pinned during a benchmark.
ANTHROPIC_PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-sonnet-4-20250514": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-sonnet-5-5": {
        "input": 2.00,
        "output": 10.00,
        "cache_write": 2.50,
        "cache_read": 0.10,
    },
    "claude-haiku-4-5-20251001": {
        "input": 1.00,
        "output": 5.00,
        "cache_write": 1.25,
        "cache_read": 0.10,
    },
    "claude-haiku-4-5": {
        "input": 1.00,
        "output": 5.00,
        "cache_write": 1.25,
        "cache_read": 0.10,
    },
    "claude-haiku-5-5": {
        "input": 0.10,
        "output": 0.50,
        "cache_write": 0.125,
        "cache_read": 0.01,
    },
    "claude-opus-4-6": {
        "input": 5.00,
        "output": 25.00,
        "cache_write": 6.25,
        "cache_read": 0.50,
    },
    "claude-opus-4-5": {
        "input": 5.00,
        "output": 25.00,
        "cache_write": 6.25,
        "cache_read": 0.50,
    },
    "claude-opus-5-5": {
        "input": 4.00,
        "output": 20.00,
        "cache_write": 5.00,
        "cache_read": 0.20,
    },
}


def calculate_call_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Exact USD cost rounded to 6 decimal places."""
    if model not in ANTHROPIC_PRICING_USD_PER_1M:
        raise ValueError(
            f"Unknown model ID {model!r}. Known models: "
            f"{sorted(ANTHROPIC_PRICING_USD_PER_1M)}"
        )
    rates = ANTHROPIC_PRICING_USD_PER_1M[model]
    cost = (
        input_tokens * rates["input"]
        + output_tokens * rates["output"]
        + cache_creation_tokens * rates["cache_write"]
        + cache_read_tokens * rates["cache_read"]
    ) / 1_000_000.0
    return round(cost, 6)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), q))


class CostLedger:
    """Persist and aggregate Anthropic call costs / latency."""

    def __init__(self, db_path: str = "db/deal_desk.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        init_db(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def record_call(
        self,
        run_id: str,
        experiment_version: str,
        case_id: str,
        step_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        latency_ms: float = 0.0,
    ) -> dict:
        cost_usd = calculate_call_cost_usd(
            model,
            input_tokens,
            output_tokens,
            cache_creation_tokens,
            cache_read_tokens,
        )
        created_at = datetime.now(timezone.utc).isoformat()
        row = {
            "run_id": run_id,
            "experiment_version": experiment_version,
            "case_id": case_id,
            "step_name": step_name,
            "model": model,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cache_creation_tokens": int(cache_creation_tokens),
            "cache_read_tokens": int(cache_read_tokens),
            "cost_usd": cost_usd,
            "latency_ms": float(latency_ms),
            "created_at": created_at,
        }
        conn = self._connect()
        cur = conn.execute(
            """
            INSERT INTO cost_ledger (
                run_id, experiment_version, case_id, step_name, model,
                input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens,
                cost_usd, latency_ms, created_at
            ) VALUES (
                :run_id, :experiment_version, :case_id, :step_name, :model,
                :input_tokens, :output_tokens, :cache_creation_tokens, :cache_read_tokens,
                :cost_usd, :latency_ms, :created_at
            )
            """,
            row,
        )
        conn.commit()
        row["id"] = cur.lastrowid
        conn.close()
        return row

    def get_run_summary(self, run_id: str) -> dict:
        """Aggregate agent call costs for a run.

        Rows with step_name='llm_judge' are excluded so cross-model judge spend
        does not contaminate per-run agent unit economics.
        """
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT * FROM cost_ledger
            WHERE run_id = ? AND step_name != 'llm_judge'
            """,
            (run_id,),
        ).fetchall()
        conn.close()

        input_tokens = sum(r["input_tokens"] for r in rows)
        output_tokens = sum(r["output_tokens"] for r in rows)
        cache_creation_tokens = sum(r["cache_creation_tokens"] for r in rows)
        cache_read_tokens = sum(r["cache_read_tokens"] for r in rows)
        denom = max(1, input_tokens + cache_creation_tokens + cache_read_tokens)
        cache_hit_rate_pct = (cache_read_tokens / denom) * 100.0

        return {
            "run_id": run_id,
            "calls": len(rows),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_hit_rate_pct": round(cache_hit_rate_pct, 4),
            "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
            "total_latency_ms": round(sum(r["latency_ms"] for r in rows), 3),
        }

    def get_experiment_summary(self, experiment_version: str) -> dict:
        conn = self._connect()
        run_ids = [
            r[0]
            for r in conn.execute(
                """
                SELECT DISTINCT run_id FROM cost_ledger
                WHERE experiment_version = ?
                ORDER BY run_id
                """,
                (experiment_version,),
            ).fetchall()
        ]
        conn.close()

        run_summaries = [self.get_run_summary(rid) for rid in run_ids]
        costs = [s["total_cost_usd"] for s in run_summaries]
        latencies = [s["total_latency_ms"] for s in run_summaries]
        hit_rates = [s["cache_hit_rate_pct"] for s in run_summaries]

        n = len(run_summaries)
        return {
            "experiment_version": experiment_version,
            "run_count": n,
            "mean_cost_usd": round(float(np.mean(costs)) if costs else 0.0, 6),
            "p10_cost_usd": round(_percentile(costs, 10), 6),
            "p50_cost_usd": round(_percentile(costs, 50), 6),
            "p90_cost_usd": round(_percentile(costs, 90), 6),
            "mean_latency_ms": round(float(np.mean(latencies)) if latencies else 0.0, 3),
            "p10_latency_ms": round(_percentile(latencies, 10), 3),
            "p50_latency_ms": round(_percentile(latencies, 50), 3),
            "p90_latency_ms": round(_percentile(latencies, 90), 3),
            "avg_cache_hit_rate_pct": round(
                float(np.mean(hit_rates)) if hit_rates else 0.0, 4
            ),
        }
