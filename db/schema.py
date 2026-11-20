"""SQLite schema for the Enterprise Deal-Desk & Margin-Guardrail Agent."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def init_db(db_path: str = "db/deal_desk.db") -> sqlite3.Connection:
    """Create the deal-desk database and all required tables/indexes."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS historical_deals (
            deal_id TEXT PRIMARY KEY,
            account_name TEXT NOT NULL,
            segment TEXT NOT NULL CHECK (segment IN ('Mid-Market', 'Enterprise')),
            annual_contract_value REAL NOT NULL,
            term_months INTEGER NOT NULL,
            discount_pct REAL NOT NULL,
            payment_terms TEXT NOT NULL CHECK (payment_terms IN ('Net-30', 'Net-60', 'Net-90')),
            custom_onboarding_cost REAL NOT NULL,
            gross_margin_pct REAL NOT NULL,
            has_nonstandard_legal INTEGER NOT NULL CHECK (has_nonstandard_legal IN (0, 1)),
            outcome TEXT NOT NULL CHECK (outcome IN ('Closed-Won', 'Closed-Lost')),
            sales_cycle_days INTEGER NOT NULL
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_historical_deals_segment "
        "ON historical_deals(segment)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_historical_deals_outcome "
        "ON historical_deals(outcome)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_historical_deals_discount "
        "ON historical_deals(discount_pct)"
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS golden_eval_cases (
            case_id TEXT PRIMARY KEY,
            scenario_name TEXT NOT NULL,
            difficulty TEXT NOT NULL CHECK (
                difficulty IN (
                    'standard',
                    'edge_margin',
                    'edge_legal',
                    'compound_trap'
                )
            ),
            deal_payload_json TEXT NOT NULL,
            expected_gross_margin_pct REAL NOT NULL,
            expected_approval_tier TEXT NOT NULL CHECK (
                expected_approval_tier IN (
                    'Auto-Approve',
                    'VP-Sales',
                    'CFO-And-Legal'
                )
            ),
            expected_risk_flags_json TEXT NOT NULL
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_golden_eval_difficulty "
        "ON golden_eval_cases(difficulty)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_golden_eval_tier "
        "ON golden_eval_cases(expected_approval_tier)"
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_chunks (
            chunk_id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL CHECK (
                source_type IN ('DEAL_DESK_POLICY', 'CUAD_CLAUSE')
            ),
            category TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding_json TEXT
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_rag_chunks_source_type "
        "ON rag_chunks(source_type)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_rag_chunks_category "
        "ON rag_chunks(category)"
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS cost_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            experiment_version TEXT NOT NULL,
            case_id TEXT NOT NULL,
            step_name TEXT NOT NULL,
            model TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            cache_creation_tokens INTEGER NOT NULL,
            cache_read_tokens INTEGER NOT NULL,
            cost_usd REAL NOT NULL,
            latency_ms REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_cost_ledger_run_id "
        "ON cost_ledger(run_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_cost_ledger_case_id "
        "ON cost_ledger(case_id)"
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_results (
            eval_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            experiment_version TEXT NOT NULL,
            case_id TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            tier_exact_match INTEGER NOT NULL CHECK (tier_exact_match IN (0, 1)),
            margin_exact_match INTEGER NOT NULL CHECK (margin_exact_match IN (0, 1)),
            flag_precision REAL NOT NULL,
            flag_recall REAL NOT NULL,
            flag_f1 REAL NOT NULL,
            deterministic_pass INTEGER NOT NULL CHECK (deterministic_pass IN (0, 1)),
            judge_model TEXT NOT NULL,
            judge_commercial_accuracy INTEGER NOT NULL CHECK (
                judge_commercial_accuracy BETWEEN 1 AND 5
            ),
            judge_policy_grounding INTEGER NOT NULL CHECK (
                judge_policy_grounding BETWEEN 1 AND 5
            ),
            judge_executive_clarity INTEGER NOT NULL CHECK (
                judge_executive_clarity BETWEEN 1 AND 5
            ),
            judge_actionability INTEGER NOT NULL CHECK (
                judge_actionability BETWEEN 1 AND 5
            ),
            judge_overall_score REAL NOT NULL,
            judge_rationale TEXT NOT NULL,
            revision_applied INTEGER NOT NULL CHECK (revision_applied IN (0, 1)),
            total_cost_usd REAL NOT NULL,
            total_latency_ms REAL NOT NULL,
            cache_hit_rate_pct REAL NOT NULL,
            memo_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_eval_results_version "
        "ON eval_results(experiment_version)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_eval_results_case "
        "ON eval_results(case_id)"
    )
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_results_version_case "
        "ON eval_results(experiment_version, case_id)"
    )

    conn.commit()
    return conn
