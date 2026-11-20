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

    conn.commit()
    return conn
