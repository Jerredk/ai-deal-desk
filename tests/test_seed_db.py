"""Verification tests for Increment 1 seed data and financial rules."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from db.schema import init_db
from db.seed_db import (
    compute_gross_margin_pct,
    determine_approval_tier,
    seed_db,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DB_PATH = PROJECT_ROOT / "db" / "test_deal_desk.db"


@pytest.fixture(scope="module")
def seeded_db():
    """Seed a dedicated test database once per module."""
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    counts = seed_db(str(TEST_DB_PATH))
    conn = sqlite3.connect(str(TEST_DB_PATH))
    conn.row_factory = sqlite3.Row
    yield conn, counts
    conn.close()
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()


def test_all_tables_exist_with_exact_row_counts(seeded_db):
    conn, counts = seeded_db
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    assert tables == {
        "historical_deals",
        "golden_eval_cases",
        "rag_chunks",
        "cost_ledger",
    }
    assert counts["historical_deals"] == 250
    assert counts["golden_eval_cases"] == 20
    assert counts["rag_chunks"] == 16
    assert counts["cost_ledger"] == 0

    assert conn.execute("SELECT COUNT(*) FROM historical_deals").fetchone()[0] == 250
    assert conn.execute("SELECT COUNT(*) FROM golden_eval_cases").fetchone()[0] == 20
    assert conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0] == 16


def test_golden_eval_expected_values_match_deterministic_rules(seeded_db):
    conn, _ = seeded_db
    rows = conn.execute("SELECT * FROM golden_eval_cases ORDER BY case_id").fetchall()
    assert len(rows) == 20

    for row in rows:
        payload = json.loads(row["deal_payload_json"])
        risk_flags = json.loads(row["expected_risk_flags_json"])

        expected_margin = compute_gross_margin_pct(
            payload["annual_contract_value"],
            payload["discount_pct"],
            payload["payment_terms"],
            payload["services_hours"],
        )
        expected_tier = determine_approval_tier(
            payload["discount_pct"],
            expected_margin,
            payload["payment_terms"],
            risk_flags,
        )

        assert row["expected_gross_margin_pct"] == pytest.approx(expected_margin)
        assert row["expected_approval_tier"] == expected_tier


def test_at_least_four_nominal_discount_margin_traps(seeded_db):
    """Nominal discount <= 20% but gross margin < 72% (margin-trap cases)."""
    conn, _ = seeded_db
    rows = conn.execute("SELECT * FROM golden_eval_cases").fetchall()

    traps = []
    for row in rows:
        payload = json.loads(row["deal_payload_json"])
        if (
            payload["discount_pct"] <= 0.20
            and row["expected_gross_margin_pct"] < 72.0
        ):
            traps.append(row["case_id"])

    assert len(traps) >= 4, f"Expected >=4 margin traps, found {traps}"
    # All margin traps must escalate to CFO
    for case_id in traps:
        tier = conn.execute(
            "SELECT expected_approval_tier FROM golden_eval_cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()[0]
        assert tier == "CFO-And-Legal"


def test_cost_ledger_accepts_inserts_and_queries(seeded_db):
    conn, _ = seeded_db
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO cost_ledger (
            run_id, experiment_version, case_id, step_name, model,
            input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens,
            cost_usd, latency_ms, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "run-test-001",
            "inc1",
            "GE-001",
            "unit_test_probe",
            "test-model",
            100,
            50,
            0,
            0,
            0.0123,
            42.5,
            now,
        ),
    )
    conn.commit()

    row = conn.execute(
        "SELECT * FROM cost_ledger WHERE run_id = ?",
        ("run-test-001",),
    ).fetchone()
    assert row is not None
    assert row["case_id"] == "GE-001"
    assert row["step_name"] == "unit_test_probe"
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 50
    assert row["cost_usd"] == pytest.approx(0.0123)
    assert row["latency_ms"] == pytest.approx(42.5)

    count = conn.execute("SELECT COUNT(*) FROM cost_ledger").fetchone()[0]
    assert count == 1


def test_init_db_is_idempotent(tmp_path):
    db_path = tmp_path / "idempotent.db"
    conn1 = init_db(str(db_path))
    conn1.close()
    conn2 = init_db(str(db_path))
    tables = {
        row[0]
        for row in conn2.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    conn2.close()
    assert "historical_deals" in tables
    assert "golden_eval_cases" in tables
    assert "rag_chunks" in tables
    assert "cost_ledger" in tables


def test_rag_chunk_source_mix(seeded_db):
    conn, _ = seeded_db
    policy = conn.execute(
        "SELECT COUNT(*) FROM rag_chunks WHERE source_type = 'DEAL_DESK_POLICY'"
    ).fetchone()[0]
    clauses = conn.execute(
        "SELECT COUNT(*) FROM rag_chunks WHERE source_type = 'CUAD_CLAUSE'"
    ).fetchone()[0]
    assert policy == 6
    assert clauses == 10
