"""Offline tests for deterministic grading + LLM-judge eval harness."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from db.schema import init_db
from db.seed_db import seed_db
from src.cost_ledger import CostLedger
from src.evals import (
    export_benchmark_reports,
    grade_deterministic,
    grade_with_llm_judge,
    run_benchmark,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DB = PROJECT_ROOT / "db" / "test_inc4_evals.db"


@pytest.fixture
def seeded_db(tmp_path):
    db_path = tmp_path / "deal_desk.db"
    seed_db(str(db_path))
    return str(db_path)


def _memo(**overrides):
    base = {
        "deal_id": "GE-001",
        "account_name": "Test Co",
        "executive_recommendation": "APPROVE",
        "required_approval_tier": "Auto-Approve",
        "calculated_gross_margin_pct": 78.42,
        "margin_floor_pct": 72.0,
        "detected_risk_flags": [],
        "executive_summary": "Clean deal.",
        "commercial_and_margin_analysis": "Margin healthy.",
        "legal_and_policy_findings": "No redlines.",
        "required_redline_fallbacks": [],
    }
    base.update(overrides)
    return base


def _case(**overrides):
    base = {
        "case_id": "GE-001",
        "scenario_name": "clean",
        "difficulty": "standard",
        "deal_payload_json": json.dumps({"account_name": "Test Co"}),
        "expected_gross_margin_pct": 78.42,
        "expected_approval_tier": "Auto-Approve",
        "expected_risk_flags_json": "[]",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Deterministic grading
# ---------------------------------------------------------------------------
def test_grade_deterministic_exact_match_empty_flags():
    result = grade_deterministic(_case(), _memo())
    assert result["tier_exact_match"] == 1
    assert result["margin_exact_match"] == 1
    assert result["flag_precision"] == 1.0
    assert result["flag_recall"] == 1.0
    assert result["flag_f1"] == 1.0
    assert result["deterministic_pass"] == 1


def test_grade_deterministic_margin_tolerance_and_misses():
    # Within 0.05 tolerance
    ok = grade_deterministic(
        _case(expected_gross_margin_pct=78.42),
        _memo(calculated_gross_margin_pct=78.46),
    )
    assert ok["margin_exact_match"] == 1

    miss = grade_deterministic(
        _case(expected_gross_margin_pct=78.42),
        _memo(calculated_gross_margin_pct=78.50),
    )
    assert miss["margin_exact_match"] == 0
    assert miss["deterministic_pass"] == 0

    tier_miss = grade_deterministic(
        _case(),
        _memo(required_approval_tier="VP-Sales"),
    )
    assert tier_miss["tier_exact_match"] == 0
    assert tier_miss["deterministic_pass"] == 0


def test_grade_deterministic_flag_precision_recall_partial():
    case = _case(
        expected_risk_flags_json=json.dumps(
            ["MARGIN_BELOW_FLOOR", "NET_90_WORKING_CAPITAL", "MFC_CLAUSE"]
        )
    )
    # Predicted 2 correct + 1 false positive → precision 2/3, recall 2/3
    memo = _memo(
        required_approval_tier="CFO-And-Legal",
        calculated_gross_margin_pct=60.0,
        detected_risk_flags=[
            "MARGIN_BELOW_FLOOR",
            "NET_90_WORKING_CAPITAL",
            "PRICE_CAP_AUTO_RENEWAL",
        ],
    )
    case["expected_approval_tier"] = "CFO-And-Legal"
    case["expected_gross_margin_pct"] = 60.0
    result = grade_deterministic(case, memo)
    assert result["flag_precision"] == pytest.approx(2 / 3, rel=1e-3)
    assert result["flag_recall"] == pytest.approx(2 / 3, rel=1e-3)
    assert result["flag_f1"] == pytest.approx(2 / 3, rel=1e-3)
    # recall < 1.0 → deterministic_pass = 0 even if tier/margin match
    assert result["deterministic_pass"] == 0

    # Perfect recall but extra FP → still pass (recall==1)
    perfect_recall = grade_deterministic(
        case,
        _memo(
            required_approval_tier="CFO-And-Legal",
            calculated_gross_margin_pct=60.0,
            detected_risk_flags=[
                "MARGIN_BELOW_FLOOR",
                "NET_90_WORKING_CAPITAL",
                "MFC_CLAUSE",
                "EXTRA_NOISE",
            ],
        ),
    )
    assert perfect_recall["flag_recall"] == 1.0
    assert perfect_recall["flag_precision"] == pytest.approx(0.75)
    assert perfect_recall["deterministic_pass"] == 1


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------
def test_grade_with_llm_judge_parses_scores_and_excludes_from_run_summary(seeded_db):
    client = MagicMock()
    judge_json = json.dumps(
        {
            "commercial_accuracy": 5,
            "policy_grounding": 4,
            "executive_clarity": 5,
            "actionability": 4,
            "rationale": (
                "Economics match ground truth within tolerance. "
                "Fallbacks are concrete and policy-aligned."
            ),
        }
    )
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=judge_json)],
        usage=SimpleNamespace(
            input_tokens=500,
            output_tokens=120,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )

    ledger = CostLedger(seeded_db)
    # Pretend agent already logged a producer turn
    ledger.record_call(
        run_id="run-judge-1",
        experiment_version="v2",
        case_id="GE-001",
        step_name="producer_turn_1",
        model="claude-sonnet-4-6",
        input_tokens=1000,
        output_tokens=200,
        latency_ms=100.0,
    )
    agent_cost_before = ledger.get_run_summary("run-judge-1")["total_cost_usd"]

    scored = grade_with_llm_judge(
        _case(),
        _memo(),
        run_id="run-judge-1",
        experiment_version="v2",
        db_path=seeded_db,
        judge_model="claude-opus-4-6",
        client=client,
        ledger=ledger,
    )
    assert scored["judge_commercial_accuracy"] == 5
    assert scored["judge_policy_grounding"] == 4
    assert scored["judge_executive_clarity"] == 5
    assert scored["judge_actionability"] == 4
    assert scored["judge_overall_score"] == pytest.approx(4.5)
    assert "Economics match" in scored["judge_rationale"]

    # Judge row persisted
    conn = sqlite3.connect(seeded_db)
    judge_rows = conn.execute(
        "SELECT step_name, model FROM cost_ledger WHERE step_name = 'llm_judge'"
    ).fetchall()
    conn.close()
    assert len(judge_rows) == 1
    assert judge_rows[0][1] == "claude-opus-4-6"

    # Agent run summary unchanged by judge spend
    agent_cost_after = ledger.get_run_summary("run-judge-1")["total_cost_usd"]
    assert agent_cost_after == pytest.approx(agent_cost_before)
    assert ledger.get_run_summary("run-judge-1")["calls"] == 1


# ---------------------------------------------------------------------------
# Benchmark resume + export
# ---------------------------------------------------------------------------
def _fake_agent_result(case_id: str, version: str, revision: bool = False):
    return {
        "run_id": f"{version}_{case_id}_deadbeef",
        "case_id": case_id,
        "experiment_version": version,
        "memo": _memo(
            deal_id=case_id,
            required_approval_tier="CFO-And-Legal",
            calculated_gross_margin_pct=59.78,
            detected_risk_flags=["MARGIN_BELOW_FLOOR", "NET_90_WORKING_CAPITAL"],
            executive_recommendation="ESCALATE_WITH_CONDITIONS",
            required_redline_fallbacks=["Convert Net-90 to Net-30"],
        ),
        "tool_trace": [],
        "critic_review": None,
        "revision_applied": revision,
        "cost_summary": {
            "run_id": f"{version}_{case_id}_deadbeef",
            "calls": 2,
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "cache_hit_rate_pct": 0.0,
            "total_cost_usd": 0.01,
            "total_latency_ms": 250.0,
        },
    }


def test_run_benchmark_resume_and_export(seeded_db, tmp_path):
    # Ensure schema has eval_results
    init_db(seeded_db)
    # Pre-insert one eval so resume skips it for v1/GE-001 (first ordered case)
    conn = sqlite3.connect(seeded_db)
    conn.execute(
        """
        INSERT INTO eval_results (
            eval_id, run_id, experiment_version, case_id, difficulty,
            tier_exact_match, margin_exact_match, flag_precision, flag_recall,
            flag_f1, deterministic_pass, judge_model, judge_commercial_accuracy,
            judge_policy_grounding, judge_executive_clarity, judge_actionability,
            judge_overall_score, judge_rationale, revision_applied, total_cost_usd,
            total_latency_ms, cache_hit_rate_pct, memo_json, created_at
        ) VALUES (?, ?, ?, ?, ?, 1, 1, 1.0, 1.0, 1.0, 1, 'claude-opus-4-6',
                  5, 5, 5, 5, 5.0, 'preseed', 0, 0.01, 100.0, 0.0, '{}',
                  '2026-01-01T00:00:00+00:00')
        """,
        ("eval_preseed", "run_pre", "v1", "GE-001", "standard"),
    )
    conn.commit()
    conn.close()

    judge_json = json.dumps(
        {
            "commercial_accuracy": 4,
            "policy_grounding": 4,
            "executive_clarity": 3,
            "actionability": 4,
            "rationale": "Solid commercial fidelity. Actionable fallbacks present.",
        }
    )

    def fake_run_case(self, case_row, experiment_version="v3", run_id=None):
        # Also record a fake agent cost row so get_run_summary works
        self.ledger.record_call(
            run_id=f"{experiment_version}_{case_row['case_id']}_deadbeef",
            experiment_version=experiment_version,
            case_id=case_row["case_id"],
            step_name="producer_turn_1",
            model="claude-sonnet-4-6",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200.0,
        )
        result = _fake_agent_result(case_row["case_id"], experiment_version)
        # Use real expected margin/tier from case for deterministic grading path
        result["memo"]["calculated_gross_margin_pct"] = case_row[
            "expected_gross_margin_pct"
        ]
        result["memo"]["required_approval_tier"] = case_row["expected_approval_tier"]
        result["memo"]["detected_risk_flags"] = json.loads(
            case_row["expected_risk_flags_json"]
        )
        return result

    mock_judge_client = MagicMock()
    mock_judge_client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=judge_json)],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=40,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )

    with patch("src.evals.DealDeskAgent.run_case", fake_run_case):
        result = run_benchmark(
            versions=("v1",),
            limit=2,
            workers=2,
            resume=True,
            db_path=seeded_db,
            judge_client=mock_judge_client,
            agent_client=MagicMock(),
        )

    # limit=2 → GE-001 (preseeded/skip) + GE-002 (run) → completed=1, skipped=1
    assert result["completed"] == 1
    assert result["skipped"] == 1

    # Second run with resume should skip both GE-001 and GE-002
    with patch("src.evals.DealDeskAgent.run_case", fake_run_case):
        result2 = run_benchmark(
            versions=("v1",),
            limit=2,
            workers=2,
            resume=True,
            db_path=seeded_db,
            judge_client=mock_judge_client,
            agent_client=MagicMock(),
        )
    assert result2["completed"] == 0
    assert result2["skipped"] == 2

    out_dir = tmp_path / "reports"
    paths = export_benchmark_reports(seeded_db, str(out_dir))
    assert paths["benchmark_summary_csv"].exists()
    assert paths["benchmark_summary_json"].exists()
    assert paths["difficulty_breakdown_csv"].exists()
    assert paths["case_level_evals_csv"].exists()

    summary = json.loads(paths["benchmark_summary_json"].read_text())
    assert isinstance(summary, list)
    assert {r["experiment_version"] for r in summary} == {"v1", "v2", "v3"}
    v1 = next(r for r in summary if r["experiment_version"] == "v1")
    assert v1["cases_evaluated"] == 2

    case_csv = paths["case_level_evals_csv"].read_text()
    assert "human_audit_score" in case_csv.splitlines()[0]
    assert "GE-001" in case_csv or "GE-002" in case_csv


def test_eval_results_table_exists_after_init(tmp_path):
    db_path = tmp_path / "empty.db"
    conn = init_db(str(db_path))
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()
    assert "eval_results" in tables
