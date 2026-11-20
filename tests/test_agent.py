"""Offline unit tests for DealDeskAgent (mocked Anthropic client — no live API spend)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from db.seed_db import seed_db
from src.agent import (
    DealDeskAgent,
    DealDeskExecutiveMemo,
    _parse_json_response,
    _version_config,
)
from src.prompts import (
    CANONICAL_HIGH_RISK_LEGAL_FLAGS,
    V1_SYSTEM_PROMPT,
    V2_SYSTEM_PROMPT,
    approximate_token_count,
)
from src.tools import V1_BASIC_TOOLS, V2_POKAYOKE_TOOLS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DB = PROJECT_ROOT / "db" / "test_inc3_agent.db"


@pytest.fixture(scope="module")
def seeded_db_path():
    if TEST_DB.exists():
        TEST_DB.unlink()
    seed_db(str(TEST_DB))
    yield str(TEST_DB)
    if TEST_DB.exists():
        TEST_DB.unlink()


@pytest.fixture
def sample_case(seeded_db_path):
    conn = sqlite3.connect(seeded_db_path)
    conn.row_factory = sqlite3.Row
    row = dict(
        conn.execute(
            "SELECT * FROM golden_eval_cases WHERE case_id = 'GE-015'"
        ).fetchone()
    )
    conn.close()
    return row


def _usage(
    input_tokens=100,
    output_tokens=50,
    cache_creation=0,
    cache_read=0,
):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
    )


def _text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _tool_block(tool_id: str, name: str, tool_input: dict):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)


def _response(stop_reason: str, content: list, usage=None):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=content,
        usage=usage or _usage(),
    )


def _valid_memo_json(**overrides) -> str:
    base = {
        "deal_id": "GE-015",
        "account_name": "Silverline Media Co",
        "executive_recommendation": "ESCALATE_WITH_CONDITIONS",
        "required_approval_tier": "CFO-And-Legal",
        "calculated_gross_margin_pct": 59.78,
        "margin_floor_pct": 72.0,
        "detected_risk_flags": ["MARGIN_BELOW_FLOOR", "NET_90_WORKING_CAPITAL"],
        "executive_summary": "Margin trap under Net-90 with heavy onboarding.",
        "commercial_and_margin_analysis": "Gross margin 59.78% below 72% floor.",
        "legal_and_policy_findings": "No high-risk CUAD flags; economic breach only.",
        "required_redline_fallbacks": [
            "Convert Net-90 to Net-30",
            "Convert excess onboarding hours to paid SOW",
        ],
    }
    base.update(overrides)
    return json.dumps(base)


# ---------------------------------------------------------------------------
# Helpers / prompts
# ---------------------------------------------------------------------------
def test_parse_json_response_strips_fences_and_errors():
    assert _parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json_response('{"b": 2}') == {"b": 2}
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        _parse_json_response("not-json")
    with pytest.raises(ValueError, match="Empty"):
        _parse_json_response("   ")


def test_canonical_flags_and_v2_prompt_cache_threshold():
    assert CANONICAL_HIGH_RISK_LEGAL_FLAGS == [
        "MFN_CLAUSE",
        "UNCAPPED_LIABILITY",
        "TERMINATION_CONVENIENCE_30D",
        "NON_STANDARD_AUDIT",
    ]
    # Static system prompt alone should be substantial; with tools >> 1300
    assert approximate_token_count(V2_SYSTEM_PROMPT) > 800
    tools_chars = sum(len(json.dumps(t)) for t in V2_POKAYOKE_TOOLS)
    combined = approximate_token_count(V2_SYSTEM_PROMPT) + tools_chars // 4
    assert combined > 1300
    assert "MFN_CLAUSE" in V2_SYSTEM_PROMPT
    assert "<tool_execution_protocol>" in V2_SYSTEM_PROMPT
    assert "XML" not in V1_SYSTEM_PROMPT or "<role>" not in V1_SYSTEM_PROMPT


def test_version_config_switches():
    v1 = _version_config("v1")
    v2 = _version_config("v2")
    v3 = _version_config("v3")
    assert v1["rag_mode"] == "bm25"
    assert v1["tools"] == V1_BASIC_TOOLS or [t["name"] for t in v1["tools"]] == [
        t["name"] for t in V1_BASIC_TOOLS
    ]
    assert v1["use_cache"] is False and v1["use_critic"] is False
    assert v2["rag_mode"] == "hybrid" and v2["use_cache"] is True and v2["use_critic"] is False
    assert v3["rag_mode"] == "hybrid" and v3["use_cache"] is True and v3["use_critic"] is True
    assert v2["system_prompt"] == V2_SYSTEM_PROMPT
    assert [t["name"] for t in v2["tools"]] == [t["name"] for t in V2_POKAYOKE_TOOLS]


# ---------------------------------------------------------------------------
# Multi-turn tool loop + cost ledger
# ---------------------------------------------------------------------------
def test_multiturn_tool_use_logs_cost_ledger(seeded_db_path, sample_case):
    client = MagicMock()
    # Turn 1: search tool
    # Turn 2: evaluate tool
    # Turn 3: benchmarks tool
    # Turn 4: final memo
    client.messages.create.side_effect = [
        _response(
            "tool_use",
            [
                _tool_block(
                    "tu1",
                    "search_deal_desk_policies_and_clauses",
                    {"query": "Net-90 margin floor", "top_k": 3},
                )
            ],
            _usage(200, 40, cache_creation=1500, cache_read=0),
        ),
        _response(
            "tool_use",
            [
                _tool_block(
                    "tu2",
                    "evaluate_deal_economics",
                    {
                        "annual_contract_value": 100000,
                        "term_months": 12,
                        "discount_pct": 0.19,
                        "payment_terms": "Net-90",
                        "custom_onboarding_hours": 100,
                    },
                )
            ],
            _usage(220, 50, cache_creation=0, cache_read=1400),
        ),
        _response(
            "tool_use",
            [
                _tool_block(
                    "tu3",
                    "query_historical_benchmarks",
                    {"segment": "Mid-Market", "discount_pct": 0.19},
                )
            ],
            _usage(240, 45, cache_creation=0, cache_read=1400),
        ),
        _response(
            "end_turn",
            [_text_block(_valid_memo_json())],
            _usage(260, 180, cache_creation=0, cache_read=1400),
        ),
    ]

    agent = DealDeskAgent(
        model="claude-sonnet-4-6",
        db_path=seeded_db_path,
        client=client,
        max_turns=8,
    )
    result = agent.run_case(sample_case, experiment_version="v2", run_id="test-v2-loop")

    assert result["case_id"] == "GE-015"
    assert result["experiment_version"] == "v2"
    assert result["revision_applied"] is False
    assert result["critic_review"] is None
    assert len(result["tool_trace"]) == 3
    assert {t["tool_name"] for t in result["tool_trace"]} == {
        "search_deal_desk_policies_and_clauses",
        "evaluate_deal_economics",
        "query_historical_benchmarks",
    }
    memo = DealDeskExecutiveMemo.model_validate(result["memo"])
    assert memo.required_approval_tier == "CFO-And-Legal"
    assert memo.calculated_gross_margin_pct == pytest.approx(59.78)

    cost = result["cost_summary"]
    assert cost["calls"] == 4
    assert cost["input_tokens"] == 200 + 220 + 240 + 260
    assert cost["output_tokens"] == 40 + 50 + 45 + 180
    assert cost["cache_creation_tokens"] == 1500
    assert cost["cache_read_tokens"] == 1400 * 3
    assert cost["total_cost_usd"] > 0

    # Verify cache_control present on v2 API kwargs
    first_kwargs = client.messages.create.call_args_list[0].kwargs
    assert isinstance(first_kwargs["system"], list)
    assert first_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first_kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in first_kwargs
    assert first_kwargs["max_tokens"] == 2048


def test_v1_has_no_cache_control_and_uses_basic_tools(seeded_db_path, sample_case):
    client = MagicMock()
    client.messages.create.side_effect = [
        _response(
            "tool_use",
            [
                _tool_block(
                    "tu1",
                    "evaluate_deal_economics",
                    {
                        "annual_contract_value": 100000,
                        "term_months": 12,
                        "discount_pct": 0.19,
                        "payment_terms": "Net-90",
                        "custom_onboarding_hours": 100,
                    },
                )
            ],
        ),
        _response("end_turn", [_text_block(_valid_memo_json())]),
    ]
    agent = DealDeskAgent(db_path=seeded_db_path, client=client)
    agent.run_case(sample_case, experiment_version="v1", run_id="test-v1")

    kwargs = client.messages.create.call_args_list[0].kwargs
    assert kwargs["system"] == V1_SYSTEM_PROMPT
    assert isinstance(kwargs["system"], str)
    tool_names = [t["name"] for t in kwargs["tools"]]
    assert tool_names == [t["name"] for t in V1_BASIC_TOOLS]
    assert "cache_control" not in kwargs["tools"][-1]


# ---------------------------------------------------------------------------
# v3 critic loop branches
# ---------------------------------------------------------------------------
def _v3_producer_then(extra_side_effects: list[Any], seeded_db_path, sample_case, run_id: str):
    client = MagicMock()
    producer = [
        _response(
            "tool_use",
            [
                _tool_block(
                    "tu1",
                    "evaluate_deal_economics",
                    {
                        "annual_contract_value": 100000,
                        "term_months": 12,
                        "discount_pct": 0.19,
                        "payment_terms": "Net-90",
                        "custom_onboarding_hours": 100,
                        "proposed_legal_flags": [],
                    },
                )
            ],
            _usage(100, 40),
        ),
        _response(
            "end_turn",
            [_text_block(_valid_memo_json())],
            _usage(120, 160),
        ),
    ]
    client.messages.create.side_effect = producer + extra_side_effects
    agent = DealDeskAgent(db_path=seeded_db_path, client=client)
    return agent.run_case(sample_case, experiment_version="v3", run_id=run_id), client


def test_v3_critic_no_revision_skips_producer_revision(seeded_db_path, sample_case):
    critic_json = json.dumps(
        {"needs_revision": False, "issues": [], "revision_instructions": ""}
    )
    result, client = _v3_producer_then(
        [_response("end_turn", [_text_block(critic_json)], _usage(80, 60))],
        seeded_db_path,
        sample_case,
        "test-v3-norev",
    )
    assert result["revision_applied"] is False
    assert result["critic_review"]["needs_revision"] is False
    steps = [
        r[0]
        for r in sqlite3.connect(seeded_db_path)
        .execute(
            "SELECT step_name FROM cost_ledger WHERE run_id = ? ORDER BY id",
            ("test-v3-norev",),
        )
        .fetchall()
    ]
    assert "critic" in steps
    assert "producer_revision" not in steps
    assert client.messages.create.call_count == 3  # 2 producer + critic


def test_v3_critic_needs_revision_triggers_producer_revision(seeded_db_path, sample_case):
    critic_json = json.dumps(
        {
            "needs_revision": True,
            "issues": ["Missing Net-90 fallback specificity"],
            "revision_instructions": (
                "Strengthen required_redline_fallbacks to demand Net-30 conversion."
            ),
        }
    )
    revised = _valid_memo_json(
        required_redline_fallbacks=[
            "Convert Net-90 to Net-30 immediately",
            "Move 100 onboarding hours to paid SOW at $150/hr",
        ],
        executive_summary="Revised after critic: restructure payment and services.",
    )
    result, client = _v3_producer_then(
        [
            _response("end_turn", [_text_block(critic_json)], _usage(90, 70)),
            _response("end_turn", [_text_block(revised)], _usage(110, 150)),
        ],
        seeded_db_path,
        sample_case,
        "test-v3-rev",
    )
    assert result["revision_applied"] is True
    assert result["critic_review"]["needs_revision"] is True
    assert "Convert Net-90 to Net-30 immediately" in result["memo"][
        "required_redline_fallbacks"
    ]
    steps = [
        r[0]
        for r in sqlite3.connect(seeded_db_path)
        .execute(
            "SELECT step_name FROM cost_ledger WHERE run_id = ? ORDER BY id",
            ("test-v3-rev",),
        )
        .fetchall()
    ]
    assert "critic" in steps
    assert "producer_revision" in steps
    assert client.messages.create.call_count == 4

    # Critic system prompt should be cached
    critic_call = client.messages.create.call_args_list[2].kwargs
    assert isinstance(critic_call["system"], list)
    assert critic_call["system"][0]["cache_control"]["type"] == "ephemeral"
    assert critic_call.get("tools") is None


# ---------------------------------------------------------------------------
# Malformed JSON recovery + max_turns
# ---------------------------------------------------------------------------
def test_malformed_json_triggers_single_repair(seeded_db_path, sample_case):
    client = MagicMock()
    client.messages.create.side_effect = [
        _response("end_turn", [_text_block("NOT VALID JSON {{{")], _usage(50, 20)),
        _response("end_turn", [_text_block(_valid_memo_json())], _usage(60, 100)),
    ]
    agent = DealDeskAgent(db_path=seeded_db_path, client=client)
    result = agent.run_case(sample_case, experiment_version="v1", run_id="test-repair")
    assert result["memo"]["deal_id"] == "GE-015"
    steps = [
        r[0]
        for r in sqlite3.connect(seeded_db_path)
        .execute(
            "SELECT step_name FROM cost_ledger WHERE run_id = ?",
            ("test-repair",),
        )
        .fetchall()
    ]
    assert "producer_json_repair" in steps


def test_max_turns_guardrail(seeded_db_path, sample_case):
    client = MagicMock()
    # Always request tools — never end_turn
    client.messages.create.side_effect = [
        _response(
            "tool_use",
            [
                _tool_block(
                    f"tu{i}",
                    "query_historical_benchmarks",
                    {"segment": "Mid-Market", "discount_pct": 0.1},
                )
            ],
        )
        for i in range(5)
    ]
    agent = DealDeskAgent(db_path=seeded_db_path, client=client, max_turns=3)
    with pytest.raises(RuntimeError, match="Exceeded max_turns"):
        agent.run_case(sample_case, experiment_version="v1", run_id="test-max-turns")
