"""Framework-free Anthropic Messages API Deal-Desk agent with caching + critic loop."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from src.cost_ledger import CostLedger
from src.prompts import (
    CRITIC_SYSTEM_PROMPT,
    V1_SYSTEM_PROMPT,
    V2_SYSTEM_PROMPT,
)
from src.tools import V1_BASIC_TOOLS, V2_POKAYOKE_TOOLS, execute_tool

ExperimentVersion = Literal["v1", "v2", "v3"]
DEFAULT_DB_PATH = "db/deal_desk.db"
DEFAULT_MODEL = "claude-sonnet-4-6"


class DealDeskExecutiveMemo(BaseModel):
    deal_id: str
    account_name: str
    executive_recommendation: Literal[
        "APPROVE", "ESCALATE_WITH_CONDITIONS", "REJECT_OR_RESTRUCTURE"
    ]
    required_approval_tier: Literal["Auto-Approve", "VP-Sales", "CFO-And-Legal"]
    calculated_gross_margin_pct: float
    margin_floor_pct: float
    detected_risk_flags: list[str] = Field(default_factory=list)
    executive_summary: str
    commercial_and_margin_analysis: str
    legal_and_policy_findings: str
    required_redline_fallbacks: list[str] = Field(default_factory=list)


def _parse_json_response(raw_text: str) -> dict:
    """Strip optional markdown fences and parse JSON; raise ValueError with context."""
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Empty model response; expected JSON object.")

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse JSON from model response: {exc}. "
            f"Raw text (truncated): {text[:500]!r}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    return data


def _content_blocks_to_dicts(content: Any) -> list[dict]:
    """Normalize Anthropic content blocks to plain dicts for message history."""
    blocks: list[dict] = []
    for block in content:
        btype = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if btype == "text":
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text", "")
            blocks.append({"type": "text", "text": text or ""})
        elif btype == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": getattr(block, "id", None)
                    or (block.get("id") if isinstance(block, dict) else None),
                    "name": getattr(block, "name", None)
                    or (block.get("name") if isinstance(block, dict) else None),
                    "input": getattr(block, "input", None)
                    or (block.get("input") if isinstance(block, dict) else {}),
                }
            )
        elif isinstance(block, dict):
            blocks.append(block)
    return blocks


def _extract_text(content: Any) -> str:
    parts: list[str] = []
    for block in content:
        btype = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if btype == "text":
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _usage_tokens(usage: Any) -> dict[str, int]:
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "cache_creation_tokens": int(
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        ),
        "cache_read_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
    }


def _version_config(experiment_version: ExperimentVersion) -> dict[str, Any]:
    if experiment_version == "v1":
        return {
            "system_prompt": V1_SYSTEM_PROMPT,
            "tools": copy.deepcopy(V1_BASIC_TOOLS),
            "rag_mode": "bm25",
            "use_cache": False,
            "use_critic": False,
        }
    if experiment_version == "v2":
        return {
            "system_prompt": V2_SYSTEM_PROMPT,
            "tools": copy.deepcopy(V2_POKAYOKE_TOOLS),
            "rag_mode": "hybrid",
            "use_cache": True,
            "use_critic": False,
        }
    if experiment_version == "v3":
        return {
            "system_prompt": V2_SYSTEM_PROMPT,
            "tools": copy.deepcopy(V2_POKAYOKE_TOOLS),
            "rag_mode": "hybrid",
            "use_cache": True,
            "use_critic": True,
        }
    raise ValueError(f"Unknown experiment_version: {experiment_version!r}")


def _apply_cache_control(tools: list[dict], use_cache: bool) -> list[dict]:
    tools = copy.deepcopy(tools)
    if use_cache and tools:
        tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
    return tools


def _system_param(system_prompt: str, use_cache: bool) -> str | list[dict]:
    if use_cache:
        return [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    return system_prompt


def _case_payload(case_row: dict) -> dict:
    if "deal_payload_json" in case_row:
        payload = case_row["deal_payload_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
    else:
        payload = case_row.get("deal_payload") or case_row
    return dict(payload)


def _build_user_message(case_row: dict, payload: dict) -> str:
    case_id = case_row.get("case_id") or payload.get("deal_id") or "UNKNOWN"
    body = {
        "case_id": case_id,
        "scenario_name": case_row.get("scenario_name"),
        "deal_payload": {
            **payload,
            "deal_id": payload.get("deal_id", case_id),
        },
    }
    return (
        "Review the following B2B SaaS deal-desk request and produce the executive "
        "memo JSON after using the required tools.\n\n"
        f"{json.dumps(body, indent=2)}"
    )


class DealDeskAgent:
    """Raw Anthropic Messages API tool-using deal-desk agent."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        db_path: str = DEFAULT_DB_PATH,
        client: anthropic.Anthropic | None = None,
        max_turns: int = 8,
    ) -> None:
        self.model = model
        self.db_path = db_path
        self.client = client or anthropic.Anthropic()
        self.max_turns = max_turns
        self.ledger = CostLedger(db_path)

    def _api_call(
        self,
        *,
        run_id: str,
        experiment_version: str,
        case_id: str,
        step_name: str,
        system: str | list[dict],
        messages: list[dict],
        tools: list[dict] | None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 2048,
            "temperature": 0.0,
            "system": system,
            "messages": messages,
        }
        if tools is not None:
            kwargs["tools"] = tools

        t0 = time.perf_counter()
        response = self.client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        tokens = _usage_tokens(response.usage)
        self.ledger.record_call(
            run_id=run_id,
            experiment_version=experiment_version,
            case_id=case_id,
            step_name=step_name,
            model=self.model,
            input_tokens=tokens["input_tokens"],
            output_tokens=tokens["output_tokens"],
            cache_creation_tokens=tokens["cache_creation_tokens"],
            cache_read_tokens=tokens["cache_read_tokens"],
            latency_ms=latency_ms,
        )
        return response

    def _parse_memo(
        self,
        raw_text: str,
        *,
        messages: list[dict],
        system: str | list[dict],
        tools: list[dict],
        run_id: str,
        experiment_version: str,
        case_id: str,
        allow_retry: bool,
    ) -> DealDeskExecutiveMemo:
        try:
            data = _parse_json_response(raw_text)
            return DealDeskExecutiveMemo.model_validate(data)
        except (ValueError, ValidationError) as exc:
            if not allow_retry:
                raise
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid DealDeskExecutiveMemo JSON. "
                        f"Validation error: {exc}\n"
                        "Return ONLY a corrected JSON object matching the required schema."
                    ),
                }
            )
            response = self._api_call(
                run_id=run_id,
                experiment_version=experiment_version,
                case_id=case_id,
                step_name="producer_json_repair",
                system=system,
                messages=messages,
                tools=tools,
            )
            messages.append(
                {"role": "assistant", "content": _content_blocks_to_dicts(response.content)}
            )
            repaired = _extract_text(response.content)
            data = _parse_json_response(repaired)
            return DealDeskExecutiveMemo.model_validate(data)

    def _run_producer_loop(
        self,
        *,
        messages: list[dict],
        system: str | list[dict],
        tools: list[dict],
        rag_mode: str,
        run_id: str,
        experiment_version: str,
        case_id: str,
        tool_trace: list[dict],
        turn_offset: int = 0,
        max_turns_override: int | None = None,
    ) -> DealDeskExecutiveMemo:
        max_turns = max_turns_override if max_turns_override is not None else self.max_turns
        for turn in range(1, max_turns + 1):
            response = self._api_call(
                run_id=run_id,
                experiment_version=experiment_version,
                case_id=case_id,
                step_name=f"producer_turn_{turn_offset + turn}",
                system=system,
                messages=messages,
                tools=tools,
            )
            assistant_content = _content_blocks_to_dicts(response.content)
            messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason == "tool_use":
                tool_results: list[dict] = []
                for block in assistant_content:
                    if block.get("type") != "tool_use":
                        continue
                    tool_name = block["name"]
                    tool_input = block.get("input") or {}
                    result_str = execute_tool(
                        tool_name,
                        tool_input,
                        db_path=self.db_path,
                        rag_mode=rag_mode,
                    )
                    tool_trace.append(
                        {
                            "turn": turn_offset + turn,
                            "tool_name": tool_name,
                            "tool_input": tool_input,
                            "tool_output": result_str,
                        }
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": result_str,
                        }
                    )
                messages.append({"role": "user", "content": tool_results})
                continue

            if response.stop_reason == "end_turn":
                raw_text = _extract_text(response.content)
                return self._parse_memo(
                    raw_text,
                    messages=messages,
                    system=system,
                    tools=tools,
                    run_id=run_id,
                    experiment_version=experiment_version,
                    case_id=case_id,
                    allow_retry=True,
                )

            # Unexpected stop — try to parse any text present
            raw_text = _extract_text(response.content)
            if raw_text:
                return self._parse_memo(
                    raw_text,
                    messages=messages,
                    system=system,
                    tools=tools,
                    run_id=run_id,
                    experiment_version=experiment_version,
                    case_id=case_id,
                    allow_retry=True,
                )

        raise RuntimeError(
            f"Exceeded max_turns={max_turns} without producing a final memo "
            f"(case_id={case_id}, run_id={run_id})."
        )

    def _run_critic(
        self,
        *,
        case_id: str,
        run_id: str,
        experiment_version: str,
        payload: dict,
        tool_trace: list[dict],
        memo: DealDeskExecutiveMemo,
        use_cache: bool,
    ) -> dict:
        critic_user = {
            "case_id": case_id,
            "deal_payload": payload,
            "tool_trace": tool_trace,
            "producer_draft_memo": memo.model_dump(),
        }
        system = _system_param(CRITIC_SYSTEM_PROMPT, use_cache=use_cache)
        messages = [
            {
                "role": "user",
                "content": (
                    "Audit the Producer draft against tool_trace and deal_payload. "
                    "Return the Critic JSON object only.\n\n"
                    f"{json.dumps(critic_user, indent=2)}"
                ),
            }
        ]
        response = self._api_call(
            run_id=run_id,
            experiment_version=experiment_version,
            case_id=case_id,
            step_name="critic",
            system=system,
            messages=messages,
            tools=None,
        )
        raw = _extract_text(response.content)
        return _parse_json_response(raw)

    def run_case(
        self,
        case_row: dict,
        experiment_version: ExperimentVersion = "v3",
        run_id: str | None = None,
    ) -> dict:
        cfg = _version_config(experiment_version)
        payload = _case_payload(case_row)
        case_id = str(case_row.get("case_id") or payload.get("deal_id") or "UNKNOWN")
        run_id = run_id or f"{experiment_version}_{case_id}_{uuid.uuid4().hex[:8]}"

        tools = _apply_cache_control(cfg["tools"], cfg["use_cache"])
        system = _system_param(cfg["system_prompt"], cfg["use_cache"])
        messages: list[dict] = [
            {"role": "user", "content": _build_user_message(case_row, payload)}
        ]
        tool_trace: list[dict] = []

        memo = self._run_producer_loop(
            messages=messages,
            system=system,
            tools=tools,
            rag_mode=cfg["rag_mode"],
            run_id=run_id,
            experiment_version=experiment_version,
            case_id=case_id,
            tool_trace=tool_trace,
        )

        critic_review: dict | None = None
        revision_applied = False

        if cfg["use_critic"]:
            critic_review = self._run_critic(
                case_id=case_id,
                run_id=run_id,
                experiment_version=experiment_version,
                payload=payload,
                tool_trace=tool_trace,
                memo=memo,
                use_cache=cfg["use_cache"],
            )
            if bool(critic_review.get("needs_revision")):
                revision_applied = True
                instructions = critic_review.get("revision_instructions") or (
                    "Revise the memo to address Critic issues."
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "CRITIC REVISION REQUIRED. Apply these instructions and "
                            "return the corrected DealDeskExecutiveMemo JSON only.\n\n"
                            f"Issues: {json.dumps(critic_review.get('issues', []))}\n"
                            f"Instructions: {instructions}"
                        ),
                    }
                )
                # One final Producer synthesis turn (may still tool-call within a short budget)
                response = self._api_call(
                    run_id=run_id,
                    experiment_version=experiment_version,
                    case_id=case_id,
                    step_name="producer_revision",
                    system=system,
                    messages=messages,
                    tools=tools,
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": _content_blocks_to_dicts(response.content),
                    }
                )

                # If revision turn requests tools, satisfy them then force a final text turn
                if response.stop_reason == "tool_use":
                    tool_results: list[dict] = []
                    for block in _content_blocks_to_dicts(response.content):
                        if block.get("type") != "tool_use":
                            continue
                        result_str = execute_tool(
                            block["name"],
                            block.get("input") or {},
                            db_path=self.db_path,
                            rag_mode=cfg["rag_mode"],
                        )
                        tool_trace.append(
                            {
                                "turn": "revision",
                                "tool_name": block["name"],
                                "tool_input": block.get("input") or {},
                                "tool_output": result_str,
                            }
                        )
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block["id"],
                                "content": result_str,
                            }
                        )
                    messages.append({"role": "user", "content": tool_results})
                    memo = self._run_producer_loop(
                        messages=messages,
                        system=system,
                        tools=tools,
                        rag_mode=cfg["rag_mode"],
                        run_id=run_id,
                        experiment_version=experiment_version,
                        case_id=case_id,
                        tool_trace=tool_trace,
                        turn_offset=100,
                        max_turns_override=3,
                    )
                else:
                    raw_text = _extract_text(response.content)
                    memo = self._parse_memo(
                        raw_text,
                        messages=messages,
                        system=system,
                        tools=tools,
                        run_id=run_id,
                        experiment_version=experiment_version,
                        case_id=case_id,
                        allow_retry=True,
                    )

        return {
            "run_id": run_id,
            "case_id": case_id,
            "experiment_version": experiment_version,
            "memo": memo.model_dump(),
            "tool_trace": tool_trace,
            "critic_review": critic_review,
            "revision_applied": revision_applied,
            "cost_summary": self.ledger.get_run_summary(run_id),
        }


def load_golden_case(case_id: str, db_path: str = DEFAULT_DB_PATH) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM golden_eval_cases WHERE case_id = ?",
        (case_id,),
    ).fetchone()
    conn.close()
    if row is None:
        raise LookupError(f"No golden_eval_cases row for case_id={case_id!r}")
    return dict(row)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run DealDeskAgent on a golden case")
    parser.add_argument("--case-id", default="GE-015")
    parser.add_argument(
        "--version",
        default="v3",
        choices=["v1", "v2", "v3"],
    )
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    case_row = load_golden_case(args.case_id, args.db_path)
    agent = DealDeskAgent(model=args.model, db_path=args.db_path)
    result = agent.run_case(case_row, experiment_version=args.version)
    memo = result["memo"]
    cost = result["cost_summary"]

    print("=== DealDeskAgent Smoke Summary ===")
    print(f"run_id:              {result['run_id']}")
    print(f"case_id:             {result['case_id']}")
    print(f"experiment_version:  {result['experiment_version']}")
    print(f"approval_tier:       {memo['required_approval_tier']}")
    print(f"gross_margin_pct:    {memo['calculated_gross_margin_pct']}")
    print(f"detected_flags:      {memo['detected_risk_flags']}")
    print(f"recommendation:      {memo['executive_recommendation']}")
    print(f"revision_applied:    {result['revision_applied']}")
    print(f"input_tokens:        {cost['input_tokens']}")
    print(f"output_tokens:       {cost['output_tokens']}")
    print(f"cache_creation:      {cost['cache_creation_tokens']}")
    print(f"cache_read:          {cost['cache_read_tokens']}")
    print(f"cache_hit_rate_pct:  {cost['cache_hit_rate_pct']}")
    print(f"total_cost_usd:      {cost['total_cost_usd']}")
    print(f"total_latency_ms:    {cost['total_latency_ms']}")


if __name__ == "__main__":
    main()
