"""Token accounting for the context visualizer.

Token math MUST match the runtime spinner (e.g. ``Tokens: 96,293/272,000``).
The spinner number originates in ``agents/_compaction.py`` -> the authoritative
``estimate_tokens_for_message(m, model_name)`` in ``agents/_history.py``, which
applies a per-model multiplier (e.g. ``opus-4-7`` x1.35). So this module always
delegates to core (``estimate_tokens_for_message`` / ``estimate_context_overhead``)
and resolves ``model_name`` once per snapshot -- never a homebrew char/N estimate,
which silently drops the multiplier and under-reports.

INTENTIONAL DIVERGENCE from ``plugins/context_indicator/usage.py``: that module
keeps a private char/2.5 heuristic (``_raw_estimate_tokens``) that is *immune* to
the token_ratio_learner monkeypatch, because the /context indicator wants a stable
reference number. This module wants the *learned* numbers that match the live
spinner. Do NOT merge the two estimators -- they answer different questions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextUsage:
    used_tokens: int
    overhead_tokens: int
    capacity: int
    system_prompt_tokens: int = 0
    agents_md_tokens: int = 0
    pydantic_tools_tokens: int = 0
    mcp_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.used_tokens + self.overhead_tokens


@dataclass(frozen=True)
class ChatSlice:
    label: str
    bucket: str
    tokens: int
    detail: str = ""


def raw_estimate_tokens(text: str, model_name: str | None = None) -> int:
    if not text:
        return 0
    from code_puppy.agents._history import _apply_multiplier, estimate_tokens

    return _apply_multiplier(estimate_tokens(text), model_name)


def _stringify_part(part: Any) -> str:
    try:
        from code_puppy.agents._history import stringify_part

        return stringify_part(part)
    except Exception:
        return str(part)


def message_hash(message: Any) -> int:
    try:
        from code_puppy.agents._history import hash_message

        return hash_message(message)
    except Exception:
        return hash(repr(message))


def raw_tokens_for_message(message: Any, model_name: str | None = None) -> int:
    from code_puppy.agents._history import estimate_tokens_for_message

    return estimate_tokens_for_message(message, model_name)


# Tool calls that spawn nested agents (sub-agents, judges, wiggum loops) all
# flow through the invoke_agent tool, so their tokens land in the parent history
# as these parts. Split them into their own bucket for visibility.
_SUBAGENT_TOOL_NAMES = {"invoke_agent"}


def _part_bucket(part: Any) -> str:
    cls_name = part.__class__.__name__.lower()
    role = str(getattr(part, "role", "") or "").lower()
    if "user" in cls_name or role == "user":
        return "chat_user"
    tool_name = getattr(part, "tool_name", None)
    if tool_name in _SUBAGENT_TOOL_NAMES:
        return "chat_subagent"
    if "tool" in cls_name or tool_name:
        return "chat_tool"
    return "chat_assistant"


def _chat_label(bucket: str) -> str:
    if bucket == "chat_user":
        return "User message"
    if bucket == "chat_subagent":
        return "Sub-agents / judges"
    if bucket == "chat_tool":
        return "Tool calls"
    return "Assistant work"


def _model_name(agent: Any) -> str | None:
    try:
        return agent.get_model_name()
    except Exception:
        return None


def _part_token_counts(parts: list[Any], total_tokens: int) -> list[int]:
    from code_puppy.agents._history import estimate_tokens

    weights = [estimate_tokens(_stringify_part(part)) for part in parts]
    weight_total = sum(weights)
    if weight_total <= 0:
        return []
    remaining = total_tokens
    counts: list[int] = []
    for index, weight in enumerate(weights):
        remaining_parts = len(weights) - index - 1
        if remaining_parts == 0:
            counts.append(max(0, remaining))
            break
        tokens = max(1, math.floor(total_tokens * weight / weight_total))
        tokens = min(tokens, max(0, remaining - remaining_parts))
        counts.append(tokens)
        remaining -= tokens
    return counts


def _detail_for(bucket: str, part: Any) -> str:
    # Tooltip text: carry the user's full typed prompt (the window renders it in
    # a scrollable popup, so we don't truncate here).
    if bucket != "chat_user":
        return ""
    return _stringify_part(part).strip()


def chat_slices(
    agent: Any, summarized_hashes: set[int] | None = None
) -> tuple[ChatSlice, ...]:
    summarized_hashes = summarized_hashes or set()
    model_name = _model_name(agent)
    # All summarized history collapses into ONE block: per-event detail lives in
    # the compaction table, and N separate slivers made one summarization look
    # like many stacked segments in the bar.
    summary_tokens = 0
    slices: list[ChatSlice] = []
    for message in agent.get_message_history() or []:
        message_tokens = raw_tokens_for_message(message, model_name)
        if message_hash(message) in summarized_hashes:
            summary_tokens += message_tokens
            continue
        parts = list(getattr(message, "parts", []) or [])
        if not parts:
            role = getattr(message, "role", None)
            bucket = "chat_user" if role == "user" else "chat_assistant"
            slices.append(ChatSlice(_chat_label(bucket), bucket, message_tokens))
            continue
        for part, tokens in zip(parts, _part_token_counts(parts, message_tokens)):
            bucket = _part_bucket(part)
            slices.append(
                ChatSlice(
                    _chat_label(bucket), bucket, tokens, _detail_for(bucket, part)
                )
            )
    summary_slices = (
        [ChatSlice("Summarized context", "summary", summary_tokens)]
        if summary_tokens > 0
        else []
    )
    return tuple(item for item in [*summary_slices, *slices] if item.tokens > 0)


def _agent_tools(agent: Any) -> dict | None:
    try:
        from code_puppy.agents.base_agent import _extract_pydantic_agent_tools
    except Exception:
        return None

    tools_source = getattr(agent, "pydantic_agent", None)
    if tools_source is None:
        probe_getter = getattr(agent, "_get_tool_probe", None)
        if callable(probe_getter):
            try:
                tools_source = probe_getter()
            except Exception:
                tools_source = None
    if tools_source is None:
        return None
    try:
        return _extract_pydantic_agent_tools(tools_source)
    except Exception:
        return None


def _live_mcp_servers_for(agent: Any) -> list[Any] | None:
    try:
        from code_puppy.config import get_value
        from code_puppy.mcp_ import get_mcp_manager

        disabled = get_value("disable_mcp_servers")
        if str(disabled or "").lower() in {"1", "true", "yes", "on"}:
            return None
        servers = get_mcp_manager().get_servers_for_agent(
            agent_name=getattr(agent, "name", None)
        )
        if servers:
            return servers
    except Exception:
        pass
    return getattr(agent, "_mcp_servers", None) or None


def _resolved_system_prompt(agent: Any) -> str:
    system_prompt = agent.get_full_system_prompt()
    try:
        from code_puppy.model_utils import prepare_prompt_for_model

        prepared = prepare_prompt_for_model(
            model_name=agent.get_model_name() or "",
            system_prompt=system_prompt,
            user_prompt="",
            prepend_system_to_user=False,
        )
        return prepared.instructions or system_prompt
    except Exception:
        return system_prompt


def _agents_md_tokens(model_name: str | None) -> int:
    # AGENTS.md / puppy rules are appended to the prompt at runtime by
    # _builder._assemble_instructions, so they count toward the live total but
    # are NOT part of get_full_system_prompt(). Surface them as their own bucket.
    try:
        from code_puppy.agents._builder import load_puppy_rules

        rules = load_puppy_rules() or ""
    except Exception:
        return 0
    return raw_estimate_tokens(rules, model_name) if rules else 0


def _model_capacity(agent: Any) -> int:
    try:
        return int(agent._get_model_context_length())
    except Exception:
        return 0


def get_current_usage() -> ContextUsage | None:
    try:
        from code_puppy.agents._history import estimate_context_overhead
        from code_puppy.agents.agent_manager import get_current_agent

        agent = get_current_agent()
    except Exception:
        return None
    if agent is None:
        return None

    try:
        capacity = _model_capacity(agent)
        if capacity <= 0:
            return None
        model_name = _model_name(agent)
        history = agent.get_message_history() or []
        used_tokens = sum(raw_tokens_for_message(m, model_name) for m in history)
        resolved_prompt = _resolved_system_prompt(agent)
        tools = _agent_tools(agent)
        mcp_servers = _live_mcp_servers_for(agent)
        agents_md_tokens = _agents_md_tokens(model_name)
        overhead = (
            estimate_context_overhead(
                resolved_prompt,
                tools,
                model_name,
                mcp_servers=mcp_servers,
            )
            + agents_md_tokens
        )
        system_prompt_tokens = estimate_context_overhead(
            resolved_prompt,
            None,
            model_name,
            mcp_servers=None,
        )
        pydantic_tools_tokens = estimate_context_overhead(
            "",
            tools,
            model_name,
            mcp_servers=None,
        )
        mcp_tokens = estimate_context_overhead(
            "",
            None,
            model_name,
            mcp_servers=mcp_servers,
        )
    except Exception:
        return None

    # Buckets are summed independently, so rounding inside the per-bucket
    # multiplier can make them drift a few tokens from the single-call overhead.
    # We pin the headline `overhead` (it drives % used) and absorb any rounding
    # delta into the system-prompt bucket so the legend reconciles exactly.
    component_total = (
        system_prompt_tokens + agents_md_tokens + pydantic_tools_tokens + mcp_tokens
    )
    if component_total != overhead:
        system_prompt_tokens = max(0, system_prompt_tokens + overhead - component_total)
    return ContextUsage(
        used_tokens=used_tokens,
        overhead_tokens=overhead,
        capacity=capacity,
        system_prompt_tokens=system_prompt_tokens,
        agents_md_tokens=agents_md_tokens,
        pydantic_tools_tokens=pydantic_tools_tokens,
        mcp_tokens=mcp_tokens,
    )
