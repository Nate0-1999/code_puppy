"""Context aggregation for the context visualizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .compaction_log import (
    get_compaction_events,
    get_compaction_summary,
    get_summarized_hashes,
)
from .usage import chat_slices, get_current_usage, raw_estimate_tokens

CONFIG_KEY = "context-visual-on-start"

COLORS = {
    "base": "#00e5ff",
    "agent": "#ffd000",
    "agents_md": "#00ff66",
    "chat_user": "#c6ff00",
    "chat_assistant": "#3d7bff",
    "chat_tool": "#ff7a00",
    "chat_subagent": "#ff4fd8",
    "summary": "#b14dff",
    "tools": "#ff00aa",
    "free": "#1b263b",
}

_AGENT_RULE_FILES = ("AGENTS.md", "AGENT.md", "agents.md", "agent.md")
_CODE_PUPPY_DIR = ".code_puppy"


@dataclass(frozen=True)
class ContextSegment:
    label: str
    tokens: int
    color: str
    detail: str = ""

    def percent_of(self, capacity: int) -> float:
        if capacity <= 0:
            return 0.0
        return self.tokens / capacity * 100.0


@dataclass(frozen=True)
class PathPart:
    label: str
    agent_rules_state: str = "none"

    @property
    def has_agent_rules(self) -> bool:
        return self.agent_rules_state != "none"

    @property
    def applies_to_prompt(self) -> bool:
        return self.agent_rules_state == "applied"


@dataclass(frozen=True)
class ContextSnapshot:
    capacity: int
    segments: tuple[ContextSegment, ...]
    config_lines: tuple[str, ...]
    status: str = ""
    instance_name: str = ""
    agent_name: str = ""
    cwd_parts: tuple[PathPart, ...] = field(default_factory=tuple)
    compaction_threshold: float = 0.0
    total_tokens: int | None = None
    compaction_note: str = ""
    # Rows: (index, datetime, strategy, tokens_at, tokens_to, duration_min|None).
    compaction_rows: tuple[tuple[int, str, str, int, int, float | None], ...] = field(
        default_factory=tuple
    )
    # Summary: (count, avg_at, avg_to, avg_duration_min|None) or None.
    compaction_summary: tuple[int, float, float, float | None] | None = None

    @property
    def used_tokens(self) -> int:
        if self.total_tokens is not None:
            return self.total_tokens
        return sum(segment.tokens for segment in self.segments)

    @property
    def percent_used(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return self.used_tokens / self.capacity * 100.0

    @property
    def free_tokens(self) -> int:
        return max(0, self.capacity - self.used_tokens)

    def segments_with_free(self) -> tuple[ContextSegment, ...]:
        if self.capacity <= 0:
            return self.segments
        return self.segments + (
            ContextSegment("Unused capacity", self.free_tokens, COLORS["free"]),
        )

    def legend_segments(self) -> tuple[ContextSegment, ...]:
        grouped: dict[tuple[str, str], int] = {}
        for segment in self.segments:
            key = (segment.label, segment.color)
            grouped[key] = grouped.get(key, 0) + segment.tokens
        # Sorted by tokens desc == % of capacity desc (shared capacity).
        return tuple(
            ContextSegment(label, tokens, color)
            for (label, color), tokens in sorted(
                grouped.items(), key=lambda kv: kv[1], reverse=True
            )
        )


def config_enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def visualizer_enabled_on_start() -> bool:
    try:
        from code_puppy.config import get_value

        return config_enabled(get_value(CONFIG_KEY))
    except Exception:
        return False


def _current_agent_or_none() -> Any | None:
    try:
        from code_puppy.agents.agent_manager import get_current_agent

        return get_current_agent()
    except Exception:
        return None


def _chat_segments(agent: Any) -> tuple[ContextSegment, ...]:
    try:
        return tuple(
            ContextSegment(item.label, item.tokens, COLORS[item.bucket], item.detail)
            for item in chat_slices(agent, get_summarized_hashes())
        )
    except Exception:
        return ()


def _has_agent_rules(folder: Path) -> bool:
    for name in _AGENT_RULE_FILES:
        if (folder / name).is_file() or (folder / _CODE_PUPPY_DIR / name).is_file():
            return True
    return False


def _applied_project_rule_folder() -> Path | None:
    code_puppy_dir = Path(_CODE_PUPPY_DIR)
    if code_puppy_dir.is_dir():
        for name in _AGENT_RULE_FILES:
            candidate = code_puppy_dir / name
            if candidate.exists():
                return Path.cwd().resolve()
    for name in _AGENT_RULE_FILES:
        if Path(name).exists():
            return Path.cwd().resolve()
    return None


def _cwd_parts() -> tuple[PathPart, ...]:
    try:
        cwd = Path.cwd().resolve()
    except Exception:
        return ()
    applied_folder = _applied_project_rule_folder()
    folders = tuple(reversed(cwd.parents)) + (cwd,)
    parts = []
    for folder in folders:
        label = folder.anchor.rstrip("\\/") or folder.name or folder.anchor or "/"
        if applied_folder and folder == applied_folder:
            state = "applied"
        elif _has_agent_rules(folder):
            state = "available"
        else:
            state = "none"
        parts.append(PathPart(label=label, agent_rules_state=state))
    return tuple(parts)


def _compaction_threshold() -> float:
    try:
        from code_puppy.config import get_compaction_threshold

        return float(get_compaction_threshold())
    except Exception:
        return 0.0


def _compaction_note(
    agent: Any | None,
    total_tokens: int,
    capacity: int,
    threshold: float,
) -> str:
    if capacity <= 0 or threshold <= 0:
        return ""
    if (total_tokens / capacity) <= threshold:
        return ""

    try:
        from code_puppy.config import get_compaction_strategy

        strategy = get_compaction_strategy()
    except Exception:
        strategy = "summarization"

    if strategy == "summarization" and agent is not None:
        try:
            from code_puppy.agents._history import (
                filter_huge_messages,
                has_pending_tool_calls,
            )

            model_name = agent.get_model_name()
            history = agent.get_message_history() or []
            if has_pending_tool_calls(filter_huge_messages(history, model_name)):
                return "⚠ compaction deferred: pending tool call(s)"
        except Exception:
            pass

    return "⚠ over threshold; compaction runs on next agent cycle"


def _startup_config_value() -> str:
    try:
        from code_puppy.config import get_value

        return get_value(CONFIG_KEY) or "false"
    except Exception:
        return "false"


def _config_lines(capacity: int) -> tuple[str, ...]:
    try:
        from code_puppy.config import (
            get_compaction_strategy,
            get_compaction_threshold,
            get_global_model_name,
        )

        return (
            f"model: {get_global_model_name()}",
            f"model context: {capacity:,} tokens",
            f"/set compaction_strategy {get_compaction_strategy()}",
            f"/set compaction_threshold {get_compaction_threshold():.0%}",
            f"/set {CONFIG_KEY} {_startup_config_value()}",
        )
    except Exception:
        return (f"model context: {capacity:,} tokens",)


def snapshot_to_payload(snapshot: ContextSnapshot) -> dict[str, Any]:
    return {
        "capacity": snapshot.capacity,
        "status": snapshot.status,
        "instance_name": snapshot.instance_name,
        "agent_name": snapshot.agent_name,
        "config_lines": list(snapshot.config_lines),
        "compaction_threshold": snapshot.compaction_threshold,
        "total_tokens": snapshot.total_tokens,
        "compaction_note": snapshot.compaction_note,
        "cwd_parts": [
            {
                "label": part.label,
                "agent_rules_state": part.agent_rules_state,
                "has_agent_rules": part.has_agent_rules,
            }
            for part in snapshot.cwd_parts
        ],
        "compaction_rows": [list(row) for row in snapshot.compaction_rows],
        "compaction_summary": (
            list(snapshot.compaction_summary)
            if snapshot.compaction_summary is not None
            else None
        ),
        "segments": [
            {
                "label": segment.label,
                "tokens": segment.tokens,
                "color": segment.color,
                "detail": segment.detail,
            }
            for segment in snapshot.segments
        ],
    }


def snapshot_from_payload(payload: dict[str, Any]) -> ContextSnapshot:
    return ContextSnapshot(
        capacity=int(payload.get("capacity") or 0),
        status=str(payload.get("status") or ""),
        instance_name=str(payload.get("instance_name") or ""),
        agent_name=str(payload.get("agent_name") or ""),
        config_lines=tuple(str(line) for line in payload.get("config_lines") or ()),
        cwd_parts=tuple(
            PathPart(
                label=str(part.get("label") or ""),
                agent_rules_state=str(
                    part.get("agent_rules_state")
                    or ("applied" if part.get("has_agent_rules") else "none")
                ),
            )
            for part in payload.get("cwd_parts") or ()
        ),
        compaction_threshold=float(payload.get("compaction_threshold") or 0.0),
        total_tokens=(
            int(payload["total_tokens"])
            if payload.get("total_tokens") is not None
            else None
        ),
        compaction_note=str(payload.get("compaction_note") or ""),
        compaction_rows=tuple(
            (
                int(row[0]),
                str(row[1]),
                str(row[2]),
                int(row[3]),
                int(row[4]),
                (None if row[5] is None else float(row[5])),
            )
            for row in payload.get("compaction_rows") or ()
        ),
        compaction_summary=(
            (
                int(payload["compaction_summary"][0]),
                float(payload["compaction_summary"][1]),
                float(payload["compaction_summary"][2]),
                (
                    None
                    if payload["compaction_summary"][3] is None
                    else float(payload["compaction_summary"][3])
                ),
            )
            if payload.get("compaction_summary") is not None
            else None
        ),
        segments=tuple(
            ContextSegment(
                label=str(segment.get("label") or ""),
                tokens=int(segment.get("tokens") or 0),
                color=str(segment.get("color") or COLORS["free"]),
                detail=str(segment.get("detail") or ""),
            )
            for segment in payload.get("segments") or ()
        ),
    )


def _short_dt(timestamp: str) -> str:
    # "%m/%d/%Y %H:%M:%S" -> "MM/DD HH:MM" for a compact table cell.
    try:
        date_part, time_part = timestamp.split(" ")
        mm, dd, _yyyy = date_part.split("/")
        hh, mi, _ss = time_part.split(":")
        return f"{mm}/{dd} {hh}:{mi}"
    except Exception:
        return timestamp


def _compaction_rows() -> tuple[tuple[int, str, str, int, int, float | None], ...]:
    return tuple(
        (
            e.index,
            _short_dt(e.timestamp),
            e.strategy,
            e.tokens_at,
            e.tokens_to,
            e.duration_min,
        )
        for e in get_compaction_events()
    )


def _compaction_summary() -> tuple[int, float, float, float | None] | None:
    s = get_compaction_summary()
    if s is None:
        return None
    return (s.count, s.avg_at, s.avg_to, s.avg_duration_min)


def _current_agent_name() -> str:
    # Surfaced in the header so /agent and /wiggum switches are visible. Each
    # agent owns its own message history, so the window legitimately resets to
    # that agent's context window on switch -- this is real behavior, not a bug.
    try:
        from code_puppy.agents.agent_manager import get_current_agent

        agent = get_current_agent()
        if agent is None:
            return ""
        return getattr(agent, "display_name", None) or getattr(agent, "name", "") or ""
    except Exception:
        return ""


def collect_snapshot() -> ContextSnapshot:
    from code_puppy.plugins.context_window_popout.instance import get_name

    instance_name = get_name()
    agent_name = _current_agent_name()
    cwd_parts = _cwd_parts()
    compaction_threshold = _compaction_threshold()

    usage = get_current_usage()

    if usage is None:
        return ContextSnapshot(
            capacity=0,
            segments=(),
            config_lines=_config_lines(0),
            status="No context data yet. Send a message after selecting an agent.",
            instance_name=instance_name,
            agent_name=agent_name,
            cwd_parts=cwd_parts,
            compaction_threshold=compaction_threshold,
            compaction_rows=_compaction_rows(),
            compaction_summary=_compaction_summary(),
        )

    agent = _current_agent_or_none()
    agent_tokens = 0
    if agent is not None:
        try:
            agent_tokens = raw_estimate_tokens(
                agent.get_system_prompt(), agent.get_model_name()
            )
        except Exception:
            agent_tokens = 0
    agent_tokens = min(agent_tokens, usage.system_prompt_tokens)
    base_tokens = max(0, usage.system_prompt_tokens - agent_tokens)
    chat_segments = _chat_segments(agent) if agent is not None else ()

    raw_segments = (
        ContextSegment("Base system prompt", base_tokens, COLORS["base"]),
        ContextSegment("Agent prompt", agent_tokens, COLORS["agent"]),
        ContextSegment("AGENTS.md", usage.agents_md_tokens, COLORS["agents_md"]),
        ContextSegment(
            "Tool schemas / MCP",
            usage.pydantic_tools_tokens + usage.mcp_tokens,
            COLORS["tools"],
        ),
        *chat_segments,
    )
    segments = tuple(segment for segment in raw_segments if segment.tokens > 0)
    return ContextSnapshot(
        capacity=usage.capacity,
        segments=segments,
        config_lines=_config_lines(usage.capacity),
        instance_name=instance_name,
        agent_name=agent_name,
        cwd_parts=cwd_parts,
        compaction_threshold=compaction_threshold,
        total_tokens=usage.total_tokens,
        compaction_note=_compaction_note(
            agent,
            usage.total_tokens,
            usage.capacity,
            compaction_threshold,
        ),
        compaction_rows=_compaction_rows(),
        compaction_summary=_compaction_summary(),
    )
