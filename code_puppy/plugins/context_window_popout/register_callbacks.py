from __future__ import annotations

from collections import Counter
from typing import Any

from code_puppy.callbacks import register_callback
from code_puppy.plugins.context_window_popout.compaction_log import (
    mark_summarized_hashes,
    prune_summarized_hashes,
    record_compaction_event,
)
from code_puppy.plugins.context_window_popout.instance import get_name, set_name
from code_puppy.plugins.context_window_popout.snapshot import (
    CONFIG_KEY,
    visualizer_enabled_on_start,
)
from code_puppy.plugins.context_window_popout.usage import (
    get_current_usage,
    message_hash,
    raw_tokens_for_message,
)
from code_puppy.plugins.context_window_popout.window import (
    close_visualizer,
    ensure_visualizer_open,
    is_visualizer_open,
    last_error,
    notify_dirty,
    open_visualizer,
    retry_count,
)

_COMMAND = "context-visual"
_SOURCE_HASHES: Counter[int] = Counter()
_SOURCE_TOKENS = 0
_SOURCE_STRATEGY = "summarization"
_PENDING_STRATEGY: str | None = None
_PENDING_TOKENS_AT: int | None = None


def _emit_info(message: str) -> None:
    from code_puppy.messaging import emit_info

    emit_info(message)


def _emit_error(message: str) -> None:
    from code_puppy.messaging import emit_error

    emit_error(message)


def _announce() -> None:
    _emit_info(f"🐶 context-visual: {get_name()}")


def _on_startup() -> None:
    if visualizer_enabled_on_start():
        _announce()
        ok, message = open_visualizer()
        (_emit_info if ok else _emit_error)(message)


def _on_agent_activity(*_args, **_kwargs) -> str:
    ensure_visualizer_open()
    notify_dirty()  # push a fresh frame at the start of each agent turn
    return ""


def _configured_strategy() -> str:
    try:
        from code_puppy.config import get_compaction_strategy

        return get_compaction_strategy()
    except Exception:
        return "summarization"


def _compaction_threshold() -> float:
    try:
        from code_puppy.config import get_compaction_threshold

        return float(get_compaction_threshold())
    except Exception:
        return 0.0


def _capacity() -> int:
    try:
        usage = get_current_usage()
        return usage.capacity if usage is not None else 0
    except Exception:
        return 0


def _overhead_tokens() -> int:
    try:
        usage = get_current_usage()
        return usage.overhead_tokens if usage is not None else 0
    except Exception:
        return 0


def _current_model_name() -> str | None:
    try:
        from code_puppy.agents.agent_manager import get_current_agent

        agent = get_current_agent()
        return agent.get_model_name() if agent is not None else None
    except Exception:
        return None


def _tokens_for(messages: list[Any]) -> int:
    model_name = _current_model_name()
    return _overhead_tokens() + sum(
        raw_tokens_for_message(message, model_name) for message in messages
    )


def _message_hashes(messages: list[Any]) -> Counter[int]:
    return Counter(message_hash(message) for message in messages)


def _on_history_start(
    _agent_name: str | None,
    _session_id: str | None,
    message_history: list[Any],
    incoming_messages: list[Any],
) -> None:
    global _PENDING_STRATEGY, _PENDING_TOKENS_AT
    global _SOURCE_HASHES, _SOURCE_OVER_THRESHOLD
    global _SOURCE_STRATEGY, _SOURCE_TOKENS
    source_messages = [*message_history, *incoming_messages]
    _SOURCE_HASHES = _message_hashes(source_messages)
    _SOURCE_TOKENS = _tokens_for(source_messages)
    _SOURCE_STRATEGY = _configured_strategy()
    _PENDING_STRATEGY = None
    _PENDING_TOKENS_AT = None
    capacity = _capacity()
    threshold = _compaction_threshold()
    _SOURCE_OVER_THRESHOLD = (
        capacity > 0 and threshold > 0 and _SOURCE_TOKENS / capacity > threshold
    )
    notify_dirty()  # normal turn-start refresh


def _on_pre_compact(
    _agent_name: str | None,
    strategy: str,
    _message_count: int,
    token_count: int,
) -> None:
    global _PENDING_STRATEGY, _PENDING_TOKENS_AT
    _PENDING_STRATEGY = strategy
    _PENDING_TOKENS_AT = (
        _SOURCE_TOKENS if _SOURCE_HASHES else max(0, int(token_count or 0))
    )
    # pre_compact fires INSIDE compact(), after incoming messages are appended
    # but before summarization replaces them -- the only moment the live history
    # is at its over-threshold peak. Refresh here so the bar visibly climbs past
    # the threshold (summarization is a multi-second LLM call, so the async
    # writer has time to snapshot the peak before history shrinks).
    notify_dirty()


def _on_history_end(
    _agent_name: str | None,
    _session_id: str | None,
    message_history: list[Any],
    _messages_added: int,
    _messages_filtered: int,
) -> None:
    global _PENDING_STRATEGY, _PENDING_TOKENS_AT
    global _SOURCE_HASHES, _SOURCE_OVER_THRESHOLD
    global _SOURCE_STRATEGY, _SOURCE_TOKENS
    final_hashes = _message_hashes(message_history)
    removed = _SOURCE_HASHES - final_hashes
    added = final_hashes - _SOURCE_HASHES
    prune_summarized_hashes(set(final_hashes))

    if (
        _SOURCE_HASHES
        and (_PENDING_STRATEGY or _SOURCE_OVER_THRESHOLD)
        and (removed or added)
    ):
        strategy = _PENDING_STRATEGY or _SOURCE_STRATEGY
        if strategy == "summarization" and added:
            mark_summarized_hashes(set(added))
        elif strategy == "summarization" and removed:
            strategy = "summarization→truncation"
        record_compaction_event(
            strategy,
            _PENDING_TOKENS_AT if _PENDING_TOKENS_AT is not None else _SOURCE_TOKENS,
            _tokens_for(message_history),
        )

    _SOURCE_HASHES = Counter()
    _SOURCE_TOKENS = 0
    _SOURCE_STRATEGY = "summarization"
    _SOURCE_OVER_THRESHOLD = False
    _PENDING_STRATEGY = None
    _PENDING_TOKENS_AT = None
    notify_dirty()  # history/compaction changed -> repaint promptly


def _custom_help():
    return [
        (
            _COMMAND,
            f"Open/status/close the context visualizer: /context-visual [open|status|close|name]; /set {CONFIG_KEY} true for startup",
        )
    ]


def _handle_custom_command(command: str, name: str):
    if name != _COMMAND:
        return None
    args = command.strip().split()[1:]
    action = args[0].lower() if args else "open"

    if action in {"close", "off"}:
        ok, message = close_visualizer()
    elif action == "status":
        state = "open" if is_visualizer_open() else "closed"
        error = last_error()
        retries = retry_count()
        message = (
            f"Context visualizer is {state}."
            + (f" Respawns: {retries}." if retries else "")
            + (f" Last error: {error}" if error else "")
        )
        ok = True
    else:
        if action == "open":
            if len(args) > 1:
                set_name(args[1])
        else:
            set_name(args[0])
        _announce()
        ok, message = open_visualizer()
    (_emit_info if ok else _emit_error)(message)
    return True


register_callback("startup", _on_startup)
register_callback("agent_run_start", _on_agent_activity)
register_callback("message_history_processor_start", _on_history_start)
register_callback("pre_compact", _on_pre_compact)
register_callback("message_history_processor_end", _on_history_end)
register_callback("custom_command", _handle_custom_command)
register_callback("custom_command_help", _custom_help)
