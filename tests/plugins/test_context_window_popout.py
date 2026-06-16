from __future__ import annotations

from types import SimpleNamespace

from code_puppy.plugins.context_window_popout import compaction_log, instance
from code_puppy.plugins.context_window_popout import register_callbacks as callbacks
from code_puppy.plugins.context_window_popout import snapshot as visualizer_snapshot
from code_puppy.plugins.context_window_popout import usage as visualizer_usage
from code_puppy.plugins.context_window_popout import window as visualizer_window
from code_puppy.plugins.context_window_popout.snapshot import (
    ContextSegment,
    ContextSnapshot,
    PathPart,
    snapshot_from_payload,
    snapshot_to_payload,
)


def test_instance_defaults_to_autosave_session_and_command_can_name(monkeypatch):
    monkeypatch.setattr(instance, "_NAME", None)
    monkeypatch.setattr(
        "code_puppy.config.get_current_autosave_session_name",
        lambda: "auto_session_20260528_130728",
    )
    messages = []
    monkeypatch.setattr(callbacks, "_emit_info", messages.append)
    monkeypatch.setattr(callbacks, "open_visualizer", lambda: (True, "opened"))

    assert instance.get_name() == "auto_session_20260528_130728"
    assert callbacks._handle_custom_command("/context-visual pane-a", "context-visual")

    assert instance.get_name() == "pane-a"
    assert messages == ["🐶 context-visual: pane-a", "opened"]


def test_context_visual_command_variants(monkeypatch):
    messages = []
    monkeypatch.setattr(callbacks, "_emit_info", messages.append)
    monkeypatch.setattr(callbacks, "open_visualizer", lambda: (True, "opened"))
    monkeypatch.setattr(callbacks, "close_visualizer", lambda: (True, "closed"))
    monkeypatch.setattr(callbacks, "is_visualizer_open", lambda: False)
    monkeypatch.setattr(callbacks, "last_error", lambda: None)

    assert callbacks._handle_custom_command(
        "/context-visual open named", "context-visual"
    )
    assert callbacks._handle_custom_command("/context-visual status", "context-visual")
    assert callbacks._handle_custom_command("/context-visual close", "context-visual")
    assert callbacks._handle_custom_command("/context-visual off", "context-visual")

    assert "🐶 context-visual: named" in messages
    assert "opened" in messages
    assert "Context visualizer is closed." in messages
    assert messages.count("closed") == 2


def test_snapshot_payload_contract_round_trips_visible_state():
    snapshot = ContextSnapshot(
        capacity=100,
        segments=(ContextSegment("User message", 40, "#c6ff00"),),
        config_lines=("model: test",),
        instance_name="pane-a",
        cwd_parts=(PathPart("repo", "applied"), PathPart("src", "none")),
        compaction_threshold=0.85,
        total_tokens=40,
        compaction_note="⚠ over threshold; compaction runs on next agent cycle",
        compaction_rows=((1, "01/05 16:45", "summarization", 10000, 4000, 2.5),),
        compaction_summary=(1, 10000.0, 4000.0, 2.5),
    )

    restored = snapshot_from_payload(snapshot_to_payload(snapshot))

    assert restored.instance_name == "pane-a"
    assert restored.cwd_parts == snapshot.cwd_parts
    assert restored.percent_used == 40.0
    assert restored.compaction_note == snapshot.compaction_note
    assert restored.compaction_rows == snapshot.compaction_rows
    assert restored.compaction_summary == snapshot.compaction_summary


def test_bar_scales_past_100k_without_clamping():
    # Items #4/#8: a >100k pre-compaction total must report its real count and
    # fill proportionally; nothing should clamp the headline at 100k.
    snapshot = ContextSnapshot(
        capacity=200000,
        segments=(ContextSegment("User message", 150000, "#c6ff00"),),
        config_lines=(),
        total_tokens=150000,
    )
    assert snapshot.used_tokens == 150000
    assert snapshot.percent_used == 75.0
    assert snapshot.free_tokens == 50000


def test_legend_segments_sorted_by_share_desc():
    snapshot = ContextSnapshot(
        capacity=1000,
        segments=(
            ContextSegment("small", 50, "#111111"),
            ContextSegment("big", 400, "#222222"),
            ContextSegment("big", 200, "#222222"),  # same key, merges to 600
            ContextSegment("mid", 100, "#333333"),
        ),
        config_lines=(),
    )
    legend = snapshot.legend_segments()
    assert [(s.label, s.tokens) for s in legend] == [
        ("big", 600),
        ("mid", 100),
        ("small", 50),
    ]


def test_cwd_marks_available_and_applied_agent_rules(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    child = repo / "child"
    child.mkdir(parents=True)
    (repo / "AGENTS.md").write_text("repo rules", encoding="utf-8")
    (child / "AGENTS.md").write_text("child rules", encoding="utf-8")
    monkeypatch.chdir(child)

    states = {
        part.label: part.agent_rules_state for part in visualizer_snapshot._cwd_parts()
    }

    assert states["repo"] == "available"
    assert states["child"] == "applied"


def test_compaction_log_records_tokens_and_summary_hashes(monkeypatch):
    compaction_log.clear_compaction_log()
    original = FakeMessage([FakePart("UserPromptPart", "old user")])
    summary = FakeMessage([FakePart("TextPart", "summary")])
    monkeypatch.setattr(callbacks, "get_current_usage", lambda: None)
    monkeypatch.setattr(callbacks, "raw_tokens_for_message", lambda *_args: 50)
    monkeypatch.setattr(callbacks, "_configured_strategy", lambda: "summarization")
    monkeypatch.setattr(callbacks, "_capacity", lambda: 100)
    monkeypatch.setattr(callbacks, "_compaction_threshold", lambda: 0.1)

    callbacks._on_history_start("agent", None, [original], [])
    callbacks._on_history_end("agent", None, [summary], 0, 0)

    events = compaction_log.get_compaction_events()
    assert len(events) == 1
    assert events[0].strategy == "summarization"
    assert events[0].tokens_at == 50 and events[0].tokens_to == 50
    assert (
        visualizer_usage.message_hash(summary) in compaction_log.get_summarized_hashes()
    )


def test_compaction_summary_averages_tokens_and_duration():
    compaction_log.clear_compaction_log()
    assert compaction_log.get_compaction_summary() is None
    compaction_log.record_compaction_event("summarization", 10_000, 4_000)
    compaction_log.record_compaction_event("truncation", 8_000, 2_000)

    summary = compaction_log.get_compaction_summary()
    assert summary.count == 2
    assert summary.avg_at == 9_000.0
    assert summary.avg_to == 3_000.0
    # First event has no prior, so only the 2nd contributes a duration; its
    # average is that single gap (>= 0 minutes).
    assert summary.avg_duration_min is not None and summary.avg_duration_min >= 0.0
    # First event's own duration is None (no previous compaction).
    assert compaction_log.get_compaction_events()[-1].duration_min is None


class FakePart:
    def __init__(self, class_name, content, tool_name=None):
        self.content = content
        self.tool_name = tool_name
        self.__class__ = type(class_name, (), {})


class FakeMessage:
    def __init__(self, parts):
        self.parts = parts


class FakeAgent:
    name = "agent"
    session_id = "session"

    def __init__(self, history=None):
        self.history = history

    def get_message_history(self):
        return self.history or [
            FakeMessage([FakePart("UserPromptPart", "old user")]),
            FakeMessage([FakePart("TextPart", "old assistant")]),
            FakeMessage([FakePart("ToolCallPart", "new tool", "read_file")]),
        ]

    def set_message_history(self, history):
        self.history = history

    def estimate_tokens_for_message(self, _message):
        return 1

    def _estimate_context_overhead(self):
        return 10

    def summarize_messages(self, messages, with_protection=True):
        return messages, []


def test_usage_matches_core_multiplier_model_token_math(monkeypatch):
    from code_puppy.agents._history import (
        estimate_context_overhead,
        estimate_tokens_for_message,
    )

    message = FakeMessage([FakePart("UserPromptPart", "x" * 100)])
    model_name = "claude-opus-4-7-test"

    class MultiplierAgent:
        pydantic_agent = None
        name = "agent"
        _mcp_servers = None

        def get_model_name(self):
            return model_name

        def get_message_history(self):
            return [message]

        def get_full_system_prompt(self):
            return "system prompt" * 10

        def get_system_prompt(self):
            return "system prompt" * 10

        def _get_model_context_length(self):
            return 1000

        def _get_tool_probe(self):
            return None

    agent = MultiplierAgent()
    monkeypatch.setattr(
        "code_puppy.agents.agent_manager.get_current_agent", lambda: agent
    )
    monkeypatch.setattr("code_puppy.agents._builder.load_puppy_rules", lambda: "")

    usage = visualizer_usage.get_current_usage()

    expected_used = estimate_tokens_for_message(message, model_name)
    expected_overhead = estimate_context_overhead(
        agent.get_full_system_prompt(), None, model_name, mcp_servers=None
    )
    assert visualizer_usage.raw_tokens_for_message(message, model_name) == expected_used
    assert usage is not None
    assert usage.used_tokens == expected_used
    assert usage.overhead_tokens == expected_overhead
    assert usage.total_tokens == expected_used + expected_overhead


def test_raw_chat_slices_preserve_temporal_order(monkeypatch):
    monkeypatch.setattr(
        "code_puppy.agents._history.stringify_part", lambda part: part.content
    )

    slices = visualizer_usage.chat_slices(FakeAgent())

    assert [(item.label, item.bucket, item.tokens) for item in slices] == [
        ("User message", "chat_user", 3),
        ("Assistant work", "chat_assistant", 5),
        ("Tool calls", "chat_tool", 3),
    ]


def test_user_slice_carries_typed_text_as_detail(monkeypatch):
    monkeypatch.setattr(
        "code_puppy.agents._history.stringify_part", lambda part: part.content
    )
    slices = visualizer_usage.chat_slices(FakeAgent())
    user = next(item for item in slices if item.bucket == "chat_user")
    assert user.detail == "old user"
    # Non-user slices never carry detail (tooltip is user-only).
    assert all(item.detail == "" for item in slices if item.bucket != "chat_user")


def test_subagent_invoke_calls_get_their_own_bucket(monkeypatch):
    monkeypatch.setattr(
        "code_puppy.agents._history.stringify_part", lambda part: part.content
    )
    agent = FakeAgent(
        [
            FakeMessage([FakePart("ToolCallPart", "sub", "invoke_agent")]),
            FakeMessage([FakePart("ToolCallPart", "file", "read_file")]),
        ]
    )
    buckets = {item.bucket for item in visualizer_usage.chat_slices(agent)}
    assert "chat_subagent" in buckets
    assert "chat_tool" in buckets


def test_agents_md_counted_as_its_own_overhead_bucket(monkeypatch):
    message = FakeMessage([FakePart("UserPromptPart", "x" * 100)])
    model_name = "claude-opus-4-7-test"

    class RulesAgent:
        pydantic_agent = None
        name = "agent"
        _mcp_servers = None

        def get_model_name(self):
            return model_name

        def get_message_history(self):
            return [message]

        def get_full_system_prompt(self):
            return "system prompt" * 10

        def get_system_prompt(self):
            return "system prompt" * 10

        def _get_model_context_length(self):
            return 100000

        def _get_tool_probe(self):
            return None

    monkeypatch.setattr(
        "code_puppy.agents.agent_manager.get_current_agent", lambda: RulesAgent()
    )
    monkeypatch.setattr(
        "code_puppy.agents._builder.load_puppy_rules", lambda: "puppy rules " * 50
    )

    usage = visualizer_usage.get_current_usage()
    assert usage is not None
    # AGENTS.md/puppy rules now land in their own bucket instead of vanishing.
    assert usage.agents_md_tokens > 0
    # And they count toward the headline overhead (was under-reported before).
    assert usage.overhead_tokens >= usage.agents_md_tokens


def test_one_overload_yields_one_event_and_one_summary_segment(monkeypatch):
    # Iteration harness: drive the REAL callback cycle for a single
    # over-threshold overload and prove what presents:
    #   * exactly ONE compaction event (table row)
    #   * exactly ONE "Summarized context" bar segment (no slivers)
    compaction_log.clear_compaction_log()

    source = [
        FakeMessage([FakePart("UserPromptPart", f"turn-{i} " * 200)]) for i in range(10)
    ]
    summary = FakeMessage([FakePart("TextPart", "compressed summary")])

    monkeypatch.setattr(callbacks, "get_current_usage", lambda: None)
    monkeypatch.setattr(callbacks, "raw_tokens_for_message", lambda *_a: 1000)
    monkeypatch.setattr(callbacks, "_configured_strategy", lambda: "summarization")
    monkeypatch.setattr(callbacks, "_capacity", lambda: 10000)
    monkeypatch.setattr(callbacks, "_compaction_threshold", lambda: 0.8)
    monkeypatch.setattr(
        "code_puppy.agents._history.stringify_part", lambda part: part.content
    )

    # One overload -> one pre_compact -> one history_end (real sequence).
    callbacks._on_history_start("agent", None, source, [])
    callbacks._on_pre_compact("agent", "summarization", len(source), 10000)
    callbacks._on_history_end("agent", None, [summary], 1, 9)

    events = compaction_log.get_compaction_events()
    agent = FakeAgent([summary])
    slices = visualizer_usage.chat_slices(agent, compaction_log.get_summarized_hashes())
    summary_segments = [s for s in slices if s.bucket == "summary"]

    print(f"DIAG events={len(events)} summary_segments={len(summary_segments)}")
    print(f"DIAG summarized_hashes={len(compaction_log.get_summarized_hashes())}")

    assert len(events) == 1
    assert len(summary_segments) == 1


def test_pre_compact_refresh_renders_over_threshold_peak(monkeypatch):
    # Item #8: the bar must visibly climb past the threshold. The over-threshold
    # peak is live ONLY inside compact() (incoming appended, summary not yet
    # applied), which is exactly when pre_compact fires. So the refresh fired at
    # _on_pre_compact must render a snapshot whose percent_used > threshold.
    from code_puppy.agents._compaction import compact

    monkeypatch.setattr(
        "code_puppy.agents._history.stringify_part", lambda part: part.content
    )

    class Agent:
        name = "agent"
        pydantic_agent = None
        _mcp_servers = None
        session_id = "s"

        def __init__(self):
            self._message_history = [FakeMessage([FakePart("TextPart", "prev")])]
            self._compacted_message_hashes = set()

        def get_message_history(self):
            return self._message_history

        def get_model_name(self):
            return "claude-test"

        def get_full_system_prompt(self):
            return "sys"

        def get_system_prompt(self):
            return "sys"

        def _get_model_context_length(self):
            return 20000

        def _estimate_context_overhead(self):
            return 1000

        def _get_tool_probe(self):
            return None

    agent = Agent()
    monkeypatch.setattr(
        "code_puppy.agents.agent_manager.get_current_agent", lambda: agent
    )

    rendered_pct = []
    monkeypatch.setattr(
        callbacks,
        "notify_dirty",
        lambda: rendered_pct.append(
            visualizer_snapshot.collect_snapshot().percent_used
        ),
    )

    # Real history_processor order: append a huge incoming turn, then compact()
    # (which fires pre_compact internally -> our refresh).
    agent._message_history.append(
        FakeMessage([FakePart("UserPromptPart", "x" * 60000)])
    )
    callbacks._on_pre_compact("agent", "summarization", 2, 60000)
    new_history, _ = compact(agent, agent._message_history, 20000, 1000)
    agent._message_history = new_history

    print(f"DIAG rendered_pct={rendered_pct}")
    assert rendered_pct, "pre_compact must trigger a refresh"
    assert max(rendered_pct) > 80.0, "bar must render the over-threshold peak"


def test_threaded_writer_captures_peak_during_slow_compaction(monkeypatch):
    # Item #8, the part that actually matters: the REAL writer runs on its own
    # thread, blocked on _dirty. This proves that when notify_dirty() fires at
    # pre_compact, the thread wakes and emits an over-threshold frame BEFORE
    # compaction shrinks history -- we genuinely catch the peak, not just in a
    # synchronous call. Compaction here is slow (sleep) like a real LLM call.
    import threading
    import time

    monkeypatch.setattr(
        "code_puppy.agents._history.stringify_part", lambda part: part.content
    )

    class Agent:
        name = "agent"
        pydantic_agent = None
        _mcp_servers = None
        session_id = "s"

        def __init__(self):
            self._message_history = [FakeMessage([FakePart("TextPart", "prev")])]
            self._compacted_message_hashes = set()

        def get_message_history(self):
            return self._message_history

        def get_model_name(self):
            return "claude-test"

        def get_full_system_prompt(self):
            return "sys"

        def get_system_prompt(self):
            return "sys"

        def _get_model_context_length(self):
            return 20000

        def _estimate_context_overhead(self):
            return 1000

        def _get_tool_probe(self):
            return None

    agent = Agent()
    monkeypatch.setattr(
        "code_puppy.agents.agent_manager.get_current_agent", lambda: agent
    )

    frames: list[float] = []
    stop = threading.Event()

    def writer():  # mirrors window._write_snapshots: emit, wait(dirty), clear
        while not stop.is_set():
            frames.append(visualizer_window.collect_snapshot().percent_used)
            visualizer_window._dirty.wait(timeout=2.0)
            visualizer_window._dirty.clear()

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    time.sleep(0.05)  # let it emit the low frame and block on _dirty

    # Slow compaction: peak history is live, fire the refresh, then summarize.
    agent._message_history.append(
        FakeMessage([FakePart("UserPromptPart", "x" * 60000)])
    )
    visualizer_window.notify_dirty()  # what _on_pre_compact does
    time.sleep(0.3)  # LLM-call analogue; history still at peak
    agent._message_history = [FakeMessage([FakePart("TextPart", "summary")])]
    visualizer_window.notify_dirty()
    time.sleep(0.05)
    stop.set()
    visualizer_window.notify_dirty()
    t.join(timeout=1)

    rounded = [round(f, 1) for f in frames]
    print(f"DIAG frames={rounded}")
    assert any(f > 80.0 for f in frames), "writer thread must capture the peak"


def test_truncation_logs_without_summary_hashes(monkeypatch):
    compaction_log.clear_compaction_log()
    first = FakeMessage([FakePart("UserPromptPart", "first")])
    middle = FakeMessage([FakePart("TextPart", "dropped")])
    last = FakeMessage([FakePart("UserPromptPart", "last")])
    monkeypatch.setattr(callbacks, "get_current_usage", lambda: None)
    monkeypatch.setattr(callbacks, "raw_tokens_for_message", lambda *_args: 50)
    monkeypatch.setattr(callbacks, "_configured_strategy", lambda: "truncation")
    monkeypatch.setattr(callbacks, "_capacity", lambda: 100)
    monkeypatch.setattr(callbacks, "_compaction_threshold", lambda: 0.1)

    callbacks._on_history_start("agent", None, [first, middle, last], [])
    callbacks._on_history_end("agent", None, [first, last], 0, 0)

    events = compaction_log.get_compaction_events()
    assert events[0].strategy == "truncation"
    assert events[0].tokens_at == 150 and events[0].tokens_to == 100
    assert compaction_log.get_summarized_hashes() == set()


def test_manual_compact_command_records_and_labels_summary(monkeypatch):
    compaction_log.clear_compaction_log()
    from code_puppy.command_line import session_commands

    original = FakeMessage([FakePart("UserPromptPart", "old user")])
    removed = FakeMessage([FakePart("TextPart", "removed")])
    summary = FakeMessage([FakePart("TextPart", "summary")])
    agent = FakeAgent([original, removed])
    monkeypatch.setattr(
        "code_puppy.agents.agent_manager.get_current_agent", lambda: agent
    )
    monkeypatch.setattr(
        "code_puppy.config.get_compaction_strategy", lambda: "summarization"
    )
    monkeypatch.setattr("code_puppy.config.get_protected_token_count", lambda: 100)
    monkeypatch.setattr(callbacks, "get_current_usage", lambda: None)
    monkeypatch.setattr(callbacks, "raw_tokens_for_message", lambda *_args: 50)
    monkeypatch.setattr(
        agent,
        "summarize_messages",
        lambda _history, with_protection=True: ([original, summary], [removed]),
    )
    monkeypatch.setattr(
        "code_puppy.messaging.emit_info", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "code_puppy.messaging.emit_success", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "code_puppy.messaging.emit_warning", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "code_puppy.messaging.emit_error", lambda *_args, **_kwargs: None
    )

    assert session_commands.handle_compact_command("/compact") is True

    event = compaction_log.get_compaction_events()[0]
    assert event.strategy == "summarization"
    assert event.tokens_at == 100 and event.tokens_to == 100
    assert (
        visualizer_usage.message_hash(summary) in compaction_log.get_summarized_hashes()
    )


def test_manual_truncate_command_records_without_summary(monkeypatch):
    compaction_log.clear_compaction_log()
    from code_puppy.command_line import session_commands

    history = [
        FakeMessage([FakePart("UserPromptPart", "system")]),
        FakeMessage([FakePart("TextPart", "older")]),
        FakeMessage([FakePart("UserPromptPart", "recent")]),
    ]
    agent = FakeAgent(history)
    monkeypatch.setattr(
        "code_puppy.agents.agent_manager.get_current_agent", lambda: agent
    )
    monkeypatch.setattr(callbacks, "get_current_usage", lambda: None)
    monkeypatch.setattr(callbacks, "raw_tokens_for_message", lambda *_args: 50)
    monkeypatch.setattr(
        "code_puppy.messaging.emit_success", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "code_puppy.messaging.emit_info", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "code_puppy.messaging.emit_warning", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "code_puppy.messaging.emit_error", lambda *_args, **_kwargs: None
    )

    assert session_commands.handle_truncate_command("/truncate 2") is True

    event = compaction_log.get_compaction_events()[0]
    assert event.strategy == "truncation"
    assert event.tokens_at == 150 and event.tokens_to == 100
    assert compaction_log.get_summarized_hashes() == set()


def test_summary_hashes_are_labeled_as_summarized_context(monkeypatch):
    monkeypatch.setattr(
        "code_puppy.agents._history.stringify_part", lambda part: part.content
    )
    message = FakeAgent().get_message_history()[1]

    slices = visualizer_usage.chat_slices(
        FakeAgent(), {visualizer_usage.message_hash(message)}
    )

    assert slices[0].label == "Summarized context"
    assert ("Summarized context", "summary", 5) in [
        (item.label, item.bucket, item.tokens) for item in slices
    ]


def test_summarized_messages_collapse_into_single_segment(monkeypatch):
    # Regression: many summarized messages must render as ONE block, not N
    # slivers (per-event detail lives in the compaction table).
    monkeypatch.setattr(
        "code_puppy.agents._history.stringify_part", lambda part: part.content
    )
    history = [FakeMessage([FakePart("UserPromptPart", f"m{i}")]) for i in range(5)]
    agent = FakeAgent(history)
    summarized = {visualizer_usage.message_hash(m) for m in history}

    slices = visualizer_usage.chat_slices(agent, summarized)
    summary = [s for s in slices if s.bucket == "summary"]
    assert len(summary) == 1
    assert summary[0].tokens == sum(
        visualizer_usage.raw_tokens_for_message(m, None) for m in history
    )


def test_window_draw_bar_grows_and_log_renders():
    snapshot = ContextSnapshot(
        capacity=100,
        segments=(ContextSegment("User message", 40, "#00f5ff"),),
        config_lines=("model: test",),
        instance_name="pane-a",
        cwd_parts=(PathPart("repo", "applied"), PathPart("src", "none")),
        compaction_threshold=0.85,
        compaction_rows=((1, "01/05 16:45", "summarization", 10000, 4000, 2.5),),
        compaction_summary=(1, 10000.0, 4000.0, 2.5),
    )
    try:
        window = visualizer_window.ContextVisualizerWindow(
            snapshot_provider=lambda: snapshot
        )
    except Exception as exc:
        import pytest

        pytest.skip(f"tkinter display unavailable: {exc}")

    def bar_height():
        bar = next(
            item
            for item in window.canvas.find_all()
            if window.canvas.type(item) == "rectangle"
            and window.canvas.itemcget(item, "fill") == "#00f5ff"
        )
        coords = window.canvas.coords(bar)
        return coords[3] - coords[1]

    try:
        window.root.geometry("380x420")
        window.root.update_idletasks()
        window._draw(snapshot)
        small_bar_h = bar_height()

        window.root.geometry("380x620")
        window.root.update_idletasks()
        window._draw(snapshot)
        large_bar_h = bar_height()

        items = window.canvas.find_all()
        texts = [
            window.canvas.itemcget(i, "text")
            for i in items
            if window.canvas.type(i) == "text"
        ]
        fills = [
            window.canvas.itemcget(i, "fill")
            for i in items
            if window.canvas.type(i) == "rectangle"
        ]

        # Bar is the main feature: it grows with the window.
        assert large_bar_h > small_bar_h
        # User-message grab tab drew in lime green.
        assert "#c6ff00" in fills
        assert "compact 85%" in texts
        assert "COMPACTION TABLE" in texts
        # Table renders headers, the summary row, and the event row.
        assert "strategy" in texts
        assert any(t.startswith("AVG (1)") for t in texts)
        assert "10,000" in texts and "4,000" in texts
    finally:
        window.close()


def test_snapshot_writer_survives_transient_snapshot_errors(monkeypatch):
    writes = []

    class Stdin:
        def write(self, text):
            writes.append(text)

        def flush(self):
            pass

    class Process:
        stdin = Stdin()

        def poll(self):
            return 0 if len(writes) >= 2 else None

    snapshots = [RuntimeError("reload gap"), ContextSnapshot(12, (), ("ok",))]

    def fake_collect_snapshot():
        result = snapshots.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(visualizer_window, "collect_snapshot", fake_collect_snapshot)
    monkeypatch.setattr(visualizer_window.time, "sleep", lambda _seconds: None)
    # Fake event so the writer never blocks on the heartbeat wait in tests.
    monkeypatch.setattr(
        visualizer_window,
        "_dirty",
        SimpleNamespace(wait=lambda timeout=None: None, clear=lambda: None),
    )

    visualizer_window._write_snapshots(Process())

    assert "Snapshot temporarily unavailable: reload gap" in writes[0]
    assert '"capacity": 12' in writes[1]


class FakeProcess:
    def __init__(self):
        self.stdin = SimpleNamespace(write=lambda _text: None, flush=lambda: None)
        self.stderr = SimpleNamespace(read=lambda: "")
        self.returncode = None

    def poll(self):
        return self.returncode


def test_open_visualizer_uses_child_process(monkeypatch):
    visualizer_window.close_visualizer()
    calls = []
    process = FakeProcess()

    class FakeThread:
        def __init__(self, *args, **kwargs):
            calls.append(("thread", kwargs))

        def start(self):
            calls.append(("start",))

    def fake_popen(args, **kwargs):
        calls.append(("popen", args, kwargs))
        return process

    monkeypatch.setattr(visualizer_window.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(visualizer_window.threading, "Thread", FakeThread)
    monkeypatch.setattr(visualizer_window.time, "sleep", lambda _seconds: None)

    assert visualizer_window.open_visualizer() == (True, "Context visualizer opening.")
    popen_call = next(call for call in calls if call[0] == "popen")
    assert popen_call[1] == [
        visualizer_window.sys.executable,
        "-m",
        "code_puppy.plugins.context_window_popout.window_child",
    ]


def test_writer_death_records_last_error_for_status(monkeypatch):
    # A broken pipe must surface via last_error() (so /context-visual status
    # shows it) instead of dying silently and freezing the window.
    class DeadStdin:
        def write(self, _text):
            raise BrokenPipeError("pipe gone")

        def flush(self):
            pass

    class Process:
        stdin = DeadStdin()

        def poll(self):
            return None

    monkeypatch.setattr(
        visualizer_window, "collect_snapshot", lambda: ContextSnapshot(1, (), ())
    )
    monkeypatch.setattr(
        visualizer_window,
        "_dirty",
        SimpleNamespace(wait=lambda timeout=None: None, clear=lambda: None),
    )
    visualizer_window._last_error = None

    visualizer_window._write_snapshots(Process())

    assert visualizer_window._last_error is not None
    assert "BrokenPipeError" in visualizer_window._last_error


def test_ensure_visualizer_open_counts_respawns(monkeypatch):
    monkeypatch.setattr(visualizer_window, "_retry_count", 0)
    monkeypatch.setattr(visualizer_window, "was_visualizer_requested", lambda: True)
    monkeypatch.setattr(visualizer_window, "is_visualizer_open", lambda: False)
    monkeypatch.setattr(visualizer_window, "open_visualizer", lambda: (True, ""))

    visualizer_window.ensure_visualizer_open()
    visualizer_window.ensure_visualizer_open()

    assert visualizer_window.retry_count() == 2


def test_status_command_reports_respawns_and_error(monkeypatch):
    messages = []
    monkeypatch.setattr(callbacks, "_emit_info", messages.append)
    monkeypatch.setattr(callbacks, "is_visualizer_open", lambda: False)
    monkeypatch.setattr(callbacks, "last_error", lambda: "pipe gone")
    monkeypatch.setattr(callbacks, "retry_count", lambda: 3)

    callbacks._handle_custom_command("/context-visual status", "context-visual")

    assert messages == [
        "Context visualizer is closed. Respawns: 3. Last error: pipe gone"
    ]


def test_staleness_footer_draws_only_when_feed_is_stale():
    snapshot = ContextSnapshot(10, (), (), "waiting")
    stale = {"s": 0.0}
    try:
        window = visualizer_window.ContextVisualizerWindow(
            snapshot_provider=lambda: snapshot,
            staleness_provider=lambda: stale["s"],
        )
    except Exception as exc:
        import pytest

        pytest.skip(f"tkinter display unavailable: {exc}")
    try:
        window.root.update_idletasks()

        window._draw(snapshot)
        fresh_texts = [
            window.canvas.itemcget(i, "text")
            for i in window.canvas.find_all()
            if window.canvas.type(i) == "text"
        ]
        assert all("feed may be stalled" not in t for t in fresh_texts)

        stale["s"] = visualizer_window.STALE_AFTER_S + 5
        window._draw(snapshot)
        stale_texts = [
            window.canvas.itemcget(i, "text")
            for i in window.canvas.find_all()
            if window.canvas.type(i) == "text"
        ]
        assert any("feed may be stalled" in t for t in stale_texts)
    finally:
        window.close()
