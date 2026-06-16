"""Parent-side process control and Tk drawing for the context visualizer."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from .snapshot import ContextSnapshot, PathPart, collect_snapshot, snapshot_to_payload

# Event-driven cadence: the writer blocks on _dirty and is woken by notify_dirty()
# from agent/history/compaction callbacks. HEARTBEAT_S is a self-healing backstop
# so a missed notify (or new code path that forgets to call it) can't freeze the
# window forever -- it still refreshes at least this often while idle.
HEARTBEAT_S = 2.0
# Child draws a glanceable staleness footer once updates stop arriving this long.
STALE_AFTER_S = 6.0
_BG = "#05010a"
_FG = "#f8f7ff"
_MUTED = "#8bd3dd"
_GRID = "#273043"
_NEON = "#00f5ff"
_PINK = "#ff2a6d"
_APPLIED_GREEN = "#9cffb1"
_AVAILABLE_AMBER = "#ffcc66"
_THRESHOLD_RED = "#ff3355"
_USER_MESSAGE_GREEN = "#c6ff00"

_process: subprocess.Popen[str] | None = None
_writer_thread: threading.Thread | None = None
_lock = threading.Lock()
_last_error: str | None = None
_retry_count = 0
_dirty = threading.Event()


def notify_dirty() -> None:
    """Wake the snapshot writer to push a frame now (event-driven refresh)."""
    _dirty.set()


def _is_user_segment(label: str) -> bool:
    return label.lower() == "user message"


def _path_color(part: PathPart, piece: str) -> str:
    if piece == "/":
        return _FG
    if part.applies_to_prompt:
        return _APPLIED_GREEN
    if part.has_agent_rules:
        return _AVAILABLE_AMBER
    return _FG


class ContextVisualizerWindow:
    def __init__(
        self,
        snapshot_provider: Callable[[], ContextSnapshot] = collect_snapshot,
        staleness_provider: Callable[[], float] | None = None,
    ):
        import tkinter as tk

        self.snapshot_provider = snapshot_provider
        # Returns seconds since the last payload arrived (child wiring). When it
        # exceeds STALE_AFTER_S we draw a glanceable red footer so a frozen feed
        # (dead writer / hung parent) is obvious without typing a status command.
        self.staleness_provider = staleness_provider
        self.root = tk.Tk()
        self.root.title("Code Puppy Context Visualizer")
        self.root.geometry("560x760")
        self.root.minsize(360, 360)
        self.root.configure(bg=_BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._raise_to_front()
        self.canvas = tk.Canvas(self.root, bg=_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind(
            "<Configure>", lambda _event: self._draw(self._latest_snapshot)
        )
        # User-tick hover tooltips: (x0, y0, x1, y1, text) hitboxes per draw.
        self._user_hitboxes: list[tuple[int, int, int, int, str]] = []
        self._tooltip_id: int | None = None
        self._hover_xy: tuple[int, int] | None = None
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda _e: self._clear_hover())
        self.canvas.bind("<Button-1>", self._on_click)
        self.closed = False
        self._latest_snapshot = ContextSnapshot(
            0, (), (), "Waiting for context data..."
        )

    def close(self) -> None:
        self.closed = True
        try:
            self.root.destroy()
        except Exception:
            pass

    def _raise_to_front(self) -> None:
        # Pop above other windows on open WITHOUT staying permanently pinned:
        # flip -topmost on, grab focus, then back off. lift() alone is unreliable
        # on macOS where new Tk windows tend to open behind the terminal. NOTE:
        # OS shortcuts like Super+Tab group windows by app; Tk can't force
        # multiple popouts into one cycle, so we just guarantee each rises once.
        try:
            self.root.lift()
            self.root.attributes("-topmost", True)
            try:
                self.root.focus_force()
            except Exception:
                pass
            self.root.after(300, lambda: self.root.attributes("-topmost", False))
        except Exception:
            pass

    def _on_click(self, event) -> None:
        for x0, y0, x1, y1, text in self._user_hitboxes:
            if x0 <= event.x <= x1 and y0 <= event.y <= y1 and text:
                self._open_detail_window(text)
                return

    def _open_detail_window(self, text: str) -> None:
        # Item #2: full (often very long) prompt in a pinned, scrollable window.
        import tkinter as tk

        top = tk.Toplevel(self.root)
        top.title("User prompt")
        top.configure(bg=_BG)
        top.geometry("520x420")
        widget = tk.Text(
            top,
            bg=_BG,
            fg=_FG,
            font=("Courier", 10),
            wrap="word",
            padx=10,
            pady=8,
            highlightthickness=0,
        )
        scroll = tk.Scrollbar(top, command=widget.yview)
        widget.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        widget.insert("1.0", text)
        widget.configure(state="disabled")
        top.lift()
        top.attributes("-topmost", True)
        top.after(300, lambda: top.attributes("-topmost", False))

    def _on_motion(self, event) -> None:
        self._hover_xy = (event.x, event.y)
        self._render_tooltip()

    def _clear_hover(self) -> None:
        self._hover_xy = None
        self._hide_tooltip()

    def _render_tooltip(self) -> None:
        # Re-evaluated on motion AND after every redraw so the tip stays put
        # while the mouse rests on a tab (the periodic redraw wipes the canvas).
        self._hide_tooltip()
        if self._hover_xy is None:
            return
        x, y = self._hover_xy
        for x0, y0, x1, y1, text in self._user_hitboxes:
            if x0 <= x <= x1 and y0 <= y <= y1 and text:
                # Hover shows a capped preview; click opens the full scrollable
                # window so a giant prompt can't blanket the canvas.
                preview = (
                    text if len(text) <= 280 else text[:280] + "…  (click to expand)"
                )
                self._show_tooltip(x, y, preview)
                return

    def _show_tooltip(self, x: int, y: int, text: str) -> None:
        self._hide_tooltip()
        width = max(320, self.canvas.winfo_width())
        height = max(320, self.canvas.winfo_height())
        tx = min(x + 14, width - 244)
        ty = y + 8
        # Draw text first, then size the background box to its bounding box.
        fg = self.canvas.create_text(
            tx,
            ty,
            anchor="nw",
            fill=_FG,
            font=("Courier", 9),
            text=text,
            width=224,
        )
        bbox = self.canvas.bbox(fg)
        if bbox and bbox[3] > height - 6:  # would overflow bottom; flip upward
            self.canvas.delete(fg)
            ty = max(6, y - (bbox[3] - bbox[1]) - 12)
            fg = self.canvas.create_text(
                tx,
                ty,
                anchor="nw",
                fill=_FG,
                font=("Courier", 9),
                text=text,
                width=224,
            )
            bbox = self.canvas.bbox(fg)
        bg = self.canvas.create_rectangle(
            bbox[0] - 6,
            bbox[1] - 6,
            bbox[2] + 6,
            bbox[3] + 6,
            fill="#10151f",
            outline=_NEON,
        )
        self.canvas.tag_raise(fg)
        self._tooltip_id = bg
        self._tooltip_text_id = fg

    def _hide_tooltip(self) -> None:
        if self._tooltip_id is not None:
            self.canvas.delete(self._tooltip_id)
            self.canvas.delete(getattr(self, "_tooltip_text_id", None))
            self._tooltip_id = None

    def run(self) -> None:
        self._redraw()
        self.root.mainloop()

    def request_redraw(self) -> None:
        """Thread-safe nudge to repaint now (called when a payload arrives)."""
        if not self.closed:
            self.root.after(0, self._redraw_once)

    def _redraw_once(self) -> None:
        if self.closed:
            return
        try:
            self._latest_snapshot = self.snapshot_provider()
            self._draw(self._latest_snapshot)
        except Exception as exc:
            self._draw_error(str(exc))

    def _redraw(self) -> None:
        # Slow tick: event-driven payloads drive content via request_redraw();
        # this loop only keeps the staleness footer's clock current while idle.
        self._redraw_once()
        if not self.closed:
            self.root.after(1000, self._redraw)

    def _draw_error(self, message: str) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(
            20,
            24,
            anchor="nw",
            fill=_PINK,
            font=("Courier", 14, "bold"),
            text=f"Context visualizer error:\n{message}",
            width=max(280, self.canvas.winfo_width() - 40),
        )

    def _draw(self, snapshot: ContextSnapshot) -> None:
        if self.closed:
            return
        self.canvas.delete("all")
        self._tooltip_id = None
        self._user_hitboxes = []
        width = max(320, self.canvas.winfo_width())
        height = max(320, self.canvas.winfo_height())
        y = self._header(snapshot, width)
        y = self._cwd(snapshot.cwd_parts, width, y)
        if snapshot.status:
            self.canvas.create_text(
                22,
                y + 12,
                anchor="nw",
                fill=_MUTED,
                font=("Courier", 12),
                text=snapshot.status,
                width=width - 44,
            )
            self._staleness_footer(width, height)
            return
        chart_top = y + 16
        # Bar is the main feature: it grows with the window. The log gets a fixed
        # bottom strip (a few lines + scrollbar).
        log_height = 150
        chart_bottom = max(chart_top + 180, height - log_height)
        self._stacked_bar(snapshot, chart_top, chart_bottom)
        self._legend(snapshot, width, chart_top, chart_bottom)
        self._compaction_log(snapshot, width, chart_bottom + 30)
        self._render_tooltip()  # keep a resting tooltip alive across redraws
        self._staleness_footer(width, height)

    def _staleness_footer(self, width: int, height: int) -> None:
        # Glanceable red strip when the feed goes quiet (dead writer / hung
        # parent). All child-local: it watches wall-clock since last payload,
        # so it fires even if the parent stops sending entirely.
        if self.staleness_provider is None:
            return
        try:
            stale_s = self.staleness_provider()
        except Exception:
            return
        if stale_s < STALE_AFTER_S:
            return
        self.canvas.create_rectangle(
            0, height - 22, width, height, fill="#2a0510", outline=""
        )
        self.canvas.create_text(
            width // 2,
            height - 11,
            fill=_THRESHOLD_RED,
            font=("Courier", 9, "bold"),
            text=f"⚠ no updates for {stale_s:.0f}s — feed may be stalled",
        )

    def _header(self, snapshot: ContextSnapshot, width: int) -> int:
        if snapshot.instance_name:
            header = f"instance: {snapshot.instance_name}"
            if snapshot.agent_name:
                header += f"   agent: {snapshot.agent_name}"
            self.canvas.create_text(
                22,
                4,
                anchor="nw",
                fill=_MUTED,
                font=("Courier", 10, "bold"),
                text=header,
            )
        self.canvas.create_text(
            22,
            22,
            anchor="nw",
            fill=_NEON,
            font=("Courier", 18, "bold"),
            text="CONTEXT VISUALIZER",
        )
        percent_x = width - 22
        self.canvas.create_text(
            percent_x,
            24,
            anchor="ne",
            fill=_PINK,
            font=("Courier", 18, "bold"),
            text=f"{snapshot.percent_used:.1f}%",
        )

        config_x = max(220, width - 360)
        config_width = max(120, percent_x - config_x - 70)
        config_block = self.canvas.create_text(
            config_x,
            22,
            anchor="nw",
            fill=_MUTED,
            font=("Courier", 9),
            text="\n".join(snapshot.config_lines),
            width=config_width,
        )

        total = f"{snapshot.used_tokens:,} / {snapshot.capacity:,} tokens"
        total_item = self.canvas.create_text(
            22,
            54,
            anchor="nw",
            fill=_FG,
            font=("Courier", 11),
            text=total,
        )
        note_bottom = 0
        if snapshot.compaction_note:
            note_item = self.canvas.create_text(
                22,
                72,
                anchor="nw",
                fill=_THRESHOLD_RED,
                font=("Courier", 10, "bold"),
                text=snapshot.compaction_note,
                width=max(240, config_x - 34),
            )
            note_bottom = (
                self.canvas.bbox(note_item)[3] if self.canvas.bbox(note_item) else 0
            )
        total_bottom = (
            self.canvas.bbox(total_item)[3] if self.canvas.bbox(total_item) else 82
        )
        config_bottom = (
            self.canvas.bbox(config_block)[3] if self.canvas.bbox(config_block) else 82
        )
        line_y = max(82, total_bottom + 10, note_bottom + 10, config_bottom + 10)
        self.canvas.create_line(22, line_y, width - 22, line_y, fill=_GRID)
        return line_y + 14

    def _cwd(self, parts: tuple[PathPart, ...], width: int, y: int) -> int:
        self.canvas.create_text(
            22,
            y,
            anchor="nw",
            fill=_MUTED,
            font=("Courier", 10, "bold"),
            text="CWD",
        )
        x = 58
        line_y = y
        max_x = width - 24
        char_w = 7
        for index, part in enumerate(parts):
            pieces = [part.label]
            if index < len(parts) - 1 and part.label not in {"/", "\\"}:
                pieces.append("/")
            for piece in pieces:
                piece_w = max(8, len(piece) * char_w)
                if x + piece_w > max_x and x > 58:
                    x = 58
                    line_y += 18
                fill = _path_color(part, piece)
                self.canvas.create_text(
                    x,
                    line_y,
                    anchor="nw",
                    fill=fill,
                    font=("Courier", 10, "bold" if part.has_agent_rules else "normal"),
                    text=piece,
                )
                x += piece_w
        line_y += 24
        self.canvas.create_text(
            58,
            line_y,
            anchor="nw",
            fill=_MUTED,
            font=("Courier", 9),
            text="green = applied AGENT(S).md; amber = present but not applied",
        )
        self.canvas.create_line(22, line_y + 22, width - 22, line_y + 22, fill=_GRID)
        return line_y + 36

    def _stacked_bar(self, snapshot: ContextSnapshot, top: int, bottom: int) -> None:
        bar_x0, bar_x1 = 150, 228
        bar_y0, bar_y1 = top + 18, max(top + 80, bottom)
        self.canvas.create_rectangle(
            bar_x0,
            bar_y0,
            bar_x1,
            bar_y1,
            outline=_NEON,
            width=2,
        )
        segments = snapshot.segments_with_free()
        capacity = max(1, snapshot.capacity)
        height = max(1, bar_y1 - bar_y0)
        cumulative = 0
        user_marks: list[tuple[int, str]] = []  # (center_y, detail)
        for index, segment in enumerate(segments):
            next_cumulative = min(capacity, cumulative + max(0, segment.tokens))
            y0 = bar_y0 + round((cumulative / capacity) * height)
            y1 = bar_y0 + round((next_cumulative / capacity) * height)
            if index == len(segments) - 1:
                y1 = bar_y1
            if y1 <= y0 and segment.tokens > 0:
                y1 = min(bar_y1, y0 + 1)
            if y1 > y0:
                self.canvas.create_rectangle(
                    bar_x0 + 2,
                    y0,
                    bar_x1 - 2,
                    y1,
                    fill=segment.color,
                    outline="",
                )
                if _is_user_segment(segment.label):
                    self.canvas.create_rectangle(
                        bar_x0 - 8,
                        y0,
                        bar_x0 + 4,
                        y1,
                        fill=_USER_MESSAGE_GREEN,
                        outline="",
                    )
                    user_marks.append(((y0 + y1) // 2, segment.detail))
            cumulative = next_cumulative
        self._draw_user_tabs(user_marks, bar_x0, bar_y0, bar_y1)
        threshold = snapshot.compaction_threshold
        if 0 < threshold < 1:
            threshold_y = bar_y0 + int(threshold * (bar_y1 - bar_y0))
            self.canvas.create_line(
                bar_x0 - 8,
                threshold_y,
                bar_x1 + 8,
                threshold_y,
                fill=_THRESHOLD_RED,
                width=2,
            )
            # Label sits in the left margin so it can't collide with the legend.
            self.canvas.create_text(
                bar_x0 - 28,
                threshold_y,
                anchor="e",
                fill=_THRESHOLD_RED,
                font=("Courier", 9, "bold"),
                text=f"compact {threshold:.0%}",
            )
        self.canvas.create_text(
            (bar_x0 + bar_x1) // 2,
            bar_y0 - 14,
            fill=_MUTED,
            font=("Courier", 9),
            text="0%",
        )
        self.canvas.create_text(
            (bar_x0 + bar_x1) // 2,
            bar_y1 + 14,
            fill=_MUTED,
            font=("Courier", 9),
            text="100%",
        )

    def _draw_user_tabs(
        self,
        marks: list[tuple[int, str]],
        bar_x0: int,
        bar_y0: int,
        bar_y1: int,
    ) -> None:
        # Fixed-size grab tabs protruding left so even a 1px slice (short prompt)
        # is hoverable. Overlapping tabs merge into one taller tab whose tooltip
        # joins each prompt.
        tab_h = 14
        clusters: list[list[tuple[int, str]]] = []
        for mark in sorted(marks):
            if clusters and mark[0] - clusters[-1][-1][0] < tab_h:
                clusters[-1].append(mark)
            else:
                clusters.append([mark])
        for cluster in clusters:
            centers = [c[0] for c in cluster]
            details = [c[1] for c in cluster if c[1]]
            top = max(bar_y0, min(centers) - tab_h // 2)
            bot = min(bar_y1, max(centers) + tab_h // 2)
            if bot - top < tab_h:
                bot = min(bar_y1, top + tab_h)
            self.canvas.create_rectangle(
                bar_x0 - 22,
                top,
                bar_x0 - 8,
                bot,
                fill=_USER_MESSAGE_GREEN,
                outline=_BG,
            )
            if len(cluster) > 1:
                self.canvas.create_text(
                    bar_x0 - 15,
                    (top + bot) // 2,
                    fill=_BG,
                    font=("Courier", 8, "bold"),
                    text=str(len(cluster)),
                )
            tip = "\n---\n".join(details)
            self._user_hitboxes.append((bar_x0 - 24, top - 2, bar_x0 + 6, bot + 2, tip))

    def _legend(
        self, snapshot: ContextSnapshot, width: int, top: int, bottom: int
    ) -> None:
        x, y = 255, top + 18
        max_y = bottom - 24
        for segment in snapshot.legend_segments():
            if y > max_y:
                self.canvas.create_text(
                    x,
                    y,
                    anchor="nw",
                    fill=_MUTED,
                    font=("Courier", 10),
                    text="...",
                )
                return
            pct = segment.percent_of(snapshot.capacity)
            self.canvas.create_rectangle(
                x,
                y + 3,
                x + 14,
                y + 17,
                fill=segment.color,
                outline="",
            )
            label = f"{segment.label}: {segment.tokens:,} ({pct:.1f}%)"
            self.canvas.create_text(
                x + 22,
                y,
                anchor="nw",
                fill=_FG,
                font=("Courier", 10),
                text=label,
                width=width - x - 30,
            )
            y += 28
        free = snapshot.free_tokens
        free_pct = snapshot.segments_with_free()[-1].percent_of(snapshot.capacity)
        self.canvas.create_rectangle(
            x,
            y + 3,
            x + 14,
            y + 17,
            fill="#1b263b",
            outline="",
        )
        self.canvas.create_text(
            x + 22,
            y,
            anchor="nw",
            fill=_MUTED,
            font=("Courier", 10),
            text=f"Unused capacity: {free:,} ({free_pct:.1f}%)",
            width=width - x - 30,
        )

    # Column right-edge fractions of usable width. Text cols (#, date, strategy)
    # left-anchor; number cols (before/after/dur) right-anchor at their edge, so
    # the table spreads/contracts with the window like the top section.
    _COL_FRACS = (0.06, 0.30, 0.58, 0.76, 0.88, 1.00)
    _COL_HEADERS = ("#", "date-time", "strategy", "before", "after", "dur(m)")

    def _compaction_log(self, snapshot: ContextSnapshot, width: int, top: int) -> None:
        self.canvas.create_line(22, top, width - 22, top, fill=_GRID)
        self.canvas.create_text(
            22,
            top + 8,
            anchor="nw",
            fill=_PINK,
            font=("Courier", 10, "bold"),
            text="COMPACTION TABLE",
        )
        self._compaction_table(snapshot, width, top + 28)

    def _col_edge(self, width: int, frac: float) -> int:
        return 22 + int((width - 44) * frac)

    def _draw_table_row(self, width, y, cells, fill, font) -> None:
        # cells: 6 strings. Cols 0-2 left-anchored at the prior edge; cols 3-5
        # right-anchored at their own edge for clean number alignment.
        for i, text in enumerate(cells):
            if i < 3:
                x = 22 if i == 0 else self._col_edge(width, self._COL_FRACS[i - 1])
                self.canvas.create_text(
                    x, y, anchor="nw", fill=fill, font=font, text=text
                )
            else:
                x = self._col_edge(width, self._COL_FRACS[i]) - 4
                self.canvas.create_text(
                    x, y, anchor="ne", fill=fill, font=font, text=text
                )

    def _compaction_table(
        self, snapshot: ContextSnapshot, width: int, top: int
    ) -> None:
        line_h = 18
        header_font = ("Courier", 9, "bold")
        cell_font = ("Courier", 9)
        self._draw_table_row(width, top, self._COL_HEADERS, _MUTED, header_font)
        y = top + line_h + 2
        self.canvas.create_line(22, y - 1, width - 22, y - 1, fill=_GRID)

        summary = snapshot.compaction_summary
        if summary is not None:
            count, avg_at, avg_to, avg_dur = summary
            dur = "-" if avg_dur is None else f"{avg_dur:.1f}"
            self._draw_table_row(
                width,
                y,
                (
                    "",
                    "",
                    f"AVG ({count})",
                    f"{round(avg_at):,}",
                    f"{round(avg_to):,}",
                    dur,
                ),
                _NEON,
                header_font,
            )
        y += line_h + 2

        rows = snapshot.compaction_rows[:4]  # newest first, last 4
        if not rows:
            self.canvas.create_text(
                22, y, anchor="nw", fill=_MUTED, font=cell_font, text="(none yet)"
            )
            return
        for index, dt, strategy, tok_at, tok_to, duration in rows:
            dur = "-" if duration is None else f"{duration:.1f}"
            self._draw_table_row(
                width,
                y,
                (str(index), dt, strategy, f"{tok_at:,}", f"{tok_to:,}", dur),
                _FG,
                cell_font,
            )
            y += line_h


def open_visualizer() -> tuple[bool, str]:
    global _last_error, _process, _writer_thread
    with _lock:
        if _process and _process.poll() is None:
            _remember_open_requested(True)
            return True, "Context visualizer is already open."
        _last_error = None
        _dirty.set()  # ensure the new writer pushes a fresh frame promptly
        try:
            _process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "code_puppy.plugins.context_window_popout.window_child",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:
            _last_error = str(exc)
            return False, f"Context visualizer failed to open: {_last_error}"
        _writer_thread = threading.Thread(
            target=_write_snapshots,
            args=(_process,),
            name="context-visualizer-snapshot-writer",
            daemon=True,
        )
        _writer_thread.start()

    time.sleep(0.3)
    if _process.poll() is not None:
        _last_error = (_process.stderr.read() if _process.stderr else "").strip()
        return (
            False,
            f"Context visualizer failed to open: {_last_error or 'unknown error'}",
        )
    _remember_open_requested(True)
    return True, "Context visualizer opening."


def close_visualizer() -> tuple[bool, str]:
    global _process
    _remember_open_requested(False)
    with _lock:
        process = _process
        _process = None
    if process is None or process.poll() is not None:
        return True, "Context visualizer is not open."
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
    return True, "Context visualizer closed."


def ensure_visualizer_open() -> None:
    global _retry_count
    if was_visualizer_requested() and not is_visualizer_open():
        # The window was wanted but the process/writer died -- respawn it and
        # count the retry so /context-visual status can report the churn.
        _retry_count += 1
        open_visualizer()


def is_visualizer_open() -> bool:
    return bool(_process and _process.poll() is None)


def last_error() -> str | None:
    return _last_error


def retry_count() -> int:
    return _retry_count


def was_visualizer_requested() -> bool:
    try:
        return _request_file().exists()
    except Exception:
        return False


def _request_file() -> Path:
    # Per-PID so concurrent Code Puppy instances don't clobber each other's
    # "window wanted" flag (a shared file made one instance's close hide the
    # others' respawn intent).
    name = f"context_visualizer.open.{os.getpid()}"
    try:
        from code_puppy.config import CONFIG_DIR

        return Path(CONFIG_DIR) / name
    except Exception:
        return Path.home() / ".code_puppy" / name


def _remember_open_requested(requested: bool) -> None:
    try:
        path = _request_file()
        if requested:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(os.getpid()), encoding="utf-8")
        elif path.exists():
            path.unlink()
    except Exception:
        pass


def _error_snapshot(message: str) -> ContextSnapshot:
    return ContextSnapshot(
        capacity=0,
        segments=(),
        config_lines=(),
        status=f"Snapshot temporarily unavailable: {message}",
    )


def _payload_or_error() -> dict:
    try:
        snapshot = collect_snapshot()
    except Exception as exc:
        snapshot = _error_snapshot(str(exc))
    return snapshot_to_payload(snapshot)


def _write_snapshots(process: subprocess.Popen[str]) -> None:
    global _last_error
    # Push the first frame immediately, then block on _dirty (woken by
    # notify_dirty) with a HEARTBEAT_S timeout backstop. Any pipe death is
    # recorded to _last_error so /context-visual status surfaces it and
    # ensure_visualizer_open() can respawn on the next agent activity.
    while process.poll() is None and process.stdin:
        try:
            process.stdin.write(json.dumps(_payload_or_error()) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            _last_error = f"snapshot writer stopped: {exc!r}"
            return
        _dirty.wait(timeout=HEARTBEAT_S)
        _dirty.clear()
