"""Child process entrypoint for the Tk context visualizer."""

from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any

from .snapshot import ContextSnapshot, snapshot_from_payload
from .window import ContextVisualizerWindow

_latest_payload: dict[str, Any] | None = None
_window: ContextVisualizerWindow | None = None
_lock = threading.Lock()
_last_payload_at = time.monotonic()


def _read_stdin() -> None:
    global _latest_payload, _last_payload_at
    for line in sys.stdin:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        with _lock:
            _latest_payload = payload
            _last_payload_at = time.monotonic()
        if _window:  # repaint immediately on arrival (event-driven)
            _window.request_redraw()
    if _window:
        _window.root.after(0, _window.close)


def _staleness() -> float:
    with _lock:
        return time.monotonic() - _last_payload_at


def _snapshot_provider() -> ContextSnapshot:
    with _lock:
        payload = dict(_latest_payload or {})
    if not payload:
        return ContextSnapshot(
            capacity=0,
            segments=(),
            config_lines=(),
            status="Waiting for context data...",
        )
    return snapshot_from_payload(payload)


def main() -> None:
    global _window
    _window = ContextVisualizerWindow(
        snapshot_provider=_snapshot_provider,
        staleness_provider=_staleness,
    )
    reader = threading.Thread(target=_read_stdin, daemon=True)
    reader.start()
    _window.run()


if __name__ == "__main__":
    main()
