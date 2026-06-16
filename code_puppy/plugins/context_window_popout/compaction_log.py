from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from threading import Lock

_MAX_ENTRIES = 8
_lock = Lock()
_counter = 0
_events: deque["CompactionEvent"] = deque(maxlen=_MAX_ENTRIES)
_summary_hashes: set[int] = set()
# Running, session-wide accumulators so averages survive deque eviction.
_sum_at = 0
_sum_to = 0
_sum_duration = 0.0
_duration_count = 0
_last_dt: datetime | None = None


@dataclass(frozen=True)
class CompactionEvent:
    index: int
    strategy: str
    tokens_at: int
    tokens_to: int
    timestamp: str
    duration_min: float | None = None


@dataclass(frozen=True)
class CompactionSummary:
    count: int
    avg_at: float
    avg_to: float
    avg_duration_min: float | None


def record_compaction_event(strategy: str, tokens_at: int, tokens_to: int) -> None:
    global _counter, _sum_at, _sum_to, _sum_duration, _duration_count, _last_dt
    now = datetime.now()
    with _lock:
        _counter += 1
        duration_min = (now - _last_dt).total_seconds() / 60 if _last_dt else None
        _last_dt = now
        _sum_at += tokens_at
        _sum_to += tokens_to
        if duration_min is not None:
            _sum_duration += duration_min
            _duration_count += 1
        _events.appendleft(
            CompactionEvent(
                _counter,
                strategy,
                tokens_at,
                tokens_to,
                now.strftime("%m/%d/%Y %H:%M:%S"),
                duration_min,
            )
        )


def mark_summarized_hashes(hashes: set[int]) -> None:
    with _lock:
        _summary_hashes.update(hashes)


def prune_summarized_hashes(live_hashes: set[int]) -> None:
    with _lock:
        _summary_hashes.intersection_update(live_hashes)


def get_summarized_hashes() -> set[int]:
    with _lock:
        return set(_summary_hashes)


def get_compaction_events() -> tuple[CompactionEvent, ...]:
    with _lock:
        return tuple(_events)


def get_compaction_summary() -> CompactionSummary | None:
    with _lock:
        if _counter == 0:
            return None
        avg_dur = _sum_duration / _duration_count if _duration_count else None
        return CompactionSummary(
            count=_counter,
            avg_at=_sum_at / _counter,
            avg_to=_sum_to / _counter,
            avg_duration_min=avg_dur,
        )


def clear_compaction_log() -> None:
    global _counter, _sum_at, _sum_to, _sum_duration, _duration_count, _last_dt
    with _lock:
        _events.clear()
        _summary_hashes.clear()
        _counter = 0
        _sum_at = 0
        _sum_to = 0
        _sum_duration = 0.0
        _duration_count = 0
        _last_dt = None
