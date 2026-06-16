from __future__ import annotations

import re

_NAME: str | None = None
_VALID = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")


def get_name() -> str:
    global _NAME
    if _NAME is None:
        try:
            from code_puppy.config import get_current_autosave_session_name

            _NAME = get_current_autosave_session_name()
        except Exception:
            _NAME = "context-visual"
    return _NAME


def set_name(value: str | None) -> str:
    global _NAME
    value = (value or "").strip()
    if value and _VALID.match(value):
        _NAME = value
    return get_name()
