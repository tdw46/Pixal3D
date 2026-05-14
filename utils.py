from __future__ import annotations

import textwrap
from pathlib import Path


def wrap_text_to_panel(text: str, context, *, min_chars: int = 8, full_width: bool = False) -> str:
    try:
        width = getattr(context.region, "width", 300) or 300
        prefs = getattr(context, "preferences", None)
        view = getattr(prefs, "view", None) if prefs else None
        scale = getattr(view, "ui_scale", 1.0) if view else 1.0
        reserved = 240 if not full_width else 60
        available = max(50, width - reserved)
        px_per_char = (13.5 if not full_width else 9.5) * max(scale, 0.5)
        max_chars = max(min_chars, int(available / px_per_char))
    except Exception:
        max_chars = min_chars
    max_cap = 75 if not full_width else 180
    max_chars = max(min_chars, min(max_cap, max_chars))
    return textwrap.fill(
        text or "",
        width=max_chars,
        break_long_words=False,
        replace_whitespace=False,
        expand_tabs=False,
    )


def unique_output_path(path: str) -> str:
    target = Path(path)
    if not target.exists():
        return str(target)
    stem = target.stem
    suffix = target.suffix or ".glb"
    for index in range(1, 10000):
        candidate = target.with_name(f"{stem}_{index:03d}{suffix}")
        if not candidate.exists():
            return str(candidate)
    return str(target)
