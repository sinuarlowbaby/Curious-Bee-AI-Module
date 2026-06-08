"""
Generic JSON list persistence helpers.

Both the calendar and announcements stores round-trip through these
two functions, so the "missing file == empty list" and "empty file ==
empty list" semantics live in one place.
"""

import json
import os


def load_json(path: str) -> list:
    """
    Load a JSON list from disk.

    Returns an empty list when the file is missing or empty. Raises the
    underlying ``json.JSONDecodeError`` when the file is present and
    non-empty but malformed.
    """
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []
        return json.loads(content)


def save_json(path: str, data: list) -> None:
    """Persist a list as a pretty-printed JSON file (UTF-8)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
