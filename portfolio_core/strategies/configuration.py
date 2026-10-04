"""JSON-owned strategy defaults and search choices; Python owns their schemas."""

from __future__ import annotations

import json
from pathlib import Path


DEFAULTS_PATH = Path(__file__).with_name("defaults.json")


def unique_json_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key {name!r}")
        result[name] = value
    return result


def read_json_object(path: str | Path) -> dict:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"),
                             object_pairs_hook=unique_json_object)
    except ValueError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must contain a JSON object: {path}")
    return payload


def parameter_defaults(strategy_id: str) -> dict:
    """Read a fresh packet only when a caller resolves strategy defaults."""
    payload = read_json_object(DEFAULTS_PATH)
    if strategy_id not in payload or not isinstance(payload[strategy_id], dict):
        raise ValueError(f"Missing parameter defaults for {strategy_id}")
    return payload[strategy_id]
