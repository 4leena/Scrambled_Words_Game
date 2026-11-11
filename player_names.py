"""
Player name lookup — maps ArUco player-card IDs to camper names.

Reads player_names.json (same folder as this file). Edit that JSON directly
to assign names; no GUI involved. Falls back to "Player <id>" for any ID
not found in the file, so the game never breaks if a card isn't registered.

Usage:
    from player_names import get_player_name
    name = get_player_name(230)   # -> "Pikachu", or "Player 230" if unknown
"""

import json
import os

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_NAMES_PATH = os.path.join(_BASE_DIR, "player_names.json")

_names_cache: dict[str, dict] | None = None


def _load() -> dict:
    global _names_cache
    if _names_cache is not None:
        return _names_cache
    try:
        with open(_NAMES_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        # Keys in JSON are strings; drop the _comment key if present.
        # Accepted entry formats:
        #   old plain:  "201": "Leo"
        #   old dict:   "201": {"name": "Leo", "age": 8}
        #   new dict:   "201": {"name": "Leo", "age": "6-8", "consent": true}
        # All normalized to {"name": str, "age": str-or-None, "consent": bool}.
        # NOTE: age is kept as-is (e.g. "6-8", "9-12"); it is NOT coerced to int,
        # so range strings from users.json survive. Empty "" becomes None.
        _names_cache = {}
        for k, v in raw.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict):
                name    = str(v.get("name", f"Player {k}"))
                age     = v.get("age")
                if age == "":
                    age = None
                consent = bool(v.get("consent", False))
            else:
                name, age, consent = str(v), None, False
            _names_cache[k] = {"name": name, "age": age, "consent": consent}
        print(f"[PlayerNames] loaded {len(_names_cache)} names from {_NAMES_PATH}")
    except FileNotFoundError:
        print(f"[PlayerNames] {_NAMES_PATH} not found — using generic 'Player <id>' labels.")
        _names_cache = {}
    except json.JSONDecodeError as e:
        print(f"[PlayerNames] {_NAMES_PATH} is corrupt ({e}) — using generic labels.")
        _names_cache = {}
    return _names_cache


def get_player_name(player_id) -> str:
    """Return the camper's name for an ArUco player ID, or a generic fallback."""
    if player_id is None:
        return "Player"
    entry = _load().get(str(player_id))
    return entry["name"] if entry else f"Player {player_id}"


def get_player_age(player_id):
    """Return the camper's age for an ArUco player ID as stored in
    player_names.json (e.g. "6-8", "9-12"), or None if the card isn't
    registered or no age was entered."""
    if player_id is None:
        return None
    entry = _load().get(str(player_id))
    return entry["age"] if entry else None


def get_player_consent(player_id) -> bool:
    """Return True only if the camper's card is registered AND consent is true.
    Unregistered cards, or entries without consent, return False."""
    if player_id is None:
        return False
    entry = _load().get(str(player_id))
    return bool(entry.get("consent")) if entry else False


def reload_names() -> dict:
    """Force re-read of player_names.json (e.g. if edited mid-session)."""
    global _names_cache
    _names_cache = None
    return _load()