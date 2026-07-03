"""Track and rank data sources by success/failure rates."""
import json
import logging
import random
from pathlib import Path
from typing import Optional

from ..config import cache_dir


def _preference_file() -> Path:
    """Path to the source preference JSON file."""
    return cache_dir() / "source_preference.json"


def _load_preferences() -> dict[str, dict[str, int]]:
    """Load source success/failure counts from disk. Returns {source: {success, failure}}."""
    path = _preference_file()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logging.getLogger("stockpredict.source").warning(
            "Failed to load source preferences: %s; starting fresh", e
        )
        return {}


def _save_preferences(prefs: dict[str, dict[str, int]]) -> None:
    """Save source success/failure counts to disk."""
    path = _preference_file()
    try:
        path.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    except Exception as e:
        logging.getLogger("stockpredict.source").warning(
            "Failed to save source preferences: %s", e
        )


def track_source_success(source: str) -> None:
    """Record a successful fetch from this source."""
    prefs = _load_preferences()
    src = source.upper()
    if src not in prefs:
        prefs[src] = {"success": 0, "failure": 0}
    prefs[src]["success"] += 1
    _save_preferences(prefs)


def track_source_failure(source: str) -> None:
    """Record a failed fetch from this source."""
    prefs = _load_preferences()
    src = source.upper()
    if src not in prefs:
        prefs[src] = {"success": 0, "failure": 0}
    prefs[src]["failure"] += 1
    _save_preferences(prefs)


def get_source_order(all_sources: Optional[list[str]] = None) -> list[str]:
    """Return sources ordered by win-rate (success / (success + failure)).

    Ties broken randomly. If a source has no history, treat as neutral (0.5).
    If all_sources is None, returns [VCI, KBS, MSN, TCBS] by default.
    """
    if all_sources is None:
        all_sources = ["VCI", "KBS", "MSN", "TCBS"]

    prefs = _load_preferences()

    def win_rate(src: str) -> tuple[float, int]:
        """Return (win_rate, random_tiebreaker) for sorting."""
        stats = prefs.get(src.upper(), {"success": 0, "failure": 0})
        s = stats.get("success", 0)
        f = stats.get("failure", 0)
        if s + f == 0:
            rate = 0.5  # neutral if no history
        else:
            rate = s / (s + f)
        return (-rate, random.random())  # descending order, then random tiebreak

    ordered = sorted(all_sources, key=win_rate)
    logger = logging.getLogger("stockpredict.source")
    prefs_str = ", ".join(
        f"{src.upper()}({prefs.get(src.upper(), {}).get('success', 0)}/{prefs.get(src.upper(), {}).get('failure', 0)})"
        for src in ordered
    )
    logger.info("source order (by win-rate): %s", prefs_str)
    return ordered


def reset_preferences() -> None:
    """Clear all source preference history."""
    _preference_file().unlink(missing_ok=True)
