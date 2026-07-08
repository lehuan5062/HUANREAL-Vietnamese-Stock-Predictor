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
    If all_sources is None, returns [VCI, KBS] by default (most reliable VN stock sources).
    """
    if all_sources is None:
        all_sources = ["VCI", "KBS"]

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


def distribute_symbols_by_preference(symbols: list[str],
                                      sources: list[str] | None = None) -> dict[str, list[str]]:
    """Distribute symbols across sources weighted by historical success rates.

    Returns a dict mapping each source to its assigned symbol list.
    Sources with higher win-rates get more symbols. If no preference history exists,
    distributes evenly.

    Args:
        symbols: List of ticker symbols to distribute
        sources: List of sources (default: [VCI, KBS, MSN, TCBS])

    Returns:
        Dict mapping source name to list of assigned symbols
    """
    if sources is None:
        sources = ["VCI", "KBS", "MSN", "TCBS"]

    if not symbols:
        return {src: [] for src in sources}

    prefs = _load_preferences()

    # Calculate weight for each source based on win-rate
    def get_weight(src: str) -> float:
        stats = prefs.get(src.upper(), {"success": 0, "failure": 0})
        s = stats.get("success", 0)
        f = stats.get("failure", 0)
        if s + f == 0:
            return 1.0  # neutral weight if no history
        return s / (s + f)  # win-rate as weight

    weights = {src: get_weight(src) for src in sources}
    total_weight = sum(weights.values())

    # Normalize weights and calculate target counts per source
    distribution = {}
    remaining_symbols = list(symbols)
    random.shuffle(remaining_symbols)  # Randomize within each bucket to avoid bias

    for i, src in enumerate(sources):
        # Calculate target count for this source
        if i == len(sources) - 1:
            # Last source gets all remaining symbols (handles rounding)
            target_count = len(remaining_symbols)
        else:
            # Proportional to normalized weight
            target_count = int(len(symbols) * weights[src] / total_weight)

        distribution[src] = remaining_symbols[:target_count]
        remaining_symbols = remaining_symbols[target_count:]

    return distribution
