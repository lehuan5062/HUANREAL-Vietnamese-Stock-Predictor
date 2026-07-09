"""Persist each source's adaptive request rate AND cooldown across sessions.

A source's calls/min cap is a one-way ratchet: every genuine provider 429
knocks 1 off, down to a floor, and the reduced rate survives process
restarts (unlike the in-memory-only ``_RateLimiter.cap``). Its cooldown grows
the same way: it starts small and grows by a fixed step every time that
source needs cooling down again, persisted so a fresh process continues from
the accumulated cooldown rather than restarting small. Neither the rate nor
the cooldown ever recovers on its own — if a provider's real limit is
believed to have relaxed, reset manually via::

    python -c "from stockpredict.data.source_rate import reset_rates; reset_rates()"

Both live in the same per-source dict in the same JSON file (e.g.
``{"VCI": {"calls_per_min": 20.0, "cooldown_seconds": 3.0}}``), so every
write merges into the existing entry rather than replacing it — otherwise a
rate ratchet would silently wipe out that source's accumulated cooldown
(and vice versa).
"""
import json
import logging
import os
import threading
from pathlib import Path

from ..config import cache_dir

# Serializes the read-modify-write in ratchet_down within a process. Two
# fetch workers (one per source) both ratchet the SAME file, so without this
# their interleaved read-modify-writes lose updates and, combined with a
# non-atomic write, a reader can hit a half-written file. Cross-process
# safety comes from the atomic os.replace in _save_rates (a reader always
# sees the whole old or whole new file, never a torn one).
_RATE_LOCK = threading.Lock()


def _rate_file() -> Path:
    """Path to the persisted per-source rate JSON file."""
    return cache_dir() / "source_rate.json"


def _load_rates() -> dict[str, dict[str, float]]:
    """Load persisted rates from disk. Returns {source: {calls_per_min}}."""
    path = _rate_file()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logging.getLogger("stockpredict.rate").warning(
            "Failed to load source rates: %s; starting fresh", e
        )
        return {}


def _save_rates(rates: dict[str, dict[str, float]]) -> None:
    """Atomically persist rates: write a temp file then os.replace it into
    place, so a concurrent reader (other thread or process) never observes a
    partially-written file. A torn read previously parsed as corrupt JSON,
    silently reset to {}, and made ratchet_down bump the rate back up toward
    ``default`` instead of monotonically down."""
    path = _rate_file()
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(rates, indent=2), encoding="utf-8")
        os.replace(tmp, path)  # atomic on Windows and POSIX
    except Exception as e:
        logging.getLogger("stockpredict.rate").warning(
            "Failed to save source rates: %s", e
        )
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def get_persisted_rate(source: str, default: float) -> float:
    """Return the persisted calls/min for ``source``, or ``default`` if none
    has been recorded yet (first run, or after ``reset_rates``)."""
    rates = _load_rates()
    entry = rates.get(source.upper())
    if entry is None:
        return default
    return float(entry.get("calls_per_min", default))


def ratchet_down(source: str, floor: float, default: float,
                 live_cap: float | None = None) -> float:
    """Decrement ``source``'s persisted calls/min by 1, never below ``floor``.

    ``default`` seeds the rate if this source has never been ratcheted
    before (mirrors the config ceiling used when first building the
    in-memory limiter, so the very first 429 steps down from the same
    baseline rather than from an unrelated value).

    ``live_cap`` (the caller's current in-memory cap) clamps the starting
    point so the ratchet stays monotonic even if a concurrent process left a
    higher value on disk: we never step down from more than the rate this
    process is already enforcing.

    The whole read-modify-write is held under ``_RATE_LOCK`` and the write is
    atomic, so two workers ratcheting the shared file can't lose updates or
    read a torn file. Returns the new rate.
    """
    src = source.upper()
    with _RATE_LOCK:
        rates = _load_rates()
        entry = rates.setdefault(src, {})
        current = float(entry.get("calls_per_min", default))
        if live_cap is not None:
            current = min(current, float(live_cap))
        new_rate = max(float(floor), current - 1.0)
        entry["calls_per_min"] = new_rate
        _save_rates(rates)
        return new_rate


def get_persisted_cooldown(source: str, default: float) -> float:
    """Return the persisted cooldown (seconds) for ``source``, or ``default``
    if none has been recorded yet (first run, or after ``reset_rates``)."""
    rates = _load_rates()
    entry = rates.get(source.upper())
    if entry is None:
        return default
    return float(entry.get("cooldown_seconds", default))


def increment_cooldown(source: str, step: float, start: float) -> float:
    """Grow ``source``'s persisted cooldown by ``step`` seconds; seed at
    ``start`` if this source has never been cooled down before.

    One-way (never shrinks except via ``reset_rates()``) and persisted
    cross-session — a fresh process picks up the accumulated cooldown, not
    ``start``, for that source's NEXT failure. Same locking/atomicity as
    ``ratchet_down``, and merges into the same per-source entry rather than
    overwriting the ``calls_per_min`` field a rate ratchet may have set.
    Returns the cooldown to apply now.
    """
    src = source.upper()
    with _RATE_LOCK:
        rates = _load_rates()
        entry = rates.setdefault(src, {})
        current = entry.get("cooldown_seconds")
        new_cooldown = float(start) if current is None else float(current) + float(step)
        entry["cooldown_seconds"] = new_cooldown
        _save_rates(rates)
        return new_cooldown


def reset_rates() -> None:
    """Clear all persisted per-source rates AND cooldowns (manual escape hatch)."""
    _rate_file().unlink(missing_ok=True)
