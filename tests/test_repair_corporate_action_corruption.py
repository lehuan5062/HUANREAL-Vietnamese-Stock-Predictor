"""Tests for scripts/repair_corporate_action_corruption.py's detection logic.

The detector requires a REVERSAL (spike then back near the pre-spike level
within a short window) -- not a bare "any band violation anywhere in
history" -- specifically so it does NOT flag genuine, permanent corporate
action level changes (already handled elsewhere), only phantom-instrument
injections (temporary, self-reverting) like the confirmed MSN/ABB incident.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from stockpredict.data import cache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import repair_corporate_action_corruption as repair  # noqa: E402


def _patch_universe(monkeypatch, mapping):
    import stockpredict.filters as filters
    import stockpredict.data.universe as u
    uni = pd.DataFrame({"symbol": list(mapping), "exchange": list(mapping.values())})
    monkeypatch.setattr(u, "load_universe", lambda: uni)
    filters._ceiling_params.cache_clear()


def _series(closes, start="2026-06-01"):
    idx = pd.date_range(start, periods=len(closes), freq="B", name="date")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [100000] * len(closes)},
        index=idx,
    )


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "ohlcv_dir", lambda: tmp_path)
    monkeypatch.setattr(cache, "ohlcv_path", lambda s: tmp_path / f"{s.upper()}.parquet")
    monkeypatch.setattr(repair, "ohlcv_dir", lambda: tmp_path)
    yield tmp_path


def test_flags_a_spike_that_reverts(monkeypatch, isolated_cache):
    """A phantom-instrument-shaped spike (jump beyond band, then back near
    the pre-spike level within a few days) IS flagged -- matches ABB's
    confirmed real incident shape (MSN's ~600-900 range vs ABB's ~5-18)."""
    _patch_universe(monkeypatch, {"CLEAN": "HSX", "SPIKY": "HSX"})

    clean = _series([10.0, 10.1, 10.2, 10.15, 10.3, 10.25, 10.4])
    cache.write_ohlcv("CLEAN", clean)

    # 5.6 -> 619 (phantom) -> 608 (still phantom) -> back to ~5.8 for several
    # bars (reverted) -- several post-spike bars near the old level, not
    # just one noisy touch, to satisfy _MIN_REVERTED_BARS.
    spiky = _series([5.64, 619.0, 608.6, 5.80, 5.88, 5.82, 5.86])
    cache.write_ohlcv("SPIKY", spiky)

    found = repair.find_phantom_spike_symbols()
    assert list(found.keys()) == ["SPIKY"]


def test_does_not_flag_a_permanent_level_change(monkeypatch, isolated_cache):
    """The key fix: a genuine, PERMANENT corporate-action-shaped drop (e.g.
    an ex-rights gap that never reverts, staying at the new level) must NOT
    be flagged -- that's real, already-handled data, not a phantom
    injection. An earlier version of this detector (bare band-violation
    check with no reversal requirement) flagged ~1343/~1550 symbols in the
    real cache precisely because it couldn't tell these apart."""
    _patch_universe(monkeypatch, {"PERM": "HSX"})
    # -38% ex-rights-style gap that STAYS down (VVS-shaped), no reversion.
    perm = _series([116.8, 72.3, 71.9, 72.5, 72.1, 71.8, 72.9, 73.0, 72.4, 72.6, 73.1, 72.8])
    cache.write_ohlcv("PERM", perm)

    found = repair.find_phantom_spike_symbols()
    assert found == {}


def test_empty_cache_returns_nothing(monkeypatch, isolated_cache):
    _patch_universe(monkeypatch, {})
    assert repair.find_phantom_spike_symbols() == {}


def test_is_confirmed_phantom_date_when_no_source_has_data(monkeypatch):
    """Neither VCI nor KBS has data for the date -> confirmed phantom, drop."""
    monkeypatch.setattr(repair, "_quote_history",
                        lambda *a, **kw: pd.DataFrame())
    is_phantom, replacement = repair._is_confirmed_phantom_date(
        "ABB", pd.Timestamp("2025-01-27")
    )
    assert is_phantom is True
    assert replacement is None


def test_is_confirmed_phantom_date_when_a_source_has_real_data(monkeypatch):
    """A source DOES have data for the date -> not phantom, return its value
    as a replacement instead of dropping the row."""
    def fake_quote_history(symbol, src, start, end, interval, bypass):
        if src == "VCI":
            return pd.DataFrame({"close": [15.83]})
        return pd.DataFrame()
    monkeypatch.setattr(repair, "_quote_history", fake_quote_history)

    is_phantom, replacement = repair._is_confirmed_phantom_date(
        "ABB", pd.Timestamp("2026-06-25")
    )
    assert is_phantom is False
    assert replacement == 15.83
