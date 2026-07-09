"""Tests for cache.py's write-time corporate-action guard.

merge_ohlcv rejects an incremental append whose 1-day move against the last
cached close is physically impossible for a normal trading day (beyond the
symbol's exchange band + margin) -- see SuspectedCorporateActionArtifact's
docstring for the ABB cache-checkerboard incident that prompted this.
"""
from __future__ import annotations

import pandas as pd
import pytest

from stockpredict.data import cache
from stockpredict.data.cache import SuspectedCorporateActionArtifact


def _patch_universe(monkeypatch, mapping):
    import stockpredict.filters as filters
    import stockpredict.data.universe as u
    uni = pd.DataFrame({"symbol": list(mapping), "exchange": list(mapping.values())})
    monkeypatch.setattr(u, "load_universe", lambda: uni)
    filters._ceiling_params.cache_clear()


def _row(close, date="2026-07-08"):
    return pd.DataFrame(
        {"open": [close], "high": [close], "low": [close], "close": [close],
         "volume": [100000]},
        index=pd.DatetimeIndex([date], name="date"),
    )


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "ohlcv_dir", lambda: tmp_path)
    monkeypatch.setattr(cache, "ohlcv_path", lambda s: tmp_path / f"{s.upper()}.parquet")
    yield tmp_path


def test_merge_ohlcv_rejects_impossible_incremental_move(monkeypatch, isolated_cache):
    """HOSE: an 18.70 -> 16.52 incremental append (-11.66%, beyond the 9%
    HOSE band+margin) against an existing cached tail is rejected."""
    _patch_universe(monkeypatch, {"ABB": "HSX"})
    cache.write_ohlcv("ABB", _row(18.70, "2026-07-07"))

    new = _row(16.52, "2026-07-08")
    with pytest.raises(SuspectedCorporateActionArtifact):
        cache.merge_ohlcv("ABB", new, validate=True)

    # Nothing was written -- the file on disk is unchanged.
    on_disk = cache.read_ohlcv("ABB")
    assert len(on_disk) == 1
    assert float(on_disk["close"].iloc[-1]) == 18.70


def test_merge_ohlcv_allows_normal_incremental_move(monkeypatch, isolated_cache):
    """A normal in-band move is appended without issue."""
    _patch_universe(monkeypatch, {"ABB": "HSX"})
    cache.write_ohlcv("ABB", _row(18.70, "2026-07-07"))

    new = _row(18.90, "2026-07-08")  # +1.07%, well within band
    merged = cache.merge_ohlcv("ABB", new, validate=True)
    assert len(merged) == 2
    assert float(merged["close"].iloc[-1]) == 18.90


def test_merge_ohlcv_validate_false_allows_the_same_jump_through(monkeypatch, isolated_cache):
    """The full-refetch path (validate=False) must never self-reject --
    it's the mechanism meant to HEAL a corrupted cache, not get blocked by
    the same guard that flags corruption for a plain incremental append."""
    _patch_universe(monkeypatch, {"ABB": "HSX"})
    cache.write_ohlcv("ABB", _row(18.70, "2026-07-07"))

    new = _row(16.52, "2026-07-08")  # same impossible jump as the rejected case
    merged = cache.merge_ohlcv("ABB", new, validate=False)
    assert len(merged) == 2
    assert float(merged["close"].iloc[-1]) == 16.52


def test_write_ohlcv_stable_sort_makes_new_data_win_the_dedup(monkeypatch, isolated_cache):
    """Regression test: sort_index()'s default (quicksort) is NOT stable, so
    when existing and new both have a row for the SAME date, the tie-break
    order was non-deterministic and duplicated(keep="last") could pick the
    STALE existing value instead of the fresh new one. Confirmed live: a
    full re-fetch that should have overwritten ABB's corrupted rows with
    clean VCI data left the corruption in place because of this -- and at
    realistic scale (~1400 fully-overlapping rows, matching a real full
    history) the default sort corrupts roughly HALF the rows, not a rare
    edge case. A tiny 1-row-vs-1-row tie is NOT a reliable repro (quicksort
    happens to preserve order for trivially small inputs), so this test
    uses a large fully-overlapping frame to actually exercise the
    instability. Must use sort_index(kind="stable") so existing
    (concatenated first) reliably sorts before new (concatenated second)
    for every tied date, letting keep="last" correctly pick new's fresher
    value every time, at any scale."""
    import numpy as np

    _patch_universe(monkeypatch, {"ABB": "HSX"})
    idx = pd.date_range("2020-01-01", periods=1400, freq="B", name="date")
    stale = pd.DataFrame(
        {"open": np.arange(1400.0), "high": np.arange(1400.0), "low": np.arange(1400.0),
         "close": np.arange(1400.0), "volume": np.ones(1400)},
        index=idx,
    )
    cache.write_ohlcv("ABB", stale)

    fresh_closes = np.arange(1400.0, 2800.0)  # every value distinct from `stale`'s
    fresh = pd.DataFrame(
        {"open": fresh_closes, "high": fresh_closes, "low": fresh_closes,
         "close": fresh_closes, "volume": np.ones(1400)},
        index=idx,  # SAME dates as stale -- full overlap, exactly like a full refetch
    )
    merged = cache.merge_ohlcv("ABB", fresh, validate=False)

    stale_survivors = (merged["close"].astype(float) < 1400).sum()
    assert stale_survivors == 0, (
        f"{stale_survivors}/1400 rows kept the stale value instead of the "
        f"fresh one -- the dedup must always prefer new data"
    )


def test_merge_ohlcv_no_existing_history_boundary_check_skipped(monkeypatch, isolated_cache):
    """A cold symbol (no existing cache) has no BOUNDARY to validate against
    -- the first-ever write succeeds when its own rows are internally
    consistent, regardless of what magnitude they'd be relative to some
    nonexistent prior history."""
    _patch_universe(monkeypatch, {"ABB": "HSX"})
    new = pd.DataFrame(
        {"open": [18.60, 18.70], "high": [18.60, 18.70], "low": [18.60, 18.70],
         "close": [18.60, 18.70], "volume": [1000, 1000]},
        index=pd.DatetimeIndex(["2026-07-01", "2026-07-02"], name="date"),
    )
    merged = cache.merge_ohlcv("ABB", new, validate=True)
    assert len(merged) == 2


def test_merge_ohlcv_validate_false_accepts_internal_violation_known_limitation(
    monkeypatch, isolated_cache
):
    """Documents a deliberately-accepted limitation, not a desired behavior:
    with validate=False (the full=True path), a bad row from a flaky source
    sandwiched between good neighbors WITHIN THE SAME fetch batch is NOT
    caught (confirmed live: MSN briefly returned ABB's close as ~1039 for
    one date while VCI/KBS both agreed on ~15.8 for the same date).

    An earlier attempt to ALSO validate internal batch consistency during
    full=True broke far worse: it rejects ANY symbol whose real multi-year
    history contains a genuine, permanent corporate-action jump (extremely
    common for VN stocks -- confirmed live, it broke full re-fetches for
    ordinary symbols like A32/AAM/AAS). A bare consecutive-move check can't
    tell "real permanent level change" from "phantom row that will revert"
    without a reversal-lookahead window (see
    scripts/repair_corporate_action_corruption.py, which DOES have that
    lookahead and is the right place for this distinction -- not here).
    Mitigation: prefer a single trusted source, not a multi-source fallback
    chain, when doing a full re-fetch to heal a specific symbol."""
    _patch_universe(monkeypatch, {"ABB": "UPCOM"})
    new = pd.DataFrame(
        {"open": [15.74, 1039.5, 15.63], "high": [15.74, 1039.5, 15.63],
         "low": [15.74, 1039.5, 15.63], "close": [15.74, 1039.5, 15.63],
         "volume": [1000, 1000, 1000]},
        index=pd.DatetimeIndex(["2026-06-24", "2026-06-25", "2026-06-26"], name="date"),
    )
    merged = cache.merge_ohlcv("ABB", new, validate=False)  # full=True path
    assert len(merged) == 3  # accepted as-is; not this layer's job to catch


def test_merge_ohlcv_full_refetch_skips_boundary_check_entirely(
    monkeypatch, isolated_cache
):
    """validate=False (full=True) skips validation entirely, including the
    boundary against existing stale/corrupted history -- so a full refetch
    can always jump past it to heal the cache, regardless of magnitude."""
    _patch_universe(monkeypatch, {"ABB": "HSX"})
    cache.write_ohlcv("ABB", _row(9999.0, "2026-07-06"))  # stale/corrupted existing tail

    new = _row(18.70, "2026-07-07")
    merged = cache.merge_ohlcv("ABB", new, validate=False)
    assert float(merged["close"].iloc[-1]) == 18.70
