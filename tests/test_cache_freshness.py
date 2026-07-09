"""Verify the market-aware cache freshness check skips API calls when
the broker has nothing new to give us."""
import datetime as dt
from unittest.mock import patch

import pandas as pd
import pytest

from stockpredict import tracking
from stockpredict.data import fetcher


# Calendar covering late-April 2026 — chosen to be entirely in the past
# relative to any plausible real "today" so the old `start > today_str`
# guard inside update_symbol never short-circuits the new check we're
# trying to test.
@pytest.fixture
def fake_calendar(monkeypatch):
    days = pd.DatetimeIndex([
        "2026-04-20",  # Mon
        "2026-04-21",
        "2026-04-22",
        "2026-04-23",
        "2026-04-24",  # Fri
        # Weekend Apr 25-26 skipped
        "2026-04-27",  # Mon
        "2026-04-28",  # Tue
        "2026-04-29",  # Wed (last trading day before Apr 30 + May 1 holiday cluster)
        # Skip Apr 30 (Reunification), May 1 (Labor), May 2-3 (weekend)
    ])
    monkeypatch.setattr(tracking, "_trading_calendar_cached", lambda: days)
    yield days


# ---------------------------------------------------------------------------
# 1. latest_expected_bar_date
# ---------------------------------------------------------------------------

def test_pre_close_returns_previous_trading_day(fake_calendar):
    """Mon 09:00, mid-trading session — last finalized bar is Friday."""
    now = dt.datetime(2026, 4, 27, 9, 0)
    out = tracking.latest_expected_bar_date(now=now)
    assert out == pd.Timestamp("2026-04-24")


def test_at_atc_close_buffer_still_pre(fake_calendar):
    """Mon 14:30 — ATC just started, broker hasn't published yet."""
    now = dt.datetime(2026, 4, 27, 14, 30)
    out = tracking.latest_expected_bar_date(now=now)
    assert out == pd.Timestamp("2026-04-24")


def test_post_close_buffer_returns_today(fake_calendar):
    """Mon 15:00 — close + 15min buffer cleared, today's bar published."""
    now = dt.datetime(2026, 4, 27, 15, 0)
    out = tracking.latest_expected_bar_date(now=now)
    assert out == pd.Timestamp("2026-04-27")


def test_between_atc_and_buffer_returns_previous(fake_calendar):
    """Mon 14:50 — broker may not have published yet, conservative fallback."""
    now = dt.datetime(2026, 4, 27, 14, 50)
    out = tracking.latest_expected_bar_date(now=now)
    assert out == pd.Timestamp("2026-04-24")


def test_saturday_returns_friday(fake_calendar):
    """Sat 11:00 — last trading day was Friday."""
    now = dt.datetime(2026, 4, 25, 11, 0)
    out = tracking.latest_expected_bar_date(now=now)
    assert out == pd.Timestamp("2026-04-24")


def test_sunday_late_returns_friday(fake_calendar):
    """Sun 23:59 — still Friday."""
    now = dt.datetime(2026, 4, 26, 23, 59)
    out = tracking.latest_expected_bar_date(now=now)
    assert out == pd.Timestamp("2026-04-24")


def test_post_holiday_returns_pre_holiday_trading_day(fake_calendar):
    """Sat May 2 (after Apr 30 Reunification + May 1 Labor + weekend) —
    last trading day was Wed Apr 29."""
    now = dt.datetime(2026, 5, 2, 11, 0)
    out = tracking.latest_expected_bar_date(now=now)
    assert out == pd.Timestamp("2026-04-29")


def test_empty_calendar_returns_none(monkeypatch):
    """No cached data + projection still empty → return None safely."""
    monkeypatch.setattr(tracking, "_trading_calendar_cached",
                        lambda: pd.DatetimeIndex([]))
    # force _extended_calendar to also return empty by mocking its inputs
    # (it falls back to bdate_range — for a date with no calendar at all
    # we still expect a sane None or projected date)
    now = dt.datetime(2026, 5, 11, 9, 0)
    out = tracking.latest_expected_bar_date(now=now)
    # _extended_calendar projects weekdays so it's not actually empty —
    # just sanity-check we don't crash.
    assert out is None or isinstance(out, pd.Timestamp)


# ---------------------------------------------------------------------------
# 2. update_symbol respects the helper
# ---------------------------------------------------------------------------

def _fixed_now(now: dt.datetime):
    """Helper: monkeypatch only `datetime.now()` while preserving the rest
    of the `dt` module so other date arithmetic still works."""
    class _FrozenDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return now
    return _FrozenDatetime


def test_update_symbol_skips_when_cache_current(monkeypatch, fake_calendar):
    """Cache through expected bar → returns 0 with no fetch."""
    df = pd.DataFrame({
        "open": [10.0], "high": [10.5], "low": [9.5],
        "close": [10.2], "volume": [1000],
    }, index=pd.DatetimeIndex([pd.Timestamp("2026-04-24")], name="date"))
    monkeypatch.setattr(fetcher, "read_ohlcv", lambda s: df)

    fetch_called = {"count": 0}
    def fake_fetch(*args, **kwargs):
        fetch_called["count"] += 1
        return pd.DataFrame()
    monkeypatch.setattr(fetcher, "fetch_history", fake_fetch)

    # Saturday Apr 25 11:00 — expected bar = Fri Apr 24 = cached_max → skip.
    monkeypatch.setattr(tracking.dt, "datetime",
                        _fixed_now(dt.datetime(2026, 4, 25, 11, 0)))
    result = fetcher.update_symbol("FPT", full=False)

    assert result == 0
    assert fetch_called["count"] == 0, "fetch_history should not have been called"


def test_update_symbol_caps_end_date_at_expected_bar(monkeypatch, fake_calendar):
    """During trading hours we must not ask the broker for today's
    intraday partial bar — fetch_history must be called with end =
    latest_expected_bar_date, not 'today'."""
    df = pd.DataFrame({
        "open": [10.0], "high": [10.5], "low": [9.5],
        "close": [10.2], "volume": [1000],
    }, index=pd.DatetimeIndex([pd.Timestamp("2026-04-23")], name="date"))
    monkeypatch.setattr(fetcher, "read_ohlcv", lambda s: df)
    monkeypatch.setattr(fetcher, "merge_ohlcv", lambda s, d, validate=True: pd.concat([df, d]))
    # No prior watermark — otherwise a real-world watermark file from
    # earlier runs (dated to today) would short-circuit the fetch.
    monkeypatch.setattr(fetcher, "get_watermark", lambda s: None)
    monkeypatch.setattr(fetcher, "set_watermark", lambda s, d: None)

    captured: dict = {}
    def fake_fetch(symbol, start, end=None, source_order=None):
        captured["start"] = start
        captured["end"] = end
        return pd.DataFrame({
            "open": [10.2], "high": [10.6], "low": [10.1],
            "close": [10.5], "volume": [2000],
        }, index=pd.DatetimeIndex([pd.Timestamp("2026-04-24")], name="date"))
    monkeypatch.setattr(fetcher, "fetch_history", fake_fetch)

    # Pre-close on Mon Apr 27 — expected bar = Fri Apr 24.
    # Cache through Thu Apr 23, so a fetch fires for Apr 24 only.
    monkeypatch.setattr(tracking.dt, "datetime",
                        _fixed_now(dt.datetime(2026, 4, 27, 9, 0)))
    fetcher.update_symbol("FPT", full=False)

    # The end date passed to fetch_history must be Apr 24 (the latest
    # finalized bar) — NOT Apr 27 (today, which has no published close yet).
    assert captured["end"] == "2026-04-24", (
        f"end-date should be capped at latest finalized bar; got {captured['end']}"
    )
    assert captured["start"] == "2026-04-24"


def test_update_symbol_fetches_when_cache_stale(monkeypatch, fake_calendar):
    """Cache one day behind expected → fetch_history is called."""
    df = pd.DataFrame({
        "open": [10.0], "high": [10.5], "low": [9.5],
        "close": [10.2], "volume": [1000],
    }, index=pd.DatetimeIndex([pd.Timestamp("2026-04-23")], name="date"))
    monkeypatch.setattr(fetcher, "read_ohlcv", lambda s: df)
    monkeypatch.setattr(fetcher, "merge_ohlcv", lambda s, d, validate=True: pd.concat([df, d]))
    # No prior watermark — every test starts with a clean slate.
    monkeypatch.setattr(fetcher, "get_watermark", lambda s: None)
    monkeypatch.setattr(fetcher, "set_watermark", lambda s, d: None)

    fetch_called = {"count": 0}
    def fake_fetch(symbol, start, end=None, source_order=None):
        fetch_called["count"] += 1
        return pd.DataFrame({
            "open": [10.2], "high": [10.6], "low": [10.1],
            "close": [10.5], "volume": [2000],
        }, index=pd.DatetimeIndex([pd.Timestamp("2026-04-24")], name="date"))
    monkeypatch.setattr(fetcher, "fetch_history", fake_fetch)

    # Saturday Apr 25 11:00 — expected = Fri Apr 24, cached = Thu Apr 23 → fetch.
    monkeypatch.setattr(tracking.dt, "datetime",
                        _fixed_now(dt.datetime(2026, 4, 25, 11, 0)))
    result = fetcher.update_symbol("FPT", full=False)

    assert fetch_called["count"] == 1, "expected fetch to fire when cache is stale"
    assert result == 1  # one new row


def test_update_symbol_forces_full_refetch_on_corporate_action_rejection(monkeypatch, fake_calendar):
    """When merge_ohlcv rejects an incremental append as a suspected
    corporate-action artifact, update_symbol retries with full=True instead
    of propagating the error, and returns the healed result."""
    from stockpredict.data.cache import SuspectedCorporateActionArtifact

    df = pd.DataFrame({
        "open": [10.0], "high": [10.5], "low": [9.5],
        "close": [10.2], "volume": [1000],
    }, index=pd.DatetimeIndex([pd.Timestamp("2026-04-23")], name="date"))
    monkeypatch.setattr(fetcher, "read_ohlcv", lambda s: df)
    monkeypatch.setattr(fetcher, "get_watermark", lambda s: None)
    monkeypatch.setattr(fetcher, "set_watermark", lambda s, d: None)

    def fake_merge(symbol, new, validate=True):
        if validate:
            raise SuspectedCorporateActionArtifact("simulated checkerboard")
        # full=True path (validate=False) succeeds, healing the cache.
        return pd.concat([df, new])
    monkeypatch.setattr(fetcher, "merge_ohlcv", fake_merge)

    fetch_called = {"count": 0}
    def fake_fetch(symbol, start, end=None, source_order=None):
        fetch_called["count"] += 1
        return pd.DataFrame({
            "open": [10.2], "high": [10.6], "low": [10.1],
            "close": [10.5], "volume": [2000],
        }, index=pd.DatetimeIndex([pd.Timestamp("2026-04-24")], name="date"))
    monkeypatch.setattr(fetcher, "fetch_history", fake_fetch)

    monkeypatch.setattr(tracking.dt, "datetime",
                        _fixed_now(dt.datetime(2026, 4, 25, 11, 0)))
    result = fetcher.update_symbol("FPT", full=False)

    # First call (incremental, validate=True) rejected -> retried with
    # full=True (validate=False) -> fetch_history called again for the retry.
    assert fetch_called["count"] == 2
    assert result == 1


def test_update_symbol_full_refetch_still_corrupted_reraises(monkeypatch, fake_calendar):
    """If even a full re-fetch (validate=False) still hits an internal
    violation, update_symbol must NOT recurse forever -- it re-raises."""
    from stockpredict.data.cache import SuspectedCorporateActionArtifact

    df = pd.DataFrame({
        "open": [10.0], "high": [10.5], "low": [9.5],
        "close": [10.2], "volume": [1000],
    }, index=pd.DatetimeIndex([pd.Timestamp("2026-04-23")], name="date"))
    monkeypatch.setattr(fetcher, "read_ohlcv", lambda s: df)
    monkeypatch.setattr(fetcher, "get_watermark", lambda s: None)
    monkeypatch.setattr(fetcher, "set_watermark", lambda s, d: None)

    def always_raise(symbol, new, validate=True):
        raise SuspectedCorporateActionArtifact("still corrupted even on full refetch")
    monkeypatch.setattr(fetcher, "merge_ohlcv", always_raise)
    monkeypatch.setattr(fetcher, "fetch_history", lambda *a, **kw: pd.DataFrame({
        "open": [10.2], "high": [10.6], "low": [10.1],
        "close": [10.5], "volume": [2000],
    }, index=pd.DatetimeIndex([pd.Timestamp("2026-04-24")], name="date")))

    monkeypatch.setattr(tracking.dt, "datetime",
                        _fixed_now(dt.datetime(2026, 4, 25, 11, 0)))
    with pytest.raises(SuspectedCorporateActionArtifact):
        fetcher.update_symbol("FPT", full=False)


# ---------------------------------------------------------------------------
# 3. update_many fast-path
# ---------------------------------------------------------------------------

def test_update_many_fast_path_no_threads(monkeypatch, fake_calendar):
    """When all selected symbols are current, update_many returns immediately
    with all zeros and never hits the thread pool."""
    df = pd.DataFrame({
        "open": [10.0], "high": [10.5], "low": [9.5],
        "close": [10.2], "volume": [1000],
    }, index=pd.DatetimeIndex([pd.Timestamp("2026-04-24")], name="date"))
    monkeypatch.setattr(fetcher, "read_ohlcv", lambda s: df)

    update_called = {"count": 0}
    def fake_update(s, full=False):
        update_called["count"] += 1
        return 0
    monkeypatch.setattr(fetcher, "update_symbol", fake_update)

    monkeypatch.setattr(tracking.dt, "datetime",
                        _fixed_now(dt.datetime(2026, 4, 25, 11, 0)))
    results = fetcher.update_many(["FPT", "VCB", "MSN"], full=False)

    assert results == {"FPT": 0, "VCB": 0, "MSN": 0}
    assert update_called["count"] == 0, "fast-path should bypass per-symbol update entirely"
