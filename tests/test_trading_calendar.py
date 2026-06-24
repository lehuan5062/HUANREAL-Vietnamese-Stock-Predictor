"""Verify that T+N target_date computation uses the real trading-day calendar
(weekends + Vietnamese holidays excluded), not naive business-day arithmetic."""
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from stockpredict.tracking import _next_trading_offset


@pytest.fixture
def fake_calendar(monkeypatch):
    """Pretend the OHLCV cache contains a calendar that skips a holiday week.
    Concretely: simulate Vietnamese Reunification Day + Labor Day (Apr 30 +
    May 1, 2026 = Thursday + Friday) as non-trading days, plus weekends."""
    days = pd.DatetimeIndex([
        # Mon-Wed of week 1
        "2026-04-27", "2026-04-28", "2026-04-29",
        # SKIP Apr 30 (Thu, Reunification Day) + May 1 (Fri, Labor Day) + weekend
        # Resume next Mon-Fri
        "2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08",
        # Following week
        "2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15",
    ])

    from stockpredict import tracking
    tracking._invalidate_trading_calendar_cache()
    monkeypatch.setattr(tracking, "_trading_calendar_cached", lambda: days)
    yield days
    # monkeypatch's auto-teardown restores the original lru-cached function;
    # nothing else to do here.


def test_t_plus_2_skips_two_day_holiday_window(fake_calendar):
    """If T = April 29 (Wed before April 30 + May 1 holidays), T+2 should
    be May 5 — the second trading day after April 29 — NOT May 1 (the
    naive Mon-Fri-arithmetic answer)."""
    t = pd.Timestamp("2026-04-29")
    target = _next_trading_offset(t, 2)
    assert target == pd.Timestamp("2026-05-05"), (
        f"expected 2026-05-05 (skipping Apr 30 + May 1 holidays + weekend); "
        f"got {target.date()}"
    )


def test_t_plus_5_walks_calendar(fake_calendar):
    """T = April 28 (pos 1), T+5 = pos 6 = May 7 (5 trading days later,
    skipping Apr 30 + May 1 holidays and the weekend)."""
    t = pd.Timestamp("2026-04-28")
    target = _next_trading_offset(t, 5)
    assert target == pd.Timestamp("2026-05-07")


def test_start_on_holiday_uses_next_trading_day_as_anchor(fake_calendar):
    """If the user records a prediction with as_of on a non-trading day
    (shouldn't happen in practice but be defensive), we anchor on the next
    trading day at-or-after that date."""
    t = pd.Timestamp("2026-05-01")  # Labor Day, not in calendar
    target = _next_trading_offset(t, 2)
    # searchsorted finds May 4 (next trading day) at pos=3; pos+2 = 5 -> May 6
    assert target == pd.Timestamp("2026-05-06")


def test_falls_back_to_bday_when_cache_empty(monkeypatch):
    """Without any cached OHLCV, fall back to business-day arithmetic."""
    from stockpredict import tracking
    tracking._invalidate_trading_calendar_cache()
    monkeypatch.setattr(tracking, "_trading_calendar_cached",
                        lambda: pd.DatetimeIndex([]))
    t = pd.Timestamp("2026-05-05")  # Tuesday
    target = _next_trading_offset(t, 2)
    assert target == pd.Timestamp("2026-05-07")  # Thursday, BDay arithmetic


def test_offset_zero_returns_start_when_trading_day(fake_calendar):
    """T+0 on a known trading day returns that day."""
    t = pd.Timestamp("2026-05-05")
    target = _next_trading_offset(t, 0)
    assert target == pd.Timestamp("2026-05-05")
