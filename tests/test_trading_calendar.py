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


# ---- end-of-month resolution -----------------------------------------------

@pytest.fixture
def may_calendar(monkeypatch):
    """Synthetic calendar covering Apr / May / June 2026 with the
    Reunification + Labor Day holiday cluster (Apr 30 + May 1) excised."""
    days = pd.DatetimeIndex([
        # April 2026 (last week)
        "2026-04-27", "2026-04-28", "2026-04-29",
        # May 2026 — full month, holidays Apr 30 / May 1 already excluded
        "2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08",
        "2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15",
        "2026-05-18", "2026-05-19", "2026-05-20", "2026-05-21", "2026-05-22",
        "2026-05-25", "2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29",
        # June 2026
        "2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05",
        "2026-06-08", "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12",
        "2026-06-15", "2026-06-16", "2026-06-17", "2026-06-18", "2026-06-19",
        "2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25", "2026-06-26",
        "2026-06-29", "2026-06-30",
    ])
    from stockpredict import tracking
    tracking._invalidate_trading_calendar_cache()
    monkeypatch.setattr(tracking, "_trading_calendar_cached", lambda: days)
    yield days


def test_days_end_mid_month_targets_current_month_last_day(may_calendar):
    """T = May 5 (mid-month). End of May = May 29. Distance is plenty (>2)
    so we target the current month."""
    from stockpredict.tracking import days_to_month_end
    n = days_to_month_end(pd.Timestamp("2026-05-05"), min_days=2)
    # May 5 -> May 29: count trading days. Indices: May 5 = 4, May 29 = 22. n = 18.
    assert n == 18


def test_days_end_one_day_before_last_rolls_to_next_month(may_calendar):
    """T = May 28. Last trading day = May 29 -> distance = 1 < 2.
    Must roll to next-month last trading day = June 30."""
    from stockpredict.tracking import days_to_month_end
    n = days_to_month_end(pd.Timestamp("2026-05-28"), min_days=2)
    # May 28 = pos 21, June 30 = pos 44. n = 23.
    assert n == 23


def test_days_end_on_last_day_rolls_to_next_month(may_calendar):
    """T = May 29 (last trading day of May) -> distance = 0 < 2.
    Must roll to June 30."""
    from stockpredict.tracking import days_to_month_end
    n = days_to_month_end(pd.Timestamp("2026-05-29"), min_days=2)
    # May 29 = pos 22, June 30 = pos 44. n = 22.
    assert n == 22


def test_days_end_exactly_two_days_before_uses_current_month(may_calendar):
    """T = May 27. Last trading day = May 29 -> distance = 2 (T+2 exactly).
    Should use current month, not roll over."""
    from stockpredict.tracking import days_to_month_end
    n = days_to_month_end(pd.Timestamp("2026-05-27"), min_days=2)
    assert n == 2


def test_days_end_at_year_boundary(monkeypatch):
    """Year-boundary rollover: December last day -> January next-month last day."""
    days = pd.DatetimeIndex([
        # late Dec 2026
        "2026-12-29", "2026-12-30", "2026-12-31",
        # Jan 2027
        "2027-01-04", "2027-01-05", "2027-01-06", "2027-01-07", "2027-01-08",
        "2027-01-11", "2027-01-12", "2027-01-13", "2027-01-14", "2027-01-15",
        "2027-01-18", "2027-01-19", "2027-01-20", "2027-01-21", "2027-01-22",
        "2027-01-25", "2027-01-26", "2027-01-27", "2027-01-28", "2027-01-29",
    ])
    from stockpredict import tracking
    monkeypatch.setattr(tracking, "_trading_calendar_cached", lambda: days)
    n = tracking.days_to_month_end(pd.Timestamp("2026-12-31"), min_days=2)
    # Dec 31 = pos 2, Jan 29 = pos 22. n = 20.
    assert n == 20
