"""Verify the after-14:30 cutoff, the run signature, and the horizon-
grouped feedback all behave correctly."""
import datetime as dt

import pandas as pd
import pytest

from stockpredict import tracking


# ---------------------------------------------------------------------------
# 1. effective_today_for_trading
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_calendar(monkeypatch):
    days = pd.DatetimeIndex([
        "2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08",
        "2026-05-11", "2026-05-12", "2026-05-13",
    ])
    monkeypatch.setattr(tracking, "_trading_calendar_cached", lambda: days)
    yield days


def test_before_cutoff_returns_today(fake_calendar):
    """Pre-14:30: T+0 = today's calendar date."""
    now = dt.datetime(2026, 5, 5, 9, 0)  # Tuesday 9 AM
    out = tracking.effective_today_for_trading(now=now)
    assert out == pd.Timestamp("2026-05-05")


def test_at_cutoff_still_today(fake_calendar):
    """Exactly 14:30 still counts as 'today' (boundary inclusive)."""
    now = dt.datetime(2026, 5, 5, 14, 30)
    out = tracking.effective_today_for_trading(now=now)
    assert out == pd.Timestamp("2026-05-05")


def test_after_cutoff_advances_to_next_trading_day(fake_calendar):
    """Post-14:30 Tuesday → T+0 = Wednesday."""
    now = dt.datetime(2026, 5, 5, 14, 31)
    out = tracking.effective_today_for_trading(now=now)
    assert out == pd.Timestamp("2026-05-06")


def test_after_cutoff_skips_weekend(fake_calendar):
    """Post-14:30 Friday → T+0 = Monday (skipping Sat / Sun)."""
    now = dt.datetime(2026, 5, 8, 16, 0)
    out = tracking.effective_today_for_trading(now=now)
    assert out == pd.Timestamp("2026-05-11")


def test_weekend_morning_rolls_to_next_trading_day(fake_calendar):
    """A run on a non-trading day rolls forward regardless of the clock:
    Sunday 9 AM (before the cutoff) → next trading day = Monday, NOT Sunday."""
    now = dt.datetime(2026, 5, 10, 9, 0)  # Sunday morning, pre-cutoff
    out = tracking.effective_today_for_trading(now=now)
    assert out == pd.Timestamp("2026-05-11")


def test_saturday_after_cutoff_rolls_to_monday(fake_calendar):
    """Saturday afternoon → Monday (weekend roll, cutoff irrelevant)."""
    now = dt.datetime(2026, 5, 9, 16, 0)  # Saturday
    out = tracking.effective_today_for_trading(now=now)
    assert out == pd.Timestamp("2026-05-11")


def test_after_cutoff_with_holiday(monkeypatch):
    """Post-14:30 on the last day before a holiday cluster → skips both
    the holiday and the weekend."""
    days = pd.DatetimeIndex([
        "2026-04-29",  # Wed before Apr 30 + May 1 holidays + weekend
        "2026-05-04",  # next trading day
        "2026-05-05",
    ])
    monkeypatch.setattr(tracking, "_trading_calendar_cached", lambda: days)
    now = dt.datetime(2026, 4, 29, 15, 0)  # Wed afternoon, post cutoff
    out = tracking.effective_today_for_trading(now=now)
    assert out == pd.Timestamp("2026-05-04")


# ---------------------------------------------------------------------------
# 2. run_signature
# ---------------------------------------------------------------------------

def test_run_signature_distinct_for_distinct_params():
    """Different parameter combos yield different signatures."""
    sigs = {
        tracking.run_signature("base", hose_only=False),
        tracking.run_signature("base", hose_only=True),
        tracking.run_signature("claude", hose_only=False),
        tracking.run_signature("base", hose_only=False, exclude=["HPG"]),
        tracking.run_signature("base", hose_only=False, include_etfs=False),
    }
    assert len(sigs) == 5, f"expected 5 distinct signatures; got {len(sigs)}: {sigs}"


def test_run_signature_idempotent():
    """Same params -> same signature. No horizon token (flexible exit)."""
    a = tracking.run_signature("claude", hose_only=True)
    b = tracking.run_signature("claude", hose_only=True)
    assert a == b
    assert a == "claude_HOSE"


def test_run_signature_no_hose_tag_when_off():
    """hose_only=False omits the HOSE tag — keeps signatures readable."""
    sig = tracking.run_signature("base", hose_only=False)
    assert "HOSE" not in sig
    assert sig == "base"


# ---------------------------------------------------------------------------
# 3. by-signature feedback grouping
# ---------------------------------------------------------------------------

def _ledger(rows):
    """Build a ledger DataFrame compatible with tracking._read."""
    return pd.DataFrame(rows, columns=tracking._LEDGER_COLUMNS)


def _row(**kw):
    """Build a ledger row with sensible defaults so tests stay terse."""
    today = pd.Timestamp.today().normalize()
    base = {
        "run_id": "20260420_claude",
        "signature": "claude",
        "as_of": today - pd.Timedelta(days=10),
        "target_date": today - pd.Timedelta(days=8),
        "mode": "claude",
        "symbol": "AAA",
        "rank": 1,
        "news_score": 0,
        "entry_price": 10.0,
        "actual_exit": 10.5,
        "realized_return": 0.05,
        "pred_days": 3.0,
        "pred_profit": 0.03,
        "evaluated": True,
    }
    base.update(kw)
    return base


def test_by_run_signature_separates_param_combos(monkeypatch):
    """Different parameter combos get distinct rows in by_run_signature."""
    df = _ledger([
        _row(signature="claude", symbol="AAA", realized_return=0.05),
        _row(signature="claude", symbol="BBB", rank=2, realized_return=-0.05),
        _row(signature="claude_HOSE", symbol="CCC", realized_return=0.10),
        _row(signature="claude_HOSE", symbol="DDD", rank=2, realized_return=0.06),
    ])
    monkeypatch.setattr(tracking, "_read", lambda: df)

    perf = tracking.recent_performance(window_days=90, mode="claude")
    sigs = perf["by_run_signature"]
    assert sigs["claude"]["n"] == 2
    assert sigs["claude"]["hit_rate"] == 0.5
    assert sigs["claude_HOSE"]["n"] == 2
    assert sigs["claude_HOSE"]["hit_rate"] == 1.0


def test_feedback_block_marks_current_signature(monkeypatch):
    """The by_run_signature table marks the row whose signature matches."""
    df = _ledger([
        _row(signature="claude", symbol="AAA", realized_return=0.05),
        _row(signature="claude_HOSE", symbol="BBB", realized_return=0.10),
    ])
    monkeypatch.setattr(tracking, "_read", lambda: df)

    block = tracking.feedback_block(window_days=90, mode="claude",
                                    current_signature="claude_HOSE")
    sig_lines = [ln for ln in block.splitlines() if ln.startswith("| `claude")]
    matching = [ln for ln in sig_lines if "THIS RUN" in ln]
    assert len(matching) == 1
    assert "claude_HOSE" in matching[0]


# ---------------------------------------------------------------------------
# 4. record() uses run_signature when run_id not provided
# ---------------------------------------------------------------------------

def test_record_uses_run_signature_for_default_run_id(monkeypatch, tmp_path):
    """When run_id is not passed, it's built from mode/hose_only."""
    monkeypatch.setattr(tracking, "ledger_path", lambda: tmp_path / "predictions.parquet")
    picks = pd.DataFrame([{
        "symbol": "AAA", "rank": 1, "close": 10.0,
        "pred_days": 3.0, "pred_profit": 0.03,
    }])
    tracking.record(picks, mode="base", as_of=pd.Timestamp("2026-05-05"),
                    hose_only=True)
    df = pd.read_parquet(tmp_path / "predictions.parquet")
    assert df.iloc[0]["run_id"] == "20260505_base_HOSE"


def test_record_distinct_runs_coexist(monkeypatch, tmp_path):
    """Same day, different params -> both rows preserved."""
    monkeypatch.setattr(tracking, "ledger_path", lambda: tmp_path / "predictions.parquet")
    picks = pd.DataFrame([{
        "symbol": "AAA", "rank": 1, "close": 10.0,
        "pred_days": 3.0, "pred_profit": 0.03,
    }])
    tracking.record(picks, mode="base", as_of=pd.Timestamp("2026-05-05"))
    tracking.record(picks, mode="base", as_of=pd.Timestamp("2026-05-05"),
                    hose_only=True)
    df = pd.read_parquet(tmp_path / "predictions.parquet")
    assert len(df) == 2
    assert set(df["run_id"]) == {"20260505_base", "20260505_base_HOSE"}


def test_record_same_run_id_replaces(monkeypatch, tmp_path):
    """Same params re-run -> row is replaced (idempotent within the day)."""
    monkeypatch.setattr(tracking, "ledger_path", lambda: tmp_path / "predictions.parquet")
    picks_a = pd.DataFrame([{
        "symbol": "AAA", "rank": 1, "close": 10.0,
        "pred_days": 3.0, "pred_profit": 0.03,
    }])
    picks_b = pd.DataFrame([{
        "symbol": "AAA", "rank": 1, "close": 10.0,
        "pred_days": 5.0, "pred_profit": 0.05,
    }])
    tracking.record(picks_a, mode="base", as_of=pd.Timestamp("2026-05-05"))
    tracking.record(picks_b, mode="base", as_of=pd.Timestamp("2026-05-05"))
    df = pd.read_parquet(tmp_path / "predictions.parquet")
    assert len(df) == 1
    # Latest estimate wins.
    assert abs(float(df.iloc[0]["pred_profit"]) - 0.05) < 1e-9
