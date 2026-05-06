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
        tracking.run_signature("base", 2, 100, hose_only=False),
        tracking.run_signature("base", 2, 200, hose_only=False),
        tracking.run_signature("base", 2, 100, hose_only=True),
        tracking.run_signature("claude", 2, 100, hose_only=False),
        tracking.run_signature("base", 5, 100, hose_only=False),
    }
    assert len(sigs) == 5, f"expected 5 distinct signatures; got {len(sigs)}: {sigs}"


def test_run_signature_idempotent():
    """Same params → same signature."""
    a = tracking.run_signature("claude", 18, 200, hose_only=True)
    b = tracking.run_signature("claude", 18, 200, hose_only=True)
    assert a == b
    assert a == "claude_d18_u200_HOSE"


def test_run_signature_no_hose_tag_when_off():
    """hose_only=False omits the HOSE tag — keeps signatures readable."""
    sig = tracking.run_signature("base", 2, 100, hose_only=False)
    assert "HOSE" not in sig
    assert sig == "base_d2_u100"


# ---------------------------------------------------------------------------
# 3. by-horizon feedback grouping
# ---------------------------------------------------------------------------

def _ledger_with_horizons(rows):
    """Build a ledger DataFrame compatible with tracking._read."""
    return pd.DataFrame(rows, columns=tracking._LEDGER_COLUMNS)


def _row(**kw):
    """Build a ledger row with sensible defaults so tests stay terse."""
    today = pd.Timestamp.today().normalize()
    base = {
        "run_id": "20260420_claude_d2_u100",
        "signature": "claude_d2_u100",
        "as_of": today - pd.Timedelta(days=10),
        "target_date": today - pd.Timedelta(days=8),
        "exit_offset_days": 2,
        "mode": "claude",
        "symbol": "AAA",
        "rank": 1,
        "pred_mean": 0.002,
        "news_score": 0,
        "adjusted": 0.002,
        "entry_price": 10.0,
        "actual_exit": 10.5,
        "realized_return": 0.05,
        "evaluated": True,
    }
    base.update(kw)
    return base


def test_by_horizon_separates_t2_and_t18(monkeypatch, tmp_path):
    """recent_performance.by_horizon has one entry per horizon used."""
    df = _ledger_with_horizons([
        _row(symbol="AAA", realized_return=0.05),
        _row(symbol="BBB", rank=2, pred_mean=0.001, adjusted=0.001,
             entry_price=20.0, actual_exit=19.0, realized_return=-0.05),
        _row(run_id="20260415_claude_d18_u200", signature="claude_d18_u200",
             exit_offset_days=18, symbol="CCC", pred_mean=0.05, news_score=1,
             adjusted=0.0525, entry_price=30.0, actual_exit=33.0,
             realized_return=0.10),
        _row(run_id="20260415_claude_d18_u200", signature="claude_d18_u200",
             exit_offset_days=18, symbol="DDD", rank=2, pred_mean=0.04,
             adjusted=0.04, entry_price=40.0, actual_exit=42.0,
             realized_return=0.05),
        _row(run_id="20260415_claude_d18_u200", signature="claude_d18_u200",
             exit_offset_days=18, symbol="EEE", rank=3, pred_mean=0.03,
             news_score=-1, adjusted=0.0285, entry_price=50.0,
             actual_exit=48.0, realized_return=-0.04),
    ])
    monkeypatch.setattr(tracking, "_read", lambda: df)

    perf = tracking.recent_performance(window_days=90, mode="claude")
    assert "by_horizon" in perf
    assert set(perf["by_horizon"].keys()) == {2, 18}
    assert perf["by_horizon"][2]["n"] == 2
    assert perf["by_horizon"][2]["hit_rate"] == 0.5
    assert perf["by_horizon"][18]["n"] == 3
    assert abs(perf["by_horizon"][18]["hit_rate"] - 2/3) < 1e-9


def test_by_run_signature_separates_param_combos(monkeypatch):
    """Different parameter combos get distinct rows in by_run_signature."""
    df = _ledger_with_horizons([
        _row(signature="claude_d2_u100", symbol="AAA", realized_return=0.05),
        _row(signature="claude_d2_u100", symbol="BBB", rank=2,
             realized_return=-0.05),
        _row(signature="claude_d18_u200_HOSE", exit_offset_days=18,
             symbol="CCC", pred_mean=0.05, adjusted=0.05,
             realized_return=0.10),
        _row(signature="claude_d18_u200_HOSE", exit_offset_days=18,
             symbol="DDD", rank=2, pred_mean=0.04, adjusted=0.04,
             realized_return=0.06),
    ])
    monkeypatch.setattr(tracking, "_read", lambda: df)

    perf = tracking.recent_performance(window_days=90, mode="claude")
    assert "by_run_signature" in perf
    sigs = perf["by_run_signature"]
    assert "claude_d2_u100" in sigs
    assert "claude_d18_u200_HOSE" in sigs
    assert sigs["claude_d2_u100"]["n"] == 2
    assert sigs["claude_d2_u100"]["hit_rate"] == 0.5
    assert sigs["claude_d18_u200_HOSE"]["n"] == 2
    assert sigs["claude_d18_u200_HOSE"]["hit_rate"] == 1.0


def test_feedback_block_marks_current_signature(monkeypatch):
    """The by_run_signature table marks the row whose signature matches."""
    df = _ledger_with_horizons([
        _row(signature="claude_d2_u100", symbol="AAA", realized_return=0.05),
        _row(signature="claude_d18_u200", exit_offset_days=18, symbol="BBB",
             pred_mean=0.05, adjusted=0.05, realized_return=0.10),
    ])
    monkeypatch.setattr(tracking, "_read", lambda: df)

    block = tracking.feedback_block(window_days=90, mode="claude",
                                    current_horizon=18,
                                    current_signature="claude_d18_u200")
    assert "claude_d18_u200" in block
    # Find the signature row that has THIS RUN
    sig_lines = [ln for ln in block.splitlines() if ln.startswith("| `claude_d")]
    matching = [ln for ln in sig_lines if "THIS RUN" in ln]
    assert len(matching) == 1
    assert "claude_d18_u200" in matching[0]


def test_feedback_block_marks_current_horizon(monkeypatch):
    """The by-horizon table highlights the row matching `current_horizon`."""
    df = _ledger_with_horizons([
        _row(symbol="AAA", realized_return=0.005),
        _row(signature="claude_d18_u200", exit_offset_days=18, symbol="BBB",
             pred_mean=0.05, adjusted=0.05, realized_return=0.10),
    ])
    monkeypatch.setattr(tracking, "_read", lambda: df)

    block = tracking.feedback_block(window_days=90, mode="claude",
                                    current_horizon=18)
    assert "T+18" in block
    assert "**THIS RUN**" in block
    t18_line = [ln for ln in block.splitlines() if ln.startswith("| T+18 ")][0]
    t2_line = [ln for ln in block.splitlines() if ln.startswith("| T+2 ")][0]
    assert "THIS RUN" in t18_line
    assert "THIS RUN" not in t2_line


def test_feedback_block_with_unseen_horizon(monkeypatch):
    """If today's horizon has no past data, render an explicit empty row."""
    df = _ledger_with_horizons([
        _row(symbol="AAA", realized_return=0.005),
    ])
    monkeypatch.setattr(tracking, "_read", lambda: df)

    block = tracking.feedback_block(window_days=90, mode="claude",
                                    current_horizon=18)
    assert "no prior history" in block
    assert "T+18" in block


# ---------------------------------------------------------------------------
# 4. record() uses run_signature when run_id not provided
# ---------------------------------------------------------------------------

def test_record_uses_run_signature_for_default_run_id(monkeypatch, tmp_path):
    """When run_id is not passed, it's built from mode/horizon/units/hose_only."""
    monkeypatch.setattr(tracking, "ledger_path", lambda: tmp_path / "predictions.parquet")
    picks = pd.DataFrame([{
        "symbol": "AAA", "pred_mean": 0.01, "rank": 1,
        "close": 10.0, "actionable": False,
    }])
    tracking.record(picks, mode="base", as_of=pd.Timestamp("2026-05-05"),
                    exit_offset_days=18, units=200, hose_only=True)
    df = pd.read_parquet(tmp_path / "predictions.parquet")
    assert df.iloc[0]["run_id"] == "20260505_base_d18_u200_HOSE"


def test_record_distinct_runs_coexist(monkeypatch, tmp_path):
    """Same day, different params → both rows preserved."""
    monkeypatch.setattr(tracking, "ledger_path", lambda: tmp_path / "predictions.parquet")
    picks_a = pd.DataFrame([{
        "symbol": "AAA", "pred_mean": 0.01, "rank": 1,
        "close": 10.0, "actionable": False,
    }])
    picks_b = pd.DataFrame([{
        "symbol": "AAA", "pred_mean": 0.05, "rank": 1,
        "close": 10.0, "actionable": True,
    }])
    tracking.record(picks_a, mode="base", as_of=pd.Timestamp("2026-05-05"),
                    exit_offset_days=2, units=100)
    tracking.record(picks_b, mode="base", as_of=pd.Timestamp("2026-05-05"),
                    exit_offset_days=18, units=200)
    df = pd.read_parquet(tmp_path / "predictions.parquet")
    assert len(df) == 2
    assert set(df["run_id"]) == {"20260505_base_d2_u100", "20260505_base_d18_u200"}


def test_record_same_run_id_replaces(monkeypatch, tmp_path):
    """Same params re-run → row is replaced (idempotent within the day)."""
    monkeypatch.setattr(tracking, "ledger_path", lambda: tmp_path / "predictions.parquet")
    picks_a = pd.DataFrame([{
        "symbol": "AAA", "pred_mean": 0.01, "rank": 1,
        "close": 10.0, "actionable": False,
    }])
    picks_b = pd.DataFrame([{
        "symbol": "AAA", "pred_mean": 0.05, "rank": 1,
        "close": 10.0, "actionable": True,
    }])
    tracking.record(picks_a, mode="base", as_of=pd.Timestamp("2026-05-05"),
                    exit_offset_days=2, units=100)
    tracking.record(picks_b, mode="base", as_of=pd.Timestamp("2026-05-05"),
                    exit_offset_days=2, units=100)  # same params
    df = pd.read_parquet(tmp_path / "predictions.parquet")
    assert len(df) == 1
    # Latest pred_mean wins.
    assert abs(float(df.iloc[0]["pred_mean"]) - 0.05) < 1e-9
