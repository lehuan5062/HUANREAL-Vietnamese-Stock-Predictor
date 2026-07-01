"""Rebound recovery targets + Kaplan-Meier estimator."""
import numpy as np
import pandas as pd

from stockpredict.model.target import recovery_episode, resolve_exit
from stockpredict.model.train import _km_curve, _km_summarize


def _frame(closes):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    return pd.DataFrame({"close": closes}, index=idx)


def test_recovery_first_profitable_day():
    # entry 100; thr 0.02 -> profitable when close >= 102.
    # closes: 100, 99, 101, 103, ...  first day >=102 for T=0 is index 3 (k=3).
    df = _frame([100.0, 99.0, 101.0, 103.0, 105.0])
    rec = recovery_episode(df, thr=0.02, max_horizon=10)
    assert bool(rec["target_recovered"].iloc[0]) is True
    assert rec["target_days_to_recover"].iloc[0] == 3
    assert np.isclose(rec["target_recovery_return"].iloc[0], 103 / 100 - 1)


def test_recovery_no_lookahead_and_censoring():
    # Monotonically falling: nothing ever recovers -> every row censored, with
    # censoring time = available forward bars (data-edge censoring).
    df = _frame([100.0, 99.0, 98.0, 97.0, 96.0])
    rec = recovery_episode(df, thr=0.01, max_horizon=10)
    assert not rec["target_recovered"].any()
    # row 0 has 4 future bars, row 3 has 1, last row has 0.
    assert rec["target_days_to_recover"].iloc[0] == 4
    assert rec["target_days_to_recover"].iloc[3] == 1
    assert rec["target_days_to_recover"].iloc[4] == 0
    # profit is undefined for censored rows.
    assert rec["target_recovery_return"].isna().all()


def test_recovery_band_break_censors_scan():
    # A >15% jump at index 2 is a phantom corporate action; row 0 must NOT be
    # counted as recovered off that fake spike. It recovers only if a legit day
    # before the break clears the bar — here index 1 (101) does not clear 102,
    # so row 0 is censored at the break (k=2 -> censor time 1).
    df = _frame([100.0, 101.0, 130.0, 131.0])
    rec = recovery_episode(df, thr=0.02, max_horizon=10)
    assert bool(rec["target_recovered"].iloc[0]) is False
    assert rec["target_days_to_recover"].iloc[0] == 1


def test_km_curve_and_summary():
    times = np.array([1, 2, 2, 3, 5])
    events = np.array([True, True, True, False, True])
    curve = _km_curve(times, events)
    # S(1)=0.8, S(2)=0.4, S(3)=0.4 (censor), S(5)=0.
    assert np.isclose(curve[0, 1], 0.8)
    assert np.isclose(curve[1, 1], 0.4)
    prob, days = _km_summarize(times, events)
    assert np.isclose(prob, 1.0)
    assert days == 2.0  # first t with S <= 0.5


def test_km_all_censored_uses_rmst():
    prob, days = _km_summarize(np.array([3, 4, 5]), np.array([False, False, False]))
    assert prob == 0.0
    assert days == 5.0  # restricted-mean fallback (area under S=1 to t=5)


# ---------------------------------------------------------------------------
# resolve_exit: hold until recovery (no stop, no cap)
# ---------------------------------------------------------------------------

def test_exit_recovery_when_it_bounces():
    # entry 100, thr 2% -> recover at >=102. closes: 101, 103 -> recovery day 2.
    ex = resolve_exit([101.0, 103.0], entry=100.0, thr=0.02)
    assert ex["reason"] == "recovery"
    assert ex["k"] == 2
    assert ex["exit_close"] == 103.0


def test_exit_open_when_never_recovers():
    # Falling knife: never clears the profit target -> still open (None),
    # regardless of how far it drops. No stop to cut it.
    ex = resolve_exit([90.0, 84.0, 80.0, 70.0], entry=100.0, thr=0.02)
    assert ex is None


def test_exit_open_when_unresolved():
    # No recovery within the available data -> still open (None).
    ex = resolve_exit([99.0, 100.0], entry=100.0, thr=0.02)
    assert ex is None
