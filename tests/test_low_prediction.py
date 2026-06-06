"""Tests for the low-prediction (limit-buy entry) head and its
integration with pricing, the ledger, and evaluate_pending.

The low head produces ``pred_low`` — a quantile prediction of
``low[T+1]/close[T] - 1``. Pricing turns this into ``entry_vnd`` (the
limit-buy price), the ledger records the quoted limit, and
``evaluate_pending`` stamps whether it actually filled when the buy day
closed.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stockpredict import tracking
from stockpredict.data import cache as cache_mod
from stockpredict.model.target import (
    attach_target,
    next_day_low_return,
)
from stockpredict.model.train import (
    RollingEmpiricalQuantileModel,
    derive_lookback,
)
from stockpredict.pricing import add_price_suggestions


# ---------------------------------------------------------------------------
# 1. next_day_low_return target
# ---------------------------------------------------------------------------

def test_next_day_low_return_basic():
    """target[T] = low[T+1] / close[T] - 1, with NaN at the last row."""
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    df = pd.DataFrame({
        "close": [100.0, 102.0, 101.0, 105.0],
        "low":   [ 99.0,  98.0, 100.0, 104.0],
    }, index=idx)
    y = next_day_low_return(df)
    # T=0: low[1]/close[0] - 1 = 98/100 - 1 = -0.02
    # T=1: low[2]/close[1] - 1 = 100/102 - 1
    # T=2: low[3]/close[2] - 1 = 104/101 - 1
    # T=3: NaN (no T+1)
    assert np.isclose(y.iloc[0], -0.02)
    assert np.isclose(y.iloc[1], 100 / 102 - 1)
    assert np.isclose(y.iloc[2], 104 / 101 - 1)
    assert np.isnan(y.iloc[3])


def test_attach_target_produces_target_low_when_low_present():
    """attach_target adds target_low when 'low' is in the OHLCV frame."""
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    df = pd.DataFrame({
        "open":  [100, 102, 101, 105, 110],
        "high":  [102, 103, 106, 111, 112],
        "low":   [ 99,  98, 100, 104, 109],
        "close": [101, 102, 105, 110, 111],
    }, index=idx, dtype=float)
    out = attach_target(df, exit_offset_days=2)
    assert "target" in out.columns
    assert "target_low" in out.columns
    # T=0: low[1]/close[0] - 1 = 98/101 - 1
    assert np.isclose(out["target_low"].iloc[0], 98 / 101 - 1)


def test_attach_target_skips_target_low_without_low_column():
    """If 'low' is missing (degenerate fixture), target_low is omitted."""
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    df = pd.DataFrame({"close": [100.0, 102.0, 105.0]}, index=idx)
    out = attach_target(df, exit_offset_days=1)
    assert "target" in out.columns
    assert "target_low" not in out.columns


# ---------------------------------------------------------------------------
# 1b. RollingEmpiricalQuantileModel — per-ticker empirical low head
# ---------------------------------------------------------------------------

def _emp_model(alpha=0.25, lookback=60, min_obs=5, global_quantile=-0.03):
    return RollingEmpiricalQuantileModel(
        alpha=alpha, target_tail_obs=min_obs, lookback=lookback,
        min_obs=min_obs, global_quantile=global_quantile,
        train_end=pd.Timestamp("2026-01-01"), train_rows=1000,
    )


def test_derive_lookback_scales_with_alpha():
    """Window auto-sizes so the tail keeps ~target_tail_obs observations,
    bounded by [30, 120]."""
    assert derive_lookback(0.25, 15) == 60     # 15/0.25 = 60
    assert derive_lookback(0.50, 15) == 30     # 15/0.50 = 30 (floor)
    assert derive_lookback(0.10, 15) == 120    # 15/0.10 = 150 → cap 120
    assert derive_lookback(0.90, 15) == 30     # tiny → floor


def test_empirical_predict_uses_per_ticker_history():
    """pred_low is the alpha-quantile of the ticker's OWN recent target_low."""
    idx = pd.date_range("2026-01-01", periods=40, freq="B")
    # 39 known dips of -2%, then the as-of row.
    tl = [-0.02] * 39 + [np.nan]
    history = pd.DataFrame({"symbol": "AAA", "target_low": tl}, index=idx)
    history.index.name = "date"
    snap = history.iloc[[-1]].copy()  # one row, as-of = last date
    model = _emp_model(alpha=0.25, lookback=60, min_obs=5)
    out = model.predict(snap, history=history)
    assert np.isclose(out.iloc[0], -0.02)


def test_empirical_predict_runner_quotes_near_zero():
    """A name that has not been dipping (all positive next-low returns, i.e. a
    gap-up runner) yields a non-negative quantile → pricing will clip to close."""
    idx = pd.date_range("2026-01-01", periods=40, freq="B")
    tl = [0.05] * 39 + [np.nan]   # never dipped — opened up every day
    history = pd.DataFrame({"symbol": "DST", "target_low": tl}, index=idx)
    history.index.name = "date"
    snap = history.iloc[[-1]].copy()
    out = _emp_model(min_obs=5).predict(snap, history=history)
    assert out.iloc[0] >= 0.0   # clipped to close downstream in pricing


def test_empirical_predict_falls_back_to_global_when_thin():
    """Too few observations → pooled global_quantile, not a per-ticker guess."""
    idx = pd.date_range("2026-01-01", periods=4, freq="B")
    tl = [-0.02, -0.02, -0.02, np.nan]   # only 3 usable obs < min_obs=5
    history = pd.DataFrame({"symbol": "NEW", "target_low": tl}, index=idx)
    history.index.name = "date"
    snap = history.iloc[[-1]].copy()
    out = _emp_model(min_obs=5, global_quantile=-0.031).predict(snap, history=history)
    assert np.isclose(out.iloc[0], -0.031)


def test_empirical_predict_is_lookahead_safe():
    """The as-of row's own target_low (which uses low[T+1]) must never enter the
    window: only observations strictly before the as-of date are used."""
    idx = pd.date_range("2026-01-01", periods=10, freq="B")
    # Early history dips -1%; the as-of-day's own (future) value is a huge -50%
    # that, if leaked, would dominate the quantile.
    tl = [-0.01] * 9 + [-0.50]
    history = pd.DataFrame({"symbol": "AAA", "target_low": tl}, index=idx)
    history.index.name = "date"
    snap = history.iloc[[-1]].copy()    # as-of = last date, value -0.50
    out = _emp_model(alpha=0.25, lookback=60, min_obs=3).predict(snap, history=history)
    # Window excludes the -0.50 leak → quantile sits at -0.01, not near -0.50.
    assert np.isclose(out.iloc[0], -0.01)


# ---------------------------------------------------------------------------
# 2. Pricing — entry uses pred_low when present
# ---------------------------------------------------------------------------

def test_pricing_uses_pred_low_when_present():
    """When pred_low is present, entry_vnd = close × (1 + pred_low) × 1000;
    stop is anchored on the limit so risk is exactly stop_atr_mult × ATR."""
    df = pd.DataFrame([{
        "close": 20.0,           # 20,000 VND
        "pred_mean": 0.05,
        "pred_std": 0.005,
        "atr_14": 0.40,
        "pred_low": -0.012,      # predicts a 1.2% dip on T+1
    }])
    out = add_price_suggestions(df).iloc[0]
    # Entry = 20 × (1 - 0.012) × 1000 = 19,760
    assert int(out["entry_vnd"]) == 19_760
    # Reference close stays available
    assert int(out["close_vnd"]) == 20_000
    # Stop = entry - 1.5 × ATR = 19.76 - 0.6 = 19.16 → 19,160
    assert int(out["stop_vnd"]) == 19_160
    # Target unchanged: 20 × 1.05 × 1000 = 21,000
    assert int(out["target_vnd"]) == 21_000
    # Limit-pct surfaced
    assert abs(float(out["entry_limit_pct"]) - (-0.012)) < 1e-9


def test_pricing_clips_positive_pred_low_to_zero():
    """pred_low > 0 means the model thinks T+1 will gap up — never quote a
    limit ABOVE today's close. Clip at 0 so entry_vnd stays at close."""
    df = pd.DataFrame([{
        "close": 20.0,
        "pred_mean": 0.03,
        "pred_std": 0.003,
        "atr_14": 0.40,
        "pred_low": +0.005,      # would imply entry above close — must clip
    }])
    out = add_price_suggestions(df).iloc[0]
    assert int(out["entry_vnd"]) == 20_000
    assert int(out["close_vnd"]) == 20_000
    assert float(out["entry_limit_pct"]) == 0.0


def test_pricing_falls_back_to_close_when_pred_low_absent():
    """No pred_low column → entry_vnd = close × 1000 (legacy behavior)."""
    df = pd.DataFrame([{
        "close": 15.35,
        "pred_mean": 0.0017,
        "pred_std": 0.0001,
        "atr_14": 0.30,
    }])
    out = add_price_suggestions(df).iloc[0]
    assert int(out["entry_vnd"]) == 15_350
    assert int(out["close_vnd"]) == 15_350
    assert float(out["entry_limit_pct"]) == 0.0


def test_pricing_recomputes_rr_against_limit_entry():
    """Filling at a cheaper limit makes net_reward_vnd LARGER than the same
    trade at close: same target, lower entry."""
    base_kwargs = {"close": 20.0, "pred_mean": 0.05,
                   "pred_std": 0.005, "atr_14": 0.40}
    no_low = add_price_suggestions(pd.DataFrame([base_kwargs])).iloc[0]
    with_low = add_price_suggestions(pd.DataFrame([{**base_kwargs, "pred_low": -0.01}])).iloc[0]
    # gross_reward at close-entry: (21000 - 20000) × 100 = 100,000
    # gross_reward at limit-entry (-1%): (21000 - 19800) × 100 = 120,000
    assert int(with_low["gross_reward_vnd"]) > int(no_low["gross_reward_vnd"])
    assert int(with_low["gross_reward_vnd"]) == 120_000


# ---------------------------------------------------------------------------
# 3. Ledger — record() captures pred_low + entry_limit_price
# ---------------------------------------------------------------------------

def test_record_captures_pred_low_and_entry_limit_price(monkeypatch, tmp_path):
    """record() reads pred_low off picks rows and persists the quoted
    limit price (= close × (1 + pred_low))."""
    monkeypatch.setattr(tracking, "ledger_path",
                        lambda: tmp_path / "predictions.parquet")
    picks = pd.DataFrame([{
        "symbol": "AAA",
        "rank": 1,
        "pred_mean": 0.03,
        "pred_low": -0.012,
        "close": 20.0,
        "news_score": 0,
        "adjusted": 0.03,
    }])
    n = tracking.record(picks, mode="claude",
                        as_of=pd.Timestamp("2026-05-05"),
                        exit_offset_days=2, units=100)
    assert n == 1
    df = tracking._read()
    assert df.iloc[0]["pred_low"] == -0.012
    # entry_limit_price = 20 × (1 - 0.012) = 19.76
    assert abs(df.iloc[0]["entry_limit_price"] - 19.76) < 1e-9
    assert bool(df.iloc[0]["entry_limit_filled"]) is False
    assert bool(df.iloc[0]["t0_evaluated"]) is False


def test_record_no_pred_low_leaves_limit_nan(monkeypatch, tmp_path):
    """If picks lack pred_low (low head not trained), entry_limit_price
    stays NaN and entry_limit_filled stays False — the row is invisible
    to limit-fill stats."""
    monkeypatch.setattr(tracking, "ledger_path",
                        lambda: tmp_path / "predictions.parquet")
    picks = pd.DataFrame([{
        "symbol": "AAA", "rank": 1, "pred_mean": 0.03,
        "close": 20.0, "news_score": 0, "adjusted": 0.03,
    }])
    tracking.record(picks, mode="base",
                    as_of=pd.Timestamp("2026-05-05"),
                    exit_offset_days=2, units=100)
    df = tracking._read()
    assert pd.isna(df.iloc[0]["pred_low"])
    assert pd.isna(df.iloc[0]["entry_limit_price"])


# ---------------------------------------------------------------------------
# 4. evaluate_pending — T+0 limit-fill stamping (independent of T+N)
# ---------------------------------------------------------------------------

def test_evaluate_pending_stamps_t0_fill_before_target_date(monkeypatch, tmp_path):
    """The buy day closes (May 6 = as_of) but T+N (May 8) hasn't elapsed
    yet. evaluate_pending should stamp t0_evaluated + entry_limit_filled
    while leaving evaluated=False. Under the corrected semantic, ``as_of``
    IS the buy day; ``entry_price`` is the close from the data anchor
    (May 5)."""
    df_ohlcv = pd.DataFrame({
        "open":  [10.0,  9.8],
        "high":  [10.2, 10.0],
        "low":   [ 9.9,  9.5],   # buy-day low 9.5 is below the limit 9.88
        "close": [10.0,  9.9],
        "volume":[1000, 1100],
    }, index=pd.DatetimeIndex([
        pd.Timestamp("2026-05-05"),  # data anchor (as_of - 1)
        pd.Timestamp("2026-05-06"),  # buy day = as_of
    ], name="date"))
    monkeypatch.setattr(cache_mod, "read_ohlcv", lambda s: df_ohlcv)
    monkeypatch.setattr(tracking, "read_ohlcv", lambda s: df_ohlcv)
    monkeypatch.setattr(tracking, "ledger_path",
                        lambda: tmp_path / "predictions.parquet")

    pending = pd.DataFrame([{
        "run_id": "20260506_claude_d2_u100", "signature": "claude_d2_u100",
        "as_of": pd.Timestamp("2026-05-06"),       # buy day = as_of
        "target_date": pd.Timestamp("2026-05-08"),  # T+N hasn't elapsed
        "exit_offset_days": 2, "mode": "claude", "symbol": "AAA", "rank": 1,
        "pred_mean": 0.05, "news_score": 0, "adjusted": 0.05,
        "entry_price": 10.0, "actual_exit": np.nan, "realized_return": np.nan,
        "evaluated": False, "t0_open": np.nan, "t0_low": np.nan,
        "t0_close": np.nan, "entry_slippage": np.nan,
        "dimensions_cited": "",
        "pred_low": -0.012, "entry_limit_price": 9.88,
        "entry_limit_filled": False, "t0_evaluated": False,
    }])
    pending.to_parquet(tmp_path / "predictions.parquet", index=False)

    # Run with today = May 7 — buy day closed, target_date hasn't.
    updated = tracking.evaluate_pending(today=dt.date(2026, 5, 7))
    assert len(updated) == 1
    row = updated.iloc[0]
    # T+0 stamping happened
    assert bool(row["t0_evaluated"]) is True
    assert row["t0_low"] == 9.5
    # 9.5 <= 9.88 → limit filled
    assert bool(row["entry_limit_filled"]) is True
    # T+N stamping did NOT happen (target_date still in the future)
    assert bool(row["evaluated"]) is False
    assert pd.isna(row["realized_return"])


def test_evaluate_pending_marks_unfilled_limit(monkeypatch, tmp_path):
    """Buy day's low is ABOVE the quoted limit → entry_limit_filled=False.
    Buy day = as_of (May 6); its low 10.8 is above the 9.88 limit, so the
    limit-buy never executes."""
    df_ohlcv = pd.DataFrame({
        "open":  [10.0, 11.0],
        "high":  [10.2, 11.5],
        "low":   [ 9.9, 10.8],   # May 6 (= as_of) low 10.8 > limit 9.88 → no fill
        "close": [10.0, 11.4],
        "volume":[1000, 1100],
    }, index=pd.DatetimeIndex([
        pd.Timestamp("2026-05-05"),  # data anchor
        pd.Timestamp("2026-05-06"),  # buy day = as_of (gap-up)
    ], name="date"))
    monkeypatch.setattr(cache_mod, "read_ohlcv", lambda s: df_ohlcv)
    monkeypatch.setattr(tracking, "read_ohlcv", lambda s: df_ohlcv)
    monkeypatch.setattr(tracking, "ledger_path",
                        lambda: tmp_path / "predictions.parquet")

    pending = pd.DataFrame([{
        "run_id": "20260506_claude_d2_u100", "signature": "claude_d2_u100",
        "as_of": pd.Timestamp("2026-05-06"),  # buy day = as_of
        "target_date": pd.Timestamp("2026-05-08"),
        "exit_offset_days": 2, "mode": "claude", "symbol": "AAA", "rank": 1,
        "pred_mean": 0.05, "news_score": 0, "adjusted": 0.05,
        "entry_price": 10.0, "actual_exit": np.nan, "realized_return": np.nan,
        "evaluated": False, "t0_open": np.nan, "t0_low": np.nan,
        "t0_close": np.nan, "entry_slippage": np.nan,
        "dimensions_cited": "",
        "pred_low": -0.012, "entry_limit_price": 9.88,
        "entry_limit_filled": False, "t0_evaluated": False,
    }])
    pending.to_parquet(tmp_path / "predictions.parquet", index=False)

    updated = tracking.evaluate_pending(today=dt.date(2026, 5, 7))
    row = updated.iloc[0]
    assert bool(row["t0_evaluated"]) is True
    assert bool(row["entry_limit_filled"]) is False


def test_evaluate_pending_handles_both_stages_in_one_pass(monkeypatch, tmp_path):
    """When today is past target_date, a single call stamps T+0 AND T+N."""
    df_ohlcv = pd.DataFrame({
        "open":  [10.0,  9.8, 10.2, 10.4],
        "high":  [10.2, 10.0, 10.5, 10.6],
        "low":   [ 9.9,  9.5, 10.0, 10.3],
        "close": [10.0,  9.9, 10.4, 10.5],
        "volume":[1000, 1100, 1200, 1300],
    }, index=pd.DatetimeIndex([
        pd.Timestamp("2026-05-05"),  # data anchor
        pd.Timestamp("2026-05-06"),  # buy day = as_of
        pd.Timestamp("2026-05-07"),
        pd.Timestamp("2026-05-08"),  # target_date
    ], name="date"))
    monkeypatch.setattr(cache_mod, "read_ohlcv", lambda s: df_ohlcv)
    monkeypatch.setattr(tracking, "read_ohlcv", lambda s: df_ohlcv)
    monkeypatch.setattr(tracking, "ledger_path",
                        lambda: tmp_path / "predictions.parquet")

    pending = pd.DataFrame([{
        "run_id": "20260506_claude_d2_u100", "signature": "claude_d2_u100",
        "as_of": pd.Timestamp("2026-05-06"),  # buy day = as_of
        "target_date": pd.Timestamp("2026-05-08"),
        "exit_offset_days": 2, "mode": "claude", "symbol": "AAA", "rank": 1,
        "pred_mean": 0.05, "news_score": 0, "adjusted": 0.05,
        "entry_price": 10.0, "actual_exit": np.nan, "realized_return": np.nan,
        "evaluated": False, "t0_open": np.nan, "t0_low": np.nan,
        "t0_close": np.nan, "entry_slippage": np.nan,
        "dimensions_cited": "",
        "pred_low": -0.012, "entry_limit_price": 9.88,
        "entry_limit_filled": False, "t0_evaluated": False,
    }])
    pending.to_parquet(tmp_path / "predictions.parquet", index=False)

    updated = tracking.evaluate_pending(today=dt.date(2026, 5, 9))
    row = updated.iloc[0]
    # Both stages set in one pass
    assert bool(row["t0_evaluated"]) is True
    assert bool(row["evaluated"]) is True
    assert bool(row["entry_limit_filled"]) is True   # 9.5 <= 9.88
    # realized_return = 10.5 / 10.0 - 1 = 0.05 (still computed off entry_price = close)
    assert abs(row["realized_return"] - 0.05) < 1e-9


# ---------------------------------------------------------------------------
# 5. recent_performance — limit_fill stats
# ---------------------------------------------------------------------------

def _row_with_limit(pred_low, entry_limit_price, entry_limit_filled,
                    t0_low, **kw):
    today = pd.Timestamp.today().normalize()
    base = {
        "run_id": "20260420_claude_d2_u100",
        "signature": "claude_d2_u100",
        "as_of": today - pd.Timedelta(days=10),
        "target_date": today - pd.Timedelta(days=8),
        "exit_offset_days": 2, "mode": "claude", "symbol": "AAA", "rank": 1,
        "pred_mean": 0.01, "news_score": 0, "adjusted": 0.01,
        "entry_price": 10.0, "actual_exit": 10.5, "realized_return": 0.05,
        "evaluated": True, "t0_open": 10.0, "t0_low": t0_low,
        "t0_close": 10.0, "entry_slippage": (t0_low - 10.0) / 10.0,
        "dimensions_cited": "",
        "pred_low": pred_low,
        "entry_limit_price": entry_limit_price,
        "entry_limit_filled": entry_limit_filled,
        "t0_evaluated": True,
    }
    base.update(kw)
    return base


def test_recent_performance_includes_limit_fill_stats(monkeypatch):
    """limit_fill is computed only over rows with both an entry_limit_price
    and t0_evaluated=True. Filled vs unfilled is averaged into fill_rate."""
    df = pd.DataFrame([
        _row_with_limit(-0.012, 9.88, True, 9.5),    # filled (dip −5%)
        _row_with_limit(-0.010, 9.90, True, 9.7, symbol="BBB"),  # filled
        _row_with_limit(-0.008, 9.92, False, 10.0, symbol="CCC"),  # unfilled
        _row_with_limit(-0.015, 9.85, True, 9.8, symbol="DDD"),  # filled
    ], columns=tracking._LEDGER_COLUMNS)
    monkeypatch.setattr(tracking, "_read", lambda: df)

    perf = tracking.recent_performance(window_days=90, mode="claude")
    fill = perf["limit_fill"]
    assert fill is not None
    assert fill["n"] == 4
    # 3 of 4 filled
    assert abs(fill["fill_rate"] - 0.75) < 1e-9
    # mean(pred_low) = (-0.012 - 0.010 - 0.008 - 0.015) / 4 = -0.01125
    assert abs(fill["mean_dip_quoted"] - (-0.01125)) < 1e-9


def test_feedback_block_renders_limit_fill_section(monkeypatch):
    """When limit_fill stats exist, feedback_block includes a labelled
    section the next Claude run can read."""
    df = pd.DataFrame([
        _row_with_limit(-0.012, 9.88, True, 9.5),
        _row_with_limit(-0.008, 9.92, False, 10.0, symbol="BBB"),
    ], columns=tracking._LEDGER_COLUMNS)
    monkeypatch.setattr(tracking, "_read", lambda: df)
    block = tracking.feedback_block(window_days=90, mode="claude",
                                    current_horizon=2)
    assert "Limit-buy fill calibration" in block
    assert "fill_rate" in block


def test_feedback_block_omits_limit_fill_when_no_data(monkeypatch):
    """Old ledgers with no entry_limit_price should not get a half-baked
    section."""
    today = pd.Timestamp.today().normalize()
    df = pd.DataFrame([{
        "run_id": "20260420_claude_d2_u100",
        "signature": "claude_d2_u100",
        "as_of": today - pd.Timedelta(days=10),
        "target_date": today - pd.Timedelta(days=8),
        "exit_offset_days": 2, "mode": "claude", "symbol": "AAA", "rank": 1,
        "pred_mean": 0.01, "news_score": 0, "adjusted": 0.01,
        "entry_price": 10.0, "actual_exit": 10.5, "realized_return": 0.05,
        "evaluated": True, "t0_open": np.nan, "t0_low": np.nan,
        "t0_close": np.nan, "entry_slippage": np.nan,
        "dimensions_cited": "",
        "pred_low": np.nan, "entry_limit_price": np.nan,
        "entry_limit_filled": False, "t0_evaluated": True,
    }], columns=tracking._LEDGER_COLUMNS)
    monkeypatch.setattr(tracking, "_read", lambda: df)
    block = tracking.feedback_block(window_days=90, mode="claude",
                                    current_horizon=2)
    assert "Limit-buy fill calibration" not in block
