"""Verify entry-slippage tracking: evaluate_pending stamps the buy-day
OHLC against each evaluated row, _read backfills old ledgers, and
feedback_block surfaces the slippage stats.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stockpredict import tracking
from stockpredict.data import cache as cache_mod


# ---------------------------------------------------------------------------
# 1. _read backfill — old ledgers without the new columns stay readable
# ---------------------------------------------------------------------------

def test_read_backfills_missing_slippage_columns(monkeypatch, tmp_path):
    """An old ledger written before the entry-slippage release must be
    readable, with the four new columns appended as NaN."""
    monkeypatch.setattr(tracking, "ledger_path", lambda: tmp_path / "predictions.parquet")
    # Write an "old" ledger missing every later-added column (both the
    # entry-slippage floats AND the dimensions_cited string).
    legacy_cols = [c for c in tracking._LEDGER_COLUMNS
                   if c not in tracking._NEW_FLOAT_COLUMNS
                   and c not in tracking._NEW_STRING_COLUMNS]
    today = pd.Timestamp("2026-05-05").normalize()
    legacy = pd.DataFrame([{
        "run_id": "20260505_claude_d2_u100",
        "signature": "claude_d2_u100",
        "as_of": today,
        "target_date": today + pd.Timedelta(days=2),
        "exit_offset_days": 2,
        "mode": "claude",
        "symbol": "AAA",
        "rank": 1,
        "pred_mean": 0.01,
        "news_score": 0,
        "adjusted": 0.01,
        "entry_price": 10.0,
        "actual_exit": np.nan,
        "realized_return": np.nan,
        "evaluated": False,
    }])[legacy_cols]
    legacy.to_parquet(tmp_path / "predictions.parquet", index=False)

    # _read should load it and append the four new columns as NaN.
    df = tracking._read()
    for col in tracking._NEW_FLOAT_COLUMNS:
        assert col in df.columns, f"missing backfilled column: {col}"
        assert df[col].isna().all(), f"{col} should be NaN-filled for legacy rows"
    # Sanity: the legacy row content survived.
    assert df.iloc[0]["symbol"] == "AAA"
    assert df.iloc[0]["entry_price"] == 10.0


# ---------------------------------------------------------------------------
# 2. evaluate_pending stamps the T+0 bar
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_ohlcv(monkeypatch):
    """Pre-cooked OHLCV with a known buy-day bar so we can check
    entry_slippage gets computed correctly."""
    df = pd.DataFrame({
        "open":  [10.0, 9.8, 10.2, 10.4],
        "high":  [10.2, 10.0, 10.5, 10.6],
        "low":   [9.9, 9.5, 10.0, 10.3],   # buy day (May 6) low = 9.5
        "close": [10.0, 9.9, 10.4, 10.5],
        "volume":[1000, 1100, 1200, 1300],
    }, index=pd.DatetimeIndex([
        pd.Timestamp("2026-05-05"),  # as_of (predicted entry = 10.0)
        pd.Timestamp("2026-05-06"),  # T+0  buy day, low 9.5 (cheaper!)
        pd.Timestamp("2026-05-07"),  # T+1
        pd.Timestamp("2026-05-08"),  # T+2  target_date  close 10.5
    ], name="date"))

    def _fake_read(symbol):
        return df

    monkeypatch.setattr(cache_mod, "read_ohlcv", _fake_read)
    monkeypatch.setattr(tracking, "read_ohlcv", _fake_read)
    return df


def test_evaluate_pending_stamps_buy_day_bar(monkeypatch, tmp_path, fake_ohlcv):
    """A pending row evaluates to (a) realized_return based on entry_price
    -> actual_exit, AND (b) the T+0 OHLC from the buy day."""
    monkeypatch.setattr(tracking, "ledger_path",
                        lambda: tmp_path / "predictions.parquet")
    pending = pd.DataFrame([{
        "run_id": "20260505_claude_d2_u100",
        "signature": "claude_d2_u100",
        "as_of": pd.Timestamp("2026-05-05"),
        "target_date": pd.Timestamp("2026-05-08"),
        "exit_offset_days": 2,
        "mode": "claude",
        "symbol": "AAA",
        "rank": 1,
        "pred_mean": 0.05,
        "news_score": 0,
        "adjusted": 0.05,
        "entry_price": 10.0,        # predicted entry = May 5 close
        "actual_exit": np.nan,
        "realized_return": np.nan,
        "evaluated": False,
        "t0_open": np.nan,
        "t0_low": np.nan,
        "t0_close": np.nan,
        "entry_slippage": np.nan,
    }])
    pending.to_parquet(tmp_path / "predictions.parquet", index=False)

    updated = tracking.evaluate_pending(today=dt.date(2026, 5, 9))
    assert len(updated) == 1
    row = updated.iloc[0]

    # Realized: 10.5 / 10.0 - 1 = 0.05
    assert abs(row["realized_return"] - 0.05) < 1e-9
    # Buy day OHLC stamped from May 6:
    assert row["t0_open"] == 9.8
    assert row["t0_low"] == 9.5
    assert row["t0_close"] == 9.9
    # entry_slippage = (9.5 - 10.0) / 10.0 = -0.05 (we could have bought 5% cheaper)
    assert abs(row["entry_slippage"] - (-0.05)) < 1e-9


def test_evaluate_pending_marks_unreachable_entry(monkeypatch, tmp_path):
    """If t0_low > entry_price, the predicted entry was never available and
    entry_slippage is positive (meaning the realized_return is fictional)."""
    df_ohlcv = pd.DataFrame({
        "open":  [10.0, 11.0, 11.5, 12.0],
        "high":  [10.2, 11.5, 12.0, 12.5],
        "low":   [9.9, 10.8, 11.4, 11.8],   # buy-day low 10.8 > predicted 10.0
        "close": [10.0, 11.4, 11.8, 12.3],
        "volume":[1000, 1100, 1200, 1300],
    }, index=pd.DatetimeIndex([
        pd.Timestamp("2026-05-05"),
        pd.Timestamp("2026-05-06"),  # gap-up day
        pd.Timestamp("2026-05-07"),
        pd.Timestamp("2026-05-08"),
    ], name="date"))
    monkeypatch.setattr(cache_mod, "read_ohlcv", lambda s: df_ohlcv)
    monkeypatch.setattr(tracking, "read_ohlcv", lambda s: df_ohlcv)
    monkeypatch.setattr(tracking, "ledger_path",
                        lambda: tmp_path / "predictions.parquet")

    pending = pd.DataFrame([{
        "run_id": "20260505_claude_d2_u100", "signature": "claude_d2_u100",
        "as_of": pd.Timestamp("2026-05-05"),
        "target_date": pd.Timestamp("2026-05-08"),
        "exit_offset_days": 2, "mode": "claude", "symbol": "AAA", "rank": 1,
        "pred_mean": 0.05, "news_score": 0, "adjusted": 0.05,
        "entry_price": 10.0, "actual_exit": np.nan, "realized_return": np.nan,
        "evaluated": False, "t0_open": np.nan, "t0_low": np.nan,
        "t0_close": np.nan, "entry_slippage": np.nan,
    }])
    pending.to_parquet(tmp_path / "predictions.parquet", index=False)

    updated = tracking.evaluate_pending(today=dt.date(2026, 5, 9))
    row = updated.iloc[0]
    # entry_slippage = (10.8 - 10.0)/10.0 = 0.08 > 0 → unreachable
    assert row["entry_slippage"] > 0
    assert abs(row["entry_slippage"] - 0.08) < 1e-9


# ---------------------------------------------------------------------------
# 3. recent_performance.entry_slippage stats
# ---------------------------------------------------------------------------

def _row_with_slippage(slippage, **kw):
    """Helper: build a complete ledger row with the given slippage."""
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
        "pred_mean": 0.01,
        "news_score": 0,
        "adjusted": 0.01,
        "entry_price": 10.0,
        "actual_exit": 10.5,
        "realized_return": 0.05,
        "evaluated": True,
        "t0_open": 10.0,
        "t0_low": 10.0 + 10.0 * slippage,
        "t0_close": 10.0,
        "entry_slippage": slippage,
    }
    base.update(kw)
    return base


def test_recent_performance_includes_slippage_stats(monkeypatch):
    df = pd.DataFrame([
        _row_with_slippage(-0.02),  # could've bought 2% cheaper
        _row_with_slippage(-0.01, symbol="BBB"),
        _row_with_slippage(+0.005, symbol="CCC"),  # unreachable
        _row_with_slippage(0.0, symbol="DDD"),     # exact fill
    ], columns=tracking._LEDGER_COLUMNS)
    monkeypatch.setattr(tracking, "_read", lambda: df)

    perf = tracking.recent_performance(window_days=90, mode="claude")
    slip = perf["entry_slippage"]
    assert slip is not None
    assert slip["n"] == 4
    # Mean of -0.02, -0.01, +0.005, 0 = -0.00625
    assert abs(slip["mean"] - (-0.00625)) < 1e-9
    # 1 of 4 had slippage > 0
    assert abs(slip["pct_unreachable"] - 0.25) < 1e-9
    # Reachable rows: -0.02, -0.01, 0 → savings (positive numbers): 0.02, 0.01, 0
    # Mean savings when reachable = (0.02 + 0.01 + 0) / 3 = 0.01
    assert abs(slip["mean_savings_when_reachable"] - 0.01) < 1e-9


def test_recent_performance_slippage_none_when_no_data(monkeypatch):
    """If every row has NaN entry_slippage (e.g. legacy ledger that hasn't
    been re-evaluated), the stats key should be None — feedback_block then
    omits the slippage section entirely."""
    df = pd.DataFrame([{
        "run_id": "20260420_claude_d2_u100",
        "signature": "claude_d2_u100",
        "as_of": pd.Timestamp.today().normalize() - pd.Timedelta(days=10),
        "target_date": pd.Timestamp.today().normalize() - pd.Timedelta(days=8),
        "exit_offset_days": 2, "mode": "claude", "symbol": "AAA", "rank": 1,
        "pred_mean": 0.01, "news_score": 0, "adjusted": 0.01,
        "entry_price": 10.0, "actual_exit": 10.5, "realized_return": 0.05,
        "evaluated": True, "t0_open": np.nan, "t0_low": np.nan,
        "t0_close": np.nan, "entry_slippage": np.nan,
    }], columns=tracking._LEDGER_COLUMNS)
    monkeypatch.setattr(tracking, "_read", lambda: df)

    perf = tracking.recent_performance(window_days=90, mode="claude")
    assert perf["entry_slippage"] is None


# ---------------------------------------------------------------------------
# 4. feedback_block rendering
# ---------------------------------------------------------------------------

def test_feedback_block_renders_slippage_section(monkeypatch):
    df = pd.DataFrame([
        _row_with_slippage(-0.02),
        _row_with_slippage(+0.01, symbol="BBB"),
    ], columns=tracking._LEDGER_COLUMNS)
    monkeypatch.setattr(tracking, "_read", lambda: df)

    block = tracking.feedback_block(window_days=90, mode="claude",
                                    current_horizon=2)
    assert "Entry-execution sanity check" in block
    assert "mean entry_slippage" in block
    # 1 of 2 unreachable = 50%
    assert "50.0%" in block


def test_feedback_block_omits_slippage_when_no_data(monkeypatch):
    """Legacy ledgers without entry_slippage should not get a half-baked
    slippage section in the feedback block."""
    df = pd.DataFrame([{
        "run_id": "20260420_claude_d2_u100",
        "signature": "claude_d2_u100",
        "as_of": pd.Timestamp.today().normalize() - pd.Timedelta(days=10),
        "target_date": pd.Timestamp.today().normalize() - pd.Timedelta(days=8),
        "exit_offset_days": 2, "mode": "claude", "symbol": "AAA", "rank": 1,
        "pred_mean": 0.01, "news_score": 0, "adjusted": 0.01,
        "entry_price": 10.0, "actual_exit": 10.5, "realized_return": 0.05,
        "evaluated": True, "t0_open": np.nan, "t0_low": np.nan,
        "t0_close": np.nan, "entry_slippage": np.nan,
    }], columns=tracking._LEDGER_COLUMNS)
    monkeypatch.setattr(tracking, "_read", lambda: df)

    block = tracking.feedback_block(window_days=90, mode="claude",
                                    current_horizon=2)
    assert "Entry-execution sanity check" not in block
