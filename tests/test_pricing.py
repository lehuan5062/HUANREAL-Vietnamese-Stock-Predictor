"""Verify the pricing module produces correct VND-scale outputs and that the
ACBS fee model + per-share P&L line up with hand-computed values."""
import pandas as pd

from stockpredict.pricing import add_price_suggestions


def _frame(rows):
    return pd.DataFrame(rows)


def test_low_predicted_return_is_not_actionable():
    """A tiny predicted return fails the directional edge gate (pred_mean must
    clear the round-trip cost), so the pick is NOT actionable — even though the
    ATR-scaled target now gives it a healthy rr. This is the typical case for
    our model output."""
    df = _frame([{
        "close": 15.35,            # thousand VND
        "pred_mean": 0.0017,
        "pred_std": 0.0001,
        "atr_14": 0.30,
    }])
    out = add_price_suggestions(df).iloc[0]
    # Per-share VND prices. Target is ATR-scaled: entry + 2 × ATR.
    assert out["entry_vnd"] == 15_350
    assert out["target_vnd"] == round((15.35 + 2.0 * 0.30) * 1000)   # 15,950
    assert out["stop_vnd"] == round((15.35 - 1.5 * 0.30) * 1000)     # 14,900
    # Gross reward per share = 2 × ATR = 600
    assert out["gross_reward_vnd"] == 600
    # rr is healthy now (reward/risk ≈ 2/1.5), but...
    assert out["rr_ratio"] >= 1.0
    # ...the forecast (0.17%) is below breakeven (~0.43%), so the quality bar
    # flags it. Low signal ⇒ below_breakeven.
    assert float(out["pred_mean"]) < float(out["breakeven_pct"])
    assert bool(out["below_breakeven"])


def test_strong_predicted_return_is_actionable():
    """A 5% predicted return with the same stop is large enough to clear
    fees + risk gate."""
    df = _frame([{
        "close": 20.0,
        "pred_mean": 0.05,
        "pred_std": 0.005,
        "atr_14": 0.40,
    }])
    out = add_price_suggestions(df).iloc[0]
    assert out["entry_vnd"] == 20_000
    # Target is ATR-scaled: entry + 2 × ATR = 20,000 + 800 = 20,800
    assert out["target_vnd"] == 20_800
    assert out["stop_vnd"] == 19_400
    # Gross reward per share: 2 × ATR = 800
    assert out["gross_reward_vnd"] == 800
    # Fees per share still ~0.4% of the per-share trade value: order of ~90
    assert out["fees_round_trip_vnd"] < 200
    # Net should be comfortably positive
    assert out["net_reward_vnd"] > 600
    # rr ratio: net_reward / net_loss; net_loss = (20-19.4)*1000 + fees
    # = 600 + ~90 = ~690; rr ~= 910/690 ~= 1.3
    assert out["rr_ratio"] >= 1.0
    assert not bool(out["below_breakeven"])


def test_quality_flag_marks_forecast_below_cost_despite_good_rr():
    """A pick with a healthy ATR-scaled rr but a forecast that barely beats
    zero (below breakeven) is flagged below_breakeven. Selection is exactly-N,
    so it'd still be returned — but the flag warns it's a weak-edge pick."""
    df = _frame([{
        "close": 20.0,
        "pred_mean": 0.002,    # +0.2%, below the ~0.43% breakeven
        "pred_std": 0.001,
        "atr_14": 0.40,
    }])
    out = add_price_suggestions(df).iloc[0]
    # rr is fine (ATR-scaled reward vs ATR-scaled risk)...
    assert out["rr_ratio"] >= 1.0
    # ...but the forecast doesn't clear the cost, so it's flagged weak.
    assert float(out["pred_mean"]) < float(out["breakeven_pct"])
    assert bool(out["below_breakeven"])


def test_breakeven_pct_around_43bps():
    """ACBS round-trip cost ~ 0.43% (0.30% commission + 0.03% VAT + 0.10% PIT)."""
    df = _frame([{
        "close": 50.0,
        "pred_mean": 0.0,
        "pred_std": 0.0,
        "atr_14": 1.0,
    }])
    out = add_price_suggestions(df).iloc[0]
    # breakeven_pct rounded to 4 decimals — should be ~ 0.0043
    assert 0.0040 < float(out["breakeven_pct"]) < 0.0046


def test_missing_atr_marks_rr_nan_and_below_breakeven():
    """If ATR is NaN we can't compute the stop, so rr is undefined and the
    pick is flagged below_breakeven (invalid trade economics)."""
    import numpy as np
    df = _frame([{
        "close": 15.0,
        "pred_mean": 0.01,
        "pred_std": 0.001,
        "atr_14": np.nan,
    }])
    out = add_price_suggestions(df).iloc[0]
    assert pd.isna(out["rr_ratio"])
    assert bool(out["below_breakeven"])


def test_vnd_columns_are_integers():
    """Order amounts must round to whole VND so the user can paste into a broker."""
    df = _frame([{"close": 7.123, "pred_mean": 0.01,
                  "pred_std": 0.002, "atr_14": 0.123}])
    out = add_price_suggestions(df).iloc[0]
    for col in ("entry_vnd", "target_vnd", "stop_vnd",
                "gross_reward_vnd",
                "fees_round_trip_vnd", "net_reward_vnd"):
        v = out[col]
        # pandas Int64 - either int or pandas NA
        assert pd.isna(v) or int(v) == v
