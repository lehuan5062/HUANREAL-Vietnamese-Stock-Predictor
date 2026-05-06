"""Verify the pricing module produces correct VND-scale outputs and that the
ACBS fee model + 100-unit position math line up with hand-computed values."""
import pandas as pd

from stockpredict.pricing import add_price_suggestions


def _frame(rows):
    return pd.DataFrame(rows)


def test_low_predicted_return_is_not_actionable():
    """Tiny predicted return + ATR-based stop -> rr near zero, net negative,
    actionable=False. This is the typical case for our model output."""
    df = _frame([{
        "close": 15.35,            # thousand VND
        "pred_mean": 0.0017,
        "pred_std": 0.0001,
        "atr_14": 0.30,
    }])
    out = add_price_suggestions(df).iloc[0]
    # Per-share VND prices
    assert out["entry_vnd"] == 15_350
    assert out["target_vnd"] == round(15_350 * (1 + 0.0017))      # 15376
    assert out["stop_vnd"] == round((15.35 - 1.5 * 0.30) * 1000)   # 14,900
    # 100-unit position
    assert out["position_units"] == 100
    assert out["position_value_vnd"] == 1_535_000
    # Gross reward: (15376 - 15350) * 100 = 2,600
    assert out["gross_reward_vnd"] == 2_600
    # Fees: buy commission 0.15% on 1.535M + VAT 10% + sell commission on
    # ~target value + sell PIT 0.10% on target value. Roughly 6,600.
    assert 6_000 < out["fees_round_trip_vnd"] < 7_000
    # Net is negative — model's predicted return doesn't cover ACBS fees.
    assert out["net_reward_vnd"] < 0
    assert not bool(out["actionable"])


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
    assert out["target_vnd"] == 21_000
    assert out["stop_vnd"] == 19_400
    # Gross reward: (21000-20000) * 100 = 100,000
    assert out["gross_reward_vnd"] == 100_000
    # Fees still ~0.4% of the ~2M trade value: order of 9k-10k
    assert out["fees_round_trip_vnd"] < 15_000
    # Net should be comfortably positive
    assert out["net_reward_vnd"] > 80_000
    # rr ratio: net_reward / net_loss; net_loss = (20-19.4)*1000*100 + fees
    # = 60,000 + ~9,000 = ~69k; rr ~= 90k/69k ~= 1.3
    assert out["rr_ratio"] >= 1.0
    assert bool(out["actionable"])


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


def test_missing_atr_marks_rr_nan_not_actionable():
    """If ATR is NaN we can't compute the stop, so rr is undefined and we
    should never flag the trade as actionable."""
    import numpy as np
    df = _frame([{
        "close": 15.0,
        "pred_mean": 0.01,
        "pred_std": 0.001,
        "atr_14": np.nan,
    }])
    out = add_price_suggestions(df).iloc[0]
    assert pd.isna(out["rr_ratio"])
    assert not bool(out["actionable"])


def test_vnd_columns_are_integers():
    """Order amounts must round to whole VND so the user can paste into a broker."""
    df = _frame([{"close": 7.123, "pred_mean": 0.01,
                  "pred_std": 0.002, "atr_14": 0.123}])
    out = add_price_suggestions(df).iloc[0]
    for col in ("entry_vnd", "target_vnd", "stop_vnd",
                "position_value_vnd", "gross_reward_vnd",
                "fees_round_trip_vnd", "net_reward_vnd"):
        v = out[col]
        # pandas Int64 - either int or pandas NA
        assert pd.isna(v) or int(v) == v
