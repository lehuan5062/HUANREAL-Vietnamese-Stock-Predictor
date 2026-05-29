"""Budget mode: size each pick to a per-pick VND budget instead of a fixed
share count. Covers the pricing conversion, the over_budget flag (over-budget
picks are kept at the 100-share minimum, never dropped), the run-signature
token, the CLI sizing resolver, and the rank_today no-drop integration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpredict import cli, tracking
from stockpredict.model import predict as predict_mod
from stockpredict.pricing import add_price_suggestions


def _frame(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. pricing: budget -> whole-lot units, equal capital per pick
# ---------------------------------------------------------------------------

def test_budget_sizes_each_pick_to_whole_lots():
    """close=20 -> entry 20,000 VND. A 5,000,000 budget buys 250 shares,
    floored to whole 100-share lots = 200 (a 4,000,000 VND position)."""
    df = _frame([{"symbol": "AAA", "close": 20.0, "pred_mean": 0.05,
                  "pred_std": 0.005, "atr_14": 0.4}])
    out = add_price_suggestions(df, budget_vnd=5_000_000).iloc[0]
    assert int(out["position_units"]) == 200
    assert int(out["position_value_vnd"]) == 4_000_000
    assert not bool(out["over_budget"])


def test_budget_exact_lot_boundary_is_not_over_budget():
    """A budget that exactly equals one lot's cost fits -> over_budget False."""
    df = _frame([{"symbol": "AAA", "close": 20.0, "pred_mean": 0.05,
                  "pred_std": 0.005, "atr_14": 0.4}])
    out = add_price_suggestions(df, budget_vnd=2_000_000).iloc[0]  # 100 * 20,000
    assert int(out["position_units"]) == 100
    assert int(out["position_value_vnd"]) == 2_000_000
    assert not bool(out["over_budget"])


def test_over_budget_pick_kept_at_min_lot_and_flagged():
    """close=200 -> entry 200,000 VND. One lot (100 shares) costs 20,000,000,
    above a 5,000,000 budget. The pick is NOT dropped: it's shown at the
    100-share minimum, flagged over_budget, and still actionable on quality."""
    df = _frame([{"symbol": "AAA", "close": 200.0, "pred_mean": 0.05,
                  "pred_std": 0.005, "atr_14": 2.0}])
    out = add_price_suggestions(df, budget_vnd=5_000_000).iloc[0]
    assert int(out["position_units"]) == 100        # min lot, never 0 / dropped
    assert int(out["position_value_vnd"]) == 20_000_000
    assert bool(out["over_budget"]) is True
    # Trade quality is independent of affordability — a real pick you'd need a
    # bigger budget for must still read actionable so the user can see it.
    assert bool(out["actionable"]) is True


def test_budget_etf_symbol_floors_to_100():
    """ETF symbols (FUE* / E1VFVN30) now use 100-share lots like stocks."""
    df = _frame([{"symbol": "FUEVFVND", "close": 15.0, "pred_mean": 0.05,
                  "pred_std": 0.005, "atr_14": 0.3}])
    out = add_price_suggestions(df, budget_vnd=5_000_000).iloc[0]  # 333 -> 300
    assert int(out["position_units"]) == 300
    assert int(out["position_units"]) % 100 == 0


def test_units_mode_unchanged_and_over_budget_false():
    """Passing --units behaves exactly as before; over_budget is always False."""
    df = _frame([{"symbol": "AAA", "close": 20.0, "pred_mean": 0.05,
                  "pred_std": 0.005, "atr_14": 0.4}])
    out = add_price_suggestions(df, units=300).iloc[0]
    assert int(out["position_units"]) == 300
    assert not bool(out["over_budget"])


# ---------------------------------------------------------------------------
# 2. run_signature: b{VND} token
# ---------------------------------------------------------------------------

def test_run_signature_budget_token():
    sig = tracking.run_signature("base", 2, budget_vnd=2_000_000)
    assert sig == "base_d2_b2000000"


def test_run_signature_budget_distinct_from_units():
    u = tracking.run_signature("base", 2, 100)
    b = tracking.run_signature("base", 2, budget_vnd=2_000_000)
    assert u == "base_d2_u100"
    assert u != b


def test_run_signature_budget_idempotent_with_flags():
    a = tracking.run_signature("claude", 5, budget_vnd=5_000_000, hose_only=True)
    b = tracking.run_signature("claude", 5, budget_vnd=5_000_000, hose_only=True)
    assert a == b == "claude_d5_b5000000_HOSE"


# ---------------------------------------------------------------------------
# 3. CLI sizing resolver: mutual exclusion + defaults
# ---------------------------------------------------------------------------

def test_resolve_sizing_defaults_to_100_units():
    assert cli._resolve_sizing(None, None) == (100, None)


def test_resolve_sizing_rounds_units_down_to_lot():
    assert cli._resolve_sizing(250, None) == (200, None)


def test_resolve_sizing_budget_mode():
    assert cli._resolve_sizing(None, 2_000_000) == (None, 2_000_000)


def test_resolve_sizing_rejects_both():
    with pytest.raises(SystemExit):
        cli._resolve_sizing(100, 2_000_000)


def test_resolve_sizing_rejects_sub_lot_units():
    with pytest.raises(SystemExit):
        cli._resolve_sizing(50, None)


def test_resolve_sizing_rejects_nonpositive_budget():
    with pytest.raises(SystemExit):
        cli._resolve_sizing(None, 0)


# ---------------------------------------------------------------------------
# 4. rank_today: budget mode keeps every pick (no drop) and flags over-budget
# ---------------------------------------------------------------------------

class _FakeMeanModel:
    def predict(self, snap):
        n = len(snap)
        return {"pred_mean": np.full(n, 0.05), "pred_std": np.full(n, 0.005)}


def test_rank_today_budget_keeps_and_flags_over_budget(monkeypatch):
    """A cheap and a pricey ticker under a 5,000,000 VND budget: both survive
    (nothing dropped), the cheap one is sized to budget, the pricey one is
    flagged over_budget at the 100-share minimum."""
    end = pd.Timestamp("2026-05-27")
    panel = pd.DataFrame(
        [
            {"symbol": "CHEAP", "close": 10.0, "atr_14": 0.3},
            {"symbol": "PRICEY", "close": 200.0, "atr_14": 5.0},
        ],
        index=pd.DatetimeIndex([end, end], name="date"),
    )
    monkeypatch.setattr(predict_mod, "tradable_symbols", lambda: None)
    monkeypatch.setattr(predict_mod, "build_panel", lambda **kw: panel)
    monkeypatch.setattr(predict_mod, "liquidity_mask",
                        lambda df: pd.Series(True, index=df.index))
    # Don't load a real low-quantile model from disk (its predict() needs the
    # full feature matrix); keep entry == close for this focused test.
    monkeypatch.setattr(predict_mod, "_try_load_low_model", lambda: None)

    out = predict_mod.rank_today(model=_FakeMeanModel(), top_k=10,
                                 budget_vnd=5_000_000, low_model=None)
    assert set(out["symbol"].astype(str)) == {"CHEAP", "PRICEY"}  # nothing dropped
    assert "over_budget" in out.columns
    row = out.set_index("symbol")
    # CHEAP: 5,000,000 / 10,000 = 500 shares -> fits the budget
    assert int(row.loc["CHEAP", "position_units"]) == 500
    assert not bool(row.loc["CHEAP", "over_budget"])
    # PRICEY: one lot (100 * 200,000 = 20,000,000) exceeds budget -> flagged
    assert int(row.loc["PRICEY", "position_units"]) == 100
    assert bool(row.loc["PRICEY", "over_budget"]) is True
