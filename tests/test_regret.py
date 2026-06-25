"""Tests for the missed-winners ("regret") analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockpredict.analyze import regret


def _panel():
    """Two buy-days, several symbols. DEAD has the biggest target but is
    illiquid (must be excluded by liquidity_mask so it isn't a 'missed winner')."""
    d1, d2 = pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-02")
    rows = [
        # date, symbol, target, close, adv_active_days_20
        (d1, "WIN",  0.08, 20.0, 20),   # liquid big winner
        (d1, "MEH",  0.01, 20.0, 20),   # liquid small
        (d1, "DEAD", 0.20, 20.0,  3),   # illiquid huge gainer -> excluded
        (d2, "WIN",  0.05, 20.0, 20),
        (d2, "CAP",  0.04, 20.0, 20),   # liquid, will be in the ledger
        (d2, "MEH", -0.02, 20.0, 20),
    ]
    idx = pd.DatetimeIndex([r[0] for r in rows], name="date")
    return pd.DataFrame(
        {"symbol": [r[1] for r in rows], "target": [r[2] for r in rows],
         "close": [r[3] for r in rows], "adv_active_days_20": [r[4] for r in rows]},
        index=idx,
    )


def _ledger():
    """The model surfaced CAP on d2 (captured) but never surfaced WIN."""
    return pd.DataFrame({
        "as_of": [pd.Timestamp("2026-06-02")],
        "symbol": ["CAP"], "rank": [1], "pred_mean": [0.01],
        "realized_return": [0.04], "evaluated": [True], "signature": ["base_d2"],
    })


def test_realized_top_n_applies_liquidity():
    """The illiquid huge gainer DEAD must NOT appear as a winner."""
    top = regret.realized_top_n(_panel(), n=2)
    assert "DEAD" not in set(top["symbol"])
    d1 = top[top["as_of"] == pd.Timestamp("2026-06-01")]
    assert list(d1.sort_values("realized_rank")["symbol"]) == ["WIN", "MEH"]


def test_missed_winners_flags_unsurfaced():
    """On d2 the model surfaced CAP (captured) but missed WIN."""
    mw = regret.missed_winners(_panel(), _ledger(), n=2)
    # Only d2 is a ledger run-day, so only d2 winners are considered.
    assert set(mw["as_of"].unique()) == {pd.Timestamp("2026-06-02")}
    by = dict(zip(mw["symbol"], mw["missed"]))
    assert by["WIN"] is True or bool(by["WIN"]) is True
    assert bool(by["CAP"]) is False


def test_aggregate_regret_math():
    a = regret.aggregate_regret(window_days=0, n=2, panel=_panel(), ledger=_ledger())
    assert a["n_winner_rows"] == 2          # d2 top-2 liquid: WIN, CAP
    assert abs(a["miss_rate"] - 0.5) < 1e-9  # 1 of 2 missed
    assert a["mean_missed_target"] is not None and a["mean_captured_target"] is not None


def test_missed_winner_weights_upweights_winners():
    w = regret.missed_winner_weights(_panel(), n=2, upweight=3.0)
    assert len(w) == len(_panel())
    # WIN on both days and MEH/CAP as top-2 get 3.0; DEAD (illiquid) stays 1.0.
    p = _panel()
    dead_mask = (p["symbol"] == "DEAD").to_numpy()
    assert np.allclose(w.to_numpy()[dead_mask], 1.0)
    assert (w > 1.0).sum() == 4             # WIN,MEH (d1) + WIN,CAP (d2)
