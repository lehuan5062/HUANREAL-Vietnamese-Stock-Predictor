"""The weighted-train path (missed-winners variant) runs and stays
backward-compatible when weights=None."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockpredict.dataset import FEATURE_COLS
from stockpredict.model.train import TrainedModel, train


def _panel(n_dates=40, n_syms=6):
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    rng = np.random.RandomState(0)
    rows = []
    idx = []
    for d in dates:
        for s in range(n_syms):
            feat = {c: float(rng.randn()) for c in FEATURE_COLS}
            feat["symbol"] = f"S{s}"
            feat["target"] = float(rng.randn() * 0.01)
            rows.append(feat)
            idx.append(d)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="date"))


def test_weights_none_trains():
    m = train(_panel(), seeds=[11])
    assert isinstance(m, TrainedModel)
    assert len(m.boosters) == 1


def test_weights_path_runs_same_shape():
    panel = _panel()
    w = pd.Series(1.0, index=panel.index)
    w.iloc[: len(w) // 4] = 3.0          # upweight a chunk
    m = train(panel, seeds=[11], weights=w)
    assert isinstance(m, TrainedModel)
    assert len(m.boosters) == 1
    # Predicts a finite pred_mean for the feature matrix.
    out = m.predict(panel.head(5))
    assert out["pred_mean"].notna().all()
