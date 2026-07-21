"""RecoveryKMModel._lookup fallback cascade + train_recovery's ticker x bucket
cells (state-aware tier inserted ahead of the flat per-ticker aggregate)."""
import numpy as np
import pandas as pd

from stockpredict import filters as filters_mod
from stockpredict.model import train as train_mod
from stockpredict.model.train import RecoveryKMModel, train_recovery


def _stat(prob, days=3.0, profit=0.05, n=100):
    return {"recovery_prob": prob, "days": days, "profit": profit, "n": n}


def _base_model(**overrides):
    kwargs = dict(
        buckets={}, pooled=_stat(0.44, n=9999),
        rsi_edges=[30, 40, 50], high_prox_edges=[-0.20, -0.10, -0.05],
        p_quantile=0.5, min_bucket_obs=50,
        train_end=pd.Timestamp("2024-01-01"), train_rows=0,
        min_ticker_obs=100, min_ticker_bucket_obs=30,
    )
    kwargs.update(overrides)
    return RecoveryKMModel(**kwargs)


# rsi=45 -> ri=2 (edges [30,40,50]); high_prox=-0.15 -> hi=1; high_prox=-0.25 -> hi=0.
_RSI = 45.0
_HP_CELL = -0.15   # bucket (2, 1)
_HP_KNIFE = -0.25  # bucket (2, 0)


def test_lookup_prefers_ticker_bucket_cell_over_aggregate():
    model = _base_model(
        ticker_bucket_stats={("AAA", 2, 1): _stat(0.11, n=40)},
        ticker_stats={"AAA": _stat(0.22, n=150)},
        buckets={(2, 1): _stat(0.33, n=60)},
    )
    got = model._lookup("AAA", _RSI, _HP_CELL)
    assert got["recovery_prob"] == 0.11  # most specific tier wins


def test_lookup_falls_back_to_aggregate_when_cell_thin():
    model = _base_model(
        ticker_bucket_stats={("AAA", 2, 1): _stat(0.11, n=10)},  # < min_ticker_bucket_obs
        ticker_stats={"AAA": _stat(0.22, n=150)},
        buckets={(2, 1): _stat(0.33, n=60)},
    )
    got = model._lookup("AAA", _RSI, _HP_CELL)
    assert got["recovery_prob"] == 0.22


def test_lookup_falls_back_to_cross_sectional_bucket_when_ticker_thin():
    model = _base_model(
        ticker_bucket_stats={("AAA", 2, 1): _stat(0.11, n=10)},
        ticker_stats={"AAA": _stat(0.22, n=5)},  # < min_ticker_obs
        buckets={(2, 1): _stat(0.33, n=60)},
    )
    got = model._lookup("AAA", _RSI, _HP_CELL)
    assert got["recovery_prob"] == 0.33


def test_lookup_falls_back_to_pooled_when_everything_thin():
    model = _base_model(
        ticker_bucket_stats={("AAA", 2, 1): _stat(0.11, n=10)},
        ticker_stats={"AAA": _stat(0.22, n=5)},
        buckets={(2, 1): _stat(0.33, n=1)},  # < min_bucket_obs
    )
    got = model._lookup("AAA", _RSI, _HP_CELL)
    assert got["recovery_prob"] == 0.44


def test_lookup_skips_cell_and_bucket_on_nan_state():
    model = _base_model(
        ticker_bucket_stats={("AAA", 2, 1): _stat(0.11, n=40)},
        ticker_stats={"AAA": _stat(0.22, n=150)},
        buckets={(2, 1): _stat(0.33, n=60)},
    )
    got = model._lookup("AAA", float("nan"), float("nan"))
    assert got["recovery_prob"] == 0.22  # cell/bucket both need finite state


def test_lookup_backward_compatible_without_new_fields():
    """A model built the old way (no ticker_bucket_stats/min_ticker_bucket_obs
    kwargs, as tests/test_rebound_rank.py::_model already does) must still
    work via the getattr defaults in _lookup."""
    model = RecoveryKMModel(
        buckets={(2, 1): _stat(0.33, n=60)}, pooled=_stat(0.44, n=9999),
        rsi_edges=[30, 40, 50], high_prox_edges=[-0.20, -0.10, -0.05],
        p_quantile=0.5, min_bucket_obs=50,
        train_end=pd.Timestamp("2024-01-01"), train_rows=0,
        ticker_stats={"AAA": _stat(0.22, n=150)}, min_ticker_obs=100,
    )
    assert model._lookup("AAA", _RSI, _HP_CELL)["recovery_prob"] == 0.22
    # ZZZ has no ticker history at all -> falls to the cross-sectional bucket
    # (which is present here), not straight to pooled.
    assert model._lookup("ZZZ", _RSI, _HP_CELL)["recovery_prob"] == 0.33


# ---------------------------------------------------------------------------
# train_recovery: builds ticker x bucket cells that can disagree with the
# ticker's own flat aggregate.
# ---------------------------------------------------------------------------

def _split_panel(n_benign=40, n_knife=40):
    """One symbol, SPLIT: n_benign rows in the benign cell (rsi 45, high_prox
    -0.15 -> bucket (2,1)), all recovered quickly; n_knife rows in the knife
    cell (rsi 45, high_prox -0.25 -> bucket (2,0)), all censored (never
    recovered). The blended per-ticker aggregate should land near 0.5 while
    the two cells disagree sharply (~1.0 vs ~0.0)."""
    rows = []
    dates = pd.date_range("2020-01-01", periods=n_benign + n_knife, freq="B")
    di = iter(dates)
    for _ in range(n_benign):
        rows.append({
            "date": next(di), "symbol": "SPLIT", "mom_20": -0.10,
            "rsi_14": _RSI, "high_prox_20": _HP_CELL,
            "target_days_to_recover": 2.0, "target_recovered": True,
            "target_recovery_return": 0.05,
        })
    for _ in range(n_knife):
        rows.append({
            "date": next(di), "symbol": "SPLIT", "mom_20": -0.10,
            "rsi_14": _RSI, "high_prox_20": _HP_KNIFE,
            "target_days_to_recover": 50.0, "target_recovered": False,
            "target_recovery_return": float("nan"),
        })
    return pd.DataFrame(rows).set_index("date")


def _fake_cfg(min_ticker_bucket_obs):
    return type("Cfg", (), {"strategy": {
        "downtrend": {"mom20_max": -0.03, "high_prox_max": -0.05,
                      "rsi_floor": 10, "rsi_ceil": 50},
        "recovery": {"p_quantile": 0.5,
                     "state_buckets": {"rsi_edges": [30, 40, 50],
                                       "high_prox_edges": [-0.20, -0.10, -0.05]},
                     "min_bucket_obs": 50, "min_ticker_obs": 100,
                     "min_ticker_bucket_obs": min_ticker_bucket_obs}}})()


def test_train_recovery_ticker_bucket_cells_disagree_with_aggregate(monkeypatch):
    panel = _split_panel(n_benign=40, n_knife=40)
    cfg = _fake_cfg(min_ticker_bucket_obs=30)
    monkeypatch.setattr(train_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(filters_mod, "load_config", lambda: cfg)

    model = train_recovery(panel)

    benign = model.ticker_bucket_stats[("SPLIT", 2, 1)]
    knife = model.ticker_bucket_stats[("SPLIT", 2, 0)]
    aggregate = model.ticker_stats["SPLIT"]
    assert benign["n"] == 40 and knife["n"] == 40
    assert benign["recovery_prob"] == 1.0
    assert knife["recovery_prob"] == 0.0
    assert 0.3 < aggregate["recovery_prob"] < 0.7  # blended, disagrees with both cells

    # predict() now moves with state instead of returning the flat aggregate
    # for both rows.
    X = pd.DataFrame({"symbol": ["SPLIT", "SPLIT"], "rsi_14": [_RSI, _RSI],
                      "high_prox_20": [_HP_CELL, _HP_KNIFE]})
    preds = model.predict(X)
    assert preds["pred_recovery_prob"].iloc[0] == 1.0
    assert preds["pred_recovery_prob"].iloc[1] == 0.0


def test_train_recovery_thin_cell_defers_to_aggregate(monkeypatch):
    # n_benign=90 so the per-ticker AGGREGATE (n=100) also clears the default
    # min_ticker_obs=100 in _fake_cfg -- otherwise it would fall past the
    # aggregate too and this test would (by coincidence, single-symbol panel)
    # land on a pooled value that happens to equal the aggregate by value but
    # not by identity.
    panel = _split_panel(n_benign=90, n_knife=10)  # knife cell below the threshold
    cfg = _fake_cfg(min_ticker_bucket_obs=30)
    monkeypatch.setattr(train_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(filters_mod, "load_config", lambda: cfg)

    model = train_recovery(panel)

    assert model.ticker_bucket_stats[("SPLIT", 2, 0)]["n"] == 10
    assert model.ticker_stats["SPLIT"]["n"] == 100
    got = model._lookup("SPLIT", _RSI, _HP_KNIFE)
    aggregate = model.ticker_stats["SPLIT"]
    assert got is aggregate  # thin cell (n=10 < 30) defers to the flat aggregate
