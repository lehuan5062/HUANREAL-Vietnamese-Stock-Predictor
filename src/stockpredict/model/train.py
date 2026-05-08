"""Train the LightGBM ensembles on the long panel of (symbol, date, features, target).

Two heads are trained from the same feature matrix:

* **Mean head** (``TrainedModel``) — regression on ``target`` (forward
  close-to-close return). This is the existing ranker that decides which
  tickers are top-K candidates.
* **Low head** (``LowQuantileModel``) — quantile regression on
  ``target_low`` (next-day low return). Used to predict a realistic
  limit-buy entry price below today's close. The quantile ``alpha`` is
  configurable: alpha=0.5 fills ~50% of the time at the median dip;
  smaller alpha = deeper dip, lower fill rate.
"""
from __future__ import annotations

import datetime as dt
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import load_config, models_dir
from ..dataset import FEATURE_COLS


@dataclass
class TrainedModel:
    boosters: list[lgb.Booster]
    feature_cols: list[str]
    train_end: pd.Timestamp
    train_rows: int

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns DataFrame [pred_mean, pred_std] aligned to X.index."""
        x = X[self.feature_cols].to_numpy(dtype=np.float32)
        preds = np.column_stack([b.predict(x) for b in self.boosters])
        return pd.DataFrame(
            {"pred_mean": preds.mean(axis=1), "pred_std": preds.std(axis=1)},
            index=X.index,
        )

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "TrainedModel":
        with open(path, "rb") as f:
            return pickle.load(f)


@dataclass
class LowQuantileModel:
    """Quantile-regression ensemble that predicts ``low[T+1]/close[T] - 1``.

    ``alpha`` is the quantile level: P(actual_next_low_return <= prediction)
    ≈ alpha. So if you place a limit-buy at ``close * (1 + prediction)``,
    you fill roughly ``alpha`` of the time.

    Stored as a separate pickle (``models/low_latest.pkl``) so it can be
    rebuilt independently of the mean head and so old installs without a
    low model fall through to the close-anchored entry.
    """

    boosters: list[lgb.Booster]
    feature_cols: list[str]
    alpha: float
    train_end: pd.Timestamp
    train_rows: int

    def predict(self, X: pd.DataFrame) -> pd.Series:
        x = X[self.feature_cols].to_numpy(dtype=np.float32)
        preds = np.column_stack([b.predict(x) for b in self.boosters])
        # Mean across seeds so single-seed instability doesn't dominate.
        return pd.Series(preds.mean(axis=1), index=X.index, name="pred_low")

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "LowQuantileModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def _temporal_split(panel: pd.DataFrame, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date so that the validation set is strictly after the train set."""
    dates = sorted(panel.index.unique())
    cutoff = dates[int(len(dates) * (1 - val_frac))]
    return panel[panel.index < cutoff], panel[panel.index >= cutoff]


def train(panel: pd.DataFrame, seeds: Iterable[int] | None = None) -> TrainedModel:
    """Fit an ensemble of LightGBM regressors on the mean ``target``."""
    if panel.empty:
        raise ValueError("empty training panel")
    cfg = load_config().model
    seeds = list(seeds) if seeds is not None else list(cfg["ensemble_seeds"])
    val_frac = float(cfg["validation_fraction"])
    early_stop = int(cfg["early_stopping_rounds"])
    params = dict(cfg["params"])

    train_df, val_df = _temporal_split(panel, val_frac)
    X_tr = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_tr = train_df["target"].to_numpy(dtype=np.float32)
    X_val = val_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_val = val_df["target"].to_numpy(dtype=np.float32)

    boosters: list[lgb.Booster] = []
    for seed in seeds:
        p = dict(params)
        p["seed"] = seed
        n_estimators = int(p.pop("n_estimators", 400))
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=list(FEATURE_COLS))
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain,
                           feature_name=list(FEATURE_COLS))
        booster = lgb.train(
            params=p,
            train_set=dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(early_stop, verbose=False),
                       lgb.log_evaluation(0)],
        )
        boosters.append(booster)

    return TrainedModel(
        boosters=boosters,
        feature_cols=list(FEATURE_COLS),
        train_end=panel.index.max(),
        train_rows=len(panel),
    )


def train_quantile(panel: pd.DataFrame, alpha: float | None = None,
                   seeds: Iterable[int] | None = None,
                   target_col: str = "target_low") -> LowQuantileModel:
    """Fit an ensemble of LightGBM quantile regressors on ``target_col``.

    ``alpha`` defaults to ``pricing.entry_low_alpha`` from config. Three
    seeds by default (vs. five for the mean head) — quantile loss is
    noisier and ensembling more boosters helps less. Early stopping is
    skipped because LightGBM's quantile loss doesn't always provide a
    stable validation metric across versions; we rely on the configured
    ``n_estimators`` (with ``min_child_samples`` already preventing
    overfit).
    """
    if panel.empty:
        raise ValueError("empty training panel")
    if target_col not in panel.columns:
        raise KeyError(f"panel missing target column: {target_col}")

    cfg = load_config()
    if alpha is None:
        alpha = float(cfg.pricing.get("entry_low_alpha", 0.5))
    alpha_f = float(alpha)
    if not 0.0 < alpha_f < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha_f}")

    model_cfg = cfg.model
    seeds = list(seeds) if seeds is not None else [11, 22, 33]
    val_frac = float(model_cfg["validation_fraction"])
    params = dict(model_cfg["params"])
    # Override loss for quantile regression. Replace any previous
    # objective/metric so we don't accidentally inherit the regression
    # MSE metric and confuse early-stopping callbacks downstream.
    params["objective"] = "quantile"
    params["alpha"] = alpha_f
    params.pop("metric", None)

    train_df, _val_df = _temporal_split(panel, val_frac)
    X_tr = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_tr = train_df[target_col].to_numpy(dtype=np.float32)

    boosters: list[lgb.Booster] = []
    for seed in seeds:
        p = dict(params)
        p["seed"] = seed
        n_estimators = int(p.pop("n_estimators", 400))
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=list(FEATURE_COLS))
        booster = lgb.train(
            params=p,
            train_set=dtrain,
            num_boost_round=n_estimators,
            callbacks=[lgb.log_evaluation(0)],
        )
        boosters.append(booster)

    return LowQuantileModel(
        boosters=boosters,
        feature_cols=list(FEATURE_COLS),
        alpha=alpha_f,
        train_end=panel.index.max(),
        train_rows=len(panel),
    )


def latest_model_path() -> Path:
    return models_dir() / "latest.pkl"


def latest_low_model_path() -> Path:
    return models_dir() / "low_latest.pkl"


def save_latest(model: TrainedModel) -> Path:
    p = latest_model_path()
    model.save(p)
    stamped = models_dir() / f"model_{dt.date.today().isoformat()}.pkl"
    model.save(stamped)
    return p


def save_latest_low(model: LowQuantileModel) -> Path:
    p = latest_low_model_path()
    model.save(p)
    stamped = models_dir() / f"low_model_{dt.date.today().isoformat()}_a{int(model.alpha * 100):03d}.pkl"
    model.save(stamped)
    return p
