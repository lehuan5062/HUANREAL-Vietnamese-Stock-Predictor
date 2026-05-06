"""Train a LightGBM ensemble on the long panel of (symbol, date, features, target)."""
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


def _temporal_split(panel: pd.DataFrame, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date so that the validation set is strictly after the train set."""
    dates = sorted(panel.index.unique())
    cutoff = dates[int(len(dates) * (1 - val_frac))]
    return panel[panel.index < cutoff], panel[panel.index >= cutoff]


def train(panel: pd.DataFrame, seeds: Iterable[int] | None = None) -> TrainedModel:
    """Fit an ensemble of LightGBM regressors. Returns a TrainedModel."""
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


def latest_model_path() -> Path:
    return models_dir() / "latest.pkl"


def save_latest(model: TrainedModel) -> Path:
    p = latest_model_path()
    model.save(p)
    stamped = models_dir() / f"model_{dt.date.today().isoformat()}.pkl"
    model.save(stamped)
    return p
