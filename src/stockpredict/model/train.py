"""Train the LightGBM ensembles on the long panel of (symbol, date, features, target).

Two heads are trained from the same feature matrix:

* **Mean head** (``TrainedModel``) — regression on ``target`` (forward
  close-to-close return). This is the existing ranker that decides which
  tickers are top-K candidates.
* **Low head** (``RollingEmpiricalQuantileModel``) — a per-ticker rolling
  empirical quantile of ``target_low`` (next-day low return). Used to
  predict a realistic limit-buy entry price below today's close. Each
  ticker is judged on its OWN recent dip distribution, so a name that has
  not been dipping lately (a momentum runner) quotes an entry near today's
  close (reachable) rather than an unfillable deep dip. The quantile
  ``alpha`` is configurable: alpha=0.5 ≈ median dip (fills ~half the time);
  smaller alpha = deeper dip, lower fill rate.

  This replaced a LightGBM quantile-regression head that had *negative*
  predictive skill (it quoted its deepest dips on names that gapped up,
  placing unreachable limits on exactly the wrong tickers). See
  ``reports/self_correction_2026-06-05_*_stage1.md``.
"""
from __future__ import annotations

import datetime as dt
import math
import pickle
from dataclasses import dataclass, field
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


def derive_lookback(alpha: float, target_tail_obs: int,
                    floor: int = 30, cap: int = 120) -> int:
    """Trading-day window sized so the alpha-quantile rests on ~``target_tail_obs``
    observations in the tail.

    ``alpha * lookback`` observations fall below the quantile, so to keep
    ~``target_tail_obs`` of them we need ``lookback ≈ target_tail_obs / alpha``.
    Bounded by ``floor`` (never so short it's jumpy) and ``cap`` (never so long
    it goes stale). Self-adjusts when ``entry_low_alpha`` changes — a deeper
    alpha automatically widens the window.
    """
    raw = math.ceil(target_tail_obs / alpha)
    return int(min(max(raw, floor), cap))


@dataclass
class RollingEmpiricalQuantileModel:
    """Per-ticker rolling empirical quantile of ``target_low`` (= ``low[T+1]/
    close[T] - 1``).

    For a prediction on date T, the entry dip for a ticker is the empirical
    ``alpha``-quantile of that ticker's OWN trailing ``lookback`` next-day-low
    returns (using only observations whose realized low is known by T — i.e.
    rows strictly before T, so no lookahead). ``P(fill) ≈ alpha`` per ticker,
    interpreted against its own recent dip distribution.

    Fallback chain when a ticker has too little history: pooled
    ``global_quantile`` (market-typical dip), then ``0.0`` (entry == close).

    Stored as a tiny pickle (``models/low_latest.pkl``) so the mean head can be
    rebuilt independently and old installs without a low model fall through to
    the close-anchored entry.

    ``boosters`` is always empty — kept only so call sites that echo
    ``len(model.boosters)`` keep working unchanged.
    """

    alpha: float
    target_tail_obs: int
    lookback: int
    min_obs: int
    global_quantile: float
    train_end: pd.Timestamp
    train_rows: int
    boosters: list = field(default_factory=list)
    # Pooled (market-wide) quantile curve: shape (G, 2) of [alpha, value], so the
    # thin-ticker fallback can serve an ARBITRARY per-row alpha by interpolation.
    # Optional + read via getattr so old pickles (without it) still load.
    global_quantile_grid: "np.ndarray | None" = None

    def _grid_quantile(self, alpha: float) -> float | None:
        """Pooled (market-wide) quantile at an ARBITRARY alpha, interpolated from
        the stored ``global_quantile_grid``. Returns None when no grid is stored
        (old pickles) so the caller can fall back to the scalar global."""
        grid = getattr(self, "global_quantile_grid", None)
        if grid is None or len(grid) == 0:
            return None
        return float(np.interp(float(alpha), grid[:, 0], grid[:, 1]))

    def _quantile_from_obs(self, obs: np.ndarray, alpha: float | None = None) -> float:
        a = self.alpha if alpha is None else float(alpha)
        if obs.size >= self.min_obs:
            return float(np.quantile(obs, a))
        # Thin ticker: prefer the pooled grid (honors an arbitrary per-row
        # alpha), then the scalar global (back-compat), then entry == close.
        g = self._grid_quantile(a)
        if g is not None and np.isfinite(g):
            return g
        if np.isfinite(self.global_quantile):
            return float(self.global_quantile)
        return 0.0

    def predict(self, X: pd.DataFrame,
                history: pd.DataFrame | None = None,
                alphas: "np.ndarray | pd.Series | None" = None) -> pd.Series:
        """Per-row empirical quantile of next-day-low for each row's symbol.

        ``X`` is a cross-section (one row per symbol) indexed by the as-of date,
        with a ``symbol`` column. ``history`` is the in-memory panel (indexed by
        date, with ``symbol`` and ``target_low`` columns) — preferred, no I/O.
        When ``history`` is None, each ticker's history is read from the OHLCV
        cache instead (used by tests / callers that pass only a snapshot).
        Built positionally so duplicate index timestamps across symbols are safe.

        ``alphas`` optionally overrides the dip quantile PER ROW (aligned to
        ``X``) so a caller can demand a deeper dip for a weaker pick. When None,
        every row uses the model's single ``alpha`` (the original behavior).
        """
        arr: np.ndarray | None = None
        if alphas is not None:
            arr = np.asarray(alphas, dtype=float)
            if arr.shape[0] != len(X):
                raise ValueError(
                    f"alphas length {arr.shape[0]} != X rows {len(X)}")

        groups: dict[str, pd.Series] = {}
        if history is not None and not history.empty and "target_low" in history.columns:
            hist = history[["symbol", "target_low"]].dropna(subset=["target_low"])
            groups = {str(sym): g["target_low"] for sym, g in hist.groupby("symbol")}

        from ..data.cache import read_ohlcv
        from .target import next_day_low_return

        out = np.empty(len(X), dtype=float)
        symbols = X["symbol"].astype(str).to_numpy()
        asofs = X.index.to_numpy()
        for i in range(len(X)):
            a_i = None if arr is None else arr[i]
            sym = symbols[i]
            asof = pd.Timestamp(asofs[i])
            series = groups.get(sym)
            if series is None and history is None:
                df = read_ohlcv(sym)
                if not df.empty and "low" in df.columns:
                    series = next_day_low_return(df).dropna()
            if series is None or series.empty:
                out[i] = self._quantile_from_obs(np.empty(0), alpha=a_i)
                continue
            # Strictly before the as-of date → only lows already known at T close.
            obs = series[series.index < asof].to_numpy()
            if self.lookback > 0:
                obs = obs[-self.lookback:]
            out[i] = self._quantile_from_obs(obs, alpha=a_i)
        return pd.Series(out, index=X.index, name="pred_low")

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "RollingEmpiricalQuantileModel":
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
                   target_col: str = "target_low") -> RollingEmpiricalQuantileModel:
    """Build the per-ticker rolling-empirical-quantile low head from ``panel``.

    ``alpha`` defaults to ``pricing.entry_low_alpha`` from config. The only
    value fitted at train time is ``global_quantile`` (the pooled alpha-quantile
    of ``target_col`` across the whole panel), used as a fallback for tickers
    without enough history; the per-ticker quantiles are computed at predict
    time from each ticker's own trailing window (see
    ``RollingEmpiricalQuantileModel.predict``).

    ``seeds`` is accepted but unused (kept so existing call sites that pass it
    keep working); the empirical quantile has no random seeds.
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

    target_tail_obs = int(cfg.pricing.get("entry_low_target_tail_obs", 15))
    # Size the window for the DEEPEST alpha prediction can request under
    # conviction coupling (base * weak_mult), so a deep weak-pick dip still
    # rests on ~target_tail_obs in its tail. With coupling off this equals the
    # base alpha, so the window is unchanged.
    hard_min = float(cfg.pricing.get("entry_alpha_hard_min", 0.05))
    hard_max = float(cfg.pricing.get("entry_alpha_hard_max", 0.75))
    if bool(cfg.pricing.get("entry_alpha_couple_conviction", True)):
        weak_mult = float(cfg.pricing.get("entry_alpha_weak_mult", 0.6))
        deepest_alpha = min(max(alpha_f * weak_mult, hard_min), hard_max)
    else:
        deepest_alpha = alpha_f
    lookback = derive_lookback(deepest_alpha, target_tail_obs)
    min_obs = target_tail_obs

    y = panel[target_col].to_numpy(dtype=float)
    has_y = bool(np.isfinite(y).any())
    global_quantile = float(np.nanquantile(y, alpha_f)) if has_y else 0.0
    # Pooled quantile curve so the thin-ticker fallback can serve any per-row
    # alpha (not just the base). Monotonic in alpha by construction.
    if has_y:
        grid_alphas = np.round(np.arange(0.01, 1.00, 0.01), 2)
        grid_vals = np.nanquantile(y, grid_alphas)
        global_quantile_grid = np.column_stack([grid_alphas, grid_vals])
    else:
        global_quantile_grid = None

    return RollingEmpiricalQuantileModel(
        alpha=alpha_f,
        target_tail_obs=target_tail_obs,
        lookback=lookback,
        min_obs=min_obs,
        global_quantile=global_quantile,
        train_end=panel.index.max(),
        train_rows=int(panel[target_col].notna().sum()),
        global_quantile_grid=global_quantile_grid,
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


def save_latest_low(model: RollingEmpiricalQuantileModel) -> Path:
    p = latest_low_model_path()
    model.save(p)
    stamped = models_dir() / f"low_model_{dt.date.today().isoformat()}_a{int(model.alpha * 100):03d}.pkl"
    model.save(stamped)
    return p
