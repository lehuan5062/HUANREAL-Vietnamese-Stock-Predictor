"""Train the rebound recovery estimator on the long panel of
(symbol, date, features, recovery-targets).

The sole head is the **Kaplan-Meier recovery estimator** (``RecoveryKMModel``):
per downtrend candidate it estimates, from history, the eventual recovery
probability, the days-to-recovery (N), and the profit at recovery (P). The
strongest signal is the ticker's OWN downtrend-recovery history WITHIN ITS
CURRENT STATE (a ticker × RSI-band × distance-below-high cell), falling back to
the ticker's flat lifetime average, then a coarse cross-sectional RSI ×
distance-below-high bucket, then a pooled all-downtrend fallback, as each tier
thins out.
"""
from __future__ import annotations

import datetime as dt
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config, models_dir


def _km_curve(times: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Kaplan-Meier survival estimate. ``times`` are integer days-to-event (or
    censoring time), ``events`` is True where the event (recovery) was observed.
    Returns an array of [t, S(t)] rows at each distinct time, S the probability
    of NOT having recovered by t. Censored observations correctly stay in the
    at-risk set up to their censoring time and never trigger a drop."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    n = times.size
    if n == 0:
        return np.empty((0, 2))
    order = np.argsort(times, kind="stable")
    t = times[order]
    e = events[order]
    uniq = np.unique(t)
    at_risk = n
    S = 1.0
    rows = []
    for tt in uniq:
        at_tt = t == tt
        d = int(np.sum(at_tt & e))       # recoveries at tt
        c = int(np.sum(at_tt & ~e))      # censored at tt
        if at_risk > 0 and d > 0:
            S *= (1.0 - d / at_risk)
        rows.append((tt, S))
        at_risk -= (d + c)
    return np.asarray(rows, dtype=float)


def _km_summarize(times: np.ndarray, events: np.ndarray) -> tuple[float, float]:
    """From a KM curve return (recovery_prob, days_to_recover).

    * ``recovery_prob`` = eventual recovery fraction = 1 - S at the last observed
      time (bounded [0, 1]).
    * ``days_to_recover`` = KM median survival time (smallest t with S(t) <= 0.5).
      When the curve never crosses 0.5 (recovery is rare in this bucket), fall
      back to the restricted mean survival time (area under S), which stays
      finite and orders buckets sensibly."""
    curve = _km_curve(times, events)
    if len(curve) == 0:
        return 0.0, float("nan")
    t = curve[:, 0]
    S = curve[:, 1]
    recovery_prob = float(min(max(1.0 - S[-1], 0.0), 1.0))
    below = np.where(S <= 0.5)[0]
    if below.size:
        days = float(t[below[0]])
    else:
        # Restricted mean survival time: area under the step function from 0..t[-1].
        edges = np.concatenate([[0.0], t])
        widths = np.diff(edges)
        heights = np.concatenate([[1.0], S[:-1]]) if len(S) > 1 else np.array([1.0])
        days = float(np.sum(widths * heights))
    return recovery_prob, max(days, 1.0)


@dataclass
class RecoveryKMModel:
    """Kaplan-Meier empirical estimator of the rebound recovery episode.

    Downtrend candidates are bucketed by a coarse state (RSI band x
    distance-below-20d-high band). Each bucket stores, from history:

    * ``recovery_prob`` — eventual fraction that turn profitable (censoring-aware).
    * ``days`` (N) — KM median days-to-recovery (restricted-mean fallback).
    * ``profit`` (P) — the ``p_quantile`` of realized profit over that bucket's
      RECOVERED episodes only (censored rows never poison the magnitude).

    Thin/empty buckets fall back to the ``pooled`` all-downtrend estimate.
    ``predict`` returns [pred_recovery_prob, pred_days, pred_profit]; the ranker
    turns these into score = P/N * recovery_prob. ``boosters`` stays empty so
    call sites echoing ``len(model.boosters)`` keep working."""

    buckets: dict
    pooled: dict
    rsi_edges: list
    high_prox_edges: list
    p_quantile: float
    min_bucket_obs: int
    train_end: pd.Timestamp
    train_rows: int
    boosters: list = field(default_factory=list)
    # Per-ticker recovery stats {symbol: {recovery_prob, days, profit, n}}. A
    # ticker's OWN downtrend history is the strongest "healthy vs falling knife"
    # signal — reliably-bouncing names cluster near recovery_prob ~1.0, chronic
    # decliners near 0.0 — so it is preferred over the coarse cross-sectional
    # bucket when the ticker has enough history. Optional (read via getattr) so
    # older pickles still load.
    ticker_stats: dict = field(default_factory=dict)
    min_ticker_obs: int = 100
    # Per-ticker × state stats {(symbol, ri, hi): {recovery_prob, days, profit,
    # n}} — more specific than ``ticker_stats``, which blends a ticker's ENTIRE
    # downtrend history into one flat number regardless of how deep the current
    # dip is. A ticker can be a reliable bouncer in a shallow dip and a chronic
    # decliner once it's fallen far below its recent high (or vice versa); once
    # a ticker clears ``min_ticker_obs`` the flat aggregate stops looking at
    # state at all, so this cell is checked FIRST and the flat aggregate is
    # only a fallback for states too thin to trust on their own. Optional (read
    # via getattr) so older pickles still load.
    ticker_bucket_stats: dict = field(default_factory=dict)
    min_ticker_bucket_obs: int = 30

    def _bucket_key(self, rsi: float, high_prox: float) -> tuple[int, int]:
        ri = int(np.digitize([float(rsi)], self.rsi_edges)[0])
        hi = int(np.digitize([float(high_prox)], self.high_prox_edges)[0])
        return (ri, hi)

    def _lookup(self, symbol: str, rsi: float, high_prox: float) -> dict:
        # 0) ticker x state cell (most specific) when it has enough history.
        if np.isfinite(rsi) and np.isfinite(high_prox):
            tbstats = getattr(self, "ticker_bucket_stats", None) or {}
            key = (str(symbol),) + self._bucket_key(rsi, high_prox)
            tb = tbstats.get(key)
            min_tb = int(getattr(self, "min_ticker_bucket_obs", 30))
            if tb is not None and int(tb.get("n", 0)) >= min_tb:
                return tb
        # 1) per-ticker flat aggregate (strongest health signal) when it has
        # enough history but no reliable state-specific cell.
        tstats = getattr(self, "ticker_stats", None) or {}
        t = tstats.get(str(symbol))
        if t is not None and int(t.get("n", 0)) >= int(getattr(self, "min_ticker_obs", 100)):
            return t
        # 2) coarse RSI x distance-below-high bucket.
        if np.isfinite(rsi) and np.isfinite(high_prox):
            b = self.buckets.get(self._bucket_key(rsi, high_prox))
            if b is not None and int(b.get("n", 0)) >= self.min_bucket_obs:
                return b
        # 3) pooled all-downtrend fallback.
        return self.pooled

    def predict(self, X: pd.DataFrame,
                history: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return [pred_recovery_prob, pred_days, pred_profit] aligned to X.index.
        ``history`` is accepted for signature parity with the other heads and is
        unused (the estimator is fully baked at train time)."""
        rsi = X.get("rsi_14", pd.Series(np.nan, index=X.index)).astype(float).to_numpy()
        hp = X.get("high_prox_20", pd.Series(np.nan, index=X.index)).astype(float).to_numpy()
        syms = X.get("symbol", pd.Series("", index=X.index)).astype(str).to_numpy()
        prob = np.empty(len(X)); days = np.empty(len(X)); profit = np.empty(len(X))
        for i in range(len(X)):
            b = self._lookup(syms[i], rsi[i], hp[i])
            prob[i] = b["recovery_prob"]
            days[i] = b["days"]
            profit[i] = b["profit"]
        return pd.DataFrame(
            {"pred_recovery_prob": prob, "pred_days": days, "pred_profit": profit},
            index=X.index,
        )

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "RecoveryKMModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def _bucket_stats(sub: pd.DataFrame, p_quantile: float) -> dict:
    """Compute (recovery_prob, days, profit, n) for a set of downtrend rows."""
    times = sub["target_days_to_recover"].to_numpy(dtype=float)
    events = sub["target_recovered"].to_numpy(dtype=bool)
    recovery_prob, days = _km_summarize(times, events)
    p_recovered = sub.loc[events, "target_recovery_return"].to_numpy(dtype=float)
    p_recovered = p_recovered[np.isfinite(p_recovered)]
    profit = float(np.quantile(p_recovered, p_quantile)) if p_recovered.size else float("nan")
    return {"recovery_prob": recovery_prob, "days": days,
            "profit": profit, "n": int(len(sub))}


def train_recovery(panel: pd.DataFrame) -> RecoveryKMModel:
    """Build the Kaplan-Meier recovery estimator from a feature panel.

    Filters the panel to downtrend rows (the rebound candidates) with a defined
    recovery episode, buckets them by (RSI band x distance-below-high band), and
    fits a KM survival curve per bucket, per ticker, per ticker x bucket cell,
    plus a pooled all-downtrend fallback. Reads its knobs from
    ``strategy.recovery``."""
    if panel.empty:
        raise ValueError("empty training panel")
    from ..filters import downtrend_mask  # local import to avoid a cycle at load

    need = {"target_days_to_recover", "target_recovered", "target_recovery_return",
            "rsi_14", "high_prox_20"}
    missing = need - set(panel.columns)
    if missing:
        raise KeyError(f"recovery panel missing columns: {sorted(missing)}")

    cfg = load_config()
    strat = dict(getattr(cfg, "strategy", {}) or {})
    recovery = dict(strat.get("recovery", {}) or {})
    p_quantile = float(recovery.get("p_quantile", 0.5))
    buckets_cfg = dict(recovery.get("state_buckets", {}) or {})
    rsi_edges = list(buckets_cfg.get("rsi_edges", [30, 40, 50]))
    high_prox_edges = list(buckets_cfg.get("high_prox_edges", [-0.20, -0.10, -0.05]))
    min_bucket_obs = int(recovery.get("min_bucket_obs", 50))
    min_ticker_obs = int(recovery.get("min_ticker_obs", 100))
    min_ticker_bucket_obs = int(recovery.get("min_ticker_bucket_obs", 30))

    # A multi-day downtrend produces one row per calendar day, but consecutive
    # days of the SAME decline are not independent trials -- they're one event
    # sampled repeatedly (a 20-day decline would otherwise outweigh a genuine
    # 2-day dip 10:1 in every stat below, and a still-open decline racks up one
    # near-duplicate "censored" row per day it stays open instead of counting
    # once). Collapse each contiguous downtrend run, per symbol, down to its
    # entry-day row -- which already carries that episode's own correctly
    # resolved (or still-censored) outcome from target.py -- before any of the
    # aggregations below see it.
    panel_sorted = panel.sort_index(kind="stable")
    full_mask = downtrend_mask(panel_sorted).to_numpy()
    grp_arr = (panel_sorted["symbol"].to_numpy() if "symbol" in panel_sorted.columns
              else np.zeros(len(panel_sorted), dtype=int))
    run_tbl = pd.DataFrame({"mask": full_mask, "grp": grp_arr})
    prev_mask = run_tbl.groupby("grp")["mask"].shift(1, fill_value=False)
    run_start = run_tbl["mask"] & ~prev_mask
    episode_num = run_start.groupby(run_tbl["grp"]).cumsum()

    dt_rows = panel_sorted[full_mask].copy()
    dt_rows["__episode__"] = episode_num[full_mask].to_numpy()
    dt_rows = dt_rows.dropna(subset=["target_days_to_recover"])
    if dt_rows.empty:
        raise ValueError("no downtrend rows with a recovery label in the panel")

    ri = np.digitize(dt_rows["rsi_14"].astype(float).to_numpy(), rsi_edges)
    hi = np.digitize(dt_rows["high_prox_20"].astype(float).to_numpy(), high_prox_edges)
    dt_rows["__ri__"] = ri
    dt_rows["__hi__"] = hi

    group_cols = ["symbol", "__episode__"] if "symbol" in dt_rows.columns else ["__episode__"]
    episodes = dt_rows.groupby(group_cols, as_index=False).first()

    buckets: dict[tuple[int, int], dict] = {}
    for (rk, hk), sub in episodes.groupby(["__ri__", "__hi__"]):
        buckets[(int(rk), int(hk))] = _bucket_stats(sub, p_quantile)

    # Per-ticker recovery reliability — the primary "healthy vs falling knife"
    # signal. A chronic decliner (e.g. LCC) lands near recovery_prob 0; a name
    # that reliably bounces near 1.0.
    ticker_stats: dict[str, dict] = {}
    ticker_bucket_stats: dict[tuple[str, int, int], dict] = {}
    if "symbol" in episodes.columns:
        for sym, sub in episodes.groupby("symbol"):
            ticker_stats[str(sym)] = _bucket_stats(sub, p_quantile)
        # Ticker x state cells — the same per-ticker history, but split by the
        # ticker's OWN state at pick time (see RecoveryKMModel.ticker_bucket_stats).
        for (sym, rk, hk), sub in episodes.groupby(["symbol", "__ri__", "__hi__"]):
            ticker_bucket_stats[(str(sym), int(rk), int(hk))] = _bucket_stats(sub, p_quantile)

    pooled = _bucket_stats(episodes, p_quantile)

    return RecoveryKMModel(
        buckets=buckets,
        pooled=pooled,
        rsi_edges=rsi_edges,
        high_prox_edges=high_prox_edges,
        p_quantile=p_quantile,
        min_bucket_obs=min_bucket_obs,
        train_end=panel.index.max(),
        train_rows=int(len(dt_rows)),
        ticker_stats=ticker_stats,
        min_ticker_obs=min_ticker_obs,
        ticker_bucket_stats=ticker_bucket_stats,
        min_ticker_bucket_obs=min_ticker_bucket_obs,
    )


def latest_recovery_model_path() -> Path:
    return models_dir() / "recovery_latest.pkl"


def save_latest_recovery(model: RecoveryKMModel) -> Path:
    p = latest_recovery_model_path()
    model.save(p)
    stamped = models_dir() / f"recovery_model_{dt.date.today().isoformat()}.pkl"
    model.save(stamped)
    return p
