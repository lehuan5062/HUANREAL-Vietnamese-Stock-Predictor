"""Analyze accumulated rebound_config_tuner.py trials to suggest config values.

Reads reports/tuning/rebound_include_held_search.jsonl (written by
scripts/rebound_config_tuner.py) and does a per-knob marginal analysis:
since the tuner samples all 6 knobs independently and jointly on every
trial, grouping/correlating one knob at a time against annualized_IRR
(averaging over the other 5) is a legitimate signal — not just eyeballing
the single best row.

Prints the analysis + a suggested config to stdout. Never writes any file,
never touches config.yaml. rebound_config_tuner.py is untouched by this
script — it only reads what the tuner already wrote.

    python -m scripts.rebound_config_suggest
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from stockpredict import PROJECT_ROOT

RESULTS_PATH = PROJECT_ROOT / "reports" / "tuning" / "rebound_include_held_search.jsonl"
MIN_TRIALS = 5
MIN_GROUP_SIZE = 3
TERCILE_MIN_TRIALS = 12

# Knob names are the FULL dotted config paths the tuner records under `config.`
# (rebound_config_tuner writes flat dotted keys). List-valued knobs
# (state_buckets.*_edges) are recorded in the JSONL but not scalar-analyzed here.
CATEGORICAL_KNOBS = [
    "backtest.train_window_years", "backtest.oos_window_months", "backtest.step_months",
    "strategy.recovery.min_ticker_obs", "strategy.recovery.min_bucket_obs",
    "strategy.recovery.label_max_horizon",
    "universe.liquidity_filter.min_close_vnd", "universe.liquidity_filter.min_adv_active_days",
    "universe.liquidity_filter.min_adv_vnd", "universe.liquidity_filter.min_history_days",
    "pricing.overbought_rsi_max", "pricing.corp_action_lookback",
    "features.rsi_period", "strategy.downtrend.rsi_floor", "strategy.downtrend.rsi_ceil",
]
CONTINUOUS_KNOBS = [
    "strategy.recovery.min_recovery_prob", "strategy.recovery.p_quantile",
    "strategy.recovery.profit_margin",
    "strategy.downtrend.mom20_max", "strategy.downtrend.high_prox_max",
    "pricing.ceiling_tol", "pricing.max_participation_pct",
]


def _load_trials() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in RESULTS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.json_normalize(rows, sep=".")


def _analyze_categorical(df: pd.DataFrame, knob: str) -> pd.DataFrame:
    col = f"config.{knob}"
    grouped = df.groupby(col).agg(
        count=("result.annualized_IRR", "size"),
        mean_irr=("result.annualized_IRR", "mean"),
        median_irr=("result.annualized_IRR", "median"),
        mean_drawdown=("result.book_max_drawdown", "mean"),
    ).reset_index()
    grouped["thin"] = grouped["count"] < MIN_GROUP_SIZE
    return grouped.sort_values("mean_irr", ascending=False)


def _corr_read(r: float) -> str:
    if pd.isna(r):
        return "not enough variation to compute"
    if abs(r) < 0.15:
        return "~0: no clear linear signal"
    if r > 0:
        return f"positive ({r:.3f}): higher tends to help"
    return f"negative ({r:.3f}): lower tends to help"


def _analyze_continuous(df: pd.DataFrame, knob: str) -> dict:
    col = f"config.{knob}"
    corr = df[col].corr(df["result.annualized_IRR"])
    out = {"correlation": corr, "read": _corr_read(corr), "terciles": None}
    if len(df) >= TERCILE_MIN_TRIALS:
        try:
            tercile = pd.qcut(df[col], 3, labels=["low", "mid", "high"], duplicates="drop")
            grp = df.groupby(tercile, observed=True).agg(
                count=("result.annualized_IRR", "size"),
                mean_irr=("result.annualized_IRR", "mean"),
                range_=(col, lambda s: f"{s.min():.4g}-{s.max():.4g}"),
            )
            out["terciles"] = grp
        except ValueError:
            pass
    return out


def main():
    df = _load_trials()
    n = len(df)
    if n < MIN_TRIALS:
        print(f"Not enough trials yet (have {n}, need {MIN_TRIALS}). "
              f"Run run_config_tuner.bat a few more times, then re-run this.")
        return

    print(f"{n} trial(s) recorded in {RESULTS_PATH}")
    cat_knobs = [k for k in CATEGORICAL_KNOBS if f"config.{k}" in df.columns]
    con_knobs = [k for k in CONTINUOUS_KNOBS if f"config.{k}" in df.columns]
    print()
    print("=== Categorical knobs (grouped by value) ===")
    for knob in cat_knobs:
        print(f"\n-- {knob} --")
        grouped = _analyze_categorical(df, knob)
        print(grouped.to_string(index=False))
        if grouped["thin"].any():
            print("  (rows flagged thin have <3 trials - low confidence, excluded from suggestion)")

    print()
    print("=== Continuous knobs (correlation with annualized_IRR) ===")
    continuous_results = {}
    for knob in con_knobs:
        res = _analyze_continuous(df, knob)
        continuous_results[knob] = res
        print(f"\n-- {knob} --")
        print(f"  correlation: {res['read']}")
        if res["terciles"] is not None:
            print(res["terciles"].to_string())
        else:
            print(f"  (need >= {TERCILE_MIN_TRIALS} trials for tercile breakdown, have {n})")

    print()
    print("=== Suggested config (advisory - cross-check before applying) ===")
    for knob in cat_knobs:
        grouped = _analyze_categorical(df, knob)
        usable = grouped[~grouped["thin"]]
        if usable.empty:
            print(f"  {knob}: no group has >=3 trials yet - no suggestion")
        else:
            best = usable.iloc[0]
            print(f"  {knob}: {best[f'config.{knob}']} "
                  f"(mean IRR {best['mean_irr']:.4f} over {int(best['count'])} trials)")
    for knob in con_knobs:
        res = continuous_results[knob]
        if res["terciles"] is not None:
            best_row = res["terciles"].sort_values("mean_irr", ascending=False).iloc[0]
            print(f"  {knob}: prefer range {best_row['range_']} "
                  f"(mean IRR {best_row['mean_irr']:.4f} over {int(best_row['count'])} trials)")
        else:
            print(f"  {knob}: {res['read']} (not enough trials for a range yet)")

    print()
    print("=== Caveats ===")
    print("- Each trial backtests a DIFFERENT random 1-year window, so raw IRR is")
    print("  NOT comparable across trials (a 2021-bull window beats a 2022-bear one")
    print("  regardless of config). Only the per-knob averages above are meaningful:")
    print("  window difficulty averages out within each knob group because the")
    print("  window is randomized independently of the knobs. Do NOT rank raw trials.")
    print("- These are correlational, not causal, and sample sizes are still small.")
    print("- Advisory only - don't copy this verbatim into config.yaml. Cross-check")
    print("  against per-pick diagnosis before proposing an edit.")


if __name__ == "__main__":
    main()
