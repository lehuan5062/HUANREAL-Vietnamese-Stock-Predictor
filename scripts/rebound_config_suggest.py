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

CATEGORICAL_KNOBS = ["train_window_years", "oos_window_months", "step_months"]
CONTINUOUS_KNOBS = ["min_recovery_prob", "p_quantile", "profit_margin"]


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
    print()
    print("=== Categorical knobs (grouped by value) ===")
    for knob in CATEGORICAL_KNOBS:
        print(f"\n-- {knob} --")
        grouped = _analyze_categorical(df, knob)
        print(grouped.to_string(index=False))
        if grouped["thin"].any():
            print("  (rows flagged thin have <3 trials - low confidence, excluded from suggestion)")

    print()
    print("=== Continuous knobs (correlation with annualized_IRR) ===")
    continuous_results = {}
    for knob in CONTINUOUS_KNOBS:
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
    for knob in CATEGORICAL_KNOBS:
        grouped = _analyze_categorical(df, knob)
        usable = grouped[~grouped["thin"]]
        if usable.empty:
            print(f"  {knob}: no group has >=3 trials yet - no suggestion")
        else:
            best = usable.iloc[0]
            print(f"  backtest.{knob}: {best[f'config.{knob}']} "
                  f"(mean IRR {best['mean_irr']:.4f} over {int(best['count'])} trials)")
    for knob in CONTINUOUS_KNOBS:
        res = continuous_results[knob]
        if res["terciles"] is not None:
            best_row = res["terciles"].sort_values("mean_irr", ascending=False).iloc[0]
            print(f"  strategy.recovery.{knob}: prefer range {best_row['range_']} "
                  f"(mean IRR {best_row['mean_irr']:.4f} over {int(best_row['count'])} trials)")
        else:
            print(f"  strategy.recovery.{knob}: {res['read']} (not enough trials for a range yet)")

    print()
    print("=== Caveats ===")
    print("- All trials backtest the SAME fixed historical window (2024-01-02..present).")
    print("  A knob that looks good here may be overfit to this exact period, not a")
    print("  robust improvement.")
    print("- These are correlational, not causal, and sample sizes are still small.")
    print("- Advisory only - don't copy this verbatim into config.yaml. Cross-check")
    print("  against per-pick diagnosis before proposing an edit.")


if __name__ == "__main__":
    main()
