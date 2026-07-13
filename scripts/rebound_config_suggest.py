"""Analyze accumulated rebound_config_tuner.py trials to suggest config values.

Reads reports/tuning/rebound_include_held_search.jsonl (written by
scripts/rebound_config_tuner.py) and does two layers of analysis:

1. Per-knob marginal analysis (always): since the tuner samples every knob
   independently and jointly on every trial, grouping/correlating one knob at
   a time against annualized_IRR (averaging over the others) is a legitimate
   signal — not just eyeballing the single best row. Window-difficulty noise
   averages out within each knob's group because the window is randomized
   independently of the knobs.
2. ML surrogate (once >= ML_MIN_TRIALS trials exist): a LightGBM model
   trained on excess_irr (each trial's IRR minus what a plain buy-and-hold
   would have returned over that same random window — this removes the
   "was it a good year" noise that raw IRR can't distinguish from config
   skill). Unlike the marginal analysis, this can see COMBINATIONS of knobs,
   not just one at a time. Prints a holdout R² so you can tell whether it
   found real signal or is just fitting noise, then feature importances and
   a few concrete candidate configs it predicts would score well.

Prints everything to stdout. Never writes any file, never touches
config.yaml. rebound_config_tuner.py is untouched by this script — it only
reads what the tuner already wrote (and reuses its knob-sampling functions to
generate ML candidate configs, so ranges/coordination never drift out of
sync between the two scripts).

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
# Incremental parse cache: the JSONL only ever grows (tuner appends), so we
# keep a parquet of the already-flattened rows plus a byte offset and only
# json_normalize the lines appended since the last run.
CACHE_PATH = RESULTS_PATH.with_suffix(".cache.parquet")
CACHE_META_PATH = RESULTS_PATH.with_suffix(".cache.meta.json")
MIN_TRIALS = 5
MIN_GROUP_SIZE = 3
TERCILE_MIN_TRIALS = 12
ML_MIN_TRIALS = 50
ML_CANDIDATE_SAMPLES = 5000
ML_TOP_N = 5
# Range-boundary check: a continuous knob is "at the edge" when the median of
# the top-quartile trials sits within EDGE_FRACTION of the sampling span from
# an endpoint; the ML surrogate's top candidates confirm within ML_EDGE_FRACTION.
EDGE_FRACTION = 0.10
ML_EDGE_FRACTION = 0.05
TOP_QUANTILE = 0.75

# Knobs whose config value is a small sorted LIST, not a scalar (the
# coordinated RSI/high-prox bucket edges). Expanded into individual
# positional features for the ML surrogate; excluded from the marginal
# per-knob tables below (which only handle scalars).
LIST_KNOBS = [
    "strategy.recovery.state_buckets.rsi_edges",
    "strategy.recovery.state_buckets.high_prox_edges",
]

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


def _parse_lines(text: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    return pd.json_normalize(rows, sep=".")


def _read_cache() -> tuple[pd.DataFrame, int] | None:
    try:
        offset = json.loads(CACHE_META_PATH.read_text(encoding="utf-8"))["byte_offset"]
        return pd.read_parquet(CACHE_PATH), int(offset)
    except Exception:
        return None


def _write_cache(df: pd.DataFrame, offset: int) -> None:
    try:
        df.to_parquet(CACHE_PATH, index=False)
        CACHE_META_PATH.write_text(json.dumps({"byte_offset": offset}), encoding="utf-8")
    except Exception:
        # A locked/unwritable cache must never break the run; next run just
        # falls back to a full re-parse.
        CACHE_META_PATH.unlink(missing_ok=True)


def _load_trials() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        return pd.DataFrame()
    file_size = RESULTS_PATH.stat().st_size
    cached = _read_cache()
    if cached is not None and cached[1] <= file_size:
        df, offset = cached
        with open(RESULTS_PATH, "rb") as f:
            f.seek(offset)
            chunk = f.read()
        # Only consume complete lines, in case the tuner is mid-append.
        end = chunk.rfind(b"\n") + 1
        if end > 0:
            new_df = _parse_lines(chunk[:end].decode("utf-8"))
            if not new_df.empty:
                df = pd.concat([df, new_df], ignore_index=True)
            _write_cache(df, offset + end)
        return df
    # No cache, corrupt cache, or the JSONL shrank (rewritten): full re-parse.
    raw = RESULTS_PATH.read_bytes()
    end = raw.rfind(b"\n") + 1
    df = _parse_lines(raw[:end].decode("utf-8"))
    _write_cache(df, end)
    return df


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


def _feature_matrix(df: pd.DataFrame, scalar_knobs: list[str]) -> pd.DataFrame:
    """ML feature matrix from a trials dataframe: scalar knob columns as-is,
    LIST_KNOBS (rsi_edges/high_prox_edges) expanded into individual
    positional columns (edge__0, edge__1, edge__2) since a model can't take
    a list-valued cell directly."""
    cols = {}
    for k in scalar_knobs:
        col = f"config.{k}"
        if col in df.columns:
            cols[k] = df[col]
    for k in LIST_KNOBS:
        col = f"config.{k}"
        if col in df.columns:
            expanded = pd.DataFrame(df[col].tolist(), index=df.index)
            for i in range(expanded.shape[1]):
                cols[f"{k}__{i}"] = expanded[i]
    return pd.DataFrame(cols)


def _flat_to_feature_row(flat: dict, scalar_knobs: list[str]) -> dict:
    """Same expansion as _feature_matrix, but for one freshly-sampled
    candidate config (rebound_config_tuner._sample_flat() output) instead of
    a historical trials dataframe row."""
    row = {}
    for k in scalar_knobs:
        if k in flat:
            row[k] = flat[k]
    for k in LIST_KNOBS:
        if k in flat:
            for i, v in enumerate(flat[k]):
                row[f"{k}__{i}"] = v
    return row


def _run_ml_surrogate(df: pd.DataFrame, cat_knobs: list[str], con_knobs: list[str]):
    """LightGBM surrogate: learns excess_irr from knob combinations (not just
    one knob at a time), reports honest holdout R², feature importances, and
    a handful of concrete candidate configs it predicts would score well.
    Skips cleanly (prints why) if there's not enough usable data yet.

    Returns the top predicted candidate configs (list of flat dicts) when the
    holdout R² shows real signal, else None — consumed by the range-boundary
    check as confirmation that an edge optimum isn't a marginal-analysis fluke."""
    if "excess_irr" not in df.columns:
        print(f"(ML surrogate: no trials have excess_irr yet - re-run the tuner "
              f"to accumulate some, then re-run this script)")
        return None

    scalar_knobs = cat_knobs + con_knobs
    X = _feature_matrix(df, scalar_knobs)
    y = df["excess_irr"]
    valid = X.notna().all(axis=1) & y.notna()
    X, y = X[valid], y[valid]

    if len(X) < ML_MIN_TRIALS:
        print(f"(ML surrogate needs >= {ML_MIN_TRIALS} complete trials with "
              f"excess_irr; have {len(X)} - skipping for now)")
        return None

    from lightgbm import LGBMRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = LGBMRegressor(n_estimators=200, max_depth=4, min_child_samples=5,
                          verbosity=-1)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))

    print()
    print("=== ML surrogate (LightGBM, trained on excess_irr) ===")
    print(f"Trained on {len(X_train)} trials, holdout R^2 on {len(X_test)} "
          f"unseen trials: {r2:.3f}")
    if r2 < 0.05:
        print("R^2 is near zero: the model has NOT found real signal yet - "
              "treat everything below as noise, not a pattern. Keep")
        print("accumulating trials before trusting this section.")

    importances = sorted(zip(X.columns, model.feature_importances_),
                         key=lambda t: -t[1])
    print()
    print("Feature importance (which knobs the model found most predictive,")
    print("accounting for combinations - this is what a pure per-knob average can't see):")
    for name, imp in importances:
        print(f"  {name}: {imp}")

    from scripts.rebound_config_tuner import _sample_flat
    candidates = [_sample_flat() for _ in range(ML_CANDIDATE_SAMPLES)]
    cand_features = pd.DataFrame(
        [_flat_to_feature_row(c, scalar_knobs) for c in candidates])
    cand_features = cand_features[X.columns]  # match training column order
    preds = model.predict(cand_features)
    order = preds.argsort()[::-1][:ML_TOP_N]

    print()
    print(f"Model's top {ML_TOP_N} predicted-best configs, out of "
          f"{ML_CANDIDATE_SAMPLES} randomly sampled candidates:")
    for rank, i in enumerate(order, 1):
        print(f"\n  #{rank}  predicted excess_irr = {preds[i]:.4f}")
        print(json.dumps(candidates[int(i)], indent=4, default=str))

    if r2 < 0.05:
        return None
    return [candidates[int(i)] for i in order]


def _range_boundary_check(df: pd.DataFrame, cat_knobs: list[str],
                          con_knobs: list[str], ml_top) -> None:
    """Detect optima piling up at the edge of the tuner's sampling range —
    the one thing neither the marginal analysis nor the ML surrogate can see,
    because the surrogate only ever samples candidates INSIDE the range.

    Continuous knobs: flag when the top-quartile trials (by excess_irr, so
    window luck is already removed) have their median value within
    EDGE_FRACTION of the range span from an endpoint. If the ML surrogate ran
    with real signal, its top candidates must agree (median within
    ML_EDGE_FRACTION of the same endpoint), otherwise the trial clustering
    alone fires with a lower-confidence note.

    Categorical knobs: flag when the best non-thin group is the smallest or
    largest extendable value in the choice grid (sentinel values like
    0 = disabled are never treated as an extendable endpoint)."""
    from scripts.rebound_config_tuner import KNOB_BOUNDS

    warnings_out = []

    if "excess_irr" in df.columns:
        for knob in con_knobs:
            spec = KNOB_BOUNDS.get(knob)
            if spec is None or spec["kind"] != "uniform":
                continue
            col = f"config.{knob}"
            sub = df[[col, "excess_irr"]].dropna()
            if len(sub) < TERCILE_MIN_TRIALS:
                continue
            span = spec["high"] - spec["low"]
            top = sub[sub["excess_irr"] >= sub["excess_irr"].quantile(TOP_QUANTILE)]
            top_median = top[col].median()
            side = None
            if top_median <= spec["low"] + EDGE_FRACTION * span:
                side, endpoint = "lower", spec["low"]
            elif top_median >= spec["high"] - EDGE_FRACTION * span:
                side, endpoint = "upper", spec["high"]
            if side is None or side in spec.get("no_extend", []):
                continue
            note = "(trial clustering only - ML surrogate not available/confident yet)"
            if ml_top:
                ml_vals = pd.Series([c[knob] for c in ml_top if knob in c])
                if ml_vals.empty:
                    continue
                ml_median = ml_vals.median()
                near = (abs(ml_median - endpoint) <= ML_EDGE_FRACTION * span)
                if not near:
                    continue  # trials cluster at the edge but the surrogate disagrees
                note = "(confirmed by ML surrogate top candidates)"
            warnings_out.append(
                f"RANGE-BOUNDARY WARNING: {knob} optimum at {side} edge "
                f"({endpoint:g} of range {spec['low']:g}-{spec['high']:g}) {note} - "
                f"consider widening this knob's sampling range in "
                f"scripts/rebound_config_tuner.py, then accumulate fresh trials.")

    for knob in cat_knobs:
        spec = KNOB_BOUNDS.get(knob)
        if spec is None or spec["kind"] != "choice":
            continue
        extendable = [v for v in spec["values"] if v not in spec.get("sentinel", [])]
        if len(extendable) < 2:
            continue
        grouped = _analyze_categorical(df, knob)
        usable = grouped[~grouped["thin"]]
        if usable.empty:
            continue
        best = usable.iloc[0][f"config.{knob}"]
        side = None
        if best == min(extendable):
            side = "lower"
        elif best == max(extendable):
            side = "upper"
        if side is None or side in spec.get("no_extend", []):
            continue
        warnings_out.append(
            f"RANGE-BOUNDARY WARNING: {knob} best group is the {side} end of its "
            f"choice grid ({best} of {extendable}) - consider extending the grid "
            f"in scripts/rebound_config_tuner.py, then accumulate fresh trials.")

    print()
    print("=== Range-boundary check ===")
    if warnings_out:
        for w in warnings_out:
            print(f"  {w}")
        print("  (An edge optimum means the true best value may lie OUTSIDE the")
        print("  current search space - the ML surrogate can never propose beyond")
        print("  it. Widen the range in the tuner, not config.yaml directly.)")
    else:
        print("  No knob optimum sits at a sampling-range boundary - the current")
        print("  search space looks wide enough.")


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

    ml_top = _run_ml_surrogate(df, cat_knobs, con_knobs)

    _range_boundary_check(df, cat_knobs, con_knobs, ml_top)

    print()
    print("=== Caveats ===")
    print("- Each trial backtests a DIFFERENT random 1-year window, so raw IRR is")
    print("  NOT comparable across trials (a 2021-bull window beats a 2022-bear one")
    print("  regardless of config). Only the per-knob averages above are meaningful:")
    print("  window difficulty averages out within each knob group because the")
    print("  window is randomized independently of the knobs. Do NOT rank raw trials.")
    print("- These are correlational, not causal, and sample sizes are still small.")
    print("- The ML surrogate above (if it ran) uses excess_irr specifically to")
    print("  remove window-luck noise, and its own holdout R^2 tells you whether it")
    print("  found real signal - a low R^2 means treat its output as noise too.")
    print("- Advisory only - don't copy this verbatim into config.yaml. Cross-check")
    print("  against per-pick diagnosis before proposing an edit.")


if __name__ == "__main__":
    main()
