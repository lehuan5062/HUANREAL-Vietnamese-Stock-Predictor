"""One-shot randomized-config trial for rebound_sim_include_held.

Each run:
  1. Picks a random combination of ALL prediction-affecting config knobs
     (backtest windows, recovery-model thresholds, liquidity/history gates,
     downtrend gates, pricing gates, and a coordinated RSI/high-prox family).
  2. Picks a random continuous 1-year backtest SLICE (a random start date →
     +1 year) drawn from the rolling actual 9 years (today-9y → today), leaving
     room for that trial's training lookback.
  3. Writes the sampled config to config.yaml, runs the sim once on that slice,
     appends (config, window, result) to a results file, then ALWAYS restores
     the original config.yaml — even on error or Ctrl+C.

The real config.yaml is never left changed. Re-run this (via
run_config_tuner.bat) as many times as you want to accumulate trials, then use
run_config_suggest.bat to see which knob values correlate with higher IRR.

NOTE: raw IRR is NOT comparable across trials (each uses a different 1-year
window, so window difficulty dominates). Only the per-knob marginal analysis in
rebound_config_suggest is meaningful — window difficulty averages out within
each knob's group because the window is randomized independently of the knobs.

Real-world constants are deliberately NOT randomized: broker fees/VAT/PIT,
settle_days, and exchange ceiling_limits.

    python -m scripts.rebound_config_tuner
"""
from __future__ import annotations

import copy
import datetime
import json
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import yaml

from stockpredict import PROJECT_ROOT
from stockpredict.config import load_config

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
BACKUP_DIR = PROJECT_ROOT / "reports" / "tuning" / "config_backups"
RESULTS_PATH = PROJECT_ROOT / "reports" / "tuning" / "rebound_include_held_search.jsonl"

HISTORY_YEARS = 9          # rolling span the random window is drawn from
WINDOW_YEARS = 1           # each trial backtests a random 1-year slice


# --- knob specs -------------------------------------------------------------
# Each entry: (dotted config path, zero-arg sampler). Independent scalar knobs
# only. Coordinated families (RSI, high-prox) are sampled separately below
# because their values must stay mutually consistent.
SIMPLE_KNOBS = [
    ("backtest.train_window_years", lambda: random.choice([1, 2, 3, 4])),
    ("backtest.oos_window_months", lambda: random.choice([3, 6, 9, 12])),
    ("backtest.step_months", lambda: random.choice([1, 2, 3, 6])),
    ("strategy.recovery.min_recovery_prob", lambda: round(random.uniform(0.70, 0.99), 4)),
    ("strategy.recovery.p_quantile", lambda: round(random.uniform(0.50, 0.85), 4)),
    ("strategy.recovery.profit_margin", lambda: round(random.uniform(0.002, 0.02), 5)),
    ("strategy.recovery.min_ticker_obs", lambda: random.choice([50, 100, 150, 200])),
    ("strategy.recovery.min_bucket_obs", lambda: random.choice([25, 50, 75, 100])),
    ("strategy.recovery.label_max_horizon", lambda: random.choice([120, 180, 250])),
    ("universe.liquidity_filter.min_close_vnd", lambda: random.choice([3, 5, 7, 10])),
    ("universe.liquidity_filter.min_adv_active_days", lambda: random.choice([5, 10, 15, 18, 20])),
    ("universe.liquidity_filter.min_adv_vnd", lambda: random.choice([500_000, 1_000_000, 2_000_000, 5_000_000])),
    ("universe.liquidity_filter.min_history_days", lambda: random.choice([120, 180, 250, 400])),
    ("strategy.downtrend.mom20_max", lambda: round(random.uniform(-0.05, 0.05), 4)),
    ("strategy.downtrend.high_prox_max", lambda: round(random.uniform(-0.15, -0.02), 4)),
    ("pricing.overbought_rsi_max", lambda: random.choice([0, 70, 75, 80, 85])),
    ("pricing.ceiling_tol", lambda: round(random.uniform(0.005, 0.03), 4)),
    ("pricing.max_participation_pct", lambda: round(random.uniform(0.5, 3.0), 3)),
    ("pricing.corp_action_lookback", lambda: random.choice([15, 20, 25])),
]


def _sample_rsi_family() -> dict:
    """rsi_period + bucket rsi_edges + downtrend rsi_floor/ceil, kept mutually
    consistent: floor <= min(edges) <= max(edges) <= ceil. (RSI is 0-100 for
    any period; coordination = ordering/nesting, not period rescaling.)"""
    period = random.choice([9, 14, 21])
    edges = sorted(random.sample([15, 20, 25, 30, 35, 40, 45, 50, 55, 60], 3))
    floor_choices = [0] + [v for v in (10, 20, 25) if v <= edges[0]]
    rsi_floor = random.choice(floor_choices)
    ceil_choices = [v for v in (50, 55, 60, 65, 70) if v >= edges[-1]] or [edges[-1]]
    rsi_ceil = random.choice(ceil_choices)
    assert rsi_floor <= edges[0] <= edges[-1] <= rsi_ceil, (rsi_floor, edges, rsi_ceil)
    return {
        "features.rsi_period": period,
        "strategy.recovery.state_buckets.rsi_edges": edges,
        "strategy.downtrend.rsi_floor": rsi_floor,
        "strategy.downtrend.rsi_ceil": rsi_ceil,
    }


def _sample_high_prox_edges(high_prox_max: float) -> dict:
    """3 sorted-ascending bucket edges in [-0.30,-0.02], with the downtrend
    high_prox_max >= max(edges) (both negative; ">=" = closer to zero)."""
    grid = [-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, -0.03]
    # keep edges strictly below the downtrend gate so max(edges) <= high_prox_max
    usable = [v for v in grid if v <= high_prox_max]
    if len(usable) < 3:
        usable = grid[:3]  # degenerate fallback; assertion below stays valid
    edges = sorted(random.sample(usable, 3))
    assert edges[0] <= edges[1] <= edges[2], edges
    return {"strategy.recovery.state_buckets.high_prox_edges": edges}


def _sample_flat() -> dict:
    """Sample every knob → flat {dotted_path: value} dict."""
    flat = {path: sampler() for path, sampler in SIMPLE_KNOBS}
    flat.update(_sample_rsi_family())
    flat.update(_sample_high_prox_edges(flat["strategy.downtrend.high_prox_max"]))
    return flat


def _deep_set(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = d
    for k in keys[:-1]:
        node = node[k]          # intermediate keys must already exist in config
    node[keys[-1]] = value


def _apply_overrides(raw: dict, flat: dict) -> dict:
    mutated = copy.deepcopy(raw)
    for dotted, value in flat.items():
        _deep_set(mutated, dotted, value)
    return mutated


def _pick_window(train_years: int):
    """A random continuous 1-year slice within the rolling actual 9 years,
    leaving `train_years` of training lookback before the slice start."""
    today = pd.Timestamp.now().normalize()
    span_floor = today - pd.DateOffset(years=HISTORY_YEARS)
    earliest_start = span_floor + pd.DateOffset(years=train_years)
    latest_start = today - pd.DateOffset(years=WINDOW_YEARS)
    span_days = (latest_start - earliest_start).days
    if span_days <= 0:
        start = earliest_start
    else:
        start = earliest_start + pd.Timedelta(days=random.randint(0, span_days))
    end = start + pd.DateOffset(years=WINDOW_YEARS)
    return start, end


def main():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    original_bytes = CONFIG_PATH.read_bytes()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"config_{timestamp}.yaml"
    backup_path.write_bytes(original_bytes)

    flat = _sample_flat()
    window_start, window_end = _pick_window(int(flat["backtest.train_window_years"]))

    raw = yaml.safe_load(original_bytes.decode("utf-8"))
    mutated = _apply_overrides(raw, flat)
    CONFIG_PATH.write_text(yaml.safe_dump(mutated, sort_keys=False), encoding="utf-8")
    load_config.cache_clear()

    print("Randomized config for this trial:")
    print(json.dumps(flat, indent=2, default=str))
    print(f"window: {window_start.date()} -> {window_end.date()}")
    print()

    try:
        from scripts.rebound_sim_include_held import _build_data, simulate

        print("Building data (retrains the recovery model per anchor)...", flush=True)
        data = _build_data(start=window_start.strftime("%Y-%m-%d"),
                           end=window_end.strftime("%Y-%m-%d"))
        print("Running simulation...", flush=True)
        result = simulate(data=data)
    finally:
        CONFIG_PATH.write_bytes(original_bytes)
        load_config.cache_clear()

    record = {
        "timestamp": timestamp,
        "window_start": window_start.strftime("%Y-%m-%d"),
        "window_end": window_end.strftime("%Y-%m-%d"),
        "config": flat,
        "result": result,
    }
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

    print()
    print("=== Trial result ===")
    print(f"window: {window_start.date()} -> {window_end.date()}")
    print(f"annualized_IRR: {result['annualized_IRR']:.4f}")
    print(f"total_profit_VND: {result['total_profit_VND']:,.0f}")
    print(f"book_max_drawdown: {result['book_max_drawdown']:.4f}")
    print()
    print(f"Appended to {RESULTS_PATH}")
    print("config.yaml restored to its original contents.")
    print("(Raw IRR is not comparable across trials - different windows. "
          "Use run_config_suggest.bat for the per-knob analysis.)")


if __name__ == "__main__":
    main()
