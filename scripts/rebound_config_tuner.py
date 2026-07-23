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
import ctypes
import datetime
import json
import os
import random
import statistics
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    # Run at Idle priority so this loop doesn't compete with foreground work.
    # Self-demoting (rather than launching via `start /low`) keeps this process
    # a normal child of the launching console, so Ctrl+C still works — `start`'s
    # /b flag puts the child in its own process group, which Windows excludes
    # from console Ctrl+C broadcasts.
    import ctypes.wintypes as _wintypes

    _kernel32 = ctypes.windll.kernel32
    _kernel32.GetCurrentProcess.restype = _wintypes.HANDLE
    _kernel32.SetPriorityClass.argtypes = [_wintypes.HANDLE, _wintypes.DWORD]
    _kernel32.SetPriorityClass.restype = _wintypes.BOOL
    IDLE_PRIORITY_CLASS = 0x00000040
    _kernel32.SetPriorityClass(_kernel32.GetCurrentProcess(), IDLE_PRIORITY_CLASS)
except (AttributeError, OSError):
    pass  # not on Windows, or the call failed — not worth failing the trial over

import numpy as np
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
# KNOB_BOUNDS is the single source of truth for the independent scalar knobs'
# search space. Each spec is either
#   {"kind": "choice",  "values": [...]}                         — discrete grid
#   {"kind": "uniform", "low": x, "high": y, "digits": n}        — continuous
# Optional "sentinel": [...] marks special values (e.g. 0 = feature disabled)
# that rebound_config_suggest's range-boundary check must not treat as an
# extendable endpoint; optional "no_extend": ["lower"|"upper"] marks a side
# that physically cannot be widened (so the boundary check stays silent on
# it). Coordinated families (RSI, high-prox) are sampled
# separately below because their values must stay mutually consistent.
KNOB_BOUNDS = {
    # train_window_years can't shrink below 1y; oos_window_months can't exceed
    # the 1-year trial slice (WINDOW_YEARS) — those sides are not widenable.
    "backtest.train_window_years": {"kind": "choice", "values": [1, 2, 3, 4], "no_extend": ["lower"]},
    "backtest.oos_window_months": {"kind": "choice", "values": [3, 6, 9, 12], "no_extend": ["upper"]},
    "backtest.step_months": {"kind": "choice", "values": [1, 2, 3, 6]},
    "strategy.recovery.min_recovery_prob": {"kind": "uniform", "low": 0.70, "high": 0.99, "digits": 4},
    "strategy.recovery.p_quantile": {"kind": "uniform", "low": 0.50, "high": 0.85, "digits": 4},
    "strategy.recovery.profit_margin": {"kind": "uniform", "low": 0.002, "high": 0.02, "digits": 5},
    "strategy.recovery.min_ticker_obs": {"kind": "choice", "values": [50, 100, 150, 200, 300, 400]},
    "strategy.recovery.min_bucket_obs": {"kind": "choice", "values": [25, 50, 75, 100]},
    "strategy.recovery.min_ticker_bucket_obs": {"kind": "choice", "values": [15, 20, 30, 50, 75]},
    "strategy.recovery.vol_penalty.k": {"kind": "choice", "values": [0, 0.5, 1.0, 1.5, 2.0], "sentinel": [0]},
    "strategy.recovery.vol_penalty.measure": {"kind": "choice", "values": ["realvol_20", "atr_pct"]},
    "strategy.recovery.label_max_horizon": {"kind": "choice", "values": [120, 180, 250]},
    "universe.liquidity_filter.min_close_vnd": {"kind": "choice", "values": [3, 5, 7, 10]},
    "universe.liquidity_filter.min_adv_active_days": {"kind": "choice", "values": [5, 10, 15, 18, 20]},
    "universe.liquidity_filter.min_adv_vnd": {"kind": "choice", "values": [500_000, 1_000_000, 2_000_000, 5_000_000]},
    "universe.liquidity_filter.min_history_days": {"kind": "choice", "values": [120, 180, 250, 400]},
    "strategy.downtrend.mom20_max": {"kind": "uniform", "low": -0.05, "high": 0.05, "digits": 4},
    "strategy.downtrend.high_prox_max": {"kind": "uniform", "low": -0.15, "high": -0.02, "digits": 4},
    "pricing.overbought_rsi_max": {"kind": "choice", "values": [0, 70, 75, 80, 85], "sentinel": [0]},
    "pricing.ceiling_tol": {"kind": "uniform", "low": 0.005, "high": 0.03, "digits": 4},
    "pricing.max_participation_pct": {"kind": "uniform", "low": 0.5, "high": 3.0, "digits": 3},
    "pricing.corp_action_lookback": {"kind": "choice", "values": [15, 20, 25]},
}


def _make_sampler(spec: dict):
    if spec["kind"] == "choice":
        return lambda: random.choice(spec["values"])
    return lambda: round(random.uniform(spec["low"], spec["high"]), spec["digits"])


SIMPLE_KNOBS = [(path, _make_sampler(spec)) for path, spec in KNOB_BOUNDS.items()]


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


def _benchmark_irr(paths: dict, window_start: pd.Timestamp, window_end: pd.Timestamp) -> float:
    """Equal-weighted buy-and-hold return across every symbol with price data
    spanning [window_start, window_end], annualized by the ACTUAL day count of
    that slice (not an assumed 365) — works identically for any two dates,
    calendar-year-aligned or not, since it's just two lookups into the same
    per-symbol price paths the sim already built for this trial.

    Median (not mean) across symbols — robust to a handful of delistings or
    data gaps skewing the average. NaN if no symbol has valid data on both
    ends (e.g. the degenerate empty-paths case)."""
    start_np = np.datetime64(window_start)
    end_np = np.datetime64(window_end)
    rets = []
    for idx_arr, _o, _l, close_arr in paths.values():
        if idx_arr.size == 0:
            continue
        i0 = int(np.searchsorted(idx_arr, start_np))
        i1 = int(np.searchsorted(idx_arr, end_np, side="right")) - 1
        if i0 >= len(idx_arr) or i1 < 0 or i0 > i1:
            continue
        start_close = float(close_arr[i0])
        end_close = float(close_arr[i1])
        if start_close > 0:
            rets.append(end_close / start_close - 1.0)
    if not rets:
        return float("nan")
    median_ret = statistics.median(rets)
    days_span = (window_end - window_start).days
    if days_span <= 0:
        return float("nan")
    return (1.0 + median_ret) ** (365.0 / days_span) - 1.0


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

    # Everything from here on touches config.yaml, so it's all inside try/finally:
    # a Ctrl+C at any point (including mid-write) still restores the original.
    try:
        CONFIG_PATH.write_text(yaml.safe_dump(mutated, sort_keys=False), encoding="utf-8")
        load_config.cache_clear()

        print("Randomized config for this trial:")
        print(json.dumps(flat, indent=2, default=str))
        print(f"window: {window_start.date()} -> {window_end.date()}")
        print()

        from scripts.rebound_sim_include_held import _build_data, simulate

        print("Building data (retrains the recovery model per anchor)...", flush=True)
        data = _build_data(start=window_start.strftime("%Y-%m-%d"),
                           end=window_end.strftime("%Y-%m-%d"))
        print("Running simulation...", flush=True)
        # Pass the trial window so the sim annualizes IRR over the whole
        # window, not the (possibly tiny) traded span.
        result = simulate(data=data,
                          start=window_start.strftime("%Y-%m-%d"),
                          end=window_end.strftime("%Y-%m-%d"))
        benchmark_irr = _benchmark_irr(data[1], window_start, window_end)
    finally:
        CONFIG_PATH.write_bytes(original_bytes)
        load_config.cache_clear()

    excess_irr = result["annualized_IRR"] - benchmark_irr

    record = {
        "timestamp": timestamp,
        "window_start": window_start.strftime("%Y-%m-%d"),
        "window_end": window_end.strftime("%Y-%m-%d"),
        "config": flat,
        "result": result,
        "benchmark_irr": benchmark_irr,
        "excess_irr": excess_irr,
    }
    # A prior trial killed mid-write (e.g. the console window's X button sends
    # CTRL_CLOSE_EVENT) can leave a partial line with no trailing newline. If we
    # appended straight onto it, the two records would merge into one garbage
    # line, destroying this good record too. So first ensure the file ends in a
    # newline, isolating any truncated line as its own (skippable) bad line.
    if RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size > 0:
        with open(RESULTS_PATH, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"
    else:
        needs_newline = False
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())

    print()
    print("=== Trial result ===")
    print(f"window: {window_start.date()} -> {window_end.date()}")
    print(f"annualized_IRR: {result['annualized_IRR']:.4f}")
    print(f"benchmark_irr (buy-and-hold, same window): {benchmark_irr:.4f}")
    print(f"excess_irr (config skill vs. just holding): {excess_irr:.4f}")
    print(f"total_profit_VND: {result['total_profit_VND']:,.0f}")
    print(f"book_max_drawdown: {result['book_max_drawdown']:.4f}")
    print()
    print(f"Appended to {RESULTS_PATH}")
    print("config.yaml restored to its original contents.")
    print("(Raw IRR is not comparable across trials - different windows. "
          "Use run_config_suggest.bat for the per-knob analysis.)")


if __name__ == "__main__":
    main()
