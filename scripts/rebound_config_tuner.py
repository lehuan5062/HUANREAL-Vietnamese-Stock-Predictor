"""One-shot randomized-config trial for rebound_sim_include_held.

Picks ONE random combination of the backtest walk-forward windows and
recovery-model knobs, writes it to config.yaml, runs the sim once, appends
the (config, result) pair to a results file, then ALWAYS restores the
original config.yaml — even on error or Ctrl+C.

The real config.yaml is never left changed. Re-run this (e.g. via
run_config_tuner.bat) as many times as you want to accumulate trials, then
inspect reports/tuning/rebound_include_held_search.jsonl to find the
randomized combination with the best annualized_IRR.

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

import yaml

from stockpredict import PROJECT_ROOT
from stockpredict.config import load_config

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
BACKUP_DIR = PROJECT_ROOT / "reports" / "tuning" / "config_backups"
RESULTS_PATH = PROJECT_ROOT / "reports" / "tuning" / "rebound_include_held_search.jsonl"

# Randomization ranges. Adjust here if you want to narrow/widen the search.
TRAIN_WINDOW_YEARS_CHOICES = [1, 2, 3, 4]
OOS_WINDOW_MONTHS_CHOICES = [3, 6, 9, 12]
STEP_MONTHS_CHOICES = [1, 2, 3, 6]
MIN_RECOVERY_PROB_RANGE = (0.70, 0.99)
P_QUANTILE_RANGE = (0.50, 0.85)
PROFIT_MARGIN_RANGE = (0.002, 0.02)


def _sample_overrides() -> dict:
    return {
        "backtest": {
            "train_window_years": random.choice(TRAIN_WINDOW_YEARS_CHOICES),
            "oos_window_months": random.choice(OOS_WINDOW_MONTHS_CHOICES),
            "step_months": random.choice(STEP_MONTHS_CHOICES),
        },
        "strategy": {
            "recovery": {
                "min_recovery_prob": round(random.uniform(*MIN_RECOVERY_PROB_RANGE), 4),
                "p_quantile": round(random.uniform(*P_QUANTILE_RANGE), 4),
                "profit_margin": round(random.uniform(*PROFIT_MARGIN_RANGE), 5),
            }
        },
    }


def _apply_overrides(raw: dict, overrides: dict) -> dict:
    mutated = copy.deepcopy(raw)
    mutated["backtest"]["train_window_years"] = overrides["backtest"]["train_window_years"]
    mutated["backtest"]["oos_window_months"] = overrides["backtest"]["oos_window_months"]
    mutated["backtest"]["step_months"] = overrides["backtest"]["step_months"]
    rec = mutated["strategy"]["recovery"]
    rec["min_recovery_prob"] = overrides["strategy"]["recovery"]["min_recovery_prob"]
    rec["p_quantile"] = overrides["strategy"]["recovery"]["p_quantile"]
    rec["profit_margin"] = overrides["strategy"]["recovery"]["profit_margin"]
    return mutated


def main():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    original_bytes = CONFIG_PATH.read_bytes()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"config_{timestamp}.yaml"
    backup_path.write_bytes(original_bytes)

    overrides = _sample_overrides()
    raw = yaml.safe_load(original_bytes.decode("utf-8"))
    mutated = _apply_overrides(raw, overrides)
    CONFIG_PATH.write_text(yaml.safe_dump(mutated, sort_keys=False), encoding="utf-8")
    load_config.cache_clear()

    print("Randomized config for this trial:")
    print(json.dumps(overrides, indent=2))
    print()

    try:
        from scripts.rebound_sim_include_held import _build_data, simulate

        print("Building data (this retrains the recovery model per anchor)...")
        data = _build_data()
        print("Running simulation...")
        result = simulate(data=data)
    finally:
        CONFIG_PATH.write_bytes(original_bytes)
        load_config.cache_clear()

    flat_config = {
        "train_window_years": overrides["backtest"]["train_window_years"],
        "oos_window_months": overrides["backtest"]["oos_window_months"],
        "step_months": overrides["backtest"]["step_months"],
        "min_recovery_prob": overrides["strategy"]["recovery"]["min_recovery_prob"],
        "p_quantile": overrides["strategy"]["recovery"]["p_quantile"],
        "profit_margin": overrides["strategy"]["recovery"]["profit_margin"],
    }
    record = {
        "timestamp": timestamp,
        "config": flat_config,
        "result": result,
    }
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

    print()
    print("=== Trial result ===")
    print(json.dumps(flat_config, indent=2))
    print(f"annualized_IRR: {result['annualized_IRR']:.4f}")
    print(f"total_profit_VND: {result['total_profit_VND']:,.0f}")
    print(f"book_max_drawdown: {result['book_max_drawdown']:.4f}")
    print()
    print(f"Appended to {RESULTS_PATH}")
    print("config.yaml restored to its original contents.")


if __name__ == "__main__":
    main()
