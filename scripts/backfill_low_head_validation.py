"""One-off: backfill the new empirical low head over historical buy days and
compare its skill against the old LightGBM head recorded in the ledger.

For every claude-mode ledger row that has a stamped T+0 fill (t0_evaluated,
entry_limit_price, t0_low all present), recompute what the NEW
RollingEmpiricalQuantileModel would have quoted for that (symbol, as_of) using
only data known by as_of (lookahead-safe). Then report, old vs new:

  - corr(pred_low, actual_dip)     : skill (want >= 0; old was -0.39)
  - corr(pred_low, overnight gap)  : momentum inversion (old was -0.53)
  - pinball loss at alpha          : want new < fixed-constant baseline (0.0142)
  - fill_rate                      : share of limits reachable at the realized low

Run via the project venv python on scripts/backfill_low_head_validation.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stockpredict.model.train import RollingEmpiricalQuantileModel, latest_low_model_path

# Repo root = scripts/'s parent, so the script runs regardless of the clone path.
_REPO_ROOT = Path(__file__).resolve().parents[1]


def pinball(y: np.ndarray, f: np.ndarray, alpha: float) -> float:
    d = y - f
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def main() -> None:
    model = RollingEmpiricalQuantileModel.load(latest_low_model_path())
    alpha = model.alpha
    print(f"new model: alpha={alpha:.2f} lookback={model.lookback}d "
          f"min_obs={model.min_obs} global_quantile={model.global_quantile:.4f}")

    df = pd.read_parquet(_REPO_ROOT / "cache" / "predictions.parquet")
    p = df[df["t0_evaluated"] & df["entry_limit_price"].notna()
           & df["t0_low"].notna() & (df["mode"] == "claude")].copy()
    p["as_of"] = pd.to_datetime(p["as_of"])
    p["actual_dip"] = p["t0_low"] / p["entry_price"] - 1.0
    p["gap"] = p["t0_open"] / p["entry_price"] - 1.0

    # Recompute new pred_low as-of each buy day (history=None → reads OHLCV
    # cache; predict slices strictly before as_of, so no lookahead).
    X = pd.DataFrame({"symbol": p["symbol"].to_numpy()},
                     index=pd.DatetimeIndex(p["as_of"].to_numpy(), name="date"))
    p["pred_low_new_raw"] = model.predict(X, history=None).to_numpy()
    # Effective limit dip after pricing's clip(upper=0).
    p["pred_low_new"] = p["pred_low_new_raw"].clip(upper=0.0)
    # New limit would fill iff the realized low reaches it.
    p["fill_new"] = p["actual_dip"] <= p["pred_low_new"]

    y = p["actual_dip"].to_numpy()
    const = float(np.quantile(y, alpha))

    def block(label: str, pred: np.ndarray, filled: np.ndarray) -> None:
        pred = np.asarray(pred, dtype=float)
        print(f"\n[{label}] n={len(pred)}")
        print(f"  corr(pred, actual_dip) = {np.corrcoef(pred, y)[0,1]:+.3f}")
        print(f"  corr(pred, gap)        = {np.corrcoef(pred, p['gap'].to_numpy())[0,1]:+.3f}")
        print(f"  pinball(alpha={alpha:.2f})   = {pinball(y, pred, alpha):.5f}")
        print(f"  fill_rate              = {np.mean(filled):.3f}")

    block("OLD (LightGBM, from ledger)",
          p["pred_low"].to_numpy(), p["entry_limit_filled"].to_numpy())
    block("NEW (empirical, backfilled)",
          p["pred_low_new"].to_numpy(), p["fill_new"].to_numpy())
    print(f"\nfixed-constant baseline pinball (q{alpha:.2f}={const:.4f}) = "
          f"{pinball(y, np.full(len(y), const), alpha):.5f}")


if __name__ == "__main__":
    main()
