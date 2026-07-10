"""One-off repair: reset ledger rows whose 'recovery' was faked by a phantom bar.

A raw-VND bar (e.g. TCI 10,450 instead of 10.45) in the OHLCV cache let
``resolve_exit`` book an impossible recovery before the band-break guard was
added (see src/stockpredict/model/target.py). Those rows stay ``evaluated=True``
forever, poisoning every pooled stat. This script finds evaluated rebound rows
with an impossible ``realized_return`` (|r| > 0.5 — far beyond any
band-compliant single trade), prints them for review, and resets them to
unevaluated so the next ``evaluate`` run re-resolves them against the healed
cache. Backs up predictions.parquet first.

Usage:  .venv\\Scripts\\python.exe -m scripts.repair_ledger_band_break [--apply]

Without --apply it only prints what would be reset (dry run).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

LEDGER = Path("cache") / "predictions.parquet"
IMPOSSIBLE_ABS_RETURN = 0.5

SHOW_COLS = ["run_id", "symbol", "as_of", "entry_price", "actual_exit",
             "actual_exit_date", "realized_return", "recovered_flag",
             "exit_reason", "evaluated"]
RESET_VALUES = {
    "evaluated": False,
    "recovered_flag": False,
    "actual_exit": float("nan"),
    "actual_exit_date": pd.NaT,
    "realized_return": float("nan"),
    "exit_reason": "",
}


def main() -> int:
    apply = "--apply" in sys.argv[1:]
    if not LEDGER.exists():
        print(f"ledger not found: {LEDGER}")
        return 1
    df = pd.read_parquet(LEDGER)

    is_rebound = df["pred_profit"].notna() if "pred_profit" in df.columns \
        else pd.Series(False, index=df.index)
    bad = (
        is_rebound
        & df["evaluated"].fillna(False).astype(bool)
        & (df["realized_return"].abs() > IMPOSSIBLE_ABS_RETURN)
    )
    n = int(bad.sum())
    if n == 0:
        print("no corrupt rows found (evaluated rebound rows with "
              f"|realized_return| > {IMPOSSIBLE_ABS_RETURN}); nothing to do")
        return 0

    print(f"found {n} corrupt row(s):")
    cols = [c for c in SHOW_COLS if c in df.columns]
    print(df.loc[bad, cols].to_string(index=False))

    if not apply:
        print("\ndry run — re-run with --apply to reset these rows "
              "(a .bak backup is written first)")
        return 0

    bak = LEDGER.with_suffix(".parquet.bak")
    shutil.copy2(LEDGER, bak)
    print(f"\nbackup -> {bak}")

    for col, val in RESET_VALUES.items():
        if col in df.columns:
            df.loc[bad, col] = val
    df.to_parquet(LEDGER, index=False)

    print("after reset:")
    print(df.loc[bad, cols].to_string(index=False))
    print("\ndone — run `.venv\\Scripts\\python.exe -m stockpredict.cli evaluate` "
          "to re-resolve these picks against the healed cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
