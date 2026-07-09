"""One-off: scan the OHLCV cache for phantom-instrument price injections and
remove them.

Root cause (confirmed live on ABB, 2026-07): MSN's ticker resolution is
unreliable and can silently return prices for the WRONG underlying
instrument for a stretch of days (confirmed: MSN returned ~600-900-range
closes for ABB, a ~5-18 range stock, across several holiday weeks -- VCI and
KBS both independently confirm those exact dates aren't real trading days
at all, i.e. Tet/holiday closures MSN fabricated data for). If an
incremental daily fetch happens to route to MSN on one of those days, the
wrong-instrument price gets permanently written into an otherwise-correct
series, producing an isolated spike-then-revert plateau: a price level jump
beyond the exchange band, held for a few days, then a reversal back to
close to the pre-spike level once a healthy source resumes.

This is DELIBERATELY narrower than "any single day's move exceeds the
exchange band" (an earlier version of this script used that criterion and
flagged ~1343/~1550 symbols in the whole cache -- almost everyone, because
that band-violation-anywhere-in-history signature also matches ordinary,
already-correctly-handled unadjusted stock dividends/splits, which ARE real
and permanent, not phantom). Requiring a REVERSAL back toward the pre-spike
level distinguishes a phantom injection (temporary, self-reverting) from a
genuine corporate action (permanent level change) -- a real dividend/split
never reverts.

Healing: a full re-fetch (update_many(..., full=True)) CANNOT fix a phantom
row whose date isn't a real trading day at all -- there's no legitimate
value from any source to overwrite it with (confirmed: neither VCI nor KBS
have ANY row for ABB's phantom dates). So instead of refetching, this script
directly DROPS phantom rows once both VCI and KBS confirm the date has no
legitimate data -- see ``_is_confirmed_phantom_date``. If either source DOES
have real data for a flagged date (a case this script hasn't hit in
practice but is a fine fallback if the spike is real per-source noise
rather than a phantom instrument), that value replaces the spike instead of
being dropped.

Run via the project venv:
    .venv\\Scripts\\python.exe scripts/repair_corporate_action_corruption.py
"""
from __future__ import annotations

import pandas as pd

from stockpredict.data.cache import _corp_action_threshold, ohlcv_dir, read_ohlcv, write_ohlcv
from stockpredict.data.fetcher import _quote_history

# How many trading days ahead to look for a reversal back toward the
# pre-spike level. MSN's confirmed incidents on ABB spanned up to 5
# consecutive days (a full holiday week); 10 gives headroom without
# drifting into "this is actually a new, permanent level" territory.
_REVERSAL_LOOKAHEAD = 10
# A spike has "reverted" if the price gets back within this fraction of the
# pre-spike close. Must be much tighter than the corp-action band threshold
# itself (~15-20%+margin) -- an earlier version used 0.25, which is roughly
# the SAME magnitude as a real permanent corporate-action jump, so almost
# any genuine, permanent level change also happened to land "within 25%" of
# its pre-spike close and got misclassified as a reverted phantom (confirmed
# live: flagged 98,720 dates / 1275 symbols in the real cache, almost
# entirely ordinary low-volatility noise plus real corp actions, not
# phantom injections). 0.05 is tight enough that only a genuine return to
# the old regime satisfies it.
_REVERSION_TOLERANCE = 0.05
# Requiring just ONE bar in the lookahead window to be close is still too
# loose (a single noisy bar can randomly land near the old level even
# without a real reversion). Require at least this many bars within
# tolerance before calling it reverted.
_MIN_REVERTED_BARS = 3


def find_phantom_spike_symbols() -> dict[str, list]:
    """Return {symbol: [phantom_dates]} for every symbol whose cached OHLCV
    shows a spike-then-revert plateau: a move beyond the exchange band that
    reverts back near the pre-spike level within _REVERSAL_LOOKAHEAD days.
    Unlike a bare "any band violation anywhere" check, this does NOT flag a
    genuine permanent corporate-action level change (which never reverts)."""
    flagged: dict[str, list] = {}
    for p in sorted(ohlcv_dir().glob("*.parquet")):
        symbol = p.stem
        df = read_ohlcv(symbol)
        if len(df) < 3 or "close" not in df.columns:
            continue
        threshold = _corp_action_threshold(symbol)
        closes = df["close"].astype(float)
        moves = closes.pct_change().abs()
        violation_positions = moves.index[moves > threshold]
        phantom_dates = []
        for spike_date in violation_positions:
            pos = df.index.get_loc(spike_date)
            if pos == 0:
                continue
            pre_spike_close = closes.iloc[pos - 1]
            window_end = min(pos + _REVERSAL_LOOKAHEAD, len(closes) - 1)
            lookahead = closes.iloc[pos:window_end + 1]
            # The spike bar itself hasn't reverted (it IS the spike), so
            # check from pos+1 onward -- need MULTIPLE later bars landing
            # back near pre_spike_close (not just one noisy touch) to call
            # this a temporary plateau rather than a permanent level change.
            if len(lookahead) <= 1:
                continue
            near_old_level = (
                (lookahead.iloc[1:] - pre_spike_close).abs() <= pre_spike_close * _REVERSION_TOLERANCE
            )
            reverted_after = near_old_level.sum() >= _MIN_REVERTED_BARS
            if reverted_after:
                phantom_dates.append(df.index[pos])
        if phantom_dates:
            flagged[symbol] = phantom_dates
    return flagged


def _is_confirmed_phantom_date(symbol: str, date) -> tuple[bool, float | None]:
    """Check VCI and KBS directly for real data on ``date``. Returns
    (is_phantom, replacement_close). is_phantom=True means neither source
    has any data for that exact date (confirmed not a real trading day --
    drop the row). If a source DOES have data, returns (False, that close)
    so the caller can replace the spike with a legitimate value instead."""
    date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
    for src in ("VCI", "KBS"):
        try:
            df = _quote_history(symbol, src, date_str, date_str, "1D", True)
        except Exception:
            continue
        if df is not None and len(df) > 0:
            return False, float(df["close"].iloc[0])
    return True, None


def main() -> None:
    flagged = find_phantom_spike_symbols()
    total_dates = sum(len(v) for v in flagged.values())
    print(f"{len(flagged)} symbol(s), {total_dates} phantom-spike date(s) flagged.")
    if not flagged:
        return

    healed_symbols = 0
    for symbol, dates in flagged.items():
        df = read_ohlcv(symbol)
        changed = False
        for date in dates:
            is_phantom, replacement = _is_confirmed_phantom_date(symbol, date)
            if is_phantom:
                df = df.drop(index=date)
                changed = True
                print(f"  {symbol} {date.date()}: dropped (no source has this date)")
            elif replacement is not None:
                df.loc[date, "close"] = replacement
                changed = True
                print(f"  {symbol} {date.date()}: replaced with {replacement} (source has real data)")
        if changed:
            write_ohlcv(symbol, df)
            healed_symbols += 1

    print(f"Healed {healed_symbols}/{len(flagged)} symbols.")
    still_bad = find_phantom_spike_symbols()
    if still_bad:
        print(f"WARNING: {len(still_bad)} symbol(s) still show the phantom-spike "
              f"signature after cleanup: {list(still_bad)}")
    else:
        print("All flagged symbols are clean after cleanup.")


if __name__ == "__main__":
    main()
