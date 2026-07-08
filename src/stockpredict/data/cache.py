"""Parquet cache for OHLCV. One file per ticker, indexed by date."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import cache_dir


def ohlcv_dir() -> Path:
    d = cache_dir() / "ohlcv"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ohlcv_path(symbol: str) -> Path:
    return ohlcv_dir() / f"{symbol.upper()}.parquet"


def read_ohlcv(symbol: str) -> pd.DataFrame:
    p = ohlcv_path(symbol)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df.sort_index()


def write_ohlcv(symbol: str, df: pd.DataFrame) -> None:
    """Persist the merged OHLCV frame for `symbol`.

    Atomic write: parquet is written to `<symbol>.parquet.tmp` then
    renamed onto the final path. This guarantees that if the process is
    killed (Ctrl+C, OS shutdown) mid-write, the on-disk file is either
    the previous complete version or the new complete version — never a
    partial / corrupt parquet that fails to read on the next run."""
    import os

    if df.empty:
        return
    out = df.copy()
    if out.index.name != "date":
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"])
            out = out.set_index("date")
    # Normalize to calendar date: some fetches stamp midnight, others stamp
    # a mid-morning ICT time for the same trading day. Keeping both as
    # distinct index values lets a same-day bar satisfy `index > as_of`
    # (as_of normalized to midnight) in tracking.py's exit scan, producing
    # phantom same-day "recoveries" that are impossible to actually trade.
    out.index = out.index.normalize()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    target = ohlcv_path(symbol)
    tmp = target.with_suffix(target.suffix + ".tmp")
    out.reset_index().to_parquet(tmp, index=False)
    # os.replace is atomic on POSIX and Windows (since Python 3.3+).
    os.replace(tmp, target)


def merge_ohlcv(symbol: str, new: pd.DataFrame) -> pd.DataFrame:
    """Append new rows into the cached parquet, dedupe on date, return merged frame."""
    existing = read_ohlcv(symbol)
    if new.empty:
        # Nothing to add (e.g. a thin ticker with no matched trades in the
        # fetched window) — skip the write and the concat entirely. concat
        # with an empty/all-NA frame is a no-op here but triggers pandas'
        # FutureWarning on every call, which spams the console across a
        # large batch with many quiet tickers.
        return existing
    if existing.empty:
        merged = new
    else:
        merged = pd.concat([existing, new])
    write_ohlcv(symbol, merged)
    return read_ohlcv(symbol)


def cached_symbols() -> list[str]:
    return sorted(p.stem for p in ohlcv_dir().glob("*.parquet"))


# ---- fetch watermarks --------------------------------------------------
#
# We persist "we attempted to fetch <symbol> through date D" so that
# permanently-stuck tickers (delisted, halted, absent from the feed) don't
# trigger a wasted fetch on every run. Each attempt — successful or not —
# stamps the watermark to `latest_expected_bar_date()` for that ticker.
# When the expected bar advances (next trading day closes), the watermark
# falls behind and the ticker gets one fresh attempt.
#
# Storage: cache/watermarks/{SYMBOL}.txt — a single ISO date string per
# file. Tiny (~10 bytes × ~1,500 = ~15 KB total), atomic per-symbol writes,
# thread-safe because each thread writes its own file.

import datetime as _dt


def _watermark_dir() -> Path:
    d = cache_dir() / "watermarks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _watermark_path(symbol: str) -> Path:
    return _watermark_dir() / f"{symbol.upper()}.txt"


def get_watermark(symbol: str) -> _dt.date | None:
    """Return the date through which we've attempted to fetch this symbol,
    or None if no attempt has ever been recorded."""
    p = _watermark_path(symbol)
    if not p.exists():
        return None
    try:
        return _dt.date.fromisoformat(p.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def set_watermark(symbol: str, attempted_through: _dt.date) -> None:
    """Stamp the latest-attempted-through date for this symbol. Atomic
    write so a Ctrl+C can't leave the watermark file half-written."""
    import os

    p = _watermark_path(symbol)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        tmp.write_text(attempted_through.isoformat(), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        # Persisted watermarks are best-effort; if disk is full we still
        # want the run to succeed rather than crashing.
        pass
