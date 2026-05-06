"""Enumerate the investable Vietnamese stock universe across HOSE/HNX/UPCOM."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from ..config import cache_dir, load_config


_UNIVERSE_FILE = "universe.parquet"


def universe_path() -> Path:
    return cache_dir() / _UNIVERSE_FILE


def fetch_universe(retries: int = 3, source: str | None = None) -> pd.DataFrame:
    """Pull the full ticker list via vnstock. Falls back across data sources.
    Pass ``source`` to force a specific provider (e.g. 'VCI' to try to pick up
    the ``exchange`` column when KBS is the configured default).

    Listing calls share the broker's per-IP rate window with Quote calls.
    We go through the global limiter so a fresh process doesn't burn its
    first few quota slots on a Listing fetch right before the OHLCV burst."""
    from vnstock import Listing  # imported lazily so tests don't need vnstock

    from .fetcher import _limiter, _looks_like_rate_limit

    cfg = load_config()
    # Listing API only supports KBS / VCI / MSN per vnstock 4.x
    if source and source.upper() in ("KBS", "VCI", "MSN"):
        pref = source.upper()
    else:
        pref = cfg.data["source"] if cfg.data["source"] in ("KBS", "VCI", "MSN") else "VCI"
    sources = [pref] + [s for s in ("VCI", "KBS") if s != pref]
    last_err: Exception | None = None
    for src in sources:
        for attempt in range(retries):
            try:
                _limiter().wait()
                df = Listing(source=src).all_symbols()
                if df is None or len(df) == 0:
                    continue
                df = _normalize_listing(df)
                df["fetched_at"] = dt.datetime.utcnow().isoformat()
                df["source"] = src
                return df
            except Exception as e:  # network / API drift / rate limit
                last_err = e
                if _looks_like_rate_limit(e):
                    _limiter().pause(65.0, reason=f"{src} Listing 429")
                continue
    raise RuntimeError(f"All vnstock sources failed for Listing: {last_err}")


def _normalize_listing(df: pd.DataFrame) -> pd.DataFrame:
    """vnstock returns slightly different column names per source — normalize."""
    rename_map = {
        "ticker": "symbol",
        "Symbol": "symbol",
        "exchange": "exchange",
        "comGroupCode": "exchange",
        "organ_name": "organ_name",
        "organName": "organ_name",
    }
    df = df.rename(columns=rename_map)
    if "symbol" not in df.columns:
        # Some sources put the ticker into the index
        df = df.reset_index().rename(columns={"index": "symbol"})
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    keep = [c for c in ["symbol", "exchange", "organ_name"] if c in df.columns]
    df = df[keep].drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    return df


def load_universe(refresh: bool = False, source: str | None = None) -> pd.DataFrame:
    p = universe_path()
    if not refresh and p.exists():
        return pd.read_parquet(p)
    df = fetch_universe(source=source)
    df.to_parquet(p, index=False)
    return df


def filter_exchanges(df: pd.DataFrame, exchanges: list[str]) -> pd.DataFrame:
    if "exchange" not in df.columns:
        return df  # if the source didn't return exchange, keep everything
    wanted = {e.upper() for e in exchanges} | {"HOSE"}  # alias HSX/HOSE
    return df[df["exchange"].astype(str).str.upper().isin(wanted)].copy()
