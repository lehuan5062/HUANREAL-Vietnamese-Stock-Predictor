"""Enumerate the investable Vietnamese stock universe across HOSE/HNX/UPCOM."""
from __future__ import annotations

import datetime as dt
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..config import cache_dir, load_config


_UNIVERSE_FILE = "universe.parquet"

# Symbol-shape heuristics for when vnstock's Listing response lacks an
# explicit security-type column. Vietnamese open-ended ETFs all start
# with `FUE`; closed-end fund certificates / REITs start with `FUC`;
# `E1VFVN30` is the legacy DCVFM VN30 ETF (predates the FUE convention).
# Both FUE and FUC trade on HOSE with 10-unit lots, so we classify both
# as ETF for sizing purposes — vnstock's all_etf() also groups them
# together under one listing.  Covered warrants encode the issuer +
# maturity as `C<XXX><YYMM>` (e.g. `CFPT2502`).
_ETF_SYMBOL_RE = re.compile(r"^(FUE|FUC|E1VFVN30)", re.IGNORECASE)
_CW_SYMBOL_RE = re.compile(r"^C[A-Z]{2,4}\d{4}$", re.IGNORECASE)

# Possible column names vnstock returns for the security-type field.
# VCI tends to use `type`; KBS / MSN drift across releases. We pick the
# first one present.
_TYPE_COLUMN_CANDIDATES = (
    "type", "securityType", "instrumentType", "productType",
    "security_type", "instrument_type", "product_type", "asset_type",
)

# Exchange labels we accept as "tradable on a Vietnamese retail broker".
# vnstock's VCI source uses HSX; KBS uses HOSE — both alias to the same
# exchange (we already canonicalize via ``filter_exchanges`` downstream).
# Anything else (DELISTED, BOND, XHNF, etc.) is dropped at fetch time so
# names like HTK never reach the candidate pool.
_TRADABLE_EXCHANGES = {"HSX", "HOSE", "HNX", "UPCOM"}

# Raw type values we drop unconditionally. Match is exact (after upper())
# so "FU" (futures) does NOT accidentally swallow "FUND" (ETFs).
_NONEQUITY_RAW_TYPES = {
    # VCI uppercase
    "BOND", "CW", "FU", "DEBENTURE",
    # KBS lowercase variants — covered by upper() at compare time
    "CORPBOND", "FUTURE",
    # Generic safety net
    "WARRANT", "DERIVATIVE",
}


def universe_path() -> Path:
    return cache_dir() / _UNIVERSE_FILE


def classify_symbol(symbol: str) -> str:
    """Pure-regex security-type classifier used as a fallback when the
    vnstock Listing response doesn't include a type column. Returns one
    of: 'STOCK', 'ETF', 'CW', 'OTHER'."""
    if not symbol:
        return "OTHER"
    s = str(symbol).upper().strip()
    if _ETF_SYMBOL_RE.match(s):
        return "ETF"
    if _CW_SYMBOL_RE.match(s):
        return "CW"
    # Vietnamese ordinary share tickers are 3 uppercase letters.
    if re.match(r"^[A-Z]{3}$", s):
        return "STOCK"
    return "OTHER"


def _normalize_type_value(value: object) -> str:
    """Coerce whatever the broker returned in the type column into our
    canonical labels. Accepts variants like 'Stock', 'CỔ PHIẾU', 'ETF',
    'FUND', 'CW', 'COVERED_WARRANT', etc."""
    if value is None:
        return "OTHER"
    s = str(value).strip().upper()
    if not s or s == "NAN":
        return "OTHER"
    if "ETF" in s or "FUND" in s or "QUỸ" in s or "QUY" in s:
        return "ETF"
    if "CW" in s or "WARRANT" in s or "CHỨNG QUYỀN" in s or "CHUNG QUYEN" in s:
        return "CW"
    if "STOCK" in s or "EQUIT" in s or "SHARE" in s or "CỔ PHIẾU" in s or "CO PHIEU" in s:
        return "STOCK"
    return "OTHER"


def fetch_universe(retries: int = 3, source: str | None = None) -> pd.DataFrame:
    """Pull the full ticker list via vnstock from both KBS and VCI sources,
    consolidated to capture stocks that may be missing or corrupt on one source.

    Note: the ``source`` parameter is now ignored and kept only for backwards
    compatibility. Both KBS and VCI are fetched and consolidated.

    MSN was removed from this fallback chain 2026-07-09: confirmed 3-for-3
    on real OHLCV corruption incidents this session (ABB, USC, EMS) via
    fetch_history/Quote, silently returning fabricated/wrong-instrument
    prices with no error. This function only calls Listing (ticker lists,
    not prices), a different, unverified endpoint — but given MSN's
    confirmed unreliability elsewhere, it's dropped here too rather than
    trusted by default until proven otherwise.

    Prefers ``Listing.symbols_by_exchange()`` over ``all_symbols()`` because the
    former returns explicit ``exchange`` / ``type`` columns we use to drop
    delisted tickers and non-equity instruments (covered warrants, futures,
    bonds). Without that filter, vnstock returns rows like HTK (delisted) and
    the pipeline writes picks for tickers the broker won't accept orders for.

    Also unions in the dedicated ETF listing from ``Listing.all_etf()`` (only
    supported by the KBS source — VCI doesn't have this endpoint at all).
    ``all_symbols()`` returns common stocks only — ETFs / fund certificates
    live on a separate endpoint and would be invisible without this second call.

    Listing calls share the broker's per-IP rate window with Quote calls.
    We go through the global limiter so a fresh process doesn't burn its
    first few quota slots on a Listing fetch right before the OHLCV burst."""
    from vnstock import Listing  # imported lazily so tests don't need vnstock

    from .fetcher import _limiter, _looks_like_rate_limit

    # Fetch from both KBS and VCI to consolidate listings
    sources = ["KBS", "VCI"]
    results = {}
    last_err: Exception | None = None

    for src in sources:
        for attempt in range(retries):
            try:
                _limiter().wait()
                listing = Listing(source=src)
                # Prefer the exchange-aware endpoint so we can drop DELISTED
                # tickers; fall back to all_symbols() if the source / vnstock
                # release doesn't implement it.
                df = _try_symbols_by_exchange(listing)
                if df is None or len(df) == 0:
                    df = listing.all_symbols()
                if df is None or len(df) == 0:
                    continue
                df = _drop_untradable(df)
                df = _normalize_listing(df)
                results[src] = df
                break
            except SystemExit as e:
                # See fetcher._disable_vnstock_hard_exit — vnstock's
                # CleanErrorContext sys.exit()s on rate-limit. Treat as 429.
                last_err = e
                _limiter().pause(65.0, reason=f"{src} Listing hard-exit")
                continue
            except Exception as e:  # network / API drift / rate limit
                last_err = e
                if _looks_like_rate_limit(e):
                    _limiter().pause(65.0, reason=f"{src} Listing 429")
                continue

    # If we got at least one source, consolidate; if only one, use it directly
    if len(results) == 0:
        raise RuntimeError(f"All vnstock sources failed for Listing: {last_err}")
    elif len(results) == 1:
        df = list(results.values())[0]
    else:
        # Consolidate both sources
        df = _merge_stock_listings(results["KBS"], results["VCI"])

    # Union in the dedicated ETF listing. Failure is non-fatal —
    # the stock listing is the primary product; ETF augmentation
    # is best-effort. Without it the curated HOSE_ETFS list still
    # supplies the 10 most-liquid ETFs via the selector.
    etf_df = _try_fetch_etf_listing()
    if etf_df is not None and not etf_df.empty:
        df = _merge_etf_rows(df, etf_df)
    df["fetched_at"] = dt.datetime.utcnow().isoformat()
    # When consolidated from both, mark as such; otherwise mark the single source
    df["source"] = "KBS+VCI" if len(results) > 1 else list(results.keys())[0]
    return df


def _merge_stock_listings(kbs_df: pd.DataFrame, vci_df: pd.DataFrame) -> pd.DataFrame:
    """Consolidate stock listings from KBS and VCI sources.

    Deduplicates on symbol by keeping the row with more non-null fields
    (more complete data), preferring VCI if both have equal completeness.
    For the consolidated row, fills missing fields from the other source.
    Re-classifies instrument types using existing symbol patterns."""
    if kbs_df is None or kbs_df.empty:
        return vci_df
    if vci_df is None or vci_df.empty:
        return kbs_df

    # Normalize symbol column to uppercase for dedup matching
    kbs = kbs_df.copy()
    vci = vci_df.copy()
    kbs["symbol"] = kbs["symbol"].astype(str).str.upper().str.strip()
    vci["symbol"] = vci["symbol"].astype(str).str.upper().str.strip()

    # Union all symbols from both sources
    all_symbols = set(kbs["symbol"].tolist()) | set(vci["symbol"].tolist())
    rows = []

    for sym in sorted(all_symbols):
        kbs_row = kbs[kbs["symbol"] == sym]
        vci_row = vci[vci["symbol"] == sym]
        kbs_has = kbs_row.shape[0] > 0
        vci_has = vci_row.shape[0] > 0

        if kbs_has and vci_has:
            # Both have the symbol — keep the more complete row
            kbs_nulls = kbs_row.iloc[0].isna().sum()
            vci_nulls = vci_row.iloc[0].isna().sum()

            if vci_nulls < kbs_nulls:
                # VCI has fewer nulls, use as base and fill from KBS
                row = vci_row.iloc[0].copy()
                for col in row.index:
                    if pd.isna(row[col]) and col in kbs_row.columns:
                        row[col] = kbs_row.iloc[0][col]
            else:
                # KBS has equal or fewer nulls, use as base and fill from VCI
                row = kbs_row.iloc[0].copy()
                for col in row.index:
                    if pd.isna(row[col]) and col in vci_row.columns:
                        row[col] = vci_row.iloc[0][col]
            rows.append(row)
        elif kbs_has:
            rows.append(kbs_row.iloc[0])
        else:
            rows.append(vci_row.iloc[0])

    out = pd.DataFrame(rows)

    # Re-classify instrument types using symbol patterns (they take precedence)
    if "instrument_type" in out.columns:
        out["instrument_type"] = out["symbol"].map(classify_symbol)

    return out.reset_index(drop=True)


def _try_symbols_by_exchange(listing) -> pd.DataFrame | None:
    """Best-effort call to ``Listing.symbols_by_exchange()``. Returns None when
    the method is missing, the broker errors out, or the response is empty —
    the caller then falls back to ``all_symbols()``."""
    fn = getattr(listing, "symbols_by_exchange", None)
    if fn is None:
        return None
    try:
        df = fn()
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    return df


def _drop_untradable(df: pd.DataFrame) -> pd.DataFrame:
    """Strip rows that aren't ordinary equities / ETFs on HOSE/HNX/UPCOM.

    ``vnstock``'s listings include DELISTED tickers, bonds, covered warrants,
    and futures alongside the names a retail account can actually trade. Drop
    them at fetch time so they never enter the OHLCV cache or candidate pool.
    No-op when neither an ``exchange`` nor a type column is present (e.g.
    KBS' ``all_symbols()`` fallback returns only ``symbol`` + ``organ_name``)."""
    out = df
    if "exchange" in out.columns:
        ex = out["exchange"].astype(str).str.upper().str.strip()
        out = out[ex.isin(_TRADABLE_EXCHANGES)]
    type_col = next((c for c in _TYPE_COLUMN_CANDIDATES if c in out.columns), None)
    if type_col is not None:
        raw = out[type_col].astype(str).str.upper().str.strip()
        out = out[~raw.isin(_NONEQUITY_RAW_TYPES)]
    return out.reset_index(drop=True)


def _try_fetch_etf_listing() -> pd.DataFrame | None:
    """Best-effort fetch of the dedicated ETF listing. Returns None on any
    failure (the ETF augmentation is optional — the curated HOSE_ETFS list
    still backfills the most-liquid names).

    vnstock 4.x: only the KBS source implements ``all_etf()`` and it returns
    a ``Series[symbol]`` (not a DataFrame). We normalize to a 2-column frame
    [symbol, instrument_type] so ``_merge_etf_rows`` can union it cleanly."""
    from vnstock import Listing  # lazy import to mirror fetch_universe
    from .fetcher import _limiter, _looks_like_rate_limit

    try:
        _limiter().wait()
        result = Listing(source="KBS").all_etf()
    except Exception as e:
        if _looks_like_rate_limit(e):
            _limiter().pause(65.0, reason="KBS all_etf 429")
        return None

    if result is None or len(result) == 0:
        return None
    # KBS returns a Series; defensively handle a DataFrame too in case a
    # future vnstock release switches the return shape.
    if isinstance(result, pd.Series):
        symbols = result.astype(str).str.upper().str.strip()
        return pd.DataFrame({"symbol": symbols.tolist(),
                             "instrument_type": ["ETF"] * len(symbols)})
    # DataFrame path
    if "symbol" not in result.columns and "ticker" in result.columns:
        result = result.rename(columns={"ticker": "symbol"})
    if "symbol" not in result.columns:
        return None
    out = pd.DataFrame({
        "symbol": result["symbol"].astype(str).str.upper().str.strip(),
        "instrument_type": "ETF",
    })
    return out


def _merge_etf_rows(stock_df: pd.DataFrame, etf_df: pd.DataFrame) -> pd.DataFrame:
    """Union the ETF listing into the stock listing, forcing
    ``instrument_type='ETF'`` for any symbol that appeared in ``all_etf()``.
    Preserves stock-listing rows (organ_name, exchange, etc.) and adds any
    ETF rows that weren't already in the stock listing."""
    etf_symbols = set(etf_df["symbol"].astype(str).str.upper())

    out = stock_df.copy()
    if "instrument_type" in out.columns:
        # Override any prior classification (regex or KBS type col) for rows
        # that the dedicated ETF endpoint confirms are ETFs. The dedicated
        # endpoint is the more authoritative source for instrument type.
        is_etf_now = out["symbol"].astype(str).str.upper().isin(etf_symbols)
        out.loc[is_etf_now, "instrument_type"] = "ETF"
    else:
        out["instrument_type"] = out["symbol"].astype(str).str.upper().map(
            lambda s: "ETF" if s in etf_symbols else "STOCK"
        )

    # Add ETFs that weren't in the stock listing.
    existing = set(out["symbol"].astype(str).str.upper())
    missing = etf_df[~etf_df["symbol"].astype(str).str.upper().isin(existing)].copy()
    if len(missing) > 0:
        # Pad with NaN for the stock-listing columns we don't have for ETFs.
        for col in out.columns:
            if col not in missing.columns:
                missing[col] = pd.NA
        out = pd.concat([out, missing[out.columns]], ignore_index=True)
    return out.drop_duplicates(subset=["symbol"]).reset_index(drop=True)


def _normalize_listing(df: pd.DataFrame) -> pd.DataFrame:
    """vnstock returns slightly different column names per source — normalize.

    Also derives a canonical ``instrument_type`` column with values
    STOCK / ETF / CW / OTHER. Priority: explicit broker column (mapped via
    ``_normalize_type_value``) → symbol regex (``classify_symbol``). The
    column is persisted to ``cache/universe.parquet`` so downstream code
    (``is_etf``, the selector's ETF filter, the news prompts) can branch
    on instrument type without hitting the broker again.
    """
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

    # Pick the first available security-type column from the broker
    # response (column names drift across vnstock sources / releases).
    type_col = next((c for c in _TYPE_COLUMN_CANDIDATES if c in df.columns), None)
    if type_col is not None:
        broker_type = df[type_col].map(_normalize_type_value)
    else:
        broker_type = pd.Series(["OTHER"] * len(df), index=df.index)
    # Fall back to the symbol-shape classifier whenever the broker didn't
    # give us a useful label.
    fallback_type = df["symbol"].map(classify_symbol)
    df["instrument_type"] = broker_type.where(broker_type != "OTHER", fallback_type)

    keep = [c for c in ["symbol", "exchange", "organ_name", "instrument_type"]
            if c in df.columns]
    df = df[keep].drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    return df


def load_universe(refresh: bool = False, source: str | None = None) -> pd.DataFrame:
    p = universe_path()
    if not refresh and p.exists():
        cached = pd.read_parquet(p)
        # Force a rebuild if the cached parquet predates the DELISTED filter
        # (no ``exchange`` column). Otherwise users would have to manually
        # delete cache/universe.parquet to pick up the new schema.
        if "exchange" in cached.columns:
            return cached
    df = fetch_universe(source=source)
    df.to_parquet(p, index=False)
    # Drop the lru_cache so the next is_etf / instrument_type lookup re-reads
    # the freshly-written parquet (might now contain new ETF rows or an
    # updated instrument_type column).
    _type_index.cache_clear()
    return df


def filter_exchanges(df: pd.DataFrame, exchanges: list[str]) -> pd.DataFrame:
    if "exchange" not in df.columns:
        return df  # if the source didn't return exchange, keep everything
    wanted = {e.upper() for e in exchanges} | {"HOSE"}  # alias HSX/HOSE
    return df[df["exchange"].astype(str).str.upper().isin(wanted)].copy()


@lru_cache(maxsize=1)
def _type_index() -> dict[str, str]:
    """{symbol: instrument_type} loaded once from the cached universe parquet.
    Returns empty dict if the parquet hasn't been built yet — callers fall
    back to the symbol regex via ``classify_symbol`` in that case."""
    p = universe_path()
    if not p.exists():
        return {}
    try:
        u = pd.read_parquet(p)
    except Exception:
        return {}
    if "symbol" not in u.columns or "instrument_type" not in u.columns:
        return {}
    return {
        str(row["symbol"]).upper(): str(row["instrument_type"]).upper()
        for _, row in u.iterrows()
    }


def instrument_type(symbol: str) -> str:
    """Look up the canonical instrument type for ``symbol``. Prefers the
    cached universe parquet (populated by ``load_universe``); falls back to
    the symbol-shape regex when the parquet is missing or doesn't list this
    ticker. Returns 'STOCK' / 'ETF' / 'CW' / 'OTHER'."""
    if not symbol:
        return "OTHER"
    key = str(symbol).upper().strip()
    cached = _type_index().get(key)
    if cached and cached != "OTHER":
        return cached
    return classify_symbol(key)


def is_etf(symbol: str) -> bool:
    """True if ``symbol`` is a Vietnamese ETF (HOSE-listed FUE* / E1VFVN30)."""
    return instrument_type(symbol) == "ETF"


def tradable_symbols() -> set[str] | None:
    """Set of symbols vnstock currently lists on HOSE/HNX/UPCOM (post the
    ``_drop_untradable`` filter applied at fetch time). Returns ``None`` when
    the universe parquet is missing — callers should treat that as "unknown,
    don't filter" rather than "nothing is tradable"."""
    p = universe_path()
    if not p.exists():
        return None
    try:
        u = pd.read_parquet(p)
    except Exception:
        return None
    if "symbol" not in u.columns:
        return None
    return {str(s).upper() for s in u["symbol"].tolist()}


def invalidate_type_cache() -> None:
    """Drop the cached ``_type_index`` so the next lookup re-reads the
    parquet. Useful after a fresh ``load_universe(refresh=True)`` writes a
    new file with updated instrument_type rows."""
    _type_index.cache_clear()
