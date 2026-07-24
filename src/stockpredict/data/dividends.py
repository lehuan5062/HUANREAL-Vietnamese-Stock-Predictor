"""Deterministic dividend-history fetcher for the dividend strategy.

Same vnai-quota-bypass technique as ``fetcher.py:fetch_history`` — reach the
per-source provider instance and call its raw ``__wrapped__`` endpoint,
skipping the metered ``@optimize_execution`` decorator. Unlike OHLCV, neither
KBS's nor VCI's installed ``Company`` explorer classes actually implement a
``dividends()`` method (the generic ``vnstock.Company.dividends()`` wrapper
just delegates to a per-source method that doesn't exist for KBS/VCI — it
raises ``AttributeError`` if called). The one real, working dividend-history
endpoint found on inspection is **VCI's company events feed**
(``Company(symbol, source="VCI").events()``), which returns ALL corporate
events (dividends, issuances, AGMs, insider deals, ...) tagged by
``event_code``; dividend-cash events carry ``event_code == "DIV"`` with
``record_date`` / ``exright_date`` / ``payout_date`` and the per-share amount
in ``value_per_share`` (absolute VND) / ``exercise_ratio`` (fraction of the
10,000 VND par value). KBS exposes an analogous ``events(event_type=2)``
"Trả cổ tức" filter on its own ``Company``, but it returned empty for every
symbol tried during implementation (looks unpopulated / dead on the current
KBS endpoint) — so VCI is the sole live source today. If KBS's dividend
events ever come back populated, add it to ``_DIVIDEND_SOURCES`` below; the
per-source rate limiter and vnai bypass already generalize to it.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from ..config import cache_dir, load_config
from .fetcher import _limiter, _looks_like_rate_limit, _cprint

_DIVIDEND_SOURCES = ("VCI",)


def dividends_cache_dir() -> Path:
    d = cache_dir() / "dividends"
    d.mkdir(parents=True, exist_ok=True)
    return d


def dividend_cache_path(symbol: str) -> Path:
    return dividends_cache_dir() / f"{symbol.upper()}.parquet"


def _events_history(symbol: str, source: str, bypass_quota: bool) -> pd.DataFrame:
    """Call the source's raw (unmetered) ``Company.events()``, same bypass
    pattern as ``fetcher._quote_history``: reach the explorer's own
    ``Company`` class (NOT the generic ``vnstock.Company`` wrapper, which
    only forwards to a per-source method that may not exist) and invoke the
    undecorated function via ``__wrapped__``."""
    if source == "VCI":
        from vnstock.explorer.vci.company import Company as _VCICompany
        c = _VCICompany(symbol=symbol)
        if bypass_quota:
            try:
                raw = c.events.__wrapped__
                return raw(c)
            except (AttributeError, TypeError):
                raise RuntimeError(
                    f"vnai bypass unavailable for VCI/{symbol} dividends "
                    "(vnstock internals changed); will retry next source")
        return c.events()
    raise ValueError(f"no dividend-events implementation for source={source!r}")


def fetch_dividend_history(symbol: str) -> pd.DataFrame:
    """Fetch the raw cash-dividend event history for ``symbol``.

    Returns a DataFrame with columns ``[ex_date, record_date, payout_date,
    cash_per_share_vnd, exercise_ratio, title]`` (one row per historical cash
    dividend, most-recent last), or an empty frame if the symbol has none /
    every source fails.
    """
    cfg = load_config()
    bypass_quota = bool(cfg.data.get("bypass_vnai_quota", True))

    last_err: Exception | None = None
    for src in _DIVIDEND_SOURCES:
        try:
            _limiter(src).wait()
            _cprint(f"{src} is fetching dividends...")
            raw = _events_history(symbol, src, bypass_quota)
        except Exception as e:  # noqa: BLE001 - try next source
            last_err = e
            if _looks_like_rate_limit(e):
                cooldowns = cfg.data.get("cooldown_seconds_overrides", {}) or {}
                cooldown = float(cooldowns.get(src, cfg.data.get("cooldown_seconds", 60)))
                _limiter(src).pause(cooldown)
            continue
        if raw is None or raw.empty or "event_code" not in raw.columns:
            continue
        div = raw[raw["event_code"].astype(str).str.upper() == "DIV"].copy()
        if div.empty:
            return pd.DataFrame(columns=[
                "ex_date", "record_date", "payout_date",
                "cash_per_share_vnd", "exercise_ratio", "title"])
        out = pd.DataFrame({
            "ex_date": pd.to_datetime(div.get("exright_date"), errors="coerce"),
            "record_date": pd.to_datetime(div.get("record_date"), errors="coerce"),
            "payout_date": pd.to_datetime(div.get("payout_date"), errors="coerce"),
            "cash_per_share_vnd": pd.to_numeric(div.get("value_per_share"), errors="coerce"),
            "exercise_ratio": pd.to_numeric(div.get("exercise_ratio"), errors="coerce"),
            "title": div.get("event_title_en", div.get("event_title_vi", "")),
        })
        out = out.dropna(subset=["ex_date"]).sort_values("ex_date").reset_index(drop=True)
        return out
    if last_err is not None:
        raise RuntimeError(f"dividend fetch failed for {symbol} on all sources: {last_err}")
    return pd.DataFrame(columns=[
        "ex_date", "record_date", "payout_date",
        "cash_per_share_vnd", "exercise_ratio", "title"])


def update_dividends(symbols: list[str]) -> dict:
    """Refresh the dividend-history parquet cache for each symbol. Returns
    ``{symbol: n_rows | error_str}``, mirroring ``fetcher.update_many``'s
    result shape."""
    results: dict = {}
    for sym in symbols:
        sym = sym.upper()
        try:
            df = fetch_dividend_history(sym)
        except Exception as e:  # noqa: BLE001
            results[sym] = str(e)
            continue
        df.to_parquet(dividend_cache_path(sym), index=False)
        results[sym] = int(len(df))
    return results


_EXPECTED_COLS = {"ex_date", "record_date", "payout_date",
                 "cash_per_share_vnd", "exercise_ratio", "title"}


def read_dividend_history(symbol: str) -> pd.DataFrame:
    p = dividend_cache_path(symbol)
    empty = pd.DataFrame(columns=[
        "ex_date", "record_date", "payout_date",
        "cash_per_share_vnd", "exercise_ratio", "title"])
    if not p.exists():
        return empty
    df = pd.read_parquet(p)
    if not _EXPECTED_COLS.issubset(df.columns):
        # A stale/foreign-schema cache file (e.g. from an earlier, unrelated
        # experiment) — treat as empty rather than crash; a real
        # `update-dividends` run will overwrite it with the current schema.
        return empty
    return df


def dividend_summary(symbol: str, close_vnd_thousand: float | None = None,
                     as_of: dt.date | None = None,
                     years: int | None = None) -> dict:
    """Compute the plain data columns the dividend mode hands the LLM agent:
    ``dividend_yield_ttm`` (trailing-12-month cash dividends / current price),
    ``years_paid_consecutive`` (consecutive calendar years with >=1 cash
    dividend, counting back from the most recent), ``last_ex_date``, and
    ``payout_trend`` (rising/flat/declining — compares the total cash/share
    paid in the most recent payout year vs the year before).

    ``close_vnd_thousand`` is the OHLCV ``close`` (thousand-VND units, same
    scale as the rest of the pipeline) used to compute the yield; if omitted,
    ``dividend_yield_ttm`` is NaN. ``years`` bounds how far back
    ``payout_trend`` looks (default: ``strategy.dividend.trend_lookback_years``
    in config)."""
    as_of = as_of or dt.date.today()
    hist = read_dividend_history(symbol)
    out = {
        "dividend_yield_ttm": float("nan"),
        "years_paid_consecutive": 0,
        "last_ex_date": None,
        "payout_trend": "unknown",
        "n_dividend_events": int(len(hist)),
    }
    if hist.empty:
        return out

    cfg = load_config()
    div_cfg = dict(getattr(cfg, "strategy", {}) or {}).get("dividend", {}) or {}
    lookback_years = int(years if years is not None
                         else div_cfg.get("trend_lookback_years", 3))

    hist = hist.dropna(subset=["ex_date"]).sort_values("ex_date")
    out["last_ex_date"] = hist["ex_date"].max().date().isoformat()

    as_of_ts = pd.Timestamp(as_of)
    ttm_start = as_of_ts - pd.DateOffset(years=1)
    ttm = hist[(hist["ex_date"] > ttm_start) & (hist["ex_date"] <= as_of_ts)]
    ttm_cash = float(ttm["cash_per_share_vnd"].fillna(0).sum())
    if close_vnd_thousand is not None and close_vnd_thousand > 0:
        close_vnd = float(close_vnd_thousand) * 1000.0
        out["dividend_yield_ttm"] = round(ttm_cash / close_vnd, 4)

    # Consecutive calendar years (back from the most recent payout year) with
    # at least one cash dividend.
    years_paid = set(hist["ex_date"].dt.year.tolist())
    last_year = hist["ex_date"].max().year
    streak = 0
    y = last_year
    while y in years_paid:
        streak += 1
        y -= 1
    out["years_paid_consecutive"] = streak

    # Payout trend: total cash/share in the most recent payout year vs the
    # year before, over up to `lookback_years` years of history.
    by_year = hist.groupby(hist["ex_date"].dt.year)["cash_per_share_vnd"].sum()
    recent_years = sorted(by_year.index)[-lookback_years:]
    by_year = by_year.loc[recent_years]
    if len(by_year) >= 2:
        latest, prior = by_year.iloc[-1], by_year.iloc[-2]
        if prior <= 0:
            out["payout_trend"] = "rising" if latest > 0 else "unknown"
        elif latest > prior * 1.05:
            out["payout_trend"] = "rising"
        elif latest < prior * 0.95:
            out["payout_trend"] = "declining"
        else:
            out["payout_trend"] = "flat"
    return out
