"""Prediction ledger + retroactive scoring.

Every prediction run appends one row per pick to `cache/predictions.parquet`
with the fields needed to evaluate it later. When OHLCV for the target date
becomes available, `evaluate_pending` fills in the realized return so the
ledger always reflects the latest known truth.

Recent performance is then exposed to the LLM modes so Claude/Gemini can
factor in "what has been working / failing" when re-ranking new candidates.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import cache_dir, load_config
from .data.cache import read_ohlcv


_LEDGER_FILE = "predictions.parquet"


def ledger_path() -> Path:
    return cache_dir() / _LEDGER_FILE


_LEDGER_COLUMNS = [
    "run_id", "signature", "as_of", "target_date", "exit_offset_days",
    "mode", "symbol", "rank",
    "pred_mean", "news_score", "adjusted", "entry_price",
    "actual_exit", "realized_return", "evaluated",
    # Execution-quality fields, stamped by evaluate_pending from the bar at
    # `as_of + 1 trading day` (the actual buy day). entry_slippage measures
    # whether the predicted entry_price was achievable: negative means the
    # day's low was BELOW predicted entry (we could have bought cheaper),
    # positive means the day's low was ABOVE predicted entry (the entry
    # was unreachable — the realized_return calc is then fictional).
    "t0_open", "t0_low", "t0_close", "entry_slippage",
    # Comma-separated kebab-case dimension tags Claude actually CITED in
    # Step 4 of the news plan (the bracketed `[tag-name]` markers it puts
    # at the start of each finding bullet). Lets recent_performance aggregate
    # hit-rate by dimension so Claude can learn which research categories
    # have actually predicted returns vs. been noise. Empty string = no
    # dimensions cited (e.g. base/gemini mode picks, or a Claude pick with
    # no Step 4 findings at all).
    "dimensions_cited",
]


# Columns that may be missing from old ledger files; _read backfills them
# with NaN so the file stays readable across schema changes.
_NEW_FLOAT_COLUMNS = ("t0_open", "t0_low", "t0_close", "entry_slippage")
_NEW_STRING_COLUMNS = ("dimensions_cited",)


def _normalize_dimensions(value) -> str:
    """Coerce a `dimensions_cited` cell value into a canonical
    comma-separated string. Accepts:
      * a string (returned trimmed and lower-cased)
      * a list/tuple/set of tag names (joined with commas)
      * anything else → empty string

    The downstream by-dimension aggregator splits on `,` so we strip out
    any whitespace around tags here too — saves the aggregator from doing
    it on every row.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        tags = [str(v).strip().lower() for v in value if str(v).strip()]
        return ",".join(tags)
    s = str(value).strip().lower()
    if not s or s == "nan":  # pandas turns missing data into the literal "nan" string sometimes
        return ""
    # Already comma-separated — re-split + re-join to normalize whitespace.
    return ",".join(t.strip() for t in s.split(",") if t.strip())


def _read() -> pd.DataFrame:
    p = ledger_path()
    if not p.exists():
        return pd.DataFrame(columns=_LEDGER_COLUMNS)
    df = pd.read_parquet(p)
    # Backfill exit_offset_days for old ledger rows (pre-T+N support).
    if "exit_offset_days" not in df.columns:
        df["exit_offset_days"] = 2
    # Backfill signature for old ledger rows: derive from run_id by
    # stripping the YYYYMMDD_ date prefix (or full run_id if no underscore).
    if "signature" not in df.columns:
        def _derive(rid):
            try:
                rid = str(rid)
                first_underscore = rid.find("_")
                if first_underscore > 0 and rid[:first_underscore].isdigit():
                    return rid[first_underscore + 1:]
            except Exception:
                pass
            return ""
        df["signature"] = df["run_id"].map(_derive) if "run_id" in df.columns else ""
    # Backfill execution-quality columns added in the entry-slippage release.
    # Old rows get NaN — they'll be filled retroactively the next time
    # evaluate_pending revisits them (idempotent: it sets evaluated=True so
    # they aren't actually revisited unless the user resets that flag).
    for col in _NEW_FLOAT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    # Backfill string columns added later (dimensions_cited). Empty string
    # is the canonical "no data" value here, NOT NaN — this keeps the
    # column dtype consistent and makes the by-dimension aggregator's
    # split logic uniform.
    for col in _NEW_STRING_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def _write(df: pd.DataFrame) -> None:
    df.to_parquet(ledger_path(), index=False)


from functools import lru_cache


@lru_cache(maxsize=1)
def _trading_calendar_cached() -> pd.DatetimeIndex:
    """Union of cached OHLCV indices = the actual Vietnamese trading-day
    calendar. Excludes weekends AND every Vietnamese holiday the market
    historically closed for (Tết, April 30, May 1, Sept 2, ad-hoc closures).
    Returns empty if the cache hasn't been populated yet."""
    from .data.cache import cached_symbols, read_ohlcv
    syms = cached_symbols()
    if not syms:
        return pd.DatetimeIndex([])
    # Cap the union to a sample of liquid tickers for speed; if a date is a
    # trading day at all, at least one of these will have traded.
    union: set = set()
    for s in syms[:50]:
        try:
            df = read_ohlcv(s)
        except Exception:
            continue
        if not df.empty:
            union.update(pd.DatetimeIndex(df.index).normalize())
    return pd.DatetimeIndex(sorted(union))


def _invalidate_trading_calendar_cache() -> None:
    """Drop the cached trading-day calendar so the next call rebuilds it.
    Call after ingesting new OHLCV data within the same Python process."""
    _trading_calendar_cached.cache_clear()


# Vietnamese fixed-date public holidays (the market is closed). Tết and the
# Hung Kings Festival are lunar so their solar dates change each year — the
# real calendar in the OHLCV cache covers historical instances; for the
# couple of weeks we ever project forward we accept that those two might
# leak through (small, recoverable error: a wasted fetch returning nothing).
VN_FIXED_HOLIDAYS: frozenset[tuple[int, int]] = frozenset({
    (1, 1),    # New Year's Day
    (4, 30),   # Reunification Day
    (5, 1),    # Labor Day
    (9, 2),    # National Day
})


def _is_vn_fixed_holiday(d: pd.Timestamp) -> bool:
    return (d.month, d.day) in VN_FIXED_HOLIDAYS


def _extended_calendar(through: pd.Timestamp,
                       calendar: pd.DatetimeIndex | None = None
                       ) -> pd.DatetimeIndex:
    """Cached trading calendar + projected future weekdays (Mon-Fri, minus
    Vietnamese fixed-date holidays) up to ``through``. We don't know
    floating-date holidays (Tết, Hung Kings) in advance, so the projection
    may include those rare weekdays — worst case is one wasted fetch that
    returns nothing, and the next run catches up."""
    cal = calendar if calendar is not None else _trading_calendar_cached()
    through = pd.Timestamp(through).normalize()
    if len(cal) == 0:
        start = pd.Timestamp.today().normalize()
        future = pd.bdate_range(start=start, end=through)
        future = pd.DatetimeIndex(d for d in future if not _is_vn_fixed_holiday(d))
        return future
    if through <= cal[-1]:
        return cal
    future_start = cal[-1] + pd.Timedelta(days=1)
    future = pd.bdate_range(start=future_start, end=through)
    future = pd.DatetimeIndex(d for d in future if not _is_vn_fixed_holiday(d))
    return cal.append(future).sort_values()


def latest_expected_bar_date(now: dt.datetime | None = None,
                              post_close_buffer_minutes: int = 15,
                              calendar: pd.DatetimeIndex | None = None
                              ) -> pd.Timestamp | None:
    """Date of the most recent end-of-day bar that the broker should already
    have published.

    The OHLCV cache should be considered current iff
    ``cache.index.max() >= latest_expected_bar_date()``. Re-fetching beyond
    that date is wasted budget — the data either doesn't exist yet (mid-
    trading) or never will (weekends / holidays).

    Decision tree:
      * If today is in the trading calendar AND we're past 14:45 + buffer
        (default 15 min, so 15:00) → today's close has been published.
      * If today is in the trading calendar but we're earlier in the day
        → today's close is not yet final; the latest finalized bar is the
        previous trading day.
      * If today is NOT in the trading calendar (weekend, holiday) →
        the latest finalized bar is the most recent trading day before
        today.

    Returns ``None`` only when the trading calendar is empty (first run on
    a brand-new install)."""
    now = now or dt.datetime.now()
    today = pd.Timestamp(now.date()).normalize()

    cal = _extended_calendar(today, calendar=calendar)
    if len(cal) == 0:
        return None

    # Latest calendar entry at-or-before today.
    pos = int(cal.searchsorted(today, side="right")) - 1
    if pos < 0:
        return None
    latest = cal[pos].normalize()

    if latest != today:
        # Today is a non-trading day — latest finalized bar is whatever
        # the most recent trading day was.
        return latest

    # Today is a trading day. Decide whether today's close has published yet.
    cutoff = dt.time(14, 45)
    cutoff_min = cutoff.hour * 60 + cutoff.minute + int(post_close_buffer_minutes)
    now_min = now.time().hour * 60 + now.time().minute
    if now_min < cutoff_min:
        # Pre-close: today's close not yet final. Step back one trading day.
        if pos >= 1:
            return cal[pos - 1].normalize()
        return None
    return latest


def effective_today_for_trading(now: dt.datetime | None = None,
                                cutoff_hour: int = 14,
                                cutoff_minute: int = 30,
                                calendar: pd.DatetimeIndex | None = None
                                ) -> pd.Timestamp:
    """Return the date the picks should be stamped against.

    Vietnamese exchanges' continuous-trading session ends at 14:30 (ATC
    runs until 14:45). After 14:30 today's close is effectively locked
    in — a buy order placed now can't fill at today's close and will
    settle starting tomorrow. So picks made after the cutoff should treat
    the next trading day as T+0.

    Returns: a normalized pandas Timestamp.
      * `now <= cutoff`           -> today (calendar date of `now`)
      * `now > cutoff`            -> the next trading day after today
    Weekends and Vietnamese holidays are skipped via the cached trading
    calendar; if the calendar is empty we fall back to BDay.
    """
    now = now or dt.datetime.now()
    today = pd.Timestamp(now.date()).normalize()
    cutoff = dt.time(cutoff_hour, cutoff_minute)
    if now.time() <= cutoff:
        return today
    # Past the cutoff — pick next trading day strictly after today.
    cal = calendar if calendar is not None else _trading_calendar_cached()
    if len(cal) == 0:
        return (today + pd.tseries.offsets.BDay(1)).normalize()
    pos = int(cal.searchsorted(today, side="right"))
    if pos < len(cal):
        return cal[pos].normalize()
    # today is past the end of the cached calendar — project forward.
    anchor = today if today.weekday() < 5 else (today + pd.tseries.offsets.BDay(1)).normalize()
    if anchor == today:
        return (today + pd.tseries.offsets.BDay(1)).normalize()
    return anchor


def _last_trading_day_of_month(reference: pd.Timestamp,
                               calendar: pd.DatetimeIndex | None = None
                               ) -> pd.Timestamp | None:
    """Last trading day of the calendar month that contains `reference`.
    Uses the cached calendar where possible; projects future weekdays when
    the month extends past the cache. Returns None if no trading days are
    available for that month at all."""
    ref = pd.Timestamp(reference).normalize()
    month_start = ref.replace(day=1)
    if month_start.month == 12:
        next_month_start = month_start.replace(year=month_start.year + 1, month=1)
    else:
        next_month_start = month_start.replace(month=month_start.month + 1)
    end_of_month = next_month_start - pd.Timedelta(days=1)

    cal = _extended_calendar(end_of_month, calendar=calendar)
    if len(cal) == 0:
        return None
    in_month = cal[(cal >= month_start) & (cal < next_month_start)]
    if len(in_month) == 0:
        return None
    return in_month[-1]


def days_to_month_end(today: pd.Timestamp,
                      min_days: int = 2,
                      calendar: pd.DatetimeIndex | None = None) -> int:
    """How many trading days from `today` to the last trading day of the
    target month, where the target is the current month if that distance
    is >= `min_days`, otherwise the next month (so T+min_days is always
    satisfiable). Returns the integer offset for `--days N`.

    Future trading days beyond the cache are projected as weekdays
    (Mon-Fri); see `_extended_calendar` for caveats."""
    base = calendar if calendar is not None else _trading_calendar_cached()
    if len(base) == 0:
        raise RuntimeError("trading calendar empty — populate the OHLCV cache first")
    t = pd.Timestamp(today).normalize()

    # Try current month first.
    last_this = _last_trading_day_of_month(t, base)
    if last_this is not None:
        cal = _extended_calendar(last_this, calendar=base)
        pos_today = int(cal.searchsorted(t, side="left"))
        pos_last = int(cal.searchsorted(last_this, side="left"))
        n = pos_last - pos_today
        if n >= min_days:
            return n

    # Roll over to next month.
    if t.month == 12:
        next_month_anchor = t.replace(year=t.year + 1, month=1, day=1)
    else:
        next_month_anchor = t.replace(month=t.month + 1, day=1)
    last_next = _last_trading_day_of_month(next_month_anchor, base)
    if last_next is None:
        raise RuntimeError(
            f"next month ({next_month_anchor.year}-{next_month_anchor.month:02d}) "
            f"is out of reach; refresh the OHLCV cache or pass an integer --days"
        )
    cal = _extended_calendar(last_next, calendar=base)
    pos_today = int(cal.searchsorted(t, side="left"))
    pos_last = int(cal.searchsorted(last_next, side="left"))
    return pos_last - pos_today


def _next_trading_offset(start_date: pd.Timestamp, offset: int) -> pd.Timestamp:
    """Return start_date + `offset` TRADING days using the actual cached
    Vietnamese trading-day calendar (weekends + holidays excluded).

    Future trading days beyond the cache are projected as weekdays
    (Mon-Fri). If start_date itself is past the end of the cache, we
    project from start_date — not from the last cached day.
    """
    cal = _trading_calendar_cached()
    start_norm = pd.Timestamp(start_date).normalize()
    if len(cal) == 0:
        # No OHLCV cache yet — fall back to BDay arithmetic from start_date.
        anchor = start_norm if start_norm.weekday() < 5 else (
            (start_norm + pd.tseries.offsets.BDay(1)).normalize()
        )
        return (anchor + pd.tseries.offsets.BDay(int(offset))).normalize()

    if start_norm > cal[-1]:
        # Start is past the end of the cached calendar — anchor on
        # start_date (or the next weekday if it's a weekend) and project
        # forward with BDay.
        anchor = start_norm if start_norm.weekday() < 5 else (
            (start_norm + pd.tseries.offsets.BDay(1)).normalize()
        )
        return (anchor + pd.tseries.offsets.BDay(int(offset))).normalize()

    # Build an extended calendar that reaches at least `offset` weekdays past
    # the start, so the position arithmetic always lands inside `cal`.
    horizon = start_norm + pd.tseries.offsets.BDay(int(offset) + 5)
    cal = _extended_calendar(horizon, calendar=cal)
    pos = int(cal.searchsorted(start_norm, side="left"))
    target_pos = pos + int(offset)
    if target_pos < len(cal):
        return cal[target_pos].normalize()
    # Should not happen with the extension above, but be safe.
    extra = target_pos - len(cal) + 1
    return (cal[-1] + pd.tseries.offsets.BDay(extra)).normalize()


def run_signature(mode: str, exit_offset_days: int, units: int,
                  hose_only: bool = False) -> str:
    """Stable signature for a parameter set: distinct combinations get
    distinct signatures so saved artifacts don't override each other,
    while a re-run of the same parameters does override (idempotent).
    Used as both the filename suffix and the ledger ``run_id`` base."""
    parts = [mode, f"d{int(exit_offset_days)}", f"u{int(units)}"]
    if hose_only:
        parts.append("HOSE")
    return "_".join(parts)


def record(picks: pd.DataFrame, mode: str, as_of: str | dt.date | None = None,
           run_id: str | None = None,
           exit_offset_days: int | None = None,
           units: int | None = None,
           hose_only: bool = False) -> int:
    """Append one row per pick to the ledger. Returns number of rows added.

    `picks` is the dataframe returned by mode runs; must have at minimum:
    columns symbol, pred_mean. Optional: news_score, adjusted, close, rank.
    The default ``run_id`` includes mode/horizon/units/hose-only so a same-
    day rerun with different parameters does not clobber prior rows.
    """
    if picks is None or len(picks) == 0:
        return 0
    cfg = load_config()
    exit_off = int(exit_offset_days) if exit_offset_days is not None else int(cfg.target["exit_offset_days"])

    as_of = (pd.to_datetime(as_of) if as_of is not None
             else effective_today_for_trading())
    target = _next_trading_offset(as_of, exit_off)
    u = int(units) if units is not None else int(
        cfg.broker.get("default_position_units", 100)
        if hasattr(cfg, "broker") else 100
    )
    sig = run_signature(mode=mode, exit_offset_days=exit_off,
                        units=u, hose_only=hose_only)
    if run_id is None:
        rid = f"{as_of.strftime('%Y%m%d')}_{sig}"
    else:
        rid = run_id

    rows = []
    for i, r in picks.reset_index(drop=True).iterrows():
        sym = str(r["symbol"]).upper()
        rows.append({
            "run_id": rid,
            "signature": sig,
            "as_of": as_of.normalize(),
            "target_date": target,
            "exit_offset_days": exit_off,
            "mode": mode,
            "symbol": sym,
            "rank": int(r.get("rank", i + 1)),
            "pred_mean": float(r.get("pred_mean", np.nan)),
            "news_score": int(r["news_score"]) if "news_score" in r and pd.notna(r["news_score"]) else 0,
            "adjusted": float(r["adjusted"]) if "adjusted" in r and pd.notna(r["adjusted"]) else float(r.get("pred_mean", np.nan)),
            "entry_price": float(r["close"]) if "close" in r and pd.notna(r["close"]) else np.nan,
            "actual_exit": np.nan,
            "realized_return": np.nan,
            "evaluated": False,
            # Execution-quality fields filled by evaluate_pending once the
            # buy-day bar (as_of + 1 trading day) lands in the cache.
            "t0_open": np.nan,
            "t0_low": np.nan,
            "t0_close": np.nan,
            "entry_slippage": np.nan,
            # Dimension tags Claude actually cited in Step 4 of the plan.
            # Comma-separated string; empty for non-claude modes or rows
            # where Step 4 was left blank.
            "dimensions_cited": _normalize_dimensions(r.get("dimensions_cited")),
        })
    add = pd.DataFrame(rows)
    df = _read()
    # de-dupe on (run_id, symbol) so re-runs replace prior rows for that day
    if not df.empty:
        keep_mask = ~df.set_index(["run_id", "symbol"]).index.isin(
            add.set_index(["run_id", "symbol"]).index
        )
        df = df[keep_mask]
    _write(pd.concat([df, add], ignore_index=True))
    return len(add)


def evaluate_pending(today: dt.date | None = None) -> pd.DataFrame:
    """Fill realized returns for any pending row whose target_date <= today AND
    whose ticker has cached OHLCV through the target date. Returns the rows
    that were updated."""
    df = _read()
    if df.empty:
        return df
    today = today or dt.date.today()
    today_ts = pd.Timestamp(today).normalize()

    pending = df[(~df["evaluated"]) & (df["target_date"] <= today_ts)]
    if pending.empty:
        return pending

    updated_rows = []
    for idx, row in pending.iterrows():
        sym = row["symbol"]
        ohlcv = read_ohlcv(sym)
        if ohlcv.empty:
            continue
        target = pd.Timestamp(row["target_date"]).normalize()
        as_of = pd.Timestamp(row["as_of"]).normalize()
        # Use the first available trading day at or after target_date
        on_or_after = ohlcv[ohlcv.index >= target]
        if on_or_after.empty:
            continue
        actual_exit = float(on_or_after.iloc[0]["close"])
        entry = row["entry_price"]
        if pd.isna(entry):
            # fall back: read entry close from cached OHLCV at as_of
            on_at = ohlcv[ohlcv.index <= as_of]
            if on_at.empty:
                continue
            entry = float(on_at.iloc[-1]["close"])
            df.at[idx, "entry_price"] = entry
        realized = actual_exit / entry - 1.0
        df.at[idx, "actual_exit"] = actual_exit
        df.at[idx, "realized_return"] = realized
        df.at[idx, "evaluated"] = True

        # Execution-quality measurement: the predicted entry_price is the
        # close at as_of (the day before the actual trade). The user is
        # supposed to BUY on `as_of + 1 trading day` ("T+0" in the user's
        # frame, the buy day). Pull that bar's open/low/close to measure
        # whether the predicted entry was actually achievable. We tolerate
        # missing fields (open/low/close) so a partial OHLCV row still
        # records what we can.
        buy_day_bars = ohlcv[ohlcv.index > as_of]
        if not buy_day_bars.empty:
            bar = buy_day_bars.iloc[0]
            t0_open = float(bar["open"]) if "open" in bar and pd.notna(bar["open"]) else np.nan
            t0_low = float(bar["low"]) if "low" in bar and pd.notna(bar["low"]) else np.nan
            t0_close = float(bar["close"]) if "close" in bar and pd.notna(bar["close"]) else np.nan
            df.at[idx, "t0_open"] = t0_open
            df.at[idx, "t0_low"] = t0_low
            df.at[idx, "t0_close"] = t0_close
            # Slippage to the day's low. Negative => market dipped below
            # predicted entry (we could have bought cheaper); positive =>
            # the day's low was ABOVE predicted entry, so the entry was
            # unreachable and the realized_return on that row is fictional.
            if pd.notna(t0_low) and entry > 0:
                df.at[idx, "entry_slippage"] = (t0_low - entry) / entry
        updated_rows.append(idx)

    _write(df)
    return df.loc[updated_rows]


def recent_performance(window_days: int = 90,
                       mode: str | None = None) -> dict:
    """Summary stats over the last ``window_days`` of evaluated rows.
    The output now includes a ``by_horizon`` dict — apples-to-apples
    grouping by ``exit_offset_days`` so an LLM can see T+2 hit-rate
    separately from T+18 hit-rate."""
    df = _read()
    if df.empty:
        return {"n": 0, "note": "no predictions recorded yet"}
    df = df[df["evaluated"]]
    if mode:
        df = df[df["mode"] == mode]
    if df.empty:
        return {"n": 0, "note": "no evaluated predictions yet (need at least one T+N to elapse)"}

    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=window_days)
    df = df[df["as_of"] >= cutoff]
    if df.empty:
        return {"n": 0, "note": f"no evaluated predictions in last {window_days} days"}

    rets = df["realized_return"].astype(float)
    out = {
        "n": int(len(df)),
        "hit_rate": float((rets > 0).mean()),
        "mean_return": float(rets.mean()),
        "median_return": float(rets.median()),
        "best_pick": _format_pick(df.loc[rets.idxmax()]),
        "worst_pick": _format_pick(df.loc[rets.idxmin()]),
        "by_news_score": _by_news_score(df),
        "by_horizon": _by_horizon(df),
        "by_run_signature": _by_run_signature(df),
        "recent_5": _recent_picks_sample(df, n=5),
        "entry_slippage": _entry_slippage_stats(df),
        "by_dimension": _by_dimension(df),
    }
    return out


def _by_dimension(df: pd.DataFrame) -> dict:
    """Hit-rate / mean-return per dimension tag Claude actually cited in
    Step 4. A pick that cites three tags contributes to all three dimensions
    (so a row can land in multiple buckets).

    Returns ``{}`` when no row has any cited dimension — typical of pre-
    feature ledgers, base/gemini-only ledgers, or fresh installs where
    Claude hasn't yet completed Step 4 on a finished prediction.
    """
    if "dimensions_cited" not in df.columns:
        return {}
    out: dict = {}
    for _, row in df.iterrows():
        tags_str = row.get("dimensions_cited", "") or ""
        if not isinstance(tags_str, str) or not tags_str.strip():
            continue
        ret = float(row["realized_return"])
        for tag in (t.strip() for t in tags_str.split(",")):
            if not tag:
                continue
            bucket = out.setdefault(tag, {"_returns": []})
            bucket["_returns"].append(ret)
    # Finalize: turn the per-tag return list into n / hit_rate / mean / median.
    final: dict = {}
    for tag, bucket in out.items():
        rets = bucket["_returns"]
        s = pd.Series(rets, dtype=float)
        final[tag] = {
            "n": int(len(s)),
            "hit_rate": float((s > 0).mean()),
            "mean_return": float(s.mean()),
            "median_return": float(s.median()),
        }
    return final


def _entry_slippage_stats(df: pd.DataFrame) -> dict | None:
    """Stats on whether predicted entry_price was actually achievable on the
    buy day. Returns ``None`` if no rows have entry_slippage filled (e.g. all
    rows pre-date the entry-slippage release and haven't been re-evaluated).

    Keys:
      - ``n`` — rows with a non-null entry_slippage
      - ``mean`` / ``median`` — average slippage to the buy day's low
      - ``pct_unreachable`` — fraction where t0_low > entry_price (the
        predicted entry was never available; realized_return is fictional)
      - ``mean_savings_when_reachable`` — average % below predicted entry
        a limit-order at t0_low would have hit, conditional on the entry
        being reachable. Always ≥ 0; bigger means we routinely could have
        gotten in cheaper than `entry_price`.
    """
    if "entry_slippage" not in df.columns:
        return None
    s = df["entry_slippage"].astype(float).dropna()
    if s.empty:
        return None
    reachable = s[s <= 0]
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "pct_unreachable": float((s > 0).mean()),
        "mean_savings_when_reachable": (
            float((-reachable).mean()) if not reachable.empty else 0.0
        ),
    }


def _by_run_signature(df: pd.DataFrame) -> dict:
    """Group evaluated picks by full run signature (mode + horizon + units +
    hose flag). Lets the LLM see, for example, that its `claude_d2_u100`
    runs hit 60% but `claude_d18_u200` runs hit 40% — finer-grained than
    horizon alone, since fees and hose-only filters change the population."""
    if "signature" not in df.columns:
        return {}
    out: dict = {}
    for sig, g in df.groupby("signature"):
        if not sig:
            continue
        out[str(sig)] = {
            "n": int(len(g)),
            "hit_rate": float((g["realized_return"] > 0).mean()),
            "mean_return": float(g["realized_return"].mean()),
            "median_return": float(g["realized_return"].median()),
        }
    return out


def _by_horizon(df: pd.DataFrame) -> dict:
    """Group evaluated picks by ``exit_offset_days`` so the report
    differentiates T+2 history from longer-hold history."""
    if "exit_offset_days" not in df.columns:
        return {}
    out: dict = {}
    for n_days, g in df.groupby("exit_offset_days"):
        out[int(n_days)] = {
            "n": int(len(g)),
            "hit_rate": float((g["realized_return"] > 0).mean()),
            "mean_return": float(g["realized_return"].mean()),
            "median_return": float(g["realized_return"].median()),
        }
    return out


def _format_pick(row: pd.Series) -> dict:
    return {
        "symbol": str(row["symbol"]),
        "as_of": str(pd.Timestamp(row["as_of"]).date()),
        "pred_mean": float(row["pred_mean"]),
        "news_score": int(row["news_score"]),
        "realized_return": float(row["realized_return"]),
    }


def _by_news_score(df: pd.DataFrame) -> dict:
    out = {}
    for s, g in df.groupby("news_score"):
        out[int(s)] = {
            "n": int(len(g)),
            "hit_rate": float((g["realized_return"] > 0).mean()),
            "mean_return": float(g["realized_return"].mean()),
        }
    return out


def _recent_picks_sample(df: pd.DataFrame, n: int = 5) -> list[dict]:
    df = df.sort_values("as_of", ascending=False).head(n)
    return [_format_pick(r) for _, r in df.iterrows()]


def feedback_block(window_days: int = 90, mode: str | None = None,
                   current_horizon: int | None = None,
                   current_signature: str | None = None) -> str:
    """Markdown block summarising recent performance, designed for inclusion
    in LLM prompts so Claude can self-correct.

    If ``current_horizon`` (the T+N being predicted right now) is provided,
    its line in the by-horizon table is highlighted — those rows are the
    apples-to-apples comparison and the LLM should weight them most.
    If ``current_signature`` (full param set, e.g. ``claude_d18_u200_HOSE``)
    is provided, an additional by-signature table highlights the EXACT
    parameter combination history — the most apples-to-apples view."""
    perf = recent_performance(window_days=window_days, mode=mode)
    if perf.get("n", 0) == 0:
        return f"## Past performance feedback\n\n_{perf.get('note', 'no data')}_\n"

    lines = [
        f"## Past performance feedback (last {window_days} days)",
        "",
        f"- **n picks evaluated**: {perf['n']}",
        f"- **hit rate (all horizons pooled)**: {perf['hit_rate']:.1%}",
        f"- **mean return (all horizons pooled)**: {perf['mean_return']:+.4f}",
        f"- **median return**: {perf['median_return']:+.4f}",
        "",
    ]

    by_sig = perf.get("by_run_signature") or {}
    if by_sig:
        lines += [
            "### By full run signature (most apples-to-apples)",
            "",
            "Each row is a distinct parameter combination (mode + horizon + units",
            "+ hose-only). Re-runs of the same combo update the same row in the",
            "ledger. **THIS RUN** marks the exact parameters of today's prediction.",
            "",
            "| signature | n | hit_rate | mean_return | median_return | match? |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for sig in sorted(by_sig.keys()):
            s = by_sig[sig]
            match = "← **THIS RUN**" if (current_signature is not None and sig == current_signature) else ""
            lines.append(
                f"| `{sig}` | {s['n']} | {s['hit_rate']:.1%} | "
                f"{s['mean_return']:+.4f} | {s['median_return']:+.4f} | {match} |"
            )
        if current_signature is not None and current_signature not in by_sig:
            lines.append(
                f"| `{current_signature}` | 0 | _no prior history_ | - | - | ← **THIS RUN** |"
            )
        lines.append("")

    by_h = perf.get("by_horizon") or {}
    if by_h:
        lines += [
            "### By horizon (broader cross-run comparison)",
            "",
            "| horizon | n | hit_rate | mean_return | median_return | match? |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for h in sorted(by_h.keys()):
            s = by_h[h]
            match = "← **THIS RUN**" if (current_horizon is not None and int(h) == int(current_horizon)) else ""
            lines.append(
                f"| T+{h} | {s['n']} | {s['hit_rate']:.1%} | "
                f"{s['mean_return']:+.4f} | {s['median_return']:+.4f} | {match} |"
            )
        if current_horizon is not None and int(current_horizon) not in by_h:
            lines.append(
                f"| T+{int(current_horizon)} | 0 | _no prior history_ | - | - | ← **THIS RUN** |"
            )
        lines.append("")

    lines += [
        "### By prior news_score",
        "",
        "| news_score | n | hit_rate | mean_return |",
        "| --- | --- | --- | --- |",
    ]
    for s, stats in sorted(perf["by_news_score"].items()):
        lines.append(f"| {s:+d} | {stats['n']} | {stats['hit_rate']:.1%} | {stats['mean_return']:+.4f} |")

    lines += ["", "### Most recent evaluated picks", ""]
    lines.append("| as_of | symbol | pred_mean | news_score | realized |")
    lines.append("| --- | --- | --- | --- | --- |")
    for p in perf["recent_5"]:
        lines.append(
            f"| {p['as_of']} | {p['symbol']} | {p['pred_mean']:+.4f} | "
            f"{p['news_score']:+d} | {p['realized_return']:+.4f} |"
        )

    by_dim = perf.get("by_dimension") or {}
    if by_dim:
        # Sort by mean_return desc so the "what's been working" cluster lands
        # at the top — that's what Claude needs to know first when deciding
        # which dimensions to weight more heavily today. Only show tags with
        # n ≥ 2 to avoid noise from single-observation buckets.
        dims_sorted = sorted(
            ((tag, s) for tag, s in by_dim.items() if s["n"] >= 2),
            key=lambda kv: kv[1]["mean_return"],
            reverse=True,
        )
        if dims_sorted:
            lines += [
                "### By dimension category cited (which research tags actually predicted returns)",
                "",
                "Each row is a `[dimension-name]` tag Claude cited in Step 4 of",
                "a past plan. A pick that cited 3 tags contributes to all 3 rows.",
                "Only tags with n ≥ 2 are shown to filter noise. Sorted by",
                "mean_return so the dimensions that have actually worked land",
                "at the top.",
                "",
                "| dimension | n | hit_rate | mean_return | median_return |",
                "| --- | --- | --- | --- | --- |",
            ]
            for tag, s in dims_sorted:
                lines.append(
                    f"| `{tag}` | {s['n']} | {s['hit_rate']:.1%} | "
                    f"{s['mean_return']:+.4f} | {s['median_return']:+.4f} |"
                )
            lines.append("")

    slip = perf.get("entry_slippage")
    if slip is not None:
        lines += [
            "",
            "### Entry-execution sanity check",
            "",
            "We record the predicted `entry_price` (= close at as_of) and",
            "later compare it against the OHLC of the actual buy day",
            "(`as_of + 1 trading day`). This measures whether the entry",
            "we quoted was actually fillable, not just whether the model",
            "got the direction right.",
            "",
            f"- **n picks with buy-day data**: {slip['n']}",
            f"- **mean entry_slippage** (t0_low vs entry_price): "
            f"{slip['mean']:+.4f}",
            f"- **median entry_slippage**: {slip['median']:+.4f}",
            f"- **% picks where entry was UNREACHABLE** "
            f"(t0_low > entry_price → realized_return is fictional): "
            f"{slip['pct_unreachable']:.1%}",
            f"- **mean savings when reachable** (how much cheaper a "
            f"limit-order at t0_low would have filled): "
            f"{slip['mean_savings_when_reachable']:.4f}",
            "",
            "Interpret: a large `pct_unreachable` means today's quoted",
            "entries are systematically too low — the market gaps past",
            "them and we never get filled. A large `mean_savings_when_",
            "reachable` means quoted entries are too high — we routinely",
            "could have bought lower. Either way, treat published",
            "`actionable=True` rows with caution if these numbers are",
            "large; the realized_return assumes we did fill at",
            "`entry_price`, which the data here may contradict.",
            "",
        ]

    lines += [
        "",
        "Use this to calibrate: weight the **THIS RUN** signature row most",
        "heavily (exact parameter match), then the **THIS RUN** horizon row",
        "(broader comparison). If a particular news_score has been consistently",
        "wrong on this signature, weight it less. If recent picks at this",
        "signature have been losing, be more conservative in scoring today.",
        "",
    ]
    return "\n".join(lines)
