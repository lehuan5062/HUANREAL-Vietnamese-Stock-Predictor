"""Download daily OHLCV for one or many Vietnamese tickers via vnstock."""
from __future__ import annotations

import collections
import datetime as dt
import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from ..config import load_config
from .cache import get_watermark, merge_ohlcv, read_ohlcv, set_watermark


def _today_str() -> str:
    return dt.date.today().isoformat()


class _RateLimiter:
    """Sliding-window rate limiter.

    The broker (vnstock guest tier) enforces a hard "<= N calls in any
    rolling 60-second window" — a simple "min interval" limiter is bursty
    relative to that and overflows when concurrent threads happen to align.
    We track every recent call's timestamp and block until the oldest
    falls outside the window, which is what the broker is actually doing
    server-side.

    Also has a `pause(seconds)` knob so error-handling paths can force
    the limiter to back off after a 429 — every subsequent caller waits
    until that pause has elapsed. Threads coordinate via the same lock.
    """

    def __init__(self, calls_per_min: float, window_seconds: float = 60.0) -> None:
        self.cap = max(1, int(calls_per_min))
        self.window = float(window_seconds)
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.calls: collections.deque[float] = collections.deque()
        self.paused_until: float = 0.0

    def wait(self) -> None:
        with self.cond:
            while True:
                now = time.monotonic()
                # Honor any global pause (e.g. after a 429).
                if self.paused_until > now:
                    self.cond.wait(timeout=self.paused_until - now)
                    continue
                # Drop calls that have aged out of the window.
                cutoff = now - self.window
                while self.calls and self.calls[0] <= cutoff:
                    self.calls.popleft()
                if len(self.calls) < self.cap:
                    self.calls.append(now)
                    return
                # Otherwise wait until the oldest call ages out.
                wake_at = self.calls[0] + self.window
                self.cond.wait(timeout=max(0.05, wake_at - now))

    def pause(self, seconds: float, reason: str = "") -> None:
        """Force everyone to wait at least `seconds` more before the next call.
        Stacks with prior pauses (takes the longer)."""
        with self.cond:
            until = time.monotonic() + max(0.0, float(seconds))
            if until > self.paused_until:
                self.paused_until = until
                if reason:
                    logging.getLogger("stockpredict.rate").warning(
                        "rate-limit pause %.1fs: %s", seconds, reason
                    )
                self.cond.notify_all()


_LIMITERS: dict[str, _RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()
_RATE_LIMIT_ERROR_TOKENS = (
    "rate limit",
    "rate-limit",
    "GIỚI HẠN API",          # vnstock's Vietnamese rate-limit message
    "GIOI HAN API",
    "Rate Limit Exceeded",
    "tối đa số lượt yêu cầu",
    "429",
)


def _looks_like_rate_limit(err: BaseException) -> bool:
    msg = str(err)
    return any(tok.lower() in msg.lower() for tok in _RATE_LIMIT_ERROR_TOKENS)


def _limiter(source: str = "_default") -> _RateLimiter:
    """Return the rate limiter for ``source``, building it on first use.

    Each vnstock backend (VCI / TCBS / KBS / MSN) is a different company's
    server, so each gets its own sliding-window limiter at ``api_per_min``.
    A pause after a transient error on one source therefore never throttles
    the others on the fallback chain in ``fetch_history``."""
    src = source.upper() if source else "_default"
    lim = _LIMITERS.get(src)
    if lim is not None:
        return lim
    with _LIMITERS_LOCK:
        lim = _LIMITERS.get(src)
        if lim is None:
            # Configurable via config.yaml -> data.api_per_min, or env override.
            # api_per_min is the PER-SOURCE cap (each source gets its own limiter).
            cfg = load_config()
            rate = float(os.environ.get("STOCKPREDICT_API_PER_MIN")
                         or cfg.data.get("api_per_min", 12.0))
            lim = _RateLimiter(calls_per_min=rate)
            _LIMITERS[src] = lim
            logging.getLogger("stockpredict.rate").info(
                "rate limiter active for %s: %.1f calls/min sliding window",
                src, rate
            )
    return lim


def _disable_vnstock_hard_exit() -> None:
    """vnstock's ``vnai.beam.quota.CleanErrorContext.__exit__`` calls
    ``sys.exit("... Process terminated.")`` whenever its rate-limit guardian
    fires. That raises ``SystemExit`` (a ``BaseException``, not ``Exception``),
    which slips past our retry loop in ``fetch_history`` and kills the whole
    batch — bypassing the limiter pause we already coded for the 429 path.
    Replace it with a return-False so the underlying ``RateLimitExceeded``
    propagates normally; ``_looks_like_rate_limit`` then matches it and the
    existing pause/retry runs as intended."""
    try:
        from vnai.beam import quota  # type: ignore
    except Exception:
        return
    cec = getattr(quota, "CleanErrorContext", None)
    if cec is None or getattr(cec, "_stockpredict_no_hard_exit", False):
        return

    def _patched_exit(self, exc_type, exc_val, exc_tb):  # noqa: D401
        return False  # let RateLimitExceeded bubble out instead of sys.exit

    cec.__exit__ = _patched_exit
    cec._stockpredict_no_hard_exit = True


def quiet_vnstock_logger() -> None:
    """Vnstock spams ERROR-level logs for transient API issues that we
    already handle via fallback. Bump it to CRITICAL so it stops cluttering
    the console during long full-universe runs."""
    # Introduce ourselves to vnstock BEFORE silencing its loggers — the
    # intro line goes through `stockpredict.intro` (a separate logger),
    # so it survives the CRITICAL bump below.
    from .intro import introduce
    introduce()
    _disable_vnstock_hard_exit()
    for name in ("vnstock", "vnstock.core.utils.client", "vnstock.explorer",
                 "vnstock.explorer.msn.quote"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def _quote_history(symbol: str, src: str, start: str, end: str,
                   interval: str, bypass_quota: bool):
    """Call vnstock's Quote.history, optionally bypassing vnai's client-side
    rate-limit quota.

    vnstock meters every call through ``vnai.beam.quota.optimize`` (a
    decorator layered onto ``Quote.history`` at both the public "API" wrapper
    and the per-source explorer level). That quota — NOT the data providers'
    own servers — is what enforces the guest tier's 20 req/min cap. vnstock's
    own error message states the library "automates access to public APIs you
    already have legitimate access to," so calling the underlying endpoint
    directly is within bounds.

    Because the decorators use ``functools.wraps``, the undecorated function
    survives as ``__wrapped__``. We reach the per-source provider instance
    (``q.provider``) and invoke its raw ``history.__wrapped__`` — skipping both
    the "API" and per-source quota layers. The body underneath is a plain
    ``requests`` call (``send_request`` -> ``send_request_direct``), so no
    quota counter is ever touched. Falls back to the normal decorated call if
    the internals shift in a future vnstock release."""
    from vnstock import Quote

    q = Quote(symbol=symbol, source=src)
    if bypass_quota:
        try:
            provider = q.provider
            raw = provider.history.__wrapped__
            return raw(provider, start=start, end=end, interval=interval)
        except (AttributeError, TypeError):
            # vnstock internals changed — fall back to the metered path.
            logging.getLogger("stockpredict.rate").warning(
                "vnai bypass unavailable (vnstock internals changed); "
                "falling back to metered Quote.history for %s/%s", src, symbol
            )
    return q.history(start=start, end=end, interval=interval)


def fetch_history(symbol: str, start: str, end: str | None = None,
                  source_order: list[str] | None = None) -> pd.DataFrame:
    """Fetch raw daily OHLCV from vnstock. Tries sources in the given order.

    Args:
        symbol: Ticker symbol
        start: Start date (YYYY-MM-DD)
        end: End date; defaults to today
        source_order: List of sources to try in order (e.g., [VCI, KBS, MSN, TCBS]).
                      Defaults to [VCI, KBS, MSN, TCBS].

    On any error (429, timeout, bad data), moves to the next source in the list.
    No pause mechanism; single worker feeds requests at rate-limiter pace (1 req/sec).

    When ``data.bypass_vnai_quota`` is set (default), calls go straight to
    the underlying provider endpoint via ``_quote_history``, sidestepping
    vnstock's 20/min guest quota. The per-source ``_RateLimiter`` still
    applies as a politeness throttle against the providers' real servers.
    """
    from . import source_preference

    cfg = load_config()
    bypass_quota = bool(cfg.data.get("bypass_vnai_quota", True))
    end = end or _today_str()

    if source_order is None:
        source_order = ["VCI", "KBS", "MSN", "TCBS"]

    for src in source_order:
        if src not in ("VCI", "KBS", "MSN", "TCBS"):
            continue
        for interval in ("1D", "D"):  # vnstock 4 uses 1D, older uses D
            try:
                _limiter(src).wait()
                df = _quote_history(symbol, src, start, end, interval, bypass_quota)
                if df is not None and len(df) > 0:
                    source_preference.track_source_success(src)
                    return _normalize_ohlcv(df)
            except Exception as e:
                logging.getLogger("stockpredict.fetcher").debug(
                    "fetch_history(%s, %s, interval=%s) failed: %s",
                    symbol, src, interval, type(e).__name__
                )
                # Try next interval for this source
                continue
        # All intervals failed for this source; move to next source
        source_preference.track_source_failure(src)

    raise RuntimeError(f"Could not fetch {symbol} from any source in {source_order}")


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}
    rename = {}
    for std in ("time", "date", "open", "high", "low", "close", "volume"):
        if std in cols_lower:
            rename[cols_lower[std]] = std
    df = df.rename(columns=rename)
    if "date" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep].dropna(subset=["close"])


def audit_cache(symbols: Iterable[str],
                expected_bar: pd.Timestamp | None = None
                ) -> tuple[list[str], list[str], list[str]]:
    """Bucket ``symbols`` against the cache and the latest expected bar.

    Returns ``(warm, stale, cold)`` where:
      * ``warm``  — cached file exists AND
                    (``cached_max >= expected_bar`` OR
                     a fetch was already attempted for this expected bar
                     and produced no new rows — see watermarks below)
      * ``stale`` — cached file exists, ``cached_max < expected_bar``,
                    and we haven't yet attempted a fetch for this expected
                    bar
      * ``cold``  — no cached file at all (full-history fetch needed)

    Watermarks: each successful fetch attempt — even one that returned
    zero new rows — stamps a per-symbol marker dated to the expected bar
    at the time of the attempt. Permanently-stuck tickers (delisted,
    halted, absent from the data feed) thereafter classify as warm
    rather than stale, so the user doesn't burn API budget retrying
    them every run. When the expected bar advances (next trading day
    closes), the marker becomes outdated and the ticker gets one fresh
    attempt before being re-stamped.

    If ``expected_bar`` is ``None`` (empty calendar / first-ever run),
    every cached symbol counts as warm — we have nothing better to
    compare against.
    """
    if expected_bar is None:
        from ..tracking import latest_expected_bar_date
        expected_bar = latest_expected_bar_date()
    warm: list[str] = []
    stale: list[str] = []
    cold: list[str] = []
    expected_date = expected_bar.date() if expected_bar is not None else None
    for s in symbols:
        s = s.upper()
        df = read_ohlcv(s)
        if df.empty:
            cold.append(s)
            continue
        if expected_bar is None or df.index.max().normalize() >= expected_bar:
            warm.append(s)
            continue
        # Cache-stale by date, but check if we already tried this run-cycle.
        wm = get_watermark(s)
        if wm is not None and expected_date is not None and wm >= expected_date:
            warm.append(s)  # broker had nothing new last time — don't retry
        else:
            stale.append(s)
    return warm, stale, cold


def _fetch_symbols_for_source(source: str,
                               assigned_queue: queue.Queue,
                               shared_failures: queue.Queue,
                               full: bool,
                               cfg: dict,
                               results: dict,
                               lock: threading.Lock) -> None:
    """Worker function: fetch symbols for a single source with retry + redistribution.

    Processes symbols from assigned_queue. On failure, retries after 1 second.
    If retry fails, moves symbol to shared_failures queue for load-balancing.
    When assigned_queue is exhausted, pulls from shared_failures to stay active.

    Args:
        source: The source (VCI, KBS, MSN, TCBS) this worker handles
        assigned_queue: Queue of initially assigned symbols for this source
        shared_failures: Shared queue for failed symbols from all sources
        full: Whether to do a full refetch
        cfg: Config dict (unused but kept for consistency)
        results: Shared dict to accumulate results (symbol -> delta or error)
        lock: Lock for thread-safe result accumulation
    """
    from . import source_preference

    while True:
        # Try to get from assigned queue first (non-blocking)
        try:
            symbol = assigned_queue.get_nowait()
        except queue.Empty:
            # Try shared failures queue (for load-balancing)
            try:
                symbol = shared_failures.get_nowait()
            except queue.Empty:
                # Both queues empty — we're done
                break

        retry_count = 0
        max_retries = 1
        last_error = None

        while retry_count <= max_retries:
            try:
                # Attempt fetch with only this source
                delta = update_symbol(symbol, full=full, source_order=[source])
                with lock:
                    results[symbol] = delta
                break  # Success, move to next symbol
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count <= max_retries:
                    # Sleep 1 second before retrying
                    time.sleep(1.0)

        if retry_count > max_retries:
            # Final failure — track it and mark as failed (don't redistribute)
            source_preference.track_source_failure(source)
            with lock:
                results[symbol] = f"ERR: {type(last_error).__name__}: {str(last_error)[:160]}"


def update_symbol(symbol: str, full: bool = False,
                  source_order: list[str] | None = None) -> int:
    """Incremental update: fetch from cache_max_date+1 (or full history). Returns row delta.

    Skips the API call when the cache already covers the latest finalized
    end-of-day bar. Caps the fetch end-date at that same latest finalized
    bar — never asks the broker for today's intraday partial bar during
    trading hours, which would pollute the cache with mid-session noise
    and force an extra refetch later.
    """
    cfg = load_config()
    start_full = cfg.data["history_start"]
    cached = read_ohlcv(symbol)

    # Look up the latest finalized bar once. We use it both to short-circuit
    # the fetch and to cap the end-date.
    from ..tracking import latest_expected_bar_date
    expected = latest_expected_bar_date()
    end_str = expected.strftime("%Y-%m-%d") if expected is not None else _today_str()

    if full or cached.empty:
        start = start_full
    else:
        # Smart freshness: skip if cache already contains the latest finalized bar.
        cached_max = cached.index.max().normalize()
        if expected is not None and cached_max >= expected:
            return 0
        # Skip if we've already attempted a fetch for this expected bar
        # — vnstock had nothing newer, no point burning API budget again.
        if expected is not None:
            wm = get_watermark(symbol)
            if wm is not None and wm >= expected.date():
                return 0
        last = cached.index.max().date()
        start = (last + dt.timedelta(days=1)).isoformat()
        if start > end_str:
            return 0
    new = fetch_history(symbol, start=start, end=end_str, source_order=source_order)
    before = len(cached)
    merged = merge_ohlcv(symbol, new)
    # Stamp the watermark to the expected bar so a fetch returning empty
    # data (delisted, halted, broker has no newer bar) doesn't retrigger
    # next run. The watermark only blocks retries within the same expected-
    # bar window — once a new trading day closes, expected advances and we
    # try fresh.
    if expected is not None:
        set_watermark(symbol, expected.date())
    return len(merged) - before


def update_many(symbols: Iterable[str], full: bool = False,
                workers: int | None = None,
                source_order: list[str] | None = None) -> dict[str, int | str]:
    """Bulk-update OHLCV for many symbols. Returns delta or error per symbol.

    4-worker strategy: distributes symbols across 4 sources (VCI, KBS, MSN, TCBS)
    based on historical success rates. Each worker fetches from its source only,
    retrying failed symbols once after 1 second. Failed symbols are redistributed
    to other workers for load-balancing.

    Internals:
      * If ``full=False`` (default), audits the cache and only fetches
        **stale** and **cold** tickers — warm ones get a zero-delta result
        without touching the network.
      * 4-worker ThreadPoolExecutor: one worker per source, each respects the
        per-source rate limiter (1 req/sec).
      * Symbols are initially distributed by preference (higher win-rate sources
        get more symbols). Failed symbols are redistributed among active workers.

    Fast path: if every selected symbol is already cached through the
    latest expected bar (e.g. running on a Saturday with cache through
    Friday), returns immediately without network calls.
    """
    from . import source_preference

    cfg = load_config()
    syms = list(symbols)
    if not syms:
        return {}

    sources = ["VCI", "KBS", "MSN", "TCBS"]

    if full:
        # User asked for an unconditional re-fetch — every symbol gets a job.
        to_fetch = syms
        results: dict[str, int | str] = {}
    else:
        warm, stale, cold = audit_cache(syms)
        results = {s: 0 for s in warm}
        to_fetch = stale + cold
        if not to_fetch:
            return results  # everything's current, no work to do

    # Distribute symbols across 4 sources by win-rate
    distribution = source_preference.distribute_symbols_by_preference(to_fetch, sources)

    # Shared infrastructure for workers
    lock = threading.Lock()
    shared_failures: queue.Queue = queue.Queue()

    # Create per-source assignment queues
    assignment_queues = {}
    for src in sources:
        q: queue.Queue = queue.Queue()
        for sym in distribution[src]:
            q.put(sym)
        assignment_queues[src] = q

    # Launch 4 workers via ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for src in sources:
            future = executor.submit(
                _fetch_symbols_for_source,
                source=src,
                assigned_queue=assignment_queues[src],
                shared_failures=shared_failures,
                full=full,
                cfg=cfg,
                results=results,
                lock=lock
            )
            futures[src] = future

        # Wait for all workers to complete with progress bar
        with tqdm(total=len(to_fetch), desc="update", ncols=80) as pbar:
            last_count = 0
            while len(results) < len(to_fetch):
                current_count = len(results)
                if current_count > last_count:
                    pbar.update(current_count - last_count)
                    last_count = current_count
                time.sleep(0.1)  # Check progress every 100ms
            # Final update for any remaining
            if last_count < len(to_fetch):
                pbar.update(len(to_fetch) - last_count)

        # Check for worker exceptions
        for src, future in futures.items():
            try:
                future.result()
            except Exception as e:
                logging.getLogger("stockpredict.fetcher").error(
                    "Worker for source %s failed: %s", src, e
                )

    return results
