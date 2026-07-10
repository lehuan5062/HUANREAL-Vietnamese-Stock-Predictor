"""Verify the sliding-window rate limiter and rate-limit error detection."""
import threading
import time

import pytest

import stockpredict.data.fetcher as fetcher
from stockpredict.data.fetcher import (
    _RateLimiter,
    _limiter,
    _looks_like_rate_limit,
    _RATE_LIMIT_ERROR_TOKENS,
)


def test_even_pacing_spaces_calls_by_min_interval():
    """Calls are evenly spaced at window/cap seconds — NOT bursted.

    cap=5 over a 1s window => one call every 0.2s. The old sliding-window
    limiter let all 5 fire instantly (a burst); that burst is what tripped
    provider 429s, so even pacing is the fix.
    """
    lim = _RateLimiter(calls_per_min=5, window_seconds=1.0)
    stamps = []
    t0 = time.monotonic()
    for _ in range(4):
        lim.wait()
        stamps.append(time.monotonic() - t0)
    # 4 calls at 0.2s spacing land near 0.0, 0.2, 0.4, 0.6 — NOT all instant.
    assert stamps[0] < 0.1, f"first call should be immediate; was {stamps[0]:.2f}s"
    gaps = [stamps[i] - stamps[i - 1] for i in range(1, len(stamps))]
    for g in gaps:
        assert 0.15 <= g <= 0.35, f"consecutive calls should be ~0.2s apart; gap was {g:.2f}s"


def test_pause_blocks_subsequent_callers():
    """Calling pause(s) makes the next wait() block at least that long."""
    lim = _RateLimiter(calls_per_min=100, window_seconds=60.0)
    lim.wait()  # one call, well under cap
    lim.pause(0.5, reason="synthetic 429")
    t0 = time.monotonic()
    lim.wait()
    waited = time.monotonic() - t0
    assert waited >= 0.4, f"pause should block ~0.5s; only waited {waited:.2f}s"


def test_concurrent_threads_are_evenly_paced():
    """Racing threads are serialized to the even min-interval, never bursting.

    cap=5 over 1s => 0.2s spacing. 6 racing threads must take at least
    ~5*0.2 = 1.0s in total (each successive grant is spaced), and no window
    ever holds more than `cap` calls.
    """
    lim = _RateLimiter(calls_per_min=5, window_seconds=1.0)
    completed: list[float] = []
    c_lock = threading.Lock()

    def call():
        lim.wait()
        with c_lock:
            completed.append(time.monotonic())

    threads = [threading.Thread(target=call) for _ in range(6)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    completed.sort()
    # 6 calls at 0.2s spacing: the last should land ~1.0s after the first.
    span = completed[-1] - completed[0]
    assert span >= 0.9, f"6 evenly-paced calls should span >=0.9s; spanned {span:.2f}s"
    # No two consecutive grants closer than ~0.2s (minus scheduling slack).
    gaps = [completed[i] - completed[i - 1] for i in range(1, len(completed))]
    assert min(gaps) >= 0.15, f"grants too close together: min gap {min(gaps):.2f}s"


def test_per_source_limiters_are_independent():
    """Each source gets its own limiter; a pause on one never blocks another."""
    fetcher._LIMITERS.clear()
    vci = _limiter("VCI")
    tcbs = _limiter("TCBS")
    assert vci is not tcbs, "distinct sources must get distinct limiters"
    # Same source name (case-insensitive) returns the cached instance.
    assert _limiter("vci") is vci

    # Pause VCI hard; TCBS must remain immediately available.
    vci.pause(5.0, reason="synthetic 429")
    t0 = time.monotonic()
    tcbs.wait()
    waited = time.monotonic() - t0
    assert waited < 0.3, f"TCBS should be unaffected by VCI's pause; waited {waited:.2f}s"
    fetcher._LIMITERS.clear()


def test_looks_like_rate_limit_detects_vnstock_strings():
    """The vnstock error variants (English + Vietnamese) all match."""
    samples = [
        "API request failed: GIỚI HẠN API ĐÃ ĐẠT TỐI ĐA",
        "Rate Limit Exceeded — wait 2 seconds",
        "RuntimeError: 429 Too Many Requests",
        "Vietnamese: Bạn đã đạt tối đa số lượt yêu cầu API",
    ]
    for s in samples:
        assert _looks_like_rate_limit(Exception(s)), f"missed: {s}"


def test_looks_like_rate_limit_does_not_false_positive():
    """Random network errors should NOT be treated as rate-limit."""
    samples = [
        "Connection reset",
        "DNS lookup failed",
        "Empty response from API",
        "JSON decode error",
    ]
    for s in samples:
        assert not _looks_like_rate_limit(Exception(s)), f"false positive: {s}"


def test_looks_like_empty_data_matches_provider_empty_errors():
    """The provider's 'no bars in range' errors are recognized as empty-data,
    not treated as fetch failures."""
    from stockpredict.data.fetcher import _looks_like_empty_data
    empties = [
        "Dữ liệu trống cho mã AME với interval 1D.",  # KBS actual message
        "Empty data for symbol XYZ",
        "Không có dữ liệu",
        # VCI's "ticker not found" -- a no-data outcome, NOT a hard error.
        "Không tìm thấy dữ liệu. Vui lòng kiểm tra lại mã chứng khoán hoặc KBS",
    ]
    for s in empties:
        assert _looks_like_empty_data(Exception(s)), f"missed empty-data: {s}"
    # A real 429 / network error is NOT empty-data.
    for s in ["429 Too Many Requests", "Connection reset", "timeout"]:
        assert not _looks_like_empty_data(Exception(s)), f"false empty-data: {s}"


def test_fetch_history_returns_empty_not_error_on_empty_data(monkeypatch):
    """A source raising 'empty data' yields an empty frame (0 new rows), never
    a RuntimeError — the thin-ticker / no-new-bars case must not fail."""
    from stockpredict.data import fetcher as fx

    def fake_quote(symbol, src, start, end, interval, bypass):
        raise ValueError(f"Dữ liệu trống cho mã {symbol} với interval {interval}.")
    monkeypatch.setattr(fx, "_quote_history", fake_quote)
    fx._LIMITERS.clear()

    df = fx.fetch_history("AME", start="2026-07-07", end="2026-07-08", source_order=["KBS"])
    assert df.empty, "empty-data should yield an empty frame, not raise"
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    fx._LIMITERS.clear()


def test_no_data_from_all_sources_is_a_distinct_warning(monkeypatch, capsys):
    """A ticker that comes back empty from EVERY source is neither an `ok`
    (int) nor a hard `ERR:` — it's a NODATA sentinel, announced once as a
    yellow warning, and no source is left cooling down."""
    import pandas as pd
    from stockpredict.data import fetcher as fx

    # Every source raises the "ticker not found" message (now recognized as
    # empty-data), so update_symbol writes nothing and the cache stays empty.
    def fake_quote(symbol, src, start, end, interval, bypass):
        raise ValueError("Không tìm thấy dữ liệu. Vui lòng kiểm tra lại mã")
    monkeypatch.setattr(fx, "_quote_history", fake_quote)
    monkeypatch.setattr(fx, "read_ohlcv", lambda s: pd.DataFrame())
    fx._LIMITERS.clear()

    results = fx.update_many(["GHOST"], full=True)

    assert isinstance(results["GHOST"], str)
    assert results["GHOST"].startswith("NODATA:")
    out = capsys.readouterr().out
    assert "GHOST: no data available from any source" in out
    for src in ("KBS", "VCI"):
        assert fx._limiter(src).paused_remaining() == 0.0, \
            "a no-data ticker must not cool down any source"
    fx._LIMITERS.clear()


def test_stale_ticker_with_no_new_bars_is_a_normal_success(monkeypatch, capsys):
    """A ticker that already has cached rows and fetches 0 NEW bars is a
    normal success (green progress), NOT flagged no-data — the got_data
    signal keys off existing cached rows, not this fetch's row delta."""
    import pandas as pd
    from stockpredict.data import fetcher as fx

    existing = pd.DataFrame(
        {"open": [10.0], "high": [10.0], "low": [10.0], "close": [10.0],
         "volume": [1]},
        index=pd.DatetimeIndex(["2026-07-01"], name="date"),
    )
    # Fetch returns nothing new, but the symbol's cache is NON-empty.
    monkeypatch.setattr(fx, "update_symbol", lambda s, full=False, source_order=None: 0)
    monkeypatch.setattr(fx, "read_ohlcv", lambda s: existing)
    fx._LIMITERS.clear()

    results = fx.update_many(["HAScache"], full=True)

    assert results["HAScache"] == 0  # int -> counted ok, not NODATA
    out = capsys.readouterr().out
    assert "no data available" not in out
    assert "progress update:" in out
    fx._LIMITERS.clear()


# --- Fixed per-source cooldown (no ratchet, nothing persisted) ---

def test_cooldown_applies_fixed_pause_and_leaves_cap_untouched():
    """cooldown() pauses the source for the fixed seconds and never changes
    the rate cap — the rate is config-driven and fixed for the process."""
    lim = _RateLimiter(calls_per_min=60, window_seconds=60.0)
    lim.cooldown("VCI", 3.0, reason="test 429")
    assert lim.cap == 60, "cooldown must not change the rate cap"
    remaining = lim.paused_remaining()
    assert 2.5 <= remaining <= 3.0, f"cooldown should pause ~3s; got {remaining:.1f}s"


def test_cooldown_is_flat_not_growing():
    """Repeated cooldowns on the SAME source apply the SAME fixed value every
    time — no growth, no accumulation (the whole point of going manual)."""
    lim = _RateLimiter(calls_per_min=60, window_seconds=60.0)
    applied = []
    for _ in range(3):
        lim.paused_until = 0.0  # clear the prior pause so we can measure afresh
        lim.cooldown("VCI", 3.0, reason="test 429")
        applied.append(round(lim.paused_remaining(), 1))
    assert all(2.5 <= a <= 3.0 for a in applied), (
        f"every cooldown should be the same fixed ~3s, never growing; got {applied}"
    )


def test_configured_cooldown_for_uses_per_source_override():
    """cooldown_seconds_overrides[source] wins over the global default —
    mirrors _configured_rate_for's override precedence."""
    from stockpredict.data.fetcher import _configured_cooldown_for

    class _Cfg:
        data = {
            "cooldown_seconds": 3.0,
            "cooldown_seconds_overrides": {"VCI": 5.0},
        }
    assert _configured_cooldown_for("VCI", _Cfg()) == 5.0
    assert _configured_cooldown_for("KBS", _Cfg()) == 3.0, (
        "a source with no override should fall back to the global default"
    )


def test_limiter_uses_configured_rate_not_persisted(monkeypatch):
    """A fresh _limiter() build takes the configured rate every time — there
    is no persisted state to seed a reduced rate from anymore."""
    class _Cfg:
        data = {"api_per_min": 60, "api_per_min_overrides": {"VCI": 20}}
    monkeypatch.setattr(fetcher, "load_config", lambda: _Cfg())
    fetcher._LIMITERS.clear()
    lim = _limiter("VCI")
    assert lim.cap == 20, "limiter must use the fixed configured rate"
    fetcher._LIMITERS.clear()


def test_worker_applies_the_fixed_cooldown_on_a_confirmed_429(monkeypatch):
    """After a confirmed 429 on every source, each source ends up paused by
    the fixed cooldown — applied once, never stacked (pause takes the max of
    equal values, so even the belt-and-suspenders worker call can't inflate
    it beyond the configured seconds)."""
    import stockpredict.data.fetcher as fx

    def fake_quote_history(symbol, src, start, end, interval, bypass):
        raise ConnectionError("Failed to fetch data: 429 - Too Many Requests")
    monkeypatch.setattr(fx, "_quote_history", fake_quote_history)
    monkeypatch.setattr(fx, "read_ohlcv", lambda s: __import__("pandas").DataFrame())

    cooldown_seconds = 3.0

    class _Cfg:
        data = {
            "api_per_min": 60, "api_per_min_overrides": {},
            "cooldown_seconds": cooldown_seconds, "cooldown_seconds_overrides": {},
            "bypass_vnai_quota": True, "history_duration_years": 9,
        }
    monkeypatch.setattr(fx, "load_config", lambda: _Cfg())
    fx._LIMITERS.clear()

    fx.update_many(["HPG"], full=True)

    for src in ("KBS", "VCI"):
        remaining = fx._limiter(src).paused_remaining()
        assert 0.0 < remaining <= cooldown_seconds + 0.05, (
            f"{src} should be paused by the fixed cooldown (~{cooldown_seconds}s), "
            f"not a stacked/inflated value; got {remaining:.1f}s"
        )
    fx._LIMITERS.clear()
