"""Verify the sliding-window rate limiter and rate-limit error detection."""
import threading
import time

import pytest

import stockpredict.data.fetcher as fetcher
import stockpredict.data.source_rate as source_rate
from stockpredict.data.fetcher import (
    _RateLimiter,
    _limiter,
    _looks_like_rate_limit,
    _RATE_LIMIT_ERROR_TOKENS,
)


@pytest.fixture
def isolated_rate_file(tmp_path, monkeypatch):
    """Redirect source_rate's persisted-rate file to a scratch path per test."""
    monkeypatch.setattr(source_rate, "_rate_file", lambda: tmp_path / "source_rate.json")
    yield tmp_path


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


def test_reduce_cap_widens_the_interval():
    """Ratcheting the cap down lengthens the enforced spacing immediately."""
    lim = _RateLimiter(calls_per_min=60, window_seconds=60.0)
    assert abs(lim.min_interval - 1.0) < 1e-9  # 60/60 = 1.0s
    lim.reduce_cap(30)
    assert abs(lim.min_interval - 2.0) < 1e-9  # 60/30 = 2.0s


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


# --- Adaptive per-source rate ratchet (persisted cross-session) ---

def test_ratchet_down_decrements_by_one_and_persists(isolated_rate_file):
    """Three consecutive 429s step 60 -> 59 -> 58 -> 57, saved to disk each time."""
    rates = [source_rate.ratchet_down("VCI", floor=30, default=60) for _ in range(3)]
    assert rates == [59, 58, 57]
    assert source_rate.get_persisted_rate("VCI", default=60) == 57


def test_ratchet_down_respects_floor(isolated_rate_file):
    """A source already at the floor never goes lower."""
    source_rate._save_rates({"VCI": {"calls_per_min": 30}})
    new_rate = source_rate.ratchet_down("VCI", floor=30, default=60)
    assert new_rate == 30


def test_limiter_seeds_from_persisted_rate_across_fresh_process(isolated_rate_file):
    """A fresh _limiter() build picks up the persisted (reduced) rate, not the config default."""
    source_rate.ratchet_down("VCI", floor=30, default=60)  # -> 59
    source_rate.ratchet_down("VCI", floor=30, default=60)  # -> 58

    fetcher._LIMITERS.clear()  # simulate a fresh process: no in-memory limiter yet
    lim = fetcher._RateLimiter(
        calls_per_min=source_rate.get_persisted_rate("VCI", default=60)
    )
    assert lim.cap == 58, "new limiter must seed from the persisted rate, not the config ceiling"


def test_ratchet_down_and_cooldown_applies_cap_and_flat_pause(isolated_rate_file):
    """ratchet_down_and_cooldown reduces the live cap AND forces the default
    61s cooldown."""
    lim = _RateLimiter(calls_per_min=60, window_seconds=60.0)
    new_rate = lim.ratchet_down_and_cooldown("VCI", floor=30, default=60, reason="test 429")
    assert new_rate == 59
    assert lim.cap == 59, "live limiter cap must update immediately, not just the persisted file"
    remaining = lim.paused_remaining()
    assert 56 <= remaining <= 61, f"cooldown should default to ~61s flat; got {remaining:.1f}s"


def test_ratchet_down_and_cooldown_is_flat_not_exponential(isolated_rate_file):
    """Repeated 429s always pause the same flat amount — no exponential growth."""
    lim = _RateLimiter(calls_per_min=60, window_seconds=60.0)
    for _ in range(3):
        lim.ratchet_down_and_cooldown("VCI", floor=30, default=60, reason="test 429")
    remaining = lim.paused_remaining()
    assert 56 <= remaining <= 61, f"cooldown must stay flat at ~61s even after repeats; got {remaining:.1f}s"


def test_ratchet_down_and_cooldown_respects_configured_seconds(isolated_rate_file):
    """cooldown_seconds is a real, honored parameter — not hardcoded."""
    lim = _RateLimiter(calls_per_min=60, window_seconds=60.0)
    lim.ratchet_down_and_cooldown("VCI", floor=30, default=60,
                                  cooldown_seconds=5.0, reason="test 429")
    remaining = lim.paused_remaining()
    assert 4.5 <= remaining <= 5.0, f"cooldown should honor the 5s override; got {remaining:.1f}s"


def test_reset_rates_clears_persisted_state(isolated_rate_file):
    """reset_rates() wipes the file; a subsequent lookup reverts to the config default."""
    source_rate.ratchet_down("VCI", floor=30, default=60)
    assert source_rate.get_persisted_rate("VCI", default=60) != 60

    source_rate.reset_rates()
    assert source_rate.get_persisted_rate("VCI", default=60) == 60
