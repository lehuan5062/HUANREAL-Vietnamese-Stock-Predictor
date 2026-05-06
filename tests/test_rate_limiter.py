"""Verify the sliding-window rate limiter and rate-limit error detection."""
import threading
import time

import pytest

from stockpredict.data.fetcher import (
    _RateLimiter,
    _looks_like_rate_limit,
    _RATE_LIMIT_ERROR_TOKENS,
)


def test_sliding_window_caps_at_n_calls_per_window():
    """N+1 calls in a window block until the oldest ages out."""
    lim = _RateLimiter(calls_per_min=5, window_seconds=1.0)
    # Burn through 5 calls — these should not block.
    t0 = time.monotonic()
    for _ in range(5):
        lim.wait()
    burst = time.monotonic() - t0
    assert burst < 0.2, f"5 calls within window should be instant; took {burst:.2f}s"

    # The 6th call must wait ~1 second (until the first call ages out).
    t1 = time.monotonic()
    lim.wait()
    waited = time.monotonic() - t1
    assert 0.7 <= waited <= 1.5, f"6th call should wait ~1s; waited {waited:.2f}s"


def test_pause_blocks_subsequent_callers():
    """Calling pause(s) makes the next wait() block at least that long."""
    lim = _RateLimiter(calls_per_min=100, window_seconds=60.0)
    lim.wait()  # one call, well under cap
    lim.pause(0.5, reason="synthetic 429")
    t0 = time.monotonic()
    lim.wait()
    waited = time.monotonic() - t0
    assert waited >= 0.4, f"pause should block ~0.5s; only waited {waited:.2f}s"


def test_concurrent_threads_cannot_exceed_cap():
    """When 10 threads race, only `cap` of them get through within the window."""
    lim = _RateLimiter(calls_per_min=3, window_seconds=1.0)
    completed: list[float] = []

    def call():
        lim.wait()
        completed.append(time.monotonic())

    threads = [threading.Thread(target=call) for _ in range(6)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # First 3 should land near t0; next 3 must each be ~1s later (window slid).
    completed.sort()
    fast = [c for c in completed if c - t0 < 0.3]
    slow = [c for c in completed if c - t0 >= 0.3]
    assert len(fast) == 3, f"first 3 should be fast; got {len(fast)}"
    assert len(slow) == 3, f"remaining 3 should be slow; got {len(slow)}"


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
