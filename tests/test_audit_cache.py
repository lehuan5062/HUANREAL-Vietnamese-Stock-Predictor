"""Verify the cache audit helper buckets correctly."""
import datetime as dt

import pandas as pd

from stockpredict.data import fetcher


def _row(date_str: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": [10.0], "high": [10.5], "low": [9.5],
         "close": [10.2], "volume": [1000]},
        index=pd.DatetimeIndex([pd.Timestamp(date_str)], name="date"),
    )


def test_audit_buckets_warm_stale_cold(monkeypatch):
    """Each symbol lands in exactly the right bucket."""
    cache = {
        "WARM_A": _row("2026-04-29"),     # >= expected
        "WARM_B": _row("2026-04-29"),
        "STALE_A": _row("2026-04-22"),    # < expected
        "COLD_A": pd.DataFrame(),         # no data at all
    }
    monkeypatch.setattr(fetcher, "read_ohlcv", lambda s: cache.get(s, pd.DataFrame()))

    expected = pd.Timestamp("2026-04-29")
    warm, stale, cold = fetcher.audit_cache(
        ["WARM_A", "WARM_B", "STALE_A", "COLD_A", "COLD_B"],
        expected_bar=expected,
    )
    assert set(warm) == {"WARM_A", "WARM_B"}
    assert set(stale) == {"STALE_A"}
    assert set(cold) == {"COLD_A", "COLD_B"}


def test_audit_treats_all_warm_when_no_expected(monkeypatch):
    """When the trading calendar can't determine an expected bar (empty
    cache + brand-new install), cached rows count as warm — we have
    nothing better to compare against."""
    cache = {
        "A": _row("2020-01-01"),
        "B": pd.DataFrame(),
    }
    monkeypatch.setattr(fetcher, "read_ohlcv", lambda s: cache.get(s, pd.DataFrame()))
    # Force latest_expected_bar_date() to return None so the audit's
    # "no expected" branch is exercised even on a populated dev cache.
    from stockpredict import tracking
    monkeypatch.setattr(tracking, "latest_expected_bar_date",
                        lambda *a, **kw: None)
    warm, stale, cold = fetcher.audit_cache(["A", "B"])
    assert warm == ["A"]
    assert stale == []
    assert cold == ["B"]


def test_audit_uppercases_symbols(monkeypatch):
    """Audit normalizes symbol case so callers don't need to."""
    cache = {"FPT": _row("2026-04-29")}
    monkeypatch.setattr(fetcher, "read_ohlcv", lambda s: cache.get(s.upper(), pd.DataFrame()))
    warm, stale, cold = fetcher.audit_cache(["fpt", "vcb"],
                                             expected_bar=pd.Timestamp("2026-04-29"))
    assert warm == ["FPT"]
    assert cold == ["VCB"]


def test_update_many_only_spawns_threads_for_non_warm(monkeypatch):
    """The thread pool is sized for stale+cold only — warm symbols get
    zero-deltas without per-symbol work."""
    cache = {
        "WARM_A": _row("2026-04-29"),
        "WARM_B": _row("2026-04-29"),
        "STALE_A": _row("2026-04-22"),
    }
    monkeypatch.setattr(fetcher, "read_ohlcv", lambda s: cache.get(s, pd.DataFrame()))

    update_called: list[str] = []
    def fake_update(s, full=False, source_order=None):
        update_called.append(s)
        return 1
    monkeypatch.setattr(fetcher, "update_symbol", fake_update)

    # Pin expected bar via patched latest_expected_bar_date.
    from stockpredict import tracking
    monkeypatch.setattr(tracking, "latest_expected_bar_date",
                        lambda *a, **kw: pd.Timestamp("2026-04-29"))

    results = fetcher.update_many(["WARM_A", "WARM_B", "STALE_A"], full=False)

    # WARM_A, WARM_B: zeros without any update_symbol invocation.
    # STALE_A: actually called.
    assert results["WARM_A"] == 0
    assert results["WARM_B"] == 0
    assert results["STALE_A"] == 1
    assert update_called == ["STALE_A"], (
        f"only stale symbols should be processed; got {update_called}"
    )


def test_update_many_full_flag_processes_everyone(monkeypatch):
    """`full=True` bypasses the audit and re-fetches every symbol."""
    cache = {"A": _row("2026-04-29"), "B": _row("2026-04-29")}
    monkeypatch.setattr(fetcher, "read_ohlcv", lambda s: cache.get(s, pd.DataFrame()))

    update_called: list[str] = []
    def fake_update(s, full=False, source_order=None):
        update_called.append(s)
        assert full is True
        return 0
    monkeypatch.setattr(fetcher, "update_symbol", fake_update)

    fetcher.update_many(["A", "B"], full=True)
    assert sorted(update_called) == ["A", "B"]


def test_shared_queue_falls_over_when_one_source_fails(monkeypatch):
    """With no pre-distribution, if one source always fails every symbol is
    still fetched via the other source (self-balancing shared queue)."""
    import threading
    monkeypatch.setattr(fetcher, "read_ohlcv", lambda s: pd.DataFrame())
    fetcher._LIMITERS.clear()

    served_by = {}
    served_lock = threading.Lock()

    def fake_update(s, full=False, source_order=None):
        src = (source_order or [None])[0]
        if src != "KBS":
            raise RuntimeError(f"simulated {src} outage")
        with served_lock:
            served_by[s] = src
        return 1

    monkeypatch.setattr(fetcher, "update_symbol", fake_update)

    syms = [f"S{i}" for i in range(6)]
    results = fetcher.update_many(syms, full=True)

    assert all(results[s] == 1 for s in syms), f"all should succeed: {results}"
    # Every symbol was ultimately served by the one healthy source (KBS),
    # regardless of how many other sources are configured/failing.
    assert set(served_by) == set(syms)
    assert all(src == "KBS" for src in served_by.values()), served_by
    fetcher._LIMITERS.clear()


def test_shared_queue_marks_error_when_all_sources_fail(monkeypatch):
    """A symbol every source fails is recorded as an ERR, and the batch
    terminates (no infinite requeue loop)."""
    monkeypatch.setattr(fetcher, "read_ohlcv", lambda s: pd.DataFrame())
    fetcher._LIMITERS.clear()

    def fake_update(s, full=False, source_order=None):
        raise RuntimeError("total outage")

    monkeypatch.setattr(fetcher, "update_symbol", fake_update)

    results = fetcher.update_many(["X", "Y"], full=True)
    assert all(str(results[s]).startswith("ERR:") for s in ["X", "Y"]), results
    fetcher._LIMITERS.clear()
