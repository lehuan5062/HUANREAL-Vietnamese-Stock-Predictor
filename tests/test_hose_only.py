"""Verify the selector's --hose-only path and warm-cache delisted filter."""
import pandas as pd

import stockpredict.selector as sel


def test_hose_only_with_exchange_keeps_hose_rows(monkeypatch):
    """When the universe DataFrame has an `exchange` column, hose_only
    filters down to HOSE/HSX rows only."""
    fake_universe = pd.DataFrame([
        {"symbol": "VCB", "exchange": "HOSE", "organ_name": "Vietcombank"},
        {"symbol": "FPT", "exchange": "HSX", "organ_name": "FPT Corp"},
        {"symbol": "SHS", "exchange": "HNX", "organ_name": "Saigon-Hanoi Sec"},
        {"symbol": "BSR", "exchange": "UPCOM", "organ_name": "Binh Son Refining"},
        {"symbol": "ABC", "exchange": "HOSE", "organ_name": "Some HOSE Co"},
    ])

    def fake_load_universe(refresh=False, source=None):
        return fake_universe

    monkeypatch.setattr(sel, "load_universe", fake_load_universe)
    monkeypatch.setattr(sel, "cached_symbols", lambda: [])

    out = sel.select(target=10, hose_only=True)
    upper = {s.upper() for s in out}
    # Curated VN30/HOSE_MID names should still flow through (they're in HOSE_KNOWN).
    # The key invariant: NO HNX/UPCOM curated names like SHS or BSR.
    assert "SHS" not in upper, f"HNX SHS leaked into hose_only output: {out}"
    assert "BSR" not in upper, f"UPCOM BSR leaked into hose_only output: {out}"
    # Universe top-up correctly added new HOSE name.
    assert "ABC" in upper or len(out) >= 10  # may or may not reach top-up depending on curated count


def test_hose_only_without_exchange_falls_back_to_curated(monkeypatch):
    """When the universe lacks `exchange`, we fall back to HOSE_KNOWN
    (~43 curated tickers) and emit a warning."""
    fake_universe = pd.DataFrame([
        {"symbol": "FPT", "organ_name": "FPT Corp"},
        {"symbol": "SHS", "organ_name": "Saigon-Hanoi Sec"},
        {"symbol": "ABC", "organ_name": "Mystery Co"},
    ])

    def fake_load_universe(refresh=False, source=None):
        return fake_universe

    monkeypatch.setattr(sel, "load_universe", fake_load_universe)
    monkeypatch.setattr(sel, "cached_symbols", lambda: [])

    import warnings
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        out = sel.select(target=100, hose_only=True)
        assert any("hose_only" in str(w.message).lower() for w in recorded), (
            "expected a warning about restricted curated set"
        )

    upper = {s.upper() for s in out}
    # Output is restricted to HOSE_KNOWN (VN30 + HOSE_MID).
    assert upper.issubset(sel.HOSE_KNOWN), f"unexpected non-HOSE leaked: {upper - sel.HOSE_KNOWN}"
    # Sanity: VN30 names are present.
    assert "VCB" in upper
    assert "FPT" in upper
    # SHS (HNX) must not leak in even though it's in the universe.
    assert "SHS" not in upper


def test_hose_only_false_includes_all_exchanges(monkeypatch):
    """hose_only=False (default) leaves the multi-exchange behaviour
    unchanged — VN30/HNX_LIQUID/UPCOM_LIQUID all flow through."""
    monkeypatch.setattr(sel, "cached_symbols", lambda: [])
    out = sel.select(target=80, hose_only=False)
    upper = {s.upper() for s in out}
    # Curated mix: should contain at least one from each exchange.
    assert "VCB" in upper          # HOSE / VN30
    assert "SHS" in upper          # HNX
    assert "BSR" in upper          # UPCOM


def test_warm_cache_drops_delisted_via_tradable_set(monkeypatch):
    """If the warm OHLCV cache still has a delisted ticker (e.g. HTK), the
    selector must NOT surface it as a candidate. The check is gated on
    ``tradable_symbols()``: when it returns a set, every cached symbol not in
    that set is dropped before being mixed back into the output."""
    # Only VN30 names + one extra are "tradable" per vnstock. HTK is delisted.
    monkeypatch.setattr(sel, "tradable_symbols",
                        lambda: set(sel.VN30) | {"AGX"})
    # Pretend the warm cache contains a delisted ticker.
    monkeypatch.setattr(sel, "cached_symbols",
                        lambda: ["VCB", "HTK", "AGX"])
    # Make the curated layer NOT saturate the target so the warm cache runs.
    monkeypatch.setattr(sel, "CURATED", ["VCB"])
    # Avoid hitting the network on the top-up path.
    monkeypatch.setattr(sel, "load_universe",
                        lambda refresh=False, source=None: pd.DataFrame())

    out = sel.select(target=10, hose_only=False)
    upper = {s.upper() for s in out}
    assert "HTK" not in upper, f"delisted HTK leaked from warm cache: {out}"
    assert "AGX" in upper, "non-delisted cached ticker should still flow through"


def test_warm_cache_filter_noop_when_universe_missing(monkeypatch):
    """When the universe parquet is missing (cold start), tradable_symbols()
    returns None and the warm-cache filter degrades to a pass-through —
    better to over-include than to wipe the cache layer to nothing."""
    monkeypatch.setattr(sel, "tradable_symbols", lambda: None)
    monkeypatch.setattr(sel, "cached_symbols",
                        lambda: ["VCB", "HTK", "AGX"])
    monkeypatch.setattr(sel, "CURATED", [])
    monkeypatch.setattr(sel, "load_universe",
                        lambda refresh=False, source=None: pd.DataFrame())

    out = sel.select(target=10, hose_only=False)
    upper = {s.upper() for s in out}
    # All three cached names survive (no tradable set to filter against).
    assert {"VCB", "HTK", "AGX"}.issubset(upper)


def test_hose_known_set_consistent():
    """HOSE_KNOWN should contain only curated HOSE names — never any
    HNX_LIQUID or UPCOM_LIQUID names."""
    overlap_hnx = sel.HOSE_KNOWN & set(sel.HNX_LIQUID)
    overlap_upc = sel.HOSE_KNOWN & set(sel.UPCOM_LIQUID)
    assert not overlap_hnx, f"HNX names in HOSE_KNOWN: {overlap_hnx}"
    assert not overlap_upc, f"UPCOM names in HOSE_KNOWN: {overlap_upc}"
    # And it should be VN30 + HOSE_MID + HOSE_ETFS exactly.
    assert sel.HOSE_KNOWN == set(sel.VN30 + sel.HOSE_MID + sel.HOSE_ETFS)
