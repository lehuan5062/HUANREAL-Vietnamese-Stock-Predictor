"""Order the universe so the most useful tickers come first.

Runs always cover the entire universe, but the fetch order still matters:
if a cold full-history fetch is interrupted (rate limit, Ctrl+C), the
priority order ensures the liquid names land on disk first.

Priority order:
  1. Curated liquid list (VN30 + HNX30 + UPCOM bluechips). These are the
     most-likely-to-pass-the-liquidity-filter names; we fetch them first so
     an interrupted run still returns useful picks.
  2. Currently-cached symbols (warm cache -> cheap incremental updates).
  3. Top-up from the full vnstock universe in alphabetical order.

`hose_only=True` restricts every layer to HOSE-listed tickers — see
`select`."""
from __future__ import annotations

import warnings
from typing import Iterable

import pandas as pd

from .data.cache import cached_symbols
from .data.universe import (filter_exchanges, is_etf, load_universe,
                             tradable_symbols)
from .filters import ceiling_lock_mask, corporate_action_mask, staleness_mask


# ---- curated bluechips (kept in code so single-file deploys still work) ----

VN30 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "LPB", "MBB", "MSN", "MWG", "PLX", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]
HNX_LIQUID = [
    "SHS", "PVS", "CEO", "MBS", "IDC", "VCS", "DTD", "TNG", "L14", "NVB",
    "BAB", "VC3", "PVI", "TIG", "DDG", "API", "AMV", "HUT", "PVC", "VFS",
]
UPCOM_LIQUID = [
    "BSR", "ACV", "MCH", "VEA", "VGI", "VTP", "MML", "QNS", "FOX", "VGT",
    "OIL", "MSR", "DVN", "BVB", "ABB",
]
HOSE_MID = [
    "DGC", "PNJ", "DCM", "DPM", "GMD", "VHC", "HDG", "KDH", "NLG", "DXG",
    "REE", "PVD", "DBC",
]
# HOSE-listed ETFs. Curated so they always enter the panel regardless of
# whether the broker's Listing API returns ETFs. All of these are listed
# on HOSE (the only Vietnamese exchange that hosts ETFs today). Liquidity
# varies — the universal liquidity filter (close >= 5k VND, ADV >= 1B VND)
# naturally drops the thinly-traded ones.
HOSE_ETFS = [
    "FUEVFVND",  # DCVFM VN Diamond  — most-liquid ETF
    "E1VFVN30",  # DCVFM VN30
    "FUESSV30",  # SSIAM VN30
    "FUEMAV30",  # Mirae Asset VN30
    "FUEKIV30",  # KIM VN30
    "FUEVN100",  # MAFM VN100
    "FUEDCMID",  # DCVFM Midcap
    "FUESSVFL",  # SSIAM VNFIN Lead
    "FUEIP100",  # IPAAM VN100
    "FUEFCV50",  # VinaCapital VN100
]

CURATED = list(dict.fromkeys(
    VN30 + HNX_LIQUID + UPCOM_LIQUID + HOSE_MID + HOSE_ETFS
))

# Tickers we know for certain are HOSE-listed (used as a strict fallback when
# `--hose-only=True` and the universe API doesn't return per-ticker exchange).
HOSE_KNOWN = set(VN30 + HOSE_MID + HOSE_ETFS)
# Tickers we know are NOT HOSE — used for the "lenient" exclusion path if ever
# enabled, and to keep the curated layer honest under hose_only.
NON_HOSE_KNOWN = set(HNX_LIQUID + UPCOM_LIQUID)


def select(target: int,
           refresh_universe: bool = False,
           exchanges: list[str] | None = None,
           hose_only: bool = False,
           include_etfs: bool = True,
           exclude: Iterable[str] | None = None) -> list[str]:
    """Return up to `target` symbols, prioritized for a time-bounded run.

    With `hose_only=True`, the universe is restricted to HOSE listings:
    we refresh the universe via VCI (which usually returns the `exchange`
    column), filter to HOSE rows, and trim every layer (curated + cached
    + top-up) to that set. If VCI also lacks `exchange`, we fall back to
    `HOSE_KNOWN` (VN30 + HOSE_MID + HOSE_ETFS, ~53 tickers) and emit a
    warning.

    With `include_etfs=False`, ETF tickers are filtered out of every layer
    (curated + cached + top-up). ETFs are identified via the
    ``data.universe.is_etf`` helper, which uses the cached universe parquet's
    ``instrument_type`` column when available and falls back to the
    ``FUE*`` / ``E1VFVN30`` symbol regex otherwise.

    ``exclude`` is a per-session blacklist of ticker symbols (case-insensitive)
    that are stripped from every layer before any other filter applies. Use
    it to suppress a single name (e.g. one you already hold) without editing
    config. Excluded tickers also feed into ``run_signature`` so the saved
    picks file is distinguishable from a no-exclude run.
    """
    exclude_set: set[str] = {s.upper() for s in (exclude or [])}
    out: list[str] = []
    seen: set[str] = set()

    def _add_many(syms):
        for s in syms:
            s = s.upper()
            if s in seen:
                continue
            out.append(s)
            seen.add(s)
            if len(out) >= target:
                return True
        return False

    # Build the HOSE-allowed set if hose_only.
    hose_allowed: set[str] | None = None
    if hose_only:
        hose_allowed = set(HOSE_KNOWN)  # always start from the curated truth
        try:
            u = load_universe(refresh=True, source="VCI")
            if "exchange" in u.columns:
                hose_rows = u[u["exchange"].astype(str).str.upper().isin({"HOSE", "HSX"})]
                hose_allowed |= set(hose_rows["symbol"].astype(str).str.upper().tolist())
            else:
                warnings.warn(
                    "[selector] hose_only=True but VCI didn't return `exchange`; "
                    f"restricted to {len(HOSE_KNOWN)} curated HOSE tickers.",
                    stacklevel=2,
                )
        except Exception as e:
            warnings.warn(
                f"[selector] hose_only=True but universe refresh failed ({e}); "
                f"restricted to {len(HOSE_KNOWN)} curated HOSE tickers.",
                stacklevel=2,
            )

    def _filter_hose(syms):
        if hose_allowed is None:
            return syms
        return [s for s in syms if s.upper() in hose_allowed]

    def _filter_etfs(syms):
        """Drop ETF tickers when include_etfs=False; pass through otherwise."""
        if include_etfs:
            return syms
        return [s for s in syms if not is_etf(s)]

    def _filter_exclude(syms):
        if not exclude_set:
            return syms
        return [s for s in syms if s.upper() not in exclude_set]

    # Snapshot of currently-tradable tickers from the (post-DELISTED-filter)
    # universe parquet. Used to scrub the warm cache layer — without it, a
    # stale ``cache/ohlcv/<DELISTED>.parquet`` file (e.g. HTK after delisting)
    # would re-surface as a candidate on every run. ``None`` means the
    # universe parquet is missing; in that case we skip the filter rather than
    # wipe the cached layer to nothing on a cold start.
    tradable = tradable_symbols()

    def _filter_delisted(syms):
        if tradable is None:
            return syms
        return [s for s in syms if s.upper() in tradable]

    def _apply_filters(syms):
        return _filter_exclude(_filter_etfs(_filter_hose(syms)))

    # Curated bluechip layer — trusted, no DELISTED filter so a flaky network
    # never wipes out the user-vetted picks.
    curated_layer = _apply_filters(CURATED)
    if _add_many(curated_layer):
        return out

    # Warm cache layer
    cached_layer = _apply_filters(_filter_delisted(cached_symbols()))
    if _add_many(cached_layer):
        return out

    # Top up from full universe. Try the API; tolerate failures since this is
    # a "best-effort more tickers" path — curated list alone is still useful.
    try:
        u = load_universe(refresh=refresh_universe)
        if exchanges:
            u = filter_exchanges(u, exchanges)
        if "symbol" in u.columns:
            top_up = sorted(u["symbol"].astype(str).str.upper().tolist())
            _add_many(_apply_filters(top_up))
    except Exception:
        pass

    return out[:target]


# ---------------------------------------------------------------------------
# Eligible universe — the pure-filter (no model) cross-section handed to the
# LLM agent. Moved here from the retired ``model/predict.py``.
# ---------------------------------------------------------------------------


def latest_cross_section(panel: pd.DataFrame, on: pd.Timestamp | None = None) -> pd.DataFrame:
    """Return one row per symbol at the most recent date in `panel` (or `on` if given)."""
    if panel.empty:
        return panel
    if on is not None:
        snap = panel[panel.index <= pd.to_datetime(on)]
    else:
        snap = panel
    snap = snap.reset_index().sort_values(["symbol", "date"])
    last = snap.groupby("symbol", as_index=False).tail(1).set_index("date")
    return last


def eligible_universe(on: str | pd.Timestamp | None = None,
                      panel: pd.DataFrame | None = None,
                      symbols: list[str] | None = None) -> pd.DataFrame:
    """Return the mechanically-gated cross-section of tradable names on the
    given date — UNCAPPED, with NO model scoring, NO ranking, and NO
    liquidity/overbought/downtrend gating.

    Applies only the true mechanical gates: ``latest_cross_section ->
    tradable_symbols -> staleness_mask -> ceiling_lock_mask ->
    corporate_action_mask``. Judgment thresholds (liquidity size, overbought
    RSI, downtrend shape) are NOT gates any more — the underlying columns
    (``adv_vnd_20``, ``adv_active_days_20``, ``close``, ``rsi_14``,
    ``history_days``, plus the technical reference columns ``mom_5``,
    ``mom_20``, ``high_prox_20``, ``atr_14``, ``vol_z_20``, ``realvol_20``)
    are simply included as plain data for the LLM agent to reason over.

    Used by all three modes (momentum / rebound / dividend) to build the
    universe frame handed to the agent's plan markdown.
    """
    from .dataset import build_panel

    if panel is None:
        panel = build_panel(symbols=symbols, require_target=False)
    elif symbols is not None:
        panel = panel[panel["symbol"].astype(str).str.upper().isin(
            {s.upper() for s in symbols}
        )]
    if panel.empty:
        return panel

    # Informational history-depth column — how many cached bars this symbol
    # has, at all (not a gate; the agent judges whether that's "enough").
    hist_days = panel.groupby("symbol").size().rename("history_days")
    panel = panel.join(hist_days, on="symbol")

    snap = latest_cross_section(panel, on=on)
    if snap.empty:
        return snap

    # Drop tickers vnstock no longer lists as tradable (DELISTED, etc.) — the
    # OHLCV cache keeps a parquet file for every ticker ever fetched, so
    # without this guard a stale cache entry (e.g. a delisted name) could
    # re-surface as a candidate. ``tradable_symbols()`` returns None on a
    # cold start (universe parquet missing); skip the filter rather than wipe
    # the cross-section.
    tradable = tradable_symbols()
    if tradable is not None:
        snap = snap[snap["symbol"].astype(str).str.upper().isin(tradable)]
    if snap.empty:
        return snap

    # Drop names whose most recent cached bar is stale relative to the pick
    # date: they'd be scored on a months-old close and that close recorded as
    # the entry price. Data-integrity gate, not a judgment call.
    ref = pd.to_datetime(on) if on is not None else pd.Timestamp(snap.index.max())
    fresh = staleness_mask(snap, ref)
    if not fresh.all():
        stale_syms = sorted(snap.loc[~fresh, "symbol"].astype(str))
        shown = (", ".join(stale_syms) if len(stale_syms) <= 20
                 else ", ".join(stale_syms[:20]) + f", ... +{len(stale_syms) - 20} more")
        print(f"[filters] dropped {len(stale_syms)} stale-data candidate(s) "
              f"(last bar too old for {ref.date()}): {shown}")
        snap = snap[fresh].copy()
    if snap.empty:
        return snap

    # Drop names locked limit-up: they closed at the daily ceiling, so the buy
    # session opens with a queue and no sellers and a limit-buy can't fill.
    snap = snap[ceiling_lock_mask(snap)].copy()
    if snap.empty:
        return snap

    # Drop names whose recent history holds a band-breaking 1-day move: that's
    # an unadjusted corporate action (split / rights / special dividend), not
    # a real crash, and it poisons mom_*/atr_14/rsi_14.
    snap = snap[corporate_action_mask(snap)].copy()
    return snap
