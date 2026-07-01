"""Rebound recovery targets.

For a downtrend candidate bought at ``close[T]`` these describe the *recovery
episode*: how many trading days until the position first turns profitable (N)
and how big that profit is (P). See :func:`recovery_episode`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..pricing import profit_threshold

# Default cap on how far forward we LABEL a recovery episode. This is not a
# trading policy cap (the strategy holds until recovery); it only bounds the
# historical labeling scan so it stays O(H) per symbol instead of O(N^2). A
# rebound that takes longer than ~1 trading year is not a rebound trade — rows
# that don't recover within the horizon are right-censored, which is exactly
# what the Kaplan-Meier estimator consumes. Override via
# ``strategy.recovery.label_max_horizon``.
_DEFAULT_LABEL_HORIZON = 250

# A single-session close-to-close move larger than this is physically
# impossible on any Vietnamese exchange (widest band UPCOM 15%) without a
# corporate action, so the unadjusted feed is showing a phantom jump. We stop
# a recovery scan at such a bar (censor there) so a split/rights spike can't be
# mistaken for a "recovery".
_BAND_BREAK = 0.15


def recovery_episode(df: pd.DataFrame, thr: float,
                     max_horizon: int | None = None) -> pd.DataFrame:
    """Label the rebound recovery episode for every row of a single symbol's
    (date-ascending) OHLCV frame.

    A position entered at ``close[T]`` is "profitable" on the first future day
    ``T+k`` (k >= 1) whose close clears the entry by ``thr``::

        close[T+k] / close[T] - 1 >= thr

    Returns a frame aligned to ``df.index`` with three columns:

    * ``target_days_to_recover`` (N) — for a **recovered** row, the smallest
      such ``k``. For a **censored** row (no profitable day observed within the
      available forward bars or the label horizon), the *censoring time*: the
      number of future bars we actually looked at (so Kaplan-Meier can use it as
      a right-censored observation, event=0).
    * ``target_recovery_return`` (P) — ``close[T+N]/close[T] - 1`` at the first
      profitable day for recovered rows; NaN for censored rows.
    * ``target_recovered`` (bool) — True iff a profitable day was observed.

    The scan is bounded by ``max_horizon`` (see ``_DEFAULT_LABEL_HORIZON``) and
    is additionally censored at the first future band-break day (an unadjusted
    corporate-action spike), so a phantom jump can't fake a recovery.
    """
    n = len(df)
    idx = df.index
    if n == 0:
        return pd.DataFrame({
            "target_days_to_recover": pd.Series(dtype=float),
            "target_recovery_return": pd.Series(dtype=float),
            "target_recovered": pd.Series(dtype=bool),
        }, index=idx)

    H = int(max_horizon if max_horizon is not None else _DEFAULT_LABEL_HORIZON)
    H = max(1, min(H, n - 1)) if n > 1 else 1

    close = df["close"].astype(float).to_numpy()
    # A non-positive / non-finite entry close (a halted or bad-data bar) has no
    # meaningful recovery: give it an unreachable target so it can never be
    # counted as a bounce (and never divides 0/0 in the P calc below). Such rows
    # fall through to censored, and are dropped downstream by the feature dropna.
    valid_entry = np.isfinite(close) & (close > 0.0)
    target_level = np.where(valid_entry, close * (1.0 + float(thr)), np.inf)

    # A future band-break at position j censors every earlier row's scan when it
    # is reached (prices past a corporate action are unadjusted / fake). We track
    # the offset of the FIRST band-break at or after T+1 per row via a scan.
    if "ret_1d" in df.columns:
        brk = (df["ret_1d"].astype(float).abs() > _BAND_BREAK).to_numpy()
    else:
        # Derive a rough 1-day return if ret_1d wasn't attached.
        prev = np.empty(n); prev[0] = np.nan; prev[1:] = close[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            brk = np.abs(close / prev - 1.0) > _BAND_BREAK
        brk[0] = False

    recovered = np.zeros(n, dtype=bool)
    N = np.full(n, np.nan)
    P = np.full(n, np.nan)
    found = np.zeros(n, dtype=bool)
    # A row is "censored so far" once its scan hits a band-break before recovery.
    censored_at = np.zeros(n, dtype=bool)

    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(1, H + 1):
            fut_close = np.full(n, np.nan)
            fut_close[:n - k] = close[k:]
            fut_brk = np.zeros(n, dtype=bool)
            fut_brk[:n - k] = brk[k:]

            active = ~found & ~censored_at & ~np.isnan(fut_close)
            # A band-break day is a phantom corporate-action spike: its close is
            # not trustworthy, so censor the scan at k-1 and never let a
            # "recovery" count on that day. k==1 break => censor time 0.
            brk_hit = active & fut_brk
            N[brk_hit] = k - 1
            censored_at |= brk_hit
            # Recovery on a trustworthy (non-break) future day. Invalid entries
            # have target_level = +inf (see above), so they never hit here.
            elig = active & ~brk_hit
            hit = elig & (fut_close >= target_level)
            N[hit] = k
            P[hit] = fut_close[hit] / close[hit] - 1.0
            recovered[hit] = True
            found |= hit
            if (found | censored_at).all():
                break

    # Rows never resolved (neither recovered nor band-censored) are right-censored
    # at min(available future bars, H) — the number of days we actually watched.
    pos = np.arange(n)
    avail = np.minimum(n - 1 - pos, H)
    unresolved = ~found & ~censored_at
    N[unresolved] = avail[unresolved]

    return pd.DataFrame({
        "target_days_to_recover": N,
        "target_recovery_return": P,
        "target_recovered": recovered,
    }, index=idx)


def resolve_exit(future_close, entry: float, thr: float) -> dict | None:
    """Resolve a rebound trade's exit by walking its forward close path.

    ``future_close`` is the sequence of closes AFTER the entry bar, in
    chronological order (offset k = 1, 2, ...). The trade exits on the first day
    the close clears the profitable point (``close >= entry*(1 + thr)``).

    Returns ``{"reason": "recovery", "k" (1-based day), "exit_close"}`` for that
    day, or ``None`` when the trade is still open (never recovered within the
    available data). Callers decide how to treat an open trade (the backtest
    marks-to-market at the last close; the ledger leaves it pending)."""
    import numpy as _np
    fc = _np.asarray(future_close, dtype=float)
    n = fc.size
    if n == 0 or entry <= 0:
        return None
    recov = entry * (1.0 + float(thr))
    for k in range(1, n + 1):
        c = float(fc[k - 1])
        if c >= recov:
            return {"reason": "recovery", "k": k, "exit_close": c}
    return None


def _label_horizon() -> int:
    cfg = load_config()
    strat = dict(getattr(cfg, "strategy", {}) or {})
    recovery = dict(strat.get("recovery", {}) or {})
    return int(recovery.get("label_max_horizon", _DEFAULT_LABEL_HORIZON))


def attach_target(df: pd.DataFrame, entry: str | None = None,
                  exit_offset_days: int | None = None) -> pd.DataFrame:
    """Attach the rebound recovery targets (N / P / recovered) to a per-symbol
    OHLCV frame. ``thr`` is the shared profit_threshold (round-trip cost +
    margin), so "profitable" means the same thing the pricing charges. ``entry``
    / ``exit_offset_days`` are accepted for call-site compatibility and unused."""
    out = df.copy()
    rec = recovery_episode(df, thr=profit_threshold(),
                           max_horizon=_label_horizon())
    out["target_days_to_recover"] = rec["target_days_to_recover"]
    out["target_recovery_return"] = rec["target_recovery_return"]
    out["target_recovered"] = rec["target_recovered"]
    return out
