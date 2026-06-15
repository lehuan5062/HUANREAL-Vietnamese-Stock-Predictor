"""Annotate the picks DataFrame with multi-category 'best choice' badges.

Among picks where ``actionable=True``, mark the leader in each of four
criteria. A single ticker can be the best in multiple categories
(e.g. highest adjusted score AND highest risk-reward).

Categories:
  - best_adjusted   : top by ``adjusted`` (= pred_mean × (1 + 0.05 × news_score))
                      — the system's overall conviction.
  - best_rr         : top by ``rr_ratio`` — most asymmetric upside-vs-downside.
  - best_net        : top by ``net_reward_vnd`` — biggest per-share dollar edge
                      (net of round-trip fees).
  - best_composite  : top by mean of the three rank columns above. Use this
                      when no single metric should dominate.

If no row is actionable (the typical T+2 case), no flags are set.
"""
from __future__ import annotations

import pandas as pd


_CATEGORY_COLS = {
    "best_adjusted": "adjusted",
    "best_rr": "rr_ratio",
    "best_net": "net_reward_vnd",
}


def annotate_best(picks: pd.DataFrame) -> pd.DataFrame:
    """Add ``best_adjusted`` / ``best_rr`` / ``best_net`` / ``best_composite``
    boolean columns. Idempotent — safe to call repeatedly. Returns the
    same frame with the new columns appended (False on every row by default,
    True only on the per-category leader within the actionable subset)."""
    if picks is None or len(picks) == 0:
        return picks
    out = picks.copy()
    # Initialise all four columns to False so the schema is stable even
    # when there are no actionable picks.
    for col in ("best_adjusted", "best_rr", "best_net", "best_composite"):
        out[col] = False
    if "actionable" not in out.columns:
        return out
    actionable_mask = out["actionable"].fillna(False).astype(bool)
    if not actionable_mask.any():
        return out
    sub = out[actionable_mask]

    # Per-category leaders (highest value wins).
    for badge, source_col in _CATEGORY_COLS.items():
        if source_col not in sub.columns:
            continue
        # idxmax returns the index of the maximum; ties broken by first occurrence
        leader_idx = sub[source_col].astype(float).idxmax()
        out.loc[leader_idx, badge] = True

    # Composite: rank within the actionable subset on each criterion (lower
    # rank = better), sum the three ranks, pick the smallest sum.
    rank_cols = []
    for col in _CATEGORY_COLS.values():
        if col in sub.columns:
            rank_cols.append(sub[col].astype(float).rank(ascending=False, method="min"))
    if rank_cols:
        composite_rank = sum(rank_cols)
        leader_idx = composite_rank.idxmin()
        out.loc[leader_idx, "best_composite"] = True

    return out


def actionable_suffix(picks: pd.DataFrame) -> str:
    """Filename suffix listing the actionable tickers in DataFrame order:
    ``"_HII-MSB-HNG"``. Empty string when no rows are actionable, the
    ``actionable`` column is missing, or the frame is empty. Used so a
    directory listing of ``reports/`` surfaces which tickers are tradable
    on each report at a glance."""
    if picks is None or len(picks) == 0:
        return ""
    if "actionable" not in picks.columns or "symbol" not in picks.columns:
        return ""
    mask = picks["actionable"].fillna(False).astype(bool)
    tickers = picks.loc[mask, "symbol"].astype(str).tolist()
    if not tickers:
        return ""
    return "_" + "-".join(tickers)
