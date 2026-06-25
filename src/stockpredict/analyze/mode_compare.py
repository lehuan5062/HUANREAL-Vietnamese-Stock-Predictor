"""Head-to-head comparison of prediction methods over the ledger.

Compares the realized performance of different modes — base (pure ML),
claude (ML/LLM hybrid), claude_llm (LLM-only), gemini — restricted to
*comparable* runs: same trading day AND same run parameters (horizon /
hose-only / etfs / exclude). A single day is far too noisy (often 1-3
picks per mode), so the verdict pools over a window of days; the named-day
breakdown is shown only as context.

Comparability key: two runs are comparable when they share the same
``param_key`` — the run signature with its leading mode token stripped
(so ``base_d2``, ``claude_d2`` and ``claude_llm_d2`` all map to ``d2``,
but ``claude_d2_HOSE`` does NOT match ``base_d2``). A ``(as_of, param_key)``
cell with two or more distinct modes is a *comparable cell*; everything
pools over those cells only.

This is advisory analysis, not a program edit: mode is a per-run user
choice, so the output is "which method has been winning", not a config
knob to tune.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from ..tracking import _read

# Friendly labels for the report; unknown modes fall through to themselves.
MODE_LABELS = {
    "base": "base (pure ML)",
    "claude": "hybrid (ML + news)",
    "claude_llm": "LLM-only",
    "gemini": "gemini (ML + news)",
}


def _label(mode: str) -> str:
    return MODE_LABELS.get(str(mode), str(mode))


def _param_key(signature: str, mode: str) -> str:
    """Run signature with its leading ``{mode}_`` token stripped, so the same
    parameters under different modes share a key. Falls back to the raw
    signature when it doesn't start with the mode prefix."""
    sig = str(signature)
    prefix = f"{mode}_"
    return sig[len(prefix):] if sig.startswith(prefix) else sig


def compare_modes(window_days: int = 90,
                  as_of: str | dt.date | None = None,
                  modes: list[str] | None = None) -> dict:
    """Pool realized performance per mode over comparable cells in the window.

    Returns a dict with: ``window_days``, ``as_of`` (the named context day or
    None), ``n_comparable_cells``, ``per_mode`` (pooled stats list),
    ``pairwise`` (matched A-vs-B over common cells), ``unique_vs_shared`` (per
    mode), and ``today`` (the named-day breakdown, when ``as_of`` is given).
    ``note`` is set instead when there is nothing comparable to report.
    """
    df = _read()
    if df.empty:
        return {"window_days": window_days, "note": "ledger is empty"}

    df = df[df["evaluated"].fillna(False) & df["realized_return"].notna()].copy()
    if df.empty:
        return {"window_days": window_days,
                "note": "no evaluated picks with realized returns yet"}

    df["as_of"] = pd.to_datetime(df["as_of"])
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=int(window_days))
    win = df[df["as_of"] >= cutoff].copy()
    if modes:
        keep = {str(m) for m in modes}
        win = win[win["mode"].astype(str).isin(keep)]
    if win.empty:
        return {"window_days": window_days,
                "note": f"no evaluated picks in the last {window_days} days"}

    win["param_key"] = [
        _param_key(s, m) for s, m in zip(win["signature"], win["mode"])
    ]
    win["cell"] = list(zip(win["as_of"], win["param_key"]))

    # Comparable cells: same (day, params) with >= 2 distinct modes.
    cell_modes = win.groupby("cell")["mode"].nunique()
    comparable_cells = set(cell_modes[cell_modes >= 2].index)
    comp = win[win["cell"].isin(comparable_cells)].copy()
    if comp.empty:
        return {
            "window_days": window_days,
            "n_comparable_cells": 0,
            "note": ("no comparable cells — need >= 2 modes run on the SAME day "
                     "with the SAME params (picks/horizon/hose-only/etfs/exclude) "
                     "in the window"),
        }

    out: dict = {
        "window_days": window_days,
        "as_of": str(pd.to_datetime(as_of).date()) if as_of is not None else None,
        "n_comparable_cells": len(comparable_cells),
        "per_mode": _per_mode_stats(comp),
        "pairwise": _pairwise(comp),
        "unique_vs_shared": _unique_vs_shared(comp),
    }
    if as_of is not None:
        out["today"] = _named_day(win, pd.to_datetime(as_of), comparable_cells)
    return out


def _agg(returns: pd.Series) -> dict:
    r = returns.dropna()
    n = int(len(r))
    return {
        "n_picks": n,
        "hit_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_return": float(r.mean()) if n else float("nan"),
        "median_return": float(r.median()) if n else float("nan"),
    }


def _per_mode_stats(comp: pd.DataFrame) -> list[dict]:
    """Pooled stats per mode over comparable cells. ``mean_per_day`` averages
    each (day, params) cell first, then across cells, so a day with many picks
    doesn't dominate a day with one."""
    rows = []
    for mode, g in comp.groupby("mode"):
        stats = _agg(g["realized_return"])
        per_cell = g.groupby("cell")["realized_return"].mean()
        rows.append({
            "mode": str(mode),
            "label": _label(mode),
            "n_days": int(g["cell"].nunique()),
            **stats,
            "mean_per_day": float(per_cell.mean()) if len(per_cell) else float("nan"),
        })
    rows.sort(key=lambda d: (d["mean_per_day"] if d["mean_per_day"] == d["mean_per_day"]
                             else -1e9), reverse=True)
    return rows


def _pairwise(comp: pd.DataFrame) -> list[dict]:
    """For every mode pair, match on the cells where BOTH ran and compare the
    per-cell mean return. ``a_wins`` / ``b_wins`` count days each method's
    cell-mean beat the other's."""
    # cell -> {mode: mean_return}
    cell_mode_ret = (comp.groupby(["cell", "mode"])["realized_return"]
                     .mean().unstack("mode"))
    modes = sorted(comp["mode"].unique())
    res = []
    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            a, b = modes[i], modes[j]
            sub = cell_mode_ret[[a, b]].dropna()
            if sub.empty:
                continue
            res.append({
                "a": a, "b": b, "a_label": _label(a), "b_label": _label(b),
                "n_matched_days": int(len(sub)),
                "a_mean": float(sub[a].mean()),
                "b_mean": float(sub[b].mean()),
                "a_wins": int((sub[a] > sub[b]).sum()),
                "b_wins": int((sub[b] > sub[a]).sum()),
                "ties": int((sub[a] == sub[b]).sum()),
            })
    return res


def _unique_vs_shared(comp: pd.DataFrame) -> list[dict]:
    """Per mode: split its picks into those UNIQUE to it within a comparable
    cell vs those also picked by another mode that day, and compare realized
    returns. Shows whether a method's distinctive picks actually add value."""
    # For each cell, count how many modes picked each symbol.
    sym_counts = (comp.groupby(["cell", "symbol"])["mode"].nunique()
                  .rename("n_modes").reset_index())
    merged = comp.merge(sym_counts, on=["cell", "symbol"], how="left")
    rows = []
    for mode, g in merged.groupby("mode"):
        uniq = g[g["n_modes"] == 1]["realized_return"]
        shared = g[g["n_modes"] >= 2]["realized_return"]
        rows.append({
            "mode": str(mode),
            "label": _label(mode),
            "n_unique": int(uniq.notna().sum()),
            "unique_mean_return": float(uniq.mean()) if uniq.notna().any() else float("nan"),
            "unique_hit_rate": float((uniq > 0).mean()) if uniq.notna().any() else float("nan"),
            "n_shared": int(shared.notna().sum()),
            "shared_mean_return": float(shared.mean()) if shared.notna().any() else float("nan"),
        })
    rows.sort(key=lambda d: d["mode"])
    return rows


def _named_day(win: pd.DataFrame, day: pd.Timestamp,
               comparable_cells: set) -> dict:
    """Context-only breakdown of the single named day across modes."""
    day = day.normalize()
    sub = win[win["as_of"] == day]
    if sub.empty:
        return {"n_modes": 0, "note": f"no evaluated picks on {day.date()}"}
    picks = []
    for mode, g in sub.groupby("mode"):
        picks.append({
            "mode": str(mode), "label": _label(mode),
            "symbols": g["symbol"].tolist(),
            "mean_return": float(g["realized_return"].mean()),
            "comparable": any((c in comparable_cells) for c in g["cell"]),
        })
    picks.sort(key=lambda d: d["mean_return"], reverse=True)
    return {"n_modes": int(sub["mode"].nunique()), "picks": picks}


def _pct(x: float) -> str:
    return "  n/a" if x != x else f"{x:+.2%}"


def _rate(x: float) -> str:
    return " n/a" if x != x else f"{x:.0%}"


def format_report(result: dict) -> str:
    """Render the compare_modes() dict as a markdown report."""
    w = result.get("window_days")
    lines = [f"# Mode comparison — pooled over the last {w} days", ""]
    if result.get("note"):
        lines.append(f"**{result['note']}.**")
        return "\n".join(lines)

    lines.append(f"Comparable cells (same day + same params, >=2 modes): "
                 f"**{result['n_comparable_cells']}**.")
    lines.append("")

    lines.append("## Per-method (pooled over comparable cells)")
    lines.append("")
    lines.append("| method | days | picks | hit | mean/pick | median | mean/day |")
    lines.append("| ------ | ---- | ----- | --- | --------- | ------ | -------- |")
    for r in result["per_mode"]:
        lines.append(
            f"| {r['label']} | {r['n_days']} | {r['n_picks']} | "
            f"{_rate(r['hit_rate'])} | {_pct(r['mean_return'])} | "
            f"{_pct(r['median_return'])} | {_pct(r['mean_per_day'])} |"
        )
    lines.append("")
    lines.append("`mean/day` weights each day equally (averages a day's picks "
                 "first, then across days) — the fairest single number.")
    lines.append("")

    if result.get("pairwise"):
        lines.append("## Head-to-head (matched on shared days only)")
        lines.append("")
        lines.append("| A vs B | matched days | A mean/day | B mean/day | A wins | B wins | ties |")
        lines.append("| ------ | ------------ | ---------- | ---------- | ------ | ------ | ---- |")
        for p in result["pairwise"]:
            lines.append(
                f"| {p['a_label']} vs {p['b_label']} | {p['n_matched_days']} | "
                f"{_pct(p['a_mean'])} | {_pct(p['b_mean'])} | "
                f"{p['a_wins']} | {p['b_wins']} | {p['ties']} |"
            )
        lines.append("")

    if result.get("unique_vs_shared"):
        lines.append("## Distinctive picks — unique vs shared")
        lines.append("")
        lines.append("How each method's picks did when they were UNIQUE to it that "
                     "day (no other method picked the same name) vs SHARED.")
        lines.append("")
        lines.append("| method | unique n | unique mean | unique hit | shared n | shared mean |")
        lines.append("| ------ | -------- | ----------- | ---------- | -------- | ----------- |")
        for u in result["unique_vs_shared"]:
            lines.append(
                f"| {u['label']} | {u['n_unique']} | {_pct(u['unique_mean_return'])} | "
                f"{_rate(u['unique_hit_rate'])} | {u['n_shared']} | "
                f"{_pct(u['shared_mean_return'])} |"
            )
        lines.append("")

    today = result.get("today")
    if today:
        lines.append(f"## Named day {result.get('as_of')} (context only — noisy)")
        lines.append("")
        if today.get("note"):
            lines.append(f"_{today['note']}._")
        else:
            for p in today["picks"]:
                tag = "" if p["comparable"] else "  (not a comparable cell)"
                lines.append(f"- **{p['label']}**: {_pct(p['mean_return'])} "
                             f"on {', '.join(p['symbols'])}{tag}")
        lines.append("")

    lines.append("---")
    lines.append("Advisory only: a knob tweak can't change which method you run "
                 "— this tells you which to *prefer*. Pooled edges from few days "
                 "are noisy; re-check as more picks evaluate.")
    return "\n".join(lines)
