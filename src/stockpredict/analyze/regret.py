"""Missed-winners ("regret") analysis.

For each past buy day, the *realized* top-N liquid tickers by T+2 return are the
winners. Comparing them to what the model actually surfaced (the ledger) tells
us which winners we MISSED — the raw material for meaningful self-correction
("why didn't we rank this?") and, optionally, a training signal.

CRITICAL: the realized top-N must use the SAME ``liquidity_mask`` universe as
``rank_today`` — otherwise illiquid penny names with huge realized returns show
up as "missed winners" the model was never allowed to pick (false regret).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..filters import liquidity_mask


def realized_top_n(panel: pd.DataFrame, n: int = 5,
                   apply_liquidity: bool = True) -> pd.DataFrame:
    """Per buy-day, the top-``n`` liquid tickers by realized ``target``
    (= close[T+2]/close[T] − 1). Returns columns
    ``[as_of, symbol, target, realized_rank]`` (rank 1 = best)."""
    if panel is None or panel.empty or "target" not in panel.columns:
        return pd.DataFrame(columns=["as_of", "symbol", "target", "realized_rank"])
    df = panel
    if apply_liquidity:
        df = df[liquidity_mask(df)]          # row-wise, same universe as live
    df = df.dropna(subset=["target"]).reset_index()
    date_col = "date" if "date" in df.columns else df.columns[0]
    df = df.sort_values([date_col, "target"], ascending=[True, False])
    top = df.groupby(date_col, sort=False).head(int(n)).copy()
    top["realized_rank"] = top.groupby(date_col, sort=False).cumcount() + 1
    out = top[[date_col, "symbol", "target", "realized_rank"]].rename(
        columns={date_col: "as_of"})
    out["as_of"] = pd.to_datetime(out["as_of"]).dt.normalize()
    return out.reset_index(drop=True)


def model_ranking(ledger: pd.DataFrame,
                  signature: str | None = None) -> pd.DataFrame:
    """What the model actually surfaced, from the ledger — evaluated rows only.
    Columns ``[as_of, symbol, rank, pred_mean, realized_return]``."""
    if ledger is None or ledger.empty:
        return pd.DataFrame(columns=["as_of", "symbol", "rank", "pred_mean",
                                     "realized_return"])
    df = ledger
    if signature is not None and "signature" in df.columns:
        df = df[df["signature"] == signature]
    if "evaluated" in df.columns:
        df = df[df["evaluated"].fillna(False).astype(bool)]
    cols = ["as_of", "symbol", "rank", "pred_mean", "realized_return"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df["as_of"] = pd.to_datetime(df["as_of"]).dt.normalize()
    return df


def missed_winners(panel: pd.DataFrame, ledger: pd.DataFrame, n: int = 5,
                   signature: str | None = None) -> pd.DataFrame:
    """Join the realized top-N (panel) against what the model surfaced (ledger),
    restricted to days the model actually ran. A realized winner is ``missed``
    when it wasn't surfaced (or ranked outside the top-N).

    Columns: ``[as_of, symbol, target, realized_rank, model_rank, pred_mean,
    missed]``."""
    actual = realized_top_n(panel, n=n)
    model = model_ranking(ledger, signature=signature)
    empty = pd.DataFrame(columns=["as_of", "symbol", "target", "realized_rank",
                                  "model_rank", "pred_mean", "missed"])
    if actual.empty or model.empty:
        return empty
    run_days = set(model["as_of"].unique())
    actual = actual[actual["as_of"].isin(run_days)]
    if actual.empty:
        return empty
    m = model[["as_of", "symbol", "rank", "pred_mean"]].rename(
        columns={"rank": "model_rank"})
    merged = actual.merge(m, on=["as_of", "symbol"], how="left")
    merged["missed"] = merged["model_rank"].isna() | (merged["model_rank"] > n)
    return merged.reset_index(drop=True)


# --------------------------------------------------------------------------
# Single-day missed winners (NO window aggregation).
#
# This is the primary view: for ONE buy day B, the top-N liquid tickers by the
# realized B -> B+2 ("bought 2 days ago, sold the eval day") return, and whether
# the model surfaced each. It deliberately does NOT pool a 90-day window and sort
# by magnitude — that surfaces long-dead spikes (a name that won weeks ago and
# has since fully reversed) at the top of the table, which is misleading.
# --------------------------------------------------------------------------

def _panel_dates(panel: pd.DataFrame) -> pd.DatetimeIndex:
    df = panel.reset_index()
    date_col = "date" if "date" in df.columns else df.columns[0]
    return pd.DatetimeIndex(sorted(pd.to_datetime(df[date_col]).dt.normalize().unique()))


def realized_buy_days(panel: pd.DataFrame) -> pd.DatetimeIndex:
    """Buy days whose forward T+2 window has fully realized (``target`` not NaN)."""
    if panel is None or panel.empty or "target" not in panel.columns:
        return pd.DatetimeIndex([])
    df = panel.dropna(subset=["target"]).reset_index()
    date_col = "date" if "date" in df.columns else df.columns[0]
    return pd.DatetimeIndex(sorted(pd.to_datetime(df[date_col]).dt.normalize().unique()))


def resolve_eval_buy_day(panel: pd.DataFrame, as_of=None):
    """The buy day to evaluate. If ``as_of`` (a prediction's buy day) is given and
    its T+2 window has realized, use it — so self-correction run at T+3 or later
    scores the prediction's EXACT T+2 day. Otherwise (its T+2 hasn't closed yet, or
    no as_of given) use the latest buy day whose window HAS realized — i.e. the most
    recent fully-closed [T-2 -> today] window. Returns a normalized Timestamp or
    None."""
    days = realized_buy_days(panel)
    if len(days) == 0:
        return None
    if as_of is not None:
        a = pd.Timestamp(as_of).normalize()
        if a in days:
            return a
    return days.max()


def _sell_day(panel: pd.DataFrame, buy_day, exit_offset: int | None = None):
    from ..config import load_config
    k = int(exit_offset if exit_offset is not None
            else load_config().target["exit_offset_days"])
    dates = _panel_dates(panel)
    b = pd.Timestamp(buy_day).normalize()
    if b not in dates:
        return None
    j = dates.get_loc(b) + k
    return dates[j] if j < len(dates) else None


def _surfaced_for_day(ledger: pd.DataFrame, buy_day,
                      signature: str | None = None) -> pd.DataFrame:
    """What the model surfaced for a given buy day (ledger rows for that ``as_of``),
    regardless of evaluation state. Columns ``[symbol, model_rank, pred_mean,
    realized_return]``."""
    empty = pd.DataFrame(columns=["symbol", "model_rank", "pred_mean",
                                  "realized_return"])
    if ledger is None or ledger.empty:
        return empty
    df = ledger.copy()
    df["as_of"] = pd.to_datetime(df["as_of"]).dt.normalize()
    df = df[df["as_of"] == pd.Timestamp(buy_day).normalize()]
    if signature is not None and "signature" in df.columns:
        df = df[df["signature"] == signature]
    if df.empty:
        return empty
    df = df.rename(columns={"rank": "model_rank"})
    cols = [c for c in ["symbol", "model_rank", "pred_mean", "realized_return"]
            if c in df.columns]
    df = df[cols]
    # A symbol can appear under several run signatures (base / claude / gemini /
    # missed variant) on the same buy day. Dedupe to one row per symbol, keeping
    # the best (lowest) rank, so the single-day comparison isn't noisy.
    if "model_rank" in df.columns:
        df = df.sort_values("model_rank").drop_duplicates(subset=["symbol"],
                                                          keep="first")
    else:
        df = df.drop_duplicates(subset=["symbol"], keep="first")
    return df.sort_values("model_rank").reset_index(drop=True)


def single_day_missed_winners(panel: pd.DataFrame, ledger: pd.DataFrame,
                              as_of=None, n: int = 5,
                              signature: str | None = None):
    """Top-N realized winners for ONE buy day and whether the model surfaced each.
    See module note. Returns a dict (``buy_day``, ``sell_day``, ``winners``,
    ``our_picks``) or None if no buy day has a realized window yet."""
    b = resolve_eval_buy_day(panel, as_of=as_of)
    if b is None:
        return None
    top = realized_top_n(panel, n=n)
    top = top[top["as_of"] == b].copy()
    picks = _surfaced_for_day(ledger, b, signature=signature)
    join = picks[["symbol", "model_rank"]] if not picks.empty else picks
    winners = top.merge(join, on="symbol", how="left") if not top.empty else top
    if "model_rank" not in winners.columns:
        winners["model_rank"] = pd.NA
    winners["ours"] = winners["model_rank"].notna()
    return {"buy_day": b, "sell_day": _sell_day(panel, b), "n": n,
            "signature": signature,
            "winners": winners.reset_index(drop=True),
            "our_picks": picks}


def single_day_markdown(panel: pd.DataFrame | None = None,
                        ledger: pd.DataFrame | None = None,
                        as_of=None, n: int = 5,
                        signature: str | None = None) -> str:
    """Markdown for the single-day missed-winners view (CLI / run flow /
    self-correct prompt)."""
    if panel is None:
        from ..dataset import build_panel
        # require_target=False keeps the post-window rows (e.g. today, T+1) in the
        # date index so the sell day (B + exit_offset trading days) resolves; the
        # winner logic dropnas `target` itself.
        panel = build_panel(require_target=False)
    if ledger is None:
        from ..tracking import _read
        ledger = _read()
    res = single_day_missed_winners(panel, ledger, as_of=as_of, n=n,
                                    signature=signature)
    if res is None:
        return ("### Missed winners (single day)\n\n"
                "_No buy day has a realized T+2 window yet._")
    b, s, w = res["buy_day"], res["sell_day"], res["winners"]
    bs = pd.Timestamp(b).date()
    ss = pd.Timestamp(s).date() if s is not None else "?"
    lines = [f"### Missed winners — top-{n} for buy day {bs} (sold {ss}, T+2)"
             + (f", sig={signature}" if signature else ""), "",
             f"_Top realized gainers if bought {bs} close and sold {ss} close. "
             f"Single closed day — no window aggregation._", "",
             "| rank | symbol | T+2 return | ours? |",
             "| --- | --- | --- | --- |"]
    for r in w.itertuples():
        lines.append(f"| {int(r.realized_rank)} | {r.symbol} | "
                     f"{float(r.target):+.2%} | {'YES' if bool(r.ours) else 'no'} |")
    op = res["our_picks"]
    lines += ["", "**What the model surfaced that buy day:**"]
    if op is None or op.empty:
        lines.append("- (no model run on that buy day to compare against)")
    else:
        for r in op.itertuples():
            rr = getattr(r, "realized_return", None)
            rrs = (f"{float(rr):+.2%}" if rr is not None and pd.notna(rr)
                   else "n/a")
            rk = (int(r.model_rank) if pd.notna(r.model_rank) else "?")
            lines.append(f"- {r.symbol} (our rank {rk}): realized {rrs}")
    return "\n".join(lines)


def aggregate_regret(window_days: int = 90, n: int = 5,
                     signature: str | None = None,
                     panel: pd.DataFrame | None = None,
                     ledger: pd.DataFrame | None = None) -> dict:
    """Window summary of missed winners. Builds the panel + reads the ledger
    when not supplied (so the CLI / self-correct prompt can call it standalone)."""
    if panel is None:
        from ..dataset import build_panel
        panel = build_panel(require_target=True)
    if ledger is None:
        from ..tracking import _read
        ledger = _read()
    mw = missed_winners(panel, ledger, n=n, signature=signature)
    if not mw.empty and window_days and window_days > 0:
        cutoff = mw["as_of"].max() - pd.Timedelta(days=int(window_days))
        mw = mw[mw["as_of"] >= cutoff]
    if mw.empty:
        return {"window_days": window_days, "n": n, "signature": signature,
                "n_winner_rows": 0, "miss_rate": None,
                "mean_missed_target": None, "mean_captured_target": None,
                "regret": None, "worst": []}
    missed = mw[mw["missed"]]
    captured = mw[~mw["missed"]]
    mean_missed = float(missed["target"].mean()) if len(missed) else None
    mean_cap = float(captured["target"].mean()) if len(captured) else None
    worst = (missed.sort_values("target", ascending=False)
             .head(15)[["as_of", "symbol", "target", "realized_rank"]])
    worst_rows = [
        {"as_of": str(pd.Timestamp(r.as_of).date()), "symbol": r.symbol,
         "target": round(float(r.target), 4), "realized_rank": int(r.realized_rank)}
        for r in worst.itertuples()
    ]
    return {
        "window_days": window_days, "n": n, "signature": signature,
        "n_winner_rows": int(len(mw)),
        "miss_rate": float(mw["missed"].mean()),
        "mean_missed_target": mean_missed,
        "mean_captured_target": mean_cap,
        "regret": (None if mean_missed is None or mean_cap is None
                   else round(mean_missed - mean_cap, 4)),
        "worst": worst_rows,
    }


def regret_markdown(window_days: int = 90, n: int = 5,
                    signature: str | None = None) -> str:
    """Markdown section for the self-correct prompt / CLI."""
    a = aggregate_regret(window_days=window_days, n=n, signature=signature)
    lines = [f"### Missed winners (top-{n}, last {window_days}d"
             + (f", sig={signature}" if signature else "") + ")"]
    if not a["n_winner_rows"]:
        lines.append("_No evaluated overlap between realized winners and ledger "
                     "runs yet._")
        return "\n".join(lines)
    lines.append(
        f"- miss_rate **{a['miss_rate']:.1%}** of realized top-{n} winners were "
        f"not surfaced  |  mean missed target **{(a['mean_missed_target'] or 0):+.2%}** "
        f"vs captured **{(a['mean_captured_target'] or 0):+.2%}**  "
        f"(regret {(a['regret'] or 0):+.2%})")
    if a["worst"]:
        lines += ["", "| as_of | symbol | realized T+2 | rank |",
                  "| --- | --- | --- | --- |"]
        for w in a["worst"]:
            lines.append(f"| {w['as_of']} | {w['symbol']} | "
                         f"{w['target']:+.2%} | {w['realized_rank']} |")
    return "\n".join(lines)


def union_candidates(standard: pd.DataFrame,
                     missed: pd.DataFrame) -> pd.DataFrame:
    """Union the standard and missed-variant candidate frames for the LLM modes,
    deduped by symbol (standard pricing wins on overlap). Adds ``also_missed``
    (the missed variant also surfaced it) and ``missed_only`` (only the missed
    variant did) so the plan/prompt can flag them and the LLM can weigh both."""
    std = standard.copy()
    if missed is None or missed.empty:
        std["also_missed"] = False
        std["missed_only"] = False
        return std
    missed_syms = set(missed["symbol"].astype(str))
    std_syms = set(std["symbol"].astype(str))
    std["also_missed"] = std["symbol"].astype(str).isin(missed_syms)
    std["missed_only"] = False
    extra = missed[~missed["symbol"].astype(str).isin(std_syms)].copy()
    if not extra.empty:
        extra["also_missed"] = True
        extra["missed_only"] = True
    return pd.concat([std, extra], ignore_index=True)


def latest_ab_summary() -> str | None:
    """One-line summary of the most recent ``backtest_ab_*.md`` report (the
    Verdict line), or None when no A/B has been run. Used to tell the LLM modes
    which mean head the out-of-sample A/B currently favors."""
    import glob
    from ..config import reports_dir
    files = sorted(glob.glob(str(reports_dir() / "backtest_ab_*.md")))
    if not files:
        return None
    try:
        from pathlib import Path
        text = Path(files[-1]).read_text(encoding="utf-8")
    except Exception:
        return None
    name = files[-1].replace("\\", "/").split("/")[-1]
    for line in text.splitlines():
        if line.startswith("**Verdict:**"):
            return f"{name}: {line.replace('**Verdict:**', '').strip().rstrip('.')}"
    return name


def missed_winner_weights(panel: pd.DataFrame, n: int = 5,
                          upweight: float = 3.0) -> pd.Series:
    """Training sample weights aligned to ``panel``: ``1.0`` everywhere,
    ``upweight`` on the realized top-N liquid winner rows (so the mean head can
    be retrained to lean toward catching them). Index matches ``panel``."""
    base = pd.Series(1.0, index=panel.index)
    if panel is None or panel.empty:
        return base
    top = realized_top_n(panel, n=n)[["as_of", "symbol"]].copy()
    if top.empty:
        return base
    top["_win"] = True
    p = panel.reset_index()
    date_col = "date" if "date" in p.columns else p.columns[0]
    p["as_of"] = pd.to_datetime(p[date_col]).dt.normalize()
    merged = p.merge(top, on=["as_of", "symbol"], how="left")
    # _win is True on matched winner rows, NaN otherwise → notna() = winner.
    w = np.where(merged["_win"].notna().to_numpy(), float(upweight), 1.0)
    return pd.Series(w, index=panel.index)
