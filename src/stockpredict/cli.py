"""Command-line entry point for the predictor."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import pandas as pd

from .config import load_config


@click.group()
def cli() -> None:
    """Vietnamese T+2 stock predictor."""


def _format_picks(picks) -> str:
    """One-line-per-pick view with entry / target / stop / fees / net (VND)
    sized for the configured position. Falls back to full dataframe if
    pricing columns aren't present."""
    if picks is None or len(picks) == 0:
        return "(no picks)"
    show_cols = [c for c in [
        "symbol", "entry_vnd", "target_vnd", "stop_vnd",
        "fees_round_trip_vnd", "net_reward_vnd", "net_loss_vnd",
        "rr_ratio", "actionable",
        "pred_mean", "news_score", "adjusted",
    ] if c in picks.columns]
    if not show_cols:
        return picks.to_string(index=False)
    fmt = picks[show_cols].copy()
    money_cols = [
        "entry_vnd", "target_vnd", "stop_vnd",
        "fees_round_trip_vnd", "net_reward_vnd", "net_loss_vnd",
    ]
    for c in money_cols:
        if c in fmt.columns:
            fmt[c] = fmt[c].map(
                lambda v: f"{int(v):>+9,}" if pd.notna(v) and c in ("net_reward_vnd",)
                          else (f"{int(v):>9,}" if pd.notna(v) else "        -")
            )
    return fmt.to_string(index=False)


def _format_picks_explained(picks) -> str:
    """Verbose paragraph-per-pick view used by claude/gemini modes when the
    LLM has produced business + key_news + rationale per ticker."""
    if picks is None or len(picks) == 0:
        return "(no picks)"
    parts: list[str] = []
    for i, r in picks.reset_index(drop=True).iterrows():
        sym = r["symbol"]
        organ = r.get("organ_name", "") or ""
        business = r.get("business", "") or ""
        header = f"=== #{i+1} {sym}"
        if organ:
            header += f"  —  {organ}"
        elif business:
            header += f"  —  {business[:60]}"
        # Multi-category BEST CHOICE badges (only set on actionable rows).
        badges = []
        for col, label in [("best_adjusted", "BEST adjusted"),
                           ("best_rr", "BEST rr"),
                           ("best_net", "BEST net"),
                           ("best_composite", "BEST composite")]:
            if bool(r.get(col, False)):
                badges.append(label)
        if badges:
            header += "  [" + " | ".join(badges) + "]"
        parts.append(header)

        if "entry_vnd" in r and pd.notna(r["entry_vnd"]):
            pos = int(r.get("position_units", 100))
            entry = int(r["entry_vnd"]); tgt = int(r["target_vnd"]); stop = int(r["stop_vnd"])
            fees = int(r.get("fees_round_trip_vnd", 0))
            net = int(r.get("net_reward_vnd", 0))
            rr = r.get("rr_ratio", float("nan"))
            actionable = bool(r.get("actionable", False))
            verdict = "ACTIONABLE" if actionable else "skip (rr/net too low)"
            parts.append(f"  Trade: buy {pos} @ {entry:,} VND  |  target {tgt:,}  |  stop {stop:,}")
            parts.append(f"  P&L (after ACBS fees {fees:,}): net {net:+,}  rr {rr:.2f}  -> {verdict}")

        ml_pred = r.get("pred_mean", None)
        ns = r.get("news_score", 0)
        adj = r.get("adjusted", None)
        if ml_pred is not None:
            line = f"  Signal: pred_mean={ml_pred:+.4f}"
            if pd.notna(ns):
                line += f"  news={int(ns):+d}"
            if adj is not None and pd.notna(adj):
                line += f"  adjusted={adj:+.4f}"
            parts.append(line)

        if business and organ:
            parts.append(f"  Business: {business}")
        elif business and not organ:
            parts.append(f"  Business: {business}")

        dims = r.get("dimensions", None)
        if isinstance(dims, (list, tuple)) and len(dims) > 0:
            parts.append("  Research dimensions: " + "; ".join(str(d) for d in dims))
        elif isinstance(dims, str) and dims.strip():
            parts.append(f"  Research dimensions: {dims}")

        drivers = r.get("drivers", None)
        if isinstance(drivers, (list, tuple)) and len(drivers) > 0:
            parts.append("  Key drivers: " + "; ".join(str(d) for d in drivers))
        elif isinstance(drivers, str) and drivers.strip():
            parts.append(f"  Key drivers: {drivers}")

        key_news = r.get("key_news", None)
        if isinstance(key_news, (list, tuple)) and len(key_news) > 0:
            parts.append("  News found:")
            for k in key_news:
                parts.append(f"    - {k}")
        elif isinstance(key_news, str) and key_news.strip():
            parts.append(f"  News found: {key_news}")

        rationale = r.get("rationale", "")
        if isinstance(rationale, str) and rationale.strip():
            parts.append(f"  Rationale: {rationale}")

        parts.append("")
    return "\n".join(parts)


def _has_explanations(picks) -> bool:
    """Pick the explained view when the LLM has produced any of these."""
    if picks is None or len(picks) == 0:
        return False
    return any(c in picks.columns for c in ("rationale", "business", "key_news"))


def _has_best_badges(picks) -> bool:
    """True when at least one row carries any of the four BEST flags —
    means we should print the verbose view to surface the badges."""
    if picks is None or len(picks) == 0:
        return False
    cols = [c for c in ("best_adjusted", "best_rr", "best_net", "best_composite")
            if c in picks.columns]
    if not cols:
        return False
    return bool(picks[cols].any().any())


# ---------------------------- data -----------------------------------------


@cli.command("update-data")
@click.option("--symbols", "-s", multiple=True, help="Specific tickers; default = full universe")
@click.option("--full", is_flag=True, help="Re-fetch full history instead of incremental")
@click.option("--limit", type=int, default=None, help="Cap symbol count (debug)")
def update_data(symbols: tuple[str, ...], full: bool, limit: int | None) -> None:
    """Refresh the OHLCV parquet cache from vnstock."""
    from .data.fetcher import update_many
    from .data.universe import filter_exchanges, load_universe

    if symbols:
        syms = [s.upper() for s in symbols]
    else:
        cfg = load_config()
        u = load_universe(refresh=True)
        u = filter_exchanges(u, cfg.data["exchanges"])
        syms = u["symbol"].tolist()
    if limit:
        syms = syms[:limit]
    click.echo(f"Updating {len(syms)} symbols (full={full})...")
    results = update_many(syms, full=full)
    ok = sum(1 for v in results.values() if isinstance(v, int))
    err = len(results) - ok
    click.echo(f"done. ok={ok} err={err}")
    if err:
        bad = [(k, v) for k, v in results.items() if not isinstance(v, int)][:10]
        for k, v in bad:
            click.echo(f"  {k}: {v}")


# ---------------------------- train ----------------------------------------


@cli.command("train")
@click.option("--start", default=None)
@click.option("--end", default=None)
def train_cmd(start: str | None, end: str | None) -> None:
    """Build the panel and fit a fresh model. Saves to models/latest.pkl."""
    from .dataset import build_panel
    from .model.train import save_latest, train

    click.echo("building panel...")
    panel = build_panel(start=start, end=end, require_target=True)
    click.echo(f"panel: {len(panel):,} rows across {panel['symbol'].nunique()} symbols")
    if panel.empty:
        click.echo("no data — run update-data first.", err=True)
        sys.exit(1)
    model = train(panel)
    path = save_latest(model)
    click.echo(f"trained {len(model.boosters)} boosters; saved -> {path}")


# ---------------------------- backtest -------------------------------------


@cli.command("backtest")
@click.option("--start", default=None)
@click.option("--end", default=None)
@click.option("--top", type=int, default=None)
def backtest_cmd(start: str | None, end: str | None, top: int | None) -> None:
    from .backtest.walk_forward import run, write_report

    click.echo("running walk-forward backtest...")
    res = run(start=start, end=end, top_k=top)
    out = write_report(res)
    click.echo(json.dumps(res.summary, indent=2))
    click.echo(f"report -> {out}")


# ---------------------------- predict --------------------------------------


@cli.command("predict")
@click.option("--mode", type=click.Choice(["base", "claude", "gemini"]), default="base")
@click.option("--top", type=int, default=5)
@click.option("--days", type=int, default=2,
              help="T+N exit window (min 2). Use the SAME days as the last train run.")
@click.option("--units", type=int, default=100, show_default=True,
              help="Position size in shares. Min 100 (ACBS rule).")
@click.option("--date", "on", default=None, help="YYYY-MM-DD; defaults to most recent cache date")
def predict_cmd(mode: str, top: int, days: int, units: int, on: str | None) -> None:
    if days < 2:
        click.echo("ERROR: --days must be >= 2.", err=True)
        sys.exit(2)
    if units < 100:
        click.echo("ERROR: --units must be >= 100 (ACBS rule).", err=True)
        sys.exit(2)
    if mode == "base":
        from .modes import base
        picks, out = base.run(top_k=top, on=on, units=units, exit_offset_days=days)
        click.echo(_format_picks(picks))
        click.echo(f"\nsaved -> {out}")
    elif mode == "claude":
        from .modes import claude
        result, out, tag = claude.run(top_k_final=top, on=on, units=units, exit_offset_days=days)
        click.echo(_format_picks(result))
        if _has_explanations(result):
            click.echo("")
            click.echo(_format_picks_explained(result))
        click.echo(f"\nsaved -> {out}  (path: {tag})")
        if tag == "interactive":
            click.echo("Next: ask Claude to fill the plan, then run claude-finalize.")
    elif mode == "gemini":
        from .modes import gemini
        result, out, tag = gemini.run(top_k_final=top, on=on, units=units, exit_offset_days=days)
        click.echo(_format_picks(result))
        if _has_explanations(result):
            click.echo("")
            click.echo(_format_picks_explained(result))
        click.echo(f"\nsaved -> {out}  (path: {tag})")
        if tag == "prompt-only":
            click.echo("Paste the prompt file's contents into Gemini Pro with browsing.")


@cli.command("claude-finalize")
@click.argument("plan_path", type=click.Path(exists=True))
@click.option("--top", type=int, default=5)
def claude_finalize_cmd(plan_path: str, top: int) -> None:
    from .modes import claude
    picks, out = claude.finalize(plan_path, top_k_final=top)
    click.echo(_format_picks(picks))
    if _has_explanations(picks):
        click.echo("")
        click.echo(_format_picks_explained(picks))
    click.echo(f"\nsaved -> {out}")


@cli.command("gemini-finalize")
@click.argument("prompt_path", type=click.Path(exists=True))
@click.option("--response", "response_path", type=click.Path(exists=True), default=None,
              help="Path to the JSON response from Gemini Chat. "
                   "Defaults to reports/gemini_response_<date>.json next to the prompt.")
@click.option("--top", type=int, default=5)
def gemini_finalize_cmd(prompt_path: str, response_path: str | None, top: int) -> None:
    """Merge Gemini Chat's JSON response with the saved candidates and produce
    the final explained top-K picks. Save Gemini's response as JSON first."""
    from .modes import gemini
    picks, out = gemini.finalize(prompt_path, response_path=response_path, top_k_final=top)
    click.echo(_format_picks(picks))
    if _has_explanations(picks):
        click.echo("")
        click.echo(_format_picks_explained(picks))
    click.echo(f"\nsaved -> {out}")


# ---------------------------- one-shot run --------------------------------


@cli.command("run")
@click.option("--duration", type=str, default="full", show_default=True,
              help="Minutes available, or 'full' for the entire universe with no time cap.")
@click.option("--mode", type=click.Choice(["base", "claude", "gemini"]), default="base")
@click.option("--days", type=str, default="2",
              help="T+N exit window. Integer (min 2 — Vietnamese T+2 settlement); "
                   "or 'end' = last trading day of the month (rolling to next "
                   "month if today is too close to month-end); "
                   "or 'earliest' = train+predict at T+N, T+N+1, T+N+2, … "
                   "(starting at --earliest-start, default T+2) and stop at "
                   "the first horizon that produces at least one actionable "
                   "pick. NO upper cap — runs until found, Ctrl+C to abort.")
@click.option("--earliest-start", type=int, default=2, show_default=True,
              help="Only used when --days earliest. T+N at which the search "
                   "begins (min 2 — Vietnamese T+2 settlement floor). "
                   "Ignored for any other --days value.")
@click.option("--top", type=int, default=5)
@click.option("--units", type=int, default=100, show_default=True,
              help="Position size in shares. Min 100 (ACBS rule). Rounds down to nearest 100.")
@click.option("--hose-only/--no-hose-only", default=False, show_default=True,
              help="Restrict the universe to HOSE-listed tickers only "
                   "(refreshes via VCI to get exchange info; falls back to "
                   "the ~43 curated HOSE tickers if exchange info is missing).")
@click.option("--warm-only", default="yes", show_default=True,
              type=click.Choice(["yes", "no", "always"], case_sensitive=False),
              help="Cache strategy. "
                   "`yes` (default) = smart lazy fetch: skip warm, fetch "
                   "only stale (newly-published bar) + cold (no parquet). "
                   "When a new trading day closes, stale auto-refreshes "
                   "on the next run, then back to zero API calls. "
                   "`always` = never fetch, drop tickers that aren't "
                   "already cached or are stale. Pure offline mode. "
                   "`no` = force full re-fetch of every selected symbol "
                   "(slow, rate-limited; use only for backfill).")
@click.option("--skip-train", is_flag=True,
              help="Use the existing models/latest.pkl instead of retraining.")
@click.option("--workers", type=int, default=2,
              help="Parallel fetcher threads. Keep low to stay under 20 req/min.")
def run_cmd(duration: str, mode: str, days: str, earliest_start: int, top: int,
            units: int, hose_only: bool, warm_only: str, skip_train: bool,
            workers: int) -> None:
    """End-to-end: size universe by time budget -> fetch -> train -> predict.

    Designed to be invoked from a double-click .bat. Enter how many minutes
    you have before market open (or `full` for no time cap), and the program
    sizes the universe accordingly.
    """
    import time as _time

    from .budget import plan as plan_budget
    from .data.cache import cached_symbols
    from .data.fetcher import update_many
    from .dataset import build_panel
    from .model.train import save_latest, train as train_model
    from .selector import select as select_symbols

    # ---- input validation ----
    # Three special values for --days:
    #   "end"      → resolve now against the calendar (one int)
    #   "earliest" → resolve LATER (after data fetch) by iterating T+N
    #   integer    → use as-is
    days_lower = str(days).strip().lower()
    earliest_mode = (days_lower == "earliest")
    if days_lower == "end":
        from .tracking import days_to_month_end
        try:
            days_int = days_to_month_end(pd.Timestamp.today().normalize(), min_days=2)
        except Exception as e:
            click.echo(f"ERROR: --days end failed: {e}", err=True)
            sys.exit(2)
        click.echo(f"--days end -> resolved to T+{days_int} "
                   f"(last trading day of {'next ' if days_int > 22 else ''}month, "
                   f"T+2 minimum respected).")
        days = days_int
    elif earliest_mode:
        if earliest_start < 2:
            click.echo("ERROR: --earliest-start must be >= 2 "
                       "(Vietnamese T+2 settlement minimum).", err=True)
            sys.exit(2)
        click.echo(f"--days earliest -> will iterate T+{earliest_start}, "
                   f"T+{earliest_start + 1}, T+{earliest_start + 2}, ... "
                   f"after data fetch (NO upper cap), stopping at the first "
                   f"horizon with >=1 actionable pick. Ctrl+C to abort.")
        days = earliest_start  # provisional; the search loop overrides this below
    else:
        try:
            days = int(days)
        except (TypeError, ValueError):
            click.echo(f"ERROR: --days must be an integer, 'end', or 'earliest'. "
                       f"Got {days!r}.", err=True)
            sys.exit(2)
    if days < 2:
        click.echo("ERROR: --days must be >= 2 (Vietnamese T+2 settlement minimum).", err=True)
        sys.exit(2)
    if units < 100:
        click.echo("ERROR: --units must be >= 100 (ACBS minimum lot rule).", err=True)
        sys.exit(2)
    if units % 100 != 0:
        new = (units // 100) * 100
        click.echo(f"[note] --units={units} not a multiple of 100; rounding down to {new}.")
        units = new
    # Parse duration: int minutes or "full"
    duration_arg: int | str
    if duration.strip().lower() == "full":
        duration_arg = "full"
    else:
        try:
            duration_arg = int(duration)
        except ValueError:
            click.echo(f"ERROR: --duration must be an integer (minutes) or 'full'. Got {duration!r}.",
                       err=True)
            sys.exit(2)

    # The model is horizon-specific. If --days != 2, force retraining; the cached
    # latest.pkl was almost certainly trained on T+2 returns.
    if days != 2 and skip_train:
        click.echo(f"[note] --days={days} != 2: forcing retrain "
                   f"(cached model is horizon-specific).")
        skip_train = False

    started = _time.time()
    rp = plan_budget(duration_arg, mode=mode)
    click.echo(rp.summary())
    click.echo(f"  horizon: T+{days}"
               + ("  (T+2: sell in afternoon session only — settlement noon T+2)"
                  if days == 2 else f"  (T+{days}: sell any time on the exit day)"))
    click.echo("")

    # Auto-evaluate any predictions that are now T+2 or later. This must run
    # AFTER the data refresh so we have closes for the target date — see below.

    syms = select_symbols(target=rp.universe_target, hose_only=hose_only)
    cached = set(cached_symbols())
    n_warm = len(set(syms) & cached)
    n_cold = len(syms) - n_warm
    label = "HOSE-only" if hose_only else "all exchanges"
    click.echo(f"selected {len(syms)} tickers  (warm={n_warm}, cold={n_cold})  [{label}]")
    click.echo("")

    # If the cold count exceeds the API budget, cap to the warm + budget tickers
    # so we don't promise more than we can fetch in time.
    if n_cold > rp.api_call_budget:
        warm = [s for s in syms if s in cached]
        cold = [s for s in syms if s not in cached][: rp.api_call_budget]
        syms = warm + cold
        click.echo(f"  capped to {len(syms)} ({len(warm)} warm + "
                   f"{len(cold)} new) to fit budget\n")

    # Quiet vnstock's noisy ERROR-level logger before bulk fetching — its
    # transient errors are already handled by our fallback + rate limiter.
    from .data.fetcher import audit_cache, quiet_vnstock_logger
    quiet_vnstock_logger()
    from .tracking import latest_expected_bar_date
    _expected_pre = latest_expected_bar_date()

    # Pre-flight cache audit so the user sees what's about to happen.
    warm, stale, cold = audit_cache(syms, expected_bar=_expected_pre)
    expected_str = (str(_expected_pre.date()) if _expected_pre is not None
                    else "(unknown)")
    click.echo(f"cache audit (expected bar = {expected_str}):")
    click.echo(f"  {len(warm):>5} cached and current  ->  no API call")
    click.echo(f"  {len(stale):>5} cached but stale    ->  incremental fetch")
    click.echo(f"  {len(cold):>5} missing             ->  full-history fetch")

    warm_mode = warm_only.lower()
    if warm_mode == "always":
        # Pure offline: drop everything that isn't already cached AND current.
        # Zero API calls, guaranteed.
        if stale or cold:
            dropped = len(stale) + len(cold)
            click.echo(f"  --warm-only=always: dropping {dropped} "
                       f"non-warm ticker(s) ({len(stale)} stale + "
                       f"{len(cold)} cold); running on {len(warm)} "
                       f"current symbols only")
            syms = warm
        else:
            click.echo(f"  --warm-only=always: all {len(warm)} symbols current, "
                       "no API calls will be made")
        if not syms:
            click.echo("ERROR: --warm-only=always but no warm symbols; "
                       "either populate the cache (run with --warm-only=yes "
                       "first) or relax the flag.", err=True)
            sys.exit(1)
    elif warm_mode == "yes":
        # Smart lazy fetch: skip warm, fetch only stale + cold via update_many.
        n_to_fetch = len(stale) + len(cold)
        if n_to_fetch == 0:
            click.echo(f"  --warm-only=yes: all {len(warm)} symbols current, "
                       "no API calls needed")
        else:
            click.echo(f"  --warm-only=yes: {len(warm)} current + "
                       f"{n_to_fetch} need fetch ({len(stale)} stale + "
                       f"{len(cold)} cold)")
    else:  # warm_mode == "no"
        click.echo(f"  --warm-only=no: forcing full re-fetch of all "
                   f"{len(syms)} symbols (rate-limited, slow)")
    click.echo("")

    if warm_mode == "always":
        # Skip the network entirely. Use cached parquets as-is.
        click.echo("skipping data refresh (--warm-only=always)")
        results = {s: 0 for s in syms}
        ok = len(results)
        err = 0
    else:
        click.echo(f"updating data (workers={workers})...")
        # warm_mode == "yes" → update_many internally audits and only
        # fetches stale + cold (lazy).
        # warm_mode == "no" → full=True forces re-fetch of every symbol.
        results = update_many(syms, full=(warm_mode == "no"), workers=workers)
        ok = sum(1 for v in results.values() if isinstance(v, int))
        err = len(results) - ok
        # Re-audit AFTER the fetch so the user can see what actually persisted
        # to disk. If pre-fetch said "10 stale + 5 cold" and post-fetch still
        # says "10 stale + 5 cold", something's wrong (writes didn't take).
        warm_after, stale_after, cold_after = audit_cache(syms, expected_bar=_expected_pre)
        moved = (len(warm_after) - len(warm))
        click.echo(f"  post-fetch audit: {len(warm_after)} warm "
                   f"(+{moved} since start)  |  {len(stale_after)} stale  |  "
                   f"{len(cold_after)} cold")
        new_rows = sum(v for v in results.values() if isinstance(v, int))
        if new_rows == 0 and (stale_after or cold_after):
            click.echo("  [warn] no new rows written yet still have stale/cold "
                       "tickers — check rate-limit count below.")
    rate_limit_errs = sum(
        1 for v in results.values()
        if not isinstance(v, int) and "rate" in str(v).lower()
    )
    click.echo(f"  fetched: ok={ok} err={err}"
               + (f" (rate-limit: {rate_limit_errs})" if rate_limit_errs else ""))
    if err and err <= 5:
        for k, v in results.items():
            if not isinstance(v, int):
                click.echo(f"    {k}: {str(v)[:120]}")
    elif err > 5:
        click.echo(f"  {err} ticker(s) failed; "
                   f"continuing with what's already cached. "
                   f"Re-run later (or with `--workers 1`) to backfill the rest.")
    click.echo("")

    # Critical: fall through and predict on whatever IS cached, even if
    # many fetches failed. The ML model will still produce useful picks
    # from the warm-cache subset; aborting the run loses that signal.
    cached_now = set(cached_symbols())
    syms_with_data = [s for s in syms if s in cached_now]
    if not syms_with_data:
        click.echo("ERROR: no cached data for any selected symbol — cannot proceed.",
                   err=True)
        sys.exit(1)
    if len(syms_with_data) < len(syms):
        click.echo(f"  proceeding with {len(syms_with_data)} cached symbols "
                   f"({len(syms) - len(syms_with_data)} skipped due to fetch errors)")
    syms = syms_with_data

    # Now that data is fresh, evaluate any predictions whose T+N has elapsed
    from .tracking import evaluate_pending, recent_performance
    updated = evaluate_pending()
    if not updated.empty:
        click.echo(f"auto-evaluated {len(updated)} prior pick(s):")
        click.echo(updated[["as_of", "mode", "symbol", "realized_return"]].to_string(index=False))
        click.echo("")
    perf = recent_performance(window_days=90, mode=mode)
    if perf.get("n", 0) > 0:
        click.echo(f"recent {mode} performance: n={perf['n']}  "
                   f"hit={perf['hit_rate']:.1%}  mean_ret={perf['mean_return']:+.4f}")
        click.echo("")

    # Pass the selected symbols through so prediction is restricted to the
    # same set we trained on (and the hose_only filter actually takes effect
    # at predict time, not just at fetch time). Computed once and reused by
    # both the earliest-search loop and the final mode invocation.
    pred_syms = syms if hose_only else None

    if earliest_mode:
        # Iterative search: train at T+earliest_start, T+earliest_start+1, ...
        # with NO upper cap, stopping at the first horizon that produces >= 1
        # actionable pick. The user can Ctrl+C to abort if the search drags on.
        from .model.predict import rank_today
        click.echo(f"searching for earliest actionable horizon "
                   f"(starting T+{earliest_start}, no upper cap; "
                   f"trains a fresh model per horizon — Ctrl+C to abort)...")
        found_horizon: int | None = None
        last_picks = None
        n = earliest_start
        # Bookkeeping for periodic milestone callouts and consecutive-empty
        # safeguard. We bail with a clear error if the data simply can't
        # support any horizon (panel empty for many tries in a row), since
        # that's a config/data problem rather than something a longer
        # search would fix.
        consecutive_empty = 0
        EMPTY_PANEL_LIMIT = 60  # ~3 trading months of horizons with no labels
        while True:
            click.echo(f"  T+{n}...", nl=False)
            panel_n = build_panel(symbols=syms, require_target=True,
                                  exit_offset_days=n)
            if panel_n.empty:
                click.echo(" no rows; skip")
                consecutive_empty += 1
                if consecutive_empty >= EMPTY_PANEL_LIMIT:
                    click.echo(f"ERROR: panel empty for {EMPTY_PANEL_LIMIT} "
                               f"consecutive horizons (last tried T+{n}). "
                               f"Either history is too short or symbol set "
                               f"is too narrow — extend cfg.data.history_start "
                               f"or fetch more tickers and retry.", err=True)
                    sys.exit(1)
                n += 1
                continue
            consecutive_empty = 0
            m = train_model(panel_n)
            save_latest(m)
            picks_n = rank_today(top_k=top, units=units,
                                 exit_offset_days=n,
                                 symbols=pred_syms)
            last_picks = picks_n
            if "actionable" in picks_n.columns and bool(picks_n["actionable"].any()):
                count = int(picks_n["actionable"].sum())
                click.echo(f" found {count} actionable pick(s)")
                found_horizon = n
                break
            click.echo(" none")
            # Milestone callout every 30 horizons so the user knows the
            # process is still alive and can decide whether to keep waiting.
            if n > earliest_start and (n - earliest_start + 1) % 30 == 0:
                click.echo(f"  ... still searching past T+{n} — "
                           f"Ctrl+C to abort if you want to stop ...")
            n += 1
        days = found_horizon
        # Model on disk is already for `days` — skip the next train below.
        skip_train = True
        click.echo("")
    elif not skip_train:
        click.echo(f"training (horizon T+{days})...")
        panel = build_panel(symbols=syms, require_target=True, exit_offset_days=days)
        if panel.empty:
            click.echo("no training rows. aborting.", err=True)
            sys.exit(1)
        click.echo(f"  panel: {len(panel):,} rows / {panel['symbol'].nunique()} symbols")
        model = train_model(panel)
        save_latest(model)
        click.echo("  model saved.")
        click.echo("")

    click.echo(f"predicting (mode={mode})...")
    if mode == "base":
        from .modes import base
        picks, out = base.run(top_k=top, units=units, exit_offset_days=days,
                              symbols=pred_syms, hose_only=hose_only)
        click.echo("")
        click.echo(_format_picks(picks))
        if _has_best_badges(picks):
            click.echo("")
            click.echo(_format_picks_explained(picks))
        click.echo(f"\nsaved -> {out}")
    elif mode == "claude":
        from .modes import claude
        result, out, tag = claude.run(top_k_final=top, units=units,
                                       exit_offset_days=days, symbols=pred_syms,
                                       hose_only=hose_only)
        click.echo("")
        click.echo(_format_picks(result))
        if _has_explanations(result) or _has_best_badges(result):
            click.echo("")
            click.echo(_format_picks_explained(result))
        click.echo(f"\nsaved -> {out}  (path: {tag})")
        if tag == "interactive":
            click.echo("")
            click.echo("==> NEXT (run inside Claude Code / Cowork):")
            click.echo("    1. Ask Claude to fetch every URL in the plan and fill the score table.")
            click.echo("    2. Then run:")
            click.echo(f"       python -m stockpredict.cli claude-finalize \"{out}\"")
            click.echo("    Tip: set ANTHROPIC_API_KEY in .env to skip this manual step.")
    elif mode == "gemini":
        from .modes import gemini
        result, out, tag = gemini.run(top_k_final=top, units=units,
                                       exit_offset_days=days, symbols=pred_syms,
                                       hose_only=hose_only)
        click.echo("")
        click.echo(_format_picks(result))
        if _has_explanations(result) or _has_best_badges(result):
            click.echo("")
            click.echo(_format_picks_explained(result))
        click.echo(f"\nsaved -> {out}  (path: {tag})")
        if tag == "prompt-only":
            click.echo("")
            click.echo("==> NEXT: paste the prompt file's contents into Gemini Pro with browsing.")
            click.echo("    Tip: set GEMINI_API_KEY in .env to run autonomously.")

    elapsed = (_time.time() - started) / 60.0
    click.echo(f"\nelapsed: {elapsed:.1f} min  /  budget: {duration}")


# ---------------------------- evaluate / track ----------------------------


@cli.command("evaluate")
@click.option("--refresh-data/--no-refresh-data", default=True,
              help="Run incremental fetch on tickers in pending evaluations first.")
def evaluate_cmd(refresh_data: bool) -> None:
    """Score past predictions whose T+2 has now passed."""
    from .data.fetcher import update_many
    from .tracking import _read, evaluate_pending, recent_performance

    if refresh_data:
        df = _read()
        if not df.empty:
            pending_syms = sorted(df[~df["evaluated"]]["symbol"].unique().tolist())
            if pending_syms:
                click.echo(f"refreshing data for {len(pending_syms)} symbols...")
                update_many(pending_syms, full=False, workers=2)

    updated = evaluate_pending()
    click.echo(f"newly evaluated: {len(updated)}")
    if not updated.empty:
        cols = ["as_of", "target_date", "mode", "symbol", "rank",
                "pred_mean", "news_score", "entry_price", "actual_exit",
                "realized_return"]
        click.echo(updated[cols].to_string(index=False))

    click.echo("\n=== recent performance (last 90 days) ===")
    for mode in ("base", "claude", "gemini", None):
        label = mode or "ALL"
        perf = recent_performance(window_days=90, mode=mode)
        if perf.get("n", 0) == 0:
            click.echo(f"  {label}: {perf.get('note')}")
        else:
            click.echo(f"  {label}: n={perf['n']}  hit={perf['hit_rate']:.1%}  "
                       f"mean_ret={perf['mean_return']:+.4f}  "
                       f"med_ret={perf['median_return']:+.4f}")


@cli.command("track")
@click.option("--mode", type=click.Choice(["base", "claude", "gemini"]), default=None)
@click.option("--limit", type=int, default=20)
def track_cmd(mode: str | None, limit: int) -> None:
    """Print the most recent prediction ledger entries."""
    from .tracking import _read

    df = _read()
    if df.empty:
        click.echo("no predictions recorded yet.")
        return
    if mode:
        df = df[df["mode"] == mode]
    df = df.sort_values(["as_of", "rank"], ascending=[False, True]).head(limit)
    cols = ["as_of", "target_date", "mode", "symbol", "rank",
            "pred_mean", "news_score", "entry_price",
            "realized_return", "evaluated"]
    click.echo(df[cols].to_string(index=False))


# ---------------------------- diagnostics ---------------------------------


@cli.command("status")
def status_cmd() -> None:
    """Show what's cached / trained on disk."""
    from .config import cache_dir, models_dir
    from .data.cache import cached_symbols

    syms = cached_symbols()
    click.echo(f"cached symbols: {len(syms)}")
    if syms:
        click.echo(f"  example: {syms[:5]}")
    m = models_dir() / "latest.pkl"
    click.echo(f"latest model: {'present' if m.exists() else 'missing'}  ({m})")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
