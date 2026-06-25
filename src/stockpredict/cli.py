"""Command-line entry point for the predictor."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import pandas as pd

from .config import load_config

# Force UTF-8 on stdout/stderr so Vietnamese text in plans/picks doesn't crash
# the default Windows cp1252 console codec.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


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
        "symbol", "close_vnd", "entry_vnd", "target_vnd", "stop_vnd",
        "fees_round_trip_vnd", "net_reward_vnd", "net_loss_vnd",
        "rr_ratio", "below_breakeven",
        "pred_mean", "news_score", "adjusted",
    ] if c in picks.columns]
    if not show_cols:
        return picks.to_string(index=False)
    fmt = picks[show_cols].copy()
    money_cols = [
        "close_vnd", "entry_vnd", "target_vnd", "stop_vnd",
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
        parts.append(header)

        if "entry_vnd" in r and pd.notna(r["entry_vnd"]):
            entry = int(r["entry_vnd"]); tgt = int(r["target_vnd"]); stop = int(r["stop_vnd"])
            fees = int(r.get("fees_round_trip_vnd", 0))
            net = int(r.get("net_reward_vnd", 0))
            rr = r.get("rr_ratio", float("nan"))
            below = bool(r.get("below_breakeven", False))
            verdict = "BELOW-BAR (weak edge)" if below else "OK"
            close_v = r.get("close_vnd", None)
            dip_pct = r.get("entry_limit_pct", None)
            if close_v is not None and pd.notna(close_v) and dip_pct is not None and pd.notna(dip_pct) and float(dip_pct) < 0:
                # Entry sits below close — surface the predicted dip so the
                # user knows the limit isn't a market order at close.
                parts.append(
                    f"  Trade: LIMIT-buy @ {entry:,} VND "
                    f"(close {int(close_v):,}, dip {float(dip_pct):+.2%})  "
                    f"|  target {tgt:,}  |  stop {stop:,}"
                )
            else:
                parts.append(f"  Trade: buy @ {entry:,} VND  |  target {tgt:,}  |  stop {stop:,}")
            parts.append(f"  P&L per share (after ACBS fees {fees:,}): net {net:+,}  rr {rr:.2f}  -> {verdict}")

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


def _print_pick_warnings(picks, requested_n: int) -> None:
    """Surface the two exact-N caveats: a SHORTFALL when the eligible universe
    couldn't supply ``requested_n`` names, and a QUALITY note counting picks
    below the break-even bar (returned anyway to honor the exact count)."""
    n = int(len(picks)) if picks is not None else 0
    if requested_n and n < int(requested_n):
        click.echo("")
        click.echo(f"==> SHORTFALL: only {n} of {requested_n} requested pick(s) "
                   f"available — the eligible universe is smaller than N "
                   f"(heavy --exclude / --hose-only / tiny cache).")
    if picks is None or n == 0 or "below_breakeven" not in picks.columns:
        return
    k = int(picks["below_breakeven"].fillna(True).astype(bool).sum())
    if k > 0:
        click.echo("")
        click.echo(f"==> QUALITY: {k} of {n} pick(s) are below the break-even "
                   f"bar (weak edge — forecast under round-trip cost). They're "
                   f"shown to honor --picks; treat them with extra caution.")


def _print_sell_reminder(picks, *, as_of, exit_offset_days, mode_label) -> None:
    """Surface a structured sell-reminder block for the returned picks. Both
    the LLM (Claude / Gemini) running in the surrounding session and the user
    reading the terminal can act on this: schedule a reminder for the target
    sell day, in GMT+7 (Asia/Ho_Chi_Minh, Vietnamese ICT)."""
    if picks is None or len(picks) == 0:
        return
    n_picks = int(len(picks))
    if exit_offset_days is None:
        return
    from .tracking import _next_trading_offset, effective_today_for_trading

    as_of_ts = (pd.Timestamp(as_of) if as_of is not None
                else effective_today_for_trading())
    n = int(exit_offset_days)
    target_date = _next_trading_offset(as_of_ts, n)
    # Reminder fires at 11:30 ICT on the sell day itself — late morning,
    # just before the lunch break. For T+2 this lands 30 min before noon
    # settlement, giving the user time to queue afternoon-session orders;
    # for T+>2 it's the natural mid-day check-in before the afternoon close.
    reminder_date = target_date
    reminder_time = "11:30 ICT"
    if n == 2:
        sell_window = ("13:00–14:30 ICT  (afternoon session, "
                       "after T+2 settlement at noon)")
        reminder_note = "30 min before T+2 settlement"
    else:
        sell_window = "09:00–14:30 ICT  (any time during the trading day)"
        reminder_note = "late morning of sell day, before lunch break"
    sym_list = ", ".join(picks["symbol"].astype(str).tolist())
    click.echo("")
    click.echo("==> SELL-REMINDER (GMT+7, Asia/Ho_Chi_Minh — Vietnamese ICT):")
    click.echo(f"    {n_picks} pick(s): {sym_list}")
    click.echo(f"    Sell day: {target_date.strftime('%Y-%m-%d (%A)')} "
               f"({sell_window}).")
    click.echo(f"    Suggested reminder: "
               f"{reminder_date.strftime('%Y-%m-%d (%A)')} {reminder_time} "
               f"({reminder_note}).")
    if mode_label in ("claude", "gemini"):
        click.echo(f"    {mode_label.title()}: ask the user whether they want a "
                   f"reminder scheduled for that day/time in GMT+7.")


# ---------------------------- data -----------------------------------------


@cli.command("update-data")
@click.option("--symbols", "-s", multiple=True, help="Specific tickers; default = full universe")
@click.option("--full", is_flag=True, help="Re-fetch full history instead of incremental")
@click.option("--limit", type=int, default=None, help="Cap symbol count (debug)")
def update_data(symbols: tuple[str, ...], full: bool, limit: int | None) -> None:
    """Refresh the OHLCV parquet cache from vnstock."""
    from .data.fetcher import update_many
    from .data.intro import introduce
    from .data.universe import filter_exchanges, load_universe

    introduce()

    if symbols:
        syms = [s.upper() for s in symbols]
    else:
        cfg = load_config()
        u = load_universe(refresh=True)
        u = filter_exchanges(u, cfg.data["exchanges"])
        syms = u["symbol"].tolist()
        # Defensive union with the curated list (which now includes HOSE_ETFS).
        # vnstock's Listing API sometimes omits ETFs depending on source; the
        # curated list guarantees they get fetched into the cache so the
        # model panel can include them after training.
        from .selector import CURATED
        seen = {s.upper() for s in syms}
        for s in CURATED:
            if s.upper() not in seen:
                syms.append(s.upper())
                seen.add(s.upper())
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
@click.option("--skip-low/--no-skip-low", default=False, show_default=True,
              help="Skip the low-quantile head (entry-limit predictor). "
                   "When skipped, predictions fall back to entry = close.")
def train_cmd(start: str | None, end: str | None, skip_low: bool) -> None:
    """Build the panel and fit fresh mean + low models.

    Saves the mean head to ``models/latest.pkl`` and the low-quantile head
    to ``models/low_latest.pkl`` (unless ``--skip-low`` is set)."""
    from .dataset import build_panel
    from .model.train import save_latest, save_latest_low, train, train_quantile

    click.echo("building panel...")
    # require_target=False so target_low rows survive; we drop NaN per-head.
    panel = build_panel(start=start, end=end, require_target=False)
    click.echo(f"panel: {len(panel):,} rows across {panel['symbol'].nunique()} symbols")
    if panel.empty:
        click.echo("no data — run update-data first.", err=True)
        sys.exit(1)

    panel_mean = panel.dropna(subset=["target"])
    click.echo(f"  mean head: {len(panel_mean):,} rows with target")
    model = train(panel_mean)
    path = save_latest(model)
    click.echo(f"  trained {len(model.boosters)} boosters; saved -> {path}")

    if skip_low:
        click.echo("  skipping low head (--skip-low).")
        return
    if "target_low" not in panel.columns:
        click.echo("  no target_low column — OHLCV cache may lack 'low' "
                   "(re-run update-data). Skipping low head.")
        return
    panel_low = panel.dropna(subset=["target_low"])
    if panel_low.empty:
        click.echo("  no rows with target_low — skipping low head.")
        return
    click.echo(f"  low head: {len(panel_low):,} rows with target_low")
    low_model = train_quantile(panel_low)
    low_path = save_latest_low(low_model)
    click.echo(f"  built empirical low head (alpha={low_model.alpha:.2f}, "
               f"lookback={low_model.lookback}d); saved -> {low_path}")


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


# ---------------------------- missed-winners -------------------------------


@cli.command("regret")
@click.option("--window", type=int, default=90, show_default=True,
              help="Look-back window in days.")
@click.option("--picks", "-n", "n", type=int, default=None,
              help="Top-N realized winners per day (default: pricing.default_picks).")
@click.option("--signature", default=None,
              help="Restrict to one run signature (e.g. base_d2).")
def regret_cmd(window: int, n: int | None, signature: str | None) -> None:
    """Report the realized top-N winners the model MISSED over a window."""
    from .analyze import regret as regret_mod
    from .config import load_config, reports_dir
    nn = int(n) if n else int(load_config().pricing.get("default_picks", 5))
    summary = regret_mod.aggregate_regret(window_days=window, n=nn,
                                          signature=signature)
    click.echo(json.dumps(summary, indent=2, default=str))
    md = regret_mod.regret_markdown(window_days=window, n=nn, signature=signature)
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    suffix = f"_{signature}" if signature else ""
    out = reports_dir() / f"regret_{today}{suffix}.md"
    out.write_text(md, encoding="utf-8")
    click.echo(f"\nreport -> {out}")


@cli.command("train-missed")
@click.option("--upweight", type=float, default=3.0, show_default=True,
              help="Sample weight on realized top-N winner rows.")
@click.option("--picks", "-n", "n", type=int, default=None,
              help="Top-N winners per day to upweight (default: pricing.default_picks).")
def train_missed_cmd(upweight: float, n: int | None) -> None:
    """Train the missed-winners VARIANT mean head (upweights realized winners).

    Saves to ``models/latest_missed.pkl`` — the standard ``latest.pkl`` is left
    untouched. Use with ``run --variant missed`` and validate via ``backtest-ab``
    before trusting it (upweighting the tail can LOWER win rate)."""
    from .analyze.regret import missed_winner_weights
    from .config import load_config
    from .dataset import build_panel
    from .model.train import save_latest_missed, train

    nn = int(n) if n else int(load_config().pricing.get("default_picks", 5))
    click.echo("building panel...")
    panel = build_panel(require_target=False)
    panel_mean = panel.dropna(subset=["target"])
    if panel_mean.empty:
        click.echo("no training rows. aborting.", err=True)
        sys.exit(1)
    weights = missed_winner_weights(panel_mean, n=nn, upweight=upweight)
    n_up = int((weights > 1.0).sum())
    click.echo(f"  upweighting {n_up:,} winner rows (x{upweight}) of "
               f"{len(panel_mean):,} total")
    model = train(panel_mean, weights=weights)
    path = save_latest_missed(model)
    click.echo(f"  saved variant -> {path}")


def _write_backtest_ab_report(bt_run, missed_winner_weights, *, n,
                              start=None, end=None, top=None, upweight=3.0):
    """Run the standard-vs-missed walk-forward A/B, print the table, and write
    reports/backtest_ab_<date>.md with BOTH models' numbers. Overwrites NO
    model — it's advisory only. Returns the report path."""
    from .config import reports_dir
    a = bt_run(start=start, end=end, top_k=top).summary
    b = bt_run(start=start, end=end, top_k=top,
               weights_fn=lambda p: missed_winner_weights(p, n=n, upweight=upweight)).summary
    keys = ["n_trades", "hit_rate", "hit_rate_net", "mean_return", "mean_return_net"]
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    click.echo(f"\n{'metric':<18}{'standard':>14}{'missed':>14}")
    for k in keys:
        click.echo(f"{k:<18}{fmt(a.get(k)):>14}{fmt(b.get(k)):>14}")
    better = (b.get("hit_rate", 0) or 0) >= (a.get("hit_rate", 0) or 0)
    verdict = ("variant win_rate >= standard — candidate looks OK" if better
               else "variant win_rate < standard — keep the standard model")
    click.echo(f"\n==> {verdict}")
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    lines = [f"# Backtest A/B — standard vs missed-winners variant ({today})", "",
             f"upweight={upweight}, picks={n}, "
             f"window={a.get('start','?')}..{a.get('end','?')}", "",
             "| metric | standard | missed |", "| --- | --- | --- |"]
    lines += [f"| {k} | {fmt(a.get(k))} | {fmt(b.get(k))} |" for k in keys]
    lines += ["", f"**Verdict:** {verdict}.",
              "", "_Advisory only — no model was overwritten. The standard model "
              "stays live; the variant is `models/latest_missed.pkl`, used only "
              "via the `_missed` reports._"]
    out = reports_dir() / f"backtest_ab_{today}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


@cli.command("backtest-ab")
@click.option("--start", default=None)
@click.option("--end", default=None)
@click.option("--top", type=int, default=None)
@click.option("--upweight", type=float, default=3.0, show_default=True)
@click.option("--picks", "-n", "n", type=int, default=None)
def backtest_ab_cmd(start, end, top, upweight, n) -> None:
    """A/B the standard model vs the missed-winners variant on win rate.

    Writes reports/backtest_ab_<date>.md with both models' numbers. Advisory
    only — overwrites NO model. You decide whether the variant is worth using
    (it stays at models/latest_missed.pkl, surfaced via the _missed reports)."""
    from .analyze.regret import missed_winner_weights
    from .backtest.walk_forward import run
    from .config import load_config

    nn = int(n) if n else int(load_config().pricing.get("default_picks", 5))
    click.echo("backtest A: standard | B: missed-winners variant (slow)...")
    out = _write_backtest_ab_report(run, missed_winner_weights, n=nn,
                                    start=start, end=end, top=top, upweight=upweight)
    click.echo(f"report -> {out}")


# ---------------------------- predict --------------------------------------


@cli.command("predict")
@click.option("--mode", type=click.Choice(["base", "claude", "gemini"]), default="base")
@click.option("--picks", "-n", "n_picks", type=int, default=None,
              help="How many picks to surface (exactly this many, top by score). "
                   "Defaults to pricing.default_picks in config.yaml.")
@click.option("--date", "on", default=None, help="YYYY-MM-DD; defaults to most recent cache date")
def predict_cmd(mode: str, n_picks: int | None, on: str | None) -> None:
    if n_picks is not None and n_picks < 1:
        click.echo("ERROR: --picks must be >= 1.", err=True)
        sys.exit(2)
    if mode == "base":
        from .modes import base
        picks, out = base.run(on=on, n_picks=n_picks)
        click.echo(_format_picks(picks))
        click.echo(f"\nsaved -> {out}")
    elif mode == "claude":
        from .modes import claude
        result, out, tag = claude.run(on=on, n_picks=n_picks)
        click.echo(_format_picks(result))
        if _has_explanations(result):
            click.echo("")
            click.echo(_format_picks_explained(result))
        click.echo(f"\nsaved -> {out}  (path: {tag})")
        # Claude mode only emits the interactive plan; the sell reminder lands
        # at claude-finalize, not here.
        click.echo("Next: ask Claude to fill the plan, then run claude-finalize.")
    elif mode == "gemini":
        from .modes import gemini
        result, out, tag = gemini.run(on=on, n_picks=n_picks)
        click.echo(_format_picks(result))
        if _has_explanations(result):
            click.echo("")
            click.echo(_format_picks_explained(result))
        click.echo(f"\nsaved -> {out}  (path: {tag})")
        if tag == "prompt-only":
            click.echo("Paste the prompt file's contents into Gemini Pro with browsing.")


@cli.command("claude-finalize")
@click.argument("plan_path", type=click.Path(exists=True))
def claude_finalize_cmd(plan_path: str) -> None:
    from .modes import claude
    picks, out = claude.finalize(plan_path)
    click.echo(_format_picks(picks))
    if _has_explanations(picks):
        click.echo("")
        click.echo(_format_picks_explained(picks))
    click.echo(f"\nsaved -> {out}")
    # Recover horizon / requested count from the saved picks JSON so the
    # reminder lands on the correct sell day and warnings reflect the request.
    try:
        payload = json.loads(Path(out).read_text(encoding="utf-8"))
        _print_pick_warnings(picks, payload.get("requested_picks") or len(picks))
        _print_sell_reminder(picks, as_of=payload.get("as_of"),
                             exit_offset_days=payload.get("exit_offset_days"),
                             mode_label="claude")
    except Exception:
        pass


@cli.command("gemini-finalize")
@click.argument("prompt_path", type=click.Path(exists=True))
@click.option("--response", "response_path", type=click.Path(exists=True), default=None,
              help="Path to the JSON response from Gemini Chat. "
                   "Defaults to reports/gemini_response_<date>.json next to the prompt.")
def gemini_finalize_cmd(prompt_path: str, response_path: str | None) -> None:
    """Merge Gemini Chat's JSON response with the saved candidates and produce
    the final explained picks. Save Gemini's response as JSON first."""
    from .modes import gemini
    picks, out = gemini.finalize(prompt_path, response_path=response_path)
    click.echo(_format_picks(picks))
    if _has_explanations(picks):
        click.echo("")
        click.echo(_format_picks_explained(picks))
    click.echo(f"\nsaved -> {out}")
    try:
        payload = json.loads(Path(out).read_text(encoding="utf-8"))
        _print_pick_warnings(picks, payload.get("requested_picks") or len(picks))
        _print_sell_reminder(picks, as_of=payload.get("as_of"),
                             exit_offset_days=payload.get("exit_offset_days"),
                             mode_label="gemini")
    except Exception:
        pass


# ---------------------------- one-shot run --------------------------------


@cli.command("run")
@click.option("--mode", type=click.Choice(["base", "claude", "gemini"]), default="base")
@click.option("--picks", "-n", "n_picks", type=int, default=None,
              help="How many picks to surface. The program always returns "
                   "EXACTLY this many — it ranks the whole scored universe by "
                   "predicted return and keeps the top N, so the difficulty "
                   "(the implicit edge cutoff) floats to whatever admits exactly "
                   "N. Picks below the break-even quality bar are still returned "
                   "but flagged, with a count in the QUALITY warning. Always T+2. "
                   "Defaults to pricing.default_picks in config.yaml.")
@click.option("--hose-only/--no-hose-only", default=False, show_default=True,
              help="Restrict the universe to HOSE-listed tickers only "
                   "(refreshes via VCI to get exchange info; falls back to "
                   "the ~43 curated HOSE tickers if exchange info is missing).")
@click.option("--etfs/--no-etfs", "include_etfs", default=True, show_default=True,
              help="Include HOSE-listed ETFs (FUEVFVND, E1VFVN30, FUESSV30, …) "
                   "in the universe. --no-etfs filters every layer (curated, "
                   "warm cache, top-up) to stocks only. ETFs are identified "
                   "via the cached universe's instrument_type column with a "
                   "symbol-shape fallback (FUE* / E1VFVN30).")
@click.option("--exclude", "exclude", multiple=True,
              help="Ticker(s) to exclude from this run. Repeatable "
                   "(--exclude ACB --exclude HPG) or comma-separated "
                   "(--exclude ACB,HPG). Per-session only — not persisted to "
                   "config. Excluded tickers are stripped from every universe "
                   "layer (curated, warm cache, top-up) and from the prediction "
                   "panel. The run signature is suffixed with `_x{TICKERS}` so "
                   "the picks JSON doesn't collide with a same-day full run.")
@click.option("--warm-only", default="yes", show_default=True,
              type=click.Choice(["yes", "no", "always"], case_sensitive=False),
              help="Cache strategy. "
                   "`yes` (default) = smart lazy fetch: skip warm, fetch "
                   "only stale (newly-published bar) + cold (no parquet). "
                   "When a new trading day closes, stale auto-refreshes "
                   "on the next run, then back to zero API calls. "
                   "`always` = never fetch; run on whatever parquet is "
                   "already on disk (warm + stale). Only cold tickers "
                   "(no parquet at all) are dropped. Pure offline mode. "
                   "`no` = force full re-fetch of every selected symbol "
                   "(slow, rate-limited; use only for backfill).")
@click.option("--skip-train", is_flag=True,
              help="Use the existing models/latest.pkl instead of retraining.")
@click.option("--missed/--no-missed", "do_missed", default=True, show_default=True,
              help="Involve the missed-winners variant. base mode: also writes a "
                   "second _missed pick report. claude/gemini: UNIONs the missed "
                   "variant's top picks into the candidates the LLM researches "
                   "(deduped), so the LLM judges both rankings. On by default; "
                   "--no-missed for standard candidates only.")
@click.option("--ab/--no-ab", "do_ab", default=None,
              help="After predicting, run the standard-vs-missed walk-forward A/B "
                   "and write reports/backtest_ab_<date>.md (overwrites no model). "
                   "Default: OFF for base, ON for claude/gemini (so the LLM can "
                   "weigh the verdict). SLOW (~10 min, retrains across years).")
@click.option("--workers", type=int, default=2,
              help="Parallel fetcher threads. Keep low to stay under 20 req/min.")
def run_cmd(mode: str, n_picks: int | None,
            hose_only: bool, include_etfs: bool,
            exclude: tuple[str, ...], warm_only: str,
            skip_train: bool, do_missed: bool, do_ab: bool | None, workers: int) -> None:
    """End-to-end: fetch -> train -> predict over the entire universe.

    Designed to be invoked from a double-click .bat. Always runs on the full
    universe (no time cap); lazy caching keeps repeat runs fast. The horizon is
    always T+2 (Vietnamese settlement); ``--picks N`` controls how many names
    are surfaced.
    """
    import time as _time

    from .data.cache import cached_symbols
    from .data.fetcher import update_many
    from .dataset import build_panel
    from .model.train import (save_latest, save_latest_low,
                              train as train_model, train_quantile)
    from .selector import select as select_symbols

    # ---- input validation ----
    if n_picks is not None and n_picks < 1:
        click.echo("ERROR: --picks must be >= 1.", err=True)
        sys.exit(2)
    # Horizon is always T+2 (Vietnamese settlement); sourced from config.
    days = int(load_config().target["exit_offset_days"])
    requested_n = int(n_picks) if n_picks else int(
        load_config().pricing.get("default_picks", 5))
    # The A/B defaults OFF for base, ON for claude/gemini (LLM weighs the verdict).
    if do_ab is None:
        do_ab = (mode != "base")
    # Normalize --exclude: split each value on commas so BAT-style single
    # comma-separated input works alongside the repeatable form, uppercase,
    # dedupe, sort for stable signature ordering.
    exclude_set: set[str] = set()
    for raw in exclude:
        for tok in str(raw).split(","):
            tok = tok.strip().upper()
            if tok:
                exclude_set.add(tok)
    exclude_list: list[str] = sorted(exclude_set)
    if exclude_list:
        click.echo(f"[note] excluding {len(exclude_list)} ticker(s): "
                   f"{', '.join(exclude_list)}")

    started = _time.time()
    click.echo(f"mode={mode}  universe=entire (no cap)  picks={requested_n}")
    click.echo(f"  horizon: T+{days}  "
               f"(T+2: sell in afternoon session only — settlement noon T+2)")
    click.echo("")

    # Auto-evaluate any predictions that are now T+2 or later. This must run
    # AFTER the data refresh so we have closes for the target date — see below.

    # Always run on the entire universe. select() clamps to the real universe
    # size, so a comfortably-oversized target means "everything".
    syms = select_symbols(target=10_000, hose_only=hose_only,
                          include_etfs=include_etfs, exclude=exclude_list)
    cached = set(cached_symbols())
    n_warm = len(set(syms) & cached)
    n_cold = len(syms) - n_warm
    label = "HOSE-only" if hose_only else "all exchanges"
    if not include_etfs:
        label += ", stocks-only"
    if exclude_list:
        label += f", excl={len(exclude_list)}"
    click.echo(f"selected {len(syms)} tickers  (warm={n_warm}, cold={n_cold})  [{label}]")
    click.echo("")

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
        # Pure offline: run on whatever parquet is already on disk.
        # Warm (current) + stale (cached but missing latest bar) both count
        # as available. Only cold tickers (no parquet) are dropped.
        # Zero API calls, guaranteed.
        available = warm + stale
        if cold:
            click.echo(f"  --warm-only=always: dropping {len(cold)} cold "
                       f"ticker(s) (no parquet); running on {len(available)} "
                       f"cached symbols ({len(warm)} current + "
                       f"{len(stale)} stale)")
        elif stale:
            click.echo(f"  --warm-only=always: running on {len(available)} "
                       f"cached symbols ({len(warm)} current + "
                       f"{len(stale)} stale); no API calls will be made")
        else:
            click.echo(f"  --warm-only=always: all {len(warm)} symbols current, "
                       "no API calls will be made")
        syms = available
        if not syms:
            click.echo("ERROR: --warm-only=always but no cached symbols; "
                       "populate the cache first by running with "
                       "--warm-only=yes (or --warm-only=no for a full "
                       "backfill).", err=True)
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
    # same set we trained on (and the hose_only / no-etfs / exclude filters
    # actually take effect at predict time, not just at fetch time). Computed
    # once and reused by both the earliest-search loop and the final mode
    # invocation. We used to set this to ``None`` when no per-session filters
    # were active, but that let delisted-but-still-cached tickers (like HTK)
    # leak into the prediction set via ``cached_symbols()``. Always pass the
    # selector-vetted list now; ``rank_today`` also intersects with
    # ``tradable_symbols()`` as a defense-in-depth check.
    pred_syms = syms

    if not skip_train:
        click.echo(f"training (horizon T+{days})...")
        # Build once with require_target=False so both heads can see all
        # rows; drop NaN per-head before fitting.
        panel = build_panel(symbols=syms, require_target=False,
                            exit_offset_days=days)
        panel_mean = panel.dropna(subset=["target"]) if not panel.empty else panel
        if panel_mean.empty:
            click.echo("no training rows. aborting.", err=True)
            sys.exit(1)
        click.echo(f"  mean head: {len(panel_mean):,} rows / "
                   f"{panel_mean['symbol'].nunique()} symbols")
        model = train_model(panel_mean)
        save_latest(model)
        click.echo("  mean model saved.")

        # Low head: same panel, dropna on target_low. Independent of
        # exit_offset_days (target_low only looks at next-day low).
        if "target_low" in panel.columns:
            panel_low = panel.dropna(subset=["target_low"])
            if not panel_low.empty:
                click.echo(f"  low head: {len(panel_low):,} rows / "
                           f"{panel_low['symbol'].nunique()} symbols")
                low_model = train_quantile(panel_low)
                save_latest_low(low_model)
                click.echo(f"  low model saved (alpha={low_model.alpha:.2f}).")
            else:
                click.echo("  low head: no rows with target_low — skipped.")

        # Missed-winners variant mean head: same panel, upweight realized
        # winners. Saved to latest_missed.pkl — the standard model is untouched.
        if do_missed:
            from .analyze.regret import missed_winner_weights
            from .model.train import save_latest_missed
            weights = missed_winner_weights(panel_mean, n=requested_n, upweight=3.0)
            n_up = int((weights > 1.0).sum())
            click.echo(f"  missed variant: upweighting {n_up:,} winner rows (x3.0)")
            save_latest_missed(train_model(panel_mean, weights=weights))
            click.echo("  missed variant saved.")
        click.echo("")

    def _maybe_ab():
        """Run the standard-vs-missed A/B and write its report (no model overwrite)."""
        if not do_ab:
            return
        click.echo("")
        click.echo("running standard-vs-missed A/B backtest (slow)...")
        from .analyze.regret import missed_winner_weights
        from .backtest.walk_forward import run as _bt_run
        click.echo(f"A/B report -> "
                   f"{_write_backtest_ab_report(_bt_run, missed_winner_weights, n=requested_n)}")

    click.echo(f"predicting (mode={mode})...")
    if mode == "base":
        from .modes import base
        picks, out = base.run(exit_offset_days=days, n_picks=requested_n,
                              symbols=pred_syms, hose_only=hose_only,
                              include_etfs=include_etfs,
                              exclude=exclude_list, variant="standard")
        click.echo("")
        click.echo("=== STANDARD model ===")
        click.echo(_format_picks(picks))
        _print_pick_warnings(picks, requested_n)
        click.echo(f"saved -> {out}")
        if do_missed:
            m_picks, m_out = base.run(exit_offset_days=days, n_picks=requested_n,
                                      symbols=pred_syms, hose_only=hose_only,
                                      include_etfs=include_etfs,
                                      exclude=exclude_list, variant="missed")
            click.echo("")
            click.echo("=== MISSED-WINNERS variant (experimental; nothing overwritten) ===")
            click.echo(_format_picks(m_picks))
            _print_pick_warnings(m_picks, requested_n)
            click.echo(f"saved -> {m_out}")
        _maybe_ab()
    elif mode == "claude":
        from .modes import claude
        result, out, tag = claude.run(exit_offset_days=days, n_picks=requested_n,
                                       symbols=pred_syms,
                                       hose_only=hose_only,
                                       include_etfs=include_etfs,
                                       exclude=exclude_list, union_missed=do_missed)
        click.echo("")
        click.echo(_format_picks(result))
        _print_pick_warnings(result, requested_n)
        if _has_explanations(result):
            click.echo("")
            click.echo(_format_picks_explained(result))
        click.echo(f"\nsaved -> {out}  (path: {tag})")
        # Claude mode emits the interactive plan; the sell reminder lands at
        # claude-finalize, not here.
        click.echo("")
        click.echo("==> NEXT (run inside Claude Code / Cowork):")
        click.echo("    1. Ask Claude to fetch every URL in the plan and fill the score table.")
        click.echo("    2. Then run:")
        click.echo(f"       python -m stockpredict.cli claude-finalize \"{out}\"")
        # Refresh the A/B verdict AFTER emitting the plan (the plan embeds the
        # PREVIOUS report, so the user isn't blocked ~10 min for the research plan).
        _maybe_ab()
    elif mode == "gemini":
        from .modes import gemini
        result, out, tag = gemini.run(exit_offset_days=days, n_picks=requested_n,
                                       symbols=pred_syms,
                                       hose_only=hose_only,
                                       include_etfs=include_etfs,
                                       exclude=exclude_list, union_missed=do_missed)
        click.echo("")
        click.echo(_format_picks(result))
        _print_pick_warnings(result, requested_n)
        if _has_explanations(result):
            click.echo("")
            click.echo(_format_picks_explained(result))
        click.echo(f"\nsaved -> {out}  (path: {tag})")
        if tag == "prompt-only":
            click.echo("")
            click.echo("==> NEXT: paste the prompt file's contents into Gemini Pro with browsing.")
        _maybe_ab()

    elapsed = (_time.time() - started) / 60.0
    click.echo(f"\nelapsed: {elapsed:.1f} min")


# ---------------------------- evaluate / track ----------------------------


@cli.command("evaluate-fills")
@click.option("--refresh-data/--no-refresh-data", default=True,
              help="Run incremental fetch on tickers with un-stamped T+0 fills first.")
def evaluate_fills_cmd(refresh_data: bool) -> None:
    """Stamp T+0 limit-fill outcomes for picks whose buy day has closed.

    Independent of T+N realized-return evaluation: this runs the moment
    the next trading day's bar lands in cache. After it succeeds, the
    Claude self-correction prompt can diagnose limit-fill calibration
    issues (fill rate, dip-quoted vs dip-actual) without waiting for
    T+N. Internally calls the same ``evaluate_pending`` — the function
    handles both stages, this command just makes the early-stage trigger
    explicit in the CLI surface.
    """
    from .data.fetcher import update_many
    from .tracking import _read, evaluate_pending

    if refresh_data:
        df = _read()
        if not df.empty:
            pending_syms = sorted(df[~df["t0_evaluated"]]["symbol"].unique().tolist())
            if pending_syms:
                click.echo(f"refreshing data for {len(pending_syms)} symbols "
                           f"with un-stamped T+0 fills...")
                update_many(pending_syms, full=False, workers=2)

    updated = evaluate_pending()
    click.echo(f"newly stamped: {len(updated)}")
    if not updated.empty:
        cols = [c for c in ["as_of", "target_date", "mode", "symbol", "rank",
                            "entry_price", "entry_limit_price", "t0_low",
                            "entry_limit_filled", "entry_slippage",
                            "evaluated", "realized_return"]
                if c in updated.columns]
        click.echo(updated[cols].to_string(index=False))


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
    click.echo(f"newly evaluated (T+0 limit-fill or T+N realized): {len(updated)}")
    if not updated.empty:
        cols = [c for c in ["as_of", "target_date", "mode", "symbol", "rank",
                            "pred_mean", "news_score", "entry_price",
                            "entry_limit_price", "entry_limit_filled",
                            "actual_exit", "realized_return"]
                if c in updated.columns]
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
    cols = [c for c in ["as_of", "target_date", "mode", "symbol", "rank",
                        "pred_mean", "news_score", "entry_price",
                        "entry_limit_price", "entry_limit_filled",
                        "realized_return", "evaluated"]
            if c in df.columns]
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
    click.echo(f"latest mean model: {'present' if m.exists() else 'missing'}  ({m})")
    lm = models_dir() / "low_latest.pkl"
    if lm.exists():
        try:
            from .model.train import RollingEmpiricalQuantileModel
            low = RollingEmpiricalQuantileModel.load(lm)
            click.echo(f"latest low model:  present  ({lm})  alpha={low.alpha:.2f}  "
                       f"lookback={low.lookback}d  trained_on={low.train_rows:,} rows")
        except Exception as e:
            click.echo(f"latest low model:  present but unreadable ({e}) "
                       f"— retrain to rebuild (entries fall back to close meanwhile)")
    else:
        click.echo(f"latest low model:  missing  ({lm}) "
                   f"— predictions fall back to entry = close")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
