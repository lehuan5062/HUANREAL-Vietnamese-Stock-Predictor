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
    """Vietnamese rebound stock predictor."""


def _format_picks(picks) -> str:
    """One-line-per-pick view with entry / target / stop / fees / net (VND)
    sized for the configured position. Falls back to full dataframe if
    pricing columns aren't present."""
    if picks is None or len(picks) == 0:
        return "(no picks)"
    show_cols = [c for c in [
        "symbol", "close_vnd", "target_vnd",
        "fees_round_trip_vnd", "net_reward_vnd", "net_loss_vnd",
        "rr_ratio", "below_breakeven",
        "score", "pred_days", "pred_profit", "pred_recovery_prob",
        "below_recovery_bar", "pred_mean", "news_score", "adjusted",
    ] if c in picks.columns]
    if not show_cols:
        return picks.to_string(index=False)
    fmt = picks[show_cols].copy()
    money_cols = [
        "close_vnd", "target_vnd", "stop_vnd",
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

        is_rebound = "score" in r and pd.notna(r.get("score"))
        buy_col = "close_vnd" if is_rebound else "entry_vnd"
        if buy_col in r and pd.notna(r.get(buy_col)):
            entry = int(r[buy_col]); tgt = int(r["target_vnd"])
            fees = int(r.get("fees_round_trip_vnd", 0))
            net = int(r.get("net_reward_vnd", 0))
            if is_rebound:
                # Rebound: flexible exit, no stop. Show buy / target / hold.
                below = bool(r.get("below_recovery_bar", False))
                verdict = "BELOW RECOVERY BAR (weak)" if below else "OK"
                hold = r.get("pred_days")
                hold_s = f"{int(round(hold))}d" if pd.notna(hold) else "?"
                parts.append(f"  Trade: buy @ {entry:,} VND  |  target {tgt:,}  |  hold ≈ {hold_s} (sell at target)")
                parts.append(f"  P&L per share (after ACBS fees {fees:,}): net {net:+,}  -> {verdict}")
            else:
                stop = int(r["stop_vnd"]) if pd.notna(r.get("stop_vnd")) else 0
                rr = r.get("rr_ratio", float("nan"))
                below = bool(r.get("below_breakeven", False))
                verdict = "BELOW-BAR (weak edge)" if below else "OK"
                close_v = r.get("close_vnd", None)
                dip_pct = r.get("entry_limit_pct", None)
                if close_v is not None and pd.notna(close_v) and dip_pct is not None and pd.notna(dip_pct) and float(dip_pct) < 0:
                    parts.append(
                        f"  Trade: LIMIT-buy @ {entry:,} VND "
                        f"(close {int(close_v):,}, dip {float(dip_pct):+.2%})  "
                        f"|  target {tgt:,}  |  stop {stop:,}"
                    )
                else:
                    parts.append(f"  Trade: buy @ {entry:,} VND  |  target {tgt:,}  |  stop {stop:,}")
                parts.append(f"  P&L per share (after ACBS fees {fees:,}): net {net:+,}  rr {rr:.2f}  -> {verdict}")

        ns = r.get("news_score", 0)
        adj = r.get("adjusted", None)
        if is_rebound:
            line = (f"  Signal: score={r.get('score'):.4f}  "
                    f"N≈{r.get('pred_days', float('nan')):.0f}d  "
                    f"P≈{r.get('pred_profit', float('nan')):+.3f}")
            prob = r.get("pred_recovery_prob")
            # LLM-only picks carry no statistical recovery probability — omit.
            if prob is not None and pd.notna(prob):
                line += f"  recovery_prob={prob:.0%}"
            if pd.notna(ns):
                line += f"  news={int(ns):+d}"
            if adj is not None and pd.notna(adj):
                line += f"  adjusted={adj:.4f}"
            parts.append(line)
        else:
            ml_pred = r.get("pred_mean", None)
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
    # Rebound uses below_recovery_bar; legacy uses below_breakeven.
    bar_col = ("below_recovery_bar" if picks is not None
               and "below_recovery_bar" in picks.columns else "below_breakeven")
    if picks is None or n == 0 or bar_col not in picks.columns:
        return
    k = int(picks[bar_col].fillna(True).astype(bool).sum())
    if k > 0:
        label = ("below the recovery bar (weak — low bounce probability)"
                 if bar_col == "below_recovery_bar"
                 else "below the break-even bar (weak edge — forecast under round-trip cost)")
        click.echo("")
        click.echo(f"==> QUALITY: {k} of {n} pick(s) are {label}. They're "
                   f"shown to honor --picks; treat them with extra caution.")


def _print_sell_reminder(picks, *, mode_label="") -> None:
    """Surface the flexible-exit plan for the returned picks. The rebound trade
    holds until the profit target (no fixed sell day), so this shows the target
    + expected hold per pick and leaves the sell timing to the user."""
    if picks is None or len(picks) == 0:
        return
    click.echo("")
    click.echo("==> EXIT PLAN (flexible — sell when the price reaches the target):")
    for _, r in picks.reset_index(drop=True).iterrows():
        tgt = r.get("target_vnd")
        hold = r.get("pred_days")
        tgt_s = f"{int(tgt):,}" if pd.notna(tgt) else "?"
        hold_s = f", expected ≈ {int(round(hold))}d" if pd.notna(hold) else ""
        click.echo(f"    {r['symbol']}: target {tgt_s} VND{hold_s}")
    if mode_label in ("claude", "gemini"):
        click.echo(f"    {mode_label.title()}: there is NO fixed sell day — do not "
                   f"schedule a hard sell alarm. Offer an optional check-in only if asked.")


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
def train_cmd(start: str | None, end: str | None) -> None:
    """Build the panel and fit the rebound recovery estimator
    (``models/recovery_latest.pkl``)."""
    from .dataset import build_panel
    from .model.train import save_latest_recovery, train_recovery

    click.echo("building panel...")
    panel = build_panel(start=start, end=end, require_target=False)
    click.echo(f"panel: {len(panel):,} rows across {panel['symbol'].nunique()} symbols")
    if panel.empty:
        click.echo("no data — run update-data first.", err=True)
        sys.exit(1)

    rec = train_recovery(panel)
    rpath = save_latest_recovery(rec)
    click.echo(f"  recovery head: {rec.train_rows:,} downtrend rows, "
               f"{len(rec.buckets)} state buckets; saved -> {rpath}")


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


@cli.command("compare-modes")
@click.option("--window", type=int, default=90, show_default=True,
              help="Look-back window in days to pool over.")
@click.option("--date", "as_of", default=None,
              help="YYYY-MM-DD: also show this single day's per-mode breakdown "
                   "as context (still pools the verdict over the window).")
@click.option("--modes", default=None,
              help="Comma-separated subset to compare (e.g. base,claude,claude_llm). "
                   "Default: every mode found in the ledger.")
def compare_modes_cmd(window: int, as_of: str | None, modes: str | None) -> None:
    """Head-to-head realized performance of the prediction methods (base /
    hybrid / LLM-only / gemini), pooled over comparable runs — same day AND
    same params (picks/horizon/hose-only/etfs/exclude). Advisory: tells you
    which method to PREFER, not a knob to tune."""
    from .analyze import mode_compare
    mode_list = ([m.strip() for m in modes.split(",") if m.strip()]
                 if modes else None)
    result = mode_compare.compare_modes(window_days=window, as_of=as_of,
                                       modes=mode_list)
    md = mode_compare.format_report(result)
    click.echo(md)
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    out = reports_dir() / f"mode_comparison_{today}.md"
    out.write_text(md, encoding="utf-8")
    click.echo(f"\nreport -> {out}")


@cli.command("compare-picks")
@click.option("--date", "as_of", required=True, help="YYYY-MM-DD: compare picks reports for this day")
def compare_picks_cmd(as_of: str) -> None:
    """Compare the actual picks selected by different modes on the same day
    (ignores resolution status). Reads picks JSON files directly."""
    from .analyze import mode_compare

    result = mode_compare.compare_picks_same_day(as_of)
    md = mode_compare.format_picks_comparison(result)
    click.echo(md)


@cli.command("compare-picks-accountability")
@click.option("--date", "as_of", required=True, help="YYYY-MM-DD: mode accountability for this day")
def compare_picks_accountability_cmd(as_of: str) -> None:
    """Show mode accountability: for each resolved pick, which modes selected it
    and which avoided it. Reveals whether errors are shared (all modes) or
    mode-specific (only LLM or only base)."""
    from .analyze import mode_compare

    result = mode_compare.mode_accountability(as_of)
    md = mode_compare.format_mode_accountability(result)
    click.echo(md)


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
    # Auto-detect the LLM-only path: a `claude_llm_plan_*` file, or a meta
    # sidecar tagged "method": "llm_only".
    _p = Path(plan_path)
    _is_llm = _p.name.startswith("claude_llm_plan_")
    if not _is_llm:
        _meta = _p.with_suffix(".meta.json")
        if _meta.exists():
            try:
                _is_llm = json.loads(_meta.read_text(encoding="utf-8")).get("method") == "llm_only"
            except Exception:
                _is_llm = False
    picks, out = claude.finalize_llm(plan_path) if _is_llm else claude.finalize(plan_path)
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
        _print_sell_reminder(picks, mode_label="claude")
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
        _print_sell_reminder(picks, mode_label="gemini")
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
              help="Use the existing models/recovery_latest.pkl instead of retraining.")
@click.option("--llm-only", is_flag=True,
              help="LLM-ONLY prediction (claude mode only): no model ranking. The "
                   "whole mechanically-filtered downtrend universe is handed to "
                   "Claude, which picks, ranks and prices every name itself. Emits "
                   "a `claude_llm_plan_<date>.md` instead of `claude_news_plan_*`.")
def run_cmd(mode: str, n_picks: int | None,
            hose_only: bool, include_etfs: bool,
            exclude: tuple[str, ...], warm_only: str,
            skip_train: bool, llm_only: bool) -> None:
    """End-to-end: fetch -> train -> predict over the entire universe.

    Designed to be invoked from a double-click .bat. Always runs on the full
    universe (no time cap); lazy caching keeps repeat runs fast. The rebound
    trade holds until recovery (flexible exit); ``--picks N`` controls how many
    names are surfaced.
    """
    import time as _time

    from .data.cache import cached_symbols
    from .data.fetcher import update_many
    from .dataset import build_panel
    from .model.train import save_latest_recovery, train_recovery
    from .selector import select as select_symbols

    # ---- input validation ----
    if n_picks is not None and n_picks < 1:
        click.echo("ERROR: --picks must be >= 1.", err=True)
        sys.exit(2)
    if llm_only and mode != "claude":
        click.echo("ERROR: --llm-only is only valid with --mode claude.", err=True)
        sys.exit(2)
    requested_n = int(n_picks) if n_picks else int(
        load_config().pricing.get("default_picks", 5))
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
    click.echo("  rebound: buy at close, hold until the profit target "
               "(flexible exit — no fixed sell day)")
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
    from .data.fetcher import audit_cache, quiet_vnstock_logger, _VALID_SOURCES
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
                   f"{len(syms)} symbols (single worker, rate-limited)")
    click.echo("")

    if warm_mode == "always":
        # Skip the network entirely. Use cached parquets as-is.
        click.echo("skipping data refresh (--warm-only=always)")
        results = {s: 0 for s in syms}
        ok = len(results)
        err = 0
    else:
        click.echo(f"updating data ({len(_VALID_SOURCES)} workers, shared queue "
                   f"across sources: {', '.join(_VALID_SOURCES)})...")
        # warm_mode == "yes" → update_many internally audits and only
        # fetches stale + cold (lazy).
        # warm_mode == "no" → full=True forces re-fetch of every symbol.
        results = update_many(syms, full=(warm_mode == "no"))
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

    if llm_only:
        click.echo("LLM-only: skipping model training (no model used).")
        click.echo("")
    elif not skip_train:
        click.echo("training rebound recovery head...")
        panel = build_panel(symbols=syms, require_target=False)
        if panel.empty:
            click.echo("no training rows. aborting.", err=True)
            sys.exit(1)
        rec = train_recovery(panel)
        save_latest_recovery(rec)
        click.echo(f"  recovery head: {rec.train_rows:,} downtrend rows / "
                   f"{len(rec.buckets)} state buckets saved.")
        click.echo("")

    click.echo(f"predicting (mode={mode})...")
    if mode == "base":
        from .modes import base
        picks, out = base.run(n_picks=requested_n,
                              symbols=pred_syms, hose_only=hose_only,
                              include_etfs=include_etfs,
                              exclude=exclude_list)
        click.echo("")
        click.echo(_format_picks(picks))
        _print_pick_warnings(picks, requested_n)
        click.echo(f"saved -> {out}")
    elif mode == "claude":
        from .modes import claude
        result, out, tag = claude.run(n_picks=requested_n,
                                       symbols=pred_syms,
                                       hose_only=hose_only,
                                       include_etfs=include_etfs,
                                       exclude=exclude_list,
                                       llm_only=llm_only)
        click.echo("")
        if not llm_only:
            click.echo(_format_picks(result))
            _print_pick_warnings(result, requested_n)
            if _has_explanations(result):
                click.echo("")
                click.echo(_format_picks_explained(result))
        else:
            click.echo(f"LLM-only universe: {len(result)} eligible name(s) for "
                       f"Claude to pick from (target {requested_n}).")
        click.echo(f"\nsaved -> {out}  (path: {tag})")
        # Claude mode emits the interactive plan; the sell reminder lands at
        # claude-finalize, not here.
        click.echo("")
        click.echo("==> NEXT (run inside Claude Code / Cowork):")
        if llm_only:
            click.echo("    1. Ask Claude to research the universe, choose & price the picks,")
            click.echo("       and fill the Results table.")
        else:
            click.echo("    1. Ask Claude to fetch every URL in the plan and fill the score table.")
        click.echo("    2. Then run:")
        click.echo(f"       python -m stockpredict.cli claude-finalize \"{out}\"")
    elif mode == "gemini":
        from .modes import gemini
        result, out, tag = gemini.run(n_picks=requested_n,
                                       symbols=pred_syms,
                                       hose_only=hose_only,
                                       include_etfs=include_etfs,
                                       exclude=exclude_list)
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
                update_many(pending_syms, full=False)

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
                update_many(pending_syms, full=False)

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
    rm = models_dir() / "recovery_latest.pkl"
    if rm.exists():
        try:
            from .model.train import RecoveryKMModel
            rec = RecoveryKMModel.load(rm)
            click.echo(f"recovery model:  present  ({rm})  "
                       f"{rec.train_rows:,} downtrend rows  "
                       f"{len(rec.buckets)} state buckets  "
                       f"{len(getattr(rec, 'ticker_stats', {}))} tickers")
        except Exception as e:
            click.echo(f"recovery model:  present but unreadable ({e}) — retrain.")
    else:
        click.echo(f"recovery model:  missing  ({rm}) — run `train` (or `run`).")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
