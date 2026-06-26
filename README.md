# HUANREAL Vietnamese T+2 Stock Predictor

A swing-trade screener for the Vietnamese stock market. Buy on day **T**, hold
**2 trading days**, sell on the **T+2** afternoon session (Vietnamese
settlement happens at noon on T+2, so that's the earliest legal sell). The
program ranks the most liquid HOSE / HNX / UPCOM tickers by predicted T+2
forward return, returns **exactly the number of picks you ask for** (`--picks
N`, top by score), and tracks every prediction so it can grade itself later.

## Two ways to run

### A. Double-click a .bat file (base / gemini)

| File | What it does |
| ---- | ------------ |
| [`predict_base.bat`](predict_base.bat) | Pure ML + technical filter. No news. |
| [`predict_gemini.bat`](predict_gemini.bat) | ML + writes a prompt file you paste into Gemini Chat (web, with browsing). Opens the prompt in Notepad automatically. |
| [`evaluate.bat`](evaluate.bat) | Refreshes data and grades any past predictions whose T+2 has now passed. |

Each predict .bat asks for picks / hose-only / etfs / exclude / warm-only and runs the entire universe.

### B. Claude mode — paste a prompt into Claude Desktop

Claude mode is **driven from inside Claude Desktop** (Claude Code or Cowork),
not from a .bat. The news research uses Claude's `WebSearch` / `WebFetch`
tools, which only exist inside Claude.

**To run Claude mode:**

1. Open Claude Code or Cowork in Claude Desktop.
2. Paste the contents of [`claude_prompt.md`](claude_prompt.md) into the chat.
3. Claude will ask you for the number of `picks` and the **prediction
   method**, then drive the entire pipeline.

The prompt is self-contained and tells Claude everything it needs to know
about the project layout, tools, scoring rubric, the seven-dimension
*reference* (with the explicit instruction that dimensions are per-ticker
emergent, not a fixed checklist), and the ACBS fee model.

**Two prediction methods (Claude mode):**

| method | who selects the picks | who sets the prices | report files |
| ------ | --------------------- | ------------------- | ------------ |
| **ML/LLM hybrid** (default) | the ML model ranks the universe by `pred_mean` and returns the top N; Claude does *news research* on those N and nudges the ranking | mechanical (low-head dip + ATR) | `claude_news_plan_*` → `picks_claude_*` |
| **LLM-only** (`--llm-only`) | **Claude** picks the N names from the *whole* mechanically-filtered universe (unranked, no `pred_mean`) on its own research, ranked by its own `conviction` | **Claude** sets `entry` / `target` / `stop` itself | `claude_llm_plan_*` → `picks_claude_llm_*` |

LLM-only uses **no ML model at all** — no training, no `latest.pkl`, no
`pred_mean`. It only reuses the mechanical universe filters (liquidity /
tradable / ceiling-lock / corporate-action). The distinct report names make the
method obvious at a glance; LLM-only picks are recorded in the ledger under
`mode="claude_llm"` so the two methods stay separable.

### C. CLI (advanced)

```bash
# Standard daily run (entire universe, T+2, default pick count)
.venv\Scripts\python -m stockpredict.cli run --mode base

# Ask for exactly 8 picks
.venv\Scripts\python -m stockpredict.cli run --picks 8 --mode base

# Other commands
.venv\Scripts\python -m stockpredict.cli evaluate
.venv\Scripts\python -m stockpredict.cli track --limit 20
.venv\Scripts\python -m stockpredict.cli backtest --start 2022-01-01

# Every base run already writes BOTH a standard and a _missed pick report
# (the missed-winners variant is on by default). To compare on win rate:
.venv\Scripts\python -m stockpredict.cli backtest-ab    # writes reports/backtest_ab_<date>.md (overwrites nothing)
.venv\Scripts\python -m stockpredict.cli run --ab       # ...or fold the A/B into a run

# Missed-winners analysis: which realized top-N did the model not surface, and why
.venv\Scripts\python -m stockpredict.cli regret --window 90

# Head-to-head: which prediction METHOD picked better (base vs hybrid vs LLM-only),
# pooled over comparable same-day same-param runs. Advisory — writes reports/mode_comparison_<date>.md
.venv\Scripts\python -m stockpredict.cli compare-modes --window 90
```

**Note**: `--mode claude` at the CLI emits the markdown research plan (the
same plan the prompt file drives). The primary Claude path is the prompt file
(option B above); the in-session Claude fills the plan and runs
`claude-finalize`.

### Run command flags

| flag | meaning | default |
| ---- | ------- | ------- |
| `--picks N`, `-n N` | How many picks to surface. The program always returns **exactly N** — it ranks the whole scored universe by predicted return and keeps the top N, so the difficulty (the implicit edge cutoff) floats to whatever admits exactly N. Picks below the break-even quality bar are still returned but flagged `below_breakeven`, with a count in the `==> QUALITY` warning. If the eligible universe is smaller than N (heavy `--exclude` / `--hose-only` / tiny cache), fewer are returned and a `==> SHORTFALL` warning prints. Horizon is always T+2. | `pricing.default_picks` (1) |
| `--hose-only` | Restrict the universe to HOSE-listed tickers. Refreshes the universe via VCI to try to get exchange info; falls back to ~43 curated HOSE bluechips (VN30 + HOSE mid-caps) if the data source doesn't return `exchange`. | `False` |
| `--include-etfs` | Include HOSE ETFs (e.g. `E1VFVN30`, `FUEVFVND`) in the pickable universe. Off by default — ETFs are classified as a separate security type and excluded unless this is set. | `False` |
| `--mode {base,claude,gemini}` | which pipeline | `base` |
| `--llm-only` | **claude mode only.** No ML model is used to select or rank — the whole mechanically-filtered universe (no `pred_mean`) is handed to Claude, which picks, ranks (by its own `conviction`) and prices every name itself. Skips ML training; emits `claude_llm_plan_<date>.md` and finalizes to `picks_claude_llm_<date>_…json`. Forces `--no-missed --no-ab` (no ML to union / backtest). | off |
| `--skip-train` | reuse cached `models/latest.pkl` instead of retraining | off |
| `--missed` / `--no-missed` | **On by default.** Involves the missed-winners variant. **base**: writes a *second* `_missed` pick report alongside the standard one. **claude/gemini**: UNIONs the variant's top picks into the candidates the LLM researches (each flagged `also-missed` / `missed-only` in the plan), so the LLM weighs both rankings and keeps the top N. Nothing is overwritten. `--no-missed` for standard only. | `--missed` |
| `--ab` / `--no-ab` | After predicting, run the standard-vs-missed walk-forward A/B and write `reports/backtest_ab_<date>.md` (advisory; overwrites no model). The LLM modes embed the verdict so they weigh the winner. Slow (~10 min). **Default: off for base, on for claude/gemini.** | off (base) / on (claude,gemini) |

## Universe coverage

Every run covers the **entire universe** (all of HOSE + HNX + UPCOM, ~1,765
tickers today and growing) with no time cap. Expect ~30 minutes the first time
(the fetcher bypasses vnstock's 20/min guest quota and self-throttles to
`api_per_min`, default 60 — see Caveats), much faster on subsequent runs since
cached, up-to-date tickers cost 0 API calls. The smart
lazy fetch (`--warm-only yes`, the default) only hits the network for stale and
cold tickers, so the second run of the day is near-instant.

## Pre-flight cache audit + `--warm-only`

Every run now prints a cache audit before any fetch attempt, so you see
exactly what's about to hit the network:

```
cache audit (expected bar = 2026-05-05):
   1304 cached and current  ->  no API call
     12 cached but stale    ->  incremental fetch
    220 missing             ->  full-history fetch
```

If the previous full run hit rate limits and left some tickers cold,
the audit shows it immediately — you can re-run with same params to
backfill, run with `--warm-only` to ignore them, or wait it out.

`--warm-only` is **tri-state**, default `yes`:

| `--warm-only` | warm cached | stale cached | cold (no parquet) | use when |
| ------------- | ----------- | ------------ | ----------------- | -------- |
| **`yes`** (default) — smart lazy fetch | skip | **fetch new bar** | **fetch full history** | every-day usage |
| **`always`** — pure offline | keep | keep | drop | guaranteed zero API calls; rate-limit panic; airplane / weekend (use whatever parquet is on disk) |
| **`no`** — force full re-fetch | refetch full | refetch full | refetch full | backfill, broker corrections, suspect cache rot |

How it plays out across the trading week with default `yes`:

- During the day → cache is current → runs are instant
- After 15:00 close → every ticker is stale → next run fetches that
  one new bar per ticker (~30s of API time) → all warm → subsequent
  runs that day are instant
- Saturday / Sunday / holiday → expected bar = last trading day → if
  you fetched after Friday's close, every run is instant

CLI examples:

```
# Daily — smart lazy fetch (default)
.venv\Scripts\python -m stockpredict.cli run --mode base

# Pure offline — predict on whatever's cached, no network at all
.venv\Scripts\python -m stockpredict.cli run --warm-only always --mode base

# Force a full re-fetch (slow, rate-limited)
.venv\Scripts\python -m stockpredict.cli run --warm-only no --mode base
```

The `.bat` prompt accepts: `y` / `a` / `n` (default `y`):
*"Warm-only? [y]es lazy fetch / [a]lways offline / [n]o full refetch [y]:"*

Why "stale OK"? A one-day-stale feature row (computed on the previous
trading day's close) is still ~99% the same as a fresh one — RSI,
momentum, ATR, etc. shift only a small amount per day. The model trains
and predicts on what's in the cache; you just won't see the very latest
bar's effect.

Typical workflow:

- **Default daily runs**: just run it. The first run after each trading
  close fetches that one new bar per ticker (~15-30s of API time);
  subsequent runs that day are instant.
- **First-ever full-universe seed**: just run it (the universe is always
  full). The smart cache then keeps it lazy for everything that follows.
- **Backfill / corrections / sanity refresh**: `--no-warm-only` to
  force a full re-fetch. Slow — schedule it overnight.

```
# Daily — smart lazy fetch by default, fetches only the new bar
.venv\Scripts\python -m stockpredict.cli run --mode base

# Backfill or force-refresh corrections
.venv\Scripts\python -m stockpredict.cli run --no-warm-only --mode base
```

The CLI prints a clear summary so you know exactly what happened:

```
cache audit (expected bar = 2026-05-06):
   1304 cached and current  ->  no API call
     12 cached but stale    ->  incremental fetch
    220 missing             ->  full-history fetch
  --warm-only=always: dropping 220 cold ticker(s) (no parquet); running on 1316 cached symbols (1304 current + 12 stale)

skipping data refresh (--warm-only=always)
```

If the cache is **completely empty** (brand-new install), `--warm-only always`
errors out — there's nothing to predict on. Run once normally to seed
the cache, then `--warm-only always` works forever after.

## Smart cache freshness — no API calls outside trading hours

The fetcher knows when the broker has published a new end-of-day bar.
If your cache is already current, it skips the API entirely (no
ThreadPoolExecutor spin-up, no progress bar) and the run prints:

```
updating data (workers=2)...
  cache is current through 2026-05-05 (latest expected bar) — no fetch needed
```

The "latest expected bar" is computed from the wall clock + the cached
trading-day calendar:

| current local time | expected most-recent bar |
| ------------------ | ------------------------ |
| Trading day, ≥ 15:00 (close + 15-min buffer) | **today** |
| Trading day, < 15:00 | the **previous** trading day |
| Saturday / Sunday | the most recent prior trading day |
| Vietnamese fixed-date holiday (Apr 30, May 1, Sep 2, Jan 1) | the prior trading day |
| Tết / Hung Kings (lunar holidays) | falls back to projection — at worst one wasted fetch returning nothing |

So you can re-run the .bat repeatedly on a Saturday and only pay for
the predict step. Same for evenings, weekends, and holiday periods.
Implementation: [`tracking.latest_expected_bar_date`](src/stockpredict/tracking.py)
+ [`data/fetcher.update_many`](src/stockpredict/data/fetcher.py) fast-path.

If you ever need to force a refresh (e.g. broker back-fills a
correction), pass `--full` to `update-data`:
`.venv\Scripts\python -m stockpredict.cli update-data --full`.

### Stuck-ticker watermarks

Some tickers in HOSE / HNX / UPCOM are delisted, halted, or simply
absent from the data feed — the broker cheerfully returns *empty* for
them, the cache file never gets a newer date stamped, and on every
subsequent run those tickers re-classify as "stale" and burn API
budget retrying for nothing. The fix is a per-symbol watermark:

- After every fetch attempt — successful **or** empty — we stamp
  `cache/watermarks/{SYMBOL}.txt` with `latest_expected_bar_date()`
  for that ticker (single ISO date, ~10 bytes per file).
- `audit_cache` and `update_symbol` both consult it: a ticker whose
  watermark is `>= expected_bar` counts as **warm** even if its
  cached parquet is older.
- When the next trading day closes, `expected_bar` advances, the
  watermark falls behind, and the ticker gets exactly one fresh
  attempt before being re-stamped.

Net: stuck tickers stop retrying every run, but you never miss new
data when it actually publishes. Implementation:
[`data/cache.get_watermark` / `set_watermark`](src/stockpredict/data/cache.py)
plumbed into [`data/fetcher.audit_cache`](src/stockpredict/data/fetcher.py).

### Atomic writes — Ctrl+C never corrupts the cache

Every parquet write goes through `os.replace(tmp, target)`:

1. Serialize to `<SYMBOL>.parquet.tmp`
2. Atomically rename onto `<SYMBOL>.parquet`

`os.replace` is atomic on both POSIX and Windows since Python 3.3.
So if you Ctrl+C mid-run (or the OS shuts down), the on-disk file is
either the previous complete version or the new complete version —
never a partial / corrupt parquet that fails to read on the next run.
Same atomicity for the watermark files. The 70 tickers you fetched
before hitting Ctrl+C *are* persisted; the next run picks up from
there instead of starting over.

## Trading-day calendar (T+2 math)

T+2 is computed using the **actual Vietnamese trading-day calendar** built
from cached OHLCV data — weekends AND every Vietnamese holiday the market
historically closed for (Tết, Reunification Day, Labor Day, National Day,
plus ad-hoc closures) are skipped.

| computation | weekends | Vietnamese holidays |
| ----------- | -------- | ------------------- |
| ML training target (`shift(-2)` on OHLCV index) | excluded | excluded |
| `target_date` in the predictions ledger ([`tracking._next_trading_offset`](src/stockpredict/tracking.py)) | excluded | excluded |
| Realized-return evaluation (`evaluate_pending`) | excluded | excluded |

Example — for T = 2025-04-29 (Tue, last trading day before the
Reunification + Labor Day cluster), T+2 lands on **2025-05-06**, not
2025-05-01 (the naive Mon-Fri arithmetic answer). Verified end-to-end
against the real cache in
[`tests/test_trading_calendar.py`](tests/test_trading_calendar.py).

## Exact-N picks (`--picks N`)

The horizon is always **T+2** (Vietnamese settlement). Instead of a horizon
search, you tell the program **how many picks** you want and it returns
exactly that many:

1. Trains the T+2 model (once), predicts the whole liquid / tradable /
   glitch-filtered universe (also dropping names with an unadjusted
   corporate-action gap — see below), and ranks every name by `pred_mean`.
2. Keeps the **top N** by predicted return. Taking the top N is the same as
   auto-tuning the edge gate to admit exactly N — the cutoff floats to the
   Nth pick's score.
3. Flags any of those N that fall below the **break-even quality bar**
   (`pred_mean < min_edge_over_cost × breakeven_pct`) with
   `below_breakeven=True`. They're still returned (to honor the count) but a
   `==> QUALITY: K of N pick(s) are below the break-even bar` warning prints.
4. If the eligible universe is smaller than N (heavy `--exclude` /
   `--hose-only` / tiny cache), it returns all available and prints a
   `==> SHORTFALL` warning. Retraining can't manufacture more candidates, so
   there's no search loop.

`--picks N` (or `-n N`) overrides the default in `pricing.default_picks`
(config.yaml, default 1).

Typical output:

```
mode=base  universe=entire (no cap)  picks=5
  horizon: T+2  (T+2: sell in afternoon session only — settlement noon T+2)

symbol  close_vnd  entry_vnd  target_vnd  stop_vnd  net_reward_vnd  rr_ratio  below_breakeven  pred_mean
   VVS     80,900     78,973      96,054    66,162         +16,696      1.27            False   0.028708
   PAN     22,750     22,464      25,123    20,470          +2,555      1.22            False   0.014885
   ...
saved -> reports/picks_<date>_base_d2_VVS-PAN-KLB-NAB-VJC.json
```

The picks JSON records `selection: "top_n"`, `requested_picks`, `n_picks`,
and `n_below_breakeven`. The filename suffix lists the returned tickers.

## Reading the output

After a run, look in two places:

1. **`reports/picks_<mode>_<date>.json`** — the final top-K with all pricing.
2. **The ranked candidate table printed in the console** — compact view.

### Trade-ready columns (the ones you need)

For each pick, the program produces actual **VND** prices on a **per-share**
basis. You size the position yourself:

| column | meaning |
| ------ | ------- |
| `close_vnd` | today's close, for reference |
| `entry_vnd` | **limit-buy price to place** — the predicted next-day dip, `close × (1 + pred_low)` where `pred_low` is the per-ticker α-quantile of recent next-day-low returns (clipped never above the close). The α is **conviction-coupled** (`pred_low_alpha`): a strong pick gets a shallow dip (fills easily), a weak / below-breakeven pick gets a deep dip (fills only at a bargain). Falls back to the close when no low head is present. |
| `entry_limit_pct` | the predicted dip vs close (≤ 0) baked into `entry_vnd` |
| `target_vnd` | take-profit to sell at on the exit day T+2 — **ATR-scaled**: `entry + target_atr_mult × ATR(14)`. (Was `entry × (1 + pred_mean)`; that put reward on the scale of a ~0.1% forecast against a ~4% ATR stop, so `rr_ratio` was structurally ~0.2.) `pred_mean` now drives only ranking and the `below_breakeven` quality flag. |
| `target_low_vnd` / `target_high_vnd` | ATR target ± 1 ensemble-std band |
| `stop_vnd` | stop-loss (`entry − stop_atr_mult × ATR(14)`) |
| `gross_reward_vnd` | `target − entry` per share (before fees) |
| `max_loss_vnd` | `entry − stop` per share (before fees) |
| `fees_round_trip_vnd` | ACBS commission + VAT + PIT for this round trip, per share |
| `net_reward_vnd` | **predicted per-share profit after all fees** — the headline number |
| `net_loss_vnd` | worst-case per-share loss if stopped out (max_loss + fees) |
| `rr_ratio` | `net_reward / net_loss` — with the ATR-scaled target this is ≈ `target_atr_mult / stop_atr_mult`, a sanity number, not the selector |
| `breakeven_pct` | what % the price needs to move just to cover fees (~0.43% at ACBS) |
| `below_breakeven` | `True` when the pick does **not** clear the quality bar (`pred_mean < min_edge_over_cost × breakeven_pct`, OR `net_reward ≤ 0`, OR rr invalid). Informational only — selection is exactly-N by `pred_mean`, so a `below_breakeven` pick is still returned but flagged weak (and counted in the `==> QUALITY` warning). Names whose `\|pred_mean\|` exceeds `pricing.max_abs_pred_mean` are dropped as data glitches before ranking. |
| `suggested_max_units` | **Advisory liquidity cap** — `floor(pricing.max_participation_pct% × adv_vnd_20 / entry_vnd)`. Stay at or below this to avoid dominating the tape on entry/exit. Purely informational; never feeds selection. Null when `adv_vnd_20` is missing or the cap is disabled (`max_participation_pct: 0`). |

### News-adjusted entry/target (claude / gemini)

The mechanical `entry_vnd` is a per-ticker dip limit that the news layer can
*not* move — on a news-driven melt-up the dip never comes and the limit never
fills. So Claude and Gemini can optionally quote their own entry/target informed
by the catalyst, which the pipeline turns into a **parallel** set of `adj_*`
columns alongside the mechanical ones:

| column | meaning |
| ------ | ------- |
| `adj_entry_vnd` / `adj_target_vnd` | LLM-supplied entry and exit (in VND). `adj_entry_vnd` is **not** clipped at the close — a strong catalyst can quote an entry above today's close to guarantee a fill. |
| `adj_stop_vnd`, `adj_gross_reward_vnd`, `adj_max_loss_vnd`, `adj_fees_round_trip_vnd`, `adj_net_reward_vnd`, `adj_net_loss_vnd`, `adj_rr_ratio`, `adj_breakeven_pct` | The full risk-reward stack recomputed against `adj_entry_vnd` / `adj_target_vnd`, so you can compare the news-adjusted trade against the mechanical one side by side. |

If the LLM didn't override a row, its `adj_*` columns mirror the mechanical
ones — they're always populated. `base` mode never sets adjusted values.

### Diagnostic columns (for sanity-checking the signal)

| column | meaning | how to read it |
| ------ | ------- | -------------- |
| `pred_mean` | predicted T+2 return | `+0.0017` = +0.17% before costs |
| `pred_std` | dispersion across 5 model seeds | high std relative to mean = low confidence |
| `rsi_14` | 14-day RSI | < 30 = oversold, > 70 = overbought |
| `mom_5`, `mom_20` | log-momentum over 5 / 20 trading days | – |
| `vol_z_20` | today's volume vs 20-day mean (σ) | > +2 = unusual flow |
| `adv_vnd_20` | 20-day avg daily traded value | bigger = easier to exit |
| `news_score` (claude only) | −1 / 0 / +1 / `DROP` | from Claude's news research |
| `adjusted` (claude only) | `pred_mean × (1 + 0.05 × news_score)` | re-ranking score |

### How fees are calculated (ACBS default)

[`config.yaml`](config.yaml) → `broker:` section. Defaults:

```
commission_pct: 0.15       # per side
vat_pct: 10                # VAT on commission
pit_pct: 0.10              # PIT on the SELL transaction value
min_fee_vnd: 0
```

Round-trip cost ≈ **0.43%** of trade value:
```
buy_fee  = trade_value × 0.15% × 1.10           = 0.165% of trade
sell_fee = trade_value × 0.15% × 1.10 + 0.10%   = 0.265% of trade
total    ≈ 0.43%
```

If you have a different broker, edit the percentages.

### Worked example (today's run, per share)

```
symbol close_vnd entry_vnd target_vnd stop_vnd net_reward rr   below_breakeven pred_mean
VVS    80,900    78,973    96,054     66,162   +16,696    1.27 False           0.0287
PAN    22,750    22,464    25,123     20,470   +2,555     1.22 False           0.0149
KLB    15,500    15,339    16,061     14,797   +654       1.07 False           0.0071
```

Reading this:
- **Buy PAN at 22,464 VND/share (a dip limit below the 22,750 close), target
  25,123, stop 20,470.**
- All figures are per share — you size the position yourself.
- With the ATR-scaled target, `rr_ratio` sits near `target_atr_mult /
  stop_atr_mult` (~1.3) by construction, so the **ranking** (`pred_mean`) is
  what decides the order, and the top N you asked for are returned.
- `below_breakeven=False` means the forecast clears the round-trip-cost bar.
  When it's `True`, the pick is still returned (to honor `--picks`) but flagged
  weak and counted in the `==> QUALITY` warning — treat those with caution.

### Tuning

- All P&L figures are per share — you size the position yourself. Fees scale
  linearly with trade value, so the % cost stays at 0.43% regardless of size.
- Want more / fewer picks? Use `--picks N` (or set `pricing.default_picks`).
  The quality bar that drives the `below_breakeven` flag is
  `pricing.min_edge_over_cost` (default 1.0). `pricing.target_atr_mult`
  (default 2.0) sets the reward distance and hence `rr_ratio ≈
  target_atr_mult / stop_atr_mult`; `pricing.min_rr_ratio` is a sanity floor.
- Different broker? Edit the `broker:` section in config.yaml.

### Sanity checks before trading

- **High `pred_mean` + high `pred_std`** = model is unsure. Today's run put **AAV** at rank #1 with `pred_mean=0.0048` *and* `pred_std=0.0021`. The std is ~44% of the mean — the prediction is noisy. AAV is also down 18% in the last month with negative earnings; the model's "buy" signal is a mean-reversion bet on a beaten-down stock, which often fails when the decline is fundamental rather than technical. **In this case I would skip AAV** despite the high score.
- **`mom_5` < 0 and `mom_20` < 0** with a high `pred_mean` is the classic mean-reversion bet. It works often enough that the model finds them, but watch the news layer — if Claude scored it –1, the technicals + news disagree with the model and the pick is unreliable.
- **Predicted return < 0.0030 (30 bps)** means the expected edge is below the round-trip transaction cost (config default 30 bps). Most of today's picks fall here. Either don't trade these, or wait for higher-conviction signals.
- **`adjusted` close to `pred_mean`** for every pick = the news layer didn't move the ranking much. The 5% news weight is intentionally small; don't expect dramatic re-ranks.

### Today's example (the run we just did)

```
symbol  pred_mean  news_score  adjusted   call
AAV     0.0048     -1          0.00456    SKIP — high uncertainty + bearish fundamentals
AMS     0.0019      0          0.00190    skip — below 30 bps cost threshold
AAA     0.0018      0          0.00180    skip — below 30 bps cost threshold
DXG     0.0017     +1          0.001785   marginal — at low end of analyst range, +mom
QNS     0.0017     +1          0.001785   marginal — at low end of analyst range, +mom
```

None of these clear the cost-of-trading bar with confidence. The honest reading
of this output is **"no high-conviction trade today"** — and that's a feature,
not a bug. The system's job is to tell you when to act *and* when to wait.

### After the exit day (T+2)

Run `evaluate.bat` (or the next predict_*.bat — it auto-evaluates). The ledger
fills in `actual_exit` and `realized_return` for each pick whose `target_date`
has elapsed (computed from each pick's own `exit_offset_days`, so a T+2 pick
evaluates two trading days later, a T+11 pick evaluates eleven trading days
later, etc.). The next Claude run sees those numbers in its prompt as a "Past
performance feedback" block.

## Explanations: why was this picked?

Both Claude and Gemini modes now produce **per-ticker explanations** in the
final picks output, so you can see what news the LLM found and why it scored
the way it did.

For each pick the explained view prints:

```
=== #1 DXG  —  CTCP Tap Doan Dat Xanh ===
  Trade: buy 100 @ 15,350 VND  |  target 15,376  |  stop 14,534
  P&L (after ACBS fees 6,607): net -4,007  rr -0.05  -> skip (rr/net too low)
  Signal: pred_mean=+0.0017  news=+1  adjusted=+0.0018
  Business: Real-estate brokerage and development (Dat Xanh Group, HOSE).
  Key drivers: Mortgage rates; property launch approvals; project pipeline.
  News found:
    - DXG hit 80% of FY profit target in Q1 2026 (cafef, May 4)
    - Profit nearly tripled vs same period last year (vnexpress, May 2)
    - Disclosed loan of over 3.3 trillion VND
```

The same fields are also stored in the saved picks JSON
(`reports/picks_<mode>_<date>.json`) under each pick's `business`,
`drivers`, `key_news`, and `rationale` keys.

### Where the explanation comes from per mode

| mode | how it runs | source of explanation |
| ---- | ----------- | --------------------- |
| **base** | `predict_base.bat` or CLI | None — pure ML, no narrative. Compact table only. |
| **claude** | Paste [`claude_prompt.md`](claude_prompt.md) into Claude Desktop (Claude Code or Cowork) | Claude in-session uses `WebSearch` + `WebFetch` to research per-ticker emergent dimensions, fills the plan, runs `claude-finalize`. After finalize, **Claude offers to schedule a sell reminder for 15:00 ICT on T+(N−1)** — 30 min after the VN market close, so you can review the day's close and queue exit orders for the next-day open. |
| **gemini** | `predict_gemini.bat` (two-step) | `predict --mode gemini` writes a prompt; you paste into Gemini Chat (web with browsing); save Gemini's JSON response to `reports/gemini_response_<date>.json`; run `gemini-finalize` to merge it into explained picks. After finalize, **you receive a `SELL-REMINDER` block keyed to 15:00 ICT on T+(N−1)** (post-close on the day before the sell day), and Gemini prompts you about scheduling a reminder. |

### Gemini two-step flow

```bash
# Step 1: ML stage + emit prompt
.venv\Scripts\python -m stockpredict.cli run --mode gemini

# (open the prompt file, paste into gemini.google.com with browsing on,
#  copy the JSON response, save to reports\gemini_response_<date>.json)

# Step 2: merge response into final picks
.venv\Scripts\python -m stockpredict.cli gemini-finalize reports\gemini_prompt_<date>.txt
```

`predict_gemini.bat` walks through both steps interactively (option 1 emits
the prompt, option 2 finalizes after you've saved the response).

## Research scope (Claude / Gemini modes)

Both Claude and Gemini are instructed to do **proper research** on each
candidate, not just a sentiment scan of company news. **The research
dimensions are emergent, not predefined** — for each ticker, the LLM
itself decides which 3-7 dimensions are relevant (based on what the
company actually does and what's actually happening), then researches
those.

The reference list of common categories that the LLM may consult lives
in [`src/stockpredict/news/research_dimensions.py`](src/stockpredict/news/research_dimensions.py)
and is embedded into all three prompt paths as **inspiration**, not as a
mandatory checklist. The LLM is told explicitly:

> *"Different companies have different drivers. Skip categories that
> don't apply, add ones that do. Idiosyncratic drivers (key customers,
> single contracts, pegs, peer M&A) often matter more than any standard
> category."*

The reference categories are: company-specific, sector / industry,
Vietnam domestic macro, Vietnamese policy / law (Nghị quyết / Nghị định
/ Thông tư / Quyết định / dự thảo luật, including upcoming and draft),
global macro, geopolitical / disruptions, legal / calendar.

Per ticker the LLM emits a `dimensions` field listing what it actually
researched (its own words), plus `key_news` with findings, dates, and
sources. The CLI prints both in the explained view so you can see what
the LLM thought was relevant for that specific name on that day.

**VN-Index trend call.** Before scoring tickers, the Claude plan and the
Gemini prompt both ask the LLM to research where the VN-Index is likely
headed over the holding window — trend / momentum, position vs the
50/200-day MAs, market breadth (incl. *xanh vỏ đỏ lòng*, where a few
large caps prop up the index while most stocks fall), foreign flows, and
scheduled macro — and state an explicit **UP / SIDEWAYS / DOWN** call with
a confidence in `global_summary`. That call then tilts every pick (more
conservative in a likely-down tape, more constructive in a likely-up one),
though a counter-trend or strongly-catalysed name can override it. This is
a purely LLM-judgement signal — there is no quantitative index model or
automated regime gate in the pipeline.

## What each mode actually searches

| mode | search mechanism | scope |
| ---- | ---------------- | ----- |
| `gemini` | Gemini Chat web with browsing on (`google_search` grounding) | **whole web** via Google. Source list in the prompt is hints. |
| `claude` (interactive) | Claude in this Code/Cowork session uses `WebFetch` + `WebSearch` against the URLs the markdown plan lists, plus broader Google searches the plan instructs. | **whole web**, but only as broadly as the in-session Claude chooses. The plan now explicitly says "these URLs are STARTING POINTS — also use WebSearch for the latest news beyond them." |

The starting URL list (in [`config.yaml`](config.yaml) under `news.vn_sources`):
baomoi, Google News VN, cafef, vietstock, vneconomy, ndh, theinvestor,
tradingview. Macro: baomoi/chungkhoan, vietstock-en, Google News VN-Index,
Google News VN-macro, Reuters Asia, Bloomberg Markets.

These hint at where ticker-specific Vietnamese coverage tends to live.
The actual research isn't restricted to them. If a ticker has a story on a
sector site, regulator notice, broker note, or international press, the LLM
can find it via search and is told to do so.

## How the news layer works (Claude / Gemini)

The LLM does **not** score on price or technicals — those are already in the
ML input. Instead the prompt forces a four-step business-aware analysis:

1. **Identify the business** from the company name (`organ_name`, attached
   from the universe parquet). Example: `CTCP Đường Quảng Ngãi` → sugar /
   dairy producer.
2. **Identify the news drivers** that move *that specific business* on the
   T+2 horizon. Real estate cares about mortgage rates and sector policy;
   sugar producers care about commodity prices and FX; banks care about SBV
   policy moves and NPL data; rubber and mining care about input prices and
   China industrial demand; pangasius exporters care about US/EU tariffs;
   etc.
3. **Search news** in those categories. Primary sources:
   - `https://baomoi.com/tim-kiem/<TICKER>.epi` (per-ticker aggregator —
     empirically the most reliable single source)
   - Google News search for the ticker + macro topic
   - Vietnamese financial press (cafef.vn, vietstock.vn, vneconomy.vn,
     ndh.vn, theinvestor.vn) and global macro (Reuters, Bloomberg)
4. **Score** −1 / 0 / +1 — or `DROP` for delisting / suspension /
   bankruptcy. `DROP` is a **hard override**: the ticker is excluded
   regardless of how high the ML score is.

### Why DROP exists

The news weight is intentionally small (5%) so a single news call doesn't
overwhelm the ML signal. But that means a ticker with `pred_mean=0.0048`
and bearish news (`-1`) only drops to `0.00456` — still possibly the top
pick. For genuinely fatal news (delisting), a small re-weight isn't enough.
`DROP` removes the ticker entirely.

In our 2026-05-05 demo, the ML model put **AAV** at rank #1 with the
highest predicted return. Reading baomoi revealed: HNX delisted AAV on
April 27, 2026. ML couldn't have known — it only sees price/volume. Without
the `DROP` override, the system would have happily recommended a stock you
literally cannot trade.

## After-14:30 cutoff: T+0 = next opening day

Vietnamese exchanges close their continuous-trading session at 14:30 (HOSE
ATC runs 14:30–14:45). After 14:30 today's close is locked in — a buy
order placed now can't fill at today's close. The program picks up on this
automatically:

- Run **at or before 14:30**: T+0 = today (calendar date).
- Run **after 14:30**: T+0 = the next trading day (weekends and Vietnamese
  holidays are skipped via the cached trading-day calendar).

The detection happens at CLI entry via [`tracking.effective_today_for_trading`](src/stockpredict/tracking.py).
Every downstream date stamp (`as_of` in the ledger, the picks JSON
filename, the news plan filename, the T+2 `target_date`) uses the same
cutoff-aware date.

## Same-day re-runs: distinct parameters → distinct files

Every saved artifact (picks JSON, news plan markdown, candidates
parquet, meta JSON) is suffixed with a **run signature** that captures
the parameters that affect the picks: `mode_d{horizon}[_HOSE][_x<TICKERS>]`.

Examples:

```
reports/picks_2026-05-05_base_d2.json
reports/picks_2026-05-05_claude_d18_HOSE.json
reports/claude_news_plan_2026-05-05_claude_d18_HOSE.md
```

So if you run multiple predictions in a single day with different
parameters, none of them overwrite each other. A re-run of the **exact
same parameters** does override (idempotent within the day).

The same signature is the basis for the ledger's `run_id` (e.g.
`20260505_claude_d18_HOSE`) and gets stored in a dedicated
`signature` column. That powers the next section.

## Self-correction (Claude mode only)

Every run records every pick to `cache/predictions.parquet` (one row per
pick, tagged with `run_id`, `signature`, `mode`, `exit_offset_days`,
`target_date`). When you run again later, `evaluate.bat` (or any
`predict_*.bat`) auto-scores predictions whose target date has elapsed
by looking up the actual close price in the cached OHLCV.

The Claude mode markdown plan for in-session Claude includes a
**Past performance feedback**
block. The block now has three apples-to-apples views, in order of
specificity:

1. **By full run signature** — exact parameter match (mode + horizon +
   hose-only). The row matching today's signature is marked
   `← THIS RUN`. This is the highest-fidelity comparison: what your
   `claude_d18` runs have actually returned.
2. **By horizon** — broader cross-run comparison. Useful when the exact
   signature has too few data points yet.
3. **By news_score** — pooled across runs, but lets Claude see whether
   its `+1` calls have been more / less predictive than its `-1` calls.

Plus the most-recent 5 evaluated picks with realized returns. Claude is
instructed to weight the **THIS RUN signature** row most heavily, then
the **THIS RUN horizon** row.

### By dimension category cited

Claude tags every Step 4 finding with a kebab-case dimension name in
square brackets — e.g. `[insider-action]`, `[sector-flow]`,
`[earnings]`. `parse_plan` extracts those tags into a
`dimensions_cited` column on the ledger, and `recent_performance`
aggregates hit-rate per tag (a pick that cites three tags contributes
to all three buckets). The feedback block renders this as:

```
### By dimension category cited
| dimension          | n  | hit_rate | mean_return | median  |
| `insider-action`   | 12 | 75.0%    | +0.0234     | +0.0156 |
| `sector-flow`      | 8  | 62.5%    | +0.0145     | +0.0089 |
| `earnings`         | 15 | 53.3%    | +0.0089     | +0.0067 |
| `governance`       | 6  | 33.3%    | -0.0067     | -0.0050 |
| `trade-policy`     | 4  | 25.0%    | -0.0156     | -0.0123 |
```

So Claude knows that on this signature, `[insider-action]` findings
have been worth +234bps but `[trade-policy]` findings have been worth
-156bps — actionable signal for which dimensions to weight more
heavily during *today's* research, not just *today's* scoring.

Tags with n < 2 are filtered out as noise. Tag-naming consistency
matters: re-using `[insider-action]` across tickers makes the
aggregation work; coining `[fpt-insider]` and `[asp-insider]`
fragments the buckets and the table becomes useless. The
[`claude_prompt.md`](claude_prompt.md) tag-naming guidelines spell
this out.

### Entry-execution sanity check

The ledger also records, for every evaluated pick, the **buy-day OHLC**
— the open / low / close on `as_of + 1 trading day`, the day the user
was actually supposed to buy. From those, we derive `entry_slippage =
(t0_low − entry_price) / entry_price`:

- **negative** → the day's low was below the predicted `entry_price`,
  i.e. a limit order would have filled cheaper than quoted (good).
- **positive** → the day's low was *above* `entry_price`, i.e. the
  market gapped past our quoted entry and we never could have filled
  at that price. The realized_return on those rows is then **fictional**
  — the model was right about direction but the trade wasn't actually
  executable at the entry we promised.

`feedback_block` adds an **Entry-execution sanity check** section
showing mean / median slippage, `% picks unreachable`, and
`mean savings when reachable`. Claude sees this on every run and can
calibrate its scoring accordingly.

Ledger columns added for this: `t0_open`, `t0_low`, `t0_close`,
`entry_slippage`. Old ledgers without these columns get them backfilled
as NaN on read; rows are filled retroactively the next time
`evaluate_pending` revisits them.

Gemini mode is prompt-only. The Gemini Chat web UI cannot read the local
ledger or update it, so Gemini does not receive past-performance feedback.

### Manual program-level self-correction

Everything above is **passive** self-correction: feedback is injected into
the prompt and Claude adjusts scoring *within a single run*. The program
itself — `claude_prompt.md`, `config.yaml`, source code — never changes.

For an **active** loop that mutates the program based on a chosen report,
use [`self_correct_prompt.md`](self_correct_prompt.md). Open Claude Code
(or Cowork) in your clone of the repo, paste the file's contents, and supply a
`reports\picks_claude_<date>_<sig>.json` whose target date has fully
elapsed. It focuses on exactly **two questions** and proposes one
narrowly-scoped, approval-gated edit per finding:

1. **Missed winners** — `regret` lists the realized top-N liquid tickers (bought
   T-2, sold today) the model didn't surface, and the loop investigates *why*
   (a gate dropped it — e.g. `overbought_rsi_max` — or the model scored it low).
   The lever is a config knob, or the `train-missed` → `backtest-ab` variant
   (promote it only if its win rate holds).
2. **Entry-price misses** — the limit-fill calibration (`entry_limit_filled` /
   `fill_margin`), now read per conviction tier (`pred_low_alpha`), since a deep
   weak-pick dip not filling is by design.

Every diff is shown and applied only after explicit per-file approval — nothing
is mutated silently. Output lands at `reports\self_correction_<date>_<sig>.md`.

This is intentionally Claude-Code-only: the work needs `Read` + `Edit` on
local files, which the prompt-only Gemini path can't do.

**Not applicable to LLM-only reports.** Both focuses are ML-pipeline concepts
— missed-winners regret targets the ML model's skill / gates, and entry-price
calibration tunes the mechanical low head. LLM-only picks (`picks_claude_llm_*`,
`mode: claude_llm`) have no ML model, no `pred_mean`, and no mechanical
`entry_limit_price`, so the self-correction loop skips them.

## Setup (one-time)

```bash
# from the repo root (your clone)
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -e ".[dev,llm]"
```

Optional API keys live in [`.env.example`](.env.example); copy it to `.env`
to set them. The .bat files load `.env` automatically.

## Daily workflow

```
                     +-------------------------+
                     |   predict_*.bat (T)     |
                     |   - fetch + train       |
                     |   - record picks        |
                     +-----------+-------------+
                                 |
                                 v
                     +-------------------------+
                     |   you trade off picks   |
                     +-----------+-------------+
                                 |
                                 v  (T+2 morning)
                     +-------------------------+
                     |   predict_*.bat (T+2)   |
                     |   - auto-evaluates T's  |
                     |     picks               |
                     |   - feeds Claude with   |
                     |     past performance    |
                     |   - records new picks   |
                     +-------------------------+
```

If you skip a day, `evaluate.bat` catches up the ledger separately.

## Backtesting

Walk-forward: rolling 2-year train, 6-month out-of-sample, sliding 6 months.

```bash
.venv\Scripts\python -m stockpredict.cli backtest --start 2022-01-01
```

**Decision gate**: do not trade live until `hit_rate_net > 0.50` and
`mean_return_net > 0` over a multi-year window. The walk-forward report drops
into `reports/backtest_<date>/` with `summary.md` and `equity.png`.

## Tests

```bash
.venv\Scripts\python -m pytest -q
```

170 tests, all use synthetic data — no network required. Coverage spans
features, pricing, the trading calendar, cache freshness + watermarks,
the rate limiter, walk-forward backtest sanity, run-signature uniqueness,
entry slippage, and per-dimension hit-rate aggregation.

## Configuration

All knobs in [`config.yaml`](config.yaml): liquidity thresholds, feature
windows, model hyperparameters, walk-forward windows, news source URLs, top-K.

Key knobs:

- `target.entry: close | next_open` — switch to `next_open` if you intend to
  buy at T+1 open instead of T close.
- `data.source: VCI | KBS | MSN` — vnstock data source. KBS is most complete;
  VCI is sometimes more reliable per-ticker.
- `backtest.cost_bps: 30` — round-trip transaction cost in basis points.
- `pricing.entry_low_alpha: 0.40` — **base/pivot** quantile for the limit-buy
  "low" head. The entry price is each ticker's α-quantile of its own recent
  next-day-low returns; lower α = deeper dip / lower fill probability, higher α =
  shallower / more likely to fill. The low head is a per-ticker empirical
  quantile, *not* an ML model — an earlier LightGBM quantile head had negative
  skill and was replaced. Auto-sizes its trailing window via
  `entry_low_target_tail_obs` (sized for the deepest reachable α).
- `pricing.entry_alpha_couple_conviction: true` + `entry_alpha_weak_mult` (0.6)
  / `entry_alpha_strong_mult` (1.25) / `entry_alpha_strong_edge` (3.0) /
  `entry_alpha_hard_min`/`max` — **conviction-coupled entry depth**. The per-pick
  α = `entry_low_alpha × multiplier`, scaled by how far `pred_mean` clears the
  cost bar: a strong pick → shallow dip (high α, "get me in"); a marginal /
  below-breakeven pick → deep dip (low α, only fills at a bargain — so the
  exact-N "always return N" rule can't force a bad trade). The band is
  multipliers of the base, so it rescales if you retune `entry_low_alpha`. Set
  `entry_alpha_couple_conviction: false` to use a single flat α for every pick.
  `pricing.entry_alpha_overbought_start` (60) / `_full` (85) / `_mult` (0.5)
  add a **soft overbought penalty**: the more overbought a pick (RSI), the lower
  its α → deeper dip → it only fills on a real pullback (`_mult: 1.0` disables).
- `pricing.overbought_rsi_max: 0` — **overbought hard gate**. Drop any candidate
  whose `rsi_14` exceeds this level (an exhaustion blow-off run too far tends to
  reverse, and RSI>80 names historically win far less often). `0` disables.
  Distinct from the liquidity volume-spike defense (`min_adv_active_days`), which
  guards tradability, not exhaustion. Pairs with the soft penalty above: the gate
  drops the worst, the penalty makes the rest only fillable at a discount.
- `pricing.ceiling_limits` / `ceiling_tol` — names locked limit-up (closed at
  the daily price-band ceiling) are excluded from the pickable universe, since a
  limit-buy can't fill against an all-buyers queue.
- `pricing.corp_action_lookback: 20` — corporate-action guard. A single-day
  close-to-close move beyond the exchange price band (`ceiling_limits`: HOSE 7%
  / HNX 10% / UPCOM 15%) is physically impossible without a corporate action, so
  it's an unadjusted split / rights / special-dividend gap in the raw feed — not
  a real crash. Such a gap poisons mom_*/atr_14/rsi_14 (and, in the target
  window, fakes the label), so a ticker is dropped from both the candidate set
  and the training panel while a band-breaking move sits inside this lookback
  window (matches the longest feature window, mom_20). This is what keeps the
  model from chasing phantom oversold bounces (e.g. a −38% ex-rights gap). Set
  to `0` to disable.
- `pricing.max_participation_pct: 1.0` — advisory cap, as a % of `adv_vnd_20`,
  for the per-pick `suggested_max_units` column. Purely informational — never
  affects selection. Set to `0` to disable the column.

## Layout

```
<repo root>/
  predict_base.bat         double-click entry: ML only
  claude_prompt.md         paste into Claude Desktop for ML + Claude research
  predict_gemini.bat       double-click entry: ML + Gemini prompt
  evaluate.bat             double-click entry: grade past picks
  .env.example             API key template
  config.yaml              all tunable knobs

  src/stockpredict/
    cli.py                 click entry point
    selector.py            curated bluechip list + universe top-up
    tracking.py            prediction ledger + evaluate_pending + feedback_block
    envfile.py             .env loader
    data/                  vnstock wrappers (+ vnai quota bypass) + parquet cache + per-source rate limiter
    features/              technical + microstructure indicators
    model/                 target alignment, LightGBM ensemble
    backtest/              walk-forward simulator
    news/
      claude_runner.py     interactive Claude — hybrid plan (ML-ranked candidates)
      claude_llm_runner.py interactive Claude — LLM-only plan (whole universe, no ML)
      gemini_prompt.py     prompt builder for Gemini Chat (web)
      sources.py           URL builders
    modes/                 base / claude / gemini orchestrators

  tests/                   pytest, synthetic data only
  cache/                   OHLCV parquet cache, predictions ledger
  models/                  saved LightGBM ensembles
  reports/                 picks, news plans, prompts, backtest reports
```

## License

- **This code** is licensed under **AGPL-3.0-or-later** (see [`LICENSE`](LICENSE)).
  Forks must remain open-source under the same terms; if you run a modified
  version as a network service, you must publish your modifications.
- **vnstock**, the data-source dependency this program uses at runtime, is
  **non-commercial / personal-research only** under its own custom license.
  See [`NOTICE`](NOTICE) for the verbatim restriction and contact details
  for commercial-use negotiation. The AGPL on this code does **not**
  override that restriction — anyone running this program for any
  commercial purpose must obtain a commercial vnstock license separately.
- **Contributions** are accepted under AGPL-3.0 by default with a DCO
  sign-off on every commit. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for
  the full workflow. The maintainer reserves the right to dual-license
  this software under separate commercial terms; substantial contributions
  may be asked to sign a one-off CLA at acceptance time.

## Caveats

- The **20 req/min "guest tier" cap is vnstock's own client-side quota**
  (the bundled `vnai` layer), not the data providers' server limit — vnstock
  only automates access to public APIs you already have legitimate access to.
  With `data.bypass_vnai_quota: true` (default) the fetcher calls the
  underlying provider endpoint directly via the undecorated
  `provider.history.__wrapped__`, so that quota never applies. What's left is
  our **own** politeness throttle, a per-source sliding-window limiter
  (`data.api_per_min: 60` in [`config.yaml`](config.yaml), override via
  `STOCKPREDICT_API_PER_MIN`) sized against the providers' real servers
  (which tolerate ~120/min). If vnstock's internals ever change, the bypass
  falls back to the metered path automatically and logs a warning; a genuine
  429 is still detected (English + Vietnamese) and triggers a 65-second pause
  on that source before retry. Failed ticker fetches never abort the run —
  the pipeline proceeds with whatever's in the parquet cache and reports how
  many tickers were skipped.
- vnstock's `Listing` API only supports `KBS`/`VCI`/`MSN`; `Quote` accepts the
  same set. The fetcher falls through these automatically.
- News modes are advisory, not autonomous trading. They re-rank ML output, they
  don't generate alpha on their own.
- Predicted returns are typically tiny (10–50 bps over 2 days). At 30 bps
  round-trip cost, the model needs a high hit-rate to net positive — verify
  with the backtest before trading.
- **Not investment advice.** Use at your own risk; backtested performance does
  not guarantee future results.
