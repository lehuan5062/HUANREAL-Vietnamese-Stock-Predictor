# HUANREAL Vietnamese T+N Stock Predictor

A swing-trade screener for the Vietnamese stock market. Buy on day **T**, hold
**N trading days**, sell on the exit day. The minimum is **T+2** (Vietnamese
settlement happens at noon on T+2, so the earliest legal sell is the T+2
afternoon session); longer holds (T+5, T+11, T+18, last-trading-day-of-month,
or *earliest-actionable* — see [`--days`](#run-command-flags)) are configurable
per run. The program ranks the most liquid HOSE / HNX / UPCOM tickers by
predicted T+N forward return and tracks every prediction so it can grade
itself later.

## Two ways to run

### A. Double-click a .bat file (base / gemini)

| File | What it does |
| ---- | ------------ |
| [`predict_base.bat`](predict_base.bat) | Pure ML + technical filter. No news. |
| [`predict_gemini.bat`](predict_gemini.bat) | ML + writes a prompt file you paste into Gemini Chat (web, with browsing). Opens the prompt in Notepad automatically. |
| [`evaluate.bat`](evaluate.bat) | Refreshes data and grades any past predictions whose T+N has now passed. |

Each predict .bat asks for days / hose-only / etfs / exclude / warm-only and runs the entire universe.

### B. Claude mode — paste a prompt into Claude Desktop

Claude mode is **driven from inside Claude Desktop** (Claude Code or Cowork),
not from a .bat. The news research uses Claude's `WebSearch` / `WebFetch`
tools, which only exist inside Claude.

**To run Claude mode:**

1. Open Claude Code or Cowork in Claude Desktop.
2. Paste the contents of [`claude_prompt.md`](claude_prompt.md) into the chat.
3. Claude will ask you for `days`, then drive the entire
   pipeline (run the ML stage → research each candidate across emergent
   per-ticker dimensions → fill the plan → finalize → report explained
   picks).

The prompt is self-contained and tells Claude everything it needs to know
about the project layout, tools, scoring rubric, the seven-dimension
*reference* (with the explicit instruction that dimensions are per-ticker
emergent, not a fixed checklist), and the ACBS fee model.

### C. CLI (advanced)

```bash
# Standard daily run (entire universe, T+2 horizon)
.venv\Scripts\python -m stockpredict.cli run --mode base

# Longer hold horizon (T+5, automatically retrains)
.venv\Scripts\python -m stockpredict.cli run --days 5 --mode base

# Last trading day of the month
.venv\Scripts\python -m stockpredict.cli run --days end --mode base

# Other commands
.venv\Scripts\python -m stockpredict.cli evaluate
.venv\Scripts\python -m stockpredict.cli track --limit 20
.venv\Scripts\python -m stockpredict.cli backtest --start 2022-01-01
```

**Note**: `--mode claude` at the CLI emits the markdown research plan (the
same plan the prompt file drives). The primary Claude path is the prompt file
(option B above); the in-session Claude fills the plan and runs
`claude-finalize`.

### Run command flags

| flag | meaning | default |
| ---- | ------- | ------- |
| `--days N`, `--days end`, or `--days earliest` | T+N exit window. Min 2 (Vietnamese T+2 settlement). `end` = last trading day of the month, rolling to next month if too close. `earliest` = iterative search: trains+predicts at T+N, T+N+1, T+N+2, … (starting at `--earliest-start`) and stops at the first horizon with at least one actionable pick. **No upper cap** — runs until found, Ctrl+C to abort. Slow — minutes per iteration. The model is horizon-specific, so non-`2` horizons force a retrain. | `earliest` |
| `--earliest-start N` | Only used when `--days earliest`. Integer ≥ 2; the search begins at T+N. Ignored for any other `--days` value. | `2` |
| `--hose-only` | Restrict the universe to HOSE-listed tickers. Refreshes the universe via VCI to try to get exchange info; falls back to ~43 curated HOSE bluechips (VN30 + HOSE mid-caps) if the data source doesn't return `exchange`. | `False` |
| `--mode {base,claude,gemini}` | which pipeline | `base` |
| `--top N` | cap how many picks to print | _(off — lists all actionable picks dynamically)_ |
| `--skip-train` | reuse cached `models/latest.pkl` (only works at `days=2`; otherwise ignored) | off |

## Universe coverage

Every run covers the **entire universe** (all of HOSE + HNX + UPCOM, ~1,765
tickers today and growing) with no time cap. Expect ~75 minutes the first time
(vnstock's free guest tier rate-limits at 20 API calls per minute), much faster
on subsequent runs since cached, up-to-date tickers cost 0 API calls. The smart
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

## Trading-day calendar (T+N math)

T+N is computed using the **actual Vietnamese trading-day calendar** built
from cached OHLCV data — weekends AND every Vietnamese holiday the market
historically closed for (Tết, Reunification Day, Labor Day, National Day,
plus ad-hoc closures) are skipped.

| computation | weekends | Vietnamese holidays |
| ----------- | -------- | ------------------- |
| ML training target (`shift(-N)` on OHLCV index) | excluded | excluded |
| `target_date` in the predictions ledger ([`tracking._next_trading_offset`](src/stockpredict/tracking.py)) | excluded | excluded |
| Realized-return evaluation (`evaluate_pending`) | excluded | excluded |

Example — for T = 2025-04-29 (Tue, last trading day before the
Reunification + Labor Day cluster), T+2 lands on **2025-05-06**, not
2025-05-01 (the naive Mon-Fri arithmetic answer). Verified end-to-end
against the real cache in
[`tests/test_trading_calendar.py`](tests/test_trading_calendar.py).

## Horizon search: `--days earliest`

Pass `--days earliest` (or type `earliest` at the .bat prompt) to find
the **shortest hold period** that produces an actionable trade. The
program:

1. Trains a fresh model at T+`earliest_start` (default T+2), predicts,
   checks for `actionable=True`.
2. If found, stops and emits picks at that horizon.
3. Otherwise repeats at T+`start+1`, T+`start+2`, … with **no upper
   cap** — the loop runs until at least one actionable pick is found.
   The user can hit Ctrl+C to abort if it takes too long, and a
   "still searching past T+N" callout prints every 30 horizons so the
   process is never silent.
4. The only safety net: if the panel comes back empty for 60
   consecutive horizons (history too short or symbol set too narrow),
   the loop bails with an error rather than spinning forever.

The starting horizon is configurable: `--earliest-start <N>` (or the
follow-up `Earliest-start: T+N` prompt in the `.bat` files). Minimum
is 2 (Vietnamese T+2 settlement floor); default is 2. Use a higher
start when you already know shorter horizons aren't going to clear
the cost gate (e.g. on illiquid tickers where T+2 < ACBS round-trip
cost is structurally guaranteed).

Typical output:

```
--days earliest -> will iterate T+2, T+3, T+4, ... after data fetch (NO upper cap), stopping at the first horizon with >=1 actionable pick. Ctrl+C to abort.
searching for earliest actionable horizon (starting T+2, no upper cap; trains a fresh model per horizon — Ctrl+C to abort)...
  T+2... none
  T+3... none
  T+4... none
  ...
  T+11... found 1 actionable pick(s)

predicting (mode=base)...
=== #1 DCL  [BEST rr | BEST net | BEST composite]
  Trade: buy 100 @ 37,500 VND  |  target 40,155  |  stop 34,747
  P&L (after ACBS fees 16,829): net +248,671  rr 0.85  -> ACTIONABLE
saved -> reports/picks_<date>_base_d11_u100.json
```

Cost: each iteration trains a model from scratch (~20-30s on the
default 50-ticker universe, longer on full). Best case (T+`start` is
already actionable) is one cycle. Worst case is unbounded — on a low-
volatility week the search may traverse dozens of horizons before
something clears the ACBS cost gate.

The discovered horizon is captured in the picks filename and ledger
signature (`d11` in the example above), so the self-correction layer
keeps separate hit-rate stats per horizon found this way.

## Horizon (`--days N`)

| `--days` | meaning | constraint |
| -------- | ------- | ---------- |
| `2` (default) | sell at the close of T+2 | **afternoon session only** — Vietnamese T+2 settlement happens at noon, shares are deliverable from ~13:00 |
| `3`, `4`, `5`… | sell at the close of T+N | any time on the exit day |
| `end` | sell on the last trading day of the month | rolls to the **next** month if today is too close to month-end to satisfy T+2 (e.g. last day, last day −1) |

`--days end` resolves at CLI entry against the actual trading-day calendar
(weekends + Vietnamese holidays excluded). Future trading days are
projected forward as weekdays. Examples (today = 2026-05-05):

```
T=2026-05-05  -> T+18 = 2026-05-29   (last trading day of May)
T=2026-05-27  -> T+2  = 2026-05-29   (T+2 satisfied within May)
T=2026-05-28  -> T+23 = 2026-06-30   (rolled to June, T+1 < 2)
T=2026-05-29  -> T+22 = 2026-06-30   (rolled to June, T+0 < 2)
```

The trading-day calendar **auto-extends** for any horizon — past days
come from the cached OHLCV index, future days are projected as weekdays.
You don't need to refresh the cache or pass extra flags for long
horizons. Worst case: a future weekday turns out to be an unannounced
holiday, the resolved target_date is off by 1 day, and `evaluate_pending`
still finds the right close from the OHLCV cache once the date arrives.

The trained model is **horizon-specific**: passing `--days 5` retrains on
T+5 forward returns. The `--skip-train` flag is therefore ignored when
`--days != 2`. The first run at a new horizon will take a couple of extra
minutes for training; subsequent runs at the same horizon can use
`--skip-train`.

Predicted returns at longer horizons are typically larger (more time for
moves), which improves the chance of clearing the ~43-bps round-trip fee
floor. But the variance also grows — verify with the backtest.

## "Best choice" badges

When at least one pick has `actionable=True`, the explained view tags the
leader in each of four categories. A single ticker can win multiple
badges:

```
=== #1 ASP  —  An Pha Petroleum  [BEST adjusted | BEST rr | BEST net | BEST composite]
  Trade: buy 200 @ 6,510 VND  |  target 6,913  |  stop 6,209
  ...
```

| badge | criterion |
| ----- | --------- |
| `BEST adjusted` | Highest `adjusted` (= `pred_mean × (1 + 0.05 × news_score)`). The system's overall conviction. |
| `BEST rr` | Highest `rr_ratio = net_reward / net_loss`. Most asymmetric upside-vs-downside. |
| `BEST net` | Highest `net_reward_vnd`. Biggest per-share dollar edge (net of fees). |
| `BEST composite` | Lowest sum of (rank by adjusted) + (rank by rr) + (rank by net). The all-rounder. |

The four boolean fields (`best_adjusted`, `best_rr`, `best_net`,
`best_composite`) also persist into `picks_<mode>_<date>.json`. If no
pick is actionable, no badges are set.

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
| `entry_vnd` | **limit-buy price to place** — the predicted next-day dip, `close × (1 + pred_low)` where `pred_low` is the per-ticker α-quantile (`pricing.entry_low_alpha`) of recent next-day-low returns (clipped never above the close). Falls back to the close when no low head is present. |
| `entry_limit_pct` | the predicted dip vs close (≤ 0) baked into `entry_vnd` |
| `target_vnd` | price to sell at on the exit day T+N (`entry × (1 + pred_mean)`) |
| `target_low_vnd` / `target_high_vnd` | mean ± 1 ensemble-std target band |
| `stop_vnd` | stop-loss (`entry − stop_atr_mult × ATR(14)`) |
| `gross_reward_vnd` | `target − entry` per share (before fees) |
| `max_loss_vnd` | `entry − stop` per share (before fees) |
| `fees_round_trip_vnd` | ACBS commission + VAT + PIT for this round trip, per share |
| `net_reward_vnd` | **predicted per-share profit after all fees** — the headline number |
| `net_loss_vnd` | worst-case per-share loss if stopped out (max_loss + fees) |
| `rr_ratio` | `net_reward / net_loss` — should be ≥ 0.8 to be `actionable` |
| `breakeven_pct` | what % the price needs to move just to cover fees (~0.43% at ACBS) |
| `actionable` | `True` only when `net_reward > 0` AND `rr_ratio ≥ 0.8` |

### Diagnostic columns (for sanity-checking the signal)

| column | meaning | how to read it |
| ------ | ------- | -------------- |
| `pred_mean` | predicted T+N return for the chosen horizon | `+0.0017` = +0.17% before costs |
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
symbol entry_vnd target_vnd stop_vnd fees   net_reward net_loss   rr   actionable
AMS    10,100    10,119     9,062    4,348  -2,448     108,148    -0.02 False
DXG    15,350    15,376     14,534   6,607  -4,007     88,207     -0.05 False
ABB    14,900    14,925     14,457   6,414  -3,914     50,714     -0.08 False
```

Reading this:
- **Buy DXG at 15,350 VND/share, sell at 15,376 VND/share, stop at 14,534.**
- All figures are per share — you size the position yourself.
- Round-trip fees (per share): 6,607 VND.
- Predicted net P&L: **−4,007 VND/share** (loss). Why? `pred_mean=+0.0017` → only
  +26 VND/share gross, which doesn't cover the ACBS round-trip fees per share.
- `actionable=False` — don't trade.

This is the tool doing its job. Most days the model's T+N predictions are
smaller than ACBS round-trip costs (especially at the T+2 minimum where the
horizon is shortest), so almost everything will show `actionable=False`.
**That's a feature.** The bar to clear before placing a real trade is
`pred_mean > breakeven_pct + risk-reward gate`, which today's picks don't.
Longer horizons (T+5, T+11, …) give pred_mean more room to clear the cost
gate — see `--days earliest` for a search that finds the *shortest* horizon
that does.

### Tuning

- All P&L figures are per share — you size the position yourself. Fees scale
  linearly with trade value, so the % cost stays at 0.43% regardless of size.
- Want to relax the actionable gate? Lower `pricing.min_rr_ratio` in
  config.yaml (default 0.8). Setting it to 0.0 would only require positive
  net_reward.
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

### After the exit day (T+N)

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
| **claude** | Paste [`claude_prompt.md`](claude_prompt.md) into Claude Desktop (Claude Code or Cowork) | Claude in-session uses `WebSearch` + `WebFetch` to research per-ticker emergent dimensions, fills the plan, runs `claude-finalize`. After finalize, **Claude offers to schedule a sell reminder for the target day in GMT+7 (Vietnamese ICT).** |
| **gemini** | `predict_gemini.bat` (two-step) | `predict --mode gemini` writes a prompt; you paste into Gemini Chat (web with browsing); save Gemini's JSON response to `reports/gemini_response_<date>.json`; run `gemini-finalize` to merge it into explained picks. After finalize, **you receive a `SELL-REMINDER` block with the target exit day in GMT+7 (Vietnamese ICT), and Gemini prompts you about scheduling a reminder.** |

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
   chosen T+N horizon. Real estate cares about mortgage rates and sector policy;
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
filename, the news plan filename, the resolved `--days end` target,
`target_date`) uses the same cutoff-aware date.

## Same-day re-runs: distinct parameters → distinct files

Every saved artifact (picks JSON, news plan markdown, candidates
parquet, meta JSON) is suffixed with a **run signature** that captures
the parameters that affect the picks: `mode_d{horizon}_u100[_HOSE][_x<TICKERS>]`.
The `u100` token is a fixed constant (position sizing was removed; pricing is
per share) kept so filenames and ledger IDs stay backward-compatible.

Examples:

```
reports/picks_2026-05-05_base_d2_u100.json
reports/picks_2026-05-05_claude_d18_u100_HOSE.json
reports/claude_news_plan_2026-05-05_claude_d18_u100_HOSE.md
```

So if you run multiple predictions in a single day with different
parameters, none of them overwrite each other. A re-run of the **exact
same parameters** does override (idempotent within the day).

The same signature is the basis for the ledger's `run_id` (e.g.
`20260505_claude_d18_u100_HOSE`) and gets stored in a dedicated
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
   `claude_d18_u100` runs have actually returned.
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
(or Cowork) in `D:\stock`, paste the file's contents, and supply a
`reports\picks_claude_<date>_<sig>.json` whose target date has fully
elapsed. Claude Code cross-references the picks with the ledger's realized
returns, diagnoses systematic errors (≥3 on-report or ≥5 in the pooled
ledger), and proposes narrowly-scoped edits to `claude_prompt.md` and
`config.yaml`. Every diff is shown and applied only after explicit
per-file approval — nothing is mutated silently. Output lands at
`reports\self_correction_<date>_<sig>.md`.

This is intentionally Claude-Code-only: the work needs `Read` + `Edit` on
local files, which the prompt-only Gemini path can't do.

## Setup (one-time)

```bash
# from D:\stock
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
                                 v  (T+N morning, N = your --days)
                     +-------------------------+
                     |   predict_*.bat (T+N)   |
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

156 tests, all use synthetic data — no network required. Coverage spans
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
- `pricing.entry_low_alpha: 0.40` — quantile for the limit-buy "low" head. The
  entry price is each ticker's α-quantile of its own recent next-day-low
  returns; lower α = deeper dip / lower fill probability, higher α = shallower /
  more likely to fill. The low head is a per-ticker empirical quantile, *not* an
  ML model — an earlier LightGBM quantile head had negative skill and was
  replaced. Auto-sizes its trailing window via `entry_low_target_tail_obs`.
- `pricing.ceiling_limits` / `ceiling_tol` — names locked limit-up (closed at
  the daily price-band ceiling) are excluded from the pickable universe, since a
  limit-buy can't fill against an all-buyers queue.

## Layout

```
D:\stock\
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
    data/                  vnstock wrappers + parquet cache + global rate limiter
    features/              technical + microstructure indicators
    model/                 target alignment, LightGBM ensemble
    backtest/              walk-forward simulator
    news/
      claude_runner.py     interactive Claude (markdown plan)
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

- vnstock free tier = **20 req/min** (sliding window, server-side). The fetcher
  enforces a sliding-window limiter (default `data.api_per_min: 12` in
  [`config.yaml`](config.yaml), override via `STOCKPREDICT_API_PER_MIN`).
  When the broker still returns 429 we detect the error string (English +
  Vietnamese) and force a 65-second global pause before retrying. Failed
  ticker fetches no longer abort the run — the pipeline proceeds with
  whatever's already in the parquet cache and reports how many tickers
  were skipped.
- vnstock's `Listing` API only supports `KBS`/`VCI`/`MSN`; `Quote` accepts the
  same set. The fetcher falls through these automatically.
- News modes are advisory, not autonomous trading. They re-rank ML output, they
  don't generate alpha on their own.
- Predicted returns are typically tiny (10–50 bps over 2 days). At 30 bps
  round-trip cost, the model needs a high hit-rate to net positive — verify
  with the backtest before trading.
- **Not investment advice.** Use at your own risk; backtested performance does
  not guarantee future results.
