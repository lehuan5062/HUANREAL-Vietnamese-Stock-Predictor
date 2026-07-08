# HUANREAL Vietnamese Rebound Stock Predictor

A mean-reversion swing-trade screener for the Vietnamese stock market. It filters
the liquid HOSE / HNX / UPCOM universe down to **names in a downtrend**, and for
each one estimates from history:

- **N** — how many trading days until it bounces back to a profitable point, and
- **P** — how big that profit is,

then ranks by **score = P / N × recovery-probability** (profit per day held,
risk-adjusted by how reliably the name recovers). You **buy at the close and hold
until the price recovers to the profit target** — a *flexible exit*: there is no
stop-loss and no time cap. Every pick is recorded so the program can grade itself
later.

The "profitable point" is defined **after fees**: a day counts as profitable when
`close/entry - 1 >= round-trip cost + margin` (≈ 0.93% at ACBS defaults). So fees
are baked into the prediction, the target, and the reported net P&L.

> **Why downtrend-only + hold-until-profit?** A per-ticker "healthy" filter
> (`min_recovery_prob`) keeps only names with a strong own-history of bouncing,
> which is what makes the bet work. Backtests showed a stop-loss or a time cap
> *hurts* this strategy (you sell right before the bounce), so neither exists.

## Two ways to run

### A. Double-click a .bat file (base / gemini)

| File | What it does |
| ---- | ------------ |
| [`predict_base.bat`](predict_base.bat) | Pure model: downtrend filter + Kaplan-Meier recovery ranking. No news. |
| [`predict_gemini.bat`](predict_gemini.bat) | Model ranks, then writes a prompt you paste into Gemini Chat (web, with browsing) to vet the candidates. |
| [`evaluate.bat`](evaluate.bat) | Refreshes data and stamps the outcome of any pick that has since recovered. |

Each predict `.bat` asks for picks / hose-only / etfs / exclude / warm-only and runs the entire universe.

### B. Claude mode — paste a prompt into Claude Desktop

Claude mode runs **inside Claude Desktop** (Claude Code or Cowork), because the
research uses Claude's `WebSearch` / `WebFetch` tools.

1. Open Claude Code or Cowork.
2. Paste the contents of [`claude_prompt.md`](claude_prompt.md) into the chat.
3. Claude asks for `picks` and the **method**, then drives the whole pipeline.

The LLM's job is to **vet the bounce** — is each candidate a healthy company in a
temporary dip that will recover, or a falling knife (fraud, delisting, insolvency)
where the drop is justified? It scores each `+1` / `0` / `-1` / `DROP`, which
nudges the P/N ranking.

**Two methods (Claude mode):**

| method | who selects | who prices | report files |
| ------ | ----------- | ---------- | ------------ |
| **hybrid** (default) | the model ranks the downtrend universe by P/N and returns the top N; Claude vets those N on news and re-ranks by `adjusted = score × (1 + news_weight × news_score)` | buy at close, target from the model's P | `claude_news_plan_*` → `picks_claude_*` |
| **LLM-only** (`--llm-only`) | **Claude** picks the names from the *whole* downtrend universe and **predicts N (days to bounce) and P (profit) itself**; finalize ranks by `score = P / N` | buy at close, target = `close × (1 + P)`; no stop | `claude_llm_plan_*` → `picks_claude_llm_*` |

Both methods use the same objective (profit per day held), the same trade shape
(buy at close, hold until target), and the same report format — the only
difference is whether N and P come from the statistical estimator or from
Claude's research.

### C. CLI (advanced)

```bash
# Standard daily run (entire universe, default pick count)
.venv\Scripts\python -m stockpredict.cli run --mode base

# Ask for exactly 8 picks
.venv\Scripts\python -m stockpredict.cli run --picks 8 --mode base

# Other commands
.venv\Scripts\python -m stockpredict.cli train        # (re)fit the recovery estimator
.venv\Scripts\python -m stockpredict.cli evaluate     # stamp recovered picks
.venv\Scripts\python -m stockpredict.cli track --limit 20
.venv\Scripts\python -m stockpredict.cli backtest --start 2024-01-01
.venv\Scripts\python -m stockpredict.cli status       # what's cached / trained
# Which METHOD has picked better (base vs hybrid vs LLM-only vs gemini),
# pooled over comparable same-day runs. Advisory — writes reports/mode_comparison_<date>.md
.venv\Scripts\python -m stockpredict.cli compare-modes --window 90
```

### Run command flags

| flag | meaning | default |
| ---- | ------- | ------- |
| `--picks N`, `-n N` | How many picks to surface. Returns **exactly N** — the top N downtrend candidates by rebound `score`. A pick whose estimated recovery probability is below the healthy floor is flagged `below_recovery_bar` (counted in a `==> QUALITY` warning); if the eligible universe is smaller than N a `==> SHORTFALL` prints. | `pricing.default_picks` (1) |
| `--hose-only` | Restrict to HOSE-listed tickers. | `False` |
| `--include-etfs` / `--no-etfs` | Include HOSE ETFs (`E1VFVN30`, `FUEVFVND`, …). | included |
| `--mode {base,claude,gemini}` | which pipeline | `base` |
| `--llm-only` | **claude only.** No model ranking — the whole downtrend universe is handed to Claude, which picks / ranks by `conviction` / sets a `target_vnd` (buy at close, no stop). Emits `claude_llm_plan_<date>.md`. | off |
| `--skip-train` | reuse `models/recovery_latest.pkl` instead of retraining | off |
| `--warm-only {yes,always,no}` | cache-fetch strategy (see below) | `yes` |
| `--exclude ACB,HPG` | per-session ticker blacklist | none |

## The strategy in detail

### Downtrend filter ([`filters.downtrend_mask`](src/stockpredict/filters.py))
A candidate is a rebound candidate when it is trending down and has pulled back:
`mom_20 < 0` and `high_prox_20 <= -0.05` (≥5% below its 20-day high) and
`rsi_14 <= 50` (and `>= rsi_floor` if set). Knobs in `config.yaml` under
`strategy.downtrend`.

### Recovery estimator ([`model.train.RecoveryKMModel`](src/stockpredict/model/train.py))
For each downtrend candidate it estimates, from history:
- `pred_recovery_prob` — the eventual recovery fraction (censoring-aware),
- `pred_days` (N) — median days-to-recovery (Kaplan-Meier),
- `pred_profit` (P) — the median profit of recovered episodes.

The dominant signal is the **ticker's own** downtrend-recovery history (reliable
bouncers cluster near prob 1.0, chronic decliners near 0.0); a coarse
RSI × distance-below-high bucket and a pooled all-downtrend curve are the
fallbacks for thin tickers. It's a transparent empirical estimator — no
LightGBM, no gradient boosting.

### Ranking + healthy gate ([`model.predict.rank_today`](src/stockpredict/model/predict.py))
`score = pred_profit / pred_days × pred_recovery_prob`. Names below
`strategy.recovery.min_recovery_prob` (default 0.85) are dropped up front — this
is the healthy filter that screens out falling knives.

### Exit
Buy at the close; sell on the first day (from T+2 settlement onward) the close
first clears the profit target. No stop, no cap. In the ledger this resolves via
[`tracking.evaluate_pending`](src/stockpredict/tracking.py); the pick stays open
until it recovers.

## Reading the output

For each pick the program prints a compact table and writes
`reports/picks_<date>_<mode>_<sig>.json`. Trade columns (VND, per share):

| column | meaning |
| ------ | ------- |
| `close_vnd` | **the buy price** — you buy at today's close (there is no entry-price prediction). |
| `target_vnd` | the sell target = `close × (1 + pred_profit)`. |
| `hold_days` | expected trading days to the bounce (`pred_days`). |
| `score` | `P/N × recovery_prob` — the ranking objective. |
| `pred_days` / `pred_profit` / `pred_recovery_prob` | the recovery estimate (N / P / probability). |
| `gross_reward_vnd` | `target − close` per share (before fees). |
| `fees_round_trip_vnd` | ACBS commission + VAT + PIT for the round trip, per share. |
| `net_reward_vnd` | per-share profit after all fees — the headline number. |
| `breakeven_pct` | move needed just to cover fees (~0.43% at ACBS). |
| `below_recovery_bar` | `True` when the pick fails the quality bar (recovery prob below the floor, or the profit doesn't clear fees). Still returned to honor `--picks`, but flagged and counted in the `==> QUALITY` warning. |
| `suggested_max_units` | advisory liquidity cap = `floor(max_participation_pct% × adv_vnd_20 / close)`. Informational; never feeds selection. |

Diagnostic columns (`rsi_14`, `mom_5`, `mom_20`, `high_prox_20`, `vol_z_20`,
`adv_vnd_20`) ride along. Claude/Gemini modes add `news_score`, `adjusted`,
`business`, `dimensions`, `key_news`, and optional `adj_entry_vnd` /
`adj_target_vnd` overrides for a gap catalyst.

### How fees are calculated (ACBS default)

[`config.yaml`](config.yaml) → `broker:`. Round-trip ≈ **0.43%** of trade value:
```
buy_fee  = trade_value × 0.15% × 1.10           = 0.165%
sell_fee = trade_value × 0.15% × 1.10 + 0.10%   = 0.265%
total    ≈ 0.43%
```
This feeds `pricing.profit_threshold()` (= round-trip + 0.5% margin ≈ 0.93%),
which is the bar the recovery labeling, the target price, and `net_reward` all use.

## Backtesting

Walk-forward: rolling 2-year train, 6-month out-of-sample, sliding 6 months. The
sim buys the top-K by score each day and holds each until it recovers (flexible
exit), marking still-open trades to market at the data edge.

```bash
.venv\Scripts\python -m stockpredict.cli backtest --start 2024-01-01
```

Summary metrics: `recovery_rate`, `mean_hold_days`, `net_return_per_day`,
`max_drawdown`. **Caveat:** recovery labels are computed on each symbol's full
series, so a model trained at an anchor can peek slightly past it — OOS returns
are mildly optimistic.

For a **realistic** money-weighted picture (real cash account, 100-share lots,
liquidity caps, explicit buy+sell fees), see the portfolio simulators in
[`scripts/`](scripts/):

- `rebound_portfolio_sim.py` — fixed rolling book (SEED_DAYS/SLOTS model).
- `rebound_sim_include_held.py` — buy-daily / reinvest / T+2-min-hold /
  sell-at-first-profit; buys the day's #1 pick **even if already held**
  (can pyramid into a position). Unsold positions are counted as losses so
  the win rate is honest.
- `rebound_sim_exclude_held.py` — same rules, but the daily buy target is the
  best-ranked pick **not already held** (currently-owned tickers are removed
  from the candidate pool before ranking). Still buys every day unless every
  candidate that day is already owned.
- `rebound_sim_predicted_exit.py` — same buy side as `include_held`, but
  exits on the model's *predicted* exit day/price instead of
  sell-at-first-profit; a diagnostic for whether the model's day/price
  predictions are any good on their own.
- `rebound_sim_hold_pred_days.py` — same buy side as `include_held`, but the
  earliest sellable day is `max(T+2, pred_days)` from entry instead of a
  flat T+2 — a minimum-hold gate (not a forced exit day) so an early
  profitable close doesn't get taken before the model's own days-to-recovery
  estimate has had time to play out.
- `compare_include_vs_exclude_held.py` — runs `include_held` and
  `exclude_held` against the same data and writes a side-by-side comparison
  to `reports/sim_outputs/` (gitignored).

All `rebound_sim_*` scripts model the actual **execution**: the order
is a pre-open limit at the signal close placed the next morning ("lệnh trước
giờ") — it does NOT fill when the stock gaps up and never dips back, fills at
the open when it opens at/below the limit, else at the limit. This is the
**only** execution model they run (no lookahead/FOMO comparison built in).

### Config tuning: brute-force search + analysis

`run_config_tuner.bat` runs `scripts/rebound_config_tuner.py`: one shot per
double-click, it picks a random combination of the walk-forward windows
(`backtest.train_window_years` / `oos_window_months` / `step_months`) and
recovery-model knobs (`strategy.recovery.min_recovery_prob` / `p_quantile` /
`profit_margin`), temporarily writes it to `config.yaml`, rebuilds and runs
`rebound_sim_include_held`, appends the `(config, result)` pair to
`reports/tuning/rebound_include_held_search.jsonl` (gitignored), then
**always restores the real `config.yaml`** — verified to hold even under a
mid-run interrupt. Broker fees and the liquidity cap are deliberately
excluded from the search: they're real fixed costs, not a strategy choice.
Re-run it as many times as you want to accumulate trials.

`run_config_suggest.bat` runs `scripts/rebound_config_suggest.py`: reads
that accumulated JSONL and does a per-knob marginal analysis — since the
tuner samples all 6 knobs independently and jointly on every trial, grouping
(categorical knobs) or correlating (continuous knobs) one at a time against
`annualized_IRR` is a legitimate signal, not just eyeballing the single best
row. Prints a suggested config + caveats (single fixed backtest window,
correlational not causal, small sample sizes) to the terminal. Purely
read-only — no files written, `config.yaml` untouched. Needs at least 5
trials to say anything.

`self_correct_prompt.md` runs this suggest script as part of its own
self-correction pass (Step 6a) as one input into its `config.yaml` edit
proposals.

## Universe coverage, cache, and `--warm-only`

Every run covers the **entire universe** (all of HOSE + HNX + UPCOM, ~1,760
tickers) with no time cap. The fetcher bypasses vnstock's 20/min guest quota and
self-throttles to `data.api_per_min` (default 60). Expect ~30 minutes the first
time; subsequent runs are near-instant because up-to-date tickers cost 0 API calls.

`--warm-only` is tri-state, default `yes`:

| `--warm-only` | warm | stale | cold (no parquet) | use when |
| ------------- | ---- | ----- | ----------------- | -------- |
| **`yes`** (smart lazy) | skip | fetch new bar | fetch full history | every-day usage |
| **`always`** (offline) | keep | keep | drop | guaranteed zero API calls |
| **`no`** (force) | refetch | refetch | refetch | backfill / corrections |

Each run prints a cache audit before any fetch. The fetcher knows when the broker
has published a new end-of-day bar (from the wall clock + the cached trading-day
calendar) and skips the network entirely when the cache is current. Stuck tickers
(delisted / halted / feed-absent) get a per-symbol watermark so they stop
retrying every run. Every parquet write is atomic (`os.replace`), so Ctrl+C never
corrupts the cache.

## Trading-day calendar & the T+2 sell floor

Vietnamese settlement is **T+2**: a bought share can't be sold before two trading
days have passed. The evaluator therefore only looks for a profitable exit from
T+2 onward. The calendar is built from the actual cached OHLCV index, so weekends
and Vietnamese holidays (Tết, Reunification Day, Labor Day, National Day, ad-hoc
closures) are skipped — see
[`tracking._next_trading_offset`](src/stockpredict/tracking.py) and
[`tests/test_trading_calendar.py`](tests/test_trading_calendar.py).

**After-14:30 cutoff:** runs at/before 14:30 stamp T+0 = today; after 14:30 (close
locked in) T+0 = the next trading day. Handled at CLI entry via
[`tracking.effective_today_for_trading`](src/stockpredict/tracking.py).

## Same-day re-runs: distinct params → distinct files

Every artifact is suffixed with a **run signature** capturing the pick-affecting
parameters (`mode[_HOSE][_noETF][_x<TICKERS>]`), and that signature is
the ledger `run_id` base. Re-running the same parameters overrides (idempotent);
different parameters never clobber each other.

## Self-correction (Claude mode)

Day-to-day runs don't see past performance — each scores on today's evidence. To
act on history, paste [`self_correct_prompt.md`](self_correct_prompt.md) into
Claude Code with a picks report (it first runs `evaluate` to refresh outcomes, then
works even before picks fully resolve). It diagnoses four things and proposes one
approval-gated edit per finding:

1. **Recovery-filter calibration** — did picks actually bounce at the predicted
   `recovery_prob`? If not, tighten `min_recovery_prob` or the downtrend gate.
2. **P/N accuracy** — realized days/profit vs predicted; lever = `state_buckets` /
   `p_quantile` (empirical, mostly self-correcting — high bar to touch).
3. **Falling-knife check** — unrecovered / long-open picks; chart-check them and,
   for claude/gemini, tighten the DROP guidance in `claude_prompt.md`.
4. **Pred_days calibration (checkpoint misses)** — picks that reached their
   predicted `pred_days` without hitting target; same `state_buckets` / `p_quantile`
   lever as #2, but caught earlier — before final resolution.

Every diff is shown and applied only after per-file approval.

## Setup (one-time)

```bash
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -e ".[dev,llm]"
```

Optional API keys live in [`.env.example`](.env.example); copy to `.env`.

## Tests

```bash
.venv\Scripts\python -m pytest -q
```

131 tests, all synthetic — no network. Coverage spans the downtrend filter,
recovery targets + Kaplan-Meier estimator, P/N ranking + healthy gate, recovery
pricing, the ledger + flexible-exit evaluator, the walk-forward backtest, the
LLM-overlay finalize, the trading calendar, cache freshness + watermarks, and the
rate limiter.

## Configuration

All knobs in [`config.yaml`](config.yaml). Key rebound knobs:

- `strategy.downtrend.{mom20_max, high_prox_max, rsi_floor, rsi_ceil}` — the
  downtrend candidate gate (widen to surface more names).
- `strategy.recovery.min_recovery_prob` (0.85) — the **healthy filter**; higher =
  only more-reliable bouncers.
- `strategy.recovery.{profit_margin, p_quantile, label_max_horizon,
  min_ticker_obs, min_bucket_obs, state_buckets}` — the recovery estimator shape.
- `pricing.default_picks` (1) — how many picks (or `--picks N`).
- `pricing.overbought_rsi_max` (0 = off), `pricing.ceiling_limits` / `ceiling_tol`,
  `pricing.corp_action_lookback` (20) — universe hygiene gates.
- `pricing.max_participation_pct` (1.0) — advisory `suggested_max_units` cap.
- `broker:` — ACBS fee model. `backtest:` — walk-forward windows + `cost_bps`.
- `modes.{claude,gemini}.news_weight` — how much the LLM's `news_score` nudges the
  P/N score.

There is **no** stop-loss or time-cap knob — the strategy holds until profit.

## Layout

```
<repo root>/
  predict_base.bat         double-click: model-only picks
  predict_gemini.bat       double-click: model + Gemini vetting prompt
  claude_prompt.md         paste into Claude Desktop for model + Claude vetting
  self_correct_prompt.md   paste into Claude Code to diagnose + tune the program
  evaluate.bat             double-click: stamp recovered picks
  config.yaml              all tunable knobs

  src/stockpredict/
    cli.py                 click entry point
    selector.py            curated bluechip list + universe top-up
    filters.py             liquidity / ceiling / corp-action / downtrend gates
    dataset.py             feature + recovery-target panel builder
    tracking.py            ledger + flexible-exit evaluator
    data/                  vnstock wrappers (+ vnai quota bypass) + parquet cache + rate limiter
    features/              technical + microstructure indicators
    model/
      target.py            recovery episode labeling + resolve_exit
      train.py             Kaplan-Meier recovery estimator
      predict.py           downtrend filter + P/N ranking
    pricing.py             recovery pricing (buy=close, target from P) + fee model
    backtest/              flexible-exit walk-forward simulator
    news/                  claude / claude-llm plan builders + gemini prompt/response
    modes/                 base / claude / gemini orchestrators

  scripts/                 portfolio simulators (realistic money-weighted P&L)
  tests/                   pytest, synthetic data only
  cache/                   OHLCV parquet cache, predictions ledger
  models/                  saved recovery estimator (recovery_latest.pkl)
  reports/                 picks, plans, prompts, backtest reports
```

## License

- **This code** is **AGPL-3.0-or-later** ([`LICENSE`](LICENSE)). Forks stay
  open-source under the same terms; running a modified version as a network
  service obliges you to publish your modifications.
- **vnstock**, the runtime data dependency, is **non-commercial / personal-research
  only** under its own license — see [`NOTICE`](NOTICE). The AGPL here does not
  override that; commercial use requires a separate vnstock license.
- **Contributions** are accepted under AGPL-3.0 with a DCO sign-off; see
  [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Caveats

- The **20 req/min "guest" cap is vnstock's own client-side quota**, not the
  providers' server limit; `data.bypass_vnai_quota: true` calls the underlying
  endpoint directly, leaving only our own `data.api_per_min` throttle. A genuine
  429 still triggers a pause; failed ticker fetches never abort the run.
- Backtests carry mild **label lookahead**; the walk-forward backtest assumes
  fills at the close, and even the realistic simulator models the sell at the
  recovery day's close — so live results run somewhat lower than simulated.
- With no stop and no cap, a rare broken name that slips past the healthy filter
  is held indefinitely — the strategy wins often and fast, but warehouses an
  underwater tail of unsold names. Monitor and prune those manually.
- **Not investment advice.** Use at your own risk; past performance ≠ future results.
```
