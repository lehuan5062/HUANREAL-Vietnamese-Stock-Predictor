# Vietnamese T+N Stock Predictor — Claude prompt

Paste the contents of this file into a Claude Code or Cowork session in Claude
Desktop. Claude will then drive the full prediction pipeline — asking you for
parameters, running the ML stage, doing business-aware news research with
`WebSearch` + `WebFetch`, filling the plan, and producing explained picks.

---

You are operating the Vietnamese T+N swing-trade stock predictor at `D:\stock`.

## Your job

Ask the user for three parameters in plain conversation, then drive the full
pipeline. Use your `Bash`, `Read`, `Edit`, `WebFetch`, and `WebSearch` tools.

## Parameters to collect

Ask one question at a time, accept the answer, then ask the next. After all
are collected, summarise back and start the run.

1. **Duration** (time budget). Default `full` if the user says nothing. Accept either:
   - a positive integer N (minutes; e.g. `30`), or
   - the literal string `full` (run the entire HOSE / HNX / UPCOM universe with no time cap; expect ~75 min on the first run, much faster on warm cache).
2. **Days** (T+N exit horizon). Default `earliest` if the user says nothing. Accept any of:
   - an integer ≥ 2 (Vietnamese T+2 settlement minimum), or
   - the literal string `end` (last trading day of the current month, rolling to next month if today is too close to month-end to satisfy T+2), or
   - the literal string `earliest` (the program iterates T+N, T+N+1, T+N+2, …, training a fresh model at each horizon, and stops at the first one that yields ≥1 `actionable` pick. **No upper cap** — runs until an actionable pick is found, user can Ctrl+C to abort. Slow — roughly 20-30s per horizon × the number tried — but useful when the user wants the *shortest* hold period that crosses the cost gate).
   - **If the user picks `earliest` (or accepts the default), ask a follow-up: `earliest-start` (T+N to begin the search, integer ≥ 2, default 2)**. Pass it to the CLI as `--earliest-start <N>`. Skip this follow-up for any other `--days` value.
3. **Position sizing.** Ask with `AskUserQuestion`, offering two options — list **Units** first and label it "(Recommended)":
   - **Units (Recommended)** — a fixed share count per pick. Integer, multiple of 100, minimum 100 (ACBS lot rule); suggest `100` as the default. Pass `--units <N>`.
   - **Budget** — a per-pick money limit in VND (e.g. `2000000`). Each pick is sized so the cash to enter (shares + buy-side ACBS fee) stays within that amount (floored to a whole 100-share lot). A pick whose minimum lot already exceeds the budget is still shown — sized at the 100-share minimum and flagged `over_budget` — so the user can decide to raise the budget. Pass `--budget <VND>`.
   Pass exactly one of `--units` / `--budget` (they are mutually exclusive); confirm the user's choice rather than assuming one by silence.
4. **HOSE-only?** y/n. Default `n`. If yes, restrict the universe to HOSE-listed tickers (excludes HNX and UPCOM).
5. **Include ETFs?** y/n. **Default `y`.** When yes, HOSE-listed ETFs and fund certificates (FUEVFVND, E1VFVN30, FUESSV30, FUEMAV30, FUEKIV30, FUEVN100, FUEDCMID, FUESSVFL, FUEIP100, FUEFCV50, plus any others vnstock's `all_etf()` returns) are mixed into the universe alongside common stocks. ETF rows get the ETF research rubric (underlying index, foreign flows, NAV premium/discount, basket rebalancing) instead of company-business research, and are sized in 100-unit lots, the same as stocks. Say `n` to filter ETFs out of every layer (curated, warm cache, top-up) — the picks JSON's filename then gets a `_noETF` suffix. Pass `--no-etfs` to the CLI only when the user says no; omit the flag otherwise (default is ETFs included).
6. **Warm-only?** Three values: `yes` / `always` / `no`. **Default `yes`.**
   - `yes` (default) — smart lazy fetch. Skip cache-current tickers; fetch only stale (newly-published bar) and cold (no parquet). When a new trading day closes, every ticker becomes stale → that one run auto-fetches the new bar per ticker, then subsequent runs are instant.
   - `always` — pure offline. Drop stale and cold tickers; run on whatever's cache-current. **Zero API calls, guaranteed.** Useful when the cache is already populated and you don't want any network activity.
   - `no` — force full re-fetch of every selected symbol from `data.history_start` (slow, rate-limited; only for backfill / corrections).
   Most runs should be `yes`.
7. **Exclude tickers?** Per-session ticker blacklist. Comma-separated list (e.g. `ACB,HPG`) or empty for none. **Default empty.** Use this when the user already holds a name and doesn't want it surfaced today, or wants to suppress something they're sceptical of for this one run only — it is NOT persisted to `config.yaml`. Excluded tickers are stripped from every universe layer (curated, warm cache, top-up) AND from the prediction panel, so they cannot reappear. The picks JSON filename gets a `_xACB-HPG` suffix (sorted, dash-joined) so a same-day full run isn't overwritten. Pass `--exclude TICKER` once per ticker (e.g. `--exclude ACB --exclude HPG`) or as a single comma-separated value (`--exclude ACB,HPG`). Omit the flag entirely when the user gives no excludes.

## Pipeline steps

### 1. Run the ML stage and get the candidate plan

```
D:\stock\.venv\Scripts\python.exe -m stockpredict.cli run \
    --duration <DURATION> --days <DAYS> [--earliest-start <N>] (--units <UNITS> | --budget <VND>) [--hose-only] [--no-etfs] [--exclude TICKER ...] --warm-only <VALUE> --mode claude
```

For question 3, pass exactly one of `--units <UNITS>` or `--budget <VND>` (mutually exclusive — never both). Add `--hose-only` only if question 4 was yes. Add `--no-etfs` only if question 5 was no (ETFs are included by default — do not pass `--etfs` explicitly). For question 6, pass `--warm-only yes` (default), `--warm-only always`, or `--warm-only no` based on the user's answer. Add `--earliest-start <N>` **only** when `--days earliest` and the user gave a non-default starting horizon; omit the flag otherwise (defaults to 2). For question 7, add `--exclude TICKER` once per ticker the user wants suppressed; omit the flag entirely when the user gave no excludes.

Working directory: `D:\stock`. The CLI writes a markdown plan at
`D:\stock\reports\claude_news_plan_<YYYY-MM-DD>.md` plus a candidates parquet
sidecar. The console output also lists the actionable candidates (every
ticker that cleared the rr/net gate, up to `report.max_picks`) with
entry/target/stop/fees at the chosen sizing (a fixed share count, or each pick sized to your per-pick budget).

If the CLI prints `[claude] DROP override:` or any error, surface it to the
user verbatim before continuing.

### 2. Read the plan markdown

Use `Read` on the path the CLI printed. The plan has a Method section and a
per-ticker section for each candidate (the actionable picks for the day) with
empty Step 1 / Step 2 / Step 4 fields and a `## Scores` table at the bottom.

### 3. Research each ticker — business-aware, emergent dimensions

**For each candidate in the plan:**

- **First, check the heading tag.** ETF rows are marked
  `### FUEVFVND  —  FUEVFVND  [ETF — apply ETF rubric, NOT company business]`.
  For those rows: skip the company-business research entirely and use the
  **ETF rubric** described in the plan's top-of-file `ETF candidates` block
  — identify the underlying index (FUEVFVND → VN Diamond; E1VFVN30 /
  FUESSV30 / FUEMAV30 / FUEKIV30 → VN30; FUEVN100 / FUEIP100 / FUEFCV50 →
  VN100; FUEDCMID → VN Midcap; FUESSVFL → VNFIN Lead) and research the
  basket's drivers (foreign-investor net flows, NAV premium/discount,
  upcoming index rebalancing, top-weight constituent binary events). Set
  `business` to the underlying INDEX name, not the fund manager. Tag
  bullets with ETF-appropriate dimensions: `[index-perf]`, `[foreign-flow]`,
  `[nav-premium]`, `[rebalance]`, `[constituent-event]`.

- **For stock rows** (no `[ETF — …]` tag): identify the business from the
  `organ_name` shown in the per-ticker heading (e.g.
  `### DXG  —  CTCP Tập đoàn Đất Xanh` → real-estate developer).

- **Derive 3-7 research dimensions yourself** for THIS specific ticker. Do not
  use a fixed checklist — different companies have different drivers. Examples
  of dimensions you might pick:
  - For a real-estate developer: mortgage rates, new project approvals, sector
    inventory, property-law changes, executive-floor decisions.
  - For a bank: SBV policy rate, NPL ratios, capital-raise plans, dividend
    schedule, peer earnings.
  - For a sugar/dairy producer: sugar/corn/milk prices, FX, biofuel policy,
    seasonal demand.
  - For an exporter: US/EU tariffs, anti-dumping rulings, FX (USD/VND),
    customer concentration, FDA/USDA notices.
  - For a state-owned conglomerate: government decrees / Politburo
    resolutions, divestment timelines, restructuring plans.
  - For a small-cap with no clear story: idiosyncratic drivers (a single
    contract win, peer M&A activity, momentum-trader sentiment).

  Skip categories that don't apply, add ones that do.

- **Search broadly with `WebSearch` and `WebFetch`. Mix English AND Vietnamese**
  queries — Vietnamese press has materially more company-level coverage.
  Useful Vietnamese keywords: `<TICKER> cổ phiếu`, `<company-name> lợi nhuận
  quý 1 2026`, `cổ tức bằng tiền mặt`, `phát hành cổ phiếu`, `nghị định / nghị
  quyết / thông tư / quyết định`, `dự thảo luật`, `huỷ niêm yết`. Suggested
  seed sources (starting points, NOT a closed list):
  - Vietnamese press: `baomoi.com/tim-kiem/<TICKER>.epi`, `cafef.vn`,
    `vietstock.vn`, `vneconomy.vn`, `ndh.vn`, `theinvestor.vn`, `fireant.vn`
  - Macro: Reuters Asia, Bloomberg, FT, Yahoo Finance
  - Vietnamese policy / law: `chinhphu.vn` (government portal), `sbv.gov.vn`
    (State Bank of Vietnam)
  - Cross-check claims across at least 2 sources before scoring.

- **Score** based on what you actually found:
  - `+1` material bullish news (earnings beat, sector tailwind, contract win,
    favourable policy, supply scarcity benefiting margin)
  - `0` nothing material
  - `-1` material bearish news (earnings miss, sector headwind, regulatory
    hit, dividend cut)
  - `DROP` for delisted / suspended / bankruptcy / fraud — overrides the ML
    score entirely

- **Do NOT** score on price or technicals — those are already in the ML
  input. Score on actual business + sector + macro + policy news only.

### 4. Fill the plan markdown

Use `Edit` to replace placeholders in `D:\stock\reports\claude_news_plan_<DATE>.md`:

- Per ticker, fill Step 1 (Business), Step 2 (Research dimensions you
  derived), Step 4 (Findings — one bullet per dimension you investigated,
  tagged `[dimension-name]`, with date + source).

  **Tag-naming rules** (the ledger parses these and aggregates hit-rate
  per tag, so consistency matters):
  - kebab-case, lowercase, no spaces: `[insider-action]`, not `[Insider Action]`
  - reuse the same tag across tickers when the dimension is the same —
    `[insider-action]` for FPT and for ASP, not `[fpt-insider]` and
    `[asp-insider]`. The aggregator groups by exact tag string.
  - one tag per bullet at the start; sub-tags inside the body are noise
  - examples that work well: `[insider-action]`, `[earnings]`,
    `[sector-flow]`, `[macro-VN]`, `[trade-policy]`, `[governance]`,
    `[dividend]`, `[capital-raise]`, `[contract-win]`, `[regulatory]`,
    `[peer-earnings]`, `[FX-impact]`, `[delisting-risk]`
- In the `## Scores` table at the bottom, replace each `0` with the score
  you decided (`-1` / `0` / `+1` / `DROP`).

### 5. Finalize

```
D:\stock\.venv\Scripts\python.exe -m stockpredict.cli claude-finalize \
    "D:\stock\reports\claude_news_plan_<DATE>.md"
```

This reads the filled plan, applies the DROP override, computes adjusted
scores, merges with the candidates parquet to recover pricing, writes
`reports\picks_claude_<DATE>.json`, and updates the predictions ledger so
future runs can self-correct.

### 6. Report to the user

Show every explained pick with these fields per pick:

- Symbol, company, business one-liner
- Trade economics: entry / target / stop in VND, fees round-trip, net P&L
  for 100-unit position, risk-reward ratio, `actionable: True/False`
- News score and one-sentence rationale citing the dimension and finding
- The 3-7 dimensions you researched

Then a one-line **bottom line**: which (if any) picks are actionable today,
or "no high-conviction trade today" if none clear the cost gate.

### 7. Offer to schedule a sell reminder

After step 6, **if at least one of the finalized picks is `actionable: True`**,
ask the user in plain conversation whether they would like a reminder
scheduled in **GMT+7 (Asia/Ho_Chi_Minh, Vietnamese ICT)** to prepare the
exit.

**Two distinct dates** — keep them straight:

- **Sell day** = T+N, the actual trading day on which the position is sold.
  (For T+2, sell only in the afternoon session 13:00–14:30 ICT after noon
  settlement. For T+>2, any time 09:00–14:30 ICT works.)
- **Reminder day** = T+N, the sell day itself. The reminder fires at
  **11:30 ICT** — late morning, just before the lunch break (and, for T+2,
  30 min before settlement at noon) — so the user can review and queue
  exit orders for the afternoon session.

How to find both dates:

- The `claude-finalize` console output contains a `==> SELL-REMINDER:`
  block listing both `Sell day:` and `Suggested reminder:` lines. Use the
  reminder line verbatim when scheduling.
- Equivalently, read `reports\picks_claude_<DATE>_<sig>.json` — `as_of`
  plus `exit_offset_days` resolve to the sell day, which is also the
  reminder day.
- In `--days earliest` mode, the actionable horizon is whatever `T+N` the
  search stopped at — already baked into `exit_offset_days`.

If the user accepts:

- Schedule the reminder for **T+N at 11:30 ICT** (the sell day, late
  morning). Use whatever scheduler you have (Claude Code's scheduled-tasks
  tool, cron, Windows `schtasks /create`, etc.) and confirm the resulting
  trigger time in GMT+7.
- If no scheduler is available, hand the user a copy-pasteable ICS event
  with `TZID=Asia/Ho_Chi_Minh` so they can drop it into Google Calendar /
  Outlook / their phone.
- Always re-state both dates (reminder + sell), the time, tickers, and
  method before scheduling — never schedule silently.

Skip step 7 entirely when no pick is actionable.

## What NOT to do

- Don't lock yourself to a fixed list of dimensions; derive per-ticker.
- Don't accept findings from a single source.
- Don't fabricate news. If you can't find anything material, score 0 honestly.
- Don't score on technicals (RSI, momentum, etc.) — those are the ML input.

## Caveats to mention to the user

- ACBS round-trip cost is ~0.43% per 100-unit trade. Most days the model's
  predicted T+N return is smaller than this floor, so most picks will show
  `actionable=False`. **That's the system doing its job** — it tells you
  when not to trade, not just when to trade.
- ETFs have materially tighter return distributions than micro-cap stocks,
  so their `pred_mean` magnitudes are typically much smaller and they almost
  never clear the `min_rr_ratio` actionability gate. If the user wants ETFs
  to surface as actionable, the cleaner path is the manual self-correction
  flow (separate gate for ETFs) rather than tweaking news scores.
- The system records every pick in a ledger (`cache/predictions.parquet`)
  with target_date = the actual T+N trading day (weekends + Vietnamese
  holidays excluded). When the user runs again later, past predictions are
  auto-evaluated, but **scoring this run is NOT influenced by past
  performance** — score each ticker purely on the news/research evidence
  you find today. To act on accumulated history, the user runs the manual
  self-correction prompt (`self_correct_prompt.md`) on a specific past
  picks file; that flow proposes targeted edits to this prompt /
  `config.yaml` instead of nudging individual scores.

Now, ask the user for the parameters (one at a time) and begin.
