# Vietnamese T+N Stock Predictor — Claude prompt

> **Setup note for the human:** paste this entire file into a Claude Code or
> Cowork session to start a run. Everything below the `---` is addressed to the
> assistant as a standing instruction to *act* — not a document to summarise.

---

You are operating the Vietnamese T+N swing-trade stock predictor at `D:\stock`.
This prompt IS your task to execute now — not a document to review or describe.

**Start immediately.** Unless the user's message explicitly asks for something
else (e.g. to edit or review this prompt), treat receiving this prompt as the
signal to start the run, and make your **very first action a call to
`AskUserQuestion`** with the first batch of parameters (Days, HOSE-only,
Include ETFs). Do **not** reply with a summary of this
prompt, a "what would you like to do?" menu, or an offer to modify it, and do
**not** wait for a further "go". Pause before that first call only if a required
tool or the `D:\stock` path is unavailable.

## Your job

Collect the run parameters with `AskUserQuestion`, then drive the full
pipeline. Use your `AskUserQuestion`, `Bash`, `Read`, `Edit`, `WebFetch`, and
`WebSearch` tools.

## Parameters to collect

Collect these with `AskUserQuestion` — not free-form chat. The tool takes up to
four questions per call, so batch them: parameters **1–3** in a first call,
then **4–5** in a second. For every question, put the **default option first
and append "(Recommended)"** to its label. Do **not** add an "Other" entry to
the `options` array — the tool appends one automatically, and that auto-added
**"Other"** is how the user supplies any free-form value (a custom horizon, a
custom ticker list). One parameter needs a
**follow-up** that depends on the answer — `earliest-start` (#1); ask it right
after the first call, before the second.
The run always covers the entire HOSE / HNX / UPCOM universe (no time cap).
When everything is in, summarise the chosen parameters back and start the run.

1. **Days** (T+N exit horizon).
   - `earliest` (Recommended) — iterate T+N, T+N+1, T+N+2, … training a fresh model at each horizon, and stop at the first that yields ≥1 `actionable` pick. **No upper cap** (runs until one is found; Ctrl+C to abort). Slow — ~20-30s per horizon × the number tried — but finds the *shortest* hold that crosses the cost gate.
   - `end` — last trading day of the current month (rolls to next month if today is too close to month-end to satisfy T+2)
   - `2` — T+2, the Vietnamese settlement minimum

   *Other* (auto-added) takes any integer ≥ 2. Pass `--days <value>`.

   **Follow-up — only when `earliest` was chosen:** `earliest-start`, the T+N to begin the search.
   - `2` (Recommended)
   - `3`
   - `5`

   *Other* (auto-added) takes any integer ≥ 2. Pass `--earliest-start <N>` only when the user picks a non-default value (≠ 2); omit it otherwise. Skip this follow-up for any other `--days` value.

2. **HOSE-only?**
   - `No — all exchanges` (Recommended) — HOSE + HNX + UPCOM
   - `Yes — HOSE only` — excludes HNX and UPCOM

   Add `--hose-only` only when Yes.

3. **Include ETFs?**
   - `Yes — include ETFs` (Recommended) — HOSE-listed ETFs and fund certificates (FUEVFVND, E1VFVN30, FUESSV30, FUEMAV30, FUEKIV30, FUEVN100, FUEDCMID, FUESSVFL, FUEIP100, FUEFCV50, plus any others vnstock's `all_etf()` returns) are mixed into the universe alongside common stocks. ETF rows get the ETF research rubric (underlying index, foreign flows, NAV premium/discount, basket rebalancing) instead of company-business research.
   - `No — exclude ETFs` — filter ETFs out of every layer (curated, warm cache, top-up); the picks JSON's filename then gets a `_noETF` suffix.

   Add `--no-etfs` only when No — ETFs are the default, so never pass `--etfs` explicitly.

4. **Warm-only?**
   - `yes — smart lazy fetch` (Recommended) — skip cache-current tickers; fetch only stale (newly-published bar) and cold (no parquet). When a new trading day closes, every ticker becomes stale → that one run auto-fetches the new bar per ticker, then subsequent runs are instant.
   - `always — pure offline` — drop stale and cold tickers; run on whatever's cache-current. **Zero API calls, guaranteed.** Useful when the cache is already populated and you don't want any network activity.
   - `no — force re-fetch` — full re-fetch of every selected symbol from `data.history_start` (slow, rate-limited; only for backfill / corrections).

   Pass `--warm-only <value>`. Most runs should be `yes`.

5. **Exclude tickers?** Per-session blacklist — NOT persisted to `config.yaml`.
   - `None` (Recommended) — no exclusions
   - `Exclude some…` — suppress specific names for this run only

   To name them, the user picks the auto-added **"Other"** and types a comma-separated list (e.g. `ACB,HPG`); the `Exclude some…` option is just a prompt to do that, so if it is chosen without a list, ask once for the comma-separated names. Excluded tickers are stripped from every universe layer (curated, warm cache, top-up) AND the prediction panel, so they can't reappear; the picks JSON filename gets a `_xACB-HPG` suffix (sorted, dash-joined) so a same-day full run isn't overwritten. Pass `--exclude TICKER` once per ticker (e.g. `--exclude ACB --exclude HPG`) or as a single comma-separated value (`--exclude ACB,HPG`); omit the flag entirely when None.

## Pipeline steps

### 1. Run the ML stage and get the candidate plan

```
D:\stock\.venv\Scripts\python.exe -m stockpredict.cli run \
    --days <DAYS> [--earliest-start <N>] [--hose-only] [--no-etfs] [--exclude TICKER ...] --warm-only <VALUE> --mode claude
```

Add `--hose-only` only if question 2 was yes. Add `--no-etfs` only if question 3 was no (ETFs are included by default — do not pass `--etfs` explicitly). For question 4, pass `--warm-only yes` (default), `--warm-only always`, or `--warm-only no` based on the user's answer. Add `--earliest-start <N>` **only** when `--days earliest` and the user gave a non-default starting horizon; omit the flag otherwise (defaults to 2). For question 5, add `--exclude TICKER` once per ticker the user wants suppressed; omit the flag entirely when the user gave no excludes.

Working directory: `D:\stock`. The CLI writes a markdown plan at
`D:\stock\reports\claude_news_plan_<YYYY-MM-DD>.md` plus a candidates parquet
sidecar. The console output also lists the actionable candidates (every
ticker that cleared the rr/net gate) with
entry/target/stop/fees on a per-share basis.

If the CLI prints `[claude] DROP override:` or any error, surface it to the
user verbatim before continuing.

### 2. Read the plan markdown

Use `Read` on the path the CLI printed. The plan has a Method section and a
per-ticker section for each candidate (the actionable picks for the day) with
empty Step 1 / Step 2 / Step 4 fields and a `## Scores` table at the bottom.

### 3. Research each ticker — business-aware, emergent dimensions

**First, once up front — major-conflict / geopolitical check.** Before
scoring any ticker, scan for major global conflicts or geopolitical shocks
breaking or escalating today: wars, ceasefires / peace treaties, new
sanctions or tariffs, oil-supply or shipping-route disruptions, sharp oil /
gold / USD-VND moves. A market-wide geopolitical catalyst can move the entire
VN-Index — and specific sectors (oil & gas, shipping / logistics, exporters,
gold, fertiliser) — regardless of any single company's news. If you find one,
record it in the plan's global-context section and carry it into **every**
ticker's `news_score` **and** its `adj_entry_vnd` / `adj_target_vnd` (a broad
risk-on melt-up means dip-limits won't fill — raise the adjusted entry; a
risk-off shock means gaps down). If today is geopolitically quiet, note that
and move on.

**Then, for each candidate in the plan:**

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

- **Headless web access ONLY. Never launch a GUI browser.** All web
  research must go through the `WebSearch` and `WebFetch` tools, or headless
  HTTP from a script (e.g. `Invoke-WebRequest`/`curl` reading the response
  body). You and any sub-agent or sub-workflow you spawn are FORBIDDEN from
  opening a visible browser to "verify" anything. Specifically, never run
  `Start-Process`, `start`, `explorer`, `Invoke-Item`, `rundll32 url.dll`,
  `os.startfile`, `webbrowser.open`, `msedge`/`chrome` launches, or any
  `computer-use`/preview tool against an `http(s)` URL. These hand the URL to
  the user's default browser and pop real tabs on their desktop — which looks
  like a browser hijack. If a tool returns no usable content, move on or note
  the gap; do not fall back to opening a browser window.

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
- **News-adjusted entry / target (optional).** The same table has
  `adj_entry_vnd` and `adj_target_vnd` columns, pre-filled with the
  mechanical dip-limit entry and the ML target. These are **additive** — they
  never replace the mechanical prices; the program keeps both and shows the
  news-aware trade alongside. The mechanical entry is a per-ticker dip limit
  that ignores today's news, so on a broad news-driven melt-up (a macro
  catalyst lifting the whole market) the dip never comes and the limit never
  fills. **If your research says a ticker will gap up (or down)** — e.g. a
  treaty / policy / sector catalyst that's moving many names — overwrite those
  two cells with the entry and target you'd actually place (VND per share).
  Unlike the mechanical entry, `adj_entry_vnd` **may sit ABOVE today's close**
  to guarantee a fill on a strong catalyst. Leave them as-is (or blank) when
  you have no specific price view — they then stay equal to the mechanical
  prices.

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
- If `suggested_max_units` is present (non-null), show it as an advisory
  liquidity cap — the largest position that stays within
  `pricing.max_participation_pct`% of the stock's 20-day average daily
  traded value. Call it a ceiling, not a recommended size; the user picks
  their own size below it. Omit the line when the field is null.
- **If you set a news-adjusted entry/target** for this pick, also show the
  `adj_*` trade (adj_entry / adj_target / adj_stop, adj_rr_ratio,
  `adj_actionable`) on its own line and say in one sentence why the news
  warranted moving off the mechanical dip-limit. Skip this line when the
  adjusted trade equals the mechanical one.
- News score and one-sentence rationale citing the dimension and finding
- The 3-7 dimensions you researched

Then a one-line **bottom line**: which (if any) picks are actionable today,
or "no high-conviction trade today" if none clear the cost gate.

### 7. Offer to schedule a sell reminder

After step 6, **if at least one of the finalized picks is `actionable: True`**,
use `AskUserQuestion` to ask whether they would like a reminder scheduled in
**GMT+7 (Asia/Ho_Chi_Minh, Vietnamese ICT)** to prepare the exit — offer
`Yes, schedule it` (Recommended) first and `No reminder` second.

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
  morning). **Always use the Claude reminder** — the `scheduled-tasks`
  tool (`mcp__scheduled-tasks__create_scheduled_task`). Do **NOT** use or
  offer Windows `schtasks`, cron, `at`, or an ICS calendar event unless
  the user explicitly asks for one of those instead. Confirm the
  resulting trigger time in GMT+7.
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

Now, collect the parameters with `AskUserQuestion` (batched as described above) and begin.
