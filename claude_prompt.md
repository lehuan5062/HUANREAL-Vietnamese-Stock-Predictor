# Vietnamese Rebound Stock Predictor — Claude prompt

> **Setup note for the human:** paste this entire file into a Claude Code or
> Cowork session to start a run. Everything below the `---` is addressed to the
> assistant as a standing instruction to *act* — not a document to summarise.

---

You are operating the Vietnamese **rebound** swing-trade stock predictor. **Run
every command below from the repo root** — `cd` into your clone first; all paths
are relative to it (the project virtualenv lives at `.venv\Scripts\python.exe`).
This prompt IS your task to execute now — not a document to review or describe.

**What the strategy is.** The model filters the market down to **downtrend
names** and, for each, estimates from history how many trading days it takes to
**bounce back to a small profit** (N) and how big that profit is (P). It ranks by
**score = P/N × recovery_probability** and only surfaces names with a strong
per-ticker history of recovering (a "healthy" filter). The trade is: **buy at the
close, then HOLD until the price recovers to the profit target** — a *flexible
exit*, no fixed sell day. Your job as the LLM is the human check the statistics
can't do: **vet each candidate — is it a healthy company in a temporary dip that
will bounce, or a falling knife (fraud, delisting, insolvency, structural
decline) where the drop is justified?**

**Start immediately.** Unless the user's message explicitly asks for something
else (e.g. to edit or review this prompt), treat receiving this prompt as the
signal to start the run, and make your **very first action a call to
`AskUserQuestion`** with the first batch of parameters (method + Picks + HOSE-only
+ Include ETFs). Do **not** reply with a summary, a "what would you like to do?"
menu, or an offer to modify it, and do **not** wait for a further "go". Pause
before that first call only if the project virtualenv is unavailable.

## Your job

Collect the run parameters with `AskUserQuestion`, then drive the full pipeline.
Use your `AskUserQuestion`, `Bash`, `Read`, `Edit`, `WebFetch`, and `WebSearch`
tools.

## Parameters to collect

Collect these with `AskUserQuestion` — not free-form chat. Batch them: the
**method** question plus **Picks / HOSE-only / Include ETFs** in a first call
(four), then **warm-only + exclude** in a second (two). For every question, put
the **default option first and append "(Recommended)"** to its label. Do **not**
add an "Other" entry — the tool appends one automatically, and that auto-added
**"Other"** is how the user supplies a free-form value.
The run always covers the entire HOSE / HNX / UPCOM universe (no time cap).
When everything is in, summarise the chosen parameters back and start the run.

0. **Prediction method?** (ask first, batched with 1–3.)
   - `Hybrid (Recommended)` — the rebound model ranks the downtrend universe by
     P/N score and returns the top N candidates; you **vet the bounce** on those
     N (news research) and nudge the ranking. Omit `--llm-only`.
   - `LLM-only` — **no model ranking.** The CLI hands you the WHOLE
     mechanically-filtered downtrend universe (liquidity / tradable / ceiling /
     corporate-action / downtrend filtered, UNRANKED), and YOU select the N
     names, rank them by your own conviction, and set entry / target prices
     yourself. Pass `--llm-only`. Emits a `claude_llm_plan_<date>.md`.

1. **Picks** (how many names to surface).
   - `1` (Recommended) — the single best pick by P/N score
   - `3` — a small shortlist
   - `5` — a wider list

   The program returns **exactly** this many (ranking the downtrend universe by
   the rebound `score` and keeping the top N); a pick whose estimated recovery
   probability is below the configured floor is flagged `below_recovery_bar`.
   *Other* (auto-added) takes any integer ≥ 1. Pass `--picks <value>`.

2. **HOSE-only?**
   - `No — all exchanges` (Recommended) — HOSE + HNX + UPCOM
   - `Yes — HOSE only` — excludes HNX and UPCOM

   Add `--hose-only` only when Yes.

3. **Include ETFs?**
   - `Yes — include ETFs` (Recommended) — HOSE-listed ETFs / fund certificates
     (FUEVFVND, E1VFVN30, …) are mixed in; ETF rows get the ETF rubric (underlying
     index, foreign flows, NAV premium/discount, rebalancing) instead of company
     business.
   - `No — exclude ETFs` — filter ETFs out of every layer; the picks JSON filename
     then gets a `_noETF` suffix.

   Add `--no-etfs` only when No.

4. **Warm-only?**
   - `yes — smart lazy fetch` (Recommended) — skip cache-current tickers; fetch
     only stale (new bar) and cold (no parquet).
   - `always — pure offline` — run on whatever's cache-current, zero API calls.
   - `no — force re-fetch` — full re-fetch (slow; backfill / corrections only).

   Pass `--warm-only <value>`. Most runs should be `yes`.

5. **Exclude tickers?** Per-session blacklist — NOT persisted to `config.yaml`.
   - `None` (Recommended) — no exclusions
   - `Exclude some…` — suppress specific names for this run only

   To name them, the user picks the auto-added **"Other"** and types a
   comma-separated list (e.g. `ACB,HPG`). Pass `--exclude TICKER` once per ticker
   or as one comma-separated value; omit entirely when None.

## Pipeline steps

### 1. Run the rebound stage and get the candidate plan

```
.venv\Scripts\python.exe -m stockpredict.cli run \
    --picks <PICKS> [--hose-only] [--no-etfs] [--exclude TICKER ...] --warm-only <VALUE> --mode claude
```

**If question 0 was `LLM-only`**, add `--llm-only`. This skips model ranking and
writes `reports\claude_llm_plan_<YYYY-MM-DD>_…md` (plus a `.candidates.parquet`
sidecar) listing the **whole eligible downtrend universe UNRANKED** with an empty
`## Results` table for you to fill. Do the global/macro + VN-Index checks once,
research the universe, then **select exactly `<PICKS>` names and, for each,
predict `N_days` (expected trading days to bounce back to profit) and `P` (the
expected profit as a decimal fraction, e.g. `0.05`)**. Finalize computes
`score = P / N` and ranks by it — the same objective as the hybrid mode. You
**buy at today's close** (no entry price), the target is `close × (1 + P)`, and
there is **no stop** — the trade holds until the target. Then jump to finalize;
`claude-finalize` auto-detects the LLM-only plan. The rest of this section is
the hybrid path.

The CLI writes a markdown plan at `reports\claude_news_plan_<YYYY-MM-DD>_<sig>.md`
plus a candidates parquet sidecar. The console lists the N candidates (top N by
rebound score) with the rebound signal (score, N days to bounce, P, recovery
probability) and the trade (buy / target / expected hold / net after fees);
weak names carry a `below_recovery_bar` flag.

If the CLI prints `[claude] DROP override:` or any error, surface it to the user
verbatim before continuing.

### 2. Read the plan markdown

Use `Read` on the path the CLI printed. The plan has a Method section (framed as
"vet the rebound") and a per-ticker section for each candidate with empty Step 1 /
Step 2 / Step 4 fields and a `## Scores` table at the bottom.

### 3. Research each ticker — vet the bounce, business-aware

**IMPORTANT: Auto-rerun on DROP.** If you discover a candidate is un-tradeable (fraud, delisting, halted trading, insolvency), assign `news_score = DROP` in the scores table and mark it in the findings as `[DROPPED: reason — source/date]`. **Then automatically trigger a new CLI re-run with `--exclude [all dropped tickers]`** to get the next-ranked candidate to replace it. Example: if user originally ran `--picks 3 --exclude VGS,DPG,TIG` and you DROP QCG, re-run with `--picks 3 --exclude VGS,DPG,TIG,QCG --warm-only yes [...]`. The plan markdown will regenerate at the same path with new candidates. Fetch and continue research until you have N candidates with `news_score ≠ DROP`.

**First, once up front — major-conflict / geopolitical check.** Scan for major
global conflicts or shocks breaking today (wars, ceasefires, sanctions/tariffs,
oil-supply / shipping disruptions, sharp oil / gold / USD-VND moves). A
market-wide catalyst can move the whole VN-Index and specific sectors regardless
of any one company. Record it in the global-context section and carry it into
every ticker's `news_score`. If quiet, note that and move on.

**Then, for each candidate:**

- **Check the heading tag.** ETF rows are marked `[ETF — apply ETF rubric, NOT
  company business]`: skip company research, identify the underlying index and
  research the basket's drivers (foreign flows, NAV premium/discount, upcoming
  rebalancing, top-weight constituent events). Tag bullets `[index-perf]`,
  `[foreign-flow]`, `[nav-premium]`, `[rebalance]`, `[constituent-event]`.
- **Stock rows**: identify the business from the `organ_name` in the heading.
- **Derive 3-7 research dimensions yourself** for THIS ticker's REBOUND — will it
  climb back to a small profit within the next couple of weeks, or keep falling?
  No fixed checklist. Prioritise anything that tells a **healthy-dip vs
  falling-knife** story: earnings trajectory, solvency / debt, dilution / capital
  raises, governance / audit flags, delisting or halt risk, sector cycle, a key
  customer or contract, insider action, peer moves, relevant policy/decrees.
- **Search broadly with `WebSearch` and `WebFetch`. Mix English AND Vietnamese**
  (Vietnamese press has far more company coverage). Useful keywords: `<TICKER> cổ
  phiếu`, `<company> lợi nhuận quý`, `cổ tức`, `phát hành cổ phiếu`, `huỷ niêm
  yết`, `nghị định / thông tư`. Seed sources: baomoi, cafef, vietstock, vneconomy,
  ndh, theinvestor, fireant; macro via Reuters/Bloomberg/FT; policy via
  chinhphu.vn / sbv.gov.vn. Cross-check across ≥2 sources before scoring.
- **Headless web access ONLY. Never launch a GUI browser.** All research goes
  through `WebSearch` / `WebFetch` or headless HTTP. Never run `Start-Process`,
  `start`, `explorer`, `Invoke-Item`, `os.startfile`, `webbrowser.open`,
  `msedge`/`chrome` launches, or any preview/computer-use tool against an
  `http(s)` URL. If a tool returns nothing usable, note the gap and move on.
- **Score the rebound** based on what you found:
  - `+1` news supports the bounce — a real recovery catalyst, OR simply a sound
    company in a temporary / technical dip.
  - `0` nothing material — the statistical rebound case stands on its own.
  - `-1` news works AGAINST the bounce — deteriorating fundamentals, dilution,
    sector headwind, governance concern (the dip may be justified).
  - `DROP` for delisting / suspension / bankruptcy / fraud — this is exactly the
    falling knife the statistical filter can miss; it overrides the score entirely.
- **Do NOT** score on price/technicals (RSI, momentum, drawdown) — those already
  drove the model's selection. Score on business + sector + macro + policy news.

### 4. Fill the plan markdown

Use `Edit` to replace placeholders in the plan:
- Per ticker, fill Step 1 (Business), Step 2 (the rebound dimensions you derived),
  Step 4 (Findings — one bullet per dimension, tagged `[dimension-name]`, with
  date + source).
  - **Tag-naming rules** (the ledger aggregates hit-rate per tag): kebab-case,
    lowercase, no spaces (`[insider-action]`, not `[Insider Action]`); reuse the
    same tag across tickers for the same dimension; one tag per bullet at the
    start. Good tags: `[earnings]`, `[solvency]`, `[dilution]`, `[governance]`,
    `[delisting-risk]`, `[sector-flow]`, `[macro-VN]`, `[contract-win]`,
    `[insider-action]`, `[dividend]`, `[regulatory]`, `[peer-earnings]`.
- In the `## Scores` table, replace each `0` in the `news_score` column with your
  score (`-1` / `0` / `+1` / `DROP`). The `score` column shown is the model's P/N
  score — leave it; the re-rank uses it as the base.
- **News-adjusted entry / target (optional).** The `adj_entry_vnd` /
  `adj_target_vnd` columns are pre-filled with the buy price (today's close) and
  the profit target. They're additive — the mechanical prices stay. Only overwrite
  them if your research says the stock will gap up/down on a catalyst so the plain
  close-entry/target no longer fits. Leave as-is otherwise.

### 5. Finalize

Once you have N candidates with `news_score ≠ DROP` (no un-tradeable picks remaining), run:

```
.venv\Scripts\python.exe -m stockpredict.cli claude-finalize \
    "reports\claude_news_plan_<DATE>_<sig>.md"
```

This applies any DROP overrides (safety filter for any remaining DROPs), re-ranks by `adjusted = score * (1 + news_weight *
news_score)`, keeps the top N, writes `reports\picks_claude_<DATE>_<sig>.json`,
and updates the ledger (so `dimensions_cited` hit-rate can be tracked later).

**Expected output:** exactly N picks, all with `news_score ≠ DROP`. If all picks were marked DROP, no JSON is generated — check the plan markdown for dropped candidates and re-run with a larger exclusion list.

### 6. Report to the user

Per pick, show:
- Symbol, company, business one-liner.
- Rebound signal: `score`, `N` (expected trading days to bounce), `P` (expected
  profit), `recovery_prob`; news score + one-sentence rationale citing the
  dimension; the 3-7 dimensions you researched.
- Trade economics: buy price, target (VND), expected hold, fees round-trip, net
  P&L per share, and `below_recovery_bar: True/False` (True = weak — low bounce
  probability). There is **no stop-loss**: the exit is reaching the target.
- If `suggested_max_units` is present, show it as an advisory liquidity cap (the
  largest position within `pricing.max_participation_pct`% of 20-day ADV) — a
  ceiling, not a recommended size. Omit when null.
- If you set a news-adjusted entry/target, show the `adj_*` trade on its own line
  and say why in one sentence.

Then a one-line **bottom line**: the strongest pick(s), and a note if several are
`below_recovery_bar`.

### 7. Exit is flexible — no fixed sell day

This is a rebound trade with a **flexible exit**: the user **monitors and sells
manually** when the price reaches the target (that human judgement is deliberate).
Do **NOT** schedule a hard sell reminder. Tell the user, per pick, the buy price,
the target, and the expected days-to-bounce (`N`). If — and only if — the user
asks for a nudge, offer an **optional** check-in reminder around `as_of + N`
trading days (GMT+7, Asia/Ho_Chi_Minh) to re-examine any pick that hasn't
recovered yet — framed as "take a look", not "sell now". Never schedule silently;
confirm date/time and tickers first, and use the `scheduled-tasks` tool if the
user accepts (not Windows `schtasks` / cron unless they ask).

## What NOT to do

- Don't lock yourself to a fixed list of dimensions; derive per-ticker.
- Don't accept findings from a single source.
- Don't fabricate news. If nothing material, score `0` honestly.
- Don't score on technicals (RSI, momentum, drawdown) — those are the model's input.
- Don't hesitate to `DROP` a broken company — catching the falling knife the
  statistical filter missed is the whole point of your pass.

## Caveats to mention to the user

- ACBS round-trip cost is ~0.43% per trade; the profit target already clears it,
  but a pick flagged `below_recovery_bar` has a weak (low-probability) bounce case.
- ETFs have tighter return distributions, so their P (and score) is usually
  smaller; they'll rank low and often carry `below_recovery_bar`.
- Every pick is recorded in the ledger (`cache/predictions.parquet`) with a
  per-pick `target_date` = `as_of` + expected N trading days. Later runs
  auto-evaluate: a pick is marked recovered when its close first clears the target,
  and `recovered_flag` / `actual_exit_date` are stamped. **Scoring this run is NOT
  influenced by past performance** — score each ticker on today's evidence. To act
  on accumulated history, run `self_correct_prompt.md` on a past picks file.
- There is no automatic stop-loss or time cap (the backtest showed both hurt this
  mean-reversion strategy). The rare broken name that slips past the filter is held
  until it recovers — which is why your `DROP` judgement and the user's manual
  monitoring matter.

Now, collect the parameters with `AskUserQuestion` (batched as described) and begin.
