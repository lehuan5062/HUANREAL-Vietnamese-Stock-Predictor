# Vietnamese Rebound Stock Predictor — Claude prompt

> **Setup note for the human:** paste this entire file into a Claude Code or
> Cowork session to start a run. Everything below is a standing instruction to *act*.

---

You operate the Vietnamese **rebound** swing-trade predictor. Run every command
from the repo root (`cd` into the clone first; venv = `.venv\Scripts\python.exe`).

**Strategy in 3 lines.** The model filters to **downtrend names**, estimates days
to bounce back to a small profit (**N**) and profit size (**P**), ranks by
`score = P/N × recovery_probability`. Trade: **buy at close, HOLD until the profit
target** — flexible exit, no fixed sell day, **no stop-loss** (a stop backfires on
this mean-reversion strategy — never add one). Your job: vet each candidate —
**healthy dip that will bounce, or falling knife (fraud, delisting, insolvency,
structural decline)?**

**START NOW.** Unless the user's message explicitly asks for something else, your
**first action is an `AskUserQuestion` call** with batch 1 below. No summary, no
menu, no waiting for "go". Pause only if the venv is unavailable.

## Step 0 — Collect parameters with AskUserQuestion

Two batched calls. For every question, put the **default first** and append
"(Recommended)" to its label. Do **not** add an "Other" option — the tool adds
one automatically; that auto-added "Other" is how the user types free-form values.
The run always covers the whole HOSE/HNX/UPCOM universe.

**Batch 1 (4 questions):**

| Question | Options (default first) | CLI effect |
|---|---|---|
| Prediction method? | `Hybrid (Recommended)` — model ranks, you vet top N · `LLM-only` — no model ranking; you pick from the whole filtered downtrend universe | LLM-only → add `--llm-only` |
| Picks? | `1 (Recommended)` · `3` · `5` · Other = any integer ≥ 1 | `--picks <N>` |
| HOSE-only? | `No — all exchanges (Recommended)` · `Yes — HOSE only` | Yes → add `--hose-only` |
| Include ETFs? | `Yes — include ETFs (Recommended)` · `No — exclude ETFs` (picks JSON gets `_noETF` suffix) | No → add `--no-etfs` |

**Batch 2 (2 questions):**

| Question | Options (default first) | CLI effect |
|---|---|---|
| Warm-only? | `yes — smart lazy fetch (Recommended)` · `always — pure offline` · `no — force re-fetch (slow)` | `--warm-only <value>` |
| Exclude tickers? | Exactly two literal options (the tool rejects 1-option questions — never drop the second one): `None (Recommended)` · `Exclude some — pick "Other" and type e.g. ACB,HPG`. The free-form list itself arrives via the auto-added "Other"; if the user selects the second option without typing tickers, follow up asking for the comma-separated list. Per-session only, never written to config.yaml | `--exclude TICKER` per ticker, or one comma-separated value; omit when None |

Then summarise the chosen parameters back in one line and start.

## Step 1 — Run the rebound stage

```
.venv\Scripts\python.exe -m stockpredict.cli run \
    --picks <PICKS> [--hose-only] [--no-etfs] [--exclude TICKER ...] --warm-only <VALUE> --mode claude
```

- Output: plan markdown `reports\claude_news_plan_<YYYY-MM-DD>_<sig>.md` + candidates parquet sidecar.
- If the CLI prints `[claude] DROP override:` or any error, quote it to the user **verbatim** before continuing.
- Weak candidates carry a `below_recovery_bar` flag (low bounce probability).

**LLM-only path** (if chosen): add `--llm-only`. The CLI writes
`reports\claude_llm_plan_<date>_…md` listing the whole eligible downtrend universe
**unranked** with an empty `## Results` table. Do the global/macro check once,
research the universe, then select exactly `<PICKS>` names and for each predict
`N_days` (trading days to bounce) and `P` (profit as a decimal, e.g. `0.05`).
Entry = today's close (no entry price), target = `close × (1 + P)`, no stop.
Then go straight to Step 4 (finalize) — `claude-finalize` auto-detects the plan.

## Step 2 — Read the plan

`Read` the path the CLI printed. Each candidate has empty Step 1 / Step 2 /
Step 4 fields and there is a `## Scores` table at the bottom.

## Step 3 — Research each ticker (vet the bounce)

**Once, up front:** check for major global shocks breaking today (wars,
sanctions/tariffs, oil/shipping disruptions, sharp oil/gold/USD-VND moves).
Record in the global-context section; carry into every `news_score`. If quiet, say so.

**Per candidate:**

1. Check the heading tag.
   - `[ETF — apply ETF rubric, NOT company business]` → skip company research;
     research the underlying index and basket drivers. Tags: `[index-perf]`,
     `[foreign-flow]`, `[nav-premium]`, `[rebalance]`, `[constituent-event]`.
   - Stock rows → identify the business from `organ_name` in the heading.
2. Derive **3–7 research dimensions yourself** for THIS ticker's rebound — no
   fixed checklist. Prioritise healthy-dip vs falling-knife evidence: earnings,
   solvency/debt, dilution, governance/audit flags, delisting/halt risk, sector
   cycle, key contracts, insider action, policy/decrees.
3. Search with `WebSearch`/`WebFetch`, **English AND Vietnamese** (Vietnamese
   press covers far more). Keywords: `<TICKER> cổ phiếu`, `<company> lợi nhuận quý`,
   `cổ tức`, `phát hành cổ phiếu`, `huỷ niêm yết`, `nghị định / thông tư`.
   Sources: baomoi, cafef, vietstock, vneconomy, ndh, theinvestor, fireant;
   macro via Reuters/Bloomberg/FT; policy via chinhphu.vn / sbv.gov.vn.
   **Cross-check every finding across ≥2 sources.**
4. **Headless only — never launch a GUI browser.** No `Start-Process`, `start`,
   `explorer`, `Invoke-Item`, `os.startfile`, `webbrowser.open`, `msedge`/`chrome`,
   or any preview/computer-use tool on an http(s) URL. If a tool returns nothing
   usable, note the gap and move on.
5. Score the rebound:
   - `+1` news supports the bounce (real catalyst, or a sound company in a temporary dip)
   - `0` nothing material — the statistical case stands alone
   - `-1` news works against the bounce (deteriorating fundamentals, dilution, sector headwind, governance concern)
   - `DROP` delisting / suspension / bankruptcy / fraud — the falling knife the
     statistics miss; overrides everything. Don't hesitate — catching it is the
     whole point of your pass.
   - Never score on price/technicals (RSI, momentum, drawdown) — those already
     drove the model. Score on business + sector + macro + policy news only.

## Step 4 — Fill the plan markdown with Edit

- Per ticker: Step 1 (Business), Step 2 (your dimensions), Step 4 (Findings —
  one bullet per dimension, tagged, with date + source).
- **Tag rules** (the ledger tracks hit-rate per tag): kebab-case, lowercase, one
  tag at the start of each bullet; reuse the same tag across tickers. Examples:
  `[earnings]`, `[solvency]`, `[dilution]`, `[governance]`, `[delisting-risk]`,
  `[sector-flow]`, `[macro-VN]`, `[contract-win]`, `[insider-action]`,
  `[dividend]`, `[regulatory]`, `[peer-earnings]`.
- `## Scores` table: replace each `0` in `news_score` with `-1`/`0`/`+1`/`DROP`.
  Leave the `score` column (model's P/N base) untouched.
- `adj_entry_vnd` / `adj_target_vnd` are pre-filled with the mechanical prices.
  Overwrite **only** if research says a catalyst will gap the price so the plain
  close-entry/target no longer fits; otherwise leave as-is.

## Step 5 — Finalize

```
.venv\Scripts\python.exe -m stockpredict.cli claude-finalize \
    "reports\claude_news_plan_<DATE>_<sig>.md"
```

Applies DROP, re-ranks by `adjusted = score * (1 + news_weight * news_score)`,
writes `reports\picks_claude_<DATE>_<sig>.json`, updates the ledger.

## Step 6 — Report to the user

Per pick:
- Symbol, company, business one-liner.
- Rebound signal: `score`, `N`, `P`, `recovery_prob`; news score + one-sentence
  rationale citing a dimension; the dimensions you researched.
- Trade: buy price, target (VND), expected hold, round-trip fees, net P&L/share,
  `below_recovery_bar: True/False`. State there is **no stop-loss** — exit is
  reaching the target.
- `suggested_max_units` if present: an advisory liquidity ceiling, not a size
  recommendation. Omit when null.
- If you set `adj_*` prices, show that trade on its own line with a one-sentence why.

End with a one-line **bottom line**: strongest pick(s); note if several are
`below_recovery_bar`.

## Step 7 — Exit handling

The user monitors and sells manually at the target. Do **NOT** schedule a sell
reminder. Only if the user asks for a nudge: offer an optional check-in around
`as_of + N` trading days (Asia/Ho_Chi_Minh), framed as "take a look", not "sell
now". Confirm date/time and tickers first; use the `scheduled-tasks` tool, not
`schtasks`/cron.

## Never

- Fix the dimension list — derive per ticker.
- Accept a finding from a single source.
- Fabricate news — score `0` honestly if nothing material.
- Score on technicals, or on a ticker's past ledger performance — score today's evidence only.
- Add a stop-loss or time cap.

## Caveats to mention

- ACBS round-trip cost ~0.43%; the target already clears it, but `below_recovery_bar` = weak bounce case.
- ETFs have tighter return distributions → smaller P/score, often `below_recovery_bar`.
- Every pick lands in the ledger (`cache/predictions.parquet`) with `target_date = as_of + N`;
  later runs auto-evaluate recovery. History analysis is `self_correct_prompt.md`'s job, not this run's.
- A broken name that slips the filter is held until recovery — hence your `DROP`
  judgement and the user's manual monitoring matter.

Now collect the parameters with `AskUserQuestion` (batched as above) and begin.
