# Vietnamese Rebound Stock Predictor — Self-correction prompt

Paste the contents of this file into a Claude Code or Cowork session. Claude runs
a one-shot **self-correction** pass over a picks report you choose: cross-reference
the predictions with what actually happened, diagnose systematic errors, and
propose program-level edits to `config.yaml` (and, where relevant, `claude_prompt.md`)
so future rebound runs do better.

This is **the only self-correction path**. Day-to-day runs do not see
past-performance feedback — each scores tickers purely on today's evidence. To act
on accumulated history you run this prompt manually on a chosen past picks file.
The output is targeted edits to the program, not nudges to individual scores.

## What the strategy is (so the diagnosis is grounded)

The model filters to **downtrend** names and, per ticker, estimates days-to-bounce
(**N** = `pred_days`), profit at the bounce (**P** = `pred_profit`), and an
eventual-**recovery probability** (`pred_recovery_prob`) — dominated by that
ticker's OWN history of bouncing. It ranks by `score = P/N × recovery_prob` and
gates out names below `strategy.recovery.min_recovery_prob` (the "healthy" filter).
The trade buys at the close and **holds until the close first clears the profit
target** (a flexible exit; there is no time cap, and the stop-loss is off by default).
So a pick **resolves** (`evaluated=True`, `recovered_flag=True`,
`actual_exit_date` set, `exit_reason="recovery"`) only when it bounces; a pick
that hasn't bounced yet stays **open** (`evaluated=False`).

## Style

Respond in simple, plain language. Short sentences. Explain jargon the first time.
Keep numbers exact but say what they mean in everyday terms.

## Your job

Take a picks report the user names, gather realized outcomes for its picks,
diagnose what went wrong / right, and propose narrowly scoped edits. Show every
diff and wait for explicit per-file approval — never silently mutate files.

## Step 1 — Scan the ledger for resolved vs open, then ask

**Get the wall clock first.** The harness injects today's date, not the time. Run
`.venv\Scripts\python.exe -c "import pandas as pd; print(pd.Timestamp.now(tz='Asia/Ho_Chi_Minh'))"`
and pin `now_vn`. HOSE closes 15:00 ICT. The buy day (T+0) = the report's `as_of`
itself, so it has closed iff `now_vn > 15:00` on that date.

Do an eligibility scan before asking — pair each recent picks JSON with its ledger
rows and classify:

```
.venv\Scripts\python.exe -c "import glob, json, os, pandas as pd; df = pd.read_parquet(r'cache\predictions.parquet')
for p in sorted(glob.glob(r'reports\picks_*.json'))[-16:]:
    d = json.load(open(p, encoding='utf-8'))
    rid = d['as_of'].replace('-','') + '_' + d['run_signature']; sub = df[df['run_id'] == rid]
    if len(sub) == 0: print(os.path.basename(p), 'as_of=' + d['as_of'], 'no ledger rows'); continue
    n = len(sub); rec = int(sub['recovered_flag'].fillna(False).sum()); ev = int(sub['evaluated'].sum())
    openn = n - ev
    print(os.path.basename(p), 'as_of=' + d['as_of'], 'mode=' + str(d.get('mode')), 'recovered=' + str(rec) + '/' + str(n), 'open=' + str(openn))"
```

Classify each report:
- **Some/all `recovered` (evaluated=True)** → **ready to diagnose** (the more
  resolved, the stronger the read).
- **All still open (0 evaluated)** → not yet useful; the picks haven't bounced.
  Offer to refresh data + evaluate first (see below), or pick an older report.
- **Buy day hasn't closed** (`now_vn <= 15:00` on `as_of`) → too early; come back
  after close.

To pull fresh outcomes, run `evaluate` (it refreshes the picked symbols and stamps
recoveries). **Never run a bare `update-data`** (it refetches the whole universe);
if a straggler persists, refresh just the picked symbols with
`update-data -s <SYM1> -s <SYM2>`.

**Default-recommend** the freshest report with the most resolved picks, and phrase
question 1 as a confirmation, not an open list. Then ask, one at a time:

1. **Picks JSON path** (required; default to the suggested report).
2. **Extra context** (optional) — anything to weight in (e.g. "I only bought 2 of
   the 3", "ignore VNM, data glitch"). Default empty.

Summarise back, then Step 2.

## Step 2 — Read the report + resolve siblings

`Read` the picks JSON. Pull: `as_of`, `mode`, `hose_only`, `include_etfs`,
`exclude`, `run_signature`, `requested_picks`, `n_picks`, and the `picks` array
(each has `symbol`, `score`, `pred_days`, `pred_profit`, `pred_recovery_prob`,
`entry_vnd`, `target_vnd`, `hold_days`, `below_recovery_bar`, `rsi_14`, `mom_5`,
`mom_20`, `high_prox_20`; claude/gemini also carry `news_score`, `business`,
`dimensions_cited`, `rationale`). If a `plan_file` is set, `Read` it for the
per-ticker reasoning. Also try the sidecar `<plan_file>.candidates.parquet`.

## Step 3 — Pull outcomes from the ledger

```
.venv\Scripts\python.exe -c "import pandas as pd; df = pd.read_parquet(r'cache\predictions.parquet'); df = df[df['run_id'] == '<RUN_ID>']; print(df[['symbol','pred_days','pred_profit','pred_recovery_prob','entry_price','target_date','actual_exit_date','realized_return','recovered_flag','exit_reason','evaluated','news_score']].to_string(index=False))"
```

`<RUN_ID>` = `<YYYYMMDD>_<run_signature>` (strip dashes from `as_of`).

For each resolved pick compute **realized days-to-recover** =
`actual_exit_date − as_of` in trading days, and compare to `pred_days`; compare
`realized_return` to `pred_profit`. For open picks, note how long they've been open.

Hold these as working evidence — quote specific numbers inside findings; don't emit
standalone tables.

## Step 4 — Cross-reference the broader ledger

```
.venv\Scripts\python.exe -c "from stockpredict.tracking import recent_performance; import json; print(json.dumps(recent_performance(window_days=90), indent=2, default=str))"
```

Use the pooled hit-rate / mean-return to judge whether this report's pattern is
systemic or a one-off.

### Step 4b — Recovery-filter calibration (Focus 1)

The highest-value question: **are picks actually bouncing at the rate the model
predicted?** Over the resolved picks (this report + the 90-day pool), compare the
realized recovered fraction to the mean `pred_recovery_prob`. If picks recover far
LESS than predicted (many stay open / turn out to be knives), the healthy filter is
too loose:
- raise `strategy.recovery.min_recovery_prob` (stricter — keeps only more-reliable
  bouncers), and/or
- tighten `strategy.downtrend.*` (e.g. raise `rsi_floor` off 0 to exclude
  free-falling knives, or make `high_prox_max` less deep so you're not buying names
  that have already collapsed).
If picks recover at or above the predicted rate, the filter is fine — don't touch it.

### Step 4c — P/N accuracy (Focus 2)

Compare realized days-to-recover vs `pred_days`, and realized profit vs
`pred_profit`, across resolved picks. Systematic bias (e.g. bounces consistently
take 2× longer than predicted, or profits undershoot) points at the empirical
estimator's buckets. Levers: `strategy.recovery.state_buckets` (finer/coarser
`rsi_edges` / `high_prox_edges`), `p_quantile` (lower = more conservative P), or
`min_ticker_obs` / `min_bucket_obs`. The estimator is empirical and largely
self-correcting as data accrues — **require a clear n≥5 systematic bias before
proposing a bucket change; don't over-tune to one report.**

### Step 4d — Falling-knife check (MANDATORY, both resolved and open picks)

For any pick that **hasn't recovered** (open, or `recovered_flag=False`), look at
what the price actually did — do not assume it's just "slow to bounce":

```
.venv\Scripts\python.exe -c "import pandas as pd; d=pd.read_parquet(r'cache\ohlcv\<SYM>.parquet'); print(d.tail(20).to_string())"
```

If it kept falling hard after entry, it was a **falling knife** the filters and (in
claude/gemini) the LLM vetting failed to catch. One knife is noise; a **pattern**
(≥3 on the report, or an n≥5 pool pattern of never-recoverers) is the trigger to:
- tighten the healthy filter (`min_recovery_prob`) or downtrend gate (Step 4b), and
- if the mode is claude/gemini, tighten the **DROP guidance** in `claude_prompt.md`
  / the prompt generator so the LLM flags that class of broken name (e.g. add the
  specific red flag it missed — a going-concern warning, a dilution event, an
  audit qualification).

This step exists because with no stop-loss a broken name is held indefinitely, so
catching the systematic source of knives is the main way to protect the strategy.

### Step 4e — Cross-method comparison (optional, advisory)

Only if the same day has picks from two+ methods (base / hybrid / LLM-only /
gemini) that have since resolved:

```
.venv\Scripts\python.exe -m stockpredict.cli compare-modes --window 90 --date <as_of>
```

Read the verdict in plain language (which method has the better realized return
over the window, and whether its edge is from distinctive picks). It's **advisory
only** — mode is a per-run user choice, so propose no edit from it. Flag thin
samples; don't recommend switching the default method off a few days.

## Step 5 — Diagnose (three focuses only)

Diagnose only: **(1) recovery-filter calibration**, **(2) P/N accuracy**, **(3)
falling knives**. Write findings as a numbered list; **every finding must drive at
least one proposed edit**. If a pattern suggests no concrete edit, skip it.

**Evidence threshold**: each finding cites ≥3 same-direction cases on this report,
**or** echoes an n≥5 pattern in the 90-day pool. A single miss doesn't clear the
bar — except an unmistakable falling knife bought into a clear collapse, which you
must state bluntly as a finding regardless.

"No systemic pattern found" is a valid outcome — say so and stop; don't manufacture one.

## Step 6 — Propose edits

Write the diagnosis + proposals to
`reports\self_correction_<YYYY-MM-DD>_<sig>.md` (use the report's `as_of` /
`run_signature`). Keep it lean: findings + proposed edits + applied tracking.

**Edit-target priority:**
1. **`config.yaml`** — the primary lever:
   - `strategy.recovery.min_recovery_prob` (current 0.85) — the healthy filter.
     Raise if picks under-recover / knives slip through.
   - `strategy.downtrend.*` — `mom20_max`, `high_prox_max`, `rsi_floor` (0=off),
     `rsi_ceil` — widen or tighten the candidate pool.
   - `strategy.recovery.state_buckets` / `p_quantile` / `min_ticker_obs` /
     `min_bucket_obs` — the empirical estimator shape (Focus 2; high bar).
   - `strategy.recovery.stop_loss_pct` (0 = off, the default) — the only exit
     override left. **Note the backtest finding: a price stop HURTS this
     mean-reversion strategy (it sells right before the bounce); off is
     deliberate.** Don't propose turning it on without a strong, user-endorsed
     reason. (There is no time cap — the strategy holds until profit.)
2. **`claude_prompt.md`** (claude/gemini only) — tighten the DROP / falling-knife
   vetting guidance when the LLM missed a broken name. Additive, narrowly scoped.
3. **Source files** — only for a concrete structural defect (parser bug, wrong
   formula, missing column). Default to NOT touching code; write the finding, stop,
   and let the user trigger a separate code task.

Each proposed edit lists: the finding it's motivated by, file + line range, the
exact diff (before/after), and a one-sentence expected effect.

## Step 7 — Apply on approval

Show the report path and all diffs. Then ask, **one file at a time**:
`Apply the change to <file>? (y/n)`. On `y`, `Edit` and append to `## Applied`:
`- <YYYY-MM-DD HH:MM> applied edit to <file> (finding #<n>): <summary>`. On `n`,
append the declined line. Never apply multiple edits with one approval; never apply
a source-code edit without a separate explicit "yes, change code here".

## Step 8 — Sanity-check suggestion

After edits, suggest a dry pass (don't run it for them):

```
.venv\Scripts\python.exe -m stockpredict.cli predict --mode base --picks 3 --skip-train
```

If a `config.yaml` knob that feeds the model was changed
(`strategy.downtrend.*`, `strategy.recovery.state_buckets` / `p_quantile` /
`min_*_obs`), remind them to **retrain the recovery head** so it takes effect:

```
.venv\Scripts\python.exe -m stockpredict.cli train
```

(`min_recovery_prob` / `profit_margin` / the exit knobs are read at predict/eval
time and need no retrain.) Remind the user: one report ≠ a backtest — re-run
self-correction on another report after more picks resolve before treating any
change as confirmed. For a portfolio-level sanity check, they can re-run
`scripts/rebound_portfolio_sim.py`.

## What NOT to do

- Don't run a full-universe `update-data` (no `-s`). `evaluate` refreshes the
  picked symbols itself.
- Don't ask the user to pick blindly from a list — run the eligibility scan and
  default-suggest.
- Don't propose edits from a single miss (except an obvious falling knife).
- Don't auto-apply; show diffs and wait for per-file approval.
- Don't touch source files when a `config.yaml` knob would do.
- Don't propose turning on the stop-loss without a strong, user-endorsed
  reason — the backtest showed they hurt this strategy.
- Don't call an unrecovered pick "just slow" without looking at its chart — it may
  be a falling knife.
- Don't fabricate a finding to fill the report. "No systemic pattern" is valid.

Now, ask the user for the picks path and begin.
