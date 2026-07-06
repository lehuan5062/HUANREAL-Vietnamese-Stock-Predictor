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

## Step 0 — Refresh ledger with current prices (mandatory)

**Always run this first.** It fetches current prices for all previously-picked symbols and updates the ledger's `recovered_flag` and `evaluated` fields so they reflect actual outcomes (not stale data).

```
.venv\Scripts\python.exe -m stockpredict.cli evaluate
```

This refreshes the picked symbols' OHLCV cache and checks which ones have bounced to their profit target. Without this, the ledger flags are stale and resolution status will be wrong.

After `evaluate` completes, proceed to Step 1.

## Step 1 — Scan the ledger for resolved vs open, then ask

**Get the wall clock first.** The harness injects today's date, not the time. Run
`.venv\Scripts\python.exe -c "import pandas as pd; print(pd.Timestamp.now(tz='Asia/Ho_Chi_Minh'))"`
and pin `now_vn`. HOSE closes 15:00 ICT. The buy day (T+0) = the report's `as_of`
itself, so it has closed iff `now_vn > 15:00` on that date.

Do an eligibility scan before asking — pair each recent picks JSON with its ledger
rows and classify:

```
.venv\Scripts\python.exe -c "import glob, json, os, pandas as pd
from stockpredict.tracking import _next_trading_offset
df = pd.read_parquet(r'cache\predictions.parquet')
today = pd.Timestamp.now(tz='Asia/Ho_Chi_Minh').tz_localize(None).normalize()
for p in sorted(glob.glob(r'reports\picks_*.json'))[-16:]:
    d = json.load(open(p, encoding='utf-8'))
    rid = d['as_of'].replace('-','') + '_' + d['run_signature']
    syms = set(x['symbol'] for x in d['picks'])
    sub = df[(df['run_id'] == rid) & (df['symbol'].isin(syms))]
    if len(sub) == 0: print(os.path.basename(p), 'as_of=' + d['as_of'], 'no ledger rows'); continue
    n = len(sub); rec = int(sub['recovered_flag'].fillna(False).sum()); ev = int(sub['evaluated'].sum())
    openn = n - ev
    # Checkpoint: pred_days elapsed but not yet evaluated (intermediate diagnostic signal).
    unresolved = sub[~sub['evaluated']]
    checkpoint = 0
    for _, r in unresolved.iterrows():
        if pd.isna(r['pred_days']): continue
        checkpoint_date = _next_trading_offset(pd.Timestamp(r['as_of']), max(int(round(r['pred_days'])), 1))
        if today >= checkpoint_date: checkpoint += 1
    print(os.path.basename(p), 'as_of=' + d['as_of'], 'mode=' + str(d.get('mode')), 'recovered=' + str(rec) + '/' + str(n), 'open=' + str(openn), 'checkpoint=' + str(checkpoint))"
```

Classify each report:
- **Some/all `recovered` (evaluated=True)** → **ready to diagnose** (the more
  resolved, the stronger the read).
- **`checkpoint > 0`** (some open picks have passed their `pred_days` date without
  recovering) → **checkpoint ready** — an intermediate diagnostic signal even though
  no pick has fully resolved. The model predicted a bounce by now and it hasn't
  happened; that's actionable evidence for Step 4d (see below), not just "too early."
- **All still open, `checkpoint == 0`** → not yet useful; no pick has even reached
  its predicted day yet. Offer to pick an older report with more recovery time, or
  wait for these picks to resolve further.
- **Buy day hasn't closed** (`now_vn <= 15:00` on `as_of`) → too early; come back
  after close.

**Note on matching**: The scan filters ledger rows by both `run_id` AND the symbols
in the picks JSON, not run_id alone. This is necessary because reruns of the same
signature on the same day may pick fewer symbols than the prior run. The prior run's
picks JSON gets overwritten, but its ledger rows for dropped symbols persist as
orphans. Filtering by symbol set ensures outcomes are matched only to the picks
actually in the surviving JSON.

(Note: `evaluate` was already run in Step 0, so the ledger is current. **Never run a bare `update-data`** — it refetches the whole universe; if a straggler persists, refresh just the picked symbols with `update-data -s <SYM1> -s <SYM2>`.)

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

Extract symbols from the picks JSON, then filter the ledger to only rows matching both the `run_id` AND the recorded picks' symbols. (A rerun of the same signature can leave orphan rows for symbols the earlier run picked but the later run dropped; filtering by symbol avoids misattributing those outcomes.)

```
.venv\Scripts\python.exe -c "import pandas as pd, json; d = json.load(open(r'<PICKS_JSON_PATH>', encoding='utf-8')); syms = set(x['symbol'] for x in d['picks']); df = pd.read_parquet(r'cache\predictions.parquet'); rid = d['as_of'].replace('-','') + '_' + d['run_signature']; df = df[(df['run_id'] == rid) & (df['symbol'].isin(syms))]; print(df[['symbol','pred_days','pred_profit','pred_recovery_prob','entry_price','target_date','actual_exit_date','realized_return','recovered_flag','exit_reason','evaluated','news_score']].to_string(index=False))"
```

Replace `<PICKS_JSON_PATH>` with the path to the picks JSON you're analyzing (e.g., `reports/picks_2026-07-02_base_d2_GEE.json`).

For each resolved pick compute **realized days-to-recover** =
`actual_exit_date − as_of` in trading days, and compare to `pred_days`; compare
`realized_return` to `pred_profit`. For open picks, note how long they've been open.

Hold these as working evidence — quote specific numbers inside findings; don't emit
standalone tables.

## Step 4 — Cross-reference the broader ledger (required: all sub-steps)

Pull the 90-day pool stats:

```
.venv\Scripts\python.exe -c "from stockpredict.tracking import recent_performance; import json; print(json.dumps(recent_performance(window_days=90), indent=2, default=str))"
```

Then run **all four sub-steps (4b, 4c, 4d, 4e)** regardless of interim findings.
They gather the data that feeds Step 5's diagnosis. The sub-steps are:

### Step 4b — Recovery-filter calibration (Focus 1, mandatory comparison)

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

### Step 4c — P/N accuracy (Focus 2, mandatory comparison)

Always compare realized days-to-recover vs `pred_days`, and realized profit vs
`pred_profit`, across resolved picks. Note whether the realized values track the
predicted values or show systematic bias (e.g. bounces consistently take 2× longer
than predicted, or profits undershoot). Hold these findings for Step 5's diagnosis.
(If a bias pattern clears the evidence threshold, the levers are `strategy.recovery.state_buckets`,
`p_quantile`, `min_ticker_obs` / `min_bucket_obs`. The estimator is empirical and
largely self-correcting — **require a clear n≥5 systematic bias before proposing a
bucket change; don't over-tune to one report.**)

### Step 4d — Falling-knife check + checkpoint misses (mandatory, both resolved and open picks)

Always check every unrecovered pick (open, or `recovered_flag=False`). For each,
pull its price action to see what actually happened — do not assume it's just
"slow to bounce":

```
.venv\Scripts\python.exe -c "import pandas as pd; d=pd.read_parquet(r'cache\ohlcv\<SYM>.parquet'); print(d.tail(20).to_string())"
```

Classify each unrecovered pick: is it a **falling knife** (kept falling hard after
entry, filters/LLM vetting failed to catch it), or is it **stuck/slow** (bouncing
sideways, waiting)? Hold these findings for Step 5. One knife is noise; a **pattern**
(≥3 knives on this report, or an n≥5 pool pattern of never-recoverers) clears the
evidence bar for a diagnosis. This step exists because with no stop-loss a broken
name is held indefinitely, so catching the systematic source of knives is the main
way to protect the strategy.

**Checkpoint misses** (picks flagged `checkpoint > 0` in Step 1's scan — still
`evaluated=False` but past their `pred_days` date): for these, additionally check
whether the close ever reached the target price (`target_vnd` from the picks JSON,
or recompute as `entry_price × (1 + pred_profit)`) by the predicted checkpoint date:
- **Reached target by the checkpoint date** → the model's timing was right; not a
  finding (informational only — it just hasn't been marked `evaluated` yet because
  `evaluate` hasn't rerun, or the close dipped back below target after touching it).
- **Never reached target by the checkpoint date, and price has since fallen** →
  **checkpoint miss** (the model's `pred_days` estimate was too optimistic, and the
  pick may also be a falling knife — check both).
- **Never reached target, but still hovering near entry** → **checkpoint miss**,
  slow-mover variant (timing wrong, but not yet a knife).

Checkpoint misses feed Focus 4 in Step 5 (pred_days calibration) — hold the count
and specifics (symbol, pred_days, days elapsed, target vs. actual price) for that
diagnosis.

### Step 4e — Cross-method comparison (required, but advisory for edits)

Always run this step. The point is to compare what **different modes picked on the
same day** (your `as_of`). Do it in two parts:

**(1a) Compare the actual pick sets — always works, even with open picks.** Run:

```
.venv\Scripts\python.exe -m stockpredict.cli compare-picks --date <as_of>
```

This reads all picks JSON files for the day across modes (base, claude, claude_llm)
and shows:
- Each mode's symbols, scores, recovery prob, and profit target.
- **Overlap** — which symbols 2+ modes independently picked (agreement is a signal).
- **LLM rationales** — where claude / claude_llm added or dropped names vs base,
  and the reasoning (dimensions_cited, rationale). That is the LLM vetting doing
  its job (or missing something).

This runs off the picks files directly, so it works the moment the reports exist —
no waiting for recovery.

**(1b) Mode accountability for resolved picks — diagnose root cause.** Run:

```
.venv\Scripts\python.exe -m stockpredict.cli compare-picks-accountability --date <as_of>
```

For **each resolved pick** (bounced or fell), this shows which modes selected it and
which avoided it. The output reveals:
- **Winners:** if only base picked it → base's distinctive picks are strong
  If only LLM picked it → LLM has good vetting
- **Losers (knives):** if **all modes** picked a knife → shared ML model problem
  (recovery-prob too loose, downtrend filter too loose)
  If **only LLM** picked a knife → LLM vetting failed; tighten claude_prompt.md
  If **only base** picked a knife → base's distinctive picks are weak; tighten filters
  If **some modes** avoided it but others didn't → mode divergence; understand why

This is diagnostic input for Step 5: use the mode accountability to pinpoint whether
the root cause is in the shared ML model or mode-specific vetting.

**(2) Compare realized outcomes — only for resolved picks.** Then run:

```
.venv\Scripts\python.exe -m stockpredict.cli compare-modes --window 90 --date <as_of>
```

Its "Named day" section (bottom) shows each mode's *resolved* symbols and mean
return for `<as_of>`; the pooled tables above are 90-day context. If the named-day
section says "no evaluated picks on [date]" or shows only one mode, the outcome
comparison isn't ready yet — say so and lean on part (1).

Both parts are **advisory only** — mode is a per-run user choice, so propose no
edit from it. Flag thin samples (e.g. only one mode ran that day, or n=1 resolved);
don't recommend switching the default method off a few days' evidence.

## Step 5 — Diagnose (four focuses only)

Diagnose only: **(1) recovery-filter calibration**, **(2) P/N accuracy**, **(3)
falling knives**, **(4) pred_days calibration**. For each finding, **use the mode
accountability data from Step 4e(1b) to identify the root cause** — is it a shared
ML model issue, or mode-specific vetting?

### Focus 4 — Pred_days calibration (checkpoint misses)

Using the checkpoint-miss data from Step 4d: are picks reaching their `pred_days`
date without hitting the target? If so, the model's bounce-timing estimate is
systematically too optimistic (or the target is set too high for that timeframe).
- **Evidence threshold**: ≥3 checkpoint misses on this report, or an n≥5 pattern in
  the 90-day pool where bounces consistently land well past `pred_days`.
- **Root cause** (via mode accountability): all modes missing timing on the same
  names → shared ML model issue. Only one mode → mode-specific (unlikely, since
  `pred_days` comes from the shared empirical estimator, not LLM vetting).
- **Lever**: `strategy.recovery.p_quantile` (lower quantile = shorter, more
  optimistic `pred_days`; raising it lengthens the estimate) or
  `strategy.recovery.state_buckets` / `min_ticker_obs` / `min_bucket_obs` (bucket
  granularity). Same high bar as Focus 2 — this is the same empirical estimator,
  just viewed through unresolved picks instead of resolved ones.
- Don't conflate with falling knives (Focus 3): a checkpoint miss that's still
  hovering near entry is a **timing** problem; a checkpoint miss that's also
  cratering is **both** a timing and a knife problem — cite it under both foci if so.

Write findings as a numbered list; **every finding must drive at least one proposed edit**.
If a pattern suggests no concrete edit, skip it.

When diagnosing, ask: **Did all modes make this error, or only some?**
- **All modes** (or base + most) picked the knife → problem is in the **shared ML model**
  (recovery-prob calibration, P/N estimation, downtrend filters)
  → Edit config.yaml (strategy.recovery.min_recovery_prob, strategy.downtrend.*)
- **Only LLM modes** picked the knife → problem is in **LLM vetting** (rationale/dimensions
  missed a red flag)
  → Edit claude_prompt.md (tighten DROP guidance)
- **Only base** picked the knife → **base's distinctive picks are weak**
  → Edit config.yaml (downtrend filters too loose for base)

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
     `min_bucket_obs` — the empirical estimator shape (Focus 2 and Focus 4; high bar).
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
- Don't narrate readiness or ask "which report?" before working — run Step 0 and
  the eligibility scan first; lead with the results.

Begin now — no preamble. Don't announce what you're about to do or say you're
ready. Immediately run Step 0 (`evaluate`) and the Step 1 eligibility scan, then
your first message to the user is the scan results with a recommended candidate.
