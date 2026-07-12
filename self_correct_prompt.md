# Vietnamese Rebound Stock Predictor — Self-correction prompt

Paste this file into a Claude Code or Cowork session. Claude runs a one-shot
**self-correction** pass over a picks report you choose: cross-reference the
predictions with what actually happened, diagnose systematic errors, and propose
program-level edits to `config.yaml` (and, where relevant, `claude_prompt.md`).
This is the only self-correction path — day-to-day runs score tickers purely on
today's evidence with no past-performance feedback. The output is targeted edits
to the program, not nudges to individual scores.

## What the strategy is (so the diagnosis is grounded)

The model filters to **downtrend** names and, per ticker, estimates days-to-bounce
(**N** = `pred_days`), profit at the bounce (**P** = `pred_profit`), and an
eventual-**recovery probability** (`pred_recovery_prob`) — dominated by that
ticker's OWN history of bouncing. It ranks by `score = P/N × recovery_prob` and
gates out names below `strategy.recovery.min_recovery_prob` (the "healthy" filter).
The trade buys at the close and **holds until the close first clears the profit
target** — no time cap, stop-loss off by default. A pick **resolves**
(`evaluated=True`, `recovered_flag=True`, `actual_exit_date` set,
`exit_reason="recovery"`) only when it bounces; otherwise it stays **open**
(`evaluated=False`).

## Style

Respond in simple, plain language. Short sentences. Explain jargon the first time.
Keep numbers exact but say what they mean in everyday terms.

## Your job

Take a picks report the user names, gather realized outcomes, diagnose what went
wrong / right, and propose narrowly scoped edits. Show every diff and wait for
explicit per-file approval — never silently mutate files.

## Step 0 — Refresh ledger with current prices (mandatory)

**Always run this first.** It fetches current prices for all previously-picked
symbols and updates the ledger's `recovered_flag` / `evaluated` fields; without
it the resolution status is stale.

```
.venv\Scripts\python.exe -m stockpredict.cli evaluate
```

Then proceed to Step 1.

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
    # T+2 settlement floor; 'recovered' exits earlier than this are not executable.
    unsellable = 0
    recovered_rows = sub[sub['recovered_flag'].fillna(False)]
    for _, r in recovered_rows.iterrows():
        sellable_floor = _next_trading_offset(pd.Timestamp(r['as_of']), 2)
        if pd.Timestamp(r['actual_exit_date']) < sellable_floor: unsellable += 1
    print(os.path.basename(p), 'as_of=' + d['as_of'], 'mode=' + str(d.get('mode')), 'recovered=' + str(rec) + '/' + str(n), 'open=' + str(openn), 'checkpoint=' + str(checkpoint), 'unsellable=' + str(unsellable))"
```

**If `unsellable > 0`**: that pick "recovered" before T+2 — the earliest a VN
purchase can actually be sold — so it's not evidence of a capturable bounce.
Exclude those picks from Step 3's resolved evidence (treat as still-open) even
though the ledger marks them `evaluated=True`. This can happen on clean data:
`resolve_exit()` in `src/stockpredict/model/target.py` has no minimum-hold gate
(unlike `scripts/rebound_final_sim.py`'s backtest). If it recurs across several
reports, note it as a source-code fix candidate per Step 6's edit-target
priority 3 — flag it, don't propose it silently.

Classify each report:
- **Some/all `recovered` (evaluated=True), and `unsellable == 0` for those** →
  **ready to diagnose** (the more genuinely resolved, the stronger the read).
- **`checkpoint > 0`** (open picks past their `pred_days` date without
  recovering) → **checkpoint ready** — actionable evidence for Step 4d, not just
  "too early."
- **All still open, `checkpoint == 0`** → not yet useful; offer an older report
  or waiting.
- **Buy day hasn't closed** (`now_vn <= 15:00` on `as_of`) → too early.

**Note on matching**: filter ledger rows by both `run_id` AND the picks JSON's
symbols, not run_id alone — a rerun of the same signature can overwrite the picks
JSON while leaving orphan ledger rows for dropped symbols; the symbol filter
keeps outcomes matched to the surviving picks.

(`evaluate` already ran in Step 0. **Never run a bare `update-data`** — it
refetches the whole universe; refresh stragglers with
`update-data -s <SYM1> -s <SYM2>`.)

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

Filter the ledger to rows matching both the `run_id` AND the picks' symbols
(same orphan-row reason as Step 1's matching note):

```
.venv\Scripts\python.exe -c "import pandas as pd, json; d = json.load(open(r'<PICKS_JSON_PATH>', encoding='utf-8')); syms = set(x['symbol'] for x in d['picks']); df = pd.read_parquet(r'cache\predictions.parquet'); rid = d['as_of'].replace('-','') + '_' + d['run_signature']; df = df[(df['run_id'] == rid) & (df['symbol'].isin(syms))]; print(df[['symbol','pred_days','pred_profit','pred_recovery_prob','entry_price','target_date','actual_exit_date','realized_return','recovered_flag','exit_reason','evaluated','news_score']].to_string(index=False))"
```

Replace `<PICKS_JSON_PATH>` with the picks JSON path (e.g.,
`reports/picks_2026-07-02_base_d2_GEE.json`).

For each resolved pick compute **realized days-to-recover** =
`actual_exit_date − as_of` in trading days vs `pred_days`, and `realized_return`
vs `pred_profit`. For open picks, note how long they've been open. Hold these as
working evidence — quote specific numbers inside findings; no standalone tables.

## Step 4 — Cross-reference the broader ledger (required: all sub-steps)

Pull the 90-day pool stats:

```
.venv\Scripts\python.exe -c "from stockpredict.tracking import recent_performance; import json; print(json.dumps(recent_performance(window_days=90), indent=2, default=str))"
```

Then run **all four sub-steps (4b, 4c, 4d, 4e)** regardless of interim findings —
they feed Step 5's diagnosis.

### Step 4b — Recovery-filter calibration (Focus 1, mandatory comparison)

The highest-value question: **are picks bouncing at the predicted rate?** Over
resolved picks (this report + the 90-day pool), compare the realized recovered
fraction to the mean `pred_recovery_prob`. If picks recover far LESS than
predicted, the healthy filter is too loose — levers:
- raise `strategy.recovery.min_recovery_prob` (stricter), and/or
- tighten `strategy.downtrend.*` (e.g. raise `rsi_floor` off 0, or make
  `high_prox_max` less deep) to exclude free-falling knives.
If recovery is at or above the predicted rate, the filter is fine — don't touch it.

### Step 4c — P/N accuracy (Focus 2, mandatory comparison)

Compare realized days-to-recover vs `pred_days` and realized profit vs
`pred_profit` across resolved picks; note any systematic bias (e.g. bounces take
2× longer, profits undershoot). Hold for Step 5. Levers if a bias clears the bar:
`strategy.recovery.state_buckets`, `p_quantile`, `min_ticker_obs` /
`min_bucket_obs`. The estimator is empirical and largely self-correcting —
**require a clear n≥5 systematic bias before proposing a bucket change; don't
over-tune to one report.**

### Step 4d — Falling-knife check + checkpoint misses (mandatory, both resolved and open picks)

Check every unrecovered pick (open, or `recovered_flag=False`). Pull its price
action — never assume it's just "slow to bounce":

```
.venv\Scripts\python.exe -c "import pandas as pd; d=pd.read_parquet(r'cache\ohlcv\<SYM>.parquet'); print(d.tail(20).to_string())"
```

Classify each: **falling knife** (kept falling hard after entry; the filters/LLM
vetting missed it) or **stuck/slow** (sideways, waiting). One knife is noise; a
**pattern** (≥3 knives on this report, or an n≥5 pool pattern of
never-recoverers) clears the evidence bar. With no stop-loss a broken name is
held indefinitely, so catching the systematic source of knives is the main
protection.

**Checkpoint misses** (picks flagged `checkpoint > 0` in Step 1's scan): check
whether the close ever reached the target (`target_vnd`, or
`entry_price × (1 + pred_profit)`) by the predicted checkpoint date:
- **Reached target by then** → timing was right; informational only (`evaluate`
  just hasn't marked it, or the close dipped back below).
- **Never reached target, price has since fallen** → **checkpoint miss** — and
  possibly also a knife; check both.
- **Never reached target, hovering near entry** → **checkpoint miss**,
  slow-mover variant.

Hold the count and specifics (symbol, pred_days, days elapsed, target vs actual
price) for Focus 4 in Step 5.

### Step 4e — Cross-method comparison (required, but advisory for edits)

Compare what **different modes picked on the same day** (`as_of`), in two parts:

**(1a) Compare the actual pick sets — always works, even with open picks.** Run:

```
.venv\Scripts\python.exe -m stockpredict.cli compare-picks --date <as_of>
```

Reads all modes' picks JSONs for the day (base, claude, claude_llm) and shows
each mode's symbols/scores/prob/profit, the **overlap** (2+ mode agreement is a
signal), and the **LLM rationales** for adds/drops vs base. Works the moment the
reports exist — no waiting for recovery.

**(1b) Mode accountability for resolved picks — diagnose root cause.** Run:

```
.venv\Scripts\python.exe -m stockpredict.cli compare-picks-accountability --date <as_of>
```

For each resolved pick this shows which modes selected it and which avoided it.
Hold the output for Step 5, which owns the root-cause mapping (shared ML model
vs mode-specific vetting).

**(2) Compare realized outcomes — only for resolved picks.** Run:

```
.venv\Scripts\python.exe -m stockpredict.cli compare-modes --window 90 --date <as_of>
```

Its "Named day" section (bottom) shows each mode's resolved symbols and mean
return for `<as_of>`; the pooled tables above are 90-day context. If it says "no
evaluated picks on [date]" or shows one mode, say so and lean on part (1).

Both parts are **advisory only** — mode is a per-run user choice, so propose no
edit from it. Flag thin samples (one mode ran, n=1 resolved); don't recommend
switching the default method off a few days' evidence.

## Step 5 — Diagnose (four focuses only)

Diagnose only: **(1) recovery-filter calibration**, **(2) P/N accuracy**, **(3)
falling knives**, **(4) pred_days calibration**. For each finding, use the Step
4e(1b) mode-accountability data to identify the root cause — ask: **Did all
modes make this error, or only some?**
- **All modes** (or base + most) picked the knife → **shared ML model** problem
  (recovery-prob calibration, P/N estimation, downtrend filters)
  → Edit config.yaml (strategy.recovery.min_recovery_prob, strategy.downtrend.*)
- **Only LLM modes** → **LLM vetting** missed a red flag
  → Edit claude_prompt.md (tighten DROP guidance)
- **Only base** → base's distinctive picks are weak
  → Edit config.yaml (downtrend filters too loose for base)
- **Winners** mirror this: only-base winners = base's distinctive strength;
  only-LLM winners = good vetting.

### Focus 4 — Pred_days calibration (checkpoint misses)

Using Step 4d's checkpoint-miss data: are picks passing `pred_days` without
hitting target? If so, the bounce-timing estimate is systematically too
optimistic (or the target too high for the timeframe).
- **Evidence threshold**: ≥3 checkpoint misses on this report, or an n≥5
  pool pattern of bounces landing well past `pred_days`.
- **Root cause**: `pred_days` comes from the shared empirical estimator, so
  all-modes misses = shared model; single-mode is unlikely.
- **Levers**: same as Focus 2 (`p_quantile` — lower = shorter/more optimistic
  `pred_days`; bucket knobs), same n≥5 bar.
- Don't conflate with Focus 3: a miss hovering near entry is a **timing**
  problem; a miss that's also cratering is **both** — cite under both foci.

Write findings as a numbered list; **every finding must drive at least one
proposed edit** — if a pattern suggests no concrete edit, skip it.

**Evidence threshold**: each finding cites ≥3 same-direction cases on this
report, **or** echoes an n≥5 pattern in the 90-day pool. A single miss doesn't
clear the bar — except an unmistakable falling knife bought into a clear
collapse, which you must state bluntly regardless.

"No systemic pattern found" is a valid outcome — say so and stop; don't
manufacture one.

## Step 6 — Propose edits

Write the diagnosis + proposals to
`reports\self_correction_<YYYY-MM-DD>_<sig>.md` (use the report's `as_of` /
`run_signature`). Keep it lean: findings + proposed edits + applied tracking.

### Step 6a — Run config-tuner search analysis (only if Step 5 points at a config.yaml knob)

**Skip entirely if no Step 5 finding implicates a `config.yaml` knob** (e.g. "no
systemic pattern," or everything routes to `claude_prompt.md`/source). When one
does, run:

```
.venv\Scripts\python.exe -m scripts.rebound_config_suggest
```

It analyzes the portfolio-level backtests accumulated by
`scripts.rebound_config_tuner` (which randomizes ~24 prediction-affecting
`config.yaml` knobs — liquidity/history gates, downtrend gates, recovery/model
thresholds, pricing, walk-forward windows — each on its own random 1-year
window, scored by `annualized_IRR`). Because knobs are sampled independently and
jointly, its per-knob marginal analysis is a legitimate signal. It prints, per
knob: grouped mean IRR or correlation with IRR, thin-group flags, and a
suggested value where supported; with 50+ trials it also prints a
LightGBM-based suggested config with a holdout-R² honesty check. It handles the
too-few-trials case itself ("not enough trials yet") — **read what it prints;
don't second-guess or re-derive any of it.**

**Once run, it gates what Step 6 may propose for `config.yaml`:**
- Step 5 identifies THAT a knob-area problem exists; it **never** determines
  the new value — **the proposed number always comes from this step's output**,
  cited alongside the Step 5 finding.
- If it says "not enough trials yet" or flags the knob as thin/no-suggestion —
  do NOT propose a number. Write the finding as-is and tell the user to
  accumulate more tuner trials first. A finding without a number is a valid,
  honest outcome.
- If its suggested direction contradicts an unmistakable Step 5 finding (e.g.
  loosen a gate vs clear falling-knife evidence) — don't propose the edit;
  surface the conflict and let the user decide.
- **Range-boundary warnings:** the output ends with a `=== Range-boundary
  check ===` section. If it prints a `RANGE-BOUNDARY WARNING` for a knob that a
  Step 5 finding implicates, you may propose (normal Step 7 per-file approval)
  an edit to `scripts/rebound_config_tuner.py` widening that knob's entry in
  `KNOB_BOUNDS` — extend in the flagged direction by roughly the current span
  (or add 1–2 grid values for a choice knob), stating the old → new range. Do
  NOT simultaneously propose a `config.yaml` value outside the old range: the
  widened range needs fresh tuner trials before any value out there has
  evidence behind it. A warning for a knob no Step 5 finding implicates is
  informational only — mention it in the report, propose nothing.

**Edit-target priority:**
1. **`config.yaml`** — the primary lever, but ONLY per the Step 6a gate above;
   never hand-pick a "raise/lower it a bit" number from diagnosis alone.
   - **Exception — `strategy.recovery.stop_loss_pct`** (0 = off, the default):
     the tuner doesn't search it, and the backtest showed a price stop HURTS
     this mean-reversion strategy (it sells right before the bounce) — off is
     deliberate. Don't propose turning it on without a strong, user-endorsed
     reason. (There is no time cap — the strategy holds until profit.)
2. **`claude_prompt.md`** (claude/gemini only) — tighten the DROP /
   falling-knife vetting guidance when the LLM missed a broken name. Additive,
   narrowly scoped.
3. **Source files** — only for a concrete structural defect (parser bug, wrong
   formula, missing column). Default to NOT touching code; write the finding,
   stop, and let the user trigger a separate code task.

Each proposed edit lists: the motivating finding, file + line range, the exact
diff (before/after), and a one-sentence expected effect.

## Step 7 — Apply on approval

Show the report path and all diffs. Then ask, **one file at a time**:
`Apply the change to <file>? (y/n)`. On `y`, `Edit` and append to `## Applied`:
`- <YYYY-MM-DD HH:MM> applied edit to <file> (finding #<n>): <summary>`. On `n`,
append the declined line. Never apply multiple edits with one approval; never
apply a source-code edit without a separate explicit "yes, change code here".

## Step 8 — Sanity-check suggestion

After edits, suggest a dry pass (don't run it for them):

```
.venv\Scripts\python.exe -m stockpredict.cli predict --mode base --picks 3 --skip-train
```

If a model-feeding knob changed (`strategy.downtrend.*`,
`strategy.recovery.state_buckets` / `p_quantile` / `min_*_obs`), remind them to
**retrain the recovery head**:

```
.venv\Scripts\python.exe -m stockpredict.cli train
```

(`min_recovery_prob` / `profit_margin` / the exit knobs are read at predict/eval
time — no retrain.) Remind the user: one report ≠ a backtest — re-run
self-correction after more picks resolve before treating a change as confirmed.
For a portfolio-level check they can re-run
`.venv\Scripts\python.exe -m scripts.rebound_portfolio_sim`; if a
backtest-window or recovery-model knob changed, also suggest accumulating more
`scripts.rebound_config_tuner` trials for future Step 6a passes.

If a tuner sampling range was widened (Step 6a range-boundary warning): prior
trials still feed the ML surrogate, but that knob's group/tercile stats now mix
old and new ranges — tell the user to accumulate a batch of fresh trials before
trusting suggestions in the newly opened region of that knob.

## What NOT to do

- Don't ask the user to pick blindly from a list — run the eligibility scan and
  default-suggest.
- Don't call an unrecovered pick "just slow" without looking at its chart — it
  may be a falling knife.
- Don't fabricate a finding to fill the report. "No systemic pattern" is valid.
- Don't narrate readiness or ask "which report?" before working — run Step 0
  and the eligibility scan first; lead with the results.

Begin now — no preamble. Don't announce what you're about to do or say you're
ready. Immediately run Step 0 (`evaluate`) and the Step 1 eligibility scan, then
your first message to the user is the scan results with a recommended candidate.
