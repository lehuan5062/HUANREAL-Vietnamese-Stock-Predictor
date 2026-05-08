# Vietnamese T+N Stock Predictor — Self-correction prompt

Paste the contents of this file into a Claude Code or Cowork session in Claude
Desktop. Claude will then run a one-shot **self-correction** pass over a picks
report you choose: cross-reference the predictions with what actually happened,
diagnose systematic errors, and propose program-level edits to
`claude_prompt.md` and `config.yaml` so future runs do better.

This is **the only self-correction path** in the system. Day-to-day Claude
runs (driven by `claude_prompt.md`) do **not** see past-performance feedback
— each run scores tickers purely on today's research evidence. To act on
accumulated history, you run this prompt manually on a chosen past picks
file. The output is targeted edits to the program (config knobs, prompt
text), not nudges to individual scores.

## Two stages

There are two evaluation stages for any picks report. **Both can run
independently**, and you should choose one based on how much time has
elapsed since the picks were emitted:

* **Stage 1 — Limit-fill self-correction (after the BUY DAY closes)**.
  Fires the trading day after `as_of`. The buy day's OHLC is now in the
  ledger (`t0_evaluated=True`), which means we know whether each pick's
  `entry_limit_price` (from the low-prediction quantile head) actually
  filled. We do **not** know the realized return yet — that needs T+N
  to elapse. Diagnoses focus narrowly on **entry calibration**: was the
  predicted dip too aggressive (limits unreachable, no fill) or too
  conservative (limits filled cheaply but only a tiny dip captured)?
* **Stage 2 — Full self-correction (after T+N closes)**. The original
  flow. All rows have `evaluated=True` and `t0_evaluated=True`. You can
  now diagnose **everything**: scoring, entry calibration, dimension-tag
  patterns, news_score weighting, and how all of those interact with
  realized return.

When you run this prompt, **first** check the ledger to see which stage
the chosen picks report is eligible for, then proceed accordingly. Stage
1 is permitted before Stage 2. You can also run Stage 1 today and Stage
2 again later on the same report.

---

You are operating the Vietnamese T+N swing-trade stock predictor at `D:\stock`.
Your tools: `Bash`, `Read`, `Edit`. No `WebFetch`/`WebSearch` needed — the
evidence for self-correction lives entirely on disk.

## Your job

Take a picks report the user names, gather the realized outcomes (and/or
limit-fill outcomes) for those picks, diagnose what went wrong (and what
went right), and propose narrowly scoped edits to the program. Show every
diff and wait for explicit approval before applying — never silently
mutate files.

## Step 1 — Ask the user for the report and stage

Ask in plain conversation, one question at a time:

1. **Picks JSON path** (required). Absolute path to a finalized picks file,
   typically `D:\stock\reports\picks_claude_<YYYY-MM-DD>_<sig>.json`. List the
   available reports first if the user is unsure:
   ```
   D:\stock\.venv\Scripts\python.exe -c "import os, glob; [print(p) for p in sorted(glob.glob(r'D:\stock\reports\picks_claude_*.json'))]"
   ```
2. **Stage** (optional). Either `1` (limit-fill only) or `2` (full
   T+N realized). If the user doesn't specify, infer from the ledger
   (Step 3) and tell them what you decided.
3. **Extra context** (optional). Anything the user wants weighted into the
   diagnosis (e.g. "ignore VNM, it was a known data issue", "I executed only
   3 of the 5 picks"). Default empty.

After all are collected, summarise back and start Step 2.

## Step 2 — Resolve sibling files

`Read` the picks JSON. Pull these fields from it:

- `as_of`, `run_signature`, `exit_offset_days`, `units`, `hose_only`, `mode`
- `weight` (the `news_weight` that was active at finalize time)
- `plan_file` (interactive mode only — points at the news plan markdown)
- `picks` (the array; each item has `symbol`, `pred_mean`, `news_score`,
  `adjusted`, `entry_vnd`, `target_vnd`, `stop_vnd`, `actionable`, plus the
  `business`, `dimensions`, `drivers`, `key_news`, `dimensions_cited` fields
  Claude wrote at finalize time)

Then:

- If `plan_file` is set, `Read` it. This is the per-ticker research plan with
  the actual reasoning Claude wrote (Step 1 / Step 2 / Step 4 fields and tag
  citations).
- If `plan_file` is missing or the file doesn't exist (autonomous mode picks,
  or an old report), proceed with picks-only and **state this limitation up
  front** in the report you produce. Research-mode improvement suggestions
  will be shallower without the plan.
- Also try the sidecar candidates parquet (`<plan_file>.candidates.parquet`)
  — it has ML features (rsi, momentum, etc.) for the original 20-ticker
  candidate pool, useful for "did the ML signal match what news said?"

## Step 3 — Pull outcomes from the ledger and choose the stage

Read `D:\stock\cache\predictions.parquet` directly. The cleanest one-liner:

```
D:\stock\.venv\Scripts\python.exe -c "import pandas as pd; df = pd.read_parquet(r'D:\stock\cache\predictions.parquet'); df = df[df['run_id'] == '<RUN_ID>']; print(df[['symbol','pred_mean','news_score','adjusted','entry_price','entry_limit_price','pred_low','t0_low','entry_limit_filled','t0_evaluated','actual_exit','realized_return','evaluated','entry_slippage','dimensions_cited','target_date']].to_string(index=False))"
```

Where `<RUN_ID>` is `<YYYYMMDD>_<run_signature>` — e.g.
`20260506_claude_d11_u100`. Compute it from `as_of` (strip dashes) and
`run_signature`.

**Stage gates** (pick the highest stage that fully passes):

- **Stage 2 gate**: every row has `evaluated=True`. → run the full T+N
  diagnosis below.
- **Stage 1 gate**: every row has `t0_evaluated=True` (but at least one
  row has `evaluated=False`). → run the limit-fill-only diagnosis.
- **Neither gate passes**: stop. Tell the user:

  > Run `D:\stock\.venv\Scripts\python.exe -m stockpredict.cli evaluate-fills`
  > to stamp T+0 limit-fill outcomes (need only the buy day to have
  > closed), or `evaluate.bat` to run the full evaluation. Then rerun
  > this prompt, or pick an older report whose data has fully landed.

  Don't proceed past this gate even if the user pushes — the diagnosis
  would be hallucinated.

If Stage 2 passes, present a per-pick table showing:

| symbol | pred_mean | news_score | adjusted | realized_return | entry_limit_filled | dimensions_cited | entry_slippage |

If only Stage 1 passes, present:

| symbol | pred_mean | pred_low | entry_limit_price | t0_low | entry_limit_filled | entry_slippage |

Plus a one-line summary appropriate to the stage:

- Stage 2: n picks, hit-rate on this run, mean realized, mean(predicted)
  vs mean(realized) gap, **fill_rate**, **mean(pred_low) vs
  mean(actual_dip)**.
- Stage 1: n picks, fill_rate, mean(pred_low) (quoted dip), mean(actual
  dip = (t0_low − entry_price) / entry_price), calibration gap.

## Step 4 — Cross-reference with the broader ledger

Pull the broader by-dimension and by-news_score stats so the diagnosis isn't
based solely on one report. Use the existing helper:

```
D:\stock\.venv\Scripts\python.exe -c "from stockpredict.tracking import recent_performance; import json; print(json.dumps(recent_performance(window_days=90, mode='claude'), indent=2, default=str))"
```

Compare:

- **(stages 1 & 2) Limit-fill calibration**: the pooled `limit_fill`
  block tells you the 90-day fill_rate and calibration gap. Compare
  this run's fill_rate to it.
  - If pooled fill_rate ≈ `pricing.entry_low_alpha` (default 0.5) AND
    this run is way off in the same direction (e.g. 0% fill on a 50%-
    target alpha), this report's tickers were unusually directional
    (gap-ups), not a model issue.
  - If pooled fill_rate is also far from alpha (say pooled is 20% on a
    50% target), the low head is mis-calibrated — propose retraining or
    bumping `pricing.entry_low_alpha`.
- **(stage 2) news_score patterns**: did this report's news_score=+1
  picks lose, while the 90-day pooled news_score=+1 stat is winning? →
  likely a *this-report* problem (specific dimensions / sectors), not a
  *scoring-model* problem.
- **(stage 2) dimension tag patterns**: did this report's `[<some-tag>]`
  dimension lose, *and* the pooled by-dimension table for that tag also
  shows it losing (n ≥ 5)? → systemic weakness in how that dimension is
  being researched or scored.
- **(stage 2) entry_slippage cluster**: is `entry_slippage` systematically
  positive (mean > 0) for this run? → the **close-anchored** entries
  are unfillable; realized_return is fictional, and scoring tweaks
  won't help — pricing tweaks (e.g. `pricing.stop_atr_mult` or the
  entry rule) might. Note: in the new low-prediction regime, the user
  is supposed to place the LIMIT entry, not buy at close. So
  `entry_slippage` is mostly diagnostic for the legacy "buy at close"
  comparison; `entry_limit_filled` is the truer fill measurement.

## Step 5 — Diagnose

Write findings as a numbered list with **evidence threshold**: each finding
must cite ≥3 same-direction misses on this report, **or** the report's
pattern echoes an n≥5 pattern in the broader ledger. Single-pick findings
get logged but do **not** drive proposed edits.

For each finding, name the failure mode plainly. Stage-1 examples:

- "0 of 5 limits filled. Mean `pred_low` was −1.2% (median dip predicted)
  but actual median dip was −0.3%. The low head is too bearish on dips
  for this signature; pooled fill_rate at the same alpha (0.5) is 48%
  (n=87), so today's report sat in a one-sided up-day."
- "All 5 limits filled but mean dip-actual was −2.1% vs quoted −0.8%.
  We could have set the limit ~1.3% lower and still filled — money on
  the table. The low head is too conservative on dips; lower
  `pricing.entry_low_alpha` (or retrain at lower alpha)."

Stage-2 examples (in addition to all stage-1 examples):

- "All 4 picks tagged `[earnings]` lost (mean −1.8%); pooled `[earnings]`
  hit-rate is 45% (n=18). Earnings is being weighted higher than its
  historical predictiveness justifies."
- "3 of 5 picks had `entry_slippage > 0` AND `entry_limit_filled=True`.
  The limit fill saved us from the slippage trap, but only because
  the limit happened to be below the gap-up. Diagnosis: ML scoring is
  fine; entry_limit_pct chose well; no edit needed."
- "All `+1` scores were on real-estate names; all 3 lost. Real-estate
  research relied solely on `[sector-flow]` without checking
  `[regulatory]` or `[capital-raise]`. Coverage gap, not a scoring
  miscalibration."

Skip categories where the evidence isn't there. Empty diagnosis is a valid
outcome — say "no systemic pattern found" and stop, don't manufacture one.

## Step 6 — Propose edits

Write the diagnosis and proposals to
`D:\stock\reports\self_correction_<YYYY-MM-DD>_<sig>_stage<N>.md` (use the
picks report's `as_of` and `run_signature`, plus the stage you ran, so
naming aligns and re-runs at a different stage don't clobber). Structure:

```
# Self-correction — picks_claude_<date>_<sig>  (Stage <N>)

## Inputs
- Picks: <path>
- Plan: <path or "none — autonomous mode">
- Run id: <run_id>
- Stage: <1 = limit-fill only / 2 = full T+N>
- n picks, n with t0_evaluated, n evaluated
- (stage 2) hit-rate, mean realized
- fill_rate, mean(pred_low), mean(actual dip), calibration gap

## Per-pick table
<the table from Step 3>

## Findings
1. <finding with evidence cite>
2. ...

## Proposed edits
### claude_prompt.md
- **Motivated by**: finding #1
- **Location**: claude_prompt.md L<a>-L<b>
- **Diff**:
  ```diff
  - <before>
  + <after>
  ```
- **Expected effect**: ...

### config.yaml
- **Motivated by**: finding #2
- **Location**: config.yaml L<a>
- **Diff**: ...
- **Expected effect**: ...

## Applied
(populated in Step 7 as the user approves edits)
```

**Edit-target priority**:

1. `claude_prompt.md` — for research / scoring guidance (e.g. add a
   reminder to check `[regulatory]` for real-estate; tighten the
   tag-naming rule; add a "what NOT to do" line; expand the dimension
   examples). Additive, narrowly scoped — never rewrite a section
   wholesale.
2. `config.yaml` — for tunable knobs:
   - `modes.claude.news_weight` (current 0.05) — raise / lower if
     news_score is consistently mis-weighted relative to ML.
   - `modes.claude.candidate_pool` (current 20) — only change if the
     pool size is plausibly the issue.
   - `pricing.stop_atr_mult` (current 1.5), `pricing.min_rr_ratio`
     (current 0.8) — for entry/exit rule mismatches.
   - **`pricing.entry_low_alpha` (current 0.5) — raise toward 0.75 if
     fill_rate is consistently below alpha (we're too bearish on
     dips); lower toward 0.25 if fills are happening at quoted dip
     much smaller than actual dip (we're leaving money on the table).
     This is the primary knob for stage-1 findings.**
3. Source files (e.g. `src/stockpredict/news/claude_runner.py`,
   `src/stockpredict/tracking.py`, `src/stockpredict/model/train.py`)
   — **only** when there's a concrete structural defect (parser bug,
   missing column, wrong formula). Default to **not** touching code;
   if the right fix is structural, write the finding, propose the
   change in the report, and **stop** — let the user trigger a
   separate code-change task.

Each proposed edit must include:
- The finding number it's motivated by.
- File + line range.
- The exact diff (before / after).
- One-sentence expected effect.

## Step 7 — Apply on approval

Show the report path and all proposed diffs to the user. Then ask, **one
file at a time**:

> Apply the change to `<file>`? (y/n)

On `y`: use `Edit` to apply, then append to the report's `## Applied`
section:

```
- <YYYY-MM-DD HH:MM> applied edit to <file> (finding #<n>): <one-line summary>
```

On `n`: append:

```
- <YYYY-MM-DD HH:MM> declined edit to <file> (finding #<n>): <user reason if given>
```

Never apply multiple edits with one approval. Never apply a source-code
edit without a separate, explicit "yes I want a code change here" — config
and prompt are reversible by editing back, code changes can have wider
blast radius.

## Step 8 — Sanity-check suggestion

After all edits are applied (or skipped), suggest the user run a small
dry pass to confirm nothing broke:

```
D:\stock\.venv\Scripts\python.exe -m stockpredict.cli predict --mode claude --top 3 --days 2 --skip-train
```

Don't run it for them — they may want to inspect diffs first.

If the user changed `pricing.entry_low_alpha`, also remind them they'll
need to **retrain the low head** before the new alpha takes effect:

```
D:\stock\.venv\Scripts\python.exe -m stockpredict.cli train
```

## What NOT to do

- Don't propose edits from a single losing or unfilled pick. Evidence
  threshold = ≥3 on-report or ≥5 in pooled ledger.
- Don't auto-apply. Always show the diff and wait for per-file approval.
- Don't touch source files when a `config.yaml` knob would do.
- Don't rewrite `claude_prompt.md` wholesale. Additive, narrowly-scoped
  edits only — adding one line, tightening one rule, expanding one example.
- Don't skip the stage gate. Stage 1 is fine to run after the buy day
  closes, but Stage 2 requires `evaluated=True` on every row.
- Don't fabricate a finding to fill the report. "No systemic pattern
  found" is a valid outcome.
- Don't conflate scoring failures with execution failures. If the
  cluster of issues is `entry_limit_filled=False` AND `pred_low` was
  too bearish, that's a low-head calibration problem, not a scoring
  problem.
- Don't propose changes to `dimensions_cited` parsing or the ledger
  schema unless you've shown a concrete bug, not a stylistic preference.

## Caveats to note in the produced report

- **Stage 1 has limited diagnostic power.** You can only tell whether
  the limit was reachable, not whether the trade would have been
  profitable. A pick that filled at a great limit price can still tank
  on the realized return; a pick that didn't fill avoided a potential
  loss. Stage 1 is *purely* about entry calibration.
- **Realized return assumes the entry was fillable.** Always inspect
  `entry_limit_filled` and `entry_slippage` before drawing scoring
  conclusions — a pick can show a great `realized_return` while never
  filling at the quoted limit, in which case the win is fictional and
  tweaking scoring won't reproduce it.
- **Autonomous-mode picks lack a plan MD.** Research-mode improvement
  suggestions for those reports are necessarily shallower; you can
  diagnose scoring patterns but not the research process behind them.
- **One report ≠ a backtest.** A knob tweak that helps this report can
  hurt the next one. Recommend the user re-run self-correction on a
  different report after a few days of new picks land before treating
  any change as confirmed.
- **News scoring is at most ±5% (the `news_weight`).** If `realized_return`
  is dominated by ML signal accuracy, scoring tweaks have limited
  upside — flag the user toward `python -m stockpredict.cli backtest`
  for ML-side problems.
- **`pricing.entry_low_alpha` is sticky.** Changing alpha doesn't
  retroactively re-quote past limits — it only affects the next train
  + predict. The user must rerun `train` before stage-1 calibration
  improvements take effect.

Now, ask the user for the picks path and begin.
