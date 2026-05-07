# Vietnamese T+N Stock Predictor — Self-correction prompt

Paste the contents of this file into a Claude Code or Cowork session in Claude
Desktop. Claude will then run a one-shot **self-correction** pass over a picks
report you choose: cross-reference the predictions with what actually happened,
diagnose systematic errors, and propose program-level edits to
`claude_prompt.md` and `config.yaml` so future runs do better.

This is the **active** counterpart to the passive feedback-block loop already
baked into `claude_prompt.md`. The passive loop adjusts scoring *within a
single run*; this prompt mutates the program itself.

---

You are operating the Vietnamese T+N swing-trade stock predictor at `D:\stock`.
Your tools: `Bash`, `Read`, `Edit`. No `WebFetch`/`WebSearch` needed — the
evidence for self-correction lives entirely on disk.

## Your job

Take a picks report the user names, gather the realized outcomes for those
picks, diagnose what went wrong (and what went right), and propose narrowly
scoped edits to the program. Show every diff and wait for explicit approval
before applying — never silently mutate files.

## Step 1 — Ask the user for the report

Ask in plain conversation, one question at a time:

1. **Picks JSON path** (required). Absolute path to a finalized picks file,
   typically `D:\stock\reports\picks_claude_<YYYY-MM-DD>_<sig>.json`. List the
   available reports first if the user is unsure:
   ```
   D:\stock\.venv\Scripts\python.exe -c "import os, glob; [print(p) for p in sorted(glob.glob(r'D:\stock\reports\picks_claude_*.json'))]"
   ```
2. **Extra context** (optional). Anything the user wants weighted into the
   diagnosis (e.g. "ignore VNM, it was a known data issue", "I executed only
   3 of the 5 picks"). Default empty.

After both are collected, summarise back and start Step 2.

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

## Step 3 — Pull realized outcomes from the ledger

Read `D:\stock\cache\predictions.parquet` directly. The cleanest one-liner:

```
D:\stock\.venv\Scripts\python.exe -c "import pandas as pd; df = pd.read_parquet(r'D:\stock\cache\predictions.parquet'); df = df[df['run_id'] == '<RUN_ID>']; print(df[['symbol','pred_mean','news_score','adjusted','entry_price','actual_exit','realized_return','evaluated','entry_slippage','t0_low','dimensions_cited','target_date']].to_string(index=False))"
```

Where `<RUN_ID>` is `<YYYYMMDD>_<run_signature>` — e.g.
`20260506_claude_d11_u100`. Compute it from `as_of` (strip dashes) and
`run_signature`.

**Refusal gate**: if any row in the run has `evaluated=False`, stop. Tell the
user:

> The run signature `<run_id>` has unevaluated rows. Without realized returns
> there's nothing to self-correct from. Please run `evaluate.bat` (or
> `python -m stockpredict.cli evaluate`) and rerun this prompt, or pick an
> older report whose target date has fully elapsed.

Don't proceed past this gate even if the user pushes — the diagnosis would
be hallucinated.

If all rows are evaluated, present a per-pick table showing:

| symbol | pred_mean | news_score | adjusted | realized_return | dimensions_cited | entry_slippage |

Plus a one-line summary (n picks, hit-rate on this run, mean realized,
mean(predicted) vs mean(realized) gap).

## Step 4 — Cross-reference with the broader ledger

Pull the broader by-dimension and by-news_score stats so the diagnosis isn't
based solely on one report. Use the existing helper:

```
D:\stock\.venv\Scripts\python.exe -c "from stockpredict.tracking import recent_performance; import json; print(json.dumps(recent_performance(window_days=90, mode='claude'), indent=2, default=str))"
```

Compare:

- Did this report's news_score=+1 picks lose, while the 90-day pooled
  news_score=+1 stat is winning? → likely a *this-report* problem (specific
  dimensions / sectors), not a *scoring-model* problem.
- Did this report's `[<some-tag>]` dimension lose, *and* the pooled
  by-dimension table for that tag also shows it losing (n ≥ 5)? → systemic
  weakness in how that dimension is being researched or scored.
- Is `entry_slippage` systematically positive (mean > 0) for this run? →
  the quoted entries are unfillable; realized_return is fictional, and
  scoring tweaks won't help — pricing tweaks (e.g. `pricing.stop_atr_mult`
  or entry rule) might.

## Step 5 — Diagnose

Write findings as a numbered list with **evidence threshold**: each finding
must cite ≥3 same-direction misses on this report, **or** the report's
pattern echoes an n≥5 pattern in the broader ledger. Single-pick findings
get logged but do **not** drive proposed edits.

For each finding, name the failure mode plainly:

- "All 4 picks tagged `[earnings]` lost (mean −1.8%); pooled `[earnings]`
  hit-rate is 45% (n=18). Earnings is being weighted higher than its
  historical predictiveness justifies."
- "3 of 5 picks had `entry_slippage > 0` (entries unreachable); the quoted
  `entry_vnd` is systematically below the buy-day low. Scoring is fine; the
  entry rule is the problem."
- "All `+1` scores were on real-estate names; all 3 lost. Real-estate
  research relied solely on `[sector-flow]` without checking
  `[regulatory]` or `[capital-raise]`. Coverage gap, not a scoring
  miscalibration."

Skip categories where the evidence isn't there. Empty diagnosis is a valid
outcome — say "no systemic pattern found" and stop, don't manufacture one.

## Step 6 — Propose edits

Write the diagnosis and proposals to
`D:\stock\reports\self_correction_<YYYY-MM-DD>_<sig>.md` (use the picks
report's `as_of` and `run_signature` so naming aligns). Structure:

```
# Self-correction — picks_claude_<date>_<sig>

## Inputs
- Picks: <path>
- Plan: <path or "none — autonomous mode">
- Run id: <run_id>
- n picks, n evaluated, hit-rate, mean realized

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
3. Source files (e.g. `src/stockpredict/news/claude_runner.py`,
   `src/stockpredict/tracking.py`) — **only** when there's a concrete
   structural defect (parser bug, missing column, wrong formula). Default
   to **not** touching code; if the right fix is structural, write the
   finding, propose the change in the report, and **stop** — let the user
   trigger a separate code-change task.

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

## What NOT to do

- Don't propose edits from a single losing pick. Evidence threshold = ≥3
  on-report or ≥5 in pooled ledger.
- Don't auto-apply. Always show the diff and wait for per-file approval.
- Don't touch source files when a `config.yaml` knob would do.
- Don't rewrite `claude_prompt.md` wholesale. Additive, narrowly-scoped
  edits only — adding one line, tightening one rule, expanding one example.
- Don't ignore `evaluated=False` rows. Refuse and point at `evaluate.bat`.
- Don't fabricate a finding to fill the report. "No systemic pattern
  found" is a valid outcome.
- Don't conflate scoring failures with execution failures. If
  `entry_slippage > 0` cluster is the issue, scoring tweaks won't help.
- Don't propose changes to `dimensions_cited` parsing or the ledger
  schema unless you've shown a concrete bug, not a stylistic preference.

## Caveats to note in the produced report

- **Realized return assumes the entry was fillable.** Always inspect
  `entry_slippage` and `% unreachable` before drawing scoring conclusions
  — a pick can show a great `realized_return` while being unfillable in
  practice, in which case the win is fictional and tweaking scoring
  won't reproduce it.
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

Now, ask the user for the picks path and begin.
