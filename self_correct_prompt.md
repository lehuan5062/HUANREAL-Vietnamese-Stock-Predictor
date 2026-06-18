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

## Style

Always respond in simple, plain language. Short sentences. Explain jargon
the first time you use it (e.g. "fill_margin — how far the limit price sat
above or below the day's low"). Keep the findings precise and the numbers
exact, but say what they mean in everyday terms so the user doesn't have to
decode them.

## Your job

Take a picks report the user names, gather the realized outcomes (and/or
limit-fill outcomes) for those picks, diagnose what went wrong (and what
went right), and propose narrowly scoped edits to the program. Show every
diff and wait for explicit approval before applying — never silently
mutate files.

## Step 1 — Scan ledger for eligibility, then ask

This prompt is most often invoked **right after market close**: the user
just saw the buy day close and wants to know whether their entry limits
filled sensibly. The natural Stage-1 candidate at that moment is the
report whose `as_of` is the **previous trading day** (so today = T+0 =
buy day, which just closed). The natural Stage-2 candidate is any
report whose `target_date` = today.

**Before asking the user anything**, do an eligibility scan: pair each
recent picks JSON on disk with its ledger rows and classify each by
stage readiness. Don't just `ls reports/` and dump the list — the user
shouldn't have to compute eligibility in their head.

```
D:\stock\.venv\Scripts\python.exe -c "import glob, json, os, pandas as pd; df = pd.read_parquet(r'D:\stock\cache\predictions.parquet')
for p in sorted(glob.glob(r'D:\stock\reports\picks_claude_*.json'))[-8:]:
    d = json.load(open(p, encoding='utf-8')); rid = d['as_of'].replace('-','') + '_' + d['run_signature']; sub = df[df['run_id'] == rid]
    if len(sub) == 0: print(os.path.basename(p), 'as_of=' + d['as_of'], 'no ledger rows'); continue
    n = len(sub); t0 = int(sub['t0_evaluated'].sum()); ev = int(sub['evaluated'].sum())
    print(os.path.basename(p), 'as_of=' + d['as_of'], 'target=' + str(sub['target_date'].iloc[0]), 't0=' + str(t0) + '/' + str(n), 'eval=' + str(ev) + '/' + str(n))"
```

**Get the wall clock first.** The harness only injects today's date,
not the time. Run `date` (or `D:\stock\.venv\Scripts\python.exe -c
"import pandas as pd; print(pd.Timestamp.now(tz='Asia/Ho_Chi_Minh'))"`)
and pin `now_vn`. HOSE closes 15:00 Asia/Ho_Chi_Minh. T+0 (buy day) =
`as_of` itself (see `tracking.py:581`), so a report's buy day has
closed iff `now_vn > 15:00` on its `as_of` date — independent of any
ledger flag. The `t0_evaluated` flag tells you what's been STAMPED,
not what has HAPPENED in the market; don't read `t0_evaluated=False`
as "buy day hasn't closed."

Classify each row:

- All `evaluated=True` → **Stage-2 ready**.
- All `t0_evaluated=True` (some `evaluated=False`) → **Stage-1 ready**.
- Most-but-not-all `t0_evaluated=True` (e.g. 4/5) on a report whose buy
  day has clearly passed → **Stage-1 with stragglers** — offer to run
  `evaluate-fills` first to backfill; usually a transient lag, not a
  real exclusion.
- All `t0_evaluated=False`:
  - Buy day has closed (`now_vn > 15:00` on `as_of`) → **Stage-1 ready
    after ingest + evaluate-fills**. Today's OHLC bars likely not yet
    in `cache/ohlcv/<sym>.parquet`; offer to refresh data and stamp.
    **Never run a bare `update-data`** — that re-fetches the entire
    universe (minutes of API calls for data this diagnosis never
    reads). `evaluate-fills` already refreshes exactly the un-stamped
    symbols itself; run it directly. If a straggler persists after
    that, refresh just the picked symbols:

    ```
    D:\stock\.venv\Scripts\python.exe -m stockpredict.cli update-data -s <SYM1> -s <SYM2>
    ```

    (symbols from the report's `picks[].symbol`).
  - Buy day hasn't closed yet → not yet; ask the user to come back
    after 15:00 Asia/Ho_Chi_Minh on the buy day.

**Default-recommend** the freshest Stage-1-ready (or Stage-1-with-
stragglers, or Stage-2-ready) report, in that priority order. Phrase
question 1 as a confirmation, not an open-ended pick-from-list.

Then ask in plain conversation, one question at a time:

1. **Picks JSON path** (required). Default to the suggested report from
   the eligibility scan and frame as a confirmation, e.g.:

   > Today is the buy day for `picks_claude_<date>_<sig>.json` (4/5
   > rows already have `t0_evaluated=True`; I can backfill the 1
   > straggler with `evaluate-fills`). Run Stage 1 on that? (y / or
   > paste a different path)

   Only show the full list if the user declines the default or no
   report is currently stage-eligible.

2. **Stage** (optional). Either `1` (limit-fill only) or `2` (full
   T+N realized). If the user doesn't specify, infer from the ledger
   classification above and tell them what you decided.
3. **Extra context** (optional). Anything the user wants weighted into the
   diagnosis (e.g. "ignore VNM, it was a known data issue", "I executed only
   3 of the 5 picks"). Default empty.

After all are collected, summarise back and start Step 2.

## Step 2 — Resolve sibling files

`Read` the picks JSON. Pull these fields from it:

- `as_of`, `mode`, `exit_offset_days`, `hose_only`, `include_etfs`, `exclude`, `run_signature`, `selection`, `n_actionable` (plus mode-specific extras like `weight`, `global_summary`)
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
`20260506_claude_d11`. Compute it from `as_of` (strip dashes) and
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

- **Partial-gate with stragglers** (e.g. 4/5 `t0_evaluated=True` on a
  report whose buy day has passed): don't declare the gate failed yet.
  Run `evaluate-fills` to try to backfill the missing row, then
  re-check. If it backfills, proceed with Stage 1. If the straggler
  persists (e.g. ticker had no trades that day, was halted, or data
  feed is broken), tell the user and ask whether to (a) exclude that
  row and proceed with n-1, or (b) wait for the data. Don't silently
  drop rows.

Hold the per-pick data and stage-appropriate summary stats as working
evidence — they're inputs to findings, not output. Quote specific
numbers inside findings instead of emitting standalone tables or
summary blocks.

Per-pick columns to keep handy. `entry_limit_price` (the actual price
paid if filled) is **mandatory** in both stages — never omit it, and
never substitute the close anchor. Stage-1 calibration is judged on
`entry_limit_price` vs `t0_low` (the realized intraday low) directly,
never via the close.

- Stage 2: `symbol, pred_mean, news_score, adjusted, entry_limit_price,
  t0_low, entry_limit_filled, fill_margin, realized_return,
  dimensions_cited`.
- Stage 1: `symbol, pred_mean, pred_low, entry_limit_price, t0_low,
  entry_limit_filled, fill_margin`.

Define `fill_margin = (entry_limit_price − t0_low) / entry_limit_price`.
Sign convention: **positive = filled with that much slack below the
limit (money left on the table); negative = unreachable, missed by that
much.** Use this column to read calibration at a glance — large
positive on filled rows means "limits too generous"; small negative on
unfilled rows means "limits just barely missed"; large negative means
"limits unreachable, too bearish." Always quote both the % form and
the VND form (limit − low in absolute price) for the user's executed
picks.

`entry_slippage` (close-anchored legacy "buy at close" metric) is
informational only — do not anchor Stage-1 findings on it.

Summary stats to compute:
- Stage 2: n picks, hit-rate, mean realized, mean(predicted) vs
  mean(realized) gap, fill_rate, mean(fill_margin), and — when scoring
  signals are in question — break-down by news_score / dimension on
  realized return.
- Stage 1: n picks, fill_rate, mean(fill_margin) across all evaluable
  rows, mean filled-slack (mean fill_margin over filled rows only,
  n_filled), mean miss-margin (mean fill_margin over unfilled rows only,
  n_unfilled). All three margins answer the same question — was the
  limit reachable at the realized low — so all three must anchor on
  `entry_limit_price` vs `t0_low`, never on the close.

## Step 4 — Cross-reference with the broader ledger

Pull the broader by-dimension and by-news_score stats so the diagnosis isn't
based solely on one report. Use the existing helper:

```
D:\stock\.venv\Scripts\python.exe -c "from stockpredict.tracking import recent_performance; import json; print(json.dumps(recent_performance(window_days=90, mode='claude'), indent=2, default=str))"
```

**Fallback if `limit_fill` returns null.** The `recent_performance`
helper was patched on 2026-05-13 to compute `limit_fill` on the
t0-eligible slice (not the fully-evaluated slice), so this should
populate as long as t0-stamped rows exist. If a future regression
returns `null` again, compute pool fill stats directly from the
parquet:

```
D:\stock\.venv\Scripts\python.exe -c "import pandas as pd
df = pd.read_parquet(r'D:\stock\cache\predictions.parquet')
pool = df[df['t0_evaluated'] & df['entry_limit_price'].notna()].copy()
pool['fill_margin'] = (pool['entry_limit_price'] - pool['t0_low']) / pool['entry_limit_price']
print('n=' + str(len(pool)), 'fill_rate=' + str(round(pool['entry_limit_filled'].mean(), 3)), 'mean_fm=' + str(round(pool['fill_margin'].mean(), 4)))
print('fm_filled=' + str(round(pool.loc[pool['entry_limit_filled'], 'fill_margin'].mean(), 4)), 'n_filled=' + str(int(pool['entry_limit_filled'].sum())))
print('fm_unfilled=' + str(round(pool.loc[~pool['entry_limit_filled'], 'fill_margin'].mean(), 4)), 'n_unfilled=' + str(int((~pool['entry_limit_filled']).sum())))"
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

Write findings as a numbered list. **Every finding must drive at least
one proposed edit in Step 6.** If a pattern doesn't suggest a concrete
edit, skip it — don't write it down as an interesting observation.

**Evidence threshold**: each finding must cite ≥3 same-direction misses
on this report, **or** the report's pattern echoes an n≥5 pattern in
the broader ledger. Single-pick patterns don't clear the threshold and
should be skipped, not logged.

For each finding, name the failure mode plainly. Stage-1 examples:

- "0 of 5 limits filled. Mean `fill_margin` was −1.0% (every
  `entry_limit_price` sat ~100bp ABOVE the realized `t0_low`; limits
  would have needed to be ~100bp deeper to reach the low). The low
  head is too bearish on dips for this signature; pooled fill_rate at
  the same alpha (0.5) is 48% (n=87), so today's report sat in a
  one-sided up-day."
- "All 5 limits filled with mean `fill_margin` +2.5% (every
  `entry_limit_price` was ~250bp above `t0_low` — limits could have
  been ~2.5% lower per pick and still filled at the realized low,
  meaning ~2.5% of cost basis per share was left on the table
  per pick). The low head is too conservative on dips; lower
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
naming aligns and re-runs at a different stage don't clobber). Keep the
report lean — findings + proposed edits + applied tracking only. Don't
inline metadata blocks, per-pick tables, or summary-stat blocks; the
filename identifies the report and the ledger holds the data. Structure:

```
# Self-correction — picks_claude_<date>_<sig>  (Stage <N>)

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
   - **Per-run plan/prompt guidance** (the research method, the
     geopolitical check, and the **VN-Index trend call**) is generated
     by `src/stockpredict/news/claude_runner.py` (`write_plan`) and
     `src/stockpredict/news/gemini_prompt.py` (`build_prompt`), NOT
     `claude_prompt.md`. Refining that wording is a legitimate
     narrowly-scoped **prose** edit there — treat it like prompt
     guidance, not as the "structural defect only" gate on #3. The
     VN-Index trend call is the LLM's own UP/SIDEWAYS/DOWN view (there
     is no quantitative index model). It surfaces in each run's
     `global_summary`: sanity-check it against what the index actually
     did over the holding window — if the call is consistently wrong or
     the pick-tilt is too strong/weak, tighten that instruction (e.g.
     demand a confidence threshold, or tell it to lean less on the index
     for counter-trend names). Don't add a config knob or model for it.
2. `config.yaml` — for tunable knobs:
   - `modes.claude.news_weight` (current 0.05) — raise / lower if
     news_score is consistently mis-weighted relative to ML.
   - `pricing.stop_atr_mult` (current 1.5), `pricing.min_rr_ratio`
     (current 0.8) — for entry/exit rule mismatches.
   - **`pricing.entry_low_alpha` (currently 0.25) — sets the quantile
     level of the per-ticker rolling empirical low head (since
     2026-06-05; see memory `project_low_head_negative_skill.md` — do
     NOT reintroduce an ML low head). Raise toward 0.75 if
     fill_rate is consistently below alpha AND `fill_margin` on
     unfilled rows is consistently small-negative (we're too bearish
     on dips, but only just); lower toward 0.25 if `fill_margin` on
     filled rows is consistently large-positive (we're filling with
     lots of slack — money on the table). Both signals must anchor on
     `entry_limit_price` vs `t0_low`, never on the close. This is the
     primary knob for stage-1 findings. First rule out a one-sided
     melt-up regime (low fills market-wide ≠ miscalibration). The
     trailing window auto-sizes with alpha via
     `pricing.entry_low_target_tail_obs` (default 15); an alpha change
     needs a `train` to take effect. Before any of this, check the picks
     JSON: if `adj_entry_vnd != entry_vnd` for a pick, the news stage
     overrode the entry and the user likely placed the ADJUSTED order —
     the ledger's mechanical `entry_limit_price` / `fill_margin` for that
     row reflects a limit that was never placed, so exclude those rows
     before judging fill calibration or tuning this knob.**
   - `universe.liquidity_filter.min_adv_active_days` (currently 15 of 20)
     — raise if the user reports a pick was effectively untradeable (thin
     volume / "I was the only buyer"). That's a universe-liquidity problem,
     NOT a scoring or `entry_low_alpha` problem — don't tune alpha for it.
     Note: this isn't ledger-observable (a thin name can still fill), so it
     only surfaces via user-supplied extra context in Step 1.
   - `pricing.max_participation_pct` (currently 1.0) — the advisory unit-cap
     participation rate (% of `adv_vnd_20`) that drives `suggested_max_units`.
     Lower it if the user reports they couldn't exit a position near the quoted
     target without walking the price down (the suggested cap was too generous
     for the real book); raise it if the cap is so small it's not useful.
     Like `min_adv_active_days`, this is a liquidity/sizing knob — it's advisory
     and never feeds the actionable gate, so it isn't ledger-observable and only
     surfaces via user-supplied extra context in Step 1. Set to 0 to disable the
     suggestion entirely.
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

Remind the user: one report ≠ a backtest — a knob tweak that helps
this report can hurt the next one. Suggest re-running self-correction
on a different report after a few days of new picks land before
treating any change as confirmed.

## What NOT to do

- Don't run a full-universe `update-data` (no `-s`). Self-correction
  only ever needs bars for the report's picked symbols, and
  `evaluate-fills` fetches those itself. Always pass `-s <symbol>` if
  a manual refresh is needed at all.
- Don't ask the user to blindly pick from a list of all reports. Run
  the eligibility scan (Step 1) first, classify each recent report by
  stage readiness, and default-suggest the right one. The user should
  confirm, not compute.
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
- Don't anchor Stage-1 calibration on the close. The whole point of
  the limit-order regime is "was my `entry_limit_price` reachable at
  the realized `t0_low`" — answered by `fill_margin =
  (entry_limit_price − t0_low) / entry_limit_price`, NOT by comparing
  to `t0_close` or to yesterday's close. `entry_slippage` is the
  legacy close-anchored metric and is diagnostic only — never lead a
  Stage-1 finding with it. Likewise, `entry_limit_price` must appear
  in every per-pick table you present; do not silently omit it.
- Don't propose changes to `dimensions_cited` parsing or the ledger
  schema unless you've shown a concrete bug, not a stylistic preference.
- Don't tune `entry_low_alpha` (or read Stage-1 fill calibration) off
  rows where the news stage overrode the entry (`adj_entry_vnd !=
  entry_vnd` in the picks JSON). The ledger records only the mechanical
  `entry_limit_price`, so its `fill_margin` for those rows describes a
  limit the user didn't place — it says nothing about the low head.

Now, ask the user for the picks path and begin.
