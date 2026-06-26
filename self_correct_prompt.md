# Vietnamese T+2 Stock Predictor — Self-correction prompt

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

**Scope: the two edit-driving focuses are ML / hybrid only — NOT LLM-only.**
Both focuses below are ML-pipeline concepts: missed-winners regret diagnoses the
ML model's skill and its gates, and entry-price calibration tunes the mechanical
low head. The **LLM-only** Claude method (`picks_claude_llm_*` files, `mode:
claude_llm`, `method: llm_only`) uses no ML model — there is no `pred_mean`, no
`train-missed` variant, and no mechanical `entry_limit_price` to calibrate (the
LLM sets prices itself). So the **self-correction focuses do not apply** to
LLM-only reports: skip them in the eligibility scan, and if the user names one
*for self-correction*, say it's out of scope (nothing here to retrain or
recalibrate) rather than inventing a diagnosis.

**Exception — the cross-method comparison (Step 4c) DOES include LLM-only.**
That step is read-only and advisory (it ranks which *method* has been winning,
base vs hybrid vs LLM-only); it produces no edits, so the "no ML to tune"
limitation doesn't block it.

## Two stages

There are two evaluation stages for any picks report. **Both can run
independently**, and you should choose one based on how much time has
elapsed since the picks were emitted:

* **Stage 1 — Limit-fill self-correction (after the BUY DAY closes)**.
  Fires the trading day after `as_of`. The buy day's OHLC is now in the
  ledger (`t0_evaluated=True`), which means we know whether each pick's
  `entry_limit_price` (from the low-prediction quantile head) actually
  filled. We do **not** know the realized return yet — that needs T+2
  to elapse. Diagnoses focus narrowly on **entry calibration**: was the
  predicted dip too aggressive (limits unreachable, no fill) or too
  conservative (limits filled cheaply but only a tiny dip captured)?
  **Even at Stage 1 you MUST still RUN the missed-winners regret look
  (Step 4b) and the recency/trend sanity-check (Step 4d)** — both pool
  already-evaluated history (or read the picks' own trend features), so
  they are valid before T+2. They are advisory at Stage 1 (Focus-1
  *edits* still wait for Stage 2), but they are NEVER skipped, and never
  only because the user asked.
* **Stage 2 — Full self-correction (after T+2 closes)**. All rows have
  `evaluated=True` and `t0_evaluated=True`, so realized returns are known. This
  unlocks **Focus 1 (missed winners)** *edits* — the realized top-N the model
  didn't surface, and why (Step 4b) — in addition to **Focus 2 (entry-price
  misses)**. (The missed-winners look itself is RUN in both stages; Stage 2 is
  only what lets it drive an edit.) Those are the ONLY two things this prompt
  diagnoses; it does not chase news_score / dimension-tag patterns anymore.

When you run this prompt, **first** check the ledger to see which stage
the chosen picks report is eligible for, then proceed accordingly. Stage
1 is permitted before Stage 2. You can also run Stage 1 today and Stage
2 again later on the same report.

---

You are operating the Vietnamese T+2 swing-trade stock predictor. **Run every
command below from the repo root** — `cd` into your clone first; all paths in
this prompt are relative to it (the project virtualenv lives at
`.venv\Scripts\python.exe`).
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
.venv\Scripts\python.exe -c "import glob, json, os, pandas as pd; df = pd.read_parquet(r'cache\predictions.parquet')
for p in sorted(glob.glob(r'reports\picks_*.json'))[-16:]:
    d = json.load(open(p, encoding='utf-8'))
    if d.get('method') == 'llm_only' or os.path.basename(p).startswith('picks_claude_llm_'): continue  # LLM-only: out of scope (no ML to retrain/recalibrate)
    rid = d['as_of'].replace('-','') + '_' + d['run_signature']; sub = df[df['run_id'] == rid]
    var = d.get('model_variant', 'standard')
    if len(sub) == 0: print(os.path.basename(p), 'as_of=' + d['as_of'], 'variant=' + var, 'no ledger rows'); continue
    n = len(sub); t0 = int(sub['t0_evaluated'].sum()); ev = int(sub['evaluated'].sum())
    print(os.path.basename(p), 'as_of=' + d['as_of'], 'variant=' + var, 'target=' + str(sub['target_date'].iloc[0]), 't0=' + str(t0) + '/' + str(n), 'eval=' + str(ev) + '/' + str(n))"
```

This globs **all** report types: base (`picks_<date>_base_d2…`), claude
(`picks_claude_…`), gemini (`picks_gemini_…`), AND the missed-winners variant
(`…_base_d2_missed_…`, `model_variant: missed`). It **skips LLM-only reports**
(`picks_claude_llm_*`, `method: llm_only`) — they're out of scope (see the note
at the top: no ML model to retrain, no mechanical limit to recalibrate). The missed variant appears
differently by mode: **base** writes a separate **standard + `_missed` pair**
(treat them together), while **claude/gemini** fold it into a single **union
report** (per-pick `also_missed` / `missed_only` flags). The mode/variant come
from each JSON's `mode` / `model_variant` / `run_signature` fields.

**Get the wall clock first.** The harness only injects today's date,
not the time. Run `date` (or `.venv\Scripts\python.exe -c
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
    .venv\Scripts\python.exe -m stockpredict.cli update-data -s <SYM1> -s <SYM2>
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
   T+2 realized). If the user doesn't specify, infer from the ledger
   classification above and tell them what you decided.
3. **Extra context** (optional). Anything the user wants weighted into the
   diagnosis (e.g. "ignore VNM, it was a known data issue", "I executed only
   3 of the 5 picks"). Default empty.

After all are collected, summarise back and start Step 2.

## Step 2 — Resolve sibling files

`Read` the picks JSON. Pull these fields from it:

- `as_of`, `mode`, `exit_offset_days`, `hose_only`, `include_etfs`, `exclude`, `run_signature`, `selection`, `requested_picks`, `n_picks`, `n_below_breakeven` (plus mode-specific extras like `weight`, `global_summary`)
- `weight` (the `news_weight` that was active at finalize time)
- `plan_file` (interactive mode only — points at the news plan markdown)
- `picks` (the array; each item has `symbol`, `pred_mean`, `news_score`,
  `adjusted`, `entry_vnd`, `target_vnd`, `stop_vnd`, `below_breakeven`,
  `pred_low_alpha` (now **varies per pick** — the dip quantile is conviction-
  coupled, NOT a constant), plus the
  `business`, `dimensions`, `drivers`, `key_news`, `dimensions_cited` fields
  Claude wrote at finalize time)
- `model_variant` (base mode: `standard` or `missed`). **claude/gemini union
  reports** additionally carry `also_missed` / `missed_only` per pick — whether
  the missed-winners variant also surfaced that name (the LLM already weighed the
  A/B verdict at emit and kept the top N from the union).

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

Read `cache\predictions.parquet` directly. The cleanest one-liner:

```
.venv\Scripts\python.exe -c "import pandas as pd; df = pd.read_parquet(r'cache\predictions.parquet'); df = df[df['run_id'] == '<RUN_ID>']; print(df[['symbol','pred_mean','news_score','adjusted','entry_price','entry_limit_price','pred_low','t0_low','entry_limit_filled','t0_evaluated','actual_exit','realized_return','evaluated','entry_slippage','dimensions_cited','target_date']].to_string(index=False))"
```

Where `<RUN_ID>` is `<YYYYMMDD>_<run_signature>` — e.g.
`20260506_claude_d11`. Compute it from `as_of` (strip dashes) and
`run_signature`.

**Stage gates** (pick the highest stage that fully passes):

- **Stage 2 gate**: every row has `evaluated=True`. → run the full T+2
  diagnosis below.
- **Stage 1 gate**: every row has `t0_evaluated=True` (but at least one
  row has `evaluated=False`). → run the limit-fill-only diagnosis.
- **Neither gate passes**: stop. Tell the user:

  > Run `.venv\Scripts\python.exe -m stockpredict.cli evaluate-fills`
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
.venv\Scripts\python.exe -c "from stockpredict.tracking import recent_performance; import json; print(json.dumps(recent_performance(window_days=90, mode='claude'), indent=2, default=str))"
```

**Fallback if `limit_fill` returns null.** The `recent_performance`
helper was patched on 2026-05-13 to compute `limit_fill` on the
t0-eligible slice (not the fully-evaluated slice), so this should
populate as long as t0-stamped rows exist. If a future regression
returns `null` again, compute pool fill stats directly from the
parquet:

```
.venv\Scripts\python.exe -c "import pandas as pd
df = pd.read_parquet(r'cache\predictions.parquet')
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
  - **The dip alpha is a DYNAMIC RANGE**, not one value: per-pick
    `pred_low_alpha = entry_low_alpha (base) × conviction-multiplier ×
    overbought-penalty`. So each pick targets its OWN alpha (≈ its expected
    fill probability) and a single fill-rate target no longer applies.
    **Group fill_rate along BOTH axes** that move the alpha: by conviction
    tier (split on `below_breakeven` / bucket `pred_low_alpha`) AND by
    overbought-ness (bucket `rsi_14`, e.g. < 60 vs 60–85). A LOW fill rate
    on the deep-dip picks — whether deep because they're *weak* (conviction)
    or because they're *overbought* (penalty) — is BY DESIGN, NOT
    miscalibration. Only the within-bucket gap (filled rate vs that bucket's
    mean `pred_low_alpha`) signals a real problem, and it tells you WHICH
    component of the range to tune (base / conviction mults / overbought).
  - If a tier's fill_rate ≈ its `pred_low_alpha` AND this run is way off
    in the same direction (e.g. 0% fill on a high-alpha tier), this
    report's tickers were unusually directional (gap-ups), not a model
    issue.
  - If a tier's fill_rate is far from its `pred_low_alpha` across the
    board, the low head is mis-calibrated — propose retraining or
    adjusting the base/multiplier knobs below.

This is **Focus 2 (entry-price misses)** — see Step 5. `entry_slippage` is a
legacy close-anchored metric; `entry_limit_filled` / `fill_margin` are the
truer measurements, so don't anchor findings on slippage.

## Step 4b — Missed winners (Focus 1)

This is the highest-value question: **which winners did we miss, and why?**
Run the analysis **anchored to THIS report's buy day** — pass its `as_of`:

```
.venv\Scripts\python.exe -m stockpredict.cli regret --on <as_of>
```

This is a **single-day** view (NO 90-day aggregation): the realized top-N liquid
tickers for ONE closed window — bought T-2, sold the eval day — and whether the
model surfaced each. The eval day is the prediction's **exact T+2 day** when it
has closed (Stage 2 / running at T+3 or later); if T+2 hasn't closed yet (e.g.
self-correcting today's prediction), it falls back to the **latest fully-closed
[T-2 -> today] window**. Returns are REAL for that one window — never a stale
spike a name posted weeks ago and has since round-tripped. The liquidity universe
matches live selection, so these are winners the model *could* have picked. Use
`-n N` for a longer winners list and `--signature <sig>` to compare against one
run only. (The old 90-day magnitude-sorted `--window` view is gone — it headlined
since-crashed names like VIW and was removed; do not reintroduce it.)

For the worst missed winners, investigate **why** each was missed — load its
T-2 features (from that day's `.candidates.parquet` sidecar, or rebuild the
panel) and check:
- Was it **excluded by a gate**? (`overbought_rsi_max` dropped it as overbought;
  `corp_action_lookback` flagged a band-break; ceiling-lock; liquidity.) If a
  gate is systematically discarding winners, that's your edit.
- Was it **scored low** by the model (low `pred_mean`)? Then it's a model-skill
  gap — note it; the lever is the `train-missed` variant (below), not a knob.
- The missed-winners variant shows up differently per mode:
  * **base** writes a **standard + `_missed` pair** — contrast them for the same
    `as_of` (cross-check `regret --signature base_d2` vs `base_d2_missed`): did
    the `_missed` report surface winners the standard one missed?
  * **claude/gemini** write a **single union report** — each pick's
    `also_missed` / `missed_only` flag tells you whether the variant agreed; a
    `missed_only` pick that won (or lost) is direct evidence on the variant.
  Either way, consult the latest `reports/backtest_ab_<date>.md` (from
  `backtest-ab` / `run --ab`) for the out-of-sample verdict: only treat the
  variant as worth using if its `hit_rate` ≥ standard there. If no A/B report
  exists, propose running `backtest-ab` — never recommend the variant on a
  single report alone.

Propose ONE improvement: a `config.yaml` knob (e.g. relax `overbought_rsi_max`)
or a `claude_prompt.md` prose edit. If the miss is pure model skill, the
proposal is to run `train-missed` + `backtest-ab` and promote the variant only
if its win rate holds.

**The flip side — overbought losers WE picked.** Same focus, opposite sign: do
the report's OWN surfaced picks lose *because they were overbought tops*? The
picks JSON carries `rsi_14` and the ledger has `realized_return`, so check: of
the picks with high `rsi_14` (say > 75), what fraction had `realized_return < 0`?
If ≥3 (or an n≥5 pooled pattern) overbought picks reversed, that's the trigger
to ENABLE / TIGHTEN the hard gate:
- gate currently off (`overbought_rsi_max: 0`) → propose **setting** it (e.g.
  `78`–`82`, just below where the losers clustered);
- gate already on but losers slipped under it → propose **lowering** it.
With the gate off you'll never see "winners dropped by the gate," so this
loser-side check is the ONLY path that turns the gate on — don't skip it.
(Before reaching for the hard gate, recall the soft entry penalty
`entry_alpha_overbought_*` is already deepening these picks' entries; if they
filled anyway and reversed, the penalty may also be too shallow.)

## Step 4c — Cross-method comparison (optional, advisory)

**When to run this**: only if the same day has picks from **two or more
methods** (base / hybrid / LLM-only / gemini) that have since evaluated — e.g.
the user ran `--mode base`, `--mode claude`, and `--mode claude --llm-only` on
the same date with the same `--picks` / horizon / hose-only / etfs / exclude. If
only one method ran per day, skip this step entirely.

This answers a different question from the two focuses: **which prediction
*method* has been picking better?** It is purely advisory — mode is a per-run
user choice, so there is **no config knob to tune and no edit to propose** from
it. Do not turn its output into a Step 6 edit; just surface the verdict to the
user.

A single day is far too noisy (often 1–3 picks per method), so the verdict pools
over a window; the named day is shown only as context. Run:

```
.venv\Scripts\python.exe -m stockpredict.cli compare-modes --window 90 --date <as_of>
```

(`--date` is the report's `as_of`; drop it to skip the single-day context.) It
restricts to **comparable cells** — same trading day AND same run parameters
(the run signature minus its mode token, so `base_d2` / `claude_d2` /
`claude_llm_d2` all match, but `claude_d2_HOSE` does not) — pools realized
returns per method over the window, prints a head-to-head + an "unique vs
shared" breakdown (how each method's *distinctive* picks did), and writes
`reports/mode_comparison_<date>.md`.

Read the verdict to the user in plain language: which method has the better
`mean/day` and head-to-head win count over the window, and whether its edge
comes from *distinctive* picks (high `unique mean`) or just from agreeing with
the others. Flag explicitly when the sample is thin (few comparable days) so the
user doesn't over-trust a noisy edge. **Do not recommend changing the default
method off a handful of days** — note the trend and suggest re-checking as more
picks evaluate.

## Step 4d — Recency / trend sanity-check (BOTH stages, MANDATORY)

Never call a name a "winner", and never frame surfacing it as a good thing,
without looking at what its price actually did over the window you're evaluating
(and since). Two checks, both required on every run (Stage 1 included):

1. **Regret winners are now single-day by construction** (Step 4b uses `regret
   --on <as_of>`), so the returns shown ARE the realized move for the one closed
   [T-2 -> eval] window — no stale 90-day spikes. You normally don't need to
   re-pull bars. The one case that still warrants a chart glance: if you are
   citing a winner to argue the model *should chase that kind of name going
   forward*, confirm it hasn't since round-tripped (a name can win one window and
   reverse the next):

   ```
   .venv\Scripts\python.exe -c "import pandas as pd; d=pd.read_parquet(r'cache\ohlcv\<SYM>.parquet'); print(d.tail(15).to_string())"
   ```

   A name that won its window but has since crashed is not a template to chase.

2. **The report's OWN picks — flag any name surfaced into a downtrend.** The
   picks JSON carries `mom_5`, `mom_20`, `rsi_14`. A pick with clearly negative
   `mom_5` (a recent multi-day decline) — **especially when `mom_20` is positive**
   (a faded, post-spike name rolling over) — means the model surfaced a FALLING
   name: chasing momentum that already ended, or catching a knife. ALWAYS confirm
   against the actual chart (`cache\ohlcv\<SYM>.parquet` tail) — the trend
   features alone are not enough, and a falling chart overrides any bullish
   `pred_mean` or news narrative. If a pick is in a clear downtrend at entry, state
   it **bluntly as a finding** — never soften it to a "watch item." At Stage 1
   this is advisory.

   **The lever is NOT a `claude_prompt.md` edit.** `claude_prompt.md` explicitly
   forbids the LLM from scoring on technicals ("Don't score on technicals (RSI,
   momentum, etc.) — those are the ML input"), so momentum/trend is by design the
   ML model's job. A downtrend-chasing pattern therefore points at either the ML
   model (it has `mom_5`/`mom_20` as inputs and under-weighted them — a retrain /
   feature question) or a **mechanical selection guard** (a new `config.yaml`
   knob, e.g. a `mom_5` floor that excludes or penalizes names in a fresh
   downtrend before ranking). Both are heavier than a prose tweak, so **require an
   n>=5 systematic pattern across evaluated reports before proposing them** — one
   downtrend pick (VIW) does not justify a model or selection-logic change. Note
   that `mom_5`/`mom_20` only began landing in the picks JSON on 2026-06-26, so
   that evidence base has to accumulate first.

This step exists because a self-correction once reported VIW as a missed "+25%
winner" and described the model "catching" it — while VIW had in fact peaked
2.5 weeks earlier and was down ~21% from that peak and falling at the time it was
surfaced (`mom_5 = -14%`). Do not repeat that: look at the chart.

## Step 5 — Diagnose (two focuses only)

Diagnose **exactly two things**, nothing else: **(1) missed winners** (Step 4b)
and **(2) entry-price misses** (the fill calibration in Step 4). Write findings
as a numbered list. **Every finding must drive at least one proposed edit in
Step 6.** If a pattern doesn't suggest a concrete edit, skip it.

The Step 4d recency/trend check feeds **both** focuses, it is not a third focus:
a stale/round-tripped "winner" *corrects* a Focus-1 regret claim (drop it — do
not count it as a real miss), and a pick surfaced into a clear downtrend is a
Focus-1 flip-side / entry-timing finding (the model chased a faded name). A
downtrend pick that also failed to fill is BOTH a Focus-2 (the limit was deep
because the empirical low head saw the fall) and a Focus-1 finding — say so.
Even with n=1 on the report, a pick bought into an unmistakable downtrend MUST be
stated as a finding (it overrides the ≥3 threshold below, which exists to stop
over-fitting *calibration* noise, not to suppress an obvious bad entry).

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

Missed-winner examples (Focus 1):

- "6 of the last 20 realized top-5 winners were missed; 4 of the 6 had RSI > 80
  at T-2 and were dropped by the `overbought_rsi_max` gate, then ran another
  +6% on average. The gate is too tight for the win side — propose raising
  `overbought_rsi_max` from 75 to 82 (or rely on the soft entry penalty alone)."
- "5 missed winners all scored low `pred_mean` at T-2 (bottom tercile) — the
  model genuinely didn't see them; not a gate. Propose `train-missed` +
  `backtest-ab`; promote the `_missed` variant only if win rate holds."

Skip categories where the evidence isn't there. Empty diagnosis is a valid
outcome — say "no systemic pattern found" and stop, don't manufacture one.

## Step 6 — Propose edits

Write the diagnosis and proposals to
`reports\self_correction_<YYYY-MM-DD>_<sig>_stage<N>.md` (use the
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
2. `config.yaml` — for tunable knobs (only the two-focus-relevant ones):
   - **`pricing.overbought_rsi_max` (current 0 = off) — the overbought hard
     gate** (lower value = stricter / excludes more; `0` = off). Two Focus-1
     triggers: if missed winners are being *dropped* by the gate (RSI just above
     the cap, then they ran), **raise** it (more permissive) or set `0`; if the
     report's own picks are overbought tops that *reverse and lose* (the flip-
     side check in Step 4b), **enable/tighten** it — e.g. set `0 → ~80` from off,
     or lower an already-on cap. Paired soft levers:
     `pricing.entry_alpha_overbought_start` / `_full` / `_mult` deepen the entry
     dip for overbought names (so they only fill on a pullback) without excluding
     them — prefer tuning these over the hard gate when the win side is hurt.
     These same penalty knobs are ALSO an entry-calibration (Focus-2) lever —
     see the `entry_low_alpha` family below for the fill-rate angle.
   - `pricing.stop_atr_mult` (current 1.5), `pricing.target_atr_mult`
     (current 2.0) — the stop/take-profit distances in ATR multiples. The
     take-profit is `entry + target_atr_mult × ATR`, so `rr_ratio ≈
     target_atr_mult / stop_atr_mult`. Tune these for entry/exit rule
     mismatches (e.g. stops too tight → frequent stop-outs before the move).
     `pricing.min_rr_ratio` (current 0.8) is now only a sanity floor on rr,
     not the selection gate.
   - **Selection is exactly-N, not a gate.** The program returns the top
     `pricing.default_picks` (current 1) names by `pred_mean` — there is no
     "actionable" filter anymore. `--picks N` overrides it. A pick is flagged
     `below_breakeven` when `pred_mean < pricing.min_edge_over_cost × breakeven_pct`
     (current `min_edge_over_cost` 1.0): a weak-edge name surfaced only because
     the count had to be filled. If a report's picks are mostly
     `below_breakeven` and realized returns are poor, the lever is to advise the
     user to request fewer picks (lower `--picks` / `default_picks`) or raise
     `min_edge_over_cost` to make the flag stricter — NOT to fabricate an
     actionability gate. `pricing.max_abs_pred_mean` (current 0.05) drops
     split/corp-action glitch forecasts before ranking; widen only if a
     legitimate high-conviction name is being filtered.
   - **`pricing.entry_low_alpha` (currently 0.40) — the BASE / pivot
     quantile level of the per-ticker rolling empirical low head (since
     2026-06-05; see memory `project_low_head_negative_skill.md` — do
     NOT reintroduce an ML low head). The actual per-pick alpha is
     conviction-coupled: `pred_low_alpha = base × multiplier`, where a
     strong pick gets a shallow dip (high alpha) and a weak / below-
     breakeven pick gets a deep dip (low alpha). So the levers are now:
     * `pricing.entry_low_alpha` (the base) — shifts ALL picks' dip depth
       together. Raise it if fill_rate is consistently below
       `pred_low_alpha` WITHIN EACH conviction tier AND `fill_margin` on
       unfilled rows is small-negative (too bearish on dips across the
       board); lower it if `fill_margin` on filled rows is consistently
       large-positive (filling with slack — money on the table).
     * `pricing.entry_alpha_weak_mult` (0.6) / `entry_alpha_strong_mult`
       (1.25) / `entry_alpha_strong_edge` (3.0) — the CONVICTION (pick-
       strength) shape. Only touch these if the miscalibration is
       conviction-tier-specific: e.g. weak picks fill far MORE than their
       (deep) alpha would predict → the weak side isn't deep enough, lower
       `weak_mult`. Keep `strong_mult` modest (chasing fills on strong picks
       is a known failure mode).
     * `pricing.entry_alpha_overbought_start` (60) / `_full` (85) / `_mult`
       (0.5) — the OVERBOUGHT shape. The same alpha range, but deepened by
       RSI: an overbought pick's entry is multiplied down (only fills on a
       pullback). This is ALSO an entry-calibration lever — if the
       overbought RSI bucket fills far LESS than its `pred_low_alpha` would
       predict AND those would-be entries were actually good (the names rose),
       the penalty is too aggressive → raise `_mult` toward 1.0 (or `_start`).
       If overbought picks fill and then reverse, the penalty isn't deep
       enough → lower `_mult`. (`_mult: 1.0` disables the overbought axis.)
     * `pricing.entry_alpha_couple_conviction` — set false to revert to a
       single flat base alpha for every pick (disables BOTH the conviction
       and overbought axes of the range).
     All signals must anchor on `entry_limit_price` vs `t0_low`, never on
     the close, and must be judged WITHIN a conviction tier (a deep-dip
     weak pick failing to fill is by design). This is the primary knob
     family for stage-1 findings. First rule out a one-sided melt-up
     regime (low fills market-wide ≠ miscalibration). The trailing window
     auto-sizes for the deepest reachable alpha via
     `pricing.entry_low_target_tail_obs` (default 15); any alpha/mult
     change needs a `train` to take effect (rebuilds the low head's
     window + pooled grid). Before any of this, check the picks JSON: if
     `adj_entry_vnd != entry_vnd` for a pick, the news stage overrode the
     entry and the user likely placed the ADJUSTED order — the ledger's
     mechanical `entry_limit_price` / `fill_margin` for that row reflects
     a limit that was never placed, so exclude those rows before judging
     fill calibration or tuning these knobs.**
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
     and never feeds selection, so it isn't ledger-observable and only
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
.venv\Scripts\python.exe -m stockpredict.cli predict --mode claude --picks 3 --skip-train
```

Don't run it for them — they may want to inspect diffs first.

If the user changed `pricing.entry_low_alpha` (or any `entry_alpha_*` knob),
remind them they'll need to **retrain the low head** before it takes effect:

```
.venv\Scripts\python.exe -m stockpredict.cli train
```

If the Focus-1 finding was a model-skill gap (winners scored low, not
gate-excluded), the proposal is the missed-winners variant — train it and A/B
it, and promote `latest_missed.pkl` only if win rate holds:

```
.venv\Scripts\python.exe -m stockpredict.cli train-missed
.venv\Scripts\python.exe -m stockpredict.cli backtest-ab
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
