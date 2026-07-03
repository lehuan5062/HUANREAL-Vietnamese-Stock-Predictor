"""Claude/Gemini finalize must re-rank rebound picks by `score` (not pred_mean),
which rebound frames don't have. Regression test for the pred_mean KeyError."""
import json

import pandas as pd
import pytest

from stockpredict.modes import claude as claude_mode


def _write_rebound_plan(tmp_path):
    # Sidecar candidates: a rebound frame (score, no pred_mean).
    cand = pd.DataFrame([
        {"symbol": "AAA", "score": 0.0100, "close": 20.0, "close_vnd": 20000,
         "target_vnd": 20600, "pred_days": 3.0, "pred_profit": 0.03,
         "pred_recovery_prob": 0.95, "below_recovery_bar": False, "atr_14": 0.5},
        {"symbol": "BBB", "score": 0.0200, "close": 30.0, "close_vnd": 30000,
         "target_vnd": 30900, "pred_days": 2.0, "pred_profit": 0.03,
         "pred_recovery_prob": 0.98, "below_recovery_bar": False, "atr_14": 0.6},
    ])
    plan = tmp_path / "claude_news_plan_2026-07-01_claude_d2_AAA-BBB.md"
    cand.to_parquet(plan.with_suffix(".candidates.parquet"), index=False)
    plan.with_suffix(".meta.json").write_text(json.dumps({
        "n_picks": 2, "hose_only": False,
        "include_etfs": True, "exclude": [], "run_signature": "claude_d2",
    }), encoding="utf-8")
    # Filled plan: AAA gets +1 news, BBB gets 0. Per-ticker sections + score table.
    plan.write_text(
        "# plan\n\n"
        "### AAA  —  Alpha Corp\n**Step 4 — Findings**\n- [earnings] beat\n\n"
        "### BBB  —  Beta Corp\n**Step 4 — Findings**\n- \n\n"
        "## Scores\n\n"
        "| symbol | score | news_score | adj_entry_vnd | adj_target_vnd |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| AAA | 0.0100 | +1 | | |\n"
        "| BBB | 0.0200 | 0 | | |\n",
        encoding="utf-8",
    )
    return plan


def test_claude_finalize_reranks_by_score(monkeypatch, tmp_path):
    plan = _write_rebound_plan(tmp_path)
    monkeypatch.setattr(claude_mode, "reports_dir", lambda: tmp_path)

    merged, out = claude_mode.finalize(plan)

    # adjusted = score * (1 + 0.10 * news_score): AAA 0.011, BBB 0.020 -> BBB first.
    assert "pred_mean" not in merged.columns
    assert list(merged["symbol"]) == ["BBB", "AAA"]
    aaa = merged[merged["symbol"] == "AAA"].iloc[0]
    assert abs(float(aaa["adjusted"]) - 0.0100 * 1.10) < 1e-6
    # Payload written, below_recovery_bar counted (not below_breakeven).
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mode"] == "claude"
    assert payload["n_below_recovery_bar"] == 0


def test_llm_only_section_stops_at_results_heading():
    """The per-ticker section splitter must stop at ANY `## ` heading, not just
    `## Scores` — the LLM-only plan ends with `## Results`, and before this fix
    the last ticker's findings swallowed the Results table and any stray
    bullets, polluting key_news / dimensions_cited (VGI 2026-07-02)."""
    from stockpredict.news.claude_llm_runner import parse_llm_plan

    plan = (
        "# Claude LLM-only pick plan\n\n"
        "### VGI  —  Viettel Global\n\n"
        "**Step 1 — Business**: telecom operator.\n- Telecom.\n\n"
        "**Step 2 — Research dimensions**: drivers.\n- earnings\n\n"
        "**Step 4 — Findings**:\n"
        "- [earnings-momentum] profit +57% YoY (AGM 2026)\n\n"
        "## Results — fill this with your chosen picks\n\n"
        "| rank | symbol | N_days | P |\n"
        "| --- | --- | --- | --- |\n"
        "| 1 | VGI | 5 | 0.05 |\n"
        "| 2 | XXX | DROP | |\n"
        "| 3 | YYY | 3 | 4% |\n"
    )
    import pathlib, tempfile
    p = pathlib.Path(tempfile.mkdtemp()) / "claude_llm_plan_test.md"
    p.write_text(plan, encoding="utf-8")

    out = parse_llm_plan(p)
    assert list(out["symbol"]) == ["VGI", "XXX", "YYY"]
    row = out.iloc[0]
    assert row["pred_days"] == 5
    assert abs(row["pred_profit"] - 0.05) < 1e-9
    assert not row["dropped"]
    # DROP in the N_days cell marks exclusion; percent form parses to a fraction.
    assert bool(out.iloc[1]["dropped"])
    assert abs(out.iloc[2]["pred_profit"] - 0.04) < 1e-9
    # Findings hold ONLY the Step 4 bullet — no Results-table or seed-URL junk.
    assert row["key_news"] == ["[earnings-momentum] profit +57% YoY (AGM 2026)"]
    assert row["dimensions_cited"] == "earnings-momentum"


def test_llm_only_finalize_ranks_by_p_over_n(monkeypatch, tmp_path):
    """LLM-only finalize must compute score = P/N, rank by it, and price like
    the base/hybrid modes (buy at close, target = close*(1+P), no stop)."""
    plan = tmp_path / "claude_llm_plan_2026-07-02_claude_llm_d2.md"
    plan.write_text(
        "# plan\n\n"
        "### AAA  —  Alpha\n**Step 4 — Findings**\n- [earnings] beat\n\n"
        "### BBB  —  Beta\n**Step 4 — Findings**\n- [sector] tailwind\n\n"
        "## Results\n\n"
        "| rank | symbol | N_days | P |\n"
        "| --- | --- | --- | --- |\n"
        "| 1 | AAA | 10 | 0.05 |\n"   # score 0.005
        "| 2 | BBB | 2 | 0.03 |\n",   # score 0.015 -> ranks first
        encoding="utf-8",
    )
    pd.DataFrame([
        {"symbol": "AAA", "close": 20.0, "rsi_14": 40.0, "mom_20": -0.08},
        {"symbol": "BBB", "close": 30.0, "rsi_14": 38.0, "mom_20": -0.06},
    ]).to_parquet(plan.with_suffix(".candidates.parquet"), index=False)
    plan.with_suffix(".meta.json").write_text(json.dumps({
        "method": "llm_only", "n_picks": 2,
        "hose_only": False, "include_etfs": True, "exclude": [],
        "run_signature": "claude_llm_d2",
    }), encoding="utf-8")
    monkeypatch.setattr(claude_mode, "reports_dir", lambda: tmp_path)

    merged, out = claude_mode.finalize_llm(plan)

    assert list(merged["symbol"]) == ["BBB", "AAA"]  # ranked by P/N
    bbb = merged.iloc[0]
    assert abs(float(bbb["score"]) - 0.015) < 1e-6
    assert bbb["close_vnd"] == 30000                  # buy at close
    assert bbb["target_vnd"] == round(30000 * 1.03)   # close * (1+P)
    assert "stop_vnd" not in merged.columns           # no stop anywhere
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mode"] == "claude_llm"
    assert payload["picks"][0]["pred_days"] == 2.0
