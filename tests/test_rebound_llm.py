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
        "exit_offset_days": 2, "n_picks": 2, "hose_only": False,
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
    assert payload["n_below_breakeven"] == 0
