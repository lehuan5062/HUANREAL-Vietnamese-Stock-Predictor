"""Rebound mode finalize: ranks by score = P/N, prices via
add_recovery_price_suggestions (buy at close, target = close*(1+P), no
stop)."""
import json

import pandas as pd
import pytest

from stockpredict.modes import rebound as rebound_mode


def test_llm_only_section_stops_at_results_heading():
    """The per-ticker section splitter must stop at ANY `## ` heading, not just
    a scores table — the plan ends with `## Results`, and before this fix the
    last ticker's findings swallowed the Results table and any stray bullets,
    polluting key_news / dimensions_cited (VGI 2026-07-02)."""
    from stockpredict.news.llm_plan_runner import parse_llm_plan

    plan = (
        "# Rebound pick plan\n\n"
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
    p = pathlib.Path(tempfile.mkdtemp()) / "rebound_plan_test.md"
    p.write_text(plan, encoding="utf-8")

    out = parse_llm_plan(p)
    assert list(out["symbol"]) == ["VGI", "XXX", "YYY"]
    row = out.iloc[0]
    assert row["pred_days"] == 5
    assert abs(row["pred_profit"] - 0.05) < 1e-9
    assert not row["dropped"]
    assert bool(out.iloc[1]["dropped"])
    assert abs(out.iloc[2]["pred_profit"] - 0.04) < 1e-9
    assert row["key_news"] == ["[earnings-momentum] profit +57% YoY (AGM 2026)"]
    assert row["dimensions_cited"] == "earnings-momentum"


def test_finalize_ranks_by_p_over_n(monkeypatch, tmp_path):
    """finalize must compute score = P/N, rank by it, and price buy at close /
    target = close*(1+P), no stop."""
    plan = tmp_path / "rebound_plan_2026-07-02_rebound_d2.md"
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
        "run_signature": "rebound_d2",
    }), encoding="utf-8")
    monkeypatch.setattr(rebound_mode, "reports_dir", lambda: tmp_path, raising=False)
    import stockpredict.modes.common as common_mod
    monkeypatch.setattr(common_mod, "reports_dir", lambda: tmp_path)

    merged, out = rebound_mode.finalize(plan)

    assert list(merged["symbol"]) == ["BBB", "AAA"]  # ranked by P/N
    bbb = merged.iloc[0]
    assert abs(float(bbb["score"]) - 0.015) < 1e-6
    assert bbb["close_vnd"] == 30000                  # buy at close
    assert bbb["target_vnd"] == round(30000 * 1.03)   # close * (1+P)
    assert "stop_vnd" not in merged.columns           # no stop anywhere
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mode"] == "rebound"
    assert payload["picks"][0]["pred_days"] == 2.0
