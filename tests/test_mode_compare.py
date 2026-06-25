"""Cross-method comparison (base vs hybrid vs LLM-only) over the ledger.

All synthetic — no network, no real ledger. Patches ``mode_compare._read``
with an in-memory frame so the comparability / pooling logic is exercised
directly.
"""
import pandas as pd

from stockpredict.analyze import mode_compare as mc


def _ledger(rows):
    return pd.DataFrame([
        dict(run_id=a.replace("-", "") + "_" + s, signature=s, as_of=a,
             mode=m, symbol=sym, realized_return=r, evaluated=ev)
        for (a, m, s, sym, r, ev) in rows
    ])


def _patch(monkeypatch, rows):
    monkeypatch.setattr(mc, "_read", lambda: _ledger(rows))


def test_param_key_strips_mode_token():
    assert mc._param_key("base_d2", "base") == "d2"
    assert mc._param_key("claude_d2", "claude") == "d2"
    # The two-token mode is stripped whole, not just "claude".
    assert mc._param_key("claude_llm_d2", "claude_llm") == "d2"
    # A HOSE-scoped run is a DIFFERENT param key.
    assert mc._param_key("claude_d2_HOSE", "claude") == "d2_HOSE"


def test_comparable_cell_needs_two_modes_same_params(monkeypatch):
    rows = [
        # day 1: base + claude on d2 -> comparable
        ("2026-06-10", "base", "base_d2", "AAA", 0.01, True),
        ("2026-06-10", "claude", "claude_d2", "BBB", 0.03, True),
        # day 2: base only -> NOT comparable
        ("2026-06-11", "base", "base_d2", "CCC", 0.02, True),
        # day 3: base d2 vs claude d2_HOSE -> different params, NOT comparable
        ("2026-06-12", "base", "base_d2", "DDD", 0.05, True),
        ("2026-06-12", "claude", "claude_d2_HOSE", "EEE", 0.09, True),
    ]
    _patch(monkeypatch, rows)
    res = mc.compare_modes(window_days=3650)
    assert res["n_comparable_cells"] == 1
    modes = {r["mode"] for r in res["per_mode"]}
    assert modes == {"base", "claude"}
    # Only the one comparable day's picks are pooled (AAA, BBB) — CCC/DDD/EEE excluded.
    by_mode = {r["mode"]: r for r in res["per_mode"]}
    assert by_mode["base"]["n_picks"] == 1
    assert by_mode["claude"]["n_picks"] == 1


def test_llm_only_included_in_comparison(monkeypatch):
    rows = [
        ("2026-06-10", "base", "base_d2", "AAA", -0.01, True),
        ("2026-06-10", "claude", "claude_d2", "BBB", 0.01, True),
        ("2026-06-10", "claude_llm", "claude_llm_d2", "CCC", 0.05, True),
    ]
    _patch(monkeypatch, rows)
    res = mc.compare_modes(window_days=3650)
    assert res["n_comparable_cells"] == 1
    by_mode = {r["mode"]: r for r in res["per_mode"]}
    assert set(by_mode) == {"base", "claude", "claude_llm"}
    # LLM-only had the best return; per_mode is sorted by mean_per_day desc.
    assert res["per_mode"][0]["mode"] == "claude_llm"
    # Pairwise covers all three pairs.
    pairs = {(p["a"], p["b"]) for p in res["pairwise"]}
    assert pairs == {("base", "claude"), ("base", "claude_llm"), ("claude", "claude_llm")}


def test_unevaluated_rows_ignored(monkeypatch):
    rows = [
        ("2026-06-10", "base", "base_d2", "AAA", 0.01, True),
        ("2026-06-10", "claude", "claude_d2", "BBB", None, False),  # not evaluated
    ]
    _patch(monkeypatch, rows)
    res = mc.compare_modes(window_days=3650)
    # Only base survives the evaluated filter -> no 2-mode cell.
    assert res.get("n_comparable_cells", 0) == 0
    assert "note" in res


def test_unique_vs_shared_split(monkeypatch):
    rows = [
        # AAA shared by base+claude; BBB unique to claude; CCC unique to base
        ("2026-06-10", "base", "base_d2", "AAA", 0.02, True),
        ("2026-06-10", "base", "base_d2", "CCC", -0.04, True),
        ("2026-06-10", "claude", "claude_d2", "AAA", 0.02, True),
        ("2026-06-10", "claude", "claude_d2", "BBB", 0.06, True),
    ]
    _patch(monkeypatch, rows)
    res = mc.compare_modes(window_days=3650)
    uvs = {u["mode"]: u for u in res["unique_vs_shared"]}
    assert uvs["claude"]["n_unique"] == 1          # BBB
    assert uvs["claude"]["n_shared"] == 1          # AAA
    assert uvs["base"]["n_unique"] == 1            # CCC
    assert abs(uvs["claude"]["unique_mean_return"] - 0.06) < 1e-9


def test_report_renders_for_empty(monkeypatch):
    _patch(monkeypatch, [])
    res = mc.compare_modes(window_days=90)
    md = mc.format_report(res)
    assert "Mode comparison" in md
    assert res.get("note")
