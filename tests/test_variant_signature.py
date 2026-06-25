"""The missed-winners variant stays distinguishable: signature tag + payload."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stockpredict.tracking import run_signature


def test_signature_tags_missed_variant():
    assert run_signature("base", 2) == "base_d2"
    assert run_signature("base", 2, variant="standard") == "base_d2"
    assert run_signature("base", 2, variant="missed") == "base_d2_missed"
    # Tag comes after the exclude block.
    assert run_signature("base", 2, exclude=["ACB"], variant="missed") \
        == "base_d2_xACB_missed"


def test_base_payload_carries_model_variant(monkeypatch, tmp_path):
    from stockpredict.modes import base

    stub = pd.DataFrame({"symbol": ["AAA"], "below_breakeven": [False]})
    monkeypatch.setattr(base, "rank_today", lambda **kw: stub)
    monkeypatch.setattr(base, "reports_dir", lambda: tmp_path)
    monkeypatch.setattr(base, "record", lambda *a, **k: 0)   # don't touch the ledger

    picks, out = base.run(on="2026-06-02", n_picks=1, variant="missed")
    payload = json.loads(Path(out).read_text(encoding="utf-8"))
    assert payload["model_variant"] == "missed"
    assert "_missed" in payload["run_signature"]
    assert "_missed" in out.name
