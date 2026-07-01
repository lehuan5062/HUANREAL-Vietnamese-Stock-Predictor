"""Parse a JSON response from Gemini Chat (web) and merge it back into the
candidates parquet to produce the final explained picks.

The Gemini prompt (built by news/gemini_prompt.py) instructs Gemini to return:
    {
      "as_of": "...",
      "global_summary": "...",
      "picks": [
        {"symbol": "DXG", "business": "...", "drivers": ["..."],
         "score": 0.0167, "news_score": 1, "adjusted": 0.0184,
         "rationale": "...", "key_news": ["..."]}
      ]
    }

The user pastes that response into `reports/gemini_response_<date>.json`.
`gemini-finalize` reads it, merges with the saved candidates parquet
(the one emit_prompt also saves as a sidecar), and writes the explained
top-K picks JSON.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def parse_response(text: str) -> dict[str, Any]:
    """Tolerant JSON extraction — handles markdown code fences and stray text."""
    text = text.strip()
    # Strip ```json ... ``` fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text.strip())
    # Find the outermost JSON object
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in Gemini response")
    depth = 0
    end = -1
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise ValueError("unterminated JSON object in Gemini response")
    return json.loads(text[start:end])


def _opt_float(v: Any) -> float:
    """Coerce an optional Gemini-supplied price to float VND; NaN when absent
    or unparseable. Tolerates strings with commas / a VND suffix."""
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "").replace("đ", "").replace("VND", "").strip()
    if not s or s == "-":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def merge_response(candidates: pd.DataFrame, response: dict[str, Any]) -> pd.DataFrame:
    """Attach business / news_score / rationale / key_news / drivers to candidates.
    Tickers Gemini flagged with rationale starting `DROP:` are excluded entirely."""
    picks_in = response.get("picks") or response.get("scores") or []
    by_sym: dict[str, dict] = {}
    for p in picks_in:
        sym = str(p.get("symbol", "")).upper()
        if sym:
            by_sym[sym] = p

    out = candidates.copy()
    out["news_score"] = out["symbol"].map(
        lambda s: int(by_sym.get(s.upper(), {}).get("news_score", 0))
    )
    out["business"] = out["symbol"].map(
        lambda s: str(by_sym.get(s.upper(), {}).get("business", ""))
    )
    out["dimensions"] = out["symbol"].map(
        lambda s: by_sym.get(s.upper(), {}).get("dimensions", [])
    )
    out["drivers"] = out["symbol"].map(
        lambda s: by_sym.get(s.upper(), {}).get("drivers", [])
    )
    out["rationale"] = out["symbol"].map(
        lambda s: str(by_sym.get(s.upper(), {}).get("rationale", ""))
    )
    out["key_news"] = out["symbol"].map(
        lambda s: by_sym.get(s.upper(), {}).get("key_news", [])
    )
    # Optional news-adjusted entry / target (VND per share). NaN when Gemini
    # omitted them → downstream falls back to the mechanical entry/target.
    out["adj_entry_vnd"] = out["symbol"].map(
        lambda s: _opt_float(by_sym.get(s.upper(), {}).get("adj_entry_vnd"))
    )
    out["adj_target_vnd"] = out["symbol"].map(
        lambda s: _opt_float(by_sym.get(s.upper(), {}).get("adj_target_vnd"))
    )
    out["dropped"] = out["rationale"].str.upper().str.startswith("DROP")
    if out["dropped"].any():
        bad = out[out["dropped"]]["symbol"].tolist()
        print(f"[gemini] DROP override: excluding {', '.join(bad)}")
    out = out[~out["dropped"]].drop(columns=["dropped"])
    return out
