"""Autonomous Claude mode: call the Anthropic API with the web_search server tool
so Claude reads news + scores tickers without a human in the loop.

Requires:
  - `anthropic` SDK (pip install anthropic)
  - ANTHROPIC_API_KEY in env (or in `.env` at project root)
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import pandas as pd

from ..config import load_config


from .research_dimensions import REFERENCE_PLAIN

SYSTEM = (
    "You are a Vietnamese equities analyst doing thorough RESEARCH on each "
    "candidate ticker — not a sentiment scan of company news.\n\n"
    "For each ticker, YOU decide what to research. We do not give you a "
    "fixed checklist — different companies have different drivers, and "
    "you know better than us what matters for THIS ticker today.\n\n"
    + REFERENCE_PLAIN + "\n\n"
    "Always include in your output, per ticker:\n"
    "  - `dimensions`: the 3-7 research dimensions you derived for this "
    "ticker (your own list, may differ from the reference categories above)\n"
    "  - `key_news`: actual findings from your searches with date + source\n"
    "  - `news_score`: -1 / 0 / +1\n"
    "  - `rationale`: 1-2 sentences explaining the score\n\n"
    "Search in BOTH English AND Vietnamese. Vietnamese press carries more "
    "company-level coverage. Useful Vietnamese keywords: `<TICKER> cổ phiếu` "
    "(stock), `<company-name> lợi nhuận quý 1 2026` (Q1 profit), `cổ tức` "
    "(dividend), `phát hành cổ phiếu` (share issuance), `nghị định / nghị "
    "quyết / thông tư / quyết định` (decree / resolution / circular / "
    "decision), `dự thảo luật` (draft law), `huỷ niêm yết` (delisting), "
    "`xuất khẩu` (exports). Mix both languages in your queries.\n\n"
    "Search broadly with web_search. The ticker pages on baomoi.com, "
    "cafef.vn, vietstock.vn, vneconomy.vn, ndh.vn, theinvestor.vn are "
    "convenient starting points but NOT a closed list. Also search Google "
    "News, Reuters / Bloomberg / FT for macro, chinhphu.vn / sbv.gov.vn for "
    "policy / decrees / circulars, and any other source you think is "
    "relevant. Cross-check claims across at least 2 sources before "
    "scoring.\n\n"
    "Score on actual business + market findings, NOT on price / technicals "
    "(those are already in the ML input).\n\n"
    "Return ONLY valid JSON.\n\n"
    "HARD OVERRIDE: if a ticker is delisted, suspended, or in bankruptcy, "
    "score -1 with rationale starting `DROP:` — never recommend it."
)

USER_TEMPLATE = """Today is {date}. Score these {n} candidates for a Vietnamese T+2 swing trade
(buy at today's close, sell on T+2 afternoon after settlement).

Candidates (with company names):
{table}

For each ticker, do thorough research:
1. From the company name + your own knowledge, identify the business.
2. **Derive 3-7 research dimensions yourself** for THIS ticker. They may
   match common categories (sector, macro, policy, geopolitics, calendar)
   or be ticker-specific (a single major customer, a peer's earnings, a
   peg, a contract). Skip categories that don't apply, add ones that do.
3. web_search broadly across the dimensions you derived. Cross-check
   across at least 2 sources before scoring.
4. Score -1 / 0 / +1 (or DROP for delisted / suspended) on actual
   research findings, not technicals.

Return JSON exactly:
{{
  "as_of": "{date}",
  "global_summary": "1-2 sentences on macro drivers relevant to VN-Index today",
  "scores": [
    {{"symbol": "XYZ",
      "business": "1-line description of what the company does",
      "dimensions": ["the 3-7 dimensions YOU decided to research for this ticker"],
      "drivers": ["the 2-3 most material drivers among the dimensions"],
      "news_score": 1,
      "rationale": "1-2 sentences citing specific findings with dates",
      "key_news": ["finding 1 (date, source)", "finding 2"]}}
  ]
}}

Include every input ticker."""


def _candidates_table(candidates: pd.DataFrame) -> str:
    from .company_info import enrich
    candidates = enrich(candidates)
    rows = ["| symbol | company | pred_mean | close | rsi_14 | mom_20 |",
            "| --- | --- | --- | --- | --- | --- |"]
    for _, r in candidates.iterrows():
        name = (r.get("organ_name") or "")[:60]
        rows.append(
            f"| {r['symbol']} | {name} | {r['pred_mean']:+.4f} | "
            f"{r.get('close', 0):.0f} | {r.get('rsi_14', 0):.1f} | "
            f"{r.get('mom_20', 0):+.4f} |"
        )
    return "\n".join(rows)


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of the model's response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # find the outermost { ... }
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in Claude response")
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
        raise ValueError("unterminated JSON object in Claude response")
    return json.loads(text[start:end])


def score(candidates: pd.DataFrame, date: str,
          current_horizon: int | None = None,
          current_signature: str | None = None) -> dict[str, Any]:
    """Call Claude to score the candidates. Raises if API not available."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to .env or the environment, "
            "or use 'claude' mode interactively inside Claude Code instead."
        )
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "`anthropic` SDK not installed. Run: pip install anthropic"
        ) from e

    client = anthropic.Anthropic(api_key=api_key)
    user_msg = USER_TEMPLATE.format(
        date=date,
        n=len(candidates),
        table=_candidates_table(candidates),
    )
    # Use the latest Claude with web_search server tool. The tool is invoked
    # automatically by Claude as it researches each ticker.
    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=8000,
        system=SYSTEM,
        tools=[{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 30,
        }],
        messages=[{"role": "user", "content": user_msg}],
    )
    # Pull text content out of the response (skip tool_use / tool_result blocks)
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    return _extract_json(text)


def merge(candidates: pd.DataFrame, scored: dict[str, Any]) -> pd.DataFrame:
    """Apply news_score from `scored` to candidates and compute adjusted score.
    Tickers whose rationale starts with `DROP:` are excluded entirely."""
    cfg = load_config().modes["claude"]
    weight = float(cfg["news_weight"])
    score_by_sym = {row["symbol"].upper(): row for row in scored.get("scores", [])}
    out = candidates.copy()
    out["news_score"] = out["symbol"].map(
        lambda s: int(score_by_sym.get(s, {}).get("news_score", 0))
    )
    out["rationale"] = out["symbol"].map(
        lambda s: str(score_by_sym.get(s, {}).get("rationale", ""))
    )
    out["business"] = out["symbol"].map(
        lambda s: str(score_by_sym.get(s, {}).get("business", ""))
    )
    out["dimensions"] = out["symbol"].map(
        lambda s: score_by_sym.get(s, {}).get("dimensions", [])
    )
    # Pull `[tag]` markers out of the API's key_news bullets so the
    # autonomous path feeds the by-dimension hit-rate ledger the same way
    # the interactive path does. If the API returns no key_news for a
    # symbol, dimensions_cited stays empty.
    from .claude_runner import _extract_dimension_tags
    def _tags_for(s: str) -> str:
        kn = score_by_sym.get(s, {}).get("key_news", [])
        if isinstance(kn, str):
            kn = [kn]
        elif not isinstance(kn, list):
            kn = []
        return ",".join(_extract_dimension_tags([str(b) for b in kn]))
    out["dimensions_cited"] = out["symbol"].map(_tags_for)
    out["dropped"] = out["rationale"].str.upper().str.startswith("DROP")
    if out["dropped"].any():
        bad = out[out["dropped"]]["symbol"].tolist()
        print(f"[claude] DROP override: excluding {', '.join(bad)}")
    out = out[~out["dropped"]].drop(columns=["dropped"])
    out["adjusted"] = out["pred_mean"] * (1.0 + weight * out["news_score"])
    return out.sort_values("adjusted", ascending=False).reset_index(drop=True)
