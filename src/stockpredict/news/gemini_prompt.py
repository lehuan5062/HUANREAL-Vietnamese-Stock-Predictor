"""Generate a self-contained Gemini prompt that asks Gemini to research news for
candidate Vietnamese tickers and return a re-ranked top-5 as JSON."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from ..config import load_config, reports_dir
from .sources import global_urls, vn_urls


def build_prompt(candidates: pd.DataFrame, on: dt.date | None = None) -> str:
    on = on or dt.date.today()
    cfg = load_config().modes["gemini"]
    weight = cfg["news_weight"]

    from .company_info import enrich
    candidates = enrich(candidates)

    from .research_dimensions import REFERENCE_MD

    parts: list[str] = []
    parts.append(
        f"You are a Vietnamese equities analyst doing thorough RESEARCH on "
        f"each candidate ticker. Today is {on.isoformat()}. An ML model has "
        f"narrowed the universe to the candidates below for a T+2 swing "
        f"trade (buy at today's close, sell at the close two trading days "
        f"later in the afternoon session, after settlement)."
    )
    parts.append("")
    parts.append(
        "**You decide what to research per ticker.** We do not give you a "
        "fixed checklist — different companies have different drivers, and "
        "you know better than us what matters for each one."
    )
    parts.append("")
    parts.append("**Method — per ticker:**")
    parts.append("")
    parts.append(
        "1. Identify the business from the company name + your own knowledge."
    )
    parts.append(
        "2. **Derive 3-7 research dimensions yourself** for THIS specific "
        "ticker. They may match common categories (sector, macro, policy, "
        "geopolitics, calendar) or be ticker-specific (a single major "
        "customer, a peer's earnings, a peg, a one-off contract). Skip "
        "categories that don't apply, add ones that do."
    )
    parts.append(
        "3. **Research broadly with Google search** across the dimensions "
        "you derived. Seed sources: baomoi.com/tim-kiem/<TICKER>.epi, "
        "cafef.vn, vietstock.vn, vneconomy.vn, ndh.vn, theinvestor.vn, "
        "fireant.vn, plus Reuters Asia, Bloomberg, FT, Yahoo Finance for "
        "macro, plus chinhphu.vn and sbv.gov.vn for policy / decrees / "
        "circulars. These are STARTING POINTS — search beyond them. Cross-"
        "check claims across at least 2 sources before scoring."
    )
    parts.append("")
    parts.append(REFERENCE_MD)
    parts.append("")
    parts.append(
        "4. **Score** -1 / 0 / +1 based on what you actually found. Price/"
        "technical noise alone = 0."
    )
    parts.append("")
    parts.append(
        "**Hard override**: if a ticker is delisted / suspended / in bankruptcy, "
        "score it -1 and write `DROP:` at the start of the rationale."
    )
    parts.append("")
    parts.append(f"Adjusted score formula: adjusted = pred_mean * (1 + {weight} * news_score)")
    parts.append("")

    parts.append("## Vietnamese news sources to consult")
    for sym in candidates["symbol"].tolist():
        urls = vn_urls(sym)
        url_list = "; ".join(f"{n}: {u}" for n, u in urls.items())
        parts.append(f"- {sym}: {url_list}")
    parts.append("")

    parts.append("## Global / macro sources")
    for name, url in global_urls().items():
        parts.append(f"- {name}: {url}")
    parts.append("")
    parts.append("Also consult vnexpress.net/kinh-doanh, cafef.vn for general VN market news.")
    parts.append("")

    parts.append("## Candidates")
    parts.append("")
    parts.append("| symbol | company | pred_mean | entry_vnd | target_vnd | stop_vnd | fees_vnd | net_vnd | rr | actionable |")
    parts.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, r in candidates.iterrows():
        name = (r.get("organ_name") or "")[:50]
        entry = int(r["entry_vnd"]) if "entry_vnd" in r and pd.notna(r.get("entry_vnd")) else 0
        target = int(r["target_vnd"]) if "target_vnd" in r and pd.notna(r.get("target_vnd")) else 0
        stop = int(r["stop_vnd"]) if "stop_vnd" in r and pd.notna(r.get("stop_vnd")) else 0
        fees = int(r["fees_round_trip_vnd"]) if "fees_round_trip_vnd" in r and pd.notna(r.get("fees_round_trip_vnd")) else 0
        net = int(r["net_reward_vnd"]) if "net_reward_vnd" in r and pd.notna(r.get("net_reward_vnd")) else 0
        rr = r.get("rr_ratio", float("nan"))
        rr_str = f"{rr:.2f}" if pd.notna(rr) else "-"
        act = "yes" if r.get("actionable", False) else "no"
        parts.append(
            f"| {r['symbol']} | {name} | {r['pred_mean']:+.4f} | "
            f"{entry:,} | {target:,} | {stop:,} | {fees:,} | {net:+,} | {rr_str} | {act} |"
        )
    parts.append("")
    parts.append("Position is 100 units (Vietnamese minimum lot). All VND values are absolute.")
    parts.append("`net_vnd` already accounts for ACBS round-trip fees (commission + VAT + PIT).")
    parts.append("")

    parts.append("## Output format")
    parts.append("")
    parts.append(
        "Return ONLY valid JSON, no prose. Save your response as "
        f"`reports/gemini_response_{on.isoformat()}.json` so the program can "
        f"merge it into the final picks via `gemini-finalize`."
    )
    parts.append("")
    parts.append("Schema:")
    parts.append("```json")
    parts.append("{")
    parts.append('  "as_of": "YYYY-MM-DD",')
    parts.append('  "global_summary": "1-2 sentences on macro drivers relevant to VN-Index today",')
    parts.append('  "picks": [')
    parts.append('    {"symbol": "XYZ",')
    parts.append('     "business": "1-line description of what the company does",')
    parts.append('     "dimensions": ["the 3-7 dimensions YOU derived for this ticker"],')
    parts.append('     "drivers": ["the 2-3 most material drivers among the dimensions"],')
    parts.append('     "ml_score": 0.0017, "news_score": 1, "adjusted": 0.001785,')
    parts.append('     "rationale": "1-2 sentences citing specific findings with dates",')
    parts.append('     "key_news": ["finding 1 (date, source)", "finding 2"]}')
    parts.append("  ]")
    parts.append("}")
    parts.append("```")
    parts.append("Top 5 picks, sorted by `adjusted` descending.")
    return "\n".join(parts)


def write_prompt(candidates: pd.DataFrame, on: dt.date | None = None) -> Path:
    on = on or dt.date.today()
    text = build_prompt(candidates, on=on)
    path = reports_dir() / f"gemini_prompt_{on.isoformat()}.txt"
    path.write_text(text, encoding="utf-8")
    return path
