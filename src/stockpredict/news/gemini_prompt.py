"""Generate a self-contained Gemini prompt that asks Gemini to research news for
candidate Vietnamese tickers and return a re-ranked top-5 as JSON."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from ..config import load_config, reports_dir
from .sources import global_urls, vn_urls


def build_prompt(candidates: pd.DataFrame, on: dt.date | None = None,
                 exit_offset_days: int | None = None) -> str:
    on = on or dt.date.today()
    cfg = load_config().modes["gemini"]
    weight = cfg["news_weight"]
    horizon = int(exit_offset_days) if exit_offset_days is not None else int(
        load_config().target["exit_offset_days"]
    )
    # Compute the target sell day in Vietnamese trading-day space so the prompt
    # can quote a concrete date back at Gemini, who can then ask the user
    # about scheduling a reminder. Reminder fires on the sell day itself at
    # 11:30 ICT — late morning, just before the noon lunch break.
    from ..tracking import _next_trading_offset
    target_date = _next_trading_offset(pd.Timestamp(on), horizon).date()
    reminder_date = target_date
    if horizon == 2:
        sell_window = "13:00–14:30 ICT (afternoon session, after T+2 settlement)"
        reminder_note = "30 min before T+2 settlement at noon"
    else:
        sell_window = "09:00–14:30 ICT (any time during the trading day)"
        reminder_note = "late morning of sell day, before lunch break"
    suggested_time = "11:30 ICT"

    from .company_info import enrich
    candidates = enrich(candidates)

    from .research_dimensions import ETF_GUIDANCE_MD, REFERENCE_MD

    # Detect whether any ETF candidates are in the frame; the ETF research
    # rubric is only appended when at least one ETF row is present so the
    # stocks-only path keeps its existing prompt intact.
    has_etf_candidates = bool(
        "instrument_type" in candidates.columns
        and (candidates["instrument_type"].astype(str).str.upper() == "ETF").any()
    )

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
    if has_etf_candidates:
        parts.append(ETF_GUIDANCE_MD)
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
    parts.append("| symbol | type | company | pred_mean | entry_vnd | target_vnd | stop_vnd | fees_vnd | net_vnd | rr | actionable |")
    parts.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, r in candidates.iterrows():
        name = (r.get("organ_name") or "")[:50]
        itype = str(r.get("instrument_type", "STOCK") or "STOCK").upper()
        entry = int(r["entry_vnd"]) if "entry_vnd" in r and pd.notna(r.get("entry_vnd")) else 0
        target = int(r["target_vnd"]) if "target_vnd" in r and pd.notna(r.get("target_vnd")) else 0
        stop = int(r["stop_vnd"]) if "stop_vnd" in r and pd.notna(r.get("stop_vnd")) else 0
        fees = int(r["fees_round_trip_vnd"]) if "fees_round_trip_vnd" in r and pd.notna(r.get("fees_round_trip_vnd")) else 0
        net = int(r["net_reward_vnd"]) if "net_reward_vnd" in r and pd.notna(r.get("net_reward_vnd")) else 0
        rr = r.get("rr_ratio", float("nan"))
        rr_str = f"{rr:.2f}" if pd.notna(rr) else "-"
        act = "yes" if r.get("actionable", False) else "no"
        parts.append(
            f"| {r['symbol']} | {itype} | {name} | {r['pred_mean']:+.4f} | "
            f"{entry:,} | {target:,} | {stop:,} | {fees:,} | {net:+,} | {rr_str} | {act} |"
        )
    parts.append("")
    if has_etf_candidates:
        parts.append("Position sizing: stock rows use 100-unit lots (HOSE/HNX/UPCOM "
                     "minimum); ETF rows use 10-unit lots (HOSE ETF rule). All VND "
                     "values are absolute.")
    else:
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
    parts.append("")
    parts.append("## Final step — sell reminder (after the JSON, in chat)")
    parts.append("")
    parts.append(
        f"After you output the JSON, look at how many of your top-5 picks "
        f"are `actionable: yes` in the candidates table above (or, if you "
        f"have re-ranked them with news, would still pass the rr/net cost "
        f"gate)."
    )
    parts.append("")
    parts.append(
        f"**If at least one pick is actionable**, ask the user — in plain "
        f"text after the JSON, NOT inside the JSON — whether they would "
        f"like to schedule a reminder in **GMT+7 (Asia/Ho_Chi_Minh, "
        f"Vietnamese ICT)** to prepare the exit.\n"
        f"- Sell day: {target_date.isoformat()} ({sell_window}).\n"
        f"- Reminder fires: {reminder_date.isoformat()} {suggested_time} — "
        f"on the sell day itself ({reminder_note}). "
        f"This gives the user time to review and queue exit orders for the "
        f"afternoon session."
    )
    parts.append("")
    parts.append(
        "Suggest concrete options: a Google Calendar event, a phone alarm, "
        "or whatever the user prefers. If the user says yes, give them an "
        "ICS-style summary they can paste in: `BEGIN:VEVENT … DTSTART;TZID="
        "Asia/Ho_Chi_Minh:…`."
    )
    parts.append("")
    parts.append("If no pick is actionable, skip this question entirely.")
    return "\n".join(parts)


def write_prompt(candidates: pd.DataFrame, on: dt.date | None = None,
                 exit_offset_days: int | None = None) -> Path:
    on = on or dt.date.today()
    text = build_prompt(candidates, on=on, exit_offset_days=exit_offset_days)
    path = reports_dir() / f"gemini_prompt_{on.isoformat()}.txt"
    path.write_text(text, encoding="utf-8")
    return path
