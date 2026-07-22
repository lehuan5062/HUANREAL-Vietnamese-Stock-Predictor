"""Reference list of categories the LLM MAY consider when deriving its own
research plan per ticker.

Important: this is intentionally a NON-EXHAUSTIVE inspiration list, not a
mandatory checklist. The prompts ask Claude to derive the
research dimensions for each ticker themselves, based on what the company
actually does and what news categories actually move it on a 2-day
horizon. They are free to skip categories below that don't apply, and to
add ticker-specific categories we haven't listed.
"""

REFERENCE_MD = """\
**Search-language guidance** — Vietnamese press has materially more company-
level coverage than English wire services. Mix BOTH languages in your search
queries. Useful Vietnamese keywords:
  - `<TICKER> cổ phiếu` (stock)
  - `<company-name> lợi nhuận quý 1 2026` (Q1 2026 profit)
  - `kết quả kinh doanh quý` (quarterly business results)
  - `cổ tức bằng tiền mặt` / `cổ tức bằng cổ phiếu` (cash / stock dividend)
  - `phát hành cổ phiếu` / `tăng vốn điều lệ` (share issuance / capital raise)
  - `nghị định <topic>` / `nghị quyết` / `thông tư` / `quyết định` (decrees /
    resolutions / circulars / decisions)
  - `dự thảo luật <topic>` (draft law)
  - `xuất khẩu` / `nhập khẩu` (exports / imports)
  - `huỷ niêm yết` / `tạm ngừng giao dịch` (delisting / trading halt)

**Categories you MAY find useful when deciding what to research.** This is a
non-exhaustive reference, NOT a mandatory checklist. For each ticker, derive
your own research plan: skip categories that don't apply, add ones we haven't
listed. The goal is to cover whatever could actually move *this* stock on a
2-day horizon — and to be honest about what's relevant.

Common categories analysts often check:

- Company-specific (earnings, guidance, capital actions, M&A, lawsuits,
  executive changes, contracts, plant capacity, accounting, insider trading)
- Sector / industry (peer earnings, sector reports, supply & demand,
  industry-association notices, sector-wide regulation)
- Vietnam domestic macro (VN-Index, foreign flows, VND/USD, SBV rates,
  CPI, GDP, retail / consumer, FTSE / MSCI weight)
- Vietnamese policy / law / regulation, **including upcoming and draft**
  (Nghị quyết / resolutions, Nghị định / decrees, Thông tư / circulars,
  Quyết định / PM decisions, Luật / draft laws, FOL / listing-rule changes)
- Global macro & events (US Fed, oil, China demand, US-China / US-Vietnam
  trade, sanctions, tariffs, EU directives, sovereign rating actions)
- Geopolitical conflicts & disruptions (wars affecting trade routes,
  regional tensions, supply-chain disruptions, weather / disasters)
- Legal & calendar events (court rulings, anti-dumping, FDA / USDA / EU
  actions, ex-div, AGMs, listing transfers, seasonality)

But these are just prompts to think with. Some tickers have idiosyncratic
drivers (a key customer's earnings, a single contract, a peg / index,
peer M&A) that don't fit any of the above — surface those too. If a
category above is irrelevant for this ticker, skip it and say so.

Method per ticker:
  1. Identify the business from the company name.
  2. **Derive 3-7 research dimensions specific to THIS ticker** — your own
     list, not ours. Note what you considered and skipped.
  3. Search broadly across the dimensions you derived. Cross-check across
     at least 2 sources before scoring.
  4. Score -1 / 0 / +1 (or DROP for delisted/halted) based on what you
     actually found."""


REFERENCE_PLAIN = """Method per ticker:
1. Identify the business from the company name.
2. DERIVE 3-7 research dimensions YOURSELF, specific to this ticker.
   You decide what's relevant; the categories below are just inspiration.
3. Search broadly across the dimensions you derived. Cross-check across
   at least 2 sources before scoring. Mix English AND Vietnamese queries
   (e.g. `<TICKER> co phieu loi nhuan`, `nghi dinh ...`, `thong tu ...`)
   — Vietnamese press carries more company-level coverage.
4. Score -1 / 0 / +1 (or DROP for delisted/halted).

Common reference categories (NOT a mandatory checklist; pick what fits,
add what's missing):
- Company-specific (earnings, M&A, contracts, lawsuits, executive moves)
- Sector / industry (peer results, supply / demand, sector regulation)
- Vietnam macro (VN-Index, FX, SBV rates, foreign flows, FTSE/MSCI)
- Vietnam policy / law (Nghi quyet, Nghi dinh, Thong tu, Quyet dinh,
  draft laws — INCLUDING upcoming, not just enacted)
- Global macro (US Fed, oil, China, trade tensions, tariffs, sanctions,
  sovereign rating actions)
- Geopolitical / disruptions (conflicts, supply-chain, weather)
- Legal / calendar (court rulings, anti-dumping, FDA/USDA, ex-div, AGMs)

Idiosyncratic drivers (key customers, single contracts, pegs, peer M&A)
often matter more than any category above — surface those when they
exist. If a category above is irrelevant for this ticker, skip it."""


ETF_GUIDANCE_MD = """\
**ETF candidates** (rows where `instrument_type == "ETF"`, e.g. FUEVFVND,
E1VFVN30, FUESSV30, FUEMAV30) are passive baskets, NOT companies. The
"identify the business" step does not apply — skip it. Switch the research
rubric for ETF rows to:

1. **Identify the underlying index** from the ticker and `organ_name`
   (DCVFM/SSIAM/Mirae Asset/KIM/MAFM/VinaCapital are fund managers; the
   index name is the actual driver):
   - FUEVFVND → VN Diamond
   - E1VFVN30, FUESSV30, FUEMAV30, FUEKIV30 → VN30
   - FUEVN100, FUEIP100, FUEFCV50 → VN100
   - FUEDCMID → VN Midcap
   - FUESSVFL → VNFIN Lead
2. **Research these dimensions** instead of company-specific ones:
   - Underlying index performance over the past 5 sessions (price + breadth)
   - Foreign-investor net flows into Vietnamese equities (the dominant
     ETF demand driver — large foreign net-buy days lift FUE* prices)
   - ETF creation / redemption activity if reported by VSDC
   - NAV vs market-price premium / discount, if quoted on the fund manager
     page (DCVFM, SSIAM, etc.)
   - Upcoming index rebalancing dates (VN30 quarterly review, VN Diamond
     basket reconstitution) within the T+N exit horizon
   - Big constituent moves: if a top-3-weight constituent has a binary
     event (earnings, FOL change, sanction) on or before the exit day, it
     dominates the ETF's return
3. Use `business` to surface the **underlying index name** (e.g.
   "VN Diamond ETF managed by DCVFM"), not the fund manager's corporate
   identity. The LLM should make the index explicit so the user knows
   what they're really tracking.
4. ETF return distributions are tighter than stocks; small `pred_mean`
   magnitudes are normal. Don't penalize an ETF for a sub-stock-sized
   ML signal — judge it on its own dispersion."""


ETF_GUIDANCE_PLAIN = """ETF candidates (instrument_type=='ETF', e.g. FUEVFVND,
E1VFVN30, FUESSV30) are passive baskets — NOT companies. Skip the "identify
the business" step for these rows. Use this rubric instead:

1. Identify the underlying index from the ticker (FUEVFVND -> VN Diamond;
   E1VFVN30 / FUESSV30 / FUEMAV30 / FUEKIV30 -> VN30; FUEVN100 / FUEIP100 /
   FUEFCV50 -> VN100; FUEDCMID -> VN Midcap; FUESSVFL -> VNFIN Lead).
2. Research dimensions for ETFs:
   - Underlying index performance + breadth over past 5 sessions
   - Foreign-investor net flows into VN equities (top driver of FUE* prices)
   - VSDC creation / redemption activity if reported
   - NAV vs market-price premium / discount (fund manager pages)
   - Upcoming index rebalancing dates (VN30 quarterly, VN Diamond) inside
     the T+N exit horizon
   - Top-3-weight constituents with binary events on/before exit day
3. Set `business` to the underlying INDEX (not the fund manager).
4. ETF return distributions are tighter than stocks — small pred_mean
   magnitudes are normal; judge each ETF on its own dispersion."""


# Backwards-compat exports — older imports use these names.
CHECKLIST_MD = REFERENCE_MD
CHECKLIST_PLAIN = REFERENCE_PLAIN
