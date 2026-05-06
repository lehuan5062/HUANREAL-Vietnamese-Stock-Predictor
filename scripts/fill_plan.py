"""Fill today's claude_news_plan with the per-ticker research synthesised in
the chat session. Demo helper only — production flow is Claude in-session
editing the markdown directly via Edit tool."""
import re
from pathlib import Path

p = Path('reports/claude_news_plan_2026-05-05.md')
text = p.read_text(encoding='utf-8')

findings = {
    'ASP': {
        'business': 'An Pha Petroleum Group: top-3 Vietnamese LPG distributor (cooking + industrial), HOSE since 2008.',
        'dimensions': 'LPG wholesale prices; supply availability under Iran-war disruption; FX (USD-denominated import); seasonal demand; PVGas competitive dynamics; government fuel-subsidy policy.',
        'findings': [
            '[Global] Iran war disrupted Strait of Hormuz; Vietnam (80% Kuwait crude) facing severe fuel crunch; LPG premiums in Asia 10-15x pre-war.',
            '[Sector] Vietnam LPG supply emergency through end of May 2026; PVGas sourcing 66k tons from US in May (vs 44k from Middle East).',
            '[Policy] Government abolished some fuel levies until mid-April; mandatory work-from-home implemented to cut consumption.',
            '[Bull case] Distributors with stocked inventory benefit from price spike + scarcity premium driving margin.',
            '[Bear case] Demand destruction from very high prices; smaller distributors squeezed if PVGas captures import flow.',
        ],
    },
    'HII': {
        'business': 'An Tien Industries: plastics (150k t/yr additive plastic pellets) + minerals (222k t/yr CaCO3 stone powder). HOSE since 2017.',
        'dimensions': 'Polymer feedstock prices (oil-linked); construction-sector demand for stone powder; Iran-war oil impact; US/EU tariff exposure on plastic exports.',
        'findings': [
            '[Sector] Plastic pellet input costs follow crude — Iran war pushes feedstock higher.',
            '[Sector] Construction sector tailwind benefits CaCO3 demand (concrete additive); PM expressway pipeline.',
            '[Company] Trading 5,750 VND on May 3; small-cap. No material company-specific news in past 2 weeks.',
            '[Policy] No specific Vietnamese policy event on plastics this week.',
        ],
    },
    'ATG': {
        'business': 'An Truong An: small-cap materials/tech, pivoting to AI / blockchain (Oct-2025 capital-raise plan announced).',
        'dimensions': 'Capital-raise approval; speculative AI/blockchain narrative; retail-trader sentiment; market risk-on/off; idiosyncratic momentum.',
        'findings': [
            '[Company] No fresh material company news in past 2 weeks; pivot story unchanged since Oct 2025.',
            '[Sector] Vietnam AI/blockchain stocks broadly speculative; no specific catalyst this week.',
            '[VN macro] VN-Index 1,869 testing 1,900-1,920 resistance; foreign net selling -6.63T VND since Apr 8 — risk-off.',
            '[Global] Iran war pushing global risk-off; speculative names typically underperform in this regime.',
        ],
    },
    'C32': {
        'business': 'CIIIIIc Construction Investment Corp 3-2: industrial concrete + construction materials, HCMC.',
        'dimensions': 'Construction-materials demand; PM expressway pipeline (Mekong Delta projects); cement/energy input costs; Iran-war fuel price spillover.',
        'findings': [
            '[Sector] Vietnam construction market 7.51% CAGR to 2030; PM Chinh pushing Mekong Delta expressway projects (Cao Lanh — An Huu phase 1: 7.49T VND).',
            '[VN macro] Domestic infra spending tailwind; double-digit growth target for 2026.',
            '[Global] Iran war = higher energy / fuel costs squeeze cement and transport margins.',
            '[Stock] C32 down 11.62% over past week — short-term mean-reversion candidate but no catalyst.',
            '[Company] Q4 2025 sudden profit growth (Feb-1 article); no fresh news in past 2 weeks.',
        ],
    },
    'NRC': {
        'business': 'Danh Khoi Group (formerly Netland) — real-estate developer on HNX, residential/commercial/industrial. 4,800 VND.',
        'dimensions': 'Property-market sentiment; new Land Law 2024 + Housing Law 2023 implementation; National Real Estate Exchange launch Q1 2026; project legal-status; credit-growth caps.',
        'findings': [
            '[Sector] Vietnam property market 2026: "Easy gains fade as shakeout begins" — selective recovery favoring developers with clear legal status.',
            '[Policy] National Real Estate Transaction Center launched Q1 2026; new Land Price Framework effective Jan 1 2026.',
            '[Policy] Credit growth target reduced to ~15% from 19% in 2025.',
            '[Company] No specific company news in past 2 weeks.',
        ],
    },
    'DXS': {
        'business': 'Dat Xanh Services: real-estate brokerage subsidiary of Dat Xanh Group (DXG), HOSE.',
        'dimensions': 'Brokerage commission flow; sector recovery (north Vietnam); peer (DXG) earnings; AGM execution; deep-value-chain expansion strategy.',
        'findings': [
            '[Company] Q1 2026 profit 290.5B VND, ~7x same period 2025; revenue 1,415B VND, nearly 3x.',
            '[Company] FY2026 plan: revenue 5,300B VND, net profit 527B VND (+50% YoY); 27% / 36% completed after Q1.',
            '[Sector] Real-estate brokerage activity recovery, particularly in northern regions.',
            '[Stock] 7,360 VND on Apr 29 (3-month high), but down ~50% from Aug 2025 peak.',
            '[Strategy] AGM approved deep value-chain participation — develop to value-exploitation lifecycle.',
        ],
    },
    'ASM': {
        'business': 'Sao Mai Group: diversified — real estate + aquaculture (seafood + feed) + solar power + travel/resort. HOSE since 2010.',
        'dimensions': 'Aquaculture export prices; solar policy; real-estate cycle; travel demand recovery; conglomerate-discount sentiment.',
        'findings': [
            '[Company] Owns 600 t/day seafood plant + 378k t/yr feed plant + 200 t/day fish-oil refinery + ~100 RE projects.',
            '[Sector] Vietnam seafood industry: $11.5B export target 2026; navigating US tariff shifts.',
            '[Sector] Solar segment: power-demand growth tailwind under PDP VIII.',
            '[Company] No specific news in past 2 weeks.',
        ],
    },
    'HNG': {
        'business': 'HAGL Agrico (Hoang Anh Gia Lai Agricultural): banana (80% revenue), durian, rubber, cattle — large operations in Laos.',
        'dimensions': 'Banana export prices; Laos operational risk; rubber prices; FX (USD revenue); cattle business ramp.',
        'findings': [
            '[Company] Laos expansion: 8,000 ha banana, 2,000 ha pineapple, 14,000 ha cattle (210k breeding) via Southern Laos Agri (8.3T VND).',
            '[Stock] Up ~50% from Nov 2024 (3,300 to 5,000 VND mid-2025).',
            '[Sector] Vietnam agricultural exports navigating US/EU food-safety regimes.',
            '[Risk] Long-term turnaround story; mixed track record.',
        ],
    },
    'TLD': {
        'business': 'Thang Long Urban Development & Construction: glue-coated/film-faced plywood + civil/industrial/transport/irrigation construction. Hanoi.',
        'dimensions': 'Construction sector demand; plywood prices; PM infra pipeline; equipment-rental segment.',
        'findings': [
            '[Sector] Vietnam construction +7.5% CAGR; massive infra push (1,037T VND profit target across listed builders).',
            '[Stock] 8,310 VND, market cap 648B; down -0.24% in past 24h.',
            '[Company] No specific news in past 2 weeks.',
        ],
    },
    'PSI': {
        'business': 'Petrovietnam Securities: stockbroker subsidiary of Petrovietnam, HNX.',
        'dimensions': 'Brokerage market share; KRX system rollout (intra-day trading, T+1); FTSE upgrade flows; PVN parent dynamics.',
        'findings': [
            '[Company] 2025 revenue 293.5B VND (137% of plan); brokerage 87.8B (160%); profit after tax 44.6B (149%).',
            '[Company] 2026 = 20th year; "BUSINESS - BENEFIT" theme; comprehensive digitalization push.',
            '[Sector] FTSE upgrade Sept 21 2026 = expected foreign-flow tailwind for brokers.',
            '[Sector] KRX system enabling new products (intra-day, secured short selling).',
            '[Macro] But foreign investors net selling -6.63T VND since Apr 8 — near-term headwind.',
        ],
    },
    'NSH': {
        'business': 'Song Hong Aluminium (Shalumi Group): aluminum products — doors, cabinets, ladders, industrial. Exports to US/Japan/Canada/EU. Viet Tri, 1999.',
        'dimensions': 'Aluminum prices (LME); US/EU tariffs on Vietnamese aluminum; construction-sector demand (door/window); export FX exposure.',
        'findings': [
            '[Company] Multi-country exporter; no dividend.',
            '[Global] Trump 20% baseline tariff on Vietnam imports + 40% on transshipments could affect aluminum.',
            '[Sector] Vietnam construction tailwind benefits domestic aluminum-product demand.',
            '[Company] No specific news in past 2 weeks.',
        ],
    },
    'FIR': {
        'business': 'First Real: Da Nang-based real-estate developer.',
        'dimensions': 'Local Da Nang property cycle; new Land Law implementation; project pipeline; small-cap liquidity.',
        'findings': [
            '[Sector] Vietnam property market 2026: "Easy gains fade as shakeout begins"; selective recovery.',
            '[Policy] New land-price framework Jan 1 2026; National RE Exchange Q1 2026.',
            '[Company] No specific news in past 2 weeks.',
        ],
    },
    'CTP': {
        'business': 'Hoa Binh Takara: coffee producer (Arabica parchment + bean), Son La province. HNX.',
        'dimensions': 'Global coffee prices (especially Arabica); FX (USD revenue); Vietnamese coffee export volumes; weather impact on Son La harvest.',
        'findings': [
            '[Sector] Vietnam is world\'s 2nd-largest coffee exporter; Arabica share growing.',
            '[Global] Coffee futures volatile; supply-chain concerns due to drought in Brazil and Vietnam Mekong.',
            '[Company] No specific news in past 2 weeks.',
        ],
    },
    'MSB': {
        'business': 'Vietnam Maritime Commercial Joint Stock Bank: commercial bank, HOSE.',
        'dimensions': 'SBV policy rate; NPL ratios; net interest margin; corporate vs retail loan mix; capital actions.',
        'findings': [
            '[Company] 2025 revenue 12.13T VND (-0.06% YoY); earnings 5.63T (+1.98%); modest growth.',
            '[Company] Last quarter net income 1.62T VND.',
            '[Sector] Banks restructuring weak credit institutions get higher growth limits; MSB at 11-13%.',
            '[Macro] Foreign net selling -6.63T VND April hits banking shares disproportionately.',
        ],
    },
    'STH': {
        'business': 'STH Holdings: education-focused diversified group — Sigma International Bilingual School + organic tea (Quan Chu, Thai Nguyen).',
        'dimensions': 'Education sector demand; bilingual-school enrollment growth; organic-tea premium pricing; multi-industry capital allocation.',
        'findings': [
            '[Company] 2026 plan: revenue 400B VND (+41.5% YoY); pre-tax profit 50B; after-tax 40B.',
            '[Strategy] Sigma International Bilingual School partnership from Iris School platform.',
            '[Strategy] Organic tea complex in Quan Chu (Thai Nguyen) as strategic highlight.',
            '[Sector] Vietnamese education sector benefiting from rising middle-class demand for international schooling.',
        ],
    },
    'TSA': {
        'business': 'Truong Son Investment & Construction: power-infrastructure / construction. Listed on HOSE Jan 7, 2026 (transferred from UPCOM, 40.4M shares).',
        'dimensions': 'PDP VIII transmission pipeline; electricity demand growth; industrial-production recovery; new-listing premium.',
        'findings': [
            '[Listing] Debuted on HOSE Jan 7, 2026 — recent transfer from UPCOM.',
            '[Sector] Outlook backed by strong electricity-demand growth + industrial-production recovery + PDP VIII.',
            '[VN macro] Vietnam targeting double-digit growth 2026; massive infra push.',
            '[Company] APG Securities (listing adviser) bullish.',
        ],
    },
    'HTN': {
        'business': 'Hung Thinh Incons: civil/industrial/road/waste/urban infra construction + real estate (subsidiary of Hung Thinh Land). HOSE since 2018.',
        'dimensions': 'Hung Thinh group financial health; construction backlog; real-estate handover schedule; debt restructuring.',
        'findings': [
            '[Stock] Trading 7,400 VND, down -1.20% in 24h. Down ~88% from Mar 2022 ATH of 59,300 VND.',
            '[Group] Hung Thinh group has been navigating real-estate developer stress over 2023-2025.',
            '[Sector] Construction tailwind from infra push; mixed for property-related construction.',
            '[Company] No specific news in past 2 weeks.',
        ],
    },
    'DVM': {
        'business': 'Vietnam Medicinal Materials: pharmaceutical herbs + materials. HNX, 7,100 VND.',
        'dimensions': 'Drug import policy; herbal medicine demand; raw-material costs; healthcare regulation.',
        'findings': [
            '[Company] Stable mid-cap; no material news in past 2 weeks.',
            '[Sector] Vietnamese pharma navigating import-dependence challenges; herbal medicine growing.',
        ],
    },
    'NAB': {
        'business': 'Nam A Bank: commercial bank, HOSE.',
        'dimensions': 'Q1 earnings momentum; dividend / capital actions; SBV policy; NPL ratios; sector flows.',
        'findings': [
            '[Company] Q1 2026 pre-tax profit 1,620B VND, +32.5% YoY.',
            '[Company] ROE 21.5% (top-6 banks); total assets 410T VND (+56% YoY); deposits 217T VND.',
            '[Company] 20% stock dividend (100:20) approved; capital raise 5,431B VND Q2-Q3 2026.',
            '[Company] 1,000B VND public bond offering (second tranche).',
            '[Sector] Banks restructuring weak institutions get higher growth limits.',
        ],
    },
    'HSG': {
        'business': 'Hoa Sen Group: galvanized steel / coated steel / steel pipe + Hoa Sen Home retail subsidiary. HOSE.',
        'dimensions': 'US tariffs on coated steel; EU import quotas; domestic-market shift; Hoa Sen Home expansion; raw-material (HRC) prices.',
        'findings': [
            '[Global] US imposed final anti-dumping + countervailing duties; combined tariffs >110% on corrosion-resistant steel — eliminates US competitiveness.',
            '[Global] EU enforced strict quotas on coated steel from April 2025 — limits 35% of HSG exports.',
            '[Strategy] Chairman Le Phuoc Vu: shift focus to domestic market; expand Hoa Sen Home retail.',
            '[Calendar] 30% stock dividend ex-date May 4 2026 (already adjusted).',
            '[Company] FY2024-25 net profit +42% to 732B VND on margin gains despite revenue -7%.',
        ],
    },
}

scores = {
    'ASP': '+1',  # Iran-war LPG spike benefits distributors
    'DXS': '+1',  # Q1 +590% profit, brokerage recovery
    'STH': '+1',  # 2026 +41.5% revenue target, bilingual-school partnership
    'TSA': '+1',  # Just listed Jan 2026, PDP VIII tailwind
    'NAB': '+1',  # Strong Q1 +32.5%, dividend, capital raise
    'HSG': '-1',  # US >110% tariffs + EU quotas crushing exports
}


def replace_section(sym, info):
    global text
    pat = re.compile(
        rf"(### {sym}\s+—.*?)(?=^### |^## Scores)",
        re.DOTALL | re.MULTILINE,
    )

    def repl(m):
        body = m.group(0)
        header = body.split('\n', 1)[0]
        signal_match = re.search(r"^ML signal:.*$", body, re.MULTILINE)
        trade_match = re.search(r"^Trade.*$", body, re.MULTILINE)
        out = [header, '']
        if signal_match: out.append(signal_match.group(0))
        if trade_match: out.append(trade_match.group(0))
        out.append('')
        out.append('**Step 1 — Business**: in one line, write what this company does and the 1-2 main revenue lines.')
        out.append('')
        out.append(f'- {info["business"]}')
        out.append('')
        out.append('**Step 2 — Research dimensions**: derive 3-7 dimensions YOU think matter for THIS ticker on a T+N horizon. Your own list — not ours. Skip categories that don\'t apply, add ones that do (idiosyncratic drivers like a key customer, a peer\'s earnings, a peg, a contract often matter more than any standard category).')
        out.append('')
        out.append(f'- {info["dimensions"]}')
        out.append('')
        out.append('**Step 3 — Research findings per dimension**:')
        for url_line in re.findall(r"^- \[[^\]]+\]\([^)]+\)$", body, re.MULTILINE):
            out.append(url_line)
        out.append('')
        out.append('**Step 4 — Findings** (one bullet per dimension you investigated, tagged `[dimension-name]`, with dates and sources):')
        out.append('')
        for f in info['findings']:
            out.append(f'- {f}')
        out.append('')
        return '\n'.join(out)
    text = pat.sub(repl, text, count=1)


for sym, info in findings.items():
    replace_section(sym, info)

for sym, sc in scores.items():
    text = re.sub(rf"^\| {sym} \| (\+\d\.\d+) \| 0 \|$",
                  rf"| {sym} | \1 | {sc} |", text, flags=re.MULTILINE)

p.write_text(text, encoding='utf-8')
print(f'plan filled: {len(findings)} tickers, {len(scores)} scored non-zero')
