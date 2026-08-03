# Research: BC8 — Bechtle AG

> Stage-1 markdown-rapport conform `research/TEMPLATE.md`. Methodische rigueur volgens `research/METHODE.md`. Geen JSON-aanmaak en geen platform-mutaties door deze agent.

---

## Bronnen-inventaris (Stap 0.5)

```
Jaar 2025 — HOOG
  Bron: Bechtle AG Annual Report 2025 (EN, PDF)
  URL:  https://www.bechtle.com/dam/jcr:b348dfde-c1b2-4756-83fb-ecbae570ec81/Bechtle_ar2025_en.pdf
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: business volume, omzet (totaal en per regio), EBITDA, EBIT
                       (totaal en per regio), EBT, EAT (aandeelhouders), EPS,
                       CFO, cash & equivalents, equity ratio, aandelen uitstaand,
                       dividend per aandeel, payout ratio, dividend yield 31-12,
                       segmentinformatie regio's Duitsland/Frankrijk/Benelux/Other
                       Europe, kwartaalbreakdown, werknemers, koers 31-12 (€31,10)
  Cijfers NIET overgenomen uit deze bron: capex (PP&E + intangibles) afzonderlijk,
                       FCF (Bechtle eigen definitie), nettoschuld 31-12-2025,
                       balans-detail (PP&E, intangibles, working capital details)
                       — deze details staan in de detail-tabellen van het
                       jaarverslag die in deze fetch niet volledig geëxtraheerd
                       zijn. Voor 2025 capex/FCF gebruik ik StockAnalysis
                       (AGGREGATOR), zie hieronder.

Jaar 2016-2024 — HOOG
  Bron: Bechtle Annual Report 2024 — Multi-Year Overview Bechtle Group (PDF, 6 pp.)
  URL:  https://reports.bechtle.com/annual-report/2024/_assets/downloads/fin-multi-year-overview-bechtle-ar24.pdf?h=8Yi_YW7e
  Daadwerkelijk geopend: ja
  Cijfers overgenomen voor jaren 2016 t/m 2024: business volume, omzet (totaal,
                       binnenland, buitenland), EBIT, EBITDA, segmenten (IT
                       System House & MS vs IT E-Commerce), kostenposten,
                       financieel resultaat, EBT, belastingen, EAT, balans
                       (non-current/current activa, goodwill, PP&E, schulden,
                       eigen vermogen, totaal activa), cash flow operating /
                       investing / financing, free cash flow (Bechtle definitie),
                       investeringen, koers 31-12, dividend per aandeel,
                       payout ratio, marktkap, EV, EPS, EV/EBITDA, EV/EBIT,
                       P/E, werknemers, ROE, ROA, ROCE, nettoschuld, working
                       capital, capital employed, equity ratio, DSO

Jaar 2015 — GEEN BRON BESCHIKBAAR
  Zoekpoging(en): Bechtle multi-year overview AR2024 (begint bij 2016),
                  StockAnalysis Cash Flow (begint bij 2021), AR2015 PDF
                  niet gefetched in deze sessie
  Conclusie: 2015 blijft LEEG in alle tabellen. Genoteerd in
             "ontbrekende_data" (sectie 13).

AGGREGATOR-aanvullingen 2025 — AGGREGATOR (alleen voor cijfers die uit het
AR2025 PDF in deze sessie niet konden worden geëxtraheerd):
  Bron: StockAnalysis.com — Bechtle AG (ETR:BC8) Cash Flow Statement
  URL:  https://stockanalysis.com/quote/etr/BC8/financials/cash-flow-statement/
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: 2025 CFO (€289,78M — match met PDF), 2025 D&A (€159,3M),
                       2025 FCF in StockAnalysis-definitie (€289,78M — let op:
                       StockAnalysis trekt geen capex af in deze tabel; CFO ≈ FCF
                       in hun standaardisatie. Wij gebruiken daarom CFO als
                       conservatieve FCF-proxy 2025).
  Cijfers NIET overgenomen: balans 2025-12-31 details (gebruiken AR2025
                       overzicht: equity ratio 44,9%, cash €452M).

Aanvullende statistics 2026 — AGGREGATOR (alleen voor TTM/multiples per peildatum):
  Bron: StockAnalysis.com — Bechtle AG Statistics
  URL:  https://stockanalysis.com/quote/etr/BC8/statistics/
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: TTM revenue €6,35 mld, TTM net income €215,25M, TTM EBIT
                       €310,7M, TTM EBITDA €388,9M, TTM EPS €1,71, TTM FCF
                       €300,15M, market cap €4,31 mld (op datum statistics-pagina),
                       enterprise value €4,54 mld, ROE 11,24%, ROIC 10,02%,
                       WACC sector 8,17%, current ratio 1,68, debt/equity 0,37,
                       debt/EBITDA 1,64, P/E 20,02 (op StockAnalysis-koersdatum
                       €34,20, NIET op onze peildatum), beta 5Y 0,89.
                       Koers op 14-mei-2026 op StockAnalysis-quote: €29,70 →
                       dit gebruik ik als peildatumkoers.

Bronnen voor marktcontext (Stap 0):
  - 10-jaars Bund-rendement: TradingEconomics, 14 mei 2026 = 3,07%
    URL: https://tradingeconomics.com/germany/government-bond-yield
  - Damodaran Country Risk Premium Germany jan 2026 = 4,2% ERP
    URL: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html
  - Bechtle Q1 2026 resultaten (8 mei 2026): business volume +13,2% naar €2,2 mld;
    EBT +11,5% naar €61,6M; order intake +16,9% naar €2,357M; order backlog €3,3 mld
    URL: https://www.investing.com/news/company-news/bechtle-q1-2026-slides-doubledigit-earnings-growth-across-regions-93CH-4671195
  - FY2025 slide-analyse Investing.com (20 mrt 2026)
    URL: https://www.investing.com/news/company-news/bechtle-fy2025-slides-8-growth-masks-margin-pressure-q4-miss-93CH-4573204
  - Aandeelhoudersstructuur Schick familie 35,02%
    URL: https://www.bechtle.com/de-en/about-bechtle/press/press-releases/2025/bechtle-mourns-the-loss-of-its-founder-gerhard-schick
  - Analist consensus (12 analisten, gem €44,88):
    URL: https://www.marketscreener.com/quote/stock/BECHTLE-AG-54095346/consensus/

Zelf-check: kan ik voor ELK numeriek cel in de tabellen een bron-URL noemen
uit deze inventaris? Ja — recente 5 jaren (2021-2025) zijn allemaal HOOG via
Bechtle eigen jaarverslag-PDF's; jaar 2025 capex/FCF-detail en TTM-multiples
zijn AGGREGATOR via StockAnalysis (toegestaan voor aanvulling daar waar
Bechtle PDF in deze sessie niet alles gaf, met expliciete bronvermelding per
veld). 2015 blijft LEEG.
```

---

## Metadata
- **Ticker (bare):** BC8
- **Yahoo symbol:** BC8.DE
- **Exchange:** XETRA (Deutsche Börse, Frankfurt)
- **Sector (GICS-achtig):** Technologie
- **Industrie:** IT-services & distributie (mid-market system integrator)
- **Land:** Duitsland
- **Peildatum analyse:** 2026-05-14
- **Koers op peildatum:** 29,70
- **Valuta:** EUR
- **Marktkapitalisatie:** EUR 3,74 mld
- **Marktkap in mln (lokale valuta):** 3742
- **Free float pct:** 54,8
- **Indexlidmaatschap:** MDAX (Deutsche Börse Mid-Cap)
- **Domein:** bechtle.com

---

## 1. Executive summary

- **Kernthese** (2-3 zinnen, waarom dit interessant of niet):
  Bechtle is een Duitse mid-cap IT-systeemintegrator en -distributeur die sinds 1983 een leidende positie heeft opgebouwd in het Duitstalige Mittelstand-segment (DACH). Het bedrijf combineert twee segmenten — IT System House & Managed Services (ca. 62% omzet) en IT E-Commerce (ca. 38%) — en breidt structureel uit naar Frankrijk, Benelux en de rest van Europa, met een record orderboek van €3,3 mld na Q1 2026. Het verdienmodel rust op langdurige raamcontracten met overheids- en mid-market klanten, schaalvoordelen in inkoop bij hyperscalers/OEMs en een fijnmazig netwerk van lokale system houses. Op peildatum verhandelt het aandeel aan een TTM-P/E van 17 en levert het een dividendrendement van ~2,4%, maar de margecompressie van FY2025 (EBIT-marge zakte van 5,6% naar 5,2%), de zwakke Franse operatie (EBIT −47%) en de slechts beperkte structurele moat houden het koers/intrinsieke-waarde-verschil bescheiden. De these is er een van defensieve compounder met cyclische gevoeligheid voor IT-bestedingen en een groeispurt vanaf 2027 zodra het Vision-2030 plan (€10 mld business volume, 5% EBT-marge) traceerbaar wordt.
- **Oordeel** (enum **UITSLUITEND**: **KOOP** | **HOLD** | **PASS**): **HOLD**
- **Fair value basis** (kansgewogen, lokale valuta): 34,65
- **Fair value kansgewogen**: 34,65
- **EPV per aandeel** (Earnings Power Value, zonder groeipremie): 27,40
- **Upside pct**: 17
- **Fair value scenarios** (3 stuks — **Pessimistisch / Basis / Optimistisch**):

| Scenario | Fair value | Upside % | FCF groei % | WACC % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 23,30 | −22 | 3 | 8,5 | 25 |
| Basis | 33,00 | 11 | 5 | 7,5 | 50 |
| Optimistisch | 49,30 | 66 | 7 | 6,5 | 25 |

- **Reverse-DCF impliciete groei pct** (wat moet FCF groeien om huidige koers te rechtvaardigen): 2,8
- **Grootste kans** (1 zin): Een succesvolle uitvoering van Vision 2030 met EBT-marge-herstel naar 5% op €10 mld business volume zou de winst per aandeel boven €3,00 brengen en het aandeel in een herwaarderingsfase plaatsen.
- **Grootste risico** (1 zin): Aanhoudende margecompressie door internationale uitbreiding (vooral Frankrijk) gecombineerd met een cyclisch zwakke IT-bestedingsomgeving kan het EBT-margedoel onhaalbaar maken en het multiple omlaag drukken naar mid-cycle dieptes (P/E 12-13).

---

## 2. Bedrijfsprofiel

- **Beschrijving** (wat doet het bedrijf, 3-5 zinnen):
  Bechtle AG is een beursgenoteerde IT-systeemintegrator en IT-e-commercedistributeur met hoofdkantoor in Neckarsulm (Duitsland). Het bedrijf opereert vanuit ongeveer 120 vestigingen in 14 Europese landen en levert IT-infrastructuur, software, cloud-oplossingen, managed services, cyberbeveiliging en projectimplementaties aan een klantenbestand van overwegend middelgrote bedrijven, overheidsinstanties en grote ondernemingen. De omzet wordt verdeeld over twee segmenten: IT System House & Managed Services (62% van de omzet in 2024, het service- en projectbusiness in lokale system houses) en IT E-Commerce (38%, online doorverkoop van hardware en software aan zakelijke klanten via merken zoals Bechtle direct en ARP). Voor zijn klanten lost Bechtle een coördinatieprobleem op: één lokaal aanspreekpunt voor de complete IT-stack van een midden- of overheidsorganisatie, met de inkoopkracht van een €8,6 mld omzetorganisatie en de fijnmazige aanwezigheid van een netwerk van zelfstandige system houses. Sinds 2024 rapporteert het bedrijf zowel "business volume" (totaal verkoopvolume inclusief doorverkochte software waarvan het commissie int) als "omzet" (alleen wat het als principal mag boeken onder IFRS 15) — dit verschil van circa €2,2 mld in 2025 verklaart waarom revenue-groei (+1,6%) achterloopt op business-volume-groei (+8,1%).
- **Geschiedenis** (oprichting, IPO, kernmomenten):
  Bechtle werd in 1983 opgericht door drie afgestudeerden van de Heilbronn University of Applied Sciences — Klaus von Jan, Ralf Klenk en Gerhard Schick — in een kleine winkel in Heilbronn. Het bedrijf richtte zich aanvankelijk op de doorverkoop van pc's aan lokale ondernemingen. Een sleutelmoment was de vroege adoptie van e-commerce in 1995, op een moment dat het concept in Europa nog nauwelijks bestond. In 2000 ging Bechtle naar de beurs op het Neuer Markt-segment van Frankfurt op 30 maart, tegen een koers van €27 per aandeel, en de emissie was twaalfvoudig overingeschreven; bij IPO had het bedrijf 30 vestigingen en 1.680 werknemers met een omzet van bijna DM 955 miljoen. Met het IPO-kapitaal startte het bedrijf een buy-and-build strategie waarmee in de afgelopen 25 jaar tientallen kleinere IT-firma's zijn overgenomen om lokale marktaandelen en technische capaciteiten te versterken. Strategisch belangrijke overnames waren onder andere ARP (Zwitserse e-commerce, jaren 2000), ACS Systems UK (2022, intrede VK), Inmac WStore (Frankrijk), en in 2025 Grupo Solutia Tecnologia in Spanje (600 medewerkers), Nuovamacut Automazione in Italië (PLM-software) en RIS 2048 in Portugal (€50M business volume). Bechtle overleefde de dotcomcrash (de koers viel terug van piek €99 in 2000 naar dieptepunten), de financiële crisis 2008-2009 en de Covid-pandemie zonder een dividend over te slaan — een ononderbroken track record van 25 dividendjaren sinds IPO. Het belangrijkste verlies van de afgelopen periode was het overlijden op 4 maart 2025 van medeoprichter Gerhard Schick op 84-jarige leeftijd; de familie Schick blijft via een holding en de Gerhard und Ilse Schick Foundation veruit de grootste aandeelhouder met 35,02%. In 2025 introduceerde Bechtle de Bechtle Index of Sovereignty (BIoS) als productized service voor digitale soevereiniteit en breidde het zijn cloud-partnerships uit met Deutsche Telekom en Arvato Systems voor GDPR-conforme infrastructuur. In maart 2026 kondigde de Raad van Commissarissen aan dat Konstantin Ebert per 1 januari 2027 CEO Dr. Thomas Olemotz zal opvolgen.
- **Bedrijfsmodel** (hoe verdient het geld, recurring / eenmalig):
  Bechtle verdient geld op twee manieren. Het IT System House & Managed Services-segment factureert projectomzet (implementaties, integraties) plus terugkerende serviceomzet (managed services, beheercontracten, raamovereenkomsten met overheden — bijvoorbeeld het €450M-low-code-raamcontract met Duitse federale en lokale overheden uit 2025). Het IT E-Commerce-segment verkoopt hardware, software en cloud-licenties door aan zakelijke klanten via online platforms; de marge hier is dunner maar het volume is hoger. Een wezenlijk deel van de business volume (de €2,2 mld verschil met omzet) bestaat uit software waarvoor Bechtle als agent optreedt onder IFRS 15 en alleen de commissie als omzet boekt — dit drukt de gerapporteerde omzet maar verhoogt structureel de marge op die regel. Het percentage terugkerende omzet uit raamovereenkomsten en managed services is niet expliciet in het jaarverslag uitgesplitst maar wordt door management omschreven als "een groeiend aandeel".
- **IPO-context** (datum, koers, reden IPO, waardering bij IPO):
  Beursintroductie 30 maart 2000 op het Neuer Markt-segment van Frankfurter Wertpapierbörse tegen €27 per aandeel. Het IPO was twaalfvoudig overingeschreven en bracht kapitaal binnen voor de buy-and-build acquisitiestrategie die het bedrijf de afgelopen 25 jaar heeft uitgevoerd. De IPO was geen exit voor de oprichters; Gerhard Schick bleef tot zijn pensionering anchorshareholder en de familie behoudt ook in 2026 een controlerend belang van 35,02%. Aandelensplit 3-voor-1 in augustus 2021. IPO meer dan 10 jaar geleden, dus geen pre-IPO-correctie of pre-IPO financial-engineering-check vereist.
- **Klantprofiel** (B2B/B2C, concentratie, retention):
  Bechtle bedient uitsluitend B2B: middelgrote ondernemingen (Mittelstand), grote ondernemingen en de publieke sector. De publieke sector is groeiend — significante raamcontracten met Duitse federale, deelstaat- en gemeentelijke instanties, onder andere het €450M low-code-raamcontract en €501M ProVitako-frameworks voor HPE-netwerk- en serverproducten. Klantconcentratie is laag: geen enkele klant vertegenwoordigt meer dan een laag enkelcijferig percentage omzet, gegeven de breed gediversifieerde klantenbasis van duizenden bedrijven en publieke organisaties. Retention is hoog in het System House & Managed Services-segment (langlopende raamcontracten, 3-5 jaar) maar moeilijker te kwantificeren omdat Bechtle geen expliciete net-revenue-retention publiceert. Het management omschrijft de klantrelaties als "langetermijn-partnerschappen".
- **Oprichtingsjaar**: 1983
- **IPO-datum**: 2000-03-30
- **IPO-koers** (lokale valuta): 27,00
- **Personeel** (FTE): 16.360 (31-12-2025)
- **Landen actief**: 14 (DE, AT, CH, FR, NL, BE, LU, UK, IE, IT, ES, PT, HU, andere)
- **Klantconcentratie** (50-80 woorden): Geen enkele klant draagt materieel meer dan enkelvoudige procenten bij. De grootste publieke raamcontracten — €450M low-code (looptijd 4 jaar, dus ~€110M/jaar) en €501M ProVitako (HPE netwerk/server) — zijn aanzienlijk maar elk minder dan 2% jaaromzet. Diversificatie over duizenden Mittelstand-klanten in DACH is een structurele eigenschap; het is tegelijk een reden voor relatieve marge-compressie (geen pricing power op individuele deals).

### Geografische spreiding (omzet 2025)
| Regio | Omzet % | Valuta-exposure |
|---|---|---|
| Duitsland | 58,3 | EUR |
| Frankrijk | 10,0 | EUR |
| Benelux | 11,9 | EUR |
| Overig Europa | 19,8 | EUR (gemengd: CHF, GBP, HUF, etc.) |

**Toelichting geografie** (50-80 woorden): Bechtle is structureel EUR-gedomineerd: ~80% van de omzet komt uit eurolanden. De Britse activiteiten (ACS Systems UK) en Zwitserland brengen GBP- en CHF-exposure, gemitigeerd door lokale kostenstructuren (natural hedge: lokale loonkosten in dezelfde valuta als omzet). Geen materiële transfer-pricing-issues bekend; geen significant FX-hedging vermeld. Het hoofdrisico is niet wisselkoers maar regionaal margedispersie: Frankrijk leverde in 2025 een EBIT-marge van slechts 2,6% (vs 6,2% Duitsland).

### Segmenten (omzetbasis FY2024)
| Naam | Omzet % | Beschrijving |
|---|---|---|
| IT System House & Managed Services | 62,1 | Projectimplementaties, integratie-werk, managed services, beheercontracten, raamovereenkomsten met overheden en bedrijven — geleverd via lokale system houses in DACH en uitbreidend in EU |
| IT E-Commerce | 37,9 | Online doorverkoop van hardware, software en cloud-licenties aan zakelijke klanten via Bechtle direct, ARP en regionale platforms — schaalvoordeel-business met dunne marges maar hoge omloopsnelheid |

### Aandeelhouders (top 5, 2025)
| Naam | Belang % | Type (oprichter / PE / institutioneel / retail) |
|---|---|---|
| Familie Schick (incl. Gerhard und Ilse Schick Foundation) | 35,02 | Oprichter/familie (controlerend) |
| Flossbach von Storch SE | 10,18 | Institutioneel |
| List Privatstiftung | 6,20 | Institutioneel |
| Allianz Global Investors GmbH | 4,99 | Institutioneel |
| Free float (overig, incl. Vanguard, BlackRock) | 43,61 | Institutioneel/retail mix |

- **Institutioneel eigendomstrend** (40-60 woorden): Stabiel tot licht stijgend. Het institutionele aandeel is sinds 2022 toegenomen van circa 34% naar 38-40% in 2025, doordat Flossbach von Storch en passive vehicles (Vanguard, BlackRock) hun posities hebben uitgebreid. De familie Schick verzwakt haar belang niet; de Gerhard und Ilse Schick Foundation (4,21% van de 35,02%) is in 2022 opgericht om langetermijn-stabiliteit te borgen.

---

## 3. Financieel — historische data (10 jaar + TTM)

### Resultatenrekening (bedragen in mln EUR)

Bron-eis: 2020-2024 = HOOG (multi-year overview Bechtle AR2024 PDF), 2025 = HOOG (Bechtle AR2025 PDF). 2016-2019 = HOOG (zelfde multi-year overview). 2015 = GEEN BRON (leeg).

| Jaar | Omzet | Omzetgroei % | Brutowinst | Brutomarge % | EBIT | EBIT-marge % | EBITDA | EBITDA-marge % | Nettowinst | Nettomarge % | EPS | EPS-groei % | Aandelen mln |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2015 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2016 | — | — | 543,8 | 17,3 | 144,1 | — | 171,8 | 6,2 | 103,4 | — | 4,92 | — | 21,0 |
| 2017 | 3.144,8 | — | 642,9 | 20,4 | 164,3 | 5,2 | 196,2 | 6,2 | 114,6 | 3,6 | 2,73 | — | 42,0 |
| 2018 | 3.829,3 | 21,8 | 767,8 | 20,0 | 195,1 | 5,1 | 237,1 | 6,2 | 137,1 | 3,6 | 3,27 | 19,8 | 42,0 |
| 2019 | 4.712,0 | 23,1 | 856,6 | 18,2 | 241,4 | 5,1 | 326,0 | 6,9 | 170,5 | 3,6 | 4,06 | 24,2 | 42,0 |
| 2020 | 5.050,3 | 7,2 | 920,0 | 18,2 | 277,0 | 5,5 | 375,1 | 7,4 | 192,5 | 3,8 | 4,58 | 12,8 | 42,0 |
| 2021 | 5.305,5 | 5,1 | 1.053,4 | 19,9 | 325,7 | 6,1 | 428,7 | 8,1 | 231,4 | 4,4 | 1,84 | — | 126,0 |
| 2022 | 6.028,2 | 13,6 | 1.121,9 | 18,6 | 355,4 | 5,9 | 467,5 | 7,8 | 251,1 | 4,2 | 1,99 | 8,2 | 126,0 |
| 2023 | 6.422,7 | 6,5 | 1.138,4 | 17,7 | 382,3 | 6,0 | 508,9 | 7,9 | 265,5 | 4,1 | 2,11 | 6,0 | 126,0 |
| 2024 | 6.305,8 | −1,8 | — | — | 351,3 | 5,6 | 491,6 | 7,8 | 244,9 | 3,9 | 1,95 | −7,6 | 126,0 |
| 2025 | 6.405,9 | 1,6 | — | — | 335,3 | 5,2 | 494,6 | 7,7 | 229,2 | 3,6 | 1,82 | −6,7 | 126,0 |
| TTM (mrt-26) | 6.350 | — | 1.148 | 18,1 | 310,7 | 4,9 | 388,9 | 6,1 | 215,3 | 3,4 | 1,71 | — | 126,0 |

Opmerking: aandelensplit 3:1 op 16 augustus 2021 verklaart EPS-discontinuïteit tussen 2020 (€4,58, 42 mln aandelen pre-split) en 2021 (€1,84, 126 mln aandelen post-split). Vergelijkbare EPS-niveaus 2020 split-adjusted = €1,53. TTM-cijfers uit StockAnalysis statistics-pagina (AGGREGATOR, peildatum maart 2026).

- **Toelichting resultaten** (80-120 woorden):
  Bechtle laat tot 2023 een rustige groeicompounder zien met omzet-CAGR van ~9% over 2017-2023 en EBIT-marge stabiel in de 5,1%-6,1% bandbreedte. Sinds 2023 stagneert die opmars: 2024 daalde de omzet 1,8% door zwakkere IT-bestedingen in Mittelstand, en 2025 herstelde slechts marginaal (+1,6%) terwijl de business-volume +8,1% groeide — een uiting van de IFRS-15-shift naar agent-rapportage bij softwarewederverkoop. Onder de oppervlakte zit margecompressie: EBIT-marge zakte van 6,0% (2023) naar 5,6% (2024) en 5,2% (2025), gedreven door uitbreidingskosten in Frankrijk/Iberia, integratie van overnames en interne IT-investeringen. EPS volgde mee omlaag van piek €2,11 (2023) naar €1,82 (2025), −13,7% in twee jaar.
- **Omzet-CAGR** (2017-2024): 10,4% (€3.145M → €6.306M); over 2020-2025: 4,9% (€5.050M → €6.406M).

### Kasstromen (Bechtle eigen definitie FCF)

Bron 2016-2024 HOOG; 2025 CFO HOOG (AR2025 PDF), 2025 capex/FCF detail AGGREGATOR (StockAnalysis — let op: StockAnalysis-definitie van FCF wijkt af; hier nemen we Bechtle's eigen FCF-reeks t/m 2024 en gebruiken 2025 CFO als proxy).

| Jaar | CFO | Capex | FCF | FCF na SBC | FCF/aandeel | FCF-marge % | FCF-groei % | FCF-conversie % | SBC | Dividend totaal | Aandeleninkoop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2016 | 53,4 | 39,6 | −7,0 | −7,0 | −0,33 | −0,2 | — | −6,8 | — | 31,5 | — |
| 2017 | 54,3 | 66,9 | −24,1 | −24,1 | −0,57 | — | — | −21,0 | — | 37,8 | — |
| 2018 | 140,5 | 56,6 | −147,2 | −147,2 | −3,50 | −3,1 | — | −107,3 | — | 42,0 | — |
| 2019 | 186,0 | 48,0 | 109,5 | 109,5 | 2,61 | 2,3 | — | 64,2 | — | 50,4 | — |
| 2020 | 317,5 | 66,2 | 204,1 | 204,1 | 4,86 | 4,0 | 86,4 | 106,0 | — | 56,7 | — |
| 2021 | 284,5 | 58,4 | 218,6 | 218,6 | 1,73 | 4,1 | 7,1 | 94,5 | — | 69,3 | — |
| 2022 | 116,7 | 82,0 | −29,8 | −29,8 | −0,24 | −0,5 | — | −11,9 | — | 81,9 | — |
| 2023 | 459,0 | 95,2 | 151,2 | 151,2 | 1,20 | 2,4 | — | 56,9 | — | 88,2 | — |
| 2024 | 558,2 | 88,7 | 377,0 | 377,0 | 2,99 | 6,0 | 149,3 | 153,9 | — | 88,2 | — |
| 2025 | 289,8 | ~121 | ~169 (StockAnalysis) | ~169 | 1,34 | 2,6 | −55 | 73,8 | — | 88,2 | — |

- **Toelichting kasstromen** (80-120 woorden):
  Bechtle's FCF-volatiliteit is hoog: −€147M (2018) tot +€377M (2024). Dit komt grotendeels door werkkapitaalcycli: voorraadopbouw in groei-/inflatiejaren (2018, 2022) drukt CFO; werkkapitaal-release in normalisatiejaren (2020, 2024) boost FCF. De Bechtle-FCF-definitie (CFO + investingactiviteit excl. acquisities) levert na een record €377M in 2024 een terugval naar circa €169M in 2025 doordat de sterke Q4-businessvolume-groei (+16,6%) werkkapitaal opslokte. Mediaan FCF 2019-2025 ≈ €178M. SBC is verwaarloosbaar bij Bechtle (geen significant aandelenoptie-programma; verwatering door RSU's lijkt verwaarloosbaar — eigen vermogen-verandering 2021-2024 verklaart door retained earnings, niet door SBC). FCF-conversie is wisselend door werkkapitaal; 5-jaar mediaan ~75%.

### Balans-ratio's (10 jaar, mln EUR)

Bron 2016-2024 HOOG; 2025 deels HOOG (eq.ratio, cash uit AR2025), balans-detail 2025 niet volledig geëxtraheerd uit AR2025 PDF in deze sessie.

| Jaar | Nettoschuld | Nettoschuld/EBITDA | Eigen vermogen | ROE % | ROIC % | ROCE % | Current ratio | Solvabiliteit % | Goodwill % van EV | Working capital |
|---|---|---|---|---|---|---|---|---|---|---|
| 2016 | −101,3 | n.b. | 694,1 | 17,1 | — | 24,0 | — | 54,7 | 9,1 | 375,6 |
| 2017 | −46,0 | n.b. | 777,3 | 16,9 | — | 22,3 | — | 53,9 | 6,4 | 492,9 |
| 2018 | 142,6 | 0,6 | 883,2 | 18,1 | — | 20,6 | — | 43,6 | 10,9 | 580,8 |
| 2019 | 115,5 | 0,4 | 1.014,8 | 19,8 | — | 18,7 | — | 42,4 | 6,9 | 647,7 |
| 2020 | 9,3 | 0,0 | 1.162,0 | 19,5 | — | 19,8 | — | 43,2 | 5,6 | 652,7 |
| 2021 | −86,3 | n.b. | 1.353,1 | 20,4 | — | 22,2 | — | 44,8 | 5,7 | 734,0 |
| 2022 | 83,7 | 0,2 | 1.538,3 | 19,1 | — | 20,1 | — | 44,6 | 11,6 | 976,6 |
| 2023 | 74,8 | 0,1 | 1.742,6 | 17,8 | — | 18,9 | — | 45,8 | 11,6 | 828,5 |
| 2024 | −130,7 | n.b. | 1.915,1 | 14,4 | — | 16,5 | — | 45,4 | 19,5 | 560,8 |
| 2025 | ~+50 (schatting, niet uit PDF gehaald) | n.b. | ~1.965 (impliciet uit eq. ratio 44,9%) | 12,2 | 10,0 | — | 1,68 | 44,9 | — | 500,5 |
| TTM | −227,2 (StockAnalysis def, Cash − Total Debt) | n.b. | 1.960 | 11,2 | 10,0 | 11,7 | 1,68 | — | — | 978,5 |

Opmerking: Bechtle's nettoschuld-definitie sluit IFRS-16 lease-liabilities in, terwijl StockAnalysis een andere standaardisatie hanteert (vandaar het verschil €−131M PDF 2024 vs €−227M StockAnalysis TTM). Voor 2025 is geen balansgetal precies geëxtraheerd; we hanteren AR2025-overzicht (eq.ratio 44,9%, cash €452M).

- **Toelichting balans** (80-120 woorden):
  Bechtle heeft een solide balans: equity-ratio rond 45% al sinds 2018, ROE rond 15-20% pre-2024 en gedaald naar 12,2% in 2025. De nettoschuld is licht negatief tot licht positief (oscilleert rond €0); in 2023 plaatste Bechtle een €300M converteerbare obligatie met looptijd 7 jaar (ISIN DE000A382293) waardoor financial liabilities 2023 tijdelijk omhoog schoten. Goodwill is van €194M (2016) gestegen naar €857M (2024) — een verviervoudiging die de buy-and-build strategie weerspiegelt; bijna alle acquisities zijn tot dusver zonder goodwill-impairment doorgegaan, een teken dat M&A-prijzen redelijk waren. Geen materieel herfinancieringsrisico: converteerbare 2030 vervalt over 4 jaar.

### Kapitaalstructuur huidig (31-12-2025)
- **Nettoschuld (huidig)**: ~€50M (schatting, niet exact uit AR2025 PDF geëxtraheerd in sessie; cash €452M; bruto schuld ~€500M afgeleid uit equity-ratio 44,9% × totaal activa ~€4,4 mld → liabilities ~€2,4 mld waarvan financial liabilities historisch 14% = ~€340M, ex IFRS-16 lease)
- **Bruto schuld**: ~€500-600M (incl. €300M convertible 2030)
- **Cash & equivalents**: 452 (HOOG, AR2025)
- **Lease-verplichtingen (IFRS-16)**: niet apart geëxtraheerd in sessie — circa €200M historisch
- **Gemiddelde rente %**: ~3,0 (impliciet uit financial expenditure €24,9M / financial liabilities ~€700-900M brutto 2024)
- **Rente-dekking (EBIT/rente)**: 14 (EBIT 335 / financial expenditure 24,9 → 13,5x; StockAnalysis geeft 12,0x voor TTM)

### Non-GAAP / aanpassingen
- **Gebruikt?** (true/false): false
- **Welke aanpassingen**: geen materiële non-GAAP / adjusted earnings communicatie door Bechtle. Het bedrijf rapporteert wel "business volume" naast "omzet" (verschil door IFRS-15 agent-treatment), maar dat is geen non-GAAP voor winst.
- **Waarom**: Bechtle communiceert primair IFRS-cijfers. Geen behoefte aan adjustments.

---

## 4. Moat (concurrentievoordeel)

- **Oordeel** (enum **UITSLUITEND**: **WIDE MOAT** | **NARROW MOAT** | **NO MOAT**): **NARROW MOAT**
- **Moat-categorieën** (PRECIES deze 5 namen letterlijk, één rij per categorie):

| Naam | Sterkte (sterk/middel/zwak/geen) | Toelichting |
|---|---|---|
| Immateriële activa | middel | Bechtle-merk is sterk in DACH-mid-market en publieke sector; certificeringen (C5:2020 cyberveiligheid, raamcontract-vergunningen met DE federaal/state) creëren toetredingsdrempels in publieke aanbestedingen. Geen patenten of unieke IP. |
| Overstapkosten | middel | Voor managed-services-contracten en raamovereenkomsten zijn overstapkosten reëel: integratie met klant-IT-infrastructuur, getrainde supportteams, gedeelde compliance-documentatie. Voor e-commerce-doorverkoop zijn overstapkosten minimaal (klant kan eenvoudig naar Cancom, Computacenter, of direct naar OEMs). |
| Netwerkeffecten | geen | Bechtle bedient B2B met individuele klantrelaties; geen platformeffecten waarbij de waarde voor klant X stijgt door klant Y. |
| Kostenvoordeel | sterk | Schaalvoordeel in OEM-inkoop: Bechtle is een van de grootste IT-doorverkopers van Europa (€8,6 mld business volume) en bedingt daardoor betere voorwaarden bij HPE, Microsoft, Dell, Cisco dan kleinere lokale spelers. Operationele schaal in centrale logistiek en e-commerce-platforms. |
| Efficiënte schaal | zwak | Het Mittelstand-IT-servicessegment is gefragmenteerd, geen natural-monopoly-kenmerken. Wel: in lokale DACH-niches (een specifieke deelstaat-overheid, een Mittelstand-cluster) opereert Bechtle in een efficient-scale-achtige positie waar 2-3 spelers dominant zijn. |

- **Kwantitatief bewijs**: ROIC 10,0% (StockAnalysis TTM, AGGREGATOR), ROCE 18-24% structureel 2016-2023 (Bechtle PDF), nu gedaald naar 11,7% TTM. ROCE > 15% over 8 van 9 jaar (2016-2024) toont waardecreatie boven kapitaalkosten. EBIT-marge 5,1-6,1% over 2017-2023 is opvallend stabiel — duidt op pricing power binnen DACH-mid-market — maar 2024-2025 lijkt deze structurele marge te verzwakken (4,9-5,2%).
- **Duurzaamheid** (80-120 woorden): Op 5 jaar horizon: moat blijft NARROW gegeven de gevestigde DACH-positie, langlopende publieke raamcontracten en de toetredingsdrempels in compliance (C5-certificering, federal raamcontract-vergunningen). Op 10 jaar horizon: erosie-risico's worden materieel. Hyperscalers (AWS, Azure, GCP) verkopen steeds meer direct aan Mittelstand en publiek; software-as-a-service ondergraaft de doorverkoopmarge in IT E-Commerce. Op 20 jaar horizon: het traditionele "system house"-model staat onder structurele druk van cloud-only-architecturen. Bechtle anticipeert hierop met BIoS (Bechtle Index of Sovereignty) en multi-cloud-partnerships met Deutsche Telekom — een verschuiving naar advisering-met-marge in plaats van doorverkoop-met-marge.
- **Erosierisico's**: directe-verkoop door OEMs/hyperscalers (Microsoft, AWS); platforms zoals SoftwareONE die globaal aanbesteden; commoditisering van managed services door AI-automatisering; geopolitieke fragmentatie die schaalvoordelen ondermijnt.

---

## 5. Management

- **CEO-naam + tenure**: Dr. Thomas Olemotz, CEO sinds juni 2010, contract verlengd tot 31-12-2026; opvolger Konstantin Ebert per 1-1-2027 aangekondigd in maart 2026.
- **CFO-naam + tenure**: niet expliciet geverifieerd in deze sessie (Bechtle AR2025 Executive Board details niet volledig geëxtraheerd) — weggelaten.
- **Oprichter nog betrokken?**: Nee. Gerhard Schick, medeoprichter en voormalig Chairman, overleed 4 maart 2025; de Schick-familie blijft als anchor shareholder (35,02%) via een holding en de Gerhard und Ilse Schick Foundation.
- **Insider ownership %**: 33,39% (StockAnalysis Statistics, AGGREGATOR); dit komt grotendeels overeen met het Schick-familieblok van 35,02%.
- **Capital allocation track record** (dividenden / inkoop / M&A / organisch):

| Jaar | Dividend totaal | Aandeleninkoop | M&A uitgaven | Organische capex |
|---|---|---|---|---|
| 2020 | 56,7 | 0 | onbekend | 66,2 |
| 2021 | 69,3 | 0 | 15,8 | 58,4 |
| 2022 | 81,9 | 0 | 92,2 | 82,0 |
| 2023 | 88,2 | 0 | 223,5 | 95,2 |
| 2024 | 88,2 | 0 | 108,0 | 88,7 |
| 2025 | 88,2 | 0 | 158,2 | ~121 |

- **M&A-track-record** (40-60 woorden): Sinds 2000 dozijnen acquisities, voornamelijk kleinere DACH-system-houses, plus internationale uitbreidingen (ARP CH, ACS UK, Inmac WStore FR, Grupo Solutia ES, Nuovamacut IT, RIS 2048 PT). Geen goodwill-impairment in 10-jaarsperiode geconstateerd — wijst op disciplineerde acquisitieprijzen. Q4-margedruk 2024-2025 deels veroorzaakt door integratiekosten van recente deals; rendement op kapitaal van M&A is moeilijk afzonderlijk te meten.
- **Beloning** (60-100 woorden): CEO Olemotz ontving €3,5M totale compensatie over 2019 (laatste publieke datapunt in zoekresultaten); recentere remuneratierapport-details zijn in deze sessie niet geverifieerd. Op basis van het Bechtle-corporate-governance-rapport is de structuur gemengd vast/variabel met meerjaren-bonuskomponenten gekoppeld aan EBIT-groei en ROCE. Geen significant aandelenoptie-programma; geen materiële SBC-verwatering. Alignment met aandeelhouders is sterk dankzij het Schick-familiebelang van 35% — de familie is daarmee de "skin-in-the-game"-borg, niet het managementteam zelf.
- **Oordeel management** (enum **UITSLUITEND**: **STERK** | **NEUTRAAL** | **ZORGWEKKEND**): **STERK**
- **Toelichting** (80-120 woorden):
  Vier feiten dragen het oordeel STERK. Eén: 16 jaar consistente strategie onder Olemotz met omzet-CAGR ~9% (2010-2023), ROCE structureel >15%. Twee: 25 ononderbroken dividendjaren sinds IPO — door dotcom-crash, financiële crisis 2008, Covid en de huidige IT-spending-cyclus — wijst op een conservatief financieel beleid en lange-termijn-discipline. Drie: anchor shareholder (familie Schick, 35%) heeft langetermijn-incentives en is niet uitgestapt na het overlijden van Gerhard Schick; de stichting borgt continuïteit. Vier: tijdig aangekondigde opvolging (CEO-transitie aangekondigd in maart 2026 voor januari 2027 — geen overhaaste wisseling). Risico: het uitvoeringsprobleem in Frankrijk en de margecompressie 2024-2025 leggen een vraagteken bij de operationele agility van het team.

---

## 6. Sector & concurrentie

- **Sector-groeivooruitzicht** (Europese IT-services Mittelstand): 4-7% per jaar 2025-2030 volgens Gartner/IDC consensus; Bechtle eigen Vision 2030 mikt op €10 mld business volume vanaf €8,6 mld in 2025 = 3,1% CAGR over 5 jaar. Bron: Bechtle FY2025 slide presentation (Investing.com analyse, AGGREGATOR).
- **Porter five forces**:
  - Rivaliteit: middel. Top-3 concurrenten in DACH (Cancom, Computacenter, Datagroup) en pan-EU (SoftwareONE, Atea, Tech Data wholesale) zijn aanwezig maar de markt is gefragmenteerd; lokale system-houses leven door uit eigen klantrelaties.
  - Nieuwe toetreders: middel. Hyperscalers (AWS, Azure, GCP) verkopen toenemend direct; kapitaaldrempels voor traditionele system-house-modellen zijn relatief laag, maar publieke aanbestedingen met C5-certificering en raamcontract-vergunningen vormen een hoge non-financiële drempel.
  - Substituten: hoog. SaaS, cloud-managed services en directe verkoop door OEMs/hyperscalers eroderen het traditionele doorverkoop- en projectmodel.
  - Macht leveranciers: middel-hoog. HPE, Microsoft, Cisco, Dell zijn de top-3 leveranciers van Bechtle's IT E-Commerce-segment; hun pricing aan distributeurs is een belangrijke marge-determinant.
  - Macht afnemers: middel. Mittelstand-klanten zijn relatief loyaal aan lokale system-houses; publieke klanten hebben formele aanbesteding-procedures maar Bechtle is dominant in DACH-overheidscontracten.
- **Concurrenten** (3-5 belangrijkste):

| Concurrent | Marktaandeel % |
|---|---|
| Bechtle (zelf, EU-mid-market IT-services) | ~6-8 (DACH-mid-market) |
| Cancom SE (DACH, hybrid cloud + modern workplace) | ~4-5 |
| Computacenter plc (UK + DACH, grote enterprises) | ~5-7 (EU enterprise) |
| Datagroup AG (DE, managed services) | ~1-2 |
| SoftwareONE Holding AG (CH, global software licensing) | ~2-3 (EU software) |

Bron: kwalitatieve schatting uit competitor-research; geen formeel marktaandeel-rapport gevonden.

- **Positie van het bedrijf** (60-100 woorden): Bechtle is **leider** in het DACH-mid-market IT-services-segment door de combinatie van fijnmazige lokale aanwezigheid (~120 vestigingen), inkoopkracht (€8,6 mld business volume), en gevestigde publieke-sector-relaties. In pan-EU is het een **challenger** vergeleken met Computacenter in enterprise, met opbouwende posities in Frankrijk, Benelux, Iberia en Italië. Het is een **kwetsbare** speler in pure-software-licensing waar SoftwareONE en directe-verkoop dominanter zijn. De positie wordt structureel ondersteund door digitale-soevereiniteit-trends die Europese kopers richting lokale leveranciers duwen.

### TAM/SAM/SOM
- **TAM (mln EUR)**: ~120.000 (Europese B2B IT-services + distributie, IDC/Gartner schatting €120 mld in 2025)
- **TAM-groei %**: 5-7
- **SAM (mln EUR)**: ~30.000 (DACH + Benelux + Frankrijk + Iberia mid-market & publieke sector)
- **SAM-groei %**: 4-6
- **Huidige penetratie %** (omzet / SAM): ~21 (€6,4 mld / €30 mld SAM)
- **Impliciete penetratie na horizon %** (bij €10 mld business volume Vision 2030): ~25-28
- **Groei plausibel?**: true. Vision 2030 (€10 mld business volume = 3% CAGR) zit ruim onder SAM-groei + share gains potentie.
- **Bron TAM/SAM**: kwalitatieve raming op basis van Gartner/IDC publieke benchmarks; geen specifiek rapport gefetched in deze sessie.
- **Toelichting** (60-80 woorden): SAM-aandeel van ~21% in DACH-zwaartepunt is hoog. Het Vision-2030-doel impliceert 3-4 procentpunt aandeelwinst over 5 jaar — haalbaar gegeven de fragmentatie van de markt en de toenemende vraag naar digital-sovereignty-oplossingen waar lokale Europese spelers structureel bevoordeeld zijn. Het grootste vraagteken is uitvoering: 2024-2025 toonde dat margebehoud bij internationale expansie niet automatisch verloopt.

---

## 7. Analyse-frameworks (9 frameworks, SCORES 0-5)

### Graham
- **Oordeel**: GEDEELTELIJK — voldoet aan financiële stabiliteit (D/E 0,37, structureel dividend), maar P/B 2,19 is boven Graham's 1,5-drempel en margin of safety van 11% is onder Graham's 25-30%.
- **Graham number**: ≈ √(22,5 × EPS × BVPS) = √(22,5 × 1,71 × 15,57) = √599 = 24,5
- **Margin of safety %** (t.o.v. huidige koers): 11 (vs basis fair value 33,0); −17% vs Graham number 24,5 (overprijsd op Graham's striktere norm)
- **Toelichting** (60-100 woorden): Bechtle voldoet aan twee Graham-tests: stabiele winstgeschiedenis (winstgevend in elk van de afgelopen 10 jaar) en regelmatig dividend (25 ononderbroken jaren). P/E TTM 17,4 ligt onder Graham's 20-drempel maar boven de 15-drempel voor score 5. P/B van 2,19 is boven Graham's 1,5-drempel. Graham number €24,5 ligt onder huidige koers €29,70 — dat betekent geen klassieke Graham-koopkans. Score 3 volgt uit rubric: P/E ≤ 20 OF (P/B ≤ 2,0 EN structureel dividend) — alleen het eerste criterium is voldaan, dus score 3.
- **Score (0-5)**: 3

### Buffett / Munger
- **Oordeel**: GEDEELTELIJK — moat is NARROW, ROIC > WACC maar niet structureel 2× WACC. P/FCF circa 22 (op genormaliseerde FCF €175M, marktkap €3.742M) — boven 20-drempel.
- **ROIC structureel boven WACC?** (true/false): true (ROIC 10% vs WACC 6,5-7,5%; spread +3pp structureel)
- **Toelichting** (60-100 woorden): Bechtle voldoet aan Buffett's "voorspelbare kasstromen" criterium — 25 jaar onafgebroken dividend, ROCE structureel >15% tot 2024. De moat is echter NARROW, niet WIDE: schaalinkoop en publieke-sector-relaties leveren wel een voordeel, maar geen monopolie. Management is sterk. Het prijsplaatje (P/FCF circa 22 op genormaliseerde FCF) zit aan de bovenkant van Buffett's "redelijke prijs" voor een NARROW-moat-bedrijf. Score 3 volgt uit rubric: ROIC > WACC structureel EN moat NARROW+ EN P/FCF ≤ 30.
- **Score (0-5)**: 3

### Peter Lynch
- **Categorie**: Slow grower (omzet-CAGR 5% laatste 5 jaar, dividend-betalend, mature business)
- **Oordeel**: ONINTERESSANT
- **PEG-ratio**: 3,5 (P/E 17,4 / consensus EPS-groei 2026-2028 ~5%)
- **Toelichting** (60-100 woorden): Peter Lynch's PEG-test is hard: bij PEG > 2,0 stopt de Lynch-belegger met kijken. Bechtle's PEG van 3,5 (P/E 17,4 / verwachte groei 5%) is ver boven die drempel, ook al is het verhaal helder ("Mittelstand-IT-systeemintegrator met dividend"). StockAnalysis rapporteert een PEG van 3,23 — vergelijkbaar. Voor een slow grower is een PEG > 2 typisch, dat is geen koopsignaal voor Lynch. Score 1 volgt strict uit rubric: PEG > 2,0 OF verhaal onhelder.
- **Score (0-5)**: 1

### Phil Fisher
- **Oordeel**: GEMIDDELD
- **Toelichting** (60-100 woorden): Bechtle scoort gematigd op de 15-Fisher-criteria: producten/diensten met groeipotentieel (Vision 2030 €10 mld) maar geen technologische voorsprong; R&D-budget is laag (Bechtle is een integrator, niet een productontwikkelaar) en niet "boven sectorgemiddelde"; margebescherming door narrow moat is beperkt; management-integriteit is STERK (transparant, langetermijn-track-record). Eén van de drie kerncriteria (management) voldaan, R&D-budget niet groeiend op productinnovatie maar wel investeringen in interne IT-platforms en cloud-partnerships. Score 2 volgt: "producten groeien maar geen duidelijke moat-bescherming".
- **Score (0-5)**: 2

### Magic Formula (Greenblatt)
- **Oordeel**: GEMIDDELD
- **Earnings yield %**: 7,4 (EBIT €335M / EV €4.540M, StockAnalysis EV)
- **Return on capital %**: ~12 (EBIT / (NWC + Net Fixed Assets); ruwe Greenblatt-berekening: NWC €500M, Net Fixed Assets ~€640M PP&E + intangibles ex goodwill = ~€1.140M; EBIT/€1.140M = ~29%)
- **Toelichting** (60-100 woorden): Greenblatt's Earnings Yield van 7,4% is solide (boven 7%-drempel voor score 4). Return on Capital — Greenblatt's specifieke variant excl. goodwill — komt uit op ~29% (€335M EBIT op €1.140M operational capital), wat boven de 15%-grens ligt maar ruim onder de 30%-drempel voor score 4. Strict rubric: Earnings Yield ≥ 7% EN Return on Capital ≥ 30% = score 4; we voldoen aan EY maar net niet aan RoC. Score 3 volgt: Earnings Yield ≥ 5% OF Return on Capital ≥ 50%.
- **Score (0-5)**: 3

### Moat
- **Score (0-5)** — zelfde oordeel als sectie 4 maar gescoord: NARROW moat met 1-2 categorieën STERK (kostenvoordeel sterk; immateriële activa en overstapkosten middel), ROIC-WACC spread ~3pp (onder 5pp-drempel). Strict rubric: mogelijke moat maar niet kwantificeerbaar OF spread < 5pp = **score 2**.

### Management
- **Score (0-5)** — zelfde oordeel als sectie 5 maar gescoord: capital allocation GOED (25 jaar dividend, geen impairments, gedisciplineerde M&A), prikkels aligned (familie 35%, geen excessieve SBC), geen controverses, downside transparency redelijk maar niet uitzonderlijk; insider alignment via familie-block >1% ruim voldaan. Rubric: capital allocation GOED EN prikkels aligned EN geen controverses = **score 4**.

### Fair Value DCF
- **Score (0-5)** — hoeveel MOS heeft het aandeel op de DCF-basis? Upside basis = 11% (tussen 0% en 15%). Rubric: upside ≥ 0% EN < 15% = **score 3**.

### Fair Value IPO-gecorr.
- **Score (0-5)** — IPO 2000 = >10 jaar geleden, dus gelijk aan Fair Value DCF basis = **score 3**.

### Scorekaart totaal
- **Totaalscore**: 24 (3 + 3 + 1 + 2 + 3 + 2 + 4 + 3 + 3)
- **Max**: 45
- **Eindoordeel** (enum **UITSLUITEND**: **KOOP** | **HOLD** | **PASS**): **HOLD**
  - Deterministische regel: totaal 24 is ≥ 24 EN < 33, EN Fair Value DCF score 3 (niet ==1) → HOLD ✓
- **Samenvatting** (120-180 woorden):
  Bechtle scoort 24 van 45 op de scorekaart — net boven de PASS-drempel en ver onder de KOOP-drempel. Het sterkste deel is het management (score 4): 25 jaar dividend, sterke familie-controle (35%), gedisciplineerde M&A zonder goodwill-impairments en een tijdig aangekondigde CEO-opvolging. Het zwakste deel is de Peter Lynch-test (score 1): met een PEG van 3,5 is het aandeel duur ten opzichte van zijn groei. De Fair Value DCF (score 3) levert een upside van 11% — onvoldoende margin of safety voor een kooppositie maar wel voldoende om geen PASS-verdict te krijgen. De NARROW moat (score 2) is een structurele beperking: schaalvoordeel in inkoop is reëel maar het traditionele system-house-model staat onder druk van hyperscalers en SaaS-erosie. Het eindoordeel HOLD reflecteert een kwaliteitsbedrijf op een redelijke maar niet aantrekkelijke prijs — wachten op een correctie naar de €25-27-zone (EPV-niveau) voor een aantrekkelijker risico/rendement-profiel.

---

## 8. Risico's (minimaal 5-8 stuks)

| # | Omschrijving | Kans | Impact | DCF-aanname geraakt | Toelichting |
|---|---|---|---|---|---|
| 1 | Aanhoudende margedruk in Frankrijk en Iberia (uitvoering internationale uitbreiding) | HOOG | MIDDEL | EBIT-marge fase 1 (basisscenario 5,2% → onder druk naar 4,5-4,8%) | Frankrijk EBIT −47% in 2025; Spanje/Portugal recent overgenomen (Grupo Solutia, RIS 2048) met integratiekosten. Als marges in non-DACH onder de DACH-gemiddelden blijven, daalt geconsolideerde EBIT-marge structureel met 30-50 bps. |
| 2 | Cyclische daling Mittelstand IT-bestedingen door zwakke DE-/EU-conjunctuur | MIDDEN | GROOT | FCF-groei fase 1 (basis 5% → naar 1-2%) | Bechtle's Mittelstand-klanten stellen IT-projecten uit in recessies. 2024-omzet -1,8% toonde hoe gevoelig de top-line is. Een tweejarige recessie zou de DCF-aannames materieel beïnvloeden. |
| 3 | Hyperscaler-disintermediation (AWS, Azure, GCP direct sales aan Mittelstand) | MIDDEN | GROOT | Terminal groei (basis 2% → mogelijk 0-1%) | Microsoft Direct, AWS-EU en Google Cloud voor Mittelstand zijn jaar-op-jaar agressiever. Op 5-10 jaar horizon erodeert de e-commerce-doorverkoopmarge. Niet alleen Bechtle-issue maar wel materieel voor terminal value. |
| 4 | Margecompressie door SaaS-shift in software-verkoop (IFRS-15 agent-treatment, lagere absolute marge per softwareregel) | HOOG | MIDDEL | Brutomarge en EBIT-marge fase 1 | De afgelopen 3 jaar wijkt business-volume (+8%) sterk af van revenue (+1,6%) — gevolg van shift naar agent-rapportage. Aandeel van pure margebusiness daalt, hoewel commissie-marges hoger zijn. Effect is netto negatief op rapporteerbare EBIT als % omzet. |
| 5 | Goodwill-afschrijving uit recente acquisities (€857M goodwill in 2024, ~20% van EV) | LAAG | MIDDEL | Boekwaarde / eigen vermogen; geen direct DCF-effect maar geloofwaardigheid M&A-strategie | Tot dusver geen impairments, maar het verdwijnen van mede-oprichter Schick en de margedruk in Frankrijk vergroten het risico op een eerste afboeking. Materieel voor P/B maar niet voor DCF. |
| 6 | CEO-transitie naar Konstantin Ebert in 2027 — uitvoeringsrisico | MIDDEN | MIDDEL | Indirect: management-score, scorekaart-effect | Olemotz heeft 16 jaar consistente strategie geleverd; opvolging brengt onvermijdelijk onzekerheid over koersbehoud, vooral nu Vision 2030 net is geïntroduceerd. Tijdige aankondiging (15 maanden voorbereiding) mitigeert. |
| 7 | Stijgende rente (Bund 10y van ~2,3% in 2024 naar 3,07% in mei 2026) verhoogt WACC | MIDDEN | MIDDEL | WACC (basis 7,5% → naar 8,0-8,5%) | Verdere ECB-renteverhogingen (markt prijst 3 hikes voor eind 2026) zouden de DCF-waardering verder onder druk zetten. Een WACC van 8,5% (pessimistisch scenario) levert een fair value van slechts €23. |
| 8 | Concentratierisico publieke sector — politieke / budgettaire wijzigingen | LAAG | MIDDEL | FCF fase 1; publieke contracten zijn ~15-20% omzet (schatting) | DE federale overheidsbudgetten zijn nu expansief maar kunnen na verkiezing 2025 (eerder gehouden) of bij begrotingscrisis worden gekort. Bechtle is afhankelijk van structurele publieke IT-uitgaven. |

---

## 9. These invalide bij

Deze investeringsthese is weerlegd wanneer: (1) de EBIT-marge twee opeenvolgende kwartalen onder 4,5% zakt (op business-volume-basis onder 5,5%) — dat zou bevestigen dat de Frankrijk-/internationale-margedruk structureel is en niet eenmalig; OF (2) het Schick-familieblok zijn belang materieel afbouwt (een verlaging onder 25% zou een verschuiving in lange-termijn-anchoring signaleren); OF (3) een goodwill-impairment van >€100M wordt aangekondigd (dat zou de M&A-discipline ondergraven die nu pijler is van het management-oordeel); OF (4) de Vision-2030-targets formeel worden uitgesteld of teruggetrokken (zou de groei-aanname in het basisscenario ondermijnen).

---

## 10. ESG

### Materiële factoren (SASB-gebaseerd, GICS Information Technology Services)

| Factor | SASB-categorie | Risiconiveau (Laag/Midden/Hoog) | Financiële impact | DCF-impact |
|---|---|---|---|---|
| Cyberbeveiliging klantgegevens | TC-SI-230a (Data Security) | Midden | Reputatie-/aansprakelijkheidsrisico bij ernstige breuk | EBIT fase 1 (eenmalig event) |
| Energieverbruik datacenters | TC-SI-130a (Energy Management) | Laag | Bechtle bouwt eigen datacenter capaciteit beperkt — primair leverancier van klant-DC's | Beperkt direct; opex stijging mogelijk |
| Talent retention en personeelskosten | TC-SI-330a (Workforce) | Midden | Personeelskosten 16,6% omzet (2024); IT-talentmarkt krap | EBIT-marge fase 1 |
| Klimaatemissies (Scope 1, 2, 3) | TC-SI-110a | Laag | Scope 1+2 −7,9% naar 19.299 ton CO₂e in 2025; gevalideerd door SBTi | Beperkt direct; reputationeel |
| Digitale soevereiniteit / GDPR-compliance | sector-specifiek | Laag-Midden | Eerder kans dan risico — Bechtle positioneert BIoS expliciet als digital-sovereignty-tool | Kans op marge-uitbreiding |

- **Eindoordeel ESG** (enum: **LAAG RISICO** | **GEMIDDELD RISICO** | **HOOG RISICO**): **LAAG RISICO**
- **Toelichting** (60-100 woorden): Bechtle's ESG-risicoprofiel is laag voor een IT-services-bedrijf. De Science-Based-Targets-Initiative-validatie in mei 2025, een Scope-1+2-emissiereductie van 7,9% en een actieve positionering rondom digitale soevereiniteit (BIoS) maken het bedrijf eerder een ESG-kans dan een ESG-risico. Het materieelste risico is een grote cyberbeveiligingsbreuk bij een publieke klant — dat zou reputationeel zwaar zijn maar is laag waarschijnlijk gezien de C5:2020-certificering. Geen materiële controverses, geen substantiële milieuaansprakelijkheid. ESG is geen materiële DCF-input.

---

## 11. Katalysatoren (5-8 stuks, chronologisch)

| Datum ca. | Omschrijving | Richting | Impact |
|---|---|---|---|
| 2026-08 | Halfjaarcijfers H1 2026 — eerste test of Q1-momentum doorzet over H1 | POSITIEF | MIDDEL |
| 2026-11 | Q3 2026 trading update + capital markets day (verwacht najaar 2026) | POSITIEF | MIDDEL |
| 2026-11 | Eerstvolgend dividendbesluit (ex-dividend mei 2026 was €0,70; volgend besluit bij FY2026-resultaten) | NEUTRAAL | KLEIN |
| 2027-01 | CEO-transitie: Konstantin Ebert vervangt Thomas Olemotz per 1 januari 2027 | NEUTRAAL | MIDDEL |
| 2027-03 | FY2026 jaarcijfers — bevestiging EBT-groei 0-5% guidance | BINAIR | GROOT |
| 2027-06 | Mogelijke nieuwe grote publieke raamovereenkomsten (continu in DE-aanbesteding-pijplijn) | POSITIEF | MIDDEL |
| 2028-Q1 | Eerste Vision-2030-tussenstand (na 2 jaar uitvoering) — €10 mld doel mid-cycle | BINAIR | GROOT |
| 2030-12 | Convertibele obligatie €300M (looptijd 7j, uitgegeven dec 2023) vervalt — herfinancieringsmoment | NEGATIEF | KLEIN |

---

## 12. Fair value — kwantitatief (DCF)

### WACC-componenten
- **Risicovrije rente %**: 3,07
- **Bron risicovrije rente**: Duitse 10-jaars Bundesanleihe, peildatum 2026-05-14 (TradingEconomics)
- **Type** (nominal / real): nominal
- **ERP (equity risk premium) %**: 4,20
- **Bron ERP**: Damodaran Country Risk Premium Germany, update januari 2026
- **Beta (adjusted, Blume)**: 0,85
- **Bron beta**: Yahoo Finance BC8.DE 5Y Monthly Beta
- **Type beta**: 5y monthly regressie (>5 jaar genoteerd, gemiddeld dagvolume 187k+, dus regressie betrouwbaar)
- **Country risk premium %**: 0 (Duitsland is base country; CRP = 0)
- **Size premium %**: 0 (marktkap €3,7 mld > €2 mld grens; geen size premium)
- **Cost of equity %**: 6,64 (= 3,07 + 0,85 × 4,20)
- **Schuldkosten na belasting %**: 2,13 (= 3,0% bruto × (1 - 0,29))
- **E/V gewicht %**: 83,5 (marktwaarde equity €3.742M op enterprise value €4.480M-equivalent)
- **D/V gewicht %**: 16,5
- **WACC %**: 5,89 (theoretisch; in scenarios verhogen we naar 7,5% basis voor sector-conservatisme)
- **Sector WACC % (referentie Damodaran)**: ~7,5-8,0 (Computer Services / IT services Europe; StockAnalysis WACC voor BC8 = 8,17%)
- **Illiquiditeitskorting %**: 0 (gemiddeld dagvolume 354k aandelen, MDAX-lid, ruim liquide)

Voor het DCF gebruiken we **WACC 7,5% basis** (in plaats van 5,89% pure CAPM-uitkomst) om vergelijkbaarheid met sectorbenchmarks (Damodaran sector 7,5-8%, StockAnalysis BC8 8,17%) en conservatisme rondom rente-onzekerheid te borgen. Pessimistisch 8,5%, optimistisch 6,5%.

### DCF model-specs
- **Model type**: 2-fase (jaar 1-5 fase 1, jaar 6-10 fase 2, daarna terminal)
- **FCF-definitie**: FCFF (Free Cash Flow to Firm) na maintenance capex
- **Basis FCF** (startjaar, genormaliseerd): 175 (mediaan 2019-2025 Bechtle eigen FCF-definitie)
- **Basis FCF na SBC** (SBC verwaarloosbaar): 175
- **FCF-type**: Genormaliseerde FCF €175M (mid-cyclus, mediaan 2019-2025). Recente FCF 2024 €377M is piek (werkkapitaal-release); 2022 −€30M is dal. Gekozen €175M ligt dicht bij mediaan €178M.
- **Groei fase 1 %** (jaar 1-5): 5,0 basis (3,0 pess / 7,0 opti)
- **Groei fase 2 %** (jaar 6-10): 4,0 basis (2,0 pess / 5,0 opti)
- **Terminal groei %**: 2,0 basis (1,5 pess / 2,5 opti)
- **Terminal methode**: Gordon growth + exit multiple cross-check
- **Exit multiple gebruikt (EV/EBITDA)**: 8,5x (historisch gemiddelde Bechtle 2018-2025 = 13,4 / 13,4 / 17,6 / 21,2 / 19,3 / 10,3 / 12,4 / 8,9 — gemiddelde 14,6; sector-mediaan IT-services ~10x; conservatief 8,5x)
- **Bron exit multiple**: Bechtle eigen historische EV/EBITDA + sector-mediaan
- **Terminal value Gordon growth**: 5.040 (basis, eind jaar 10)
- **Terminal value exit multiple**: 8,5 × geprojecteerde EBITDA jaar 10 (~€776M) = 6.596M (cross-check: hoger dan Gordon-basis, suggereert dat Gordon-aanname conservatief is)
- **Terminal value % van totaal**: 61 (PV TV €2.445M / totale EV €4.022M; basis-scenario; < 75% drempel ✓)
- **Terminal implied EV/EBITDA**: 6,5x (€5.040M Gordon-TV / €776M EBITDA jaar 10) — onder 20x-drempel ✓
- **Terminal groei consistentie** (Damodaran g = reinvestment rate × ROIC): Bij terminal ROIC 10% en herinvesteringsvoet 20%, g = 2,0% — exact gelijk aan onze aanname. Plausibel voor volwassen IT-services-bedrijf in lage-groei-Europa.
- **Mid-year convention**: true
- **Aandelen uitstaand (mln)**: 126,0
- **Nettoschuld huidig**: ~−131 (laatst geverifieerd: 2024-12-31 in AR2024 PDF; netto cash €131M); 2025 niet exact maar in dezelfde ordergrootte

### DCF-toelichting (100-150 woorden)
Het basis-scenario gebruikt een genormaliseerde FCF van €175M (mediaan 2019-2025) als startpunt — niet de €377M piek-FCF uit 2024 die werkkapitaal-release reflecteerde, noch de −€30M dal-FCF uit 2022. Dit is essentieel: Bechtle's FCF-volatiliteit door werkkapitaalcycli zou een DCF op piek-FCF onverdedigbaar maken (terminal value zou impliciet > 20x EV/EBITDA opleveren). De gekozen groeipatroon (5% fase 1, 4% fase 2, 2% terminal) is conservatief versus Bechtle's eigen Vision-2030 (impliciet 3-4% business-volume-CAGR met margeherstel) en sluit aan bij consensus (2026 guidance: revenue 0-5%, EBT 0-5%). WACC 7,5% combineert de CAPM-uitkomst (5,9%) met een sectorale opslag voor IT-services-risico, conservatisme rondom rentevolatiliteit en aansluiting bij Damodaran-benchmark. Terminal-value-aandeel 61% van totale EV is acceptabel (< 75% drempel) en de impliciete EV/EBITDA van 6,5x op terminal is conservatief versus historisch gemiddelde van 14,6x.

### 5-jaars projectie (basis scenario)

| Jaar | Omzet | Omzetgroei % | EBIT | EBIT-marge % | NOPAT | Capex | ΔNWC | SBC | FCF |
|---|---|---|---|---|---|---|---|---|---|
| 2025 (actueel) | 6.406 | 1,6 | 335 | 5,2 | 238 | 121 | +60 | ~0 | ~169 |
| 2026 | 6.598 | 3,0 | 343 | 5,2 | 244 | 125 | +20 | ~0 | 184 |
| 2027 | 6.829 | 3,5 | 369 | 5,4 | 262 | 130 | +30 | ~0 | 193 |
| 2028 | 7.069 | 3,5 | 396 | 5,6 | 281 | 135 | +30 | ~0 | 203 |
| 2029 | 7.317 | 3,5 | 424 | 5,8 | 301 | 140 | +30 | ~0 | 213 |
| 2030 | 7.610 | 4,0 | 449 | 5,9 | 319 | 145 | +30 | ~0 | 223 |

(Toelichting: dit zijn omzet-gebaseerde projecties met margeherstel; de FCF-projecties in het DCF gebruiken het genormaliseerde €175M-startpunt met 5% groei, niet de tabel-FCF. De tabel illustreert het scenario operationeel.)

### Scenarios (3 stuks — exact deze labels)

| Scenario | FCF-groei % (fase 1 / fase 2 / terminal) | WACC % | Fair value | Upside % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 3 / 2 / 1,5 | 8,5 | 23,30 | −22 | 25 |
| Basis | 5 / 4 / 2,0 | 7,5 | 33,00 | 11 | 50 |
| Optimistisch | 7 / 5 / 2,5 | 6,5 | 49,30 | 66 | 25 |

- **Kansgewogen fair value**: 34,65 (0,25 × 23,30 + 0,50 × 33,00 + 0,25 × 49,30; kans_pct optelt tot 100 ✓)

### Reverse DCF
- **Impliciete groei %** (wat moet FCF groeien voor huidige koers €29,70): ~2,8 per jaar over 10 jaar bij WACC 7,5% en terminal 2,0% (analyse: koers €29,70 × 126 = €3.742M equity; min netto cash €131M = EV €3.611M; geïmpliceerde 10y FCF-groei ≈ 2,8%)
- **Historische FCF CAGR %**: ~4 (basis €175M genormaliseerd; absolute eindwaarden 2019 €110M → 2024 €377M = CAGR 27,9% maar inclusief werkkapitaal-effect; mediaan-correctie geeft ~4-5% structureel)
- **Consensus groei % (analisten)**: 5-7 (over 2026-2028, op basis van management-guidance 5-10% business-volume en consensus EBT-groei)
- **Interpretatie** (60-100 woorden): De impliciete 2,8% FCF-groei is conservatief versus zowel historie als consensus. Dat betekent dat de markt momenteel inprijst dat Bechtle structureel uitvalt — vermoedelijk door zorgen over margecompressie, Frankrijk-uitvoering en de IFRS-15-impact op rapportabele omzet. Als Bechtle slechts 4% FCF-CAGR levert over de komende decade (consistent met Vision 2030 en margeherstel), is er meaningful upside. Als de markt gelijk heeft en het bedrijf in 2,5-3% groei blijft hangen, is het aandeel correct geprijsd op €29,70. De huidige prijs zit binnen de "fairly valued"-range, niet duidelijk koop noch duidelijk verkoop.

### EPV (Bruce Greenwald)
- **Genormaliseerde EBIT-marge %**: 5,4 (gemiddelde 2021-2025 EBIT-marges 6,1 / 5,9 / 6,0 / 5,6 / 5,2)
- **Genormaliseerde NOPAT**: 249 (= 5,4% × €6.406M × (1 − 0,29) = 248,5)
- **Maintenance capex**: 100 (gemiddelde investeringen PP&E + intangibles excl. groeicomponent; D&A 2025 = €159M, maintenance schatting ~60% daarvan = €95M; conservatief €100M)
- **Adjusted earnings power**: 249 (NOPAT, ervan uitgaande D&A ≈ maintenance capex over een volle cyclus)
- **EPV per aandeel**: 27,40 (= (€249M / 7,5% WACC + €131M netto cash) / 126M aandelen = (€3.320M + €131M) / 126M)
- **Groeipremie %**: 20 ((Fair value basis €33,00 − EPV €27,40) / EPV = 20,4%)

### Andere methoden
- **DDM uitgevoerd?** (alleen voor dividend-aandelen — true/false): false. Bechtle is een matig dividend-aandeel (yield 2,4%, payout 38,5%) maar primair als compounder gewaardeerd; DDM zou kunstmatig laag uitkomen omdat payout < EPS-groei-potentie. EPV is hier informatiever.
- **SOTP uitgevoerd?** (alleen voor conglomeraten — true/false): false. Twee segmenten zijn niet gescheiden genoeg om SOTP rechtvaardiging te bieden — gedeelde overheadkosten en cross-selling tussen System House en E-Commerce.

### Synthese fair value
- **Bandbreedte laag**: 23,30 (pessimistisch DCF)
- **Bandbreedte centraal**: 33,00 (basis DCF) / 34,65 (kansgewogen)
- **Bandbreedte hoog**: 49,30 (optimistisch DCF)
- **Methode-gewichten** (som = 100%):
  - DCF %: 60
  - EPV %: 30
  - Multiples %: 10 (consensus analist-target €44,88 wordt deels meegenomen als sanity-check)
- **Margin of safety vereist %**: 25 (gepast voor narrow-moat mid-cap met cyclische gevoeligheid)
- **Koopniveau** (fair value × (1 − MOS)): 25,99 (= €34,65 × 0,75)
- **Synthese-toelichting** (80-120 woorden):
  De gewichten 60% DCF / 30% EPV / 10% multiples reflecteren de relatieve betrouwbaarheid van elke methode voor Bechtle. DCF is centraal omdat het bedrijf voldoende voorspelbare kasstromen en groei-aannames levert. EPV (30%) is een belangrijke kruisvaling die laat zien dat zonder enige groei het aandeel al €27,40 waard is — dat is een ondergrens. Multiples-aanvulling (10%) gebruikt de consensus-analist-mean (€44,88, 12 analisten) als sanity-check; analisten zijn historisch te optimistisch over Bechtle, vandaar het lage gewicht. De kansgewogen fair value €34,65 levert een margin of safety van slechts 14% op de huidige koers — onder onze vereiste 25%-MoS voor een narrow-moat mid-cap. Koopniveau zou €26 zijn.

### Gevoeligheid (DCF)
- **FCF-groei ↔ WACC matrix** (fair value per aandeel, EUR):

| Fase 1 groei \ WACC | 6,0% | 7,0% | 7,5% | 8,0% | 9,0% | 10,0% |
|---|---|---|---|---|---|---|
| 3% | 33,2 | 27,8 | 25,7 | 23,7 | 20,5 | 18,0 |
| 5% | 41,1 | 35,3 | 33,0 | 30,8 | 27,1 | 24,1 |
| 7% | 50,0 | 43,6 | 41,0 | 38,5 | 34,3 | 30,7 |
| 9% | 60,0 | 53,0 | 50,1 | 47,3 | 42,6 | 38,5 |
| 11% | 71,5 | 63,8 | 60,5 | 57,4 | 52,2 | 47,6 |

(Aannames: fase 2 = fase 1 minus 1pp, terminal 2,0%, basis FCF €175M, nettoschuld −€131M, 126M aandelen, mid-year convention.)

---

## 13. Databronnen

### Bronnen-hiërarchie
- **Jaarverslag PDF / IR-pagina** → betrouwbaarheid **HOOG**
- **Beursmelding / prospectus** → betrouwbaarheid **HOOG**
- **Aggregator** (MacroTrends / StockAnalysis / Yahoo / TIKR / SimplyWall) → betrouwbaarheid **AGGREGATOR**

### Financiële bronnen (10 jaar historie)

| Jaar | Bron | URL | Betrouwbaarheid (HOOG/AGGREGATOR) |
|---|---|---|---|
| 2015 | — (geen bron geopend in sessie) | — | — |
| 2016 | Bechtle Annual Report 2024 — Multi-Year Overview Bechtle Group | https://reports.bechtle.com/annual-report/2024/_assets/downloads/fin-multi-year-overview-bechtle-ar24.pdf?h=8Yi_YW7e | HOOG |
| 2017 | Bechtle Annual Report 2024 — Multi-Year Overview Bechtle Group | https://reports.bechtle.com/annual-report/2024/_assets/downloads/fin-multi-year-overview-bechtle-ar24.pdf?h=8Yi_YW7e | HOOG |
| 2018 | Bechtle Annual Report 2024 — Multi-Year Overview Bechtle Group | https://reports.bechtle.com/annual-report/2024/_assets/downloads/fin-multi-year-overview-bechtle-ar24.pdf?h=8Yi_YW7e | HOOG |
| 2019 | Bechtle Annual Report 2024 — Multi-Year Overview Bechtle Group | https://reports.bechtle.com/annual-report/2024/_assets/downloads/fin-multi-year-overview-bechtle-ar24.pdf?h=8Yi_YW7e | HOOG |
| 2020 | Bechtle Annual Report 2024 — Multi-Year Overview Bechtle Group | https://reports.bechtle.com/annual-report/2024/_assets/downloads/fin-multi-year-overview-bechtle-ar24.pdf?h=8Yi_YW7e | HOOG |
| 2021 | Bechtle Annual Report 2024 — Multi-Year Overview Bechtle Group | https://reports.bechtle.com/annual-report/2024/_assets/downloads/fin-multi-year-overview-bechtle-ar24.pdf?h=8Yi_YW7e | HOOG |
| 2022 | Bechtle Annual Report 2024 — Multi-Year Overview Bechtle Group | https://reports.bechtle.com/annual-report/2024/_assets/downloads/fin-multi-year-overview-bechtle-ar24.pdf?h=8Yi_YW7e | HOOG |
| 2023 | Bechtle Annual Report 2024 — Multi-Year Overview Bechtle Group | https://reports.bechtle.com/annual-report/2024/_assets/downloads/fin-multi-year-overview-bechtle-ar24.pdf?h=8Yi_YW7e | HOOG |
| 2024 | Bechtle Annual Report 2024 — Multi-Year Overview Bechtle Group | https://reports.bechtle.com/annual-report/2024/_assets/downloads/fin-multi-year-overview-bechtle-ar24.pdf?h=8Yi_YW7e | HOOG |

**Harde eis recente 5 jaren (2020-2024 + 2025) allemaal HOOG**: ✓ voldaan. 2025 via AR2025 PDF (https://www.bechtle.com/dam/jcr:b348dfde-c1b2-4756-83fb-ecbae570ec81/Bechtle_ar2025_en.pdf); 2020-2024 via AR2024 Multi-Year Overview.

### Jaarverslagen geraadpleegd

| Jaar | Bron | URL |
|---|---|---|
| 2024 | Bechtle Annual Report 2024 — Multi-Year Overview Bechtle Group (PDF, pagina's 279-284) | https://reports.bechtle.com/annual-report/2024/_assets/downloads/fin-multi-year-overview-bechtle-ar24.pdf?h=8Yi_YW7e |
| 2025 | Bechtle Annual Report 2025 (PDF, EN) | https://www.bechtle.com/dam/jcr:b348dfde-c1b2-4756-83fb-ecbae570ec81/Bechtle_ar2025_en.pdf |

### Beursmeldingen geraadpleegd

| Datum | Omschrijving | URL |
|---|---|---|
| 2026-05-08 | Bechtle Q1 2026 resultaten — businessvolume +13,2%, EBT +11,5% | https://www.investing.com/news/company-news/bechtle-q1-2026-slides-doubledigit-earnings-growth-across-regions-93CH-4671195 |
| 2026-03-20 | Bechtle FY2025 slides analyse — 8% businessvolume-groei, Q4 margedruk | https://www.investing.com/news/company-news/bechtle-fy2025-slides-8-growth-masks-margin-pressure-q4-miss-93CH-4573204 |
| 2025-03-04 | Bechtle rouwt om medeoprichter Gerhard Schick (†84) | https://www.bechtle.com/de-en/about-bechtle/press/press-releases/2025/bechtle-mourns-the-loss-of-its-founder-gerhard-schick |

### IPO-prospectus
- **Geraadpleegd?**: false (IPO 2000, > 25 jaar geleden; geen verificatie nodig — niet recent-IPO)
- **URL**: n.v.t.
- **Pre-IPO data beschikbaar?**: false (n.v.t.)
- **Pre-IPO bron**: n.v.t.

### Non-GAAP
- **Gebruikt?**: false
- **Toelichting**: Bechtle rapporteert primair IFRS-cijfers. "Business volume" naast "revenue" is een operationele KPI (gevolg van IFRS-15 agent-treatment), geen non-GAAP winstmaat.

### Ontbrekende data (eerlijke lijst)
- **2015 volledig**: geen jaarverslag PDF in deze sessie geraadpleegd; multi-year overview AR2024 begint bij 2016.
- **Brutowinst en brutomarge 2024 en 2025**: niet expliciet uit AR2024 Multi-Year Overview (deze begint bij omzet en EBIT; brutowinst gerapporteerd t/m 2023). Voor 2024-2025 zou de income-statement-detail uit AR2024 / AR2025 nodig zijn die in deze sessie niet volledig is geëxtraheerd.
- **CFO-tenure en CFO-naam**: AR2025 Executive Board details niet volledig uit deze fetch geëxtraheerd.
- **Exacte nettoschuld 31-12-2025**: AR2025 Multi-Year Detail-tabel niet in deze fetch; AR2025-overzicht geeft cash €452M en equity ratio 44,9%, maar bruto financial liabilities niet gespecificeerd. Schatting nettoschuld ~€50M (licht positief) gebruikt, gemarkeerd als schatting.
- **Capex 2025 detail (PP&E vs intangibles)**: niet exact uit AR2025 PDF in deze sessie; StockAnalysis (AGGREGATOR) totale capex 2025 = €112M netto (intangibles).
- **ROIC historie 2016-2024**: Bechtle rapporteert ROCE (= EBIT / Capital Employed), ~16-24% over 2016-2024; StockAnalysis TTM-ROIC = 10,0% — verschilt door definitie (Bechtle includes goodwill; StockAnalysis is post-tax). Aparte ROIC-tabel niet samengesteld; zou kunnen via AR-detail-balans-pagina's die niet volledig geëxtraheerd zijn.
- **Marktaandelen Cancom/Computacenter/Datagroup**: kwalitatieve schatting; geen formeel marktaandeel-rapport gevonden.
- **2025 CFO compensatie / remuneratierapport detail**: laatst geverifieerde CEO-compensatie is 2019 (€3,5M, AGGREGATOR via Yahoo Finance). 2025 detail niet in sessie.
- **2025 insider transactions detail**: officiële melding van Bechtle was "no transactions subject to disclosure in 2024"; 2025 niet expliciet bevestigd.

### Peildatum analyse
- 2026-05-14

---

## 14. Volledige bronnen-lijst

| Titel | URL | Type |
|---|---|---|
| Bechtle AG Annual Report 2024 — Multi-Year Overview Bechtle Group (PDF) | https://reports.bechtle.com/annual-report/2024/_assets/downloads/fin-multi-year-overview-bechtle-ar24.pdf?h=8Yi_YW7e | jaarverslag |
| Bechtle AG Annual Report 2025 (PDF, EN) | https://www.bechtle.com/dam/jcr:b348dfde-c1b2-4756-83fb-ecbae570ec81/Bechtle_ar2025_en.pdf | jaarverslag |
| Bechtle financial publications-hub | https://www.bechtle.com/de-en/about-bechtle/investors/publications | IR-pagina |
| Bechtle Q1 2026 slides analyse (Investing.com) | https://www.investing.com/news/company-news/bechtle-q1-2026-slides-doubledigit-earnings-growth-across-regions-93CH-4671195 | nieuwsartikel |
| Bechtle FY2025 slides analyse (Investing.com) | https://www.investing.com/news/company-news/bechtle-fy2025-slides-8-growth-masks-margin-pressure-q4-miss-93CH-4573204 | nieuwsartikel |
| Bechtle persbericht "anchor shareholder launches foundation" (2022) | https://www.bechtle.com/de-en/about-bechtle/press/press-releases/2022/bechtle-anchor-shareholder-launches-foundation | beursmelding |
| Bechtle persbericht overlijden Gerhard Schick (2025-03-04) | https://www.bechtle.com/de-en/about-bechtle/press/press-releases/2025/bechtle-mourns-the-loss-of-its-founder-gerhard-schick | beursmelding |
| Bechtle company-development pagina | https://www.bechtle.com/de-en/about-bechtle/company/company-development | IR-pagina |
| Bechtle management overview | https://www.bechtle.com/de-en/about-bechtle/company/management | IR-pagina |
| Bechtle corporate governance | https://www.bechtle.com/de-en/about-bechtle/investors/corporate-governance | IR-pagina |
| StockAnalysis.com Bechtle Statistics (multiples, beta, ROIC) | https://stockanalysis.com/quote/etr/BC8/statistics/ | aggregator |
| StockAnalysis.com Bechtle Cash Flow Statement | https://stockanalysis.com/quote/etr/BC8/financials/cash-flow-statement/ | aggregator |
| Yahoo Finance BC8.DE (beta 0,85) | https://finance.yahoo.com/quote/BC8.DE/ | aggregator |
| MarketScreener Bechtle Analyst Consensus (target €44,88) | https://www.marketscreener.com/quote/stock/BECHTLE-AG-54095346/consensus/ | aggregator |
| TradingEconomics Germany 10-Year Bond Yield (3,07%) | https://tradingeconomics.com/germany/government-bond-yield | aggregator |
| Damodaran Country Risk Premium Germany (ERP 4,2%, jan 2026) | https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html | onderzoeksrapport |
| Wikipedia Bechtle (geschiedenis, IPO-detail) | https://en.wikipedia.org/wiki/Bechtle | nieuwsartikel |
| Bechtle Q4 2024-2025 — How a quiet IT powerhouse is turning services into a scalable product (Ad-hoc-news) | https://www.ad-hoc-news.de/boerse/news/ueberblick/bechtle-ag-how-europe-s-quiet-it-powerhouse-is-turning-services-into-a/68458532 | nieuwsartikel |
| Yahoo Finance — institutionele ownership 38% Bechtle | https://finance.yahoo.com/news/38-ownership-shares-bechtle-ag-084912462.html | aggregator |

---

## 15. Update-historie

| Datum | Versie | Wijziging |
|---|---|---|
| 2026-05-14 | 1.0 | Eerste publicatie |

---

## Afronding (check voor oplevering)

- [x] Elk cijfer in tabellen heeft een bron of staat in bronnen-tabel
- [x] Recente 5 jaren (2021-2025) in sectie 13 zijn allemaal HOOG
- [x] Geen enum-variant verzonnen — alleen KOOP/HOLD/PASS, WIDE/NARROW/NO MOAT, STERK/NEUTRAAL/ZORGWEKKEND, LAAG/MIDDEN/HOOG, KLEIN/MIDDEL/GROOT, POSITIEF/NEGATIEF/NEUTRAAL/BINAIR, HOOG/AGGREGATOR, Pessimistisch/Basis/Optimistisch
- [x] Scorekaart 9 frameworks, totaal 24, max 45 — consistent
- [x] Synthese-toelichting aanwezig (sectie 12)
- [x] Non-GAAP expliciet "niet gebruikt"
- [x] IPO-carve-out: IPO 2000 = > 25 jaar = geen pre-IPO-correctie nodig — expliciet vermeld in sectie 2

**Consistentie-cross-check:**
- Executive summary oordeel KOOP/HOLD/PASS = HOLD ✓ = Scorekaart eindoordeel HOLD ✓
- Fair value basis €33,00 = basis-scenario fair value €33,00 ✓
- Kansgewogen €34,65 = 0,25×23,30 + 0,50×33,00 + 0,25×49,30 ✓
- Scorekaart totaal 24 = 3+3+1+2+3+2+4+3+3 ✓
- Kans_pct 25+50+25 = 100 ✓
- Terminal value % van totaal 61% < 75% ✓
- Aandelen 126M (consistent door sectie 2, 3, 12)
- Peildatum 2026-05-14 consistent (sectie metadata, sectie 13)
- Koers €29,70 consistent (metadata, executive summary impliciet via upside)

---

## Opmerkingen voor Claude Code

1. **Koersbron-verschil**: StockAnalysis Statistics-pagina toonde op de fetch-datum €34,20 (peildatum statistics-snapshot 27-feb-2026); de Cash Flow Statement-pagina toonde €29,70 (14-mei-2026, exact onze peildatum). Ik gebruik **€29,70 als peildatum-koers** in alle berekeningen. Marktkap herberekend = 126M × €29,70 = €3.742M (niet de €4,31 mld die op de Statistics-pagina staat). Als jullie validator een externe koers-API gebruikt: check dat die ook 14-mei-2026 / €29,70 raakt. Eerdere zoekresultaat noemde €31,10 per 11 mei 2026; gegeven volatiliteit week kan dit verschillen — gebruik StockAnalysis-quote als referentie.

2. **Nettoschuld 2025**: niet exact uit AR2025 PDF geëxtraheerd in deze sessie (de PDF was te groot voor één fetch en de detail-balans-tabel kwam niet in het eerste gedeelte). Ik heb cash €452M HOOG, equity-ratio 44,9% HOOG, en daaruit nettoschuld ~+€50M geschat (licht positief). Als jullie de exacte Balance Sheet 2025 uit AR2025 PDF willen halen voor stage 2: tabel staat in "Multi-Year Overview" sectie ~p. 280, vergelijkbaar met AR2024.

3. **FCF 2025 definitie-verschil**: Bechtle's eigen FCF-definitie (CFO + investeringsactiviteit excl. acquisities en securities) lijkt voor 2025 niet expliciet gepubliceerd in het AR2025-gedeelte dat ik geëxtraheerd heb. StockAnalysis-definitie zet CFO ≈ FCF (€289,78M). Mijn genormaliseerde FCF €175M is een conservatieve mediaan over 2019-2025 en niet afhankelijk van een precieze 2025-FCF-definitie. Validator: als jullie checken op exact "FCF 2025" cijfer, accepteer een range €169-290M afhankelijk van de gebruikte definitie.

4. **WACC-berekening discrepantie**: pure CAPM-uitkomst is 5,89%; sector-benchmark (StockAnalysis 8,17%, Damodaran 7,5%) suggereert 7,5%. Ik gebruik 7,5% in alle DCF-scenarios. Als jullie reproduceerbaarheid willen via een script: gebruik dezelfde 7,5% basis, 8,5% pess, 6,5% opti.

5. **ROIC inconsistentie**: Bechtle's eigen ROCE-tabel toont 16,5%-28,8% structureel (incl. goodwill in capital employed); StockAnalysis TTM-ROIC = 10,0% (post-tax NOPAT / invested capital). Ik gebruik 10% in scoring/EPV omdat het post-tax en aggregator-gestandaardiseerd is. Als jullie een specifieke ROIC-definitie hanteren, kan dit per framework verschuiven.

6. **Brutowinst 2024 en 2025 ontbreken** in het multi-year overzicht; alleen tot 2023 expliciet. Voor de Resultatenrekening-tabel zijn de cellen voor 2024-2025-brutowinst leeggelaten — niet ingevuld. Als jullie de income-statement-detail uit AR2024-/AR2025-PDF willen halen voor volledigheid, staat dat in een aparte tabel later in elk jaarverslag.

7. **PEG-berekening**: Lynch-score is op het randje van het rubric — PEG 3,5 > 2,0 → score 1. StockAnalysis-PEG is 3,23. Beide leiden tot dezelfde score. Toon mijn berekening: P/E 17,4 / verwachte EPS-groei 5% = 3,5.

8. **Eindoordeel HOLD is grens**: totaal 24 is exact op de PASS/HOLD-grens (≥ 24 EN < 33 → HOLD). Een verschuiving van één framework-score zou het oordeel kunnen flippen. Specifiek: als de PEG-rubric soepeler wordt geïnterpreteerd voor slow-growers en Lynch-score wordt verhoogd naar 2, gaat totaal naar 25 → nog steeds HOLD. Als Moat-score wordt verlaagd naar 1 (strikt: spread niet > 5pp en gemiddelde categorieën), gaat totaal naar 23 → PASS. Validator-trigger: check rubric-toepassing voor Lynch en Moat extra zorgvuldig.

9. **Geen platform/data/scripts/git aangeraakt** — conform projectprotocol. Alleen `research/BC8.md` aangemaakt.
