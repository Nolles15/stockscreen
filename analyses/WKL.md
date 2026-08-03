# Research: WKL — Wolters Kluwer N.V.

> **Stage 1 output van cowork.** Claude Code neemt het over voor JSON-injectie, validator en deploy.
> Methode: `research/METHODE.md`. Structuur: `research/TEMPLATE.md`.
> Deze versie vervangt de analyse van 2026-04-28 volledig: peildatum geactualiseerd (koers €57,56),
> financiële historie nu 2015-2025 volledig HOOG-gedekt via officiële full-year persberichten.

---

## Bronnen-inventaris (Stap 0.5)

```
Jaar 2025 — HOOG
  Bron: Wolters Kluwer 2025 Full-Year Report persbericht (25-02-2026), GlobeNewswire
  URL:  https://www.globenewswire.com/news-release/2026/02/25/3244280/0/en/Wolters-Kluwer-2025-Full-Year-Report.html
  Daadwerkelijk geopend: ja (volledig gelezen)
  Cijfers overgenomen: omzet 6.125, organische groei +6%, adjusted operating profit 1.687
                       (marge 27,5%), IFRS operating profit 1.735, IFRS nettowinst 1.308,
                       adjusted net profit 1.225, diluted adjusted EPS 5,29, diluted IFRS EPS 5,64,
                       gewogen gem. verwaterde aandelen 231,8 mln, ultimo 226,2 mln,
                       CFO 1.668, capex 303, adjusted FCF 1.348, dividend 2,52,
                       buyback 1.100 (8,6 mln à €128,45), nettoschuld 4.024, bruto schuld 4.972,
                       net-debt/EBITDA 2,0x, eigen vermogen 798, totale activa 9.584,
                       goodwill 4.787, ROIC 18,0%, divisie- en regiosplitsing, guidance 2026,
                       adjusted EBITDA 2.007, IFRS EBITDA 2.212, adj. net financing costs 86,
                       benchmark tax rate 23,6%, M&A 2025 (RASi 386, Brightflag 436, Libra ≤90,
                       FRR-divestment netto 399, boekwinst 232)
  Cijfers NIET overgenomen: SBC apart (niet in persbericht), current ratio (geen volledige
                       balansdetaillering in persbericht), omzet per regio in € absoluut

Jaar 2024 — HOOG
  Bron: Wolters Kluwer 2024 Full-Year Report persbericht (26-02-2025), GlobeNewswire
  URL:  https://www.globenewswire.com/news-release/2025/02/26/3032585/0/en/Wolters-Kluwer-2024-Full-Year-Report.html
  Daadwerkelijk geopend: ja (volledig gelezen)
  Cijfers overgenomen: omzet 5.916 (+6% org), adj OP 1.600 (marge 27,1%), IFRS OP 1.441,
                       IFRS nettowinst 1.079, adj net profit 1.185, diluted adj EPS 4,97,
                       diluted IFRS EPS 4,52, verwaterde aandelen 238,4 mln, CFO 1.654,
                       capex 313, adj FCF 1.276, dividend 2,33 (totaal 545), buyback 1.000
                       (6,7 mln à €149,23), nettoschuld 3.134, bruto schuld 4.090, ND/EBITDA 1,6x,
                       eigen vermogen 1.545, totale activa 9.498, goodwill 4.710, ROIC 18,1%,
                       adj EBITDA 1.930, IFRS EBITDA 1.920, divisies, regio (NA 64%/EU 28%/RoW 8%)
  Cijfers NIET overgenomen: SBC apart, current ratio

Jaar 2023 — HOOG
  Bron: vergelijkende kolom in Wolters Kluwer 2024 Full-Year Report (zelfde URL als 2024)
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: omzet 5.584 (+6% org), adj OP 1.476 (26,4%), IFRS OP 1.323,
                       IFRS nettowinst 1.007, adj net profit 1.119, diluted adj EPS 4,55,
                       diluted IFRS EPS 4,09, verwaterde aandelen 246,0 mln, CFO 1.545,
                       capex 323, adj FCF 1.164, dividend 2,08 (totaal 500), buyback 1.000
                       (8,7 mln à €114,44), nettoschuld 2.612, bruto schuld 3.749, ND/EBITDA 1,5x,
                       eigen vermogen 1.749, totale activa 9.094, goodwill 4.322, ROIC 16,8%,
                       adj EBITDA 1.775, IFRS EBITDA 1.768
  Cijfers NIET overgenomen: SBC apart, current ratio

Jaren 2021 en 2022 — HOOG
  Bron: Wolters Kluwer 2022 Full-Year Report persbericht (22-02-2023), GlobeNewswire
  URL:  https://www.globenewswire.com/en/news-release/2023/02/22/2612844/0/en/Wolters-Kluwer-2022-Full-Year-Report.html
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: 2022: omzet 5.453 (+6% org), adj OP 1.424 (26,1%), IFRS OP 1.333,
                       IFRS NW 1.027, adj NP 1.059, adj EPS 4,14, IFRS EPS 4,01, aandelen 255,8,
                       CFO 1.582, capex 295, adj FCF 1.220, DPS 1,81, buyback 1.000
                       (10,1 mln à €98,75), nettoschuld 2.253, ND/EBITDA 1,3x, ROIC 15,5%.
                       2021: omzet 4.771 (+6% org), adj OP 1.205 (25,3%), IFRS OP 1.012,
                       IFRS NW 728, adj NP 885, adj EPS 3,38, IFRS EPS 2,78, aandelen 261,8,
                       CFO 1.292, capex 239, adj FCF 1.010, buyback 410, nettoschuld 2.131,
                       ND/EBITDA 1,4x, ROIC 13,7%
  Cijfers NIET overgenomen: DPS 2021 (zie aggregator-regel), balansdetail (EV/activa in EUR)

Jaren 2019 en 2020 — HOOG
  Bron: Wolters Kluwer 2020 Full-Year Report persbericht (24-02-2021), GlobeNewswire
  URL:  https://www.globenewswire.com/news-release/2021/02/24/2181083/0/en/Wolters-Kluwer-2020-Full-Year-Report.html
  Aanvullend: Wolters Kluwer 2019 Full-Year Report (26-02-2020),
  URL:  https://www.globenewswire.com/news-release/2020/02/26/1990636/0/en/Wolters-Kluwer-2019-Full-Year-Report.html
  Daadwerkelijk geopend: ja (beide)
  Cijfers overgenomen: 2020: omzet 4.603 (+2% org), adj OP 1.124 (24,4%), IFRS OP 972,
                       IFRS NW 721, adj NP 835, adj EPS 3,13, IFRS EPS 2,70, aandelen 266,6,
                       CFO 1.197, capex 231, adj FCF 907, DPS 1,36, buyback 350 (5,1 mln à €68,41),
                       nettoschuld 2.383, ND/EBITDA 1,7x, ROIC 12,3%.
                       2019: omzet 4.612 (+4% org), adj OP 1.089 (23,6%), IFRS OP 908,
                       IFRS NW 669, adj NP 790, adj EPS 2,90, IFRS EPS 2,46, aandelen 272,2,
                       CFO 1.102, capex 226, adj FCF 807, DPS 1,18, buyback 350 (5,5 mln à €63,80),
                       nettoschuld 2.199, ND/EBITDA 1,6x, ROIC 11,8%
  Cijfers NIET overgenomen: eigen vermogen in EUR (niet in persbericht opgenomen)

Jaren 2017 en 2018 — HOOG
  Bron: Wolters Kluwer 2018 Full-Year Report persbericht (20-02-2019), GlobeNewswire
  URL:  https://www.globenewswire.com/news-release/2019/02/20/1738056/0/en/Wolters-Kluwer-2018-Full-Year-Report.html
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: 2018: omzet 4.260 (+4% org), adj OP 980 (23,0%), IFRS OP 961,
                       IFRS NW 657, adj NP 683, adj EPS 2,45, IFRS EPS 2,35, aandelen 278,8,
                       CFO 934, capex 214, adj FCF 762, DPS 0,98, buyback 550 (11,5 mln),
                       nettoschuld 1.994 (pre-IFRS 16; IFRS16-restated 2.249, ND/EBITDA 1,8x),
                       ROIC 10,9% (restated 10,6%).
                       2017 (IFRS 15-restated): omzet 4.368 (+3% org, IAS 18-basis), adj OP 970
                       (22,2%), IFRS OP 830, IFRS NW 637, adj NP 639, adj EPS 2,22, IFRS EPS 2,21,
                       aandelen 287,7, CFO 940, capex 210, adj FCF 746, DPS 0,85, buyback 302,
                       nettoschuld 2.069, ND/EBITDA 1,8x, ROIC 10,0%
  Cijfers NIET overgenomen: eigen vermogen in EUR

Jaren 2015 en 2016 — HOOG
  Bron: Wolters Kluwer 2016 Full-Year Report persbericht (22-02-2017), GlobeNewswire
  URL:  https://www.globenewswire.com/news-release/2017/02/22/926171/0/en/Wolters-Kluwer-2016-Full-Year-Report.html
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: 2016: omzet 4.297 (+3% org), adj OP 950 (22,1%), IFRS OP 766,
                       IFRS NW 490, adj NP 618, adj EPS 2,10, IFRS EPS 1,66, CFO 927, capex 224,
                       adj FCF 708, DPS 0,79, buyback 200 (5,8 mln), nettoschuld 1.927,
                       ND/EBITDA 1,7x, ROIC 9,8%.
                       2015: omzet 4.208 (+3% org), adj OP 902 (21,4%), IFRS OP 667, IFRS NW 423,
                       adj NP 583, adj EPS 1,96, IFRS EPS 1,42, CFO 843, adj FCF 647, DPS 0,75,
                       nettoschuld 1.788, ROIC 9,3%
  Cijfers NIET overgenomen: verwaterde aandelen 2015/2016 (persbericht noemt alleen
                       procentuele daling), capex 2015 in € (alleen "4,5% van omzet"),
                       buyback-bedrag 2015 (alleen anti-dilutieprogramma)

Q1/H1 2026 — HOOG (context, geen jaartabel)
  Bron: Wolters Kluwer First-Quarter 2026 Trading Update (06-05-2026), GlobeNewswire
  URL:  https://www.globenewswire.com/news-release/2026/05/06/3288504/0/en/wolters-kluwer-first-quarter-2026-trading-update.html
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: organische groei Q1 +5% (recurring 85% van omzet, +7%; cloud 23%, +14%),
                       adj OP +11% CC, nettoschuld 31-03-2026 3.788 (ND/EBITDA 1,9x),
                       aandelen uitstaand 224,5 mln, buyback 2026 tot 500 (164 gedaan per 4 mei,
                       2,3 mln à €71,71), guidance herbevestigd, kalender (HY 5-8-2026, 9M 4-11-2026)

AGGREGATOR-bronnen (aanvullend, expliciet gelabeld):
  - StockAnalysis.com (https://stockanalysis.com/quote/ams/WKL/statistics/ en /financials/):
    koers/marktkap-snapshot, aandelen 223,88 mln, institutioneel eigendom 57,2%, insiders 0,25%,
    beta 5j 0,15, EBITDA 2021 1.485 / 2022 1.799 (EUR), eigen vermogen 2021 2.417 / 2022 2.310
    (EUR), DPS 2021 1,57, consensus 3j omzet +4,4%/EPS +9,0%, koersdoel ~€103-104
  - MacroTrends WTKWY (USD, ADR — alleen ter indicatie eigen vermogen/activa 2015-2020, NIET
    in EUR-tabellen opgenomen): https://www.macrotrends.net/stocks/charts/WTKWY/wolters-kluwer/
  - Trading Economics (NL 10-jaars rente 3,03%, 02-07-2026):
    https://tradingeconomics.com/netherlands/government-bond-yield
  - Damodaran (ERP 4,23% per 01-01-2026; Europa-betas jan 2026 — betaEurope.xls daadwerkelijk
    geopend en uitgelezen: Information Services levered beta 0,82, unlevered cash-corrected 0,44):
    https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datacurrent.html
  - InsiderScreener (insider-transacties, met gemarkeerde parsingfouten):
    https://www.insiderscreener.com/en/company/wolters-kluwer-nv
  - Morningstar (moat-downgrade naar Narrow):
    https://www.morningstar.com/stocks/downgrading-wolters-kluwer-thomson-reuters-moats-narrow-ai-disruption-potential
  - TipRanks / MarketScreener (analistenconsensus): https://www.tipranks.com/stocks/nl:wkl/forecast

Zelf-check uitgevoerd: elk numeriek cel in de tabellen hieronder is traceerbaar naar een van
bovenstaande geopende bronnen. Cellen zonder bron staan op "—" en zijn genoteerd in sectie 13
onder ontbrekende data.
```

**Bronnen-inventaris-conclusie:** alle elf jaren 2015-2025 zijn gedekt met HOOG-bronnen (officiële full-year persberichten met IFRS-cijfers, elk daadwerkelijk geopend). Aggregators zijn alleen gebruikt voor aanvullende context (marktdata, consensus, beta, enkele balansdetails 2021-2022) en zijn overal expliciet gelabeld. Daarmee is de dekking wezenlijk beter dan in de vorige versie van deze analyse (die 2015-2019 leeg liet). Resterende gaten: SBC per jaar, current/quick ratio, eigen vermogen 2015-2020 in EUR — genoteerd in sectie 13.

---

## Metadata
- **Ticker (bare):** WKL
- **Yahoo symbol:** WKL.AS
- **Exchange:** AEX (Euronext Amsterdam)
- **Sector (GICS-achtig):** Industrie / Professionele diensten (Information Services)
- **Industrie:** Professionele informatie, software en services (health, tax, legal, finance, compliance)
- **Land:** Nederland (Alphen aan den Rijn)
- **Peildatum analyse:** 2026-07-02
- **Koers op peildatum:** 57,56
- **Valuta:** EUR
- **Marktkapitalisatie:** EUR 12,9 mld
- **Marktkap in mln (lokale valuta):** 12922
- **Free float pct:** ~99% (insiders 0,25%; geen controlerende aandeelhouder)
- **Indexlidmaatschap:** AEX, Stoxx Europe 600
- **Domein:** wolterskluwer.com

---

## 1. Executive summary

- **Kernthese:** Wolters Kluwer is een wereldwijde leverancier van professionele informatie, software en diensten voor artsen, accountants, juristen, banken en compliance-afdelingen — domeinen waar regelgeving complex is en fouten duur zijn. Ruim 83% van de omzet is terugkerend (abonnementen en cloudsoftware), de organische groei ligt al vijf jaar stabiel op 6% en de aangepaste operationele marge steeg van 21,4% (2015) naar 27,5% (2025), met een ROIC van 18%. Het aandeel is in twaalf maanden echter met bijna 60% gedaald — niet door slechte cijfers (de guidance werd in mei 2026 juist herbevestigd), maar door de vrees dat generatieve AI juridisch en fiscaal onderzoek commoditiseert; de hele informatiesector (RELX, Thomson Reuters) is mee gederate en Morningstar verlaagde de moat-rating naar Narrow. Daardoor noteert een structurele compounder nu tegen circa 11 keer de aangepaste winst en een vrije-kasstroomrendement van ruim 10%, terwijl de markt in een omgekeerde DCF feitelijk een eeuwige krimp van circa 2% per jaar inprijst. De centrale vraag is dus niet óf WKL goedkoop is op de huidige cijfers, maar of die cijfers de komende tien jaar door AI worden uitgehold. Wie gelooft dat geverifieerde content plus workflow-integratie verdedigbaar blijft, kijkt naar een uitzonderlijke discrepantie tussen prijs en verdienkracht; wie gelooft in structurele disruptie ziet een value trap in wording.
- **Oordeel** (enum): **HOLD**
- **Fair value basis** (lokale valuta): 170
- **Fair value kansgewogen**: 152
- **EPV per aandeel** (Earnings Power Value, zonder groeipremie): 63
- **Upside pct**: 195,3
- **Fair value scenarios**:

| Scenario | Fair value | Upside % | FCF groei % | WACC % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 58 | 0,8 | -2,0 | 7,37 | 30 |
| Basis | 170 | 195,3 | 5,0 | 6,37 | 50 |
| Optimistisch | 250 | 334,3 | 7,5 | 5,87 | 20 |

- **Reverse-DCF impliciete groei pct**: -2 (bij WACC 6,37% en mid-year convention moet de FCF tien jaar lang én in de terminal met ~2% per jaar KRIMPEN om de huidige koers van €57,56 te rechtvaardigen; historische FCF-CAGR was +7,6%)
- **Grootste kans:** De markt prijst permanente AI-disruptie in terwijl de operationele cijfers (organisch +5-6%, marge-expansie, herbevestigde guidance) daar vooralsnog niets van laten zien — elke bevestiging van veerkracht kan een forse herwaardering triggeren.
- **Grootste risico:** Generatieve AI commoditiseert juridische, fiscale en medische content en workflow-tools sneller dan WKL zijn eigen AI-producten kan monetariseren, waardoor groei en pricing power structureel eroderen en het pessimistische scenario werkelijkheid wordt.

---

## 2. Bedrijfsprofiel

- **Beschrijving:** Wolters Kluwer levert vakinformatie, software en diensten aan professionals die geen fouten mogen maken: artsen en verpleegkundigen (klinische beslisondersteuning via UpToDate), accountants en fiscalisten (CCH Axcess, TeamMate), juristen en bedrijfsjuristen (Kluwer/Legisway, Brightflag), banken en verzekeraars (OneSumX, Wiz-producten voor compliance) en corporate-afdelingen (CT Corporation voor entity management, Enablon en TeamMate voor risk en ESG). Het bedrijf zit aan het eind van de waardeketen van kennis: het combineert geredigeerde, geverifieerde domeincontent met workflowsoftware die diep in de dagelijkse processen van de klant is geïntegreerd. Het probleem dat WKL oplost is regulatoire en professionele complexiteit: wetten, jurisprudentie, medische literatuur en boekhoudregels veranderen continu, en professionals hebben gevalideerde, actuele en aansprakelijkheidsbestendige antwoorden nodig. De omzet komt voor 83% uit terugkerende bronnen — abonnementen, cloudsoftware (21% van de omzet in 2025, +15% organisch) en onderhoudscontracten — aangevuld met transactionele omzet en een krimpend printsegment. Het unieke ten opzichte van generieke software- of contentaanbieders is de combinatie van beide: content zonder software is vervangbaar, software zonder gezaghebbende content ook, maar de integratie van de twee in gecertificeerde workflows is dat veel minder.
- **Geschiedenis:** De wortels gaan terug tot 1836, toen Jan-Berend Wolters in Groningen een schoolboekenuitgeverij begon; Æbele Kluwer startte zijn uitgeverij in 1889 in Deventer. Na een reeks fusies in het Nederlandse uitgeefwezen (Wolters-Noordhoff 1968, ICU 1972, Wolters-Samsom 1983) ontstond Wolters Kluwer in 1987, toen Kluwer fuseerde met Wolters-Samsom om een vijandige overname door Elsevier af te weren; het gecombineerde bedrijf werd genoteerd aan de Amsterdamse beurs. In de jaren negentig groeide het via acquisities uit tot een internationale juridisch-fiscale uitgever (o.a. CCH in de VS, 1995). Begin jaren 2000 volgde onder de in 2003 aangetreden CEO Nancy McKinstry de strategische pivot die het huidige bedrijf definieert: van print naar digitaal en van content naar geïntegreerde software. Print daalde van de meerderheid van de omzet naar een marginaal restsegment; digitale producten en diensten vormen nu ruim 90%. Grote mijlpalen waren de uitbouw van UpToDate in gezondheidszorg, CCH Axcess als cloudplatform voor accountants, en de verkoop van niet-kernactiviteiten (onderwijsuitgeverijen, transportmedia). De financiële crisis van 2008-2009 en de coronapandemie (organische groei zakte in 2020 slechts naar +2%) toonden de defensieve aard van het abonnementsmodel. Recente jaren: versnelling van cloudmigratie, overnames in compliance en legal software (Isabel Group 2024; RASi, Brightflag en Libra in 2025, samen circa €900 mln), verkoop van Finance, Risk & Regulatory Reporting (FRR, 2025, netto €399 mln met €232 mln boekwinst), en in februari 2026 de leiderschapswissel: Stacey Caywood volgde Nancy McKinstry op na 23 jaar CEO-schap. In 2025-2026 halveerde de beurskoers op AI-disruptievrees — de scherpste derating sinds de dotcom-periode, terwijl de operationele resultaten op koers bleven.
- **Bedrijfsmodel:** WKL verdient geld met abonnementen op digitale informatieproducten en software (recurring, 83% van omzet in 2025), aangevuld met transactionele diensten (bijvoorbeeld incorporations bij CT Corporation), licenties on-premise, implementatiediensten en een krimpende printtak. Cloudsoftware is de groeimotor: 21% van de omzet, +15% organisch. Prijszetting is per gebruiker of per module met jaarlijkse indexatie; retentie is hoog omdat producten in de kernprocessen van klanten zitten (een accountantskantoor stapt niet zomaar over van CCH Axcess, een ziekenhuis niet van UpToDate). De vijf divisies (2025-omzet): Health (€1.596 mln, 26%), Tax & Accounting (€1.660 mln, 27%), Financial & Corporate Compliance (€1.239 mln, 20%), Legal & Regulatory (€1.005 mln, 16%) en Corporate Performance & ESG (€625 mln, 10%). De marges verschillen sterk per divisie: T&A en FCC ruim 35%, L&R 18%, CP&ESG 7,5% (nog in investeringsfase).
- **IPO-context:** Wolters Kluwer is sinds de fusie van 1987 genoteerd aan (de voorloper van) Euronext Amsterdam — een beursleeftijd van bijna veertig jaar. Er is dus geen recente IPO, geen lock-up-problematiek en geen pre-IPO-financial-engineering-risico. De kapitaalstructuur is in de afgelopen tien jaar bewust geoptimaliseerd richting aandeelhoudersrendement: het aantal uitstaande aandelen daalde van circa 288 mln (gewogen verwaterd, 2017) naar 224,5 mln (maart 2026) door jaarlijkse inkoopprogramma's, terwijl de nettoschuld beheerst bleef (1,3x-2,0x EBITDA).
- **Klantprofiel:** Vrijwel volledig B2B en B2P (business-to-professional): ziekenhuizen, artsen, accountants- en belastingadvieskantoren (van Big Four tot mkb-kantoren), advocatenkantoren, bedrijfsjuridische afdelingen, banken, verzekeraars en corporates. De klantenbase is zeer gefragmenteerd — honderdduizenden instellingen en kantoren wereldwijd — waardoor geen enkele klant materieel is voor de omzet. Retentie op recurring omzet is structureel hoog (het persbericht 2025 meldt recurring groei van +7% organisch, boven de groepsgroei, wat op lage churn duidt). De klantconcentratie is daarmee een sterk punt: WKL heeft geen top-10-klantenrisico zoals veel enterprise-softwarebedrijven.
- **Oprichtingsjaar**: 1836 (J.B. Wolters; huidige vennootschap door fusie 1987)
- **IPO-datum**: 1987 (notering na fusie Wolters-Samsom/Kluwer; geen klassieke IPO)
- **IPO-koers** (lokale valuta): — (niet verifieerbaar binnen deze run; 1987-fusienotering)
- **Personeel** (FTE): ~21.900 (bron: wolterskluwer.com/en/about-us, "approximately 21,900 employees"; AGGREGATOR-niveau, niet uit jaarverslag-PDF geverifieerd in deze run)
- **Landen actief**: 40+ landen, klanten in 180+ landen (bedrijfsopgave about-us; zelfde voorbehoud)
- **Klantconcentratie**: Geen materiële klantconcentratie; honderdduizenden professionele klanten, grootste klanten zijn Big Four-kantoren en grote ziekenhuisketens maar geen enkele klant is individueel materieel (geen concentratie-disclosure in persberichten — wat op zichzelf al aangeeft dat er geen meldingsplichtige concentratie is).

### Geografische spreiding (omzet)
| Regio | Omzet % | Valuta-exposure |
|---|---|---|
| Noord-Amerika | 63 | USD |
| Europa | 29 | EUR (deels GBP) |
| Azië-Pacific & rest van wereld | 8 | divers |

**Toelichting geografie:** De dominantie van Noord-Amerika (63% van de omzet in 2025) maakt WKL in de kern een dollarbedrijf met een euronotering: een verzwakkende dollar drukt de gerapporteerde cijfers fors. Dat was in Q1 2026 direct zichtbaar — de omzet daalde 3% gerapporteerd bij +4% in constante valuta, puur door een euro/dollar-beweging van 1,05 naar 1,17. De guidance 2026 hanteert €/$ 1,175 voor de marge en 1,13 voor de FCF. Kosten vallen deels in dezelfde valuta (natural hedge via Amerikaanse organisatie), maar de translatie-exposure op winst en waardering blijft de grootste valutafactor. Bron: FY2025- en Q1 2026-persberichten.

### Segmenten
| Naam | Omzet % | Beschrijving |
|---|---|---|
| Tax & Accounting | 27 | Software en content voor accountants en fiscalisten (CCH Axcess, TeamMate); adj. marge 35,2%; kern van de cloudmigratie en het meest winstgevende groeisegment. |
| Health | 26 | Klinische beslisondersteuning (UpToDate), medische literatuur en drug information voor ziekenhuizen en zorgverleners; adj. marge 32,1%; defensief en sticky. |
| Financial & Corporate Compliance | 20 | Compliance-oplossingen voor banken, verzekeraars en corporates (CT Corporation, Wiz, Isabel); adj. marge 35,2%; deels transactioneel (incorporations). |
| Legal & Regulatory | 16 | Juridische informatie en legal-tech (Kluwer, Legisway, Brightflag) voor advocatuur en bedrijfsjuristen; adj. marge 18,2%; meest blootgesteld aan AI-substitutievrees. |
| Corporate Performance & ESG | 10 | EHS/ESG-, audit- en performance-software (Enablon, TeamMate+, CCH Tagetik); adj. marge 7,5%; investeringsfase met hoogste groeipotentieel. |

### Aandeelhouders (top 5)
| Naam | Belang % | Type (oprichter / PE / institutioneel / retail) |
|---|---|---|
| Institutionele beleggers (totaal) | 57,2 | institutioneel (StockAnalysis, AGGREGATOR) |
| Insiders (bestuur en commissarissen) | 0,25 | management |
| — individuele top-5 houders niet verifieerbaar binnen deze run | — | — |

- **Institutioneel eigendomstrend:** Stabiel tot licht dalend in de crash; het institutioneel eigendom van 57,2% (StockAnalysis, juni 2026) is normaal voor een AEX-hoofdfonds zonder controlerende aandeelhouder. Een substantiëlere trendanalyse (13F-achtige reeks) was binnen deze run niet uit een geverifieerde bron beschikbaar; de vorige piek-naar-dal-beweging suggereert dat momentum- en kwaliteitsfondsen zijn uitgestapt terwijl value-beleggers instappen, maar dat is interpretatie, geen gemeten feit.

---

## 3. Financieel — historische data (10 jaar + TTM)

Alle bedragen in EUR mln tenzij anders vermeld. "Adj." = benchmark-cijfers zoals door WKL gerapporteerd (non-IFRS); IFRS-cijfers staan er telkens naast. Bronnen per jaar: zie bronnen-inventaris en sectie 13. WKL rapporteert geen brutowinst-regel; die kolom is daarom weggelaten in plaats van geschat.

### Resultatenrekening

| Jaar | Omzet | Org. groei % | Adj. op. winst | Adj. marge % | IFRS op. winst | IFRS nettowinst | Adj. nettowinst | Adj. EPS (dil.) | IFRS EPS (dil.) | Aandelen mln (gew. dil.) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2015 | 4.208 | +3 | 902 | 21,4 | 667 | 423 | 583 | 1,96 | 1,42 | — |
| 2016 | 4.297 | +3 | 950 | 22,1 | 766 | 490 | 618 | 2,10 | 1,66 | — |
| 2017 | 4.368 | +3 | 970 | 22,2 | 830 | 637 | 639 | 2,22 | 2,21 | 287,7 |
| 2018 | 4.260 | +4 | 980 | 23,0 | 961 | 657 | 683 | 2,45 | 2,35 | 278,8 |
| 2019 | 4.612 | +4 | 1.089 | 23,6 | 908 | 669 | 790 | 2,90 | 2,46 | 272,2 |
| 2020 | 4.603 | +2 | 1.124 | 24,4 | 972 | 721 | 835 | 3,13 | 2,70 | 266,6 |
| 2021 | 4.771 | +6 | 1.205 | 25,3 | 1.012 | 728 | 885 | 3,38 | 2,78 | 261,8 |
| 2022 | 5.453 | +6 | 1.424 | 26,1 | 1.333 | 1.027 | 1.059 | 4,14 | 4,01 | 255,8 |
| 2023 | 5.584 | +6 | 1.476 | 26,4 | 1.323 | 1.007 | 1.119 | 4,55 | 4,09 | 246,0 |
| 2024 | 5.916 | +6 | 1.600 | 27,1 | 1.441 | 1.079 | 1.185 | 4,97 | 4,52 | 238,4 |
| 2025 | 6.125 | +6 | 1.687 | 27,5 | 1.735 | 1.308 | 1.225 | 5,29 | 5,64 | 231,8 |
| TTM/Q1'26 | — (Q1: +5% org, +4% CC) | +5 | — (+11% CC) | — | — | — | — | — | — | 224,5 (ultimo mrt) |

EBITDA waar beschikbaar: 2021: 1.485 (AGGREGATOR) | 2022: 1.799 (AGGREGATOR) | 2023: 1.775 adj. / 1.768 IFRS | 2024: 1.930 adj. / 1.920 IFRS | 2025: 2.007 adj. / 2.212 IFRS (incl. FRR-boekwinst). Kanttekeningen: 2017 IFRS 15-restated; 2018 pre-IFRS 16 (restated: omzet 4.259, adj. OP 986); 2024-marge bevat eenmalige pensioenbate van €27 mln (excl. 26,6%); 2025 IFRS-cijfers bevatten €232 mln boekwinst op FRR-divestment.

- **Toelichting resultaten:** Het patroon over tien jaar is een schoolvoorbeeld van een kwaliteitscompounder in transitie: de omzet groeide bescheiden met 3,8% per jaar (CAGR 2015-2025, deels gedrukt door desinvesteringen en print-krimp), maar de organische groei versnelde van +3% naar een stabiele +6% vanaf 2021, en de aangepaste operationele marge steeg elf jaar op rij, van 21,4% naar 27,5% — ruim 600 basispunten. Die combinatie van bescheiden groei, structurele marge-expansie en agressieve aandeleninkoop (van ~288 mln naar ~225 mln aandelen) leverde een aangepaste EPS-CAGR van 10,4% op. Er zijn geen verliesjaren, geen omzetimplosies en zelfs in coronajaar 2020 bleef de organische groei positief (+2%). De kwaliteit van de groei is bovendien verbeterd: cloudsoftware groeit 14-15% en recurring omzet groeit boven het groepsgemiddelde. De uitschieter in IFRS-cijfers 2025 (nettowinst €1.308 mln) komt door de eenmalige FRR-boekwinst en zegt niets over de onderliggende trend; de aangepaste reeks is de betere maatstaf voor verdienkracht.
- **Omzet-CAGR (2015-2025):** 3,8% (gerapporteerd, incl. FX en portfolio-effecten); organisch de laatste vijf jaar consistent +6%.

### Kasstromen

| Jaar | CFO | Capex | Adj. FCF | FCF/aandeel | FCF-marge % | FCF-groei % | FCF-conversie %* | SBC | Dividend cash | Aandeleninkoop |
|---|---|---|---|---|---|---|---|---|---|---|
| 2015 | 843 | — | 647 | — | 15,4 | — | 111 | — | — | — |
| 2016 | 927 | 224 | 708 | — | 16,5 | +9,4 | 115 | — | 223 | 200 |
| 2017 | 940 | 210 | 746 | 2,59 | 17,1 | +5,4 | 117 | — | — | 302 |
| 2018 | 934 | 214 | 762 | 2,73 | 17,9 | +2,1 | 112 | — | 277 | 550 |
| 2019 | 1.102 | 226 | 807 | 2,96 | 17,5 | +5,9 | 102 | — | 280 | 350 |
| 2020 | 1.197 | 231 | 907 | 3,40 | 19,7 | +12,4 | 109 | — | 334 | 350 |
| 2021 | 1.292 | 239 | 1.010 | 3,86 | 21,2 | +11,4 | 114 | — | 373 | 410 |
| 2022 | 1.582 | 295 | 1.220 | 4,77 | 22,4 | +20,8 | 115 | — | 424 | 1.000 |
| 2023 | 1.545 | 323 | 1.164 | 4,73 | 20,8 | -4,6 | 104 | — | 500 | 1.000 |
| 2024 | 1.654 | 313 | 1.276 | 5,35 | 21,6 | +9,6 | 108 | — | 545 | 1.000 |
| 2025 | 1.668 | 303 | 1.348 | 5,82 | 22,0 | +5,6 | 110 | — | ~570** | 1.100 |

\* FCF-conversie = adjusted FCF / adjusted nettowinst (beide zoals gerapporteerd). \** Dividend 2025: €2,52 per aandeel voorgesteld; cash-uitbetaling deels in 2026 (finaal €1,59 betaald 17 juni 2026). SBC wordt in de persberichten niet apart vermeld — zie sectie 13, ontbrekende data.

- **Toelichting kasstromen:** De vrije kasstroom is over tien jaar meer dan verdubbeld, van €647 mln naar €1.348 mln (CAGR 7,6%), met een FCF-marge die opliep van 15,4% naar 22,0% — het bewijs dat de softwaretransitie niet alleen boekhoudkundige marges maar ook echte cash oplevert. De FCF-conversie ligt structureel boven 100% van de aangepaste nettowinst (102-117%), een kenmerk van het abonnementsmodel: klanten betalen vooruit, waardoor het werkkapitaal negatief is en groei cash genereert in plaats van absorbeert. De enige daling in de reeks (2023, -4,6%) had een timing-oorzaak in werkkapitaal en hogere capex (5,8% van omzet), geen structurele; 2024 en 2025 herstelden direct. Capex is bescheiden (5,0-5,8% van omzet) en verschuift naar productontwikkeling. De kasstroom werd vrijwel volledig teruggegeven: sinds 2022 jaarlijks €1,0-1,1 mld buybacks plus een groeiend dividend. Kanttekening bij de buyback-timing: de inkopen van 2022-2024 gebeurden tegen €98-149 per aandeel, fors boven de huidige koers — achteraf dure inkopen (zie H5).

### Balans-ratio's (10 jaar)

| Jaar | Nettoschuld | Bruto schuld | Nettoschuld/EBITDA | Eigen vermogen | Totale activa | Goodwill | ROIC % (gerapporteerd) |
|---|---|---|---|---|---|---|---|
| 2015 | 1.788 | — | — | — | — | — | 9,3 |
| 2016 | 1.927 | — | 1,7 | — | — | — | 9,8 |
| 2017 | 2.069 | — | 1,8 | — | — | — | 10,0 |
| 2018 | 1.994 (2.249 na IFRS 16) | — | 1,7 (1,8) | — | — | — | 10,9 (10,6) |
| 2019 | 2.199 | — | 1,6 | — | — | — | 11,8 |
| 2020 | 2.383 | — | 1,7 | — | — | — | 12,3 |
| 2021 | 2.131 | — | 1,4 | 2.417 (AGGR.) | 9.028 (AGGR.) | — | 13,7 |
| 2022 | 2.253 | — | 1,3 | 2.310 (AGGR.) | 9.510 (AGGR.) | — | 15,5 |
| 2023 | 2.612 | 3.749 | 1,5 | 1.749 | 9.094 | 4.322 | 16,8 |
| 2024 | 3.134 | 4.090 | 1,6 | 1.545 | 9.498 | 4.710 | 18,1 |
| 2025 | 4.024 | 4.972 | 2,0 | 798 | 9.584 | 4.787 | 18,0 |
| Q1'26 | 3.788 | — | 1,9 | — | — | — | — |

Eigen vermogen en totale activa 2015-2020 zijn in de geopende persberichten niet in EUR opgenomen; MacroTrends heeft alleen USD-waarden voor het ADR (niet in de tabel opgenomen om valutavermenging te voorkomen — zie sectie 13). Current/quick ratio: niet in persberichten gedetailleerd; —.

- **Toelichting balans:** Let bij deze balans op het onderscheid tussen bruto en netto: de bruto schuld steeg in 2025 naar €4.972 mln (mede door twee nieuwe eurobonds van elk €500 mln tegen 3,0% en 3,375%), en de nettoschuld naar €4.024 mln (2,0x EBITDA) — maar per eind maart 2026 was de nettopositie alweer verbeterd naar €3.788 mln (1,9x). De stijging van de leverage is een bewuste keuze: circa €900 mln aan overnames plus €1,1 mld buybacks in één jaar. Zorgwekkender oogt het eigen vermogen, dat door de jarenlange inkopen boven boekwaarde is uitgehold tot €798 mln — boekwaarde per aandeel is daarmee €3,53 en de P/B-ratio een betekenisloze 16x. Dat is geen solvabiliteitsprobleem (rente-dekking: adj. operationele winst €1.687 mln tegenover €86 mln financieringslasten, bijna 20x), maar het maakt boekwaarde-gebaseerde maatstaven zoals het Graham-raamwerk voor dit aandeel structureel onbruikbaar. Goodwill van €4.787 mln (50% van de activa) weerspiegelt de acquisitiestrategie; er zijn in de geopende bronnen geen impairments in de reeks gemeld.

### Kapitaalstructuur huidig
- **Nettoschuld (huidig)**: 3.788 (31-03-2026)
- **Bruto schuld**: 4.972 (31-12-2025)
- **Cash & equivalents**: 932 (31-12-2025)
- **Lease-verplichtingen (IFRS-16)**: inbegrepen in schuld sinds 2019; aparte specificatie niet in persberichten
- **Gemiddelde rente %**: ~1,7% effectief op bestaande schuld (adj. netto financieringslasten €86 mln / bruto schuld €4.972 mln); nieuwe emissies 3,0-3,375%
- **Rente-dekking (adj. op. winst/financieringslasten)**: ~19,6x

### Non-GAAP / aanpassingen
- **Gebruikt?**: true
- **Welke aanpassingen**: WKL rapporteert "benchmark"-cijfers (adjusted operating profit, adjusted net profit, adjusted EPS, adjusted FCF) naast IFRS. Uitgesloten worden m.n. amortisatie van geacquireerde immateriële activa, boekwinsten/verliezen op divestments (zoals de €232 mln FRR-winst in 2025) en eenmalige posten (pensioenbate €27 mln in 2024, herstructureringen). Deze analyse gebruikt de adjusted reeks voor marges, EPS en FCF en vermeldt IFRS er telkens naast.
- **Waarom**: de adjusted reeks geeft het beste beeld van de onderliggende verdienkracht over tijd; de uitsluitingen zijn overwegend non-cash (acquisitie-amortisatie) of werkelijk eenmalig (FRR-boekwinst). Let op: acquisitie-amortisatie is elk jaar aanwezig — wie het streng bekijkt, rekent met IFRS-EPS die circa 5-12% lager ligt. De DCF is op kasstromen gebaseerd en heeft hier geen last van.

### Earnings quality
Accruals ratio ((IFRS-nettowinst − CFO) / gem. totale activa): 2023: −5,8% | 2024: −6,2% | 2025: −3,8%. De ratio is diep negatief — de kasstroom is structureel groter dan de winst — wat op conservatieve winstverantwoording duidt; de minder negatieve 2025-waarde komt door de (non-cash meegetelde) FRR-boekwinst in de IFRS-winst, niet door verslechterende cash-generatie. Het verschil adjusted vs. IFRS nettowinst was +11% (2023), +10% (2024) en −6% (2025, IFRS hoger door FRR); de aanpassingen zijn transparant gespecificeerd in de persberichten en jaar-op-jaar consistent van aard. SBC kon niet apart worden gekwantificeerd uit de geopende bronnen (zie sectie 13) — de verwatering wordt in de praktijk ruimschoots overgecompenseerd door buybacks (aandelenaantal −22% in tien jaar), maar een expliciete FCF-na-SBC-reeks ontbreekt daardoor.

### Dividend

| Jaar | DPS (€) | Groei % | Payout op adj. EPS % | FCF-dekking (adj. FCF / cash-dividend) | Type |
|---|---|---|---|---|---|
| 2015 | 0,75 | — | 38 | — | regulier |
| 2016 | 0,79 | +5,3 | 38 | 3,2 | regulier |
| 2017 | 0,85 | +7,6 | 38 | — | regulier |
| 2018 | 0,98 | +15,3 | 40 | 2,8 | regulier |
| 2019 | 1,18 | +20,4 | 41 | 2,9 | regulier |
| 2020 | 1,36 | +15,3 | 43 | 2,7 | regulier |
| 2021 | 1,57 (AGGR.) | +15,4 | 46 | 2,7 | regulier |
| 2022 | 1,81 | +15,3 | 44 | 2,9 | regulier |
| 2023 | 2,08 | +14,9 | 46 | 2,3 | regulier |
| 2024 | 2,33 | +12,0 | 47 | 2,3 | regulier |
| 2025 | 2,52 | +8,2 | 48 | ~2,4 | regulier |

- **Dividend-CAGR 2015-2025:** 12,9%. Geen enkele verlaging of overslag in de reeks — ook niet in 2020 — en sinds 2007 een progressief beleid (persberichten spreken consistent van dividendverhogingen). Speciale of stockdividenden komen niet voor; keuzedividend evenmin in de recente jaren.
- **Huidig rendement:** 2,52 / 57,56 = 4,4% — historisch uitzonderlijk hoog voor WKL (bij koersen van €120-160 was het 1,6-2,1%) en ruim boven de Nederlandse 10-jaarsrente van 3,03%. Dat hoge rendement is een koerseffect, geen beleidswijziging: de payout (48% van adj. EPS; ~42% van FCF) is comfortabel en de FCF-dekking van 2,4x ruim.
- **Eerstvolgend besluit:** interimdividend 2026 (~40% van het totaal 2025 = ca. €1,01), ex-datum 1 september 2026. Gegeven guidance (high single-digit EPS-groei) en de dekkingsratio's is een verdere verhoging aannemelijk; dat is een verwachting, geen toezegging.
- **Oordeel houdbaarheid:** Conservatief gefinancierd dividend: minder dan de helft van winst en kasstroom, dekking 2,4x, progressief beleid dat door de kredietcrisis noch corona is onderbroken. Zelfs in het pessimistische AI-scenario (FCF −2% per jaar) blijft het huidige dividend jaren houdbaar; het risico zit in groei, niet in continuïteit.

### Sector-KPI's (informatie/software)

| KPI | Eenheid | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| Organische omzetgroei | % | 6 | 6 | 6 | 6 | 6 |
| Adjusted operating margin | % | 25,3 | 26,1 | 26,4 | 27,1 | 27,5 |
| Recurring omzet als % totaal | % | — | — | — | — | 83 (Q1'26: 85) |
| Cloudsoftware als % omzet | % | — | — | — | — | 21 (Q1'26: 23; +15% org) |
| ROIC (gerapporteerd) | % | 13,7 | 15,5 | 16,8 | 18,1 | 18,0 |
| FCF-conversie (FCF/adj. winst) | % | 114 | 115 | 104 | 108 | 110 |

Toelichting: de KPI-set laat precies zien waarom dit bedrijfsmodel wordt gewaardeerd zoals SaaS-bedrijven wórden gewaardeerd — behalve nu bij WKL zelf. Vijf jaar lang exact 6% organische groei is een zeldzaam consistentieniveau; de recurring-ratio van 85% (Q1 2026) en cloudgroei van 14-15% wijzen op een gezond migratiepad; ROIC bijna verdubbelde sinds 2015. Churn-, ARR- en netto-retentiecijfers publiceert WKL niet — de recurring-groei van +7% organisch is de beste beschikbare proxy en impliceert lage churn plus prijsverhogingen. De vraag die deze KPI's niet kunnen beantwoorden: of AI-substitutie de komende jaren in deze cijfers gaat verschijnen. Tot en met Q1 2026 is daarvan niets zichtbaar.

---

## 4. Moat (concurrentievoordeel)

- **Oordeel** (enum): **NARROW MOAT**
- **Moat-categorieën**:

| Naam | Sterkte (sterk/middel/zwak/geen) | Toelichting |
|---|---|---|
| Immateriële activa | sterk | Bijna twee eeuwen geredigeerde, geverifieerde vakcontent (UpToDate, CCH, Kluwer-jurisprudentie), merken die in beroepsgroepen de facto standaard zijn, en regulatoire expertise die continu wordt bijgehouden door duizenden vakredacteuren. Dit corpus is niet legaal kopieerbaar en vormt de trainings- en verificatiebasis die generieke AI-modellen juist missen — maar AI verlaagt wél de waarde van pure content-toegang, vandaar geen wide-moat-oordeel meer. |
| Overstapkosten | sterk | CCH Axcess zit in de kernprocessen van accountantskantoren, UpToDate in klinische protocollen van ziekenhuizen, CT Corporation in juridische entiteitsadministratie. Overstappen betekent dataconversie, hertraining, workflow-herbouw en compliance-risico. De recurring-retentie (organische groei recurring +7%, boven groepsgemiddelde) bevestigt dit gedrag in de cijfers. |
| Netwerkeffecten | zwak | Beperkt aanwezig: UpToDate wint aan gezag naarmate meer clinici bijdragen en citeren, en compliance-netwerken (Isabel) hebben transactienetwerk-elementen, maar het klassieke marktplaats-effect ontbreekt. Dit is geen kern van de moat. |
| Kostenvoordeel | middel | Schaalvoordelen in contentproductie en productontwikkeling: de kosten van het actueel houden van wet- en regelgevingsdatabases spreiden zich over honderdduizenden abonnees. Een nieuwkomer moet die vaste kosten dragen bij nul omzet. AI verlaagt echter de reproductiekosten van content, wat dit voordeel geleidelijk kan verdunnen. |
| Efficiënte schaal | middel | In niches als klinische beslisondersteuning en bank-compliance is ruimte voor twee à drie spelers; de markt is te specialistisch voor generieke techreuzen om er vol op in te zetten, maar groot genoeg voor WKL om er schaal te hebben. AI-native startups (aangejaagd door goedkoop kapitaal richting legal-AI) testen deze barrière momenteel actief. |

- **Kwantitatief bewijs:** ROIC steeg van 9,3% (2015) naar 18,0-18,1% (2024-2025) — bij een WACC van 6,4% een spread van ruim 11 procentpunten, vijf jaar op rij boven de 7 punten. De aangepaste marge steeg elf jaar op rij. Prijszettingsmacht blijkt uit organische groei van 6% bij volwassen eindmarkten die zelf 2-4% groeien. Dit is het cijferprofiel van een moat; de onzekerheid zit uitsluitend in de houdbaarheid ervan.
- **Duurzaamheid:** 5-10 jaar mits WKL de AI-transitie zelf leidt. Morningstar verlaagde in 2025/2026 de rating van Wide naar Narrow precies op dit punt: het risico is niet dat de huidige klanten morgen vertrekken, maar dat de volgende generatie professionals (en hun software-stack) om de gevestigde content-plus-workflow heen wordt gebouwd. Deze analyse volgt dat oordeel: NARROW MOAT met twee sterke categorieën.
- **Erosierisico's:** (1) Generatieve AI die "goed genoeg" juridisch/fiscaal onderzoek levert tegen een fractie van de prijs, met name in het L&R-segment; (2) AI-native concurrenten die workflow plus AI bouwen zonder legacy-content-kosten; (3) grote klanten (Big Four, ziekenhuisketens) die eigen AI-oplossingen op eigen data trainen; (4) druk op prijs-per-seat-modellen als AI het aantal benodigde professionals per kantoor verlaagt — een tweede-orde-effect dat zelden wordt ingeprijsd.

---

## 5. Management

- **CEO-naam + tenure**: Stacey Caywood, CEO sinds februari 2026; daarvoor CEO van Wolters Kluwer Health en ruim 30 jaar bij het bedrijf (o.a. CEO Legal & Regulatory). Opvolging van Nancy McKinstry (CEO 2003-2026) is ruim een jaar van tevoren aangekondigd en intern ingevuld — een ordelijke, cultuurbestendige transitie.
- **CFO-naam + tenure**: Kevin Entricken, CFO sinds 2013; blijft aan onder de nieuwe CEO (bevestigd bij de opvolgingsaankondiging van 26-02-2025).
- **Oprichter nog betrokken?**: n.v.t. (bedrijf uit 1836/1987; geen oprichtersfamilie meer betrokken)
- **Insider ownership %**: ~0,25% van de aandelen (StockAnalysis); Executive Board hield per 31-12-2025 samen 535.921 aandelen (FY2025-rapport) — bij de huidige koers ~€31 mln, substantieel in absolute termen maar geen owner-operator-profiel.
- **Capital allocation track record**:

| Jaar | Dividend cash | Aandeleninkoop | M&A-uitgaven (bruto) | Capex |
|---|---|---|---|---|
| 2022 | 424 | 1.000 | — | 295 |
| 2023 | 500 | 1.000 | — | 323 |
| 2024 | 545 | 1.000 | 342 (o.a. Isabel) | 313 |
| 2025 | ~570 | 1.100 | 896 (RASi, Brightflag, Libra) | 303 |

- **M&A-track-record**: Consequent bolt-on-beleid: kleinere software-overnames die in bestaande divisies worden geïntegreerd (Isabel Group 2024; RASi €386 mln, Brightflag €436 mln, Libra tot €90 mln in 2025; StandardFusion ~€32 mln in januari 2026), gecombineerd met discipline aan de verkoopkant (FRR-divestment 2025 voor netto €399 mln mét €232 mln boekwinst). Geen megadeals, geen gerapporteerde goodwill-impairments in de geopende bronnen. De stijgende ROIC (9,3% → 18,0%) ondanks €4,8 mld goodwill is het beste bewijs dat de acquisities per saldo waarde toevoegen.
- **Beloning**: Details van de remuneratiestructuur (bonusdoelen, LTI-KPI's, pay ratio) staan in het remuneratierapport binnen het jaarverslag, dat in deze run niet als PDF is doorgenomen — zie sectie 13. Bekend uit de persberichten: guidance stuurt op adjusted EPS-groei, marge, FCF en ROIC — precies de maatstaven waarop ook de LTI's historisch zijn geënt. Geen beloningscontroverses gevonden in het nieuws van de afgelopen 24 maanden.
- **Insider-transacties (24 maanden):** Opvallend eenduidig patroon sinds de koersval: uitsluitend aankopen. Caywood (toen CEO-designate) kocht 15-08-2025 3.775 aandelen à €112,92 (~€426k); CFO Entricken kocht 27-08-2025 2.700 à €111,30 (~€301k); commissaris Sides kocht 18-08-2025 1.875 ADR's à $131,52; commissarissen De Kreij (25-02-2026) en Vogelzang (26-02-2026, à €61,79) kochten na de jaarcijfers — de exacte volumes van die laatste twee bevatten parsingfouten in de bron (InsiderScreener) en zijn niet exact geverifieerd, de richting (KOOP) wel. Laatste verkopen dateren van februari 2024 (à €147). Netto-oordeel: NETTO KOPER — het management koopt op de daling met eigen geld.
- **Oordeel management** (enum): **STERK**
- **Toelichting**: Het managementoordeel steunt op vier poten. Eén: een capital-allocation-track-record van meer dan tien jaar waarin ROIC verdubbelde, marges elf jaar op rij stegen en desinvesteringen op het juiste moment gebeurden. Twee: prikkels die aan langetermijn-KPI's (EPS, ROIC, FCF) zijn gekoppeld en een bestuur dat sinds augustus 2025 consequent eigen geld in het aandeel steekt. Drie: transparantie — guidance wordt gegeven, gehaald en in mei 2026 ondanks de koersval onverkort herbevestigd; tegenvallers zoals valuta-effecten worden expliciet gekwantificeerd. Vier: een ordelijke CEO-successie met een interne opvolger en een blijvende CFO. Kritiekpunten zijn er ook: de buybacks van 2022-2024 (à €98-149) waren achteraf duur — hoewel dat deels hindsight-bias is — en het insider-belang van 0,25% is te klein om van echte skin-in-the-game te spreken. Per saldo: STERK, maar zonder de owner-operator-premie.

---

## 6. Sector & concurrentie

- **Sector-groeivooruitzicht**: De professionele informatie- en compliance-softwaremarkt groeit structureel 4-7% per jaar, gedreven door toenemende regelgevingscomplexiteit (ESG-rapportage, fiscale hervormingen zoals Pillar 2, zorgdigitalisering); WKL's eigen consensus-omzetgroei staat op +4,4% per jaar (3-jaars, StockAnalysis). Het sectorsentiment is echter fors negatief sinds 2025: beleggers herprijzen het hele cluster op AI-disruptierisico, los van de gerapporteerde groei.
- **Porter five forces**:
  - Rivaliteit: MIDDEL — per niche twee à drie gevestigde spelers (Thomson Reuters en LexisNexis/RELX in legal-tax, Elsevier in medisch, Diligent/AuditBoard in GRC) die op kwaliteit en integratie concurreren, niet op prijs; maar AI-startups introduceren voor het eerst in decennia echte prijscompetitie aan de onderkant.
  - Nieuwe toetreders: MIDDEL — klassiek waren de drempels zeer hoog (contentcorpus, redactie-infrastructuur, distributie, vertrouwen); generatieve AI verlaagt de contentdrempel wezenlijk, al blijft de workflow-integratie en aansprakelijkheidsbestendigheid een barrière. De dreiging is reëler dan vijf jaar geleden.
  - Substituten: HOOG — dit ís het kernrisico: generieke LLM's en AI-native tools als substituut voor opzoek- en researchabonnementen. Voor klinische beslissingen en compliance-workflows is het substitutierisico lager (validatie- en aansprakelijkheidseisen), voor juridisch-fiscale naslag het hoogst.
  - Macht leveranciers: LAAG — de belangrijkste inputs zijn publieke bronnen (wetten, jurisprudentie, literatuur) en eigen redacties; geen geconcentreerde leveranciersmacht. Nieuwe afhankelijkheid van LLM-leveranciers (compute, modellen) is een aandachtspunt maar momenteel geen prijszettingsprobleem.
  - Macht afnemers: MIDDEL — klanten zijn gefragmenteerd en gebonden via overstapkosten, maar accountants- en advocatenkantoren staan zelf onder AI-druk op hun facturabele uren, wat op termijn seat-gebaseerde omzet kan raken; grote klanten (Big Four) hebben onderhandelingsmacht en eigen AI-programma's.
- **Concurrenten**:

| Concurrent | Marktaandeel % | Kerncijfers (peildatum juni/juli 2026, AGGREGATOR) |
|---|---|---|
| RELX (LexisNexis, Elsevier) | — | P/E fwd ~17,2; EV/EBITDA ~15,7; underlying groei +8% (FY25); adj. marge 34,8%; koers −41% vanaf top |
| Thomson Reuters | — | P/E fwd ~21,1; EV/EBITDA-bronnen tegenstrijdig (11,5-20,6); koers −48% vanaf top |
| SS&C Technologies | — | P/E fwd ~10,5; EV/EBITDA ~12,3; omzetgroei +8,8% (Q1'26); koers −12% |
| Clarivate | — | Krimpende omzet; EV/EBITDA ~6,2; het waarschuwende voorbeeld van een info-dienstverlener zonder groei |
| Intuit / AI-native legal-tech (Harvey e.a.) | — | Niet-genoteerd of niet direct vergelijkbaar; relevant als disruptiedreiging, niet als multiple-referentie |

Betrouwbare marktaandeelcijfers per niche zijn niet uit een verifieerbare bron beschikbaar — weggelaten (zie sectie 13).

- **Positie van het bedrijf**: Leider of nummer twee in elk van zijn vijf divisies, met de hoogste recurring-ratio van de peer-groep en — na de derating — de laagste waardering van de kwaliteitsnamen: WKL doet fwd P/E ~10 tegen RELX ~17 en Thomson Reuters ~21, bij vergelijkbare of betere organische groei. De korting ten opzichte van RELX is deels te verklaren (RELX heeft met exhibitions en risk-analytics een bredere mix en toonde +8% groei), maar de omvang van de korting — bijna een halvering van de multiple — impliceert dat de markt WKL het hardst afstraft voor hetzelfde sectorrisico. Binnen de sectortrends (AI, cloud, compliance-groei) is WKL eerder koploper dan volger: 21-23% van de omzet is al cloud en het bedrijf verhoogt de productontwikkelingsuitgaven naar 12-13% van de omzet in 2026 — de facto een verdedigingsinvestering tegen disruptie.

### TAM/SAM/SOM
- **TAM (mln lokale valuta)**: — (geen verifieerbare recente bron geopend in deze run)
- **TAM-groei %**: —
- **SAM (mln)**: —
- **SAM-groei %**: —
- **Huidige penetratie %**: —
- **Impliciete penetratie na horizon %**: —
- **Groei plausibel?**: true
- **Bron TAM/SAM**: geen — bewust leeggelaten in plaats van een niet-verifieerbaar marktrapport te citeren
- **Toelichting**: Een formele TAM-berekening ontbreekt bij gebrek aan een geopende, betrouwbare bron. De plausibiliteitscheck kan wel kwalitatief: WKL groeit 6% organisch in eindmarkten (zorg, fiscaal, juridisch, compliance) die zelf laag-enkelcijferig groeien, dus de groei komt uit prijs, cloudmigratie-upsell en aandeelwinst in software-niches — geen heroïsche penetratie-aannames nodig. De DCF-basisaanname van 5% FCF-groei vereist geen marktaandeelwonder; dat maakt de groeiprognose robuust voor TAM-onzekerheid.

---

## 7. Analyse-frameworks (9 frameworks, SCORES 0-5)

Scores volgen de deterministische rubrics uit METHODE.md H9; kwantitatieve inputs: P/E 10,9 (adj., trailing), P/B 16,3, P/FCF 9,6, PEG 1,21, ROIC 18,0%, WACC 6,37%, earnings yield 10,1%, moat NARROW (2 categorieën sterk), management STERK, DCF-upside +195%.

### Graham
- **Oordeel**: VOLDOET NIET
- **Graham number**: 20,5 (√(22,5 × adj. EPS 5,29 × boekwaarde p.a. 3,53))
- **Margin of safety %** (koers t.o.v. Graham number): −64
- **Toelichting**: Voor Graham draait alles om tastbare zekerheid, en die meetlat past fundamenteel niet op WKL. De P/E van 10,9 zou Graham juist aanspreken, maar de boekwaarde per aandeel is door twintig jaar aandeleninkoop boven boekwaarde uitgehold tot €3,53, waardoor P/B op 16,3 staat en het Graham-getal (€20,5) ver onder de koers ligt. Dit zegt meer over de beperkingen van boekwaarde bij asset-light softwarebedrijven dan over WKL — vrijwel de hele waarde zit in niet-geactiveerde content, klantrelaties en merken. Toch is de rubric hier terecht streng: wie uitsluitend Graham-criteria hanteert, heeft bij dit aandeel geen vangnet van tastbare activa. Score volgt hard uit P/B > 3,0.
- **Score (0-5)**: 1

### Buffett / Munger
- **Oordeel**: GEDEELTELIJK
- **ROIC structureel boven WACC?**: true
- **Toelichting**: Veel Buffett-boxen worden afgevinkt: een begrijpelijk bedrijf (abonnementen op onmisbare vakinformatie), voorspelbare kasstromen (vijf jaar exact 6% organische groei, FCF-conversie boven 100%), ROIC van 18% tegen een WACC van 6,4% — bijna drie keer de kapitaalkosten — en een prijs van minder dan tien keer de vrije kasstroom. Waarom dan geen volmondig VOLDOET? Omdat het cruciale Buffett-criterium — een moat waarvan je zeker weet dat hij er over tien jaar nog is — precies het punt is dat de markt betwist en dat ook deze analyse met NARROW beoordeelt. Buffett koopt zekerheid over duurzaamheid; die zekerheid is hier verlaagd, hoe goedkoop het aandeel ook is. Score 3 volgt uit de rubric (ROIC > WACC structureel, moat NARROW, P/FCF ≤ 30).
- **Score (0-5)**: 3

### Peter Lynch
- **Categorie**: Stalwart
- **Oordeel**: INTERESSANT
- **PEG-ratio**: 1,21
- **Toelichting**: WKL is een klassieke stalwart: een grote, gevestigde onderneming met bescheiden maar betrouwbare winstgroei (consensus EPS +9% per jaar) waar je volgens Lynch op moet inzetten wanneer het sentiment tegenzit — en dat doet het nu uitzonderlijk hard. De PEG van 1,21 (P/E 10,9 gedeeld door 9% groei) is voor een bedrijf van deze kwaliteit historisch laag; begin 2025 stond dezelfde PEG boven de 3. Het verhaal is bovendien in twee zinnen uitlegbaar: professionals betalen jaarlijks voor informatie en software waar hun beroepsaansprakelijkheid van afhangt, en het bedrijf verhoogt al tien jaar marges en koopt aandelen in. De Lynch-vraag "is het verhaal intact?" is tegelijk de AI-vraag — vandaar interessant, geen tafelspringer. Score 3 volgt uit PEG ≤ 1,5 met helder verhaal.
- **Score (0-5)**: 3

### Phil Fisher
- **Oordeel**: STERK
- **Toelichting**: Fisher zou van de vijftien punten er veel afvinken: producten met langjarig groeipotentieel (compliance en zorgdigitalisering groeien autonoom), een organisatie die consequent in productontwikkeling investeert (opgevoerd naar 12-13% van de omzet in 2026 — de facto het R&D-antwoord op AI), bewezen margediscipline (elf jaar expansie op rij) en een management met lange interne loopbanen, transparante communicatie en insider-aankopen op de daling. Wat ontbreekt voor de topscore: een geverifieerde vergelijking van R&D-intensiteit versus sectorgemiddelde (niet beschikbaar uit geopende bronnen) en de Fisher-achtige zekerheid dat verkooporganisatie en innovatie de disruptiegolf vóórblijven in plaats van volgen. Twee van de drie rubric-criteria (margebescherming door moat, integriteit STERK) zijn hard aantoonbaar; dat geeft score 4.
- **Score (0-5)**: 4

### Magic Formula (Greenblatt)
- **Oordeel**: AANTREKKELIJK
- **Earnings yield %**: 10,1 (adj. EBIT 1.687 / EV 16.710, met EV = marktkap 12.922 + nettoschuld 3.788)
- **Return on capital %**: 18,0 (conservatieve proxy: gerapporteerde ROIC incl. goodwill; Greenblatt-ROC op tastbaar kapitaal zou wezenlijk hoger uitvallen omdat het werkkapitaal negatief is en de tastbare activa klein zijn, maar de daarvoor benodigde balansdetaillering zat niet in de geopende bronnen)
- **Toelichting**: De Magic Formula zoekt de combinatie van goedkoop (hoge earnings yield) en goed (hoog rendement op kapitaal), en WKL scoort op de eerste as uitzonderlijk: een earnings yield van ruim 10% betekent dat de operationele winst in tien jaar de hele ondernemingswaarde terugverdient — voor een bedrijf met 83% terugkerende omzet is dat een waarde die je normaal alleen bij krimpbedrijven ziet. Op de tweede as is 18% ROIC solide maar geen uitschieter — al is dat cijfer gedrukt door €4,8 mld acquisitie-goodwill; op tastbaar kapitaal is het rendement veel hoger. De rubric-score valt op 3 doordat de geverifieerde ROC onder de 30%-drempel blijft; met een Greenblatt-zuivere ROC was dit vermoedelijk een 4 of 5 — het cijfermatige voordeel van de twijfel is bewust niet genomen.
- **Score (0-5)**: 3

### Moat
- **Score (0-5)**: 3 — NARROW MOAT met twee sterke categorieën (immateriële activa, overstapkosten) en een ROIC-WACC-spread van ruim 11 procentpunten die vijf jaar op rij boven de 5-puntendrempel ligt. Geen 4 of 5 omdat het oordeel WIDE vereist zou zijn en juist de duurzaamheid van het voordeel onder AI-druk staat (Morningstar-downgrade naar Narrow, door deze analyse gevolgd).

### Management
- **Score (0-5)**: 4 — capital allocation GOED (ROIC verdubbeld, gedisciplineerde bolt-ons, tijdige FRR-verkoop), prikkels gekoppeld aan EPS/ROIC/FCF, geen controverses, insiders netto kopers. Geen 5 omdat het insider-belang (0,25%) onder de 1%-alignmentdrempel blijft en de buyback-timing 2022-2024 achteraf duur was.

### Fair Value DCF
- **Score (0-5)**: 5 — basisscenario fair value €170 tegen koers €57,56: upside +195%, ver boven de 30%-drempel voor de topscore. Zelfs het pessimistische scenario (blijvende FCF-krimp) komt op €58 uit — op of boven de huidige koers.

### Fair Value IPO-gecorr.
- **Score (0-5)**: 5 — beursnotering dateert van 1987 (> 10 jaar), dus per rubric gelijk aan de Fair Value DCF-score. Er is geen pre-IPO-vertekening om voor te corrigeren.

### Scorekaart totaal
- **Totaalscore**: 31
- **Max**: 45 (9 × 5)
- **Eindoordeel** (enum): **HOLD** (totaal 31 < 33, dus geen KOOP ondanks DCF-score 5; totaal ≥ 24 en DCF-score ≠ 1, dus geen PASS)
- **Samenvatting**: De scorekaart vertelt een dubbel verhaal: op waardering scoort WKL maximaal (twee keer 5), op kwaliteitsframeworks solide (Fisher 4, Management 4, Buffett/Lynch/Greenblatt/Moat 3), maar Graham's boekwaarde-eis (score 1) trekt het totaal naar 31 — net onder de KOOP-drempel van 33. Dat deterministische HOLD-oordeel is te verdedigen als "koop niet blind": de lage score op tastbare zekerheid en de betwiste moat-duurzaamheid zijn precies de reden dat het aandeel zo goedkoop is. De voornaamste onzekerheid is of AI-substitutie in de komende kwartalen zichtbaar wordt in organische groei en retentie; de katalysatorkalender (halfjaarcijfers 5 augustus, negenmaandsupdate 4 november) levert daarvoor de eerste testmomenten. Wie instapt doet dat met de wetenschap dat het pessimistische scenario (€58) vrijwel de huidige koers is — het neerwaartse risico op de aannames is beperkt, het opwaartse potentieel groot, maar de bewijslast ligt bij de komende cijfers. Minimale margin of safety: 25%, ruimschoots aanwezig ten opzichte van de kansgewogen fair value van €152.

---

## 8. Risico's

| # | Omschrijving | Kans | Impact | DCF-aanname geraakt | Toelichting |
|---|---|---|---|---|---|
| 1 | Generatieve AI commoditiseert juridisch/fiscaal/medisch onderzoek en drukt abonnementsprijzen | MIDDEN | GROOT | omzetgroei fase 1-2, terminal groei | Het kernrisico dat de halvering van de koers verklaart. Generieke LLM's leveren steeds beter "goed genoeg"-onderzoek; als klanten daardoor abonnementen afschalen, raakt dat eerst L&R (16% omzet) en daarna T&A. Tot en met Q1 2026 onzichtbaar in de cijfers (recurring +7%), maar de bewijslast vernieuwt zich elk kwartaal. Het pessimistische scenario (-2% eeuwig) kwantificeert dit; het weegt 30% mee in de kansgewogen fair value. |
| 2 | AI-native concurrenten (goed gefinancierde legal-tech/tax-tech startups) winnen de volgende productgeneratie | MIDDEN | GROOT | omzetgroei fase 2, terminal groei | Anders dan risico 1 gaat dit niet om substitutie door generieke AI maar om gespecialiseerde nieuwkomers zonder legacy-kosten. WKL's antwoord — productontwikkeling naar 12-13% van omzet — drukt op korte termijn de marge-expansie die in de guidance zit. Duurt de investeringsfase langer dan gepland, dan raakt dat de margeaanname in de projectie. |
| 3 | USD-verzwakking drukt gerapporteerde resultaten en waardering | HOOG | MIDDEL | FCF-basis (in EUR), geen effect op organische groei | 63% van de omzet is Noord-Amerikaans. Q1 2026 toonde het effect: +4% CC werd -3% gerapporteerd bij €/$ van 1,05 naar 1,17. De guidance hanteert 1,175; verdere dollarzwakte verlaagt de EUR-kasstromen in de DCF vrijwel één-op-één met het NA-aandeel. Structureel risico dat al deels is gematerialiseerd. |
| 4 | Klantenbestand onder druk: AI verlaagt personeelsbehoefte bij accountants- en advocatenkantoren (seat-erosie) | MIDDEN | MIDDEL | omzetgroei fase 2 | Tweede-orde-effect: ook als kantoren klant blijven, daalt bij per-seat-prijsmodellen de omzet mee met het aantal professionals. WKL migreert deels naar waarde-/modulegebaseerde prijzen, maar het tempo daarvan is niet gepubliceerd. Raakt vooral de jaren 6-10 van de projectie. |
| 5 | M&A-integratie en goodwill-risico na ~€900 mln acquisities in 2025 | LAAG | MIDDEL | FCF-basis, marge | RASi, Brightflag en Libra moeten de beloofde groei leveren; goodwill staat op €4,8 mld (50% van activa). Het track record (geen impairments, stijgende ROIC) maakt de kans laag, maar een impairment zou het toch al dunne eigen vermogen (€798 mln) raken en het sentiment verder drukken. |
| 6 | CEO-transitie: eerste strategiewijzigingen onder Caywood vallen tegen of leiden tot vertrek sleutelpersoneel | LAAG | MIDDEL | omzetgroei fase 1, marge | Caywood is een interne opvolger met een sterk track record (Health-divisie) en de CFO blijft — de continuïteit is bewust geborgd. Het risico is vooral dat een nieuwe CEO in een crisissfeer tot een strategische koerswijziging (bijv. veel grotere AI-overname) wordt verleid die kapitaaldiscipline doorbreekt. |
| 7 | Leverage-stijging beperkt flexibiliteit: nettoschuld/EBITDA 2,0x bij uitgehold eigen vermogen | LAAG | KLEIN | WACC (schuldkosten) | 2,0x is voor dit kasstroomprofiel comfortabel en Q1 2026 toonde alweer 1,9x; herfinanciering is gedekt met nieuwe eurobonds (3,0-3,375%). Alleen in combinatie met een diepe FCF-krimp (risico 1) zou de schuld gaan knellen — dan is de schuld het symptoom, niet de oorzaak. |
| 8 | Pre-IPO financial engineering | LAAG | KLEIN | geen | Verplicht checkpunt: niet van toepassing en niet geconstateerd. WKL is sinds 1987 genoteerd; er zijn geen pre-IPO-schulden bij gerelateerde partijen, geen IPO-opbrengsten naar insiders en geen dividend-recapitalisatie. Geen correctie op de fair value nodig; het IPO-gecorrigeerde scenario is identiek aan de basis. |

---

## 9. These invalide bij

Deze these is weerlegd wanneer de organische groei twee kwartalen op rij onder de 3% zakt of de recurring-groei onder de groepsgroei duikt (eerste harde bewijs van AI-churn), wanneer het management de marge- of FCF-guidance verlaagt onder verwijzing naar prijsdruk of klantverlies aan AI-alternatieven, of wanneer een grote klantengroep (bijv. een Big Four-kantoor of grote ziekenhuisketen) publiek overstapt op een AI-native alternatief voor CCH of UpToDate. Omgekeerd vervalt de koopcase op prijs wanneer de koers zonder fundamentele verbetering boven het koopniveau van ~€128 uitstijgt.

---

## 10. ESG

### Materiële factoren (SASB-gebaseerd)

| Factor | SASB-categorie | Risiconiveau (Laag/Midden/Hoog) | Financiële impact | DCF-impact |
|---|---|---|---|---|
| Databeveiliging & privacy (klinische en fiscale klantdata) | Customer Privacy / Data Security | Midden | Een materieel datalek in UpToDate- of CCH-omgevingen zou boetes (AVG/HIPAA), klantverlies en reputatieschade geven; preventiekosten zijn structureel stijgend | Marge (hogere security-opex); geen aparte DCF-aftrek |
| Betrouwbaarheid en aansprakelijkheid van (AI-)content | Product Quality & Safety | Midden | Foutieve klinische of fiscale AI-adviezen kunnen tot claims en vertrouwensverlies leiden; tegelijk is gevalideerde content juist WKL's verkoopargument | Terminal groei (vertrouwen = moat); opwaarts én neerwaarts |
| Talentbehoud (redacteuren, domein-experts, AI-engineers) | Employee Engagement | Midden | De AI-transitie vergt schaars talent; verloop onder domeinexperts zou de contentkwaliteit — de kern van de moat — uithollen | Marge (loonkosten), fase-2-groei |
| CO2-voetafdruk | GHG Emissions | Laag | Asset-light dienstenmodel met beperkte directe emissies; geen materiële transitiekosten | Geen |

- **Eindoordeel ESG** (enum): **LAAG RISICO**
- **Toelichting**: De materiële ESG-factoren van WKL zijn de klassieke risico's van een informatiebedrijf — data, contentintegriteit en mensen — en geen daarvan wijkt negatief af van de sector; de milieu-voetafdruk is verwaarloosbaar. De AI-ethiek-dimensie verdient monitoring omdat WKL's waardepropositie juist op gevalideerde, aansprakelijkheidsbestendige informatie rust: één groot incident met AI-gegenereerde fouten in klinische of fiscale producten zou onevenredig schadelijk zijn. Een gedetailleerde beoordeling van het duurzaamheidsverslag viel buiten de geopende bronnen van deze run.

---

## 11. Katalysatoren (chronologisch)

| Datum ca. | Omschrijving | Richting | Impact |
|---|---|---|---|
| 2026-07 | Doorlopende wekelijkse buyback-executie (tranche ~€80 mln t/m 3 aug; totaal 2026 tot €500 mln) | POSITIEF | KLEIN |
| 2026-08 | Halfjaarcijfers 2026 (5 augustus): eerste volledige cijferset onder CEO Caywood; markt zoekt naar AI-churn-signalen in recurring-groei | BINAIR | GROOT |
| 2026-09 | Ex-dividend interimdividend 2026 (1 september, ~€1,01) | POSITIEF | KLEIN |
| 2026-11 | Negenmaands trading update (4 november): herbevestiging of bijstelling guidance 2026 | BINAIR | MIDDEL |
| 2026-Q4 | Peers-cijfers RELX/Thomson Reuters als sector-sentimentkatalysator (AI-impact wel/niet zichtbaar bij concurrenten) | NEUTRAAL | MIDDEL |
| 2027-02 | FY2026-cijfers + guidance 2027 (24 februari): test van de margedoelstelling ~28% en dividendverhoging | BINAIR | GROOT |
| 2027-H1 | Aankondiging buyback-programma 2027 — omvang is signaal van managementvertrouwen bij deze koers | POSITIEF | MIDDEL |
| doorlopend | AI-productlanceringen en -partnerships van WKL (monetisatie van eigen AI in CCH/UpToDate) versus doorbraken van AI-native concurrenten | BINAIR | GROOT |

De halfjaarcijfers van 5 augustus 2026 zijn de belangrijkste korte-termijnkatalysator: bij aanhoudende organische groei van ~5-6% en herbevestigde guidance ontbreekt het disruptiebewijs opnieuw en is de kans op herwaardering reëel; bij een groeivertraging krijgt het pessimistische scenario juist gewicht. Beide uitkomsten raken de DCF-groeiaannames direct.

---

## 12. Fair value — kwantitatief (DCF)

### WACC-componenten
- **Risicovrije rente %**: 3,03
- **Bron risicovrije rente**: Nederlandse 10-jaars staatsobligatie, 02-07-2026 (Trading Economics); kasstromen zijn in EUR gemodelleerd, NL-Bund-spread verwaarloosbaar
- **Type**: nominal, spot
- **ERP (equity risk premium) %**: 4,23
- **Bron ERP**: Damodaran implied ERP per 01-01-2026 (datacurrent.html; meest recente geverifieerde stand — juli-update niet beschikbaar in deze run)
- **Beta (adjusted)**: 0,82
- **Bron beta**: Damodaran betaEurope.xls (jan 2026), industrie "Information Services": levered beta 0,82 (unlevered cash-corrected 0,44). De eigen 5-jaars regressiebeta (0,15 per StockAnalysis) is door de idiosyncratische crash onbruikbaar en bewust NIET gebruikt.
- **Type beta**: bottom-up (sector-beta; eigen regressie verworpen)
- **Country risk premium %**: 0 (Nederland)
- **Size premium %**: 0 (large cap, €12,9 mld)
- **Cost of equity %**: 6,50
- **Schuldkosten na belasting %**: 2,44 (pre-tax 3,2% conform recente eurobond-coupons; benchmark tax 23,6%)
- **E/V gewicht %**: 72,21
- **D/V gewicht %**: 27,79 (bruto schuld €4.972 mln tegen marktkap €12.922 mln)
- **WACC %**: 5,37 (componenten-uitkomst)
- **Sector WACC % (referentie)**: ~7-8 (Damodaran-gebaseerde sectorreferenties voor information services)
- **Illiquiditeitskorting %**: null (AEX-hoofdfonds, zeer liquide)

**Belangrijke keuze:** de componenten-WACC van 5,37% ligt onder de sectorreferentie en voelt te licht voor een aandeel waarvan de markt de terminale kasstromen openlijk betwist. Daarom hanteren alle scenario's een expliciete risico-opslag bovenop de componenten-WACC: basis +1,0pp → **6,37%**, pessimistisch +2,0pp → 7,37%, optimistisch +0,5pp → 5,87%. Dit is conservatief gedocumenteerd in plaats van verstopt in de beta.

### DCF model-specs
- **Model type**: 2-fase (5 + 5 jaar) + Gordon growth terminal
- **FCF-definitie**: FCFF (free cash flow to firm), verdisconteerd tegen WACC; nettoschuld afgetrokken voor equity value
- **Basis FCF**: 1.414 (= adjusted FCF 2025 van 1.348 + netto financieringslasten na belasting van 66; daarmee omgerekend van equity- naar firm-basis)
- **Basis FCF na SBC**: — (SBC niet apart beschikbaar uit geopende bronnen; zie sectie 13 — verwatering wordt feitelijk geneutraliseerd door buybacks, maar een expliciete aftrek was niet construeerbaar)
- **FCF-type**: adjusted (gerapporteerde adjusted FCF als basis; 2025 is géén piekjaar — de reeks groeit al tien jaar vrijwel monotoon, cycliciteitscheck negatief: WKL is met 83% recurring omzet niet-cyclisch)
- **Groei fase 1 %** (jaar 1-5): 5,0 (basis)
- **Groei fase 2 %** (jaar 6-10): 3,5 (basis)
- **Terminal groei %**: 2,25 (basis)
- **Terminal methode**: Gordon growth
- **Exit multiple gebruikt**: 12x EV/EBITDA als cross-check (onder het huidige peer-mediaan-niveau: RELX ~15,7, SS&C ~12,3, Clarivate ~6,2)
- **Bron exit multiple**: peer-multiples per juni/juli 2026 (StockAnalysis, AGGREGATOR)
- **Terminal value Gordon growth**: 53.194 (TV in jaar 10, basis: FCFF jaar 10 ≈ 2.143 × 1,0225 / (6,37% − 2,25%); PV ≈ 28.700)
- **Terminal value exit multiple**: ~36.700 (12 × terminale EBITDA-benadering ≈ 3.062, met EBITDA ≈ FCF/0,70 en FCFF jaar 10 ≈ 2.143) — ca. 31% lager dan Gordon growth; het verschil illustreert dat de Gordon-terminal een herstel van vertrouwen veronderstelt, de exit multiple het huidige gedeprecieerde sentiment bevriest. Op exit-multiple-basis zou de fair value ~€131 zijn in plaats van €170.
- **Terminal value % van totaal**: 68,3 (basis; < 75%-drempel)
- **Terminal implied EV/EBITDA**: 17,4 (basis; onder de 20x-heroverwegingsdrempel maar aan de hoge kant — bij het optimistische scenario 21,3, wat tot terughoudendheid over dat scenario maant)
- **Terminal groei consistentie**: 2,25% terminale groei vereist bij een terminale ROIC van conservatief 12% een herinvesteringsvoet van ~19% van NOPAT — ruimschoots haalbaar voor een bedrijf dat nu ~18% capex+productontwikkeling van de operationele kasstroom herinvesteert. Consistent met nominale BBP-groei van de ontwikkelde wereld (2-3,5%); geen agressieve aanname.
- **Mid-year convention**: true
- **Aandelen uitstaand (mln)**: 224,5 (31-03-2026)
- **Nettoschuld huidig**: 3.788 (31-03-2026)

### DCF-toelichting
De DCF werkt met FCFF: de gerapporteerde adjusted FCF (€1.348 mln, een equity-maatstaf na rente) is verhoogd met €66 mln netto financieringslasten na belasting om op firm-niveau te komen, en wordt verdisconteerd tegen de scenario-WACC met mid-year convention; daarna gaat de actuele nettoschuld (€3.788 mln per maart 2026) eraf en wordt gedeeld door 224,5 mln aandelen. De basisgroei van 5% (fase 1) ligt bewust ónder de historische FCF-CAGR van 7,6% en onder de consensus-EPS-groei van 9%, omdat de AI-onzekerheid een korting op extrapolatie rechtvaardigt; fase 2 (3,5%) en terminal (2,25%) bouwen verder af. De terminal value is 68% van de totale waarde — binnen de norm, maar het bevestigt dat deze case over de lange termijn gaat. De impliciete terminale EV/EBITDA van 17,4x is aan de bovenkant van redelijk; de exit-multiple-crosscheck op 12x levert een fair value rond €131 op — nog altijd ruim 125% boven de koers. De belangrijkste keuze is niet de groei maar de WACC-opslag: zonder de +1,0pp zou de basis-fair-value nog fors hoger liggen.

### 5-jaars projectie (basisscenario)

| Jaar | Omzet | Omzetgroei % | Adj. EBIT | EBIT-marge % | NOPAT | Capex | ΔNWC | SBC | FCFF |
|---|---|---|---|---|---|---|---|---|---|
| 2026 | 6.401 | 4,5 | 1.779 | 27,8 | 1.359 | 333 | — | — | 1.485 |
| 2027 | 6.689 | 4,5 | 1.880 | 28,1 | 1.436 | 348 | — | — | 1.559 |
| 2028 | 6.990 | 4,5 | 1.985 | 28,4 | 1.517 | 363 | — | — | 1.637 |
| 2029 | 7.304 | 4,5 | 2.096 | 28,7 | 1.602 | 380 | — | — | 1.719 |
| 2030 | 7.633 | 4,5 | 2.214 | 29,0 | 1.691 | 397 | — | — | 1.805 |

Aannames: omzetgroei 4,5% (consensus 4,4% afgerond, onder organische trend van 6% vanwege print-krimp en FX-neutraliteit), margepad naar 29% conform guidance-richting (2026: ~28,0%), belasting 23,6%, capex 5,2% van omzet, FCFF groeit met het scenario-groeipad van 5%. ΔNWC is structureel licht negatief (vooruitbetaalde abonnementen — cash-genererend) maar niet apart gekwantificeerd uit de bronnen; SBC idem (sectie 13).

### Scenarios

| Scenario | FCF-groei % (fase 1 / fase 2 / terminal) | WACC % | Fair value | Upside % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | -2,0 / -2,0 / 0,0 | 7,37 | 58 | 0,8 | 30 |
| Basis | 5,0 / 3,5 / 2,25 | 6,37 | 170 | 195,3 | 50 |
| Optimistisch | 7,5 / 5,0 / 2,5 | 5,87 | 250 | 334,3 | 20 |

Scenariologica: het pessimistische scenario ís het AI-disruptiescenario — tien jaar FCF-krimp van 2% per jaar, nul terminale groei en een verhoogde WACC; dat het dan nog steeds op €58 uitkomt (de huidige koers) laat zien hoeveel slecht nieuws al is ingeprijsd. De kans van 30% op dit scenario is bewust hoog gezet: de markt is niet gek, en het staartrisico verdient serieus gewicht. Het optimistische scenario veronderstelt dat WKL AI juist monetariseert (premium-AI-modules bovenop bestaande abonnementen) en de historische groei hervat.

- **Kansgewogen fair value**: 152 (0,30 × 58 + 0,50 × 170 + 0,20 × 250)

### Reverse DCF
- **Impliciete groei %**: -2 (constante FCF-groei over 10 jaar én terminal, WACC 6,37%, mid-year: de koers van €57,56 impliceert blijvende jaarlijkse krimp van circa 2%)
- **Historische FCF CAGR %**: 7,6 (2015-2025)
- **Consensus groei % (analisten)**: 4,4 (omzet, 3 jaar) / 9,0 (EPS, 3 jaar) — StockAnalysis
- **Interpretatie**: Het contrast kan bijna niet groter: een bedrijf dat tien jaar lang zijn vrije kasstroom met gemiddeld 7,6% per jaar liet groeien, waarvan analisten 9% winstgroei verwachten en dat zijn eigen guidance in mei 2026 herbevestigde, wordt geprijsd alsof het voor altijd 2% per jaar gaat krimpen. Dat is geen genuanceerde afwaardering, dat is een disruptie-verdict. De reverse DCF maakt de belegging daarmee overzichtelijk: wie gelooft dat WKL zelfs maar stabiel blijft (0% groei), heeft bij deze koers circa 45% opwaarts potentieel (fair value ~€83 bij WACC 6,4%); pas wie structurele krimp verwacht, vindt de huidige prijs correct. De vraag is niet of de aannames optimistisch genoeg zijn, maar of het pessimisme van de markt terecht is.

### EPV (Bruce Greenwald)
- **Genormaliseerde EBIT-marge %**: 24,4 (mediaan adjusted operating margin 2015-2025 — bewust lager dan de actuele 27,5% om niet op piekmarges te kapitaliseren)
- **Genormaliseerde NOPAT**: 1.142 (6.125 × 24,4% × (1 − 23,6%))
- **Maintenance capex**: 303 (gerapporteerde capex 2025; benadering — D&A ex acquisitie-amortisatie ligt in dezelfde orde: adj. EBITDA 2.007 − adj. EBIT 1.687 = 320)
- **Adjusted earnings power**: 1.142 (maintenance capex ≈ D&A, vallen tegen elkaar weg)
- **EPV per aandeel**: 63 (1.142 / 6,37% = 17.928; minus nettoschuld 3.788 = 14.140; / 224,5 mln aandelen). Bij de niet-opgeslagen componenten-WACC van 5,37% zou de EPV €78 zijn; de conservatieve variant is aangehouden.
- **Groeipremie %**: -9 (koers 57,56 t.o.v. EPV 63: de koers ligt ónder de no-growth-waarde — de markt betaalt momenteel minder dan de waarde van de huidige verdiencapaciteit zonder enige groei, en prijst dus effectief krimp in; consistent met de reverse DCF)

### Andere methoden
- **DDM uitgevoerd?**: false — het dividend (payout ~48%) vangt maar de helft van de kasstroom; FCF-methoden zijn superieur. Ter indicatie: het huidige dividendrendement van 4,4% plus zelfs maar 3% dividendgroei impliceert al een verwacht rendement boven de cost of equity.
- **SOTP uitgevoerd?**: false — WKL is een geïntegreerd bedrijf met gedeelde technologie en klantoverlap, geen conglomeraat; een sum-of-the-parts zou schijnprecisie geven. Wel noemenswaardig: tegen RELX-multiples zouden T&A en FCC alleen al meer waard zijn dan de hele huidige EV.

### Synthese fair value
- **Bandbreedte laag**: 58
- **Bandbreedte centraal**: 170
- **Bandbreedte hoog**: 250
- **Methode-gewichten**:
  - DCF %: 60
  - EPV %: 20
  - Multiples %: 20
- **Margin of safety vereist %**: 25
- **Koopniveau**: 128 (fair value basis 170 × (1 − 0,25))
- **Synthese-toelichting**: De drie invalshoeken vertellen een consistent verhaal met verschillende volumes. De EPV (€63) zegt: zelfs zonder één cent groei is het bedrijf meer waard dan de koers — de groeipremie is negatief, wat bij een bedrijf met tien jaar 7,6% FCF-groei hoogst ongebruikelijk is. De relatieve waardering zegt: tegen 12x de huidige adjusted EBITDA (onder peer-mediaan) hoort hier ~€90, tegen RELX' 15,7x ~€123 — zelfs conservatieve peer-multiples rechtvaardigen de huidige prijs niet. De DCF (€170 basis, €152 kansgewogen) zegt: bij gematigde aannames is de upside een veelvoud. De vereiste margin of safety van 25% — passend bij de betwiste moat-duurzaamheid en de AI-staartrisico's — geeft een koopniveau van €128, ruim boven de huidige koers van €57,56. De waardering is hier niet het discussiepunt; het vertrouwen in de terminale kasstromen is dat wel. Vandaar dat het scorekaart-oordeel (HOLD) strenger is dan de waarderingsconclusie suggereert.

### Gevoeligheid (DCF)
- **FCF-groei ↔ WACC matrix** (fair value in EUR per aandeel; groei = constante 10-jaarsgroei, terminal 2,0% behalve bij negatieve groeirijen waar terminal = groeivoet; basis-FCFF 1.414, nettoschuld 3.788, 224,5 mln aandelen, mid-year):
  - WACC range: [5,4%, 5,9%, 6,4%, 6,9%, 7,4%, 7,9%]
  - Groei range: [-2%, 0%, 2%, 4%, 6%]
  - Matrix:

| Groei ↓ / WACC → | 5,4% | 5,9% | 6,4% | 6,9% | 7,4% | 7,9% |
|---|---|---|---|---|---|---|
| -2% | 68 | 62 | 58 | 54 | 50 | 47 |
| 0% | 101 | 91 | 83 | 76 | 70 | 64 |
| +2% | 173 | 149 | 131 | 116 | 104 | 94 |
| +4% | 209 | 179 | 157 | 139 | 124 | 112 |
| +6% | 250 | 215 | 187 | 166 | 148 | 134 |

De matrix maakt het asymmetrische profiel zichtbaar: de huidige koers (€57,56) correspondeert met de meest sombere hoek (blijvende krimp bij verhoogde WACC), terwijl elke niet-negatieve groeiaanname bij elke redelijke WACC een fair value van €64 of (veel) hoger geeft.

---

## 13. Databronnen

### Bronnen-hiërarchie
- **Jaarverslag PDF / IR-pagina / officieel persbericht** → betrouwbaarheid **HOOG**
- **Beursmelding / prospectus** → betrouwbaarheid **HOOG**
- **Aggregator** (MacroTrends / StockAnalysis / Yahoo / Trading Economics / InsiderScreener) → betrouwbaarheid **AGGREGATOR**

### Financiële bronnen (10+ jaar historie)

| Jaar | Bron | URL | Betrouwbaarheid (HOOG/AGGREGATOR) |
|---|---|---|---|
| 2015 | Wolters Kluwer 2016 Full-Year Report (vergelijkende kolom) | https://www.globenewswire.com/news-release/2017/02/22/926171/0/en/Wolters-Kluwer-2016-Full-Year-Report.html | HOOG |
| 2016 | Wolters Kluwer 2016 Full-Year Report | https://www.globenewswire.com/news-release/2017/02/22/926171/0/en/Wolters-Kluwer-2016-Full-Year-Report.html | HOOG |
| 2017 | Wolters Kluwer 2018 Full-Year Report (vergelijkende kolom, IFRS 15-restated) | https://www.globenewswire.com/news-release/2019/02/20/1738056/0/en/Wolters-Kluwer-2018-Full-Year-Report.html | HOOG |
| 2018 | Wolters Kluwer 2018 Full-Year Report (+ IFRS 16-restatement uit 2019 Full-Year Report) | https://www.globenewswire.com/news-release/2019/02/20/1738056/0/en/Wolters-Kluwer-2018-Full-Year-Report.html | HOOG |
| 2019 | Wolters Kluwer 2019/2020 Full-Year Reports | https://www.globenewswire.com/news-release/2020/02/26/1990636/0/en/Wolters-Kluwer-2019-Full-Year-Report.html | HOOG |
| 2020 | Wolters Kluwer 2020 Full-Year Report | https://www.globenewswire.com/news-release/2021/02/24/2181083/0/en/Wolters-Kluwer-2020-Full-Year-Report.html | HOOG |
| 2021 | Wolters Kluwer 2022 Full-Year Report (vergelijkende kolom) | https://www.globenewswire.com/en/news-release/2023/02/22/2612844/0/en/Wolters-Kluwer-2022-Full-Year-Report.html | HOOG |
| 2022 | Wolters Kluwer 2022 Full-Year Report | https://www.globenewswire.com/en/news-release/2023/02/22/2612844/0/en/Wolters-Kluwer-2022-Full-Year-Report.html | HOOG |
| 2023 | Wolters Kluwer 2024 Full-Year Report (vergelijkende kolom) | https://www.globenewswire.com/news-release/2025/02/26/3032585/0/en/Wolters-Kluwer-2024-Full-Year-Report.html | HOOG |
| 2024 | Wolters Kluwer 2024 Full-Year Report | https://www.globenewswire.com/news-release/2025/02/26/3032585/0/en/Wolters-Kluwer-2024-Full-Year-Report.html | HOOG |
| 2025 | Wolters Kluwer 2025 Full-Year Report | https://www.globenewswire.com/news-release/2026/02/25/3244280/0/en/Wolters-Kluwer-2025-Full-Year-Report.html | HOOG |

Harde eis voldaan: de vijf meest recente jaren (2021-2025) zijn allemaal HOOG. Aanvullende AGGREGATOR-cellen (EBITDA 2021-2022, eigen vermogen 2021-2022, DPS 2021) zijn in de tabellen expliciet gelabeld.

### Jaarverslagen / officiële rapporten geraadpleegd

| Jaar | Bron | URL |
|---|---|---|
| 2025 | 2025 Full-Year Report persbericht (25-02-2026) | https://www.globenewswire.com/news-release/2026/02/25/3244280/0/en/Wolters-Kluwer-2025-Full-Year-Report.html |
| 2024 | 2024 Full-Year Report persbericht (26-02-2025) | https://www.globenewswire.com/news-release/2025/02/26/3032585/0/en/Wolters-Kluwer-2024-Full-Year-Report.html |
| 2022 | 2022 Full-Year Report persbericht (22-02-2023) | https://www.globenewswire.com/en/news-release/2023/02/22/2612844/0/en/Wolters-Kluwer-2022-Full-Year-Report.html |
| 2020 | 2020 Full-Year Report persbericht (24-02-2021) | https://www.globenewswire.com/news-release/2021/02/24/2181083/0/en/Wolters-Kluwer-2020-Full-Year-Report.html |
| 2019 | 2019 Full-Year Report persbericht (26-02-2020) | https://www.globenewswire.com/news-release/2020/02/26/1990636/0/en/Wolters-Kluwer-2019-Full-Year-Report.html |
| 2018 | 2018 Full-Year Report persbericht (20-02-2019) | https://www.globenewswire.com/news-release/2019/02/20/1738056/0/en/Wolters-Kluwer-2018-Full-Year-Report.html |
| 2016 | 2016 Full-Year Report persbericht (22-02-2017) | https://www.globenewswire.com/news-release/2017/02/22/926171/0/en/Wolters-Kluwer-2016-Full-Year-Report.html |

### Beursmeldingen geraadpleegd

| Datum | Omschrijving | URL |
|---|---|---|
| 2026-05-06 | First-Quarter 2026 Trading Update (guidance herbevestigd) | https://www.globenewswire.com/news-release/2026/05/06/3288504/0/en/wolters-kluwer-first-quarter-2026-trading-update.html |
| 2026-07-02 | Share Buyback Transaction Details June 25 – July 1, 2026 | https://www.globenewswire.com/news-release/2026/07/02/3321127/0/en/Share-Buyback-Transaction-Details-June-25-July-1-2026.html |
| 2025-02-26 | CEO-opvolging: McKinstry met pensioen, Caywood genomineerd | https://www.globenewswire.com/news-release/2025/02/26/3032584/0/en/Wolter-Kluwer-CEO-Nancy-McKinstry-announces-retirement-in-early-2026-Stacey-Caywood-nominated-as-successor.html |

### IPO-prospectus
- **Geraadpleegd?**: false (notering dateert van 1987; niet relevant en niet digitaal beschikbaar)
- **URL**: —
- **Pre-IPO data beschikbaar?**: false (niet van toepassing: beursnotering ~39 jaar)
- **Pre-IPO bron**: —

### Non-GAAP
- **Gebruikt?**: true
- **Toelichting**: De analyse gebruikt WKL's "benchmark"-cijfers (adjusted operating profit/marge, adjusted net profit, adjusted EPS, adjusted FCF) voor trends, ratio's en de DCF-basis. Uitsluitingen betreffen amortisatie van geacquireerde immateriële activa, divestment-boekwinsten (FRR: €232 mln in 2025) en eenmalige posten (pensioenbate €27 mln in 2024). Rechtvaardiging: acquisitie-amortisatie is non-cash en vertekent de verdienkrachttrend bij een seriële acquirer; de FRR-boekwinst is werkelijk eenmalig. IFRS-cijfers zijn in alle tabellen naast de adjusted reeks gerapporteerd zodat de brug zichtbaar blijft. De DCF-basis (adjusted FCF, omgerekend naar FCFF met rente na belasting) is een kasstroommaatstaf en daarmee het minst gevoelig voor non-GAAP-keuzes.

### Ontbrekende data (eerlijke lijst)
- SBC (stock-based compensation) per jaar: niet apart vermeld in de full-year-persberichten; vereist de jaarverslag-PDF's. Gevolg: kolom SBC leeg, geen expliciete FCF-na-SBC-reeks. Mitigatie: aandelenaantal daalde 10 jaar op rij, dus netto-verwatering is negatief.
- Eigen vermogen en totale activa 2015-2020 in EUR: niet in de geopende persberichten; MacroTrends heeft alleen USD (ADR) — bewust niet gemengd in de EUR-tabellen.
- Current ratio / quick ratio: geen volledige balansdetaillering in de persberichten.
- Verwaterde aandelenaantallen 2015-2016 (alleen procentuele mutaties vermeld); capex 2015 in € (alleen "4,5% van omzet"); buyback-bedrag 2015.
- Remuneratiedetails (bonus-KPI's, LTI-structuur, CEO pay ratio, exacte SBC % marktkap): remuneratierapport niet doorgenomen.
- Exacte volumes van de insider-aankopen De Kreij en Vogelzang (feb 2026): bronpagina bevat parsingfouten; richting (koop) wel zeker.
- Marktaandelen per niche, formele TAM/SAM-cijfers, churn/netto-retentie: geen verifieerbare bron — leeggelaten.
- Recurring % en cloud % van omzet vóór 2025: niet in de geopende persberichten teruggevonden.
- Damodaran ERP-stand juli 2026 (gebruikt: jan 2026); personeel/landen-tal alleen van bedrijfswebsite (about-us), niet uit jaarverslag geverifieerd.

### Peildatum analyse
- 2026-07-02 (koers €57,56, slot 2 juli 2026)

---

## 14. Volledige bronnen-lijst

| Titel | URL | Type |
|---|---|---|
| Wolters Kluwer 2025 Full-Year Report | https://www.globenewswire.com/news-release/2026/02/25/3244280/0/en/Wolters-Kluwer-2025-Full-Year-Report.html | jaarverslag |
| Wolters Kluwer 2024 Full-Year Report | https://www.globenewswire.com/news-release/2025/02/26/3032585/0/en/Wolters-Kluwer-2024-Full-Year-Report.html | jaarverslag |
| Wolters Kluwer 2022 Full-Year Report | https://www.globenewswire.com/en/news-release/2023/02/22/2612844/0/en/Wolters-Kluwer-2022-Full-Year-Report.html | jaarverslag |
| Wolters Kluwer 2020 Full-Year Report | https://www.globenewswire.com/news-release/2021/02/24/2181083/0/en/Wolters-Kluwer-2020-Full-Year-Report.html | jaarverslag |
| Wolters Kluwer 2019 Full-Year Report | https://www.globenewswire.com/news-release/2020/02/26/1990636/0/en/Wolters-Kluwer-2019-Full-Year-Report.html | jaarverslag |
| Wolters Kluwer 2018 Full-Year Report | https://www.globenewswire.com/news-release/2019/02/20/1738056/0/en/Wolters-Kluwer-2018-Full-Year-Report.html | jaarverslag |
| Wolters Kluwer 2016 Full-Year Report | https://www.globenewswire.com/news-release/2017/02/22/926171/0/en/Wolters-Kluwer-2016-Full-Year-Report.html | jaarverslag |
| Wolters Kluwer Q1 2026 Trading Update | https://www.globenewswire.com/news-release/2026/05/06/3288504/0/en/wolters-kluwer-first-quarter-2026-trading-update.html | beursmelding |
| Share Buyback Details 25 jun – 1 jul 2026 | https://www.globenewswire.com/news-release/2026/07/02/3321127/0/en/Share-Buyback-Transaction-Details-June-25-July-1-2026.html | beursmelding |
| CEO-opvolging McKinstry → Caywood | https://www.globenewswire.com/news-release/2025/02/26/3032584/0/en/Wolter-Kluwer-CEO-Nancy-McKinstry-announces-retirement-in-early-2026-Stacey-Caywood-nominated-as-successor.html | beursmelding |
| Yahoo Finance WKL.AS (koers) | https://finance.yahoo.com/quote/WKL.AS/ | aggregator |
| StockAnalysis.com WKL statistieken/financials | https://stockanalysis.com/quote/ams/WKL/statistics/ | aggregator |
| Trading Economics — NL 10Y yield | https://tradingeconomics.com/netherlands/government-bond-yield | aggregator |
| Damodaran datacurrent (ERP jan 2026) | https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datacurrent.html | onderzoeksrapport |
| Damodaran betaEurope.xls (industrie-betas jan 2026) | https://pages.stern.nyu.edu/~adamodar/pc/datasets/betaEurope.xls | onderzoeksrapport |
| Morningstar — moat-downgrade WKL/TRI naar Narrow | https://www.morningstar.com/stocks/downgrading-wolters-kluwer-thomson-reuters-moats-narrow-ai-disruption-potential | analistenrapport |
| Morningstar — WKL in-line growth, shares undervalued | https://www.morningstar.com/company-reports/1480514-wolters-kluwer-in-line-growth-not-enough-to-quell-markets-ai-fears-shares-undervalued | analistenrapport |
| Investing.com — WKL -12% op AI-concurrentievrees | https://ng.investing.com/news/earnings/wolters-kluwer-shares-drop-over-12-on-ai-competition-concerns-93CH-2484380 | nieuwsartikel |
| Stockopedia/Reuters — Thomson Reuters slumps on legal-AI worries | https://www.stockopedia.com/share-prices/wolters-kluwer-nv-AMS:WKL/news/thomson-reuters-slumps-amid-worries-over-legal-ai-disruption-019e6a09-54aa-78f2-be05-6f74c1b2e9cd/ | nieuwsartikel |
| InsiderScreener — WKL insider-transacties | https://www.insiderscreener.com/en/company/wolters-kluwer-nv | aggregator |
| TipRanks — WKL analistenconsensus | https://www.tipranks.com/stocks/nl:wkl/forecast | aggregator |
| MarketScreener — WKL consensus | https://www.marketscreener.com/quote/stock/WOLTERS-KLUWER-N-V-6291/consensus/ | aggregator |
| StockAnalysis — RELX / TRI / SSNC / CLVT statistieken | https://stockanalysis.com/stocks/relx/statistics/ | aggregator |
| RELX FY2025 results persbericht | https://www.relx.com/~/media/Files/R/RELX-Group/documents/press-releases/2026/results-2025-pressrelease.pdf | jaarverslag |
| Wolters Kluwer — Our heritage (geschiedenis) | https://www.wolterskluwer.com/en/about-us/our-heritage | IR-pagina |
| Wikipedia — Wolters Kluwer (geschiedenis 1836/1889/1987, kruisverificatie) | https://en.wikipedia.org/wiki/Wolters_Kluwer | aggregator |
| Stocksguide — WKL dividendkalender | https://stocksguide.com/en/dividends/Wolters-Kluwer-NL0000395903 | aggregator |

---

## 15. Update-historie

| Datum | Versie | Wijziging |
|---|---|---|
| 2026-04-28 | 1.0 | Eerste publicatie (koers €68; historie 2015-2019 ontbrak) |
| 2026-07-02 | 2.0 | Volledige herbouw: peildatum en koers geactualiseerd (€57,56), financiële historie 2015-2025 volledig HOOG-gedekt via zeven officiële full-year-persberichten, WACC/DCF/EPV herrekend met Damodaran-sectorbeta en expliciete AI-risico-opslag, insider-transacties en Q1 2026 verwerkt, scorekaart herrekend (31/45, HOLD) |

---

## Afronding (check voor oplevering)

- [x] Elk cijfer in de tabellen heeft een bron in de bronnen-inventaris of staat op "—"
- [x] De recente 5 jaren in sectie 13 (2021-2025) zijn allemaal HOOG
- [x] Geen enum-variant verzonnen — KOOP/HOLD/PASS, NARROW MOAT, STERK, LAAG/MIDDEN/HOOG, KLEIN/MIDDEL/GROOT, POSITIEF/NEGATIEF/NEUTRAAL/BINAIR, Pessimistisch/Basis/Optimistisch, HOOG/AGGREGATOR conform template
- [x] Scorekaart heeft 9 frameworks (1+3+3+4+3+3+4+5+5 = 31, max 45), eindoordeel volgt deterministische drempels (31 < 33 → HOLD)
- [x] Synthese-toelichting aanwezig (sectie 12)
- [x] Non-GAAP adjustments expliciet toegelicht (sectie 3 en 13)
- [x] IPO-carve-out n.v.t. (notering 1987) — expliciet behandeld in risico 8 en sectie 13

---

## Opmerkingen voor Claude Code

1. **Consistentie-ankers:** koers 57,56 (peildatum 2026-07-02); fair value basis 170; kansgewogen 152; EPV per aandeel 63; upside basis 195,3%; scenario's 58/170/250 met kansen 30/50/20; scorekaart 1,3,3,4,3,3,4,5,5 = totaal 31/45 → HOLD; koopniveau 128; MOS 25%; reverse-DCF impliciete groei -2.
2. **WACC-structuur vergt aandacht bij JSON-injectie:** de componenten (Rf 3,03 + beta 0,82 × ERP 4,23 → Ke 6,50; Kd na tax 2,44; E/V 72,21 / D/V 27,79) geven wacc_pct 5,37, maar alle scenario's gebruiken een expliciete opslag (basis 6,37 = 5,37 + 1,00). Als de validator WACC herrekent uit componenten en vergelijkt met scenario-WACC's, is het verschil de gedocumenteerde `wacc_adj` per scenario (+2,0 / +1,0 / +0,5). EPV is berekend op 6,37 (niet 5,37) — bewust conservatief; bij 5,37 zou EPV 78 zijn.
3. **EPV-scriptafwijking:** dcf_calculator.py gaf EPV 78 (rekent op componenten-WACC 5,37); in het rapport staat 63 (op basis-WACC 6,37) omwille van interne consistentie met de DCF. Keuze documenteren of script-waarde herstellen naar smaak van de validator — beide zijn uit dezelfde inputs reproduceerbaar.
4. **Reverse-DCF-conventie:** script gaf 0,0% met eigen conventie; gerapporteerd is -2% met conventie "constante groei 10 jaar én terminal gelijk aan die groei, WACC 6,37, mid-year". Bij herberekening deze conventie aanhouden.
5. **P/E-keuze:** trailing P/E 10,9 op adjusted EPS 5,29; op IFRS EPS 5,64 zou het 10,2 zijn (vertekend door FRR-boekwinst). Graham-score verandert er niet door (P/B 16,3 is bepalend).
6. **Magic-Formula-ROC:** gerapporteerde ROIC 18,0 gebruikt als conservatieve proxy (score 3). Greenblatt-zuivere ROC (excl. goodwill, negatief werkkapitaal) zou hoger zijn maar was niet berekenbaar uit de geopende bronnen — bewuste keuze, geen omissie.
7. **SBC ontbreekt** in alle persberichten; `fcf_na_sbc` moet null blijven tenzij stage 2 de jaarverslag-PDF's opent. Zelfde geldt voor current ratio, EV 2015-2020 in EUR en remuneratiedetails.
8. **Insider-data:** twee transacties (De Kreij, Vogelzang, feb 2026) hebben corrupte volumes in de bron (InsiderScreener parsing); alleen richting en datum als vaststaand behandelen. AFM-register is niet scriptbaar-fetchbaar.
9. **Divisiestructuur-breuk:** FRR-transfer maakt 2023-divisiecijfers (oude structuur) niet 1-op-1 vergelijkbaar met 2024-herzien/2025 (FCC 1.228 en CP&ESG 597 in de herziene 2024-kolom van het 2025-rapport). In segmententabellen alleen 2025 gebruikt.
10. **Vorige versie:** dit bestand vervangt de analyse van 2026-04-28 (koers €68, oordeel HOLD, FV basis 72) volledig. De fair value is fors hoger dan toen omdat (a) de historie nu 11 jaar HOOG-gedekt is waardoor de FCF-basis en groei beter onderbouwd zijn, (b) de FCFF-conversie correct is toegepast, en (c) de koersdaling de reverse-DCF-asymmetrie heeft vergroot. Het scorekaart-oordeel blijft HOLD door de deterministische drempels (Graham 1 weegt zwaar).
11. **Geen wijzigingen aangebracht** buiten research/WKL.md. Geen observaties van defecten in platform/ of docs/ (niet bekeken, conform instructie).
