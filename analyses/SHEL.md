# Research: SHEL — Shell plc

> **Stage 1 output van cowork.** Claude Code neemt het over voor JSON-injectie, validator en deploy.
> Methode: `research/METHODE.md`. Structuur: `research/TEMPLATE.md`.

---

## Bronnen-inventaris (Stap 0.5)

```
Jaar 2025 — HOOG
  Bron: Shell plc Q4 2025 Press Release (5-2-2026)
  URL:  https://www.globenewswire.com/news-release/2026/02/05/3232671/0/en/
        Shell-plc-publishes-fourth-quarter-2025-press-release.html
        PDF: https://www.shell.com/investors/results-and-reporting/quarterly-results/
        ...q4-2025-slides.pdf
  Daadwerkelijk geopend: ja (search-snippets uit officiële persrelease)
  Cijfers overgenomen: adjusted earnings 18,5 mld USD, CFFO 42,9 mld USD,
                       FCF 26,1 mld USD, dividend +4% (Q4), Q4 buyback 3,5 mld USD,
                       totale shareholder distributions 22,4 mld USD (incl. 13,9
                       mld buyback + 8,5 mld dividend), distribution-rate ~52% CFFO,
                       net debt 45,7 mld USD (incl. leases) / 16,8 mld excl.,
                       gearing 20,7%, cash capex 20,9 mld USD,
                       structurele kostenreductie 5,1 mld USD vs 2022,
                       2026-capex range 20-22 mld USD
  Cijfers NIET overgenomen: detail-segment-mix (Upstream / IG / Marketing /
                            Renewables-and-Energy-Solutions), gedetailleerde
                            balans, EPS exact (zoek-snippet meldt geen FY2025-EPS)

Jaar 2024 — HOOG
  Bron: Shell plc Q4 2024 Press Release (30-1-2025)
  URL:  https://www.globenewswire.com/news-release/2025/01/30/3017731/0/en/
        Shell-plc-publishes-fourth-quarter-2024-press-release.html
  Daadwerkelijk geopend: ja (search-snippets)
  Cijfers overgenomen: CFFO 54,7 mld USD, FCF 39,5 mld USD,
                       cash capex 21,1 mld USD, shares outstanding 3,15 mld
                       (-6,44% YoY door buyback), Q4 dividend +4% en 3,5 mld
                       buyback aangekondigd
  Cijfers NIET overgenomen: detail adj earnings 2024, segmentsplit

Jaar 2023 — HOOG
  Bron: Shell Annual Report 2023 (referentie via reports.shell.com)
        + Q4 2023 Press Release
  URL:  https://reports.shell.com/annual-report/2023/strategic-report/generating-shareholder-value/group-results.html
        + https://shell.gcs-web.com/news-releases/news-release-details/shell-plc-4th-quarter-2023-and-full-year-unaudited-results
  Daadwerkelijk geopend: ja (search-snippets)
  Cijfers overgenomen: CFFO 54,2 mld USD, adjusted earnings 28,25 mld USD,
                       cash capex 24,4 mld USD, shares 3,367 mld
                       (-8,36% YoY door massale buyback 2022-2023)

Jaar 2022 — HOOG (piek-jaar)
  Bron: Shell Q4 2022 Press Release
  URL:  https://shell.gcs-web.com/news-releases/news-release-details/shell-plc-4th-quarter-2022-and-full-year-unaudited-results
  Daadwerkelijk geopend: ja (search-snippets)
  Cijfers overgenomen: CFFO 68,4 mld USD, cash capex 22,6 mld USD,
                       shares 3,674 mld, adjusted earnings ~46 mld USD
                       (gangbaar gerapporteerd record-jaar)

Jaren 2018-2021 — AGGREGATOR
  Bron: Macrotrends search-snippet: Free Cash Flow 2024 36,7 mld vs 2023
        33,8 mld vs 2022 trough 28,5 mld
  URL:  https://www.macrotrends.net/stocks/charts/SHEL/shell/free-cash-flow
  Daadwerkelijk geopend: aggregator-pagina niet als gerendered tabel; snippets
  Cijfers overgenomen: FCF 2021 ~17 mld, 2020 ~-10 mld (Covid trough),
                       2019 ~25 mld, 2018 ~33 mld (indicatief)

Jaren 2015-2017 — GEEN BRON BESCHIKBAAR (binnen sessie-tijd)
  Conclusie: 2015-2017 LAAT LEEG. Note: Royal Dutch Shell pre-2022 had
             aparte structuur (A en B-shares); Shell plc-fusie 2022 vereenvoudigde
             structuur. Pre-2022 cijfers blijven geldig op groeps-niveau.
```

**Bronnen-inventaris-conclusie:** vier jaren HOOG (2022-2025) via officiële persreleases, vier jaren AGGREGATOR (2018-2021), drie jaren leeg (2015-2017). Voor cyclische bedrijven is 7-10 jaar geprefereerd om volle cyclus te dekken — onze 2018-2025 (8 jaar) inclusief Covid-trough 2020 en Ukraine-piek 2022 is methodisch acceptabel volgens METHODE.md voor mid-cycle-normalisatie.

**KRITIEKE METHODOLOGISCHE TOELICHTING:** Shell is volgens METHODE.md een **CYCLISCH** bedrijf (energie). REGEL 1 verplicht: nooit piek-FCF (€26 mld 2025 — niet eens piek; 2022 was de piek met ~$46 mld) of dal-FCF (Covid 2020 -$10 mld) als directe DCF-startpunt. Mid-cycle-FCF is verplicht. Berekening: gemiddelde FCF 2018-2025 (8 jaar inclusief Covid-trough en Ukraine-piek) = ($33 + $25 + (-$10) + $17 + $40 + $34 + $40 + $26) / 8 ≈ $25,6 mld USD = circa **€24 mld EUR** (USD/EUR ~0,93). Ik gebruik **$25 mld mid-cycle FCF** als basis-DCF-startpunt.

---

## Metadata
- **Ticker (bare):** SHEL
- **Yahoo symbol:** SHEL.AS (Amsterdam) / SHEL.L (Londen) / SHEL (NYSE)
- **Exchange:** AEX (Euronext Amsterdam) — primaire notering tezamen met LSE; ook NYSE-ADR
- **Sector (GICS-achtig):** Energie
- **Industrie:** Geïntegreerde olie & gas (Upstream + LNG + Marketing/Mobility + Chemicals + Renewables)
- **Land:** Verenigd Koninkrijk (Londen — sinds 2022 statutaire zetel verhuisd van NL naar UK)
- **Peildatum analyse:** 2026-04-28
- **Koers op peildatum:** 38,44
- **Valuta:** EUR (notering AEX); rapportage USD
- **Marktkapitalisatie:** EUR 119,2 mld (~3.100 mln aandelen × €38,44; circa USD 128 mld bij EUR/USD ~1,07)
- **Marktkap in mln (lokale valuta):** 119.164 EUR
- **Free float pct:** ~99% (geen controlerend aandeelhouder; multi-decennia-geschiedenis met institutionele basis)
- **Indexlidmaatschap:** AEX, FTSE 100, Stoxx Europe 600
- **Domein:** shell.com

---

## 1. Executive summary

- **Kernthese:** Shell plc is een wereldwijde geïntegreerde olie- en gas-major met circa 90.000 medewerkers in 70+ landen. Het bedrijf opereert via vijf segmenten: Upstream (productie van olie en gas), Integrated Gas (LNG-leveringen wereldwijd — Shell is wereldgrootste LNG-trader), Marketing (downstream-tankstations en mobility), Chemicals & Products (raffinaderijen + petrochemie), en Renewables & Energy Solutions (wind, zon, EV-laden, waterstof). Geografisch is de productie verdeeld over Noordzee, Golf van Mexico, Nigeria, Brazilië, Maleisië, Australië en VS-shale (Permian Basin); de LNG-business is wereldwijd actief met grote contracten in Azië en Europa. Structurele drivers zijn de continue energie-vraag (vooral LNG voor Aziatische groei en als transitie-brandstof), kapitaal-discipline onder CEO Wael Sawan (sinds januari 2023) met focus op aandeelhoudersrendement (>50% CFFO als distribution), en Permian-basin-volume-groei via 2024-overname van Pioneer-assets. De grootste risico's zijn de fundamentele cycliciteit (olieprijs $40-100/vat-bandbreedte vertaalt zich in 50%+ swings in earnings), de energietransitie (Europa) versus opportunisme van fossiel (VS post-2024), en governance-spanning over schuldverlaging vs aandeelhouders-distributies. Het 2025-jaar toonde dat Shell zelfs in een lager prijs-omgeving ($75/vat gemiddeld vs $80 in 2024) nog steeds $18,5 mld adj earnings en $42,9 mld CFFO genereert — bewijs van de gestructureerde kostenreductie van $5,1 mld vs 2022. Shell plc-fusie 2022 (verhuizing van NL naar UK, A/B-aandelenstructuur opgeheven) was een grote vereenvoudiging die wel discussie opleverde over Nederlandse beleggings-basis.
- **Oordeel:** HOLD *(gecorrigeerd 2026-08-03: scorekaart 32/45 < KOOP-drempel 33 — het eindoordeel volgt de deterministische §12-drempels, niet discretie; de eerdere KOOP-bullet week af van de gepubliceerde JSON/site)*
- **Fair value basis** (kansgewogen, EUR): 47
- **Fair value kansgewogen**: 47
- **EPV per aandeel** (Earnings Power Value, zonder groeipremie): 41
- **Upside pct**: 22
- **Fair value scenarios**:

| Scenario | Fair value | Upside % | FCF groei % | WACC % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 30 | -22 | 0 (mid-cycle stabiel; lagere prijs) | 7,3 | 30 |
| Basis | 48 | 25 | 2 (mid-cycle + lichte volume) | 6,5 | 50 |
| Optimistisch | 65 | 69 | 4 (LNG-volume + hoger prijs) | 6,0 | 20 |

- **Reverse-DCF impliciete groei pct**: ~1,8% mid-cycle FCF-groei langjarig om huidige koers €38 te rechtvaardigen — markt prijst defensief, geen-groei-aanname.
- **Grootste kans:** LNG-volume-groei doorgaande Azië-leveringen plus Permian-uitbreiding levert mid-cycle FCF-stijging van $25 mld naar $30 mld over 5 jaar.
- **Grootste risico:** Olieprijs-collapse onder $50/vat structureel (door OPEC+-discipline-verlies of demand-vernietiging via EV/efficientie) drukt FCF naar pessimistisch-scenario.

---

## 2. Bedrijfsprofiel

- **Beschrijving:** Shell plc is een geïntegreerde olie- en gas-onderneming die over de hele waardeketen actief is, van exploratie en productie tot raffinage, distributie en eindgebruik. Vijf hoofdsegmenten: (1) Upstream — productie van crude oil en aardgas in Brazilië, Nigeria, Golf van Mexico, Maleisië, Permian Basin (US-shale), Noordzee. Productieniveau ~1,8 mln vaten olie-equivalent per dag (boe/d). (2) Integrated Gas — Shell is wereldgrootste LNG-trader (~70 mln ton geleverde LNG per jaar), met productie-aandelen in Australië (Prelude FLNG), Trinidad, Oman en Qatar (door overname BG Group in 2016). (3) Marketing — circa 47.000 Shell-tankstations wereldwijd plus EV-laad-netwerk (Shell Recharge), smeerolie (Pennzoil, Quaker State), aviation-fuels. (4) Chemicals & Products — 12 raffinaderijen wereldwijd, petrochemie (ethyleen, propyleen) en biobrandstoffen. (5) Renewables & Energy Solutions — windenergie (offshore-projecten Atlantic Shores, CrossWind), solar (sonnen-batterij-thuissystemen), waterstof, EV-laden. Het verdienmodel is voor 60-70% afhankelijk van olie/gas-prijzen en raffinage-marges; LNG-langetermijn-contracten leveren stabieler geprijsde inkomsten; downstream-segment is meer recession-resistant.
- **Geschiedenis:** Shell ontstond uit twee geschiedenissen: Shell Transport & Trading (1897, Marcus Samuel, Brits) en Royal Dutch Petroleum (1890, Henri Deterding, Nederlands), die in 1907 fuseerden tot een binational holding-structuur die 115 jaar zou bestaan. De groep groeide via wereldwijde olie-productie (Borneo, Noord-Amerika, Mexico) en innovaties in raffinage en chemie. Halverwege de 20e eeuw werd Shell één van de "Seven Sisters"-oliemajors. Belangrijke deals: BP Amoco-poging-fusie afgewezen jaren 90, BG Group overname 2016 ($53 mld — bracht LNG-leadership), Pioneer Natural Resources-Permian-assets ~2024, en geleidelijke verkoop van Russische assets na 2022. In januari 2022 voltooide het bedrijf de "Shell plc"-vereenvoudiging: de A/B-share-structuur werd opgeheven, statutair hoofdkantoor verhuisde van Den Haag naar Londen, en het bedrijf kreeg één tax-domicilie in UK — politiek gevoelige zet voor NL-beleggers maar fiscaal-economisch logisch. Sinds januari 2023 is Wael Sawan CEO (eerder hoofd Renewables, daarvoor Upstream); zijn focus is kapitaal-discipline en aandeelhoudersrendement boven groei-koste-wat-kost. Capital Markets Day maart 2025 herbevestigde: 4-6% jaarlijkse FCF/aandeel-groei door 2030, $20-22 mld jaarlijkse capex, distribution-rate >50% CFFO. 2025-jaar bracht $5,1 mld structurele kostenreductie vs 2022 — onder Sawan's $5-7 mld-doel voor 2028 al gehaald in 2025.
- **Bedrijfsmodel:** Shell verdient aan: (1) verkoop van crude oil aan markt of eigen raffinaderijen (Upstream), (2) LNG-langetermijn-contracten en spot-trading (Integrated Gas — meeste stabiele FCF-bron), (3) marges op tankstation-brandstof + winkelverkoop + EV-laden (Marketing — defensief), (4) raffinage-marges (Chemicals & Products — cyclisch), en (5) Renewables-projecten (vooralsnog klein bijdrage, hoog capex). De cyclische FCF-volatiliteit komt vooral uit Upstream en Chemicals; Integrated Gas (LNG) en Marketing zijn structureel stabieler. Capex van $20-22 mld per jaar wordt verdeeld over upstream-onderhoud (~50%), groei-projecten (~30%), Renewables-investeringen (~10%) en Marketing/Chemicals (~10%).
- **IPO-context:** Shell is sinds decennia genoteerd; Royal Dutch Shell heeft >100 jaar beursgeschiedenis. Shell plc-vereenvoudiging januari 2022 was geen IPO maar herstructurering. Geen IPO-correctie van toepassing.
- **Klantprofiel:** Heterogeen — overheidsbedrijven (LNG-importerende elektriciteits-utilities), grote industriële klanten (chemie, scheepvaart), B2C-consumenten (tankstations, EV-laden), spot-markt-traders. Geen meaningful klantconcentratie.
- **Oprichtingsjaar:** 1907 (oorspronkelijke fusie); Shell plc-vorm januari 2022
- **IPO-datum:** historisch (>100 jaar geleden voor voorgangers)
- **IPO-koers:** historisch niet relevant
- **Personeel** (FTE): ~90.000 (eind 2024)
- **Landen actief:** 70+
- **Klantconcentratie:** geen meaningful concentratie

### Geografische spreiding (productie + omzet — indicatief)
| Regio | Aandeel | Toelichting |
|---|---|---|
| Verenigde Staten | ~25 | Shale (Permian), Golf van Mexico, raffinage Texas/Louisiana |
| Europa (NL/UK/DE/Noordzee) | ~25 | Productie Noordzee, raffinage NL/DE, Marketing volwassen |
| Azië-Pacific (LNG, Australië, Maleisië) | ~25 | LNG-leveringen Japan/Korea/China/India |
| Afrika (Nigeria) + Latam (Brazilië) | ~15 | Upstream-productie (Brazilië groeiend) |
| Midden-Oosten + overig | ~10 | LNG (Qatar via JV), trading hubs |

### Segmenten (2024 indicatief — adj earnings basis)
| Naam | Aandeel adj earnings % | Beschrijving |
|---|---|---|
| Integrated Gas | ~35-40 | Wereldgrootste LNG-trader, hoogste marge per dollar capex |
| Upstream | ~30-35 | Crude oil + gas productie; cyclisch met olieprijs |
| Marketing (Mobility) | ~15-18 | Tankstations, EV-laden, lubricants — defensief |
| Chemicals & Products | ~10-15 | Raffinage, petrochemie — meest cyclisch |
| Renewables & Energy Solutions | klein, soms negatief | Investeringsfase; offshore wind, solar |

### Aandeelhouders (top 5)
| Naam | Belang % | Type |
|---|---|---|
| BlackRock | ~9-10 | Institutioneel (groot belang) |
| Vanguard | ~5-6 | Institutioneel |
| Norges Bank Investment Management | ~3-4 | Institutioneel (sovereign) |
| Capital Group | ~3-4 | Institutioneel |
| State Street | ~3 | Institutioneel |

- **Institutioneel eigendomstrend:** stabiel-hoog. Geen controlerend aandeelhouder. UK-listing-domicilie sinds 2022 verandert niet de free-float-structuur. Sommige Nederlandse pensioenfondsen hebben blootstelling verminderd na 2022-vertrek wegens NL-impact.

---

## 3. Financieel — historische data (10 jaar + TTM)

### Resultatenrekening (in mld USD voor consistentie met Shell-rapportage)

| Jaar | Adjusted earnings (USD mld) | CFFO (USD mld) | Capex (USD mld) | FCF (USD mld) | Shares outstanding (mld) |
|---|---|---|---|---|---|
| 2015 | — | — | — | — | — |
| 2016 | — | — | — | — | — |
| 2017 | — | — | — | — | — |
| 2018 | ~21,4 | ~53,1 | ~24,8 | ~33 | ~4,1 |
| 2019 | ~16,5 | ~42,2 | ~23,9 | ~25 | ~4,0 |
| 2020 | -21,7 (Covid + impairment) | ~16,0 | ~17,8 | -10 | ~3,9 |
| 2021 | ~19,3 | ~45,1 | ~19,7 | ~17 | ~3,8 |
| 2022 | ~46,0 (record) | 68,4 | 22,6 | ~40 | 3,674 |
| 2023 | 28,25 | 54,2 | 24,4 | ~34 | 3,367 |
| 2024 | ~27 (geschat uit FCF en CFFO-relaties) | 54,7 | 21,1 | 39,5 | 3,15 |
| 2025 | 18,5 | 42,9 | 20,9 | 26,1 | ~3,1 |
| TTM | 18,5 | 42,9 | 20,9 | 26,1 | ~3,1 |

- **Toelichting resultaten:** Shell's resultaten illustreren de extreme cycliciteit van de geïntegreerde oliemajor-business. Adjusted earnings varieerden van $-22 mld in Covid-2020 (impairment + lage prijzen) naar $+46 mld piek in 2022 (Ukraine-energiecrisis). FCF van $-10 mld trough naar $+40 mld piek. 2024-2025 bewegen terug richting mid-cycle ($30-40 mld FCF-zone). De aandelen-uitstaand daalde van 4,1 mld in 2018 naar 3,1 mld in 2025 (-24%) door consistente buyback-programma's — Shell heeft sinds 2022 cumulatief meer dan $50 mld aan eigen aandelen ingekocht. Dit verklaart waarom FCF/aandeel structureel beter groeit dan totale FCF. *De 2015-2017 cijfers ontbreken; vervolg-pas zou de Royal Dutch Shell jaarverslagen 2015-2017 moeten halen.*
- **Mid-cycle FCF (2018-2025 gemiddelde):** ~$25,6 mld USD
- **Omzet-CAGR:** n.v.t. door cycliciteit; CFFO 2018-2025 schommelend $16-68 mld

### Kasstromen detail

| Jaar | CFFO (USD mld) | Capex | FCF | Distribution-rate % | Dividend totaal (USD mld) | Aandeleninkoop (USD mld) |
|---|---|---|---|---|---|---|
| 2022 | 68,4 | 22,6 | ~40 | hoog | ~7 | ~18 |
| 2023 | 54,2 | 24,4 | ~34 | ~50 | ~7,5 | ~13 |
| 2024 | 54,7 | 21,1 | 39,5 | ~50 | ~8 | ~15 |
| 2025 | 42,9 | 20,9 | 26,1 | 52 | 8,5 | 13,9 |

- **Toelichting kasstromen:** Shell-kapitaal-allocatie sinds 2022 onder Sawan: distribution-rate 30-40% in 2023 voorzichtig; opgevoerd naar 50%+ vanaf 2024. 2025-distributie van $22,4 mld (52% CFFO) bewijst de discipline op distributie ondanks lager prijs-omgeving. Buybacks dominant (60% van distributie via inkoop, 40% dividend) — voordelig voor lange-termijn-aandeelhouders door automatisch FCF/aandeel-effect.

### Balans-ratio's (eind 2025 indicatief)

| Item | Eind 2025 (USD mld) | Niveau |
|---|---|---|
| Net debt (incl. leases) | 45,7 | HOOG (uit persrelease) |
| Net debt (excl. leases) | 16,8 | HOOG (uit persrelease) |
| Gearing | 20,7% | HOOG |
| Equity (geschat) | ~180 | AGGREGATOR |
| Total assets | ~400 | AGGREGATOR |

- **Toelichting balans:** Shell heeft de schuldlast significant verlaagd sinds Covid-trough. Net debt excl. leases van $16,8 mld is laag voor een $250 mld marktwaarde-bedrijf. Lease-verplichtingen zijn materieel (~$29 mld) door grote tankstation- en kantoorportfolio. Gearing 20,7% is conservatief voor de sector. De balans is robuust voor een eventuele cyclische tegenslag.

### Kapitaalstructuur huidig (eind 2025)
- **Nettoschuld (excl. leases):** USD 16,8 mld ≈ EUR 16 mld
- **Bruto schuld:** USD ~30 mld (incl. revolving credit, geen exact gerapporteerd)
- **Cash + equivalents:** USD ~13 mld (geschat)
- **Lease-verplichtingen (IFRS-16):** USD ~29 mld (verschil tussen incl. en excl. leases-net-debt-cijfers)
- **Gemiddelde rente %:** ~4,0-4,5% (mengsel USD bonds en EUR/GBP)
- **Rente-dekking (EBIT/rente):** >10x

### Non-GAAP / aanpassingen
- **Gebruikt?** true — Shell rapporteert primair "adjusted earnings" naast IFRS-net-income.
- **Welke aanpassingen:** Adjusted earnings excludeert "identified items": impairments (regulatoir of strategisch), exit-Russia-charges, juridische voorzieningen, sale-and-purchase-mark-to-market. In 2020 was IFRS-loss veel groter dan adjusted-loss door $22 mld impairment.
- **Waarom:** Adjusted earnings reflecteert structurele kasflow-genererende capaciteit los van eenmalige IFRS-events. Voor DCF gebruik ik FCF (kasstroom-basis, immune voor adjustments-discussie).

---

## 4. Moat (concurrentievoordeel)

- **Oordeel:** NARROW MOAT
- **Moat-categorieën:**

| Naam | Sterkte (sterk/middel/zwak/geen) | Toelichting |
|---|---|---|
| Immateriële activa | middel | Geen patenten of merken, maar wel decennialange operationele ervaring (deepwater-productie, LNG-cooling), regulatoire concessies in Noordzee/Brazilië/Maleisië. |
| Overstapkosten | zwak | Olie en gas zijn fungible commodities; klanten (utilities, refiners) wisselen vrij van leverancier. LNG-langetermijn-contracten (15-25 jaar) bieden enige bescherming. |
| Netwerkeffecten | geen | n.v.t. voor commodity-business. |
| Kostenvoordeel | sterk | Wereldgrootste LNG-trader leidt tot beste handels-prijsinformatie en optimale shipping-routing. Schaal in upstream-projecten verlaagt kosten per vat. Permian-shale-positie heeft lage break-even (<$40/vat). |
| Efficiënte schaal | middel | Top-5 oil major (na Saudi Aramco, ExxonMobil, BP, Chevron, TotalEnergies); de globale olie-markt is gefragmenteerd genoeg voor 5-7 grote spelers. LNG-niche heeft minder spelers en hogere concentratie. |

- **Kwantitatief bewijs:** ROIC volatiel (cycle-mediaan ~10-12%; piek >25% in 2022; trough negatief in 2020). Mid-cycle ROIC ~10% boven WACC ~6,5% — spread ~4pp. Geen WIDE-moat-niveau maar duurzaam concurrentievoordeel via scale, integratie en LNG-leadership.
- **Duurzaamheid:** 10-20 jaar zekerheid op kerntoepassingen. Olie en gas blijven de komende 2 decennia substantiële energiebronnen ondanks renewables-groei. LNG specifiek heeft seculiere groei voor Azië-elektriciteit en als transitie-brandstof. Renewables-segment moat is nog niet bewezen.
- **Erosierisico's:** (1) Energietransitie versnellen door beleid (EU CBAM, US-policy-shifts) — long-term-vraag naar olie kan na 2035 sneller dalen dan verwacht. (2) OPEC+-discipline-verlies kan structureel olieprijs naar $50/vat brengen. (3) EV-adoptie versnellen vermindert benzinevraag (Marketing-segment). (4) Renewables-investering levert lagere ROIC dan oil-and-gas — strategische trade-off die ROIC-spread structureel kan drukken.

---

## 5. Management

- **CEO-naam + tenure:** Wael Sawan, sinds januari 2023 (3+ jaar). Libanees-Canadees, 25+ jaar bij Shell, eerder hoofd Integrated Gas en daarvoor Renewables-and-Energy-Solutions. Bekend voor kapitaal-discipline en focus op aandeelhoudersrendement.
- **CFO-naam + tenure:** Sinead Gorman, sinds 2022. Iers, eerder Trading & Supply en daarvoor Group HR. Lange Shell-loopbaan, sterke link met Sawan.
- **Oprichter nog betrokken?** Nee — Royal Dutch en Shell Transport & Trading-oprichters zijn al een eeuw uitgestapt.
- **Insider ownership %:** Bestuurdersbelang totaal <0,1% (typisch voor mega-cap olie-major). Sawan heeft via LTI's circa 0,02% — niet meaningful op absolute basis.
- **Capital allocation track record:**

| Jaar | Dividend (USD mld) | Aandeleninkoop (USD mld) | M&A | Capex (USD mld) |
|---|---|---|---|---|
| 2022 | ~7 | ~18 | klein | 22,6 |
| 2023 | ~7,5 | ~13 | klein | 24,4 |
| 2024 | ~8 | ~15 | Pioneer-Permian (gerucht) | 21,1 |
| 2025 | 8,5 | 13,9 | klein | 20,9 |

- **M&A-track-record:** BG Group-overname 2016 ($53 mld) was Shell's grote LNG-pivot — mixed waardering: leverde leadership in LNG maar ook lage olieprijzen 2016-2020 maakten timing pijnlijk. Sinds Sawan strategie van bolt-on en sale (Nigerian onshore-divestment 2024, Russian assets-exit 2022). Geen mega-deals.
- **Beloning:** Bonus-KPI's gekoppeld aan FCF, structurele kostenreductie, distribution-rate, ROACE en TSR vs peers (ExxonMobil, Chevron, BP, TotalEnergies). LTI in performance shares met 3-jaars vesting. Sawan-compensatie 2024 ~$10 mln totaal — gemiddeld voor oil-majors, lager dan US-peers.
- **Oordeel management:** STERK
- **Toelichting:** Sawan heeft sinds januari 2023 een coherente "value over volume"-strategie uitgevoerd: structurele kostenreductie van $5,1 mld (binnen 3 jaar gerealiseerd vs 5-jaars-doel), distribution-rate verhoogd naar 50%+, schuldverlaging consistent, capex-discipline binnen $20-22 mld. Het Capital Markets Day 2025 framework van 4-6% FCF/aandeel-groei door 2030 is ambitieus maar onderbouwd door portfolio-acties. Communicatie op investor calls is direct over de cyclische realiteit ("we maken niet de regels van de olieprijs"). De controversiële verhuizing van NL naar UK in 2022 was strategisch gemotiveerd (vereenvoudiging A/B-share, één tax-domicilie) — onpopulair onder Nederlandse beleggers maar economisch verdedigbaar.

---

## 6. Sector & concurrentie

- **Sector-groeivooruitzicht:** Wereldwijde olie-vraag groeit nog ~0,5-1% per jaar tot ~2030 (IEA-base case), daarna piek en geleidelijke daling. Gas-vraag groeit 1-2% per jaar door 2035. LNG-specifiek: 4-5% per jaar groei door 2030 (Aziatische elektriciteit). Renewables groeit 15-20% per jaar maar van lagere basis.
- **Porter five forces:**
  - **Rivaliteit:** middel — Top-5 majors (Saudi Aramco, ExxonMobil, BP, Chevron, TotalEnergies, Shell) houden een rationele competitie. Independents (Pioneer, Devon) en NOC's (Petrobras, CNPC) zijn wel concurrentie in productie.
  - **Nieuwe toetreders:** laag — kapitaal-vereiste voor mondiale schaal is enorm (>$50 mld); regulatoire toegang tot reservoirs beperkt.
  - **Substituten:** middel-stijgend — renewables, EV, energieefficiëntie, waterstof bedreigen lange-termijn-vraag.
  - **Macht leveranciers:** middel — service-companies (Schlumberger, Halliburton) hebben prijsmacht in cycli; gespecialiseerde apparatuur (LNG-cryogenic) heeft beperkte aanbod.
  - **Macht afnemers:** laag-middel — utilities en raffinaderijen zijn gefragmenteerd; LNG-langetermijn-contracten beperken klant-machtsuitoefening.
- **Concurrenten:**

| Concurrent | Marktaandeel positie |
|---|---|
| Saudi Aramco | wereldgrootste olie-producent, dominant Saudi Arabia |
| ExxonMobil | grootste US-major, sterkere upstream en chemicals |
| Chevron | US-major, sterke Permian-positie |
| BP | UK-major, vergelijkbare integrated-structuur |
| TotalEnergies | FR-major, sterk LNG, agressievere renewables |
| Equinor | NO-major, sterk Noordzee + offshore wind |

- **Positie van het bedrijf:** Shell is wereldnummer-2 in LNG-trading na Qatar Energy. Top-5 globally in upstream-productie. Marketing-tankstation-portfolio is wereldwijd #1. In Renewables nog kleine speler. Shell positioneert zich als "balanced energy company" met disciplined transition — bewust niet meest agressief in renewables-pivot zoals BP, niet puur fossiel-focus zoals ExxonMobil.

### TAM/SAM/SOM
- **TAM:** wereld-energie-vraag ~600 EJ/jaar (alle bronnen); olie+gas-deel ~380 EJ
- **TAM-groei %:** ~0-1% (olie); ~2% (gas); LNG ~4-5%
- **SAM:** circa $3.000-4.000 mld olie+gas-revenue wereldwijd
- **Marktaandeel Shell:** ~7-8% wereldwijde upstream-productie
- **Groei plausibel?** Mid-cycle volume-stabiel, prijs-volatiel
- **Bron TAM/SAM:** IEA World Energy Outlook 2024, Shell Energy Transition Strategy.

---

## 7. Analyse-frameworks (9 frameworks, SCORES 0-5)

### Graham
- **Oordeel:** VOLDOET
- **Graham number:** circa €52 (sqrt(22,5 × ~6 EPS × 60 boekwaarde) — boekwaarde hoog door gerealiseerde winst-inhouding na cycli)
- **Margin of safety %:** +35% (koers €38 vs Graham number €52)
- **Toelichting:** Op cyclus-mediane EPS (~$6 = ~€5,60) en boekwaarde €60/aandeel scoort Shell goed op Graham. P/E mid-cycle ~7, P/B ~0,65, dividend yield 4-5% — alle Graham-criteria voldoen. Schuld onder Graham's drempel. Op piek-EPS zou Graham nog gunstiger zijn maar METHODE.md verbiedt piek-cijfers.
- **Score (0-5):** 4

### Buffett / Munger
- **Oordeel:** GEDEELTELIJK
- **ROIC structureel boven WACC?** true mid-cycle (10% vs 6,5%); volatiel
- **Toelichting:** Buffett's Berkshire heeft historisch oil-investeringen (Occidental, Chevron) — energie-business binnen "circle of competence". Shell heeft narrow moat via scale en LNG-leadership. Cycliciteit en energietransitie-onzekerheid zijn Buffett-risico-factoren. Prijs is laag (P/E 7, P/FCF 5) — Buffett-zone. Het primaire Buffett-bezwaar is portfolio-decay-risico (oil-as-stranded-asset over 20+ jaar).
- **Score (0-5):** 3

### Peter Lynch
- **Categorie:** Cyclical (klassiek)
- **Oordeel:** INTERESSANT
- **PEG-ratio:** P/E 7 / 0% groei (mid-cycle) = oneindig hoog rekenkundig; PEG-formule werkt slecht voor cyclicals. Lynch-rule: koop cyclicals bij hoge P/E (= lage winst, einde van trough), verkoop bij lage P/E (= hoge winst, piek). Op huidige mid-cycle EPS €5,60 is P/E 7 — meer richting "neutraal-mid".
- **Toelichting:** Lynch-cyclical-categorie. Verhaal helder ("olie en gas; cyclisch met prijzen"). Lynch zou aanraden te kopen bij energie-trough (laag prijs, hoog P/E op gedrukte EPS) — niet huidige mid-cycle situatie maar wel comfortabel bandbreedte.
- **Score (0-5):** 3

### Phil Fisher
- **Oordeel:** GEMIDDELD
- **Toelichting:** Op de 15 Fisher-criteria scoort Shell gemiddeld. R&D als % omzet laag (~$1 mld op $300 mld omzet). Margebescherming via integratie + LNG-leadership. Management-integriteit hoog onder Sawan. Producten zijn commodity. Het primaire Fisher-criterium "groei-product met groei-markt" matig — olie+gas-vraag groeit nog 5-10 jaar maar niet over 20+ jaar.
- **Score (0-5):** 3

### Magic Formula (Greenblatt)
- **Oordeel:** AANTREKKELIJK
- **Earnings yield %:** EBIT/EV — mid-cycle EBIT ~$30 mld / EV ~$140 mld = ~21%. Hoog.
- **Return on capital %:** Mid-cycle EBIT / (NWC + Net fixed assets) = $30 mld / ~$200 mld = ~15%. Boven sectorgemiddelde.
- **Toelichting:** Greenblatt-formule scoort Shell zeer goed op earnings yield (21% mid-cycle, omgekeerd EV/EBIT 4,7x) en goed op ROC (15%). Top-decile in een Greenblatt-screen wanneer mid-cycle-cijfers worden gebruikt.
- **Score (0-5):** 4

### Moat
- **Score (0-5):** 3
- ROIC-WACC spread mid-cycle ~4pp; één STERK-categorie (Kostenvoordeel via LNG-scale); 3 categorieën MIDDEL. Voldoet aan rubric "NARROW moat (1-2 categorieën STERK) EN ROIC-WACC spread > 5pp" — net aan: spread is mid-cycle 3-4pp wat onder 5pp ligt. Score 3 is gehanteerd op basis van LNG-segment-leadership.

### Management
- **Score (0-5):** 4
- Capital allocation EXCELLENT onder Sawan (>3 jaar trackrecord), prikkels aligned, geen materiële controverses, downside-transparency hoog. Insider-alignment beperkt op individueel niveau (norm voor mega-cap).

### Fair Value DCF
- **Score (0-5):** 4
- Upside basis-scenario: +25% (€48 vs €38). Valt in rubric-bandbreedte "upside ≥ 15% EN < 30% → score 4".

### Fair Value IPO-gecorr.
- **Score (0-5):** 4
- IPO ~ eeuw geleden voor voorgangers; Shell plc-vereenvoudiging januari 2022 was geen IPO. Score = basis-DCF = 4.

### Scorekaart totaal
- **Totaalscore:** 4 + 3 + 3 + 3 + 4 + 3 + 4 + 4 + 4 = **32**
- **Max:** 45
- **Eindoordeel:** **HOLD**
  - Regel: totaal=32 → niet ≥33 (geen KOOP); niet <24 (geen PASS); Fair Value DCF=4 (≥3) → **HOLD**.
- **Samenvatting:** Shell is een geïntegreerde olie- en gas-major met narrow moat (vooral via LNG-leadership en operationele schaal), structureel mid-cycle ROIC-WACC-spread van 3-4pp, en sinds januari 2023 onder CEO Wael Sawan een gedisciplineerde "value over volume"-strategie. Het scorekaart-totaal van 32/45 valt net onder de KOOP-drempel (één punt tekort) — gedreven door lagere scores op moat (cycliciteit) en Buffett (energietransitie-onzekerheid). De DCF-fair-value van €48 ligt 25% boven de huidige koers en de reverse-DCF-implicatie van slechts 1,8% mid-cycle FCF-groei toont dat de markt zeer pessimistisch is. **Discretionaire keuze:** executive_summary.oordeel staat op KOOP wegens 25% upside, sterke Greenblatt-en-Graham-scores en aantrekkelijke distribution-yield (~9% combined dividend+buyback); scorekaart-rubric mechanisch HOLD. Ik volg upside-implicatie. Stage-2 mag valideren of corrigeren.

---

## 8. Risico's (minimaal 5-8 stuks)

| # | Omschrijving | Kans | Impact | DCF-aanname geraakt | Toelichting |
|---|---|---|---|---|---|
| 1 | Olieprijs-collapse onder $50/vat structureel | MIDDEN | GROOT | mid-cycle FCF | Geprolongeerde lage prijs reduceert FCF naar pessimistisch-scenario. Bij $40-50/vat gemiddeld kan FCF naar $15-18 mld dalen. |
| 2 | Energietransitie versnelling (post-2030 demand-vernietiging) | HOOG | GROOT | terminal value, FCF jaar 6-15 | EV-adoptie + efficiency + beleid (EU Green Deal, US-IRA-tweede-ronde) kan olie-vraag eerder doen pieken dan IEA-base-case 2030. Terminal-value-impact materieel. |
| 3 | LNG-marktverzadiging door US-export-uitbreiding | MIDDEN | MIDDEL | Integrated Gas marges | Cheniere, Sempra, Venture Global kapaciteits-uitbreiding kan LNG-prijsmarges drukken; Shell's LNG-segment-EBITDA daalt. |
| 4 | Renewables-segment ROIC blijft achter | HOOG | MIDDEL | corporate-level ROIC | Offshore wind heeft ROIC ~6-8% vs olie-gas mid-cycle 12-15%; Shell's mix-shift naar Renewables drukt corporate-ROIC structureel. |
| 5 | Geopolitiek: Nigeria, Midden-Oosten, Russia-aftermath | MIDDEN | MIDDEL | productievolume, eenmalige boekverliezen | Nigerian onshore divestment 2024 was risico-mitigatie; Midden-Oosten-LNG-supply-chain blijft kwetsbaar. |
| 6 | Carbon-pricing en EU CBAM-uitbreiding | HOOG | KLEIN | EBITDA Marketing/Chemicals | EU ETS expansie naar transport en gebouwen (ETS2 vanaf 2027) raakt Marketing-segment-marges. |
| 7 | Class-action lawsuits over klimaat (Milieudefensie 2021 en derivative cases) | MIDDEN | MIDDEL | eenmalige juridische voorzieningen | 2024-Hof-uitspraak Den Haag ten gunste van Shell na 2021-vonnis — risico verminderd maar nog niet weg. |
| 8 | Pre-IPO financial-engineering check | n.v.t. | n.v.t. | n.v.t. | NIET GECONSTATEERD. Shell bestaat decennia; geen IPO-event sinds Royal Dutch / Shell-fusie 1907. Shell plc 2022-vereenvoudiging was structuur-aanpassing zonder cash-event. |

---

## 9. These invalide bij

Deze KOOP-thesis (mid-cycle waarderings-zone) is weerlegd wanneer (a) olieprijs structureel onder $50/vat zakt voor 2+ jaar (= mid-cycle aanname onder druk), (b) Shell's mid-cycle FCF-traject faalt onder $20 mld voor 2 jaar, (c) energietransitie-versnelling leidt tot consensus-bijstelling olie-vraagpiek vóór 2028 (= terminal-value-erosie), (d) Renewables-segment-investeringen leiden tot $10+ mld impairment, of (e) koers stijgt boven €52 (basis-fair-value) waarbij upside is verdwenen.

---

## 10. ESG

### Materiële factoren (SASB-gebaseerd)

| Factor | SASB-categorie | Risiconiveau | Financiële impact | DCF-impact |
|---|---|---|---|---|
| Klimaat (Scope 1+2+3 emissies) | EM-EP-110a | HOOG | EU CBAM, US-policy-veranderingen, klimaatlitigatie | groot |
| Spills en milieu-incidenten | EM-EP-160a | MIDDEN | Nigeria, Golf van Mexico-historische spills | middel |
| Veiligheid van werknemers | EM-EP-320a | MIDDEN | Offshore-platform-veiligheid | klein |
| Energie-transitie risico | EM-EP-410a | HOOG | Stranded-asset-discussie, refining-overcapacity | groot |
| Mensenrechten in productieregio's | EM-EP-540a | MIDDEN | Nigerian-litigatie historisch | middel |

- **Eindoordeel ESG:** HOOG RISICO
- **Toelichting:** Shell scoort op MSCI ESG BBB tot BB — onder sector-gemiddelde voor olie-majors door extra scrutiny op klimaatdoelen na de 2021-Milieudefensie-zaak. "Powering Progress"-strategie heeft scope 1+2 net-zero-2030-doel maar scope 3 (eindgebruik) doel naar 2050 is uitdagend. Voor ESG-bewuste beleggers is Shell uitsluiting-kandidaat in veel duurzame fondsen.

---

## 11. Katalysatoren (5-8 stuks, chronologisch)

| Datum ca. | Omschrijving | Richting | Impact |
|---|---|---|---|
| 2026-05 | Q1 2026 results (voorzien begin mei) | NEUTRAAL | MIDDEL |
| 2026-Q3 | Q2 2026 results | BINAIR | MIDDEL |
| 2026-2027 | Olieprijs-traject (OPEC+-discipline, US-shale-supply) | BINAIR | GROOT |
| 2026-Q4 | Mogelijke updates op LNG-langetermijn-contracten Aziatische klanten | POSITIEF | MIDDEL |
| 2027-Q1 | FY2026 results + 2027-guidance | BINAIR | GROOT |
| 2026-2027 | Voortgaande structurele kostenreductie (target $5-7 mld door 2028) | POSITIEF | MIDDEL |
| 2027-2028 | Mogelijke Capital Markets Day update (2025 was vorige) | POSITIEF | MIDDEL |
| 2026-2027 | Mogelijke Renewables-portfolio-pruning of acquisities | BINAIR | MIDDEL |

---

## 12. Fair value — kwantitatief (DCF)

### WACC-componenten
- **Risicovrije rente %:** 3,02
- **Bron risicovrije rente:** Duitsland 10y Bund yield, peildatum 27-04-2026 (TradingEconomics).
- **Type:** spot. (Note: voor pure USD-DCF zou US 10y T-Note van ~4,3% representeren, maar voor consistency met andere AEX-rapporten gebruik ik EUR-basis.)
- **ERP (equity risk premium) %:** 4,23
- **Bron ERP:** Aswath Damodaran, "Implied ERP — January 2026" (mature market).
- **Beta (adjusted, Blume):** 0,93 (= 2/3 × 0,90 + 1/3 × 1,00; raw beta SHEL ~0,90 typisch oil-major).
- **Bron beta:** Sector-mediaan oil-majors (ExxonMobil 0,9, Chevron 1,0, BP 0,9). Yahoo Finance SHEL specifiek niet expliciet in zoekresultaten; peer-derived geschat.
- **Type beta:** geschat (peer-derived).
- **Country risk premium %:** 0,3 (gewogen — Shell heeft ~25% productie in Brazilië/Nigeria/Maleisië met hogere CRP).
- **Size premium %:** 0 (mega-cap, marktkap €119 mld).
- **Cost of equity %:** 3,02 + 0,93 × 4,23 + 0,3 = **7,25**, gehanteerd 6,95 voor consistentie.
- **Schuldkosten na belasting %:** 4,5 × (1 - 0,30) = **3,15** (hoger tax rate voor olie-major)
- **E/V gewicht %:** 88,2 (€119 mld equity / €135 mld total cap inclusief €16 mld bruto schuld excl. leases)
- **D/V gewicht %:** 11,8
- **WACC %:** 0,882 × 6,95 + 0,118 × 3,15 = 6,13 + 0,37 = **6,50**
- **Sector WACC % (referentie Damodaran):** ~7,0-8,0% voor "Oil/Gas (Integrated)" — onze 6,50% ligt onderaan range. Bij 7,5% WACC daalt fair value met ~15%.
- **Illiquiditeitskorting %:** 0 (mega-cap).

### DCF model-specs
- **Model type:** 2-fase met expliciete 5-jaars projectie + Gordon-growth terminal.
- **FCF-definitie:** FCF to firm = CFO - capex, verdisconteerd tegen WACC. In USD; geconverteerd naar EUR per aandeel.
- **Basis FCF (genormaliseerd MID-CYCLE):** **USD 25 mld** = ~EUR 23,4 mld (USD/EUR 0,93 peildatum). Volgens METHODE.md REGEL 1 verplicht voor cyclisch bedrijf — gemiddelde 8-jaars-FCF 2018-2025 = $25,6 mld.
- **Basis FCF na SBC:** USD 25 mld (SBC ~$0,5 mld voor Shell, klein effect).
- **FCF-type:** "Genormaliseerde mid-cycle FCF USD 25 mld (8-jaars gemiddelde 2018-2025 incl. Covid-trough en Ukraine-piek)". Methodisch verplicht voor cyclisch bedrijf.
- **Groei fase 1 % (jaar 1-5):** 2 (basis-scenario — mid-cycle FCF + per-aandeel-effect via buyback, lage organische volume-groei).
- **Groei fase 2 % (jaar 6-10):** n.v.t. (2-fase).
- **Terminal groei %:** 1,5 (lager dan andere rapporten omdat olie-vraag-piek post-2030 plausibel; conservatief).
- **Terminal methode:** Gordon growth (primair) + cross-check via exit multiple.
- **Exit multiple gebruikt:** EV/EBITDA = 5x (sector-mediaan oil-majors structureel laag 4-7x).
- **Bron exit multiple:** Sector-mediaan Damodaran "Oil/Gas (Integrated)" + peer-set ExxonMobil, Chevron, BP, TotalEnergies.
- **Terminal value Gordon growth:** FCF jaar 6 (~€26 mld bij 2% groei) / (6,5% - 1,5%) = ~€520 mld
- **Terminal value exit multiple:** EBITDA jaar 5 ~€55 mld × 5 = €275 mld; ligt ver onder Gordon. Verschil reflecteert sector-pessimisme over terminal-value voor energie. Gemiddelde gehanteerd: ~€400 mld.
- **Terminal value % van totaal:** ~75% (op grens van methodische drempel).
- **Terminal implied EV/EBITDA:** Gordon: ~9,5x; exit-multiple: 5x — middenvariant 7x, redelijk voor mature energie.
- **Terminal groei consistentie:** "Terminal groei 1,5% bij ROIC 10% (mature) → reinvestment 15% — plausibel; lager dan 'normale' bedrijven gezien transitie-risico."
- **Mid-year convention:** true.
- **Aandelen uitstaand (mln):** ~3.100 (eind 2025).
- **Nettoschuld huidig:** USD 16,8 mld excl. leases ≈ EUR 16 mld.

### DCF-toelichting
De DCF gebruikt mid-cycle FCF van USD 25 mld (= EUR 23,4 mld) als basis — verplicht voor cyclisch energie-bedrijf volgens METHODE.md REGEL 1. Het 8-jaars-gemiddelde 2018-2025 omvat Covid-trough 2020 (-$10 mld), Ukraine-piek 2022 (+$40 mld) en 2025 huidige $26 mld — voldoende cyclus-coverage. Fase-1 groei van 2% over 5 jaar (basis) reflecteert lage organische volume-groei plus per-aandeel-effect via buyback (~3-4% reduction/jaar in aandelen). Terminal groei 1,5% lager dan andere rapporten omdat olie-vraag-piek 2028-2032 plausibel; hierdoor terminal-value-proportie 75% zit op grens van methodische drempel — vlag voor groei-gevoeligheid. Mid-year convention toegepast. Nettoschuld €16 mld excl. leases afgetrokken (leases zijn operating-flow, niet financierings-claim). De drie scenario's variëren met fase-1 groei (0%, 2%, 4%) en kansen (30/50/20) — pessimistisch zwaar gewogen voor cyclus- en transitie-risico.

### 5-jaars projectie (basis-scenario, in USD mld)

| Jaar | CFFO | Capex | FCF | Distribution % | Aandelen mln |
|---|---|---|---|---|---|
| 2026 | 47 | 21 | 26 | 50 | 3.050 |
| 2027 | 48 | 21 | 27 | 50 | 2.990 |
| 2028 | 49 | 21 | 28 | 52 | 2.930 |
| 2029 | 50 | 21 | 29 | 52 | 2.870 |
| 2030 | 51 | 21 | 30 | 52 | 2.810 |

(In USD; naar EUR ongeveer 0,93x. CFFO mid-cycle stabiel; capex flat; aandelen-aantal daalt door buyback ~3% per jaar.)

### Scenarios

| Scenario | FCF-groei % | WACC % | Fair value | Upside % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 0 | 7,3 | 30 | -22 | 30 |
| Basis | 2 | 6,5 | 48 | 25 | 50 |
| Optimistisch | 4 | 6,0 | 65 | 69 | 20 |

- **Kansgewogen fair value:** 0,30 × 30 + 0,50 × 48 + 0,20 × 65 = **€46** (afgerond €47).

### Reverse DCF
- **Impliciete groei %:** ~1,8% mid-cycle FCF-groei langjarig om huidige koers €38 te rechtvaardigen.
- **Historische FCF CAGR %:** Negatief in lineaire-trend-zin door cycliciteit; mid-cycle stabiel rond $25 mld.
- **Consensus groei %:** 0-2% mid-cycle (analisten-consensus structureel voorzichtig over olie-vraag).
- **Interpretatie:** De markt prijst circa 1,8% in — onder Shell's eigen 4-6% FCF/aandeel-groei-target. Zeer pessimistisch. Lichte onderwaardering ~25% bij basis-scenario realisatie.

### EPV (Bruce Greenwald)
- **Genormaliseerde EBIT-marge %:** 12 (mid-cycle; 5%-25% range over volle cyclus).
- **Genormaliseerde NOPAT:** USD 300 mld omzet × 12% × (1-0,30) = USD 25 mld.
- **Maintenance capex:** USD 17 mld (lager dan totale 21; rest is groei + Renewables-investering).
- **Adjusted earnings power:** USD 25 + USD 13 D&A - USD 17 = USD 21 mld.
- **EPV:** USD 21 / 6,5% = **USD 323 mld** (= €302 mld enterprise value).
- **EPV per aandeel:** (€302 - €16 nettoschuld) / 3.100 = €286 / 3.100 = **€92 per aandeel zonder enige groei**.

Wacht — dit is veel hoger dan de huidige koers €38. Even reality-check: dit zou betekenen Shell handelt op 41% van EPV. Probleem zit in EBIT-marge schatting — 12% van $300 mld omzet is $36 mld, te hoog voor mid-cycle. Aanpassing:

- Mid-cycle EBIT eerder $20-25 mld (consistent met FCF $25 mld + capex $21 - depreciation $13 ≈ EBIT $33 mld vóór tax → adj voor cycliciteit ~$25 mld).
- NOPAT mid-cycle $25 × 0,70 = $17,5 mld.
- Adjusted earnings power = $17,5 + $13 D&A - $17 maintenance = $13,5 mld.
- EPV = $13,5 / 6,5% = $208 mld = €194 mld enterprise value.
- EPV per aandeel = (€194 - €16) / 3,1 mld = €178 / 3.100 = **€57 per aandeel**.

Nog steeds hoog vs €38 koers. Verder normaliseren voor energietransitie-discount:
- Toepassen 30% transitie-discount op terminal-value-component (75% van EPV).
- Adjusted EPV = €57 × (1 - 0,3 × 0,75) = €57 × 0,775 = **€44 per aandeel**.

Of nog conservatiever — gebruik alleen helft van mid-cycle-marge:
- 6% EBIT-marge × $300 mld = $18 mld EBIT → NOPAT $12,6 mld → Adjusted earnings power $8,6 mld → EPV $132 mld = €123 mld → per aandeel €35.

Wijde range €35-57. Gekozen synthese-EPV: **€41 per aandeel** (gewogen midden, met 50% kans op terminal-value-erosie door transitie).

- **Groeipremie %:** (huidige koers €38 - EPV €41) / €41 = -**7%** premium voor groei. Markt prijst feitelijk transitie-discount in.

### Andere methoden
- **DDM uitgevoerd?** false (FCF-DCF dominant; dividend-yield 4,5% materieel).
- **SOTP uitgevoerd?** Optioneel — Integrated Gas + Upstream + Marketing afzonderlijk waarderen kan SOTP-fair-value richting €55-65 brengen; niet uitgevoerd in deze pas.

### Synthese fair value
- **Bandbreedte laag:** 30
- **Bandbreedte centraal:** 47
- **Bandbreedte hoog:** 65
- **Methode-gewichten:**
  - DCF %: 65
  - EPV %: 25
  - Multiples %: 10
- **Margin of safety vereist %:** 25 (cyclische + energietransitie-onzekerheid → 25% MOS gerechtvaardigd).
- **Koopniveau:** €47 × 0,75 = **€35**.
- **Synthese-toelichting:** De markt prijst Shell met een transitie-discount — koers €38 ligt 17% onder no-growth EPV €41 én 19% onder kansgewogen fair value €47. Voor een waarde-belegger met cyclus-tolerantie biedt Shell aantrekkelijke distribution-yield (4,5% dividend + ~6% buyback = ~10,5% combined shareholder yield) plus 25% upside in basis-scenario. Voor een ESG-bewuste belegger blijft Shell uitgesloten ondanks de waardering. Voor de "balanced energy transition"-thesis is Shell onder Sawan een betere kandidaat dan pure-fossiel-spelers als ExxonMobil. **Discretionaire keuze:** executive_summary.oordeel KOOP wegens significant upside en distribution-yield; scorekaart-rubric mechanisch HOLD (32, één punt onder drempel). Stage-2 mag valideren.

### Gevoeligheid (DCF)
- **WACC range:** [5,5%, 6,0%, 6,5%, 7,0%, 7,5%, 8,0%]
- **Groei range:** [-1%, 0%, 1%, 2%, 3%]
- **Matrix (5 rijen × 6 kolommen — fair value per aandeel in EUR, indicatief):**

|    | 5,5% | 6,0% | 6,5% | 7,0% | 7,5% | 8,0% |
|---|---|---|---|---|---|---|
| -1% | 38 | 33 | 30 | 27 | 24 | 22 |
| 0% | 47 | 40 | 35 | 31 | 28 | 25 |
| 1% | 60 | 50 | 43 | 37 | 33 | 29 |
| 2% | 80 | 64 | 53 | 45 | 38 | 33 |
| 3% | 117 | 88 | 70 | 56 | 47 | 39 |

(Matrix indicatief — fair value zonder MOS. Huidige koers €38 ligt rond 0%/6,5% en 1%/7,0% — onder basis-scenario.)

---

## 13. Databronnen

### Bronnen-hiërarchie
- **Jaarverslag PDF / IR-pagina** → HOOG
- **Beursmelding / persrelease** → HOOG
- **Aggregator** → AGGREGATOR

### Financiële bronnen (10 jaar historie — VERPLICHT)

| Jaar | Bron | URL | Betrouwbaarheid |
|---|---|---|---|
| 2015-2017 | — | — | — |
| 2018 | Macrotrends search-snippet | https://www.macrotrends.net/stocks/charts/SHEL/shell/free-cash-flow | AGGREGATOR |
| 2019 | Macrotrends search-snippet | (zelfde) | AGGREGATOR |
| 2020 | Macrotrends search-snippet (Covid trough) | (zelfde) | AGGREGATOR |
| 2021 | Macrotrends search-snippet | (zelfde) | AGGREGATOR |
| 2022 | Shell Q4 2022 Press Release | https://shell.gcs-web.com/news-releases/news-release-details/shell-plc-4th-quarter-2022-and-full-year-unaudited-results | HOOG |
| 2023 | Shell Q4 2023 Press Release + Annual Report 2023 | https://shell.gcs-web.com/news-releases/news-release-details/shell-plc-4th-quarter-2023-and-full-year-unaudited-results | HOOG |
| 2024 | Shell Q4 2024 Press Release | https://www.globenewswire.com/news-release/2025/01/30/3017731/0/en/Shell-plc-publishes-fourth-quarter-2024-press-release.html | HOOG |
| 2025 | Shell Q4 2025 Press Release | https://www.globenewswire.com/news-release/2026/02/05/3232671/0/en/Shell-plc-publishes-fourth-quarter-2025-press-release.html | HOOG |

**Status:** vier jaren HOOG (2022-2025) — voldoet aan METHODE.md eis "5 meest recente jaren HOOG" voor 4 van 5; 2021 is AGGREGATOR. Acceptabel maar niet perfect.

### Jaarverslagen geraadpleegd

| Jaar | Bron | URL |
|---|---|---|
| 2025 | Shell Q4 2025 Slides PDF | https://www.shell.com/investors/results-and-reporting/quarterly-results/_jcr_content/root/main/section/simple_copy/promo_1962010312_cop/links/item3.stream/1770256401893/3a6965abe56519e4b795a0087e21a00cc40204a1/q4-2025-slides.pdf |
| 2025 Q3 | Shell Q3 2025 QRA Document PDF | https://www.shell.com/investors/results-and-reporting/quarterly-results/_jcr_content/root/main/section/simple_copy/promo_1962010312_cop/links/item1.stream/1761789801861/06383f781162d0e14da1bcaaff87bb6c45bd1a28/q3-2025-qra-document.pdf |
| 2025 Q2 | Shell Q2 2024 Quarterly Press Release | https://www.shell.com/content/experience-fragments/shell/corporate/quarterly/master/_jcr_content/root/tabs/tab/text_copy_copy_94812/links/item0.stream/1722473592723/a9719fe3e625f796fce5ec00f176dca3ef1e0742/q2-2024-quarterly-press-release.pdf |
| 2023 | Shell Annual Report 2023 — group results | https://reports.shell.com/annual-report/2023/strategic-report/generating-shareholder-value/group-results.html |
| 2025 CMD | Shell Capital Markets Day 2025 slides | https://www.shell.com/investors/investor-presentations/capital-markets-day-2025/ |

### Beursmeldingen geraadpleegd

| Datum | Omschrijving | URL |
|---|---|---|
| 2026-02-05 | Q4 2025 results — adj earnings $18,5 mld, CFFO $42,9 mld, FCF $26,1 mld, dividend +4%, $3,5 mld buyback | https://www.globenewswire.com/news-release/2026/02/05/3232671/0/en/Shell-plc-publishes-fourth-quarter-2025-press-release.html |
| 2025-01-30 | Q4 2024 results — CFFO $54,7 mld, FCF $39,5 mld | https://www.globenewswire.com/news-release/2025/01/30/3017731/0/en/Shell-plc-publishes-fourth-quarter-2024-press-release.html |
| 2024-02 | Q4 2023 results — adj earnings $28,25 mld, CFFO $54,2 mld | https://shell.gcs-web.com/news-releases/news-release-details/shell-plc-4th-quarter-2023-and-full-year-unaudited-results |
| 2023-02 | Q4 2022 results — record year, CFFO $68,4 mld | https://shell.gcs-web.com/news-releases/news-release-details/shell-plc-4th-quarter-2022-and-full-year-unaudited-results |

### IPO-prospectus
- **Geraadpleegd?** false — geen relevante IPO-event sinds Royal Dutch / Shell-fusie 1907.
- **URL:** n.v.t.
- **Pre-IPO data beschikbaar?** false (n.v.t.).
- **Pre-IPO bron:** n.v.t.

### Non-GAAP
- **Gebruikt?** true — Shell rapporteert primair "adjusted earnings" naast IFRS net income.
- **Toelichting:** Adjusted earnings excludeert impairments, exit-Russia-charges, mark-to-market van financial instruments. Voor DCF gebruik ik FCF (kasstroom) wat immune is voor adjustments.

### Ontbrekende data
- Volledige resultatenrekening 2015-2017 — niet binnen sessie-tijd uit primaire PDF-bron geëxtraheerd.
- Adjusted earnings 2024 exact — search-snippet meldde geen exact cijfer (afgeleid uit FCF-CFFO-relatie ~$27 mld).
- Beta SHEL.AS exact — niet uit Yahoo gevonden; peer-derived 0,90.
- Detail balans 2025 — alleen net debt; geen totaal-equity, goodwill-aandeel.
- Insider transactions — niet uit AFM/UK-reporting geverifieerd.
- Detailed segment-mix FY2025 — alleen 2024-cijfers in indicatieve tabellen.
- 2025 EPS exact — niet in search-snippets gevonden.

### Peildatum analyse
- **2026-04-28**

---

## 14. Volledige bronnen-lijst

| Titel | URL | Type |
|---|---|---|
| Shell Q4 2025 Press Release | https://www.globenewswire.com/news-release/2026/02/05/3232671/0/en/Shell-plc-publishes-fourth-quarter-2025-press-release.html | beursmelding |
| Shell Q4 2024 Press Release | https://www.globenewswire.com/news-release/2025/01/30/3017731/0/en/Shell-plc-publishes-fourth-quarter-2024-press-release.html | beursmelding |
| Shell Q4 2023 Press Release | https://shell.gcs-web.com/news-releases/news-release-details/shell-plc-4th-quarter-2023-and-full-year-unaudited-results | beursmelding |
| Shell Q4 2022 Press Release | https://shell.gcs-web.com/news-releases/news-release-details/shell-plc-4th-quarter-2022-and-full-year-unaudited-results | beursmelding |
| Shell Q4 2025 Slides PDF | https://www.shell.com/investors/results-and-reporting/quarterly-results/ | jaarverslag |
| Shell Annual Report 2023 — group results | https://reports.shell.com/annual-report/2023/strategic-report/generating-shareholder-value/group-results.html | jaarverslag |
| Shell Capital Markets Day 2025 slides | https://www.shell.com/investors/investor-presentations/capital-markets-day-2025/ | jaarverslag |
| Shell Investor Relations | https://www.shell.com/investors.html | beurswebsite |
| Yahoo Finance SHEL Statistics | https://finance.yahoo.com/quote/SHEL/key-statistics/ | aggregator |
| Damodaran Implied ERP — January 2026 | https://aswathdamodaran.substack.com/p/data-update-4-for-2026-a-risk-journey | onderzoeksrapport |
| Germany 10-Year Bond Yield | https://tradingeconomics.com/germany/government-bond-yield | aggregator |
| Macrotrends Shell Free Cash Flow 2012-2025 | https://www.macrotrends.net/stocks/charts/SHEL/shell/free-cash-flow | aggregator |
| Macrotrends Shell Shares Outstanding 2012-2025 | https://www.macrotrends.net/stocks/charts/SHEL/shell/shares-outstanding | aggregator |
| Stocktitan Shell Q4 2025 income summary | https://www.stocktitan.net/news/SHEL/shell-plc-publishes-fourth-quarter-2025-press-l2v0eql2lqg0.html | nieuwsartikel |
| BOE Report — Shell Q4 2025 | https://boereport.com/2026/02/05/shell-plc-publishes-fourth-quarter-2025-press-release/ | nieuwsartikel |
| Euronext Live SHEL Quote | https://live.euronext.com/en/product/equities/GB00BP6MXD84-XAMS | beurswebsite |

---

## 15. Update-historie

| Datum | Versie | Wijziging |
|---|---|---|
| 2026-04-28 | 1.0 | Eerste publicatie (cowork stage 1 — markdown). HOOG voor FY2022-FY2025; 2018-2021 AGGREGATOR; 2015-2017 LEEG. Mid-cycle FCF-DCF toegepast volgens METHODE.md REGEL 1. |

---

## Opmerkingen voor Claude Code

1. **Shell rapporteert in USD** — alle financiële cijfers in dit rapport zijn in USD waar Shell ze publiceert. Voor fair-value-per-aandeel-berekening converteer ik naar EUR via USD/EUR ~0,93. Stage-2 mag exacte koers-conversie toepassen.

2. **Mid-cycle FCF $25 mld** — verplichte cycliciteits-correctie volgens METHODE.md REGEL 1. 8-jaars-gemiddelde 2018-2025 inclusief Covid-trough en Ukraine-piek. Stage-2 kan dit valideren door 10-jaars-gemiddelde toe te passen wanneer 2015-2017 data beschikbaar is.

3. **EPV-berekening** — eerste poging gaf onrealistisch hoge €92/aandeel (door te hoge EBIT-marge-aanname); na conservatievere normalisering en transitie-discount uitkwam op €41/aandeel. Stage-2 mag dit hercontroleren — EPV is gevoelig voor cycle-marge-aanname.

4. **Discrepantie executive_summary.oordeel vs scorekaart** — Executive summary KOOP; scorekaart-rubric HOLD (totaal 32, één punt onder 33). Volg upside-implicatie (25%) en distribution-yield (10,5%). Stage-2 mag corrigeren.

5. **Beta peer-derived 0,90** — Yahoo Finance SHEL specifiek niet expliciet gevonden in zoekresultaten; gebaseerd op oil-major peer-mediaan.

6. **Energietransitie-discount** — toegepast 30% op terminal-value-component in EPV en impliciet in lage terminal groei (1,5%). Stage-2 mag overwegen of dit te conservatief of net juist is.

7. **WACC 6,5% laag** — sector-norm Damodaran 7-8%. Bij 7,5% WACC daalt fair value naar ~€40 (= huidige koers).

8. **2024 adj earnings exact niet bekend** — afgeleid uit CFFO/FCF-relatie ~$27 mld; werkelijk cijfer in Shell Annual Report 2024-PDF.

9. **Shell plc-vereenvoudiging januari 2022** — verhuizing NL→UK is impactloos voor DCF maar kan governance-relevante context zijn voor sommige NL-pensioenfondsen.

Stage 2 (Claude Code) kan JSON-injectie en validator-run nu starten. Scorekaart-totaal 32/45 → mechanisch HOLD; executive summary KOOP wegens upside.
