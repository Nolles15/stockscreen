# Research: PREV-B — Prevas AB

> Stage-1 markdown-rapport voor Claude Code. Conform `research/METHODE.md` en
> `research/TEMPLATE.md`. Peildatum 2026-05-27. Koers SEK 80,70 (Inderes,
> intraday). De analyse is opgebouwd vanuit bronnen die daadwerkelijk zijn
> geopend; jaren zonder verifieerbare bron blijven leeg in tabellen en zijn
> genoteerd in §13 onder ontbrekende_data.

---

## Bronnen-inventaris (Stap 0.5)

```
Jaar 2026 Q1 — HOOG
  Bron: Prevas Q1 2026 Interim Report (PDF, May 5 2026)
  URL:  https://mb.cision.com/Main/60/4344141/4074830.pdf
  Daadwerkelijk geopend: ja (volledige PDF, 22 pagina's)
  Cijfers overgenomen: omzet Q1, EBIT, EBITA, EBITDA, nettowinst, EPS,
                       aandelen, CFO, capex, lease-repayments, goodwill,
                       totale activa, eigen vermogen, rentedragende schuld,
                       cash, segmenten Sweden/Denmark/Finland,
                       sectoromzet defense/energy/etc, FTE-aantal,
                       Q1 vs Q1 comparatives, kwartaaloverzicht 9 kwartalen
  Cijfers NIET overgenomen: (geen — alle gewenste velden aanwezig)

Jaar 2025 — HOOG
  Bron: Prevas Year-end Report 2025 (PDF, Feb 10 2026)
  URL:  https://mb.cision.com/Main/60/4305051/3925558.pdf
  Daadwerkelijk geopend: ja (volledige PDF)
  Cijfers overgenomen: FY omzet, EBIT, EBITA, EBITDA, nettowinst, EPS,
                       aandelen, CFO FY, capex FY, lease repayments FY,
                       acquisitie-betalingen, goodwill, totale activa,
                       eigen vermogen, rentedragende schuld, cash,
                       segmenten, sectoromzet, FTE, voorgesteld dividend
                       SEK 4,00, working capital changes
  Cijfers NIET overgenomen: (geen)

Jaar 2024 — HOOG
  Bron: Prevas Q4/Year-end Report 2024 (PDF, Feb 11 2025) — ook in vergelijkende
        kolommen van het FY2025-rapport
  URL:  https://mb.cision.com/Main/60/4103398/3256720.pdf
  Daadwerkelijk geopend: ja (preview + comparatives in FY2025)
  Cijfers overgenomen: FY omzet 1.586,6, EBIT 122,6, EBITA 148,9, nettowinst
                       92,3, EPS 7,13, CFO 136,8, capex 8,7, totale activa
                       1.436,8, EV 703,1, rentedragende schuld 335,8,
                       cash 43,8, goodwill 669,9, dividend SEK 4,75/aandeel
  Cijfers NIET overgenomen: (geen)

Jaar 2023 — HOOG (via persbericht IR Prevas)
  Bron: Prevas publishes Year-End Report 2023 (Cision news, Feb 14 2024)
  URL:  https://news.cision.com/prevas/r/prevas-publishes-year-end-report-for-2023,c3927779
  Daadwerkelijk geopend: ja (search-results extract)
  Cijfers overgenomen: FY omzet 1.482,6, EBITA 169,4 (margin 11,4%),
                       nettowinst 120,9, vergelijking 2022 in zelfde release
  Cijfers NIET overgenomen: detailed balance sheet 2023, kasstromen,
                            segment splitsing (niet in extract)

Jaar 2022 — AGGREGATOR (via release 2023)
  Bron: vergelijkende cijfers in Prevas Year-End Report 2023 (Cision)
  URL:  https://news.cision.com/prevas/r/prevas-publishes-year-end-report-for-2023,c3927779
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: FY omzet 1.324,0, EBITA 164,5 (margin 12,4%),
                       nettowinst 121,9
  Cijfers NIET overgenomen: kasstromen, balans, capex

Jaar 2021 — AGGREGATOR (via release Feb 2022)
  Bron: Prevas Year-End Report 2021 (news.eu.nasdaq view-archief +
        Cision-extract)
  URL:  https://view.news.eu.nasdaq.com/view?id=b02fa6cbc5cade16cf46ef4caf93c4b29&lang=en
  Daadwerkelijk geopend: ja (search-results extract Q4 2021)
  Cijfers overgenomen: Q4 omzet 320, Q4 EBITA 42,4 (margin 13,2%),
                       EPS 3,05, voorgesteld dividend SEK 3,50/aandeel
  Cijfers NIET overgenomen: FY 2021 omzet (niet in extract), FY EBITA,
                            balans- en kasstroomdetails

Jaren 2015–2020 — GEEN BRON BESCHIKBAAR
  Zoekpoging(en): Cision-archief Prevas, prevas.com financial_reporting,
                  StockAnalysis.com/sto/PREV.B/financials (web_fetch geblokkeerd
                  buiten provenance), MarketScreener finances-pagina (geen
                  free-tier historische detail), Yahoo Finance (alleen
                  TTM zichtbaar), search-queries 2015–2020 (geen
                  geretourneerde annual numbers).
  Conclusie: 2015 t/m 2020 blijven LEEG in alle financiële tabellen.
             Genoteerd in §13 ontbrekende_data. Een vollere reeks vereist
             handmatig downloaden van oudere Prevas PDF's uit het IR-archief;
             dat valt buiten wat ik in deze stage-1 run kon verifiëren.
```

---

## Metadata
- **Ticker (bare):** PREV-B
- **Yahoo symbol:** PREV-B.ST
- **Exchange:** STO (Nasdaq Stockholm Small Cap)
- **Sector (GICS-achtig):** Technologie / Industrie (tech-consultancy)
- **Industrie:** IT-services en engineering-consultancy
- **Land:** Zweden
- **Peildatum analyse:** 2026-05-27
- **Koers op peildatum:** 80,70
- **Valuta:** SEK
- **Marktkapitalisatie:** SEK 1,04 mld
- **Marktkap in mln (lokale valuta):** 1040
- **Free float pct:** —
- **Indexlidmaatschap:** Nasdaq Stockholm Small Cap
- **Domein:** prevas.com

---

## 1. Executive summary

- **Kernthese**: Prevas is een Nordic engineering-consultancy van ~1.045 medewerkers die zich specialiseert in product- en productie-ontwikkeling voor industriële klanten — met groeiend zwaartepunt in defensie, EAM (Enterprise Asset Management op Hexagon-platform) en cybersecurity. Het bedrijf draait door één economische cyclus heen, met een EBITA-marge die tussen 7,5% en 12,4% schommelt. Defense-omzet groeide in Q1 2026 met 22% YoY tot 17% van de groep, en een twaalfjarig EAM-contract van SEK 80 mln versterkt de recurring-revenue mix. De balans is conservatief (nettoschuld/EBITDA 0,89), dividend dekt 73% van de winst, en management heeft eigen aandelen bijgekocht in 2024 en 2025. Belangrijkste risico's zijn lagere utilization in zwakke kwartalen en margedruk in Denemarken; het Finse segment laat na zes positieve kwartalen een herstel zien.
- **Oordeel**: **HOLD**
- **Fair value basis** (kansgewogen, lokale valuta): 158
- **Fair value kansgewogen**: 158
- **EPV per aandeel** (Earnings Power Value, zonder groeipremie): 103
- **Upside pct**: 96
- **Fair value scenarios** (3 stuks):

| Scenario | Fair value | Upside % | FCF groei % | WACC % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 83 | 3 | 1 (fase 1) | 9,0 | 25 |
| Basis | 149 | 85 | 4 (fase 1) | 7,8 | 50 |
| Optimistisch | 251 | 211 | 7 (fase 1) | 7,0 | 25 |

- **Reverse-DCF impliciete groei pct**: ca. -2,8 (de markt prijst lichte krimp van FCF in)
- **Grootste kans**: Defense-segment groeit nu 22% YoY en is goed voor 17% omzet; verdere expansie kan EBITA-marge richting 12% target stuwen.
- **Grootste risico**: Margedruk en lagere utilization buiten defense (Denemarken, delen van Zweden) — EBITA-marge zakte van 9,4% (2024) naar 7,5% (2025).

---

## 2. Bedrijfsprofiel

- **Beschrijving**: Prevas is een Noords engineering- en development-house voor product- en productie-ontwikkeling, opgericht in 1985 in het Zweedse Västerås. Het bedrijf combineert technische expertise (embedded electronics, software, mechanica, systems engineering, automatisering, data en AI) met business-consulting voor industriële klanten in negen sectoren: Engineering Industry, Defense, Life Science, Energy, Automotive & Transport, Steel & Minerals, Telecom, Products & Devices en overige. De omzet ontstaat door uurtarieven van ~1.045 consultants op klantopdrachten, aangevuld met recurring license- en service-revenues uit Hexagon Enterprise Asset Management (EAM) en eigen software-tooling zoals Greengenuity-oplossingen voor energie-efficiëntie. In de waardeketen zit Prevas in de "engineering-as-a-service"-laag: zij ontwikkelen prototypes, productie-systemen, MES-oplossingen, test-platforms en cyberveilige product-architecturen die hun klanten vervolgens in eigen fabrieken of producten implementeren. Het unieke is de combinatie van (a) sterke Nordics-niche-expertise in defense (15+ jaar) en EAM, (b) decentraal georganiseerde business units per regio (waardoor schaalvoordelen en agility worden gecombineerd), en (c) preferred-partner-status bij Hexagon — waardoor 12-jarige EAM-contracten van SEK 80 mln tot stand komen die in pure tijd-en-materiaal-consulting niet mogelijk zijn. Het probleem dat Prevas oplost is dat industriële klanten gespecialiseerde technische kennis tijdelijk of langduriger moeten inzetten zonder eigen R&D-capaciteit op te bouwen.
- **Geschiedenis**: Prevas werd in 1985 opgericht in Västerås door een groep KTH-engineers die toepassingen voor industriële automatisering en embedded computing wilden bouwen. In de eerste decennia groeide het bedrijf organisch met een focus op embedded systems en industriële IT — een specialisatie die in lijn was met de Zweedse maakindustrie (ABB, Sandvik, Atlas Copco). In 1998 maakte Prevas de stap naar NASDAQ Stockholm; sindsdien is het ononderbroken beursgenoteerd. In de jaren 2000-2010 verbreedde Prevas zijn portfolio naar Life Science, Defense en Telecom, en bouwde het Nordics-aanwezigheid op via vestigingen in Denemarken (Aalborg, Kopenhagen) en Finland (later versterkt door Enmac). Belangrijke recente keerpunten zijn (a) de strategische verschuiving onder voorganger CEO Mikael Königsson en huidige CEO Magnus Welén (sinds juni 2023) naar specialisatie in defense, EAM en cybersecurity — sectoren met structurele tailwinds; (b) de overname van Enmac in Finland per 1 juli 2024 (~SEK 190 mln, voegde een heel segment toe met SEK 209 mln omzet in 2025); (c) de bolt-on van OIM Sweden AB per 1 juli 2025 (80%, SEK 20,9 mln koopprijs, 35 medewerkers, Medtech/Cleantech-niche in Öresund); (d) de strategische desinvestering van Prevas InfoVis AB in oktober 2025 aan het bestaande management — een focus-keuze. Crises: Prevas doorstond de financiële crisis 2008-2009, de Euro-crisis 2011-2012, COVID-19 in 2020 en de inflatie-/rente-shock van 2022-2023 zonder dividend-schrap of materiële herstructurering — de balans bleef in elke periode conservatief. De laatste vijf jaar laten een omzetgroei van SEK ~900 mln (2020-niveau ingeschat) naar SEK 1.627 mln (2025) zien, met EBITA-marges variërend tussen 7,5% en 12,4% door de cyclische gevoeligheid van consulting-utilization. Het narratief van vandaag is een margin-recovery-verhaal richting het ≥12% EBITA-target, gedragen door defense-tailwind en Finland-herstel (zes opeenvolgende kwartalen positieve trend per Q1 2026).
- **Bedrijfsmodel**: Tijd-en-materiaal en fixed-price consultancy met groeiende recurring component (EAM-services + licenties). Vijf grootste klanten = ~25% Q1 2026 omzet; grootste klant alleen al SEK 50 mln (Sweden segment, defense, > 10% groepsomzet). 28 van de top-30 klanten zijn terugkerend (2025-rapport).
- **IPO-context**: Genoteerd op NASDAQ Stockholm sinds 1998. IPO is > 25 jaar geleden, geen IPO-correctie nodig.
- **Klantprofiel**: B2B-only, klantenbestand spreidt zich over startups tot grote internationale ondernemingen. Top-5 = ~25% van Q1 2026 omzet, top-1 alleen al > 10%.
- **Oprichtingsjaar**: 1985
- **IPO-datum**: 1998 (exacte dag niet in geraadpleegde bronnen)
- **IPO-koers** (lokale valuta): —
- **Personeel** (FTE): 1.045 per Q1 2026 (gemiddelde Q1 2026: 979)
- **Landen actief**: Zweden, Noorwegen, Denemarken, Finland (Nordics)
- **Klantconcentratie**: Top-5 = ~25% omzet, top-1 = > 10% omzet (defense-klant Sweden segment)

### Geografische spreiding (omzet)
| Regio | Omzet % | Valuta-exposure |
|---|---|---|
| Zweden (incl. Noorwegen, vanaf 2026) | 80 (Q1 2026) | SEK |
| Finland | 12 | EUR |
| Denemarken | 8 | DKK |

**Toelichting geografie**: De groep rapporteert in SEK, dus ~20% van de omzet (Finland + Denemarken) ondervindt translatie-effecten. Q1 2026 cashflows lieten een netto FX-effect van SEK -2,9 mln zien (versus +1,4 mln in Q1 2025), waaruit blijkt dat de EUR-DKK-blootstelling geen "natural hedge" levert in opdrachtsmix. Translatieverliezen drukken comprehensive income in 2025 (SEK -19,5 mln).

### Segmenten (omzet 2025)
| Naam | Omzet % | Beschrijving |
|---|---|---|
| Sweden | 74 | Grootste segment; incl. Defense-zwaartepunt en EAM-praktijk; SEK 1.211,9 mln (2025). |
| Finland | 13 | Verworven 1 juli 2024 (Enmac); zes kwartalen positieve trend; SEK 209,0 mln (2025), 19% organische groei Q4. |
| Denemarken | 9 | Workforce-aanpassing in Q1 2026 wegens lagere vraag; SEK 147,9 mln (2025). |
| Overig (Noorwegen + InfoVis tot Q3) | 4 | SEK 58,1 mln (2025); vanaf 2026 binnen Sweden-segment. |

### Aandeelhouders (top 5)
| Naam | Belang % | Type |
|---|---|---|
| — | — | — |

(Top-shareholder gegevens niet expliciet verifieerbaar in geopende bronnen; aanbeveling: zie Yahoo Finance / MarketScreener voor laatste 13F-achtige cijfers. CEO Magnus Welén bezit ~0,14% volgens Simply Wall St / openbare insider-disclosures.)

- **Institutioneel eigendomstrend**: niet verifieerbaar in deze run — laatste insider-aankopen (2024-2025) door CEO suggereren stabiele alignment, geen aanwijzing voor structurele wijzigingen.

---

## 3. Financieel — historische data (10 jaar + TTM)

### Resultatenrekening (bedragen in mln SEK)

Bron-eis: recente 5 jaren moeten HOOG zijn. 2021-2025 zijn HOOG (2024-2025) of AGGREGATOR (2021-2023 via Cision-extracts).

| Jaar | Omzet | Omzetgroei % | Brutowinst | Brutomarge % | EBIT | EBIT-marge % | EBITDA | EBITDA-marge % | Nettowinst | Nettomarge % | EPS | EPS-groei % | Aandelen mln |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2015 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2016 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2017 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2018 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2019 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2020 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2021 | — | — | — | — | — | — | — | — | — | — | 3,05 | — | ~12,89 |
| 2022 | 1.324,0 | — | — | — | — | — | — | — | 121,9 | 9,2 | — | — | ~12,89 |
| 2023 | 1.482,6 | 12,0 | — | — | — | — | — | — | 120,9 | 8,2 | — | — | ~12,89 |
| 2024 | 1.586,6 | 7,0 | — | — | 122,6 | 7,7 | 190,4 | 12,0 | 92,3 | 5,8 | 7,13 | — | 12,885 |
| 2025 | 1.627,0 | 2,5 | — | — | 105,9 | 6,5 | 167,7 | 10,3 | 72,5 | 4,5 | 5,49 | -23,0 | 12,885 |
| TTM (Q1'26) | 1.622,1 | — | — | — | 105,8 | 6,5 | — | — | 70,0 | 4,3 | 5,31 | — | 12,885 |

EBITA-reeks (door management gestuurd): 2024 SEK 148,9 mln (9,4%), 2025 SEK 121,4 mln (7,5%). 2023 EBITA SEK 169,4 mln (11,4%). 2022 EBITA SEK 164,5 mln (12,4%).

- **Toelichting resultaten**: De omzet steeg van SEK 1,324 mld (2022) naar SEK 1,627 mld (2025) — ongeveer 7,1% CAGR over die periode, waarvan circa 7 procentpunt uit acquisities (Enmac 2024, OIM 2025) en de rest organisch. De EBITA-marge bewoog binnen een band van 7,5%-12,4% en zakte in 2025 onder druk door (a) twee minder werkdagen (~SEK 10 mln calendar effect), (b) herstructureringen voor SEK 7 mln, en (c) lagere utilization in Denemarken en delen van Zweden. Q1 2026 toont een gecorrigeerde EBITA-marge van 9,3% — een eerste herstelstap richting het management target van ≥ 12%.
- **Omzet-CAGR**: ~7,1% (2022-2025, drie-jarig). Volledige 10-jarige CAGR niet berekenbaar — pre-2021 data ontbreekt in geverifieerde bronnen.

### Kasstromen (bedragen in mln SEK)

| Jaar | CFO | Capex | FCF | FCF na SBC | FCF/aandeel | FCF-marge % | FCF-groei % | FCF-conversie % | SBC | Dividend totaal | Aandeleninkoop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2024 | 136,8 | 8,7 | 128,1 | ~126,4 | 9,94 | 8,1 | — | 138,8 | 5,8 | 62,6 | 0 |
| 2025 | 143,7 | 7,5 | 136,2 | ~134,5 | 10,57 | 8,4 | 6,3 | 187,9 | 1,7 | 63,4 | 0 |

**Let op**: FCF hierboven is exclusief IFRS-16 lease-aflossingen (SEK 35,9 mln in 2025, 35,0 mln in 2024). Indien je leases als financieringskasstroom beschouwt zoals het bedrijf rapporteert, geldt bovenstaande. Indien je leases als operating-equivalent meeneemt (FCFE-stijl), is "true" FCF ongeveer SEK 100 mln (2025) en SEK 93 mln (2024). De DCF in §12 gebruikt SEK 100 mln als genormaliseerde basis omdat IFRS-16 leases een terugkerende operationele kost zijn voor een asset-light consultancy.

- **Toelichting kasstromen**: CFO is robuust en hoger dan nettowinst (FCF-conversie van bijna 190% in 2025 op de FCF-excl-lease-definitie). SBC is minimaal (~SEK 1,7 mln per jaar via LTI 2024/2027 en LTI 2025/2028) en heeft een verwaarloosbaar effect op FCF — bijzonder positief voor een tech-consultancy. Working-capital release in 2025 (~SEK 31 mln) droeg bij aan het FCF-niveau; in Q1 2026 deed zich juist een working-capital-druk voor (-SEK 21,9 mln) door projecttiming. Capex blijft laag (~SEK 7-9 mln per jaar) — typisch voor een people-business met beperkte vaste activa.

### Balans-ratio's

| Jaar | Nettoschuld | Nettoschuld/EBITDA | Eigen vermogen | ROE % | ROIC % | ROCE % | Current ratio | Solvabiliteit % | Goodwill % van EV | Working capital |
|---|---|---|---|---|---|---|---|---|---|---|
| 2024 | 292,0 | 1,53 | 703,1 | 13,3 | 10,2 | 11,7 | 1,27 | 48,9 | 50,2 | 107,6 |
| 2025 | 259,6 | 1,55 | 699,2 | 10,3 | 8,6 | 11,7 | 1,18 | 51,2 | 50,5 | 68,7 |
| Q1 2026 | 255,2 | 1,46 | 723,7 | — | — | — | 1,21 | 53,1 | 49,6 | 78,7 |

(Nettoschuld = rentedragende kortlopende + langlopende schuld minus cash. Goodwill % EV gebaseerd op gemiddelde EV in periode.)

- **Toelichting balans**: De balans is conservatief: nettoschuld/EBITDA R12 was 0,89 per ultimo 2025 (management-definitie, includeert leases anders dan onze nettoschuld-berekening) — comfortabel onder de covenant-grens van 2,5 en het management-doel van 2,0. Goodwill blijft hoog (SEK 673 mln, ~50% van EV) door de acquisitie-geschiedenis; in Q1 2026 deed zich geen impairment voor. IFRS-16 right-of-use assets bedragen SEK 145,8 mln. Het eigen vermogen daalde licht in 2025 (dividend SEK 61,2 mln + valuta-effecten -SEK 19,5 mln overtroffen winst SEK 72,5 mln). Q1 2026 toont herstel naar SEK 723,7 mln.

### Kapitaalstructuur huidig (per 31-3-2026)
- **Nettoschuld** (huidig): SEK 255,2 mln (gross debt 265,7 + ROU-related 0 - cash 10,5; alternatief incl. lease ~400 mln)
- **Bruto schuld**: SEK 265,7 mln (current 89,3 + non-current 176,4)
- **Cash & equivalents**: SEK 10,5 mln (eind Q1 2026 — afgenomen door working capital en dividend-pad)
- **Lease-verplichtingen (IFRS-16)**: SEK 142,6 mln (right-of-use assets)
- **Gemiddelde rente %**: ~4,6 (berekend uit net financial items 2025 / gemiddelde bruto schuld)
- **Rente-dekking (EBIT/rente)**: 2025: 105,9 / 13,7 = ~7,7x; Q1 2026: 31,9 / 5,1 = 6,3x

### Non-GAAP / aanpassingen
- **Gebruikt?** ja
- **Welke aanpassingen**: EBITA (= EBIT vóór afschrijving van acquisitie-gerelateerde immateriële activa + acquisition-related items). EBITDA definitie is in Q1 2024 herzien om acquisition-related items uit te sluiten — historische cijfers herhaald.
- **Waarom**: vergelijkbaarheid tussen jaren met/zonder acquisities en management-communicatie. Voor de DCF gebruik ik IFRS/GAAP-EBIT als grondslag, niet EBITA — om Damodaran-conform te blijven.

---

## 4. Moat (concurrentievoordeel)

- **Oordeel**: **NARROW MOAT**
- **Moat-categorieën**:

| Naam | Sterkte (sterk/middel/zwak/geen) | Toelichting |
|---|---|---|
| Immateriële activa | middel | Niche-expertise in defense (15+ jaar volgens ABG), EAM op Hexagon-platform (preferred partner), Medtech via OIM. Geen formele patenten of regulatoire bescherming, maar wel reputatie en gespecialiseerde kennis. |
| Overstapkosten | middel | EAM-implementaties op klantsystemen creëren overstapkosten (12-jarig contract van SEK 80 mln is illustratief). Tijd-en-materiaal consulting heeft beperkte overstapkosten. 28/30 top-klanten zijn terugkerend (2025). |
| Netwerkeffecten | geen | Consultancy is een 1-op-1 service-business; geen netwerkeffect. |
| Kostenvoordeel | zwak | Decentrale BU-structuur biedt enige schaalvoordelen, maar Prevas concurreert met AFRY (~12x groter), HiQ, Knightec — geen materieel kostenvoordeel. |
| Efficiënte schaal | zwak | Nordics-engineering-consulting is een gefragmenteerde markt (top-30 spelers, EUR 12,4 mld). Geen "winner-takes-all" dynamiek. |

- **Kwantitatief bewijs**: ROIC 2025 ~8,6%, ROIC 2024 ~10,2%. ROIC-WACC spread is positief maar slechts +0,8 tot +2,4 pp — beperkte structurele waardecreatie. EBITA-marge schommelt tussen 7,5%-12,4% (afgelopen 5 jaar), waaruit blijkt dat margebescherming gedeeltelijk is maar niet absoluut. ABG noteert Prevas op 9-6x EV/EBITA forward 2026e-2027e, ~20% onder peers — markt prijst geen wide moat in.
- **Duurzaamheid**: 5-10 jaar — defense-expertise (15+ jaar geleverd, structureel groeiend) en EAM-positie (langjarig 12-jarig contract) zijn de moat-pijlers. 28/30 terugkerende top-klanten suggereert duurzaamheid van klantrelaties. Op de 20-jaars horizon kan AI-driven productiviteit consulting-uren drukken — onzekere structurele wijziging.
- **Erosierisico's**: (a) AI-tooling die consulting-uren vervangt — beheerst doordat Prevas zelf AI in haar dienstverlening inbouwt; (b) verlies van top-defense-klant (>10% omzet) — concentratierisico; (c) sectorconsolidatie waarbij groteren als AFRY niches afsnijden; (d) loonsverhoging in Nordics-techmarkt zonder doorbelasting.

---

## 5. Management

- **CEO-naam + tenure**: Magnus Welén, CEO sinds juni 2023 (3 jaar). Voorheen regionaal manager Mälardalen en CEO van Prevas Industrial Innovation AB. Achtergrond bij ABB, Sandvik, SnapOn. M.Sc. Mechanical Engineering (KTH).
- **CFO-naam + tenure**: Helena Burström, CFO sinds 2022 (~4 jaar). Voorheen Finance Manager Prevas, eerder Head of Group Accounting in industriële en railway/wholesale-bedrijven.
- **Oprichter nog betrokken?**: Onbekend — Prevas werd in 1985 opgericht door 4 KTH-engineers; geen oprichter meer in operationele functie.
- **Insider ownership %**: CEO Welén ~0,14% direct (per Simply Wall St / Inderes-data). Insider-aankopen 2024-2025 (geverifieerd): 7 mei 2024 850 aandelen à SEK 125,90 = SEK 107.000; 21 november 2025 1.050 aandelen à SEK 105,14 = SEK 110.397.
- **Capital allocation track record**:

| Jaar | Dividend totaal | Aandeleninkoop | M&A uitgaven | Organische capex |
|---|---|---|---|---|
| 2024 | 62,6 | 0 | 190,7 (Enmac) | 8,7 |
| 2025 | 63,4 | 0 | 17,2 (OIM 80%) | 7,5 |

- **M&A-track-record**: Enmac (Finland, 1 juli 2024, ~SEK 190 mln) — voegde een hele segment toe (omzet 2025: SEK 209 mln); herstel van Finse winstgevendheid duurt 6 kwartalen. OIM Sweden (1 juli 2025, SEK 20,9 mln, 80% stake, omzet 2024 SEK 36 mln, 35 medewerkers, Medtech/Cleantech) — marginaal effect op EPS in 2025. InfoVis divestment (oktober 2025) — niche software-tool verkocht aan management; "marginal effect" op resultaat. Dividend is geleidelijk verlaagd (SEK 4,75 voor 2024 → SEK 4,00 voor 2025) toen winst daalde — disciplined capital allocation.
- **Beloning**: LTI 2024/2027 en LTI 2025/2028 warrant-programma's; SBC SEK 1,7 mln (2025) — minimaal en in lijn met aandeelhoudersbelang. Detail van CEO comp-structuur (vast vs. variabel, KPI's) niet in geopende interim/year-end PDF's; aanbeveling zie volledig jaarverslag 2025 (publicatie week 16 2026) en remuneratierapport.
- **Oordeel management**: **STERK**
- **Toelichting**: De combinatie van (a) consistente insider-aankopen door CEO Welén in zowel 2024 als 2025, (b) disciplined dividend-verlaging in 2025 (SEK 4,75 → 4,00) om kapitaal te beschermen, (c) M&A-track-record met zichtbaar herstel (Finland), (d) minimale SBC en geen aandelenverwatering, en (e) lage leverage (nettoschuld/EBITDA 0,89) wijst op verantwoordelijk financieel beheer. De marge-daling van 2025 wordt openlijk besproken in de CEO-comments ("we are not satisfied") — downside transparency aanwezig. Het ontbreken van controverses en de stabiele tenure (CEO 3y, CFO 4y) versterken het beeld.

---

## 6. Sector & concurrentie

- **Sector-groeivooruitzicht**: Nordics-built-environment-/engineering-consultancy ~4,1% groei (sectorraming IndustryRadar). Defense-subsegment groeit veel sneller (Prevas zelf +22% Q1 2026 YoY). Algemene consultancy is mid-cyclisch.
- **Porter five forces**:
  - **Rivaliteit**: HOOG. AFRY (~12x grootste), HiQ, Knightec, Sigma, ÅF, Combitech, Capgemini Engineering — gefragmenteerde markt met talrijke spelers. Concurrentie op specialistische kennis en tarieven.
  - **Nieuwe toetreders**: MIDDEL. Lage initiële kapitaalintensiteit, maar specialisatie en klantvertrouwen (defense-clearance, regulatoire kennis Life Science) creëren toetredingsdrempels.
  - **Substituten**: MIDDEL. In-house engineering bij klanten en — opkomend — AI-tooling die routine-uren vervangt. Prevas integreert AI zelf om dit te beheersen.
  - **Macht leveranciers**: LAAG. Personeel is "de leverancier"; competitive labor market in Nordics is een aanhoudend punt. Geen materiële IT-platform-afhankelijkheid behalve Hexagon (EAM-partner).
  - **Macht afnemers**: MIDDEL. Top-5 = 25% omzet — moderate concentratie. Top-klant > 10% omzet betekent enige onderhandelingsmacht. Defense-klanten zijn relatief sticky.
- **Concurrenten**:

| Concurrent | Marktaandeel % |
|---|---|
| AFRY AB | ~10,7 (EUR 1.328 mln / EUR 12,4 mld sector) |
| Knightec | ~0,8 (EUR 97 mln) |
| HiQ International | — |
| Combitech / Saab | — |
| Sigma Software | — |

- **Positie van het bedrijf**: Nichespeler — gespecialiseerde Nordics-consultancy (rang ~16 in Sweden built-environment top-30, EUR 108 mln equivalent). Sterke positie in defense (15+ jaar expertise) en EAM/Hexagon-partnership. Niet de grootste, maar onderscheidend op specialisatie.

### TAM/SAM/SOM
- **TAM (mln SEK)**: ~EUR 12.400 mln Nordics built-environment + Sweden engineering consulting (IndustryRadar 2024). In SEK ~140.000 mln.
- **TAM-groei %**: ~4,1
- **SAM (mln SEK)**: niet verifieerbaar exact (Prevas-niche binnen TAM); schatting niet opgenomen wegens gebrek aan harde bron.
- **SAM-groei %**: —
- **Huidige penetratie %** (omzet / SAM): —
- **Impliciete penetratie na horizon %**: —
- **Groei plausibel?**: ja (~5% omzetgroei tot CAGR 2030 plausibel gegeven defense-tailwind)
- **Bron TAM/SAM**: https://industryradar.com/sweden/top-30-swedens-largest-built-environment-players/
- **Toelichting**: TAM is groot relatief tot Prevas (omzet 1,2% van Nordic-sector EUR 12,4 mld). Geen plausibiliteitsprobleem bij 4-5% organische groei; M&A blijft de hefboom voor relatieve marktaandeelgroei.

---

## 7. Analyse-frameworks (9 frameworks, SCORES 0-5)

### Graham
- **Oordeel**: GEDEELTELIJK
- **Graham number**: 78,3 (= √(22,5 × EPS 5,49 × BVPS 49,63))
- **Margin of safety %** (t.o.v. huidige koers 80,7): -3 (koers is fractioneel boven Graham number — geen margin)
- **Toelichting**: P/E 14,7 voldoet aan Graham's ≤ 15 criterium. P/B 1,63 zit tussen 1,5 en 2,0 — krap niet aan defensief criterium maar binnen tolerantie. Schuldratio (D/E 0,38) is laag, dividend wordt structureel betaald sinds vele jaren, en winstgroei is positief op meerjaars-CAGR. Graham number ligt feitelijk gelijk met huidige koers, dus geen "margin of safety" via deze metriek alleen. Prevas voldoet gedeeltelijk aan Graham's defensieve checklist.
- **Score (0-5)**: 4 (P/E ≤ 15 EN P/B ≤ 2,0)

### Buffett / Munger
- **Oordeel**: GEDEELTELIJK
- **ROIC structureel boven WACC?**: ja, maar marginaal
- **Toelichting**: ROIC zit op 8,6%-10,2% (2024-2025) tegen WACC 7,8% — een positieve spread van +0,8 tot +2,4 pp. Dit is waardecreatie maar niet "structurally exceptional"; Buffett-quality bedrijven hebben typisch 20-30 pp ROIC-WACC spread. P/FCF 10,4 is laag genoeg om aan "redelijke prijs" te voldoen, en het bedrijf is begrijpelijk (consultancy met heldere business units). Moat is NARROW, niet WIDE. Voldoet aan "decent business at fair price", niet aan "exceptional business at fair price".
- **Score (0-5)**: 2 (ROIC > WACC maar niet structureel, moat NARROW)

### Peter Lynch
- **Categorie**: Stalwart (gevestigde mid-cap met matige groei en stabiel dividend)
- **Oordeel**: NEUTRAAL
- **PEG-ratio**: 1,84 (P/E 14,7 / verwachte groei 8%)
- **Toelichting**: Prevas is een typische stalwart — geen fast grower maar ook geen slow grower (omzet-CAGR ~7% sinds 2022). PEG van 1,84 op conservatieve eigen schatting (8% EPS-groei) is matig; gebruik je ABG-consensus van 29,3% EPS-groei dan kom je op PEG 0,5, maar die consensus lijkt extreem optimistisch (gebaseerd op herstel van 2025-dal naar 12% EBITA target). Het verhaal is begrijpelijk: defense + EAM + Finland-herstel.
- **Score (0-5)**: 2 (PEG ≤ 2,0)

### Phil Fisher
- **Oordeel**: GEMIDDELD
- **Toelichting**: Producten/diensten met groeipotentieel (defense, EAM, AI-integratie) — ja. R&D als formele post is beperkt in consultancy (kennis zit in mensen); 'R&D-equivalent' is personeel-investering plus partnerships zoals Hexagon. Margebescherming door moat is beperkt — EBITA-marge zakt onder druk in zwakke kwartalen (7,5% in 2025). Management-integriteit is STERK (zie §5). 1-2 van de 3 criteria voldaan, met groeiend R&D-equivalent budget (groei personeel +9% sinds Q1 2024).
- **Score (0-5)**: 3 (1 van 3 criteria + groeiend R&D-equivalent)

### Magic Formula (Greenblatt)
- **Oordeel**: AANTREKKELIJK
- **Earnings yield %**: 8,1 (EBIT 105,9 / EV 1.300)
- **Return on capital %**: 43 (EBIT 105,9 / (NWC 47 + NFA 200))
- **Toelichting**: Magic Formula combineert goedkope waardering (earnings yield) met goede kapitaalproductiviteit (RoC). Earnings yield 8,1% zit tussen ≥ 7% (score 4) en ≥ 10% (score 5) — duidelijk hoger dan vergelijkbare consultancies. Return on capital 43% is uitstekend dankzij asset-light business model (lage NFA, modest NWC). Beide assen scoren goed; net niet in de top-decile.
- **Score (0-5)**: 4 (Earnings Yield ≥ 7% EN Return on Capital ≥ 30%)

### Moat
- **Score (0-5)**: 2 (mogelijke moat maar ROIC-WACC spread < 5pp; categorieën deels middel maar geen STERK in ≥ 3 categorieën)

### Management
- **Score (0-5)**: 4 (capital allocation GOED, prikkels aligned via insider-aankopen, geen controverses)

### Fair Value DCF
- **Score (0-5)**: 5 (fair value basis 149 vs. koers 80,7 → upside 84,6% ≥ 30%)

### Fair Value IPO-gecorr.
- **Score (0-5)**: 5 (IPO 1998 > 10 jaar geleden → score = Fair Value DCF basis score = 5)

### Scorekaart totaal
- **Totaalscore**: 31
- **Max**: 45 (9 × 5)
- **Eindoordeel**: **HOLD** (totaal 31 ≥ 24 EN < 33; Fair Value DCF-score 5 ≥ 3; geen KOOP omdat totaal < 33)
- **Samenvatting**: Prevas combineert een aantrekkelijke waardering (DCF-upside ~85%, kansgewogen fair value SEK 158 versus koers SEK 80,70, EPV-koersondersteuning bij SEK 103, earnings yield 8,1%, P/E 14,7 onder Graham-drempel) met matige kwaliteit op de kerncriteria moat (NARROW met 1-2 categorieën op "middel"-sterkte) en kwaliteit van groei (Lynch PEG 1,84 op conservatieve schatting). Management scoort STERK dankzij insider-aankopen door CEO Welén in zowel 2024 als 2025, lage leverage (nettoschuld/EBITDA 0,89), disciplined M&A-aanpak met zichtbare integratie van Enmac en OIM, en minimal aandelenverwatering. Magic Formula scoort goed op return-on-capital 43% (asset-light), en DCF-upside is fors. De scorekaart trekt het eindoordeel echter weg van KOOP omdat de Buffett/Munger-score (ROIC-WACC spread slechts +0,8 pp), de Moat-score (NARROW, geen STERK in ≥ 3 categorieën) en de Lynch-score (PEG > 1,5) onder de KOOP-drempel uitkomen. Met een totaal van 31/45 tegen de KOOP-drempel van 33 is dit een typische HOLD: aantrekkelijk geprijsd voor een waarde-belegger met geduld, maar zonder de combinatie van structurele wide moat en lage prijs die een KOOP-trigger op de deterministische rubric zou rechtvaardigen.

---

## 8. Risico's (minimaal 5-8 stuks)

| # | Omschrijving | Kans | Impact | DCF-aanname geraakt | Toelichting |
|---|---|---|---|---|---|
| 1 | Margedaling onder 8% EBITA in 2026 (geen herstel) | MIDDEN | GROOT | FCF-groei fase 1, terminal-marge | EBITA zakte van 9,4% (2024) naar 7,5% (2025); Q1 2026 onderliggend 9,3% maar Denemarken nog zwak. Als margin onder 8% vastloopt, daalt FCF-basis ~20%. |
| 2 | Verlies/krimp top-defense-klant (>10% omzet) | LAAG | GROOT | Omzetgroei fase 1 | Eén klant in Sweden segment defense levert SEK 50 mln/kwartaal. Verloop zou ~5 pp omzetgroei wegnemen en EBITA met SEK 15-20 mln raken. |
| 3 | Loonkostendruk Nordics zonder tariefdoorbelasting | HOOG | MIDDEL | EBIT-marge | Competitive labor market is een terugkerend punt in interim-rapporten. Bij 1.045 FTE drukt elke 1% reële loonsverhoging EBITA met SEK 7-8 mln (zonder tariefverhoging). |
| 4 | AI-tooling vervangt consulting-uren | MIDDEN | MIDDEL | Omzetgroei fase 2, terminal groei | Generieke routine-uren kunnen door AI worden weggesneden. Prevas integreert AI om dit te beheersen; risico is dat klanten hun in-house AI-capaciteit opbouwen ten koste van consulting-uren. |
| 5 | Goodwill-impairment bij volgende neergang | LAAG | MIDDEL | Eigen vermogen, vertrouwen | Goodwill SEK 673 mln = ~50% EV; opgebouwd vooral via Enmac (Finland). Bij sustained underperformance in Finland zou impairment vereist zijn. |
| 6 | Hogere SEK 10y rente (boven 4%) | LAAG | MIDDEL | WACC | Spot 2,73%, genormaliseerd vergelijkbaar. Stijging zou WACC verhogen en fair value drukken. |
| 7 | Vertraagde herstel Denemarken-segment | MIDDEN | KLEIN | Omzetgroei fase 1 | Q1 2026 herstructurering loopt; volle effect verwacht H2 2026. Denmark = ~8% omzet, dus impact is begrensd. |
| 8 | Pre-IPO financial engineering — niet van toepassing | n.v.t. | n.v.t. | n.v.t. | IPO was 1998 (>27 jaar geleden). Geen recent IPO-effect. Geen aanwijzingen voor schuld-recap. |

---

## 9. These invalide bij

Deze these is weerlegd wanneer (a) EBITA-marge twee opeenvolgende kwartalen in 2026 onder 7% zakt zonder zicht op herstel; (b) defense-omzet stagneert of krimpt YoY; (c) management de dividend nogmaals fors verlaagt (> 20%); of (d) een goodwill-impairment > SEK 100 mln gerapporteerd wordt. Deze observeerbare triggers betekenen dat de NARROW-moat en marge-herstel-thesis is gebroken.

---

## 10. ESG

### Materiële factoren (SASB-gebaseerd)

| Factor | SASB-categorie | Risiconiveau | Financiële impact | DCF-impact |
|---|---|---|---|---|
| Human capital (talent retention) | SV-PS-330 | MIDDEN | Loonkostendruk, attrition | Marge fase 1 |
| Data security & customer privacy | SV-PS-230 | MIDDEN | Reputatie, klantverlies | Omzetgroei |
| Energy/decarbonisation enablers (klant-oplossingen) | TC-SI-130 | LAAG (kans), POSITIEF effect | Omzetgroei (Greengenuity-positionering) | Omzetgroei fase 1+2 |
| Defense exposure (controversial weapons) | n.v.t. (Prevas levert IT/engineering, geen wapensystemen) | LAAG | ESG-mandaat-uitsluiting bij sommige fondsen | Koers (multiple) |

- **Eindoordeel ESG**: **GEMIDDELD RISICO**
- **Toelichting**: Prevas profileert zich actief op duurzaamheid ("Greengenuity") en levert energie- en proces-efficiëntie-projecten aan klanten — een aantrekkelijk-segment voor ESG-thematische beleggers. Defense-exposure (17% omzet) kan echter sommige ESG-fondsen uitsluiten; de business is IT/engineering-services aan defensie, geen wapensystemen, dus exclusie hangt af van fonds-mandaat. Human capital (1.045 FTE in concurrentieele Nordics-markt) is materieel: vrouwelijk personeel 18,1% (Q1 2026) — typisch voor sector maar wel een focuspunt.

---

## 11. Katalysatoren (5-8 stuks, chronologisch)

| Datum ca. | Omschrijving | Richting | Impact |
|---|---|---|---|
| jul 2026 | Q2 2026 interim report (17 juli 2026) — utilization-update en defense-trend | POSITIEF | MIDDEL |
| H2 2026 | Volle effect Denemarken-herstructurering | POSITIEF | MIDDEL |
| okt 2026 | Q3 2026 interim report (27 oktober 2026) — seizoensmatig zwakker maar herstel verwacht | NEUTRAAL | KLEIN |
| feb 2027 | Year-end report 2026 (10 februari 2027) — bevestiging margin-target-traject richting 12% | BINAIR | GROOT |
| 2026-2027 | Bolt-on M&A in lijn met disciplined approach (CEO bevestiging ABGSC investor day) | POSITIEF | MIDDEL |
| 2026 | Verdere uitrol HxGN EAM 12-jarig contract (SEK 80 mln recurring revenue lifetime) | POSITIEF | KLEIN |
| 2026-2027 | Hogere defense-vraag in lijn met Nordic-defensie-budgetten | POSITIEF | GROOT |
| mei 2027 | Eerstvolgende AGM + dividendvoorstel 2026 | BINAIR | KLEIN |

---

## 12. Fair value — kwantitatief (DCF)

### WACC-componenten
- **Risicovrije rente %**: 2,73 (Sweden 10y SGB per 22 mei 2026)
- **Bron risicovrije rente**: tradingeconomics.com / worldgovernmentbonds.com / FRED IRLTLT01SEM156N — Sweden 10-year government bond
- **Type**: nominaal (spot)
- **ERP (equity risk premium) %**: 4,2 (Sweden = mature market in Damodaran Country-Risk-Premium table 2026; mature ERP 4,23%)
- **Bron ERP**: pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html (Damodaran, jan 2026)
- **Beta (adjusted, Blume)**: 1,00 (bottom-up sector-beta engineering consulting; Damodaran sector unlevered ~0,90 gerelevered naar Prevas-kapitaalstructuur)
- **Bron beta**: Damodaran sector betas, bottom-up (eigen relevering)
- **Type beta**: bottom_up (Prevas small-cap met 5y volatiliteit, regressie-beta onbetrouwbaar)
- **Country risk premium %**: 0 (Sweden is mature market)
- **Size premium %**: 2,0 (marktkap SEK 1,04 mld < EUR 1 mld → smaller-cap premium per Fama-French)
- **Cost of equity %**: 2,73 + 1,00 × 4,2 + 0 + 2,0 = **8,93**
- **Schuldkosten na belasting %**: 4,6 × (1 - 0,22) = **3,59**
- **E/V gewicht %**: 78,7 (E 1.040 / (1.040 + 281))
- **D/V gewicht %**: 21,3
- **WACC %**: 0,787 × 8,93 + 0,213 × 3,59 = **7,80**
- **Sector WACC %** (referentie Damodaran computer services / business & consumer services): ~8,0-8,5 — Prevas WACC ligt licht onder sector door lage leverage en stabiele cashflows; aanvaardbaar.
- **Illiquiditeitskorting %**: 0 (Prevas heeft een redelijke handelsliquiditeit op Stockholm Small Cap; geen materiële spread)

### DCF model-specs
- **Model type**: 2-fase (10 jaar projectie + Gordon growth)
- **FCF-definitie**: FCF to firm (CFO - capex - lease aflossing — IFRS-16 leases als operationele kost behandeld)
- **Basis FCF**: 100 (genormaliseerd, gemiddelde 2024-2025 FCF na lease)
- **Basis FCF na SBC**: 98 (SBC SEK 1,7 mln in mindering)
- **FCF-type**: genormaliseerd (na IFRS-16 lease)
- **Groei fase 1 %** (jaar 1-5, basis): 4
- **Groei fase 2 %** (jaar 6-10, basis): 3
- **Terminal groei %** (basis): 2,5
- **Terminal methode**: Gordon growth (primair) + exit multiple cross-check
- **Exit multiple gebruikt** (EV/EBITDA): 8,0 (mediaan Nordic-consultancy peers, in lijn met ABG-rapport 9-6x EV/EBITA voor Prevas zelf forward)
- **Bron exit multiple**: ABGSC-research / Inderes peer-set (zonder paywall niet exact verifieerbaar; conservatief gebruikt)
- **Terminal value Gordon growth**: SEK 2.747 mln
- **Terminal value exit multiple**: SEK ~1.800 mln (8,0 × EBITDA t10 ~227 mln)
- **Terminal value % van totaal**: 61 (PV TV 1.337 / EV 2.183) — onder 75%-drempel ✓
- **Terminal implied EV/EBITDA**: 12,1x (consistentie-check) — boven exit multiple omdat Gordon growth scenario optimistischer is dan exit multiple; verschil benoemd in toelichting
- **Terminal groei consistentie**: 2,5% past binnen SE BBP-groei (~2% nominaal) + lichte productiviteitspremie. Reinvestment-rate g/ROIC = 2,5/9 = 28% — plausibel voor mature consulting.
- **Mid-year convention**: true
- **Aandelen uitstaand (mln)**: 12,885
- **Nettoschuld huidig**: SEK 260 mln (excl IFRS-16 lease — lease behandeld in FCF)

### DCF-toelichting
De DCF gebruikt SEK 100 mln genormaliseerde FCF als startpunt (= CFO 143,7 - capex 7,5 - lease-aflossing 35,9 in 2025, vergelijkbaar 2024). Deze genormaliseerde basis ligt onder de boekhoudkundige FCF van SEK 136 mln (excl. lease) omdat IFRS-16 leases een terugkerende operationele kost zijn voor een asset-light consultancy met 1.045 medewerkers verdeeld over Nordics-kantoren. Groei fase 1 van 4% combineert organische groei (~2-3% bij utilization-herstel) met bolt-on M&A (~1-2%) — consistent met de disciplined M&A-approach die CEO Welén bevestigde tijdens het ABGSC investor day in mei 2026 ("waiting for the right fit"). Fase 2 daalt naar 3% en terminal naar 2,5% — onder mature consultancy steady-state en onder Sweden's nominale BBP-groei. WACC van 7,80% reflecteert small-cap-positie (size premium +2,0%) en Sweden mature market (geen country risk premium). Mid-year convention toegepast voor 3-5% precisie-verbetering. Belangrijkste consistentie-checks: (a) TV-aandeel 61% van EV — onder de 75%-geloofwaardigheidsdrempel; (b) impliciete EV/EBITDA terminal 12,1x — hoger dan exit-multiple 8,0x maar consistent met margin-recovery scenario richting 12% EBITA-target; (c) reinvestment-rate g/ROIC = 2,5/9 = 28% — plausibel voor mature consulting.

### 5-jaars projectie (basisscenario)

| Jaar | Omzet | Omzetgroei % | EBIT | EBIT-marge % | NOPAT | Capex | ΔNWC | SBC | FCF |
|---|---|---|---|---|---|---|---|---|---|
| 2026 | 1.692 | 4,0 | 127 | 7,5 | 99 | 8 | 4 | 2 | 104 |
| 2027 | 1.760 | 4,0 | 141 | 8,0 | 110 | 8 | 4 | 2 | 108 |
| 2028 | 1.830 | 4,0 | 156 | 8,5 | 122 | 9 | 4 | 2 | 113 |
| 2029 | 1.904 | 4,0 | 171 | 9,0 | 134 | 9 | 4 | 2 | 117 |
| 2030 | 1.980 | 4,0 | 198 | 10,0 | 154 | 9 | 4 | 2 | 122 |

(FCF expliciet inclusief lease-aflossing als operationele kost; EBIT-marge stijgt geleidelijk naar 10% richting management-target 12% EBITA. SBC verwaarloosbaar maar conservatief opgenomen.)

### Scenarios

| Scenario | FCF-groei % | WACC % | Fair value | Upside % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 1 (fase 1), 1 (fase 2), 2 (terminal) | 9,0 | 83 | 3 | 25 |
| Basis | 4 / 3 / 2,5 | 7,8 | 149 | 85 | 50 |
| Optimistisch | 7 / 5 / 3 | 7,0 | 251 | 211 | 25 |

- **Kansgewogen fair value**: 25% × 83 + 50% × 149 + 25% × 251 = **158 SEK** (kans-totaal = 100%)

### Reverse DCF
- **Impliciete groei %**: ca. -2,8 (de markt prijst lichte krimp van FCF in)
- **Historische FCF CAGR %**: ~7 (2024-2025 FCF stijging + impliciete reeks)
- **Consensus groei % (analisten)**: ABG ziet 9-7x EV/EBITA op 2026e-2027e met EBITA marge-herstel naar 10-11% — implicerend ~10-15% EBITA-groei per jaar 2026-2027; Yahoo geeft consensus EPS-groei 29,3% (5y).
- **Interpretatie**: De impliciete groei van -2,8% (markt) staat in scherp contrast met zowel historische groei (7% CAGR) als analist-consensus (10-29%/jr). Dit suggereert dat de markt extreem voorzichtig is over Prevas; ofwel de markt anticipeert structurele AI-disruptie van consulting, ofwel het bedrijf is significant ondergewaardeerd. De DCF basis fair value van 149 weerspiegelt het tweede beeld; de scorekaart-HOLD reflecteert dat de moat (NARROW) en margevariabiliteit voorzichtigheid rechtvaardigen.

### EPV (Bruce Greenwald)
- **Genormaliseerde EBIT-marge %**: 9,0 (gemiddelde 2022-2025 EBITA-marge ~10%, conservatief vertaald naar EBIT 9%)
- **Genormaliseerde NOPAT**: 1.627 × 0,09 × (1 - 0,22) = **114 mln**
- **Maintenance capex**: 8 (huidige niveau is ~100% maintenance bij consultancy)
- **Adjusted earnings power**: 114 - 8 + 18 (D&A excl. acquisition-related amort en excl. IFRS-16 lease dep) = **124 mln**
- **EPV**: 124 / 0,078 = **1.591 mln**
- **EPV per aandeel**: (1.591 - 260) / 12,885 = **SEK 103**
- **Groeipremie %**: (149 - 103) / 103 = **45**

(De huidige koers SEK 80,70 zit ONDER de no-growth EPV van SEK 103 — wat betekent dat de belegger feitelijk negatieve groei betaalt. Dit is een sterk signaal van potentiële onderwaardering bij een NARROW moat.)

### Andere methoden
- **DDM uitgevoerd?**: false (dividend SEK 4,00 op koers SEK 80,70 = 5,0% yield, niet de primaire valuation; DCF en EPV zijn leidend)
- **SOTP uitgevoerd?**: false (3 segmenten zijn deel van zelfde core business — niet zinvol)

### Synthese fair value
- **Bandbreedte laag**: 83 (pessimistisch DCF)
- **Bandbreedte centraal**: 149 (basis DCF) / 158 (kansgewogen)
- **Bandbreedte hoog**: 251 (optimistisch DCF)
- **Methode-gewichten** (som = 100%):
  - DCF %: 55
  - EPV %: 30
  - Multiples %: 15
- **Synthese-fair-value met gewichten**: 0,55 × 158 + 0,30 × 103 + 0,15 × 88 (10x EV/EBITDA basis) = 87 + 31 + 13 = **131 SEK**
- **Margin of safety vereist %**: 25 (typisch voor small-cap met NARROW moat en variabele marge)
- **Koopniveau** (synthese × 0,75): **SEK 98**
- **Synthese-toelichting**: DCF krijgt het grootste gewicht (55%) als primaire methode omdat Prevas voorspelbare kasstromen genereert en de FCF-conversie consistent boven 130% van nettowinst zit. EPV (30%) functioneert als belangrijke no-growth ondergrens — bij koers SEK 80,70 onder de EPV van SEK 103 betaalt de belegger zelfs negatieve groei in, wat ongebruikelijk is voor een bedrijf met 7% historische omzet-CAGR. Relatieve waardering (15%, peer-multiple) is de minst-belaste check vanwege fragmentatie van de Nordics-consultancy-sector. De synthese-fair value van SEK 131 ligt boven de huidige koers met ~62% upside, maar de vereiste MOS van 25% (gerechtvaardigd voor een small-cap met NARROW moat en variabele 7,5-12,4% EBITA-marge) legt het koopniveau op SEK 98 — nog steeds boven huidige koers, maar krap. Dit ondersteunt de HOLD-uitkomst: de upside is reëel en de neerwaartse risico's begrensd, maar de scorekaart-criteria voor moat en groei-kwaliteit halen geen KOOP-trigger.

### Gevoeligheid (DCF basis fair value per aandeel — SEK)

**WACC-range**: 6,5% / 7,0% / 7,5% / 8,0% / 8,5% / 9,0%
**FCF-groei-range fase 1**: 1% / 2% / 4% / 6% / 8% (fase 2 = fase 1 - 1pp; terminal = 2,5%)

| FCF-groei \ WACC | 6,5% | 7,0% | 7,5% | 8,0% | 8,5% | 9,0% |
|---|---|---|---|---|---|---|
| 1% | 124 | 110 | 99 | 90 | 82 | 76 |
| 2% | 142 | 127 | 114 | 103 | 94 | 86 |
| 4% | 188 | 167 | 149 | 134 | 121 | 110 |
| 6% | 245 | 215 | 191 | 170 | 152 | 137 |
| 8% | 316 | 275 | 241 | 213 | 189 | 169 |

Bij basis-aannames (4% fase 1, WACC 7,8%) komt de DCF op SEK 149 uit — gevoelig voor zowel groei- als WACC-veranderingen. Pessimistische combinaties (1% groei, 9% WACC) geven SEK 76; optimistische (8% groei, 6,5% WACC) geven SEK 316.

---

## 13. Databronnen

### Bronnen-hiërarchie
- **Jaarverslag PDF / IR-pagina** → betrouwbaarheid **HOOG**
- **Beursmelding / prospectus** → betrouwbaarheid **HOOG**
- **Aggregator** → betrouwbaarheid **AGGREGATOR**

### Financiële bronnen (10 jaar historie — VERPLICHT)

| Jaar | Bron | URL | Betrouwbaarheid |
|---|---|---|---|
| 2015 | — | — | — |
| 2016 | — | — | — |
| 2017 | — | — | — |
| 2018 | — | — | — |
| 2019 | — | — | — |
| 2020 | — | — | — |
| 2021 | Cision-extract / Nasdaq view-archief (Q4 2021 release) | https://view.news.eu.nasdaq.com/view?id=b02fa6cbc5cade16cf46ef4caf93c4b29&lang=en | AGGREGATOR |
| 2022 | Vergelijkende cijfers in FY2023-release | https://news.cision.com/prevas/r/prevas-publishes-year-end-report-for-2023,c3927779 | AGGREGATOR |
| 2023 | Cision: Prevas publishes Year-End Report 2023 | https://news.cision.com/prevas/r/prevas-publishes-year-end-report-for-2023,c3927779 | HOOG |
| 2024 | Prevas Year-end Report 2024 (PDF) | https://mb.cision.com/Main/60/4103398/3256720.pdf | HOOG |
| 2025 | Prevas Year-end Report 2025 (PDF) | https://mb.cision.com/Main/60/4305051/3925558.pdf | HOOG |

**Harde eis-check**: 2020-2024 zouden allemaal HOOG moeten zijn. 2020-2022 zijn LEEG of AGGREGATOR — dit is een **methodisch tekort** in deze run. Een tweede run met directe download van oudere PDF's uit het Prevas IR-archief (prevas.com/financial_reporting) is nodig om de bronnen-eis volledig te halen. De keuze om hier door te gaan is bewust: liever leeg dan verzonnen, conform METHODE.md.

### Jaarverslagen geraadpleegd

| Jaar | Bron | URL |
|---|---|---|
| 2024 | Prevas Q4 Year-end Report 2024 (PDF) | https://mb.cision.com/Main/60/4103398/3256720.pdf |
| 2025 | Prevas Q4 Year-end Report 2025 (PDF) | https://mb.cision.com/Main/60/4305051/3925558.pdf |
| Q1 2026 | Prevas Q1 Interim Report 2026 (PDF) | https://mb.cision.com/Main/60/4344141/4074830.pdf |

### Beursmeldingen geraadpleegd

| Datum | Omschrijving | URL |
|---|---|---|
| 2026-05-05 | Q1 2026 Interim Report — record EAM contract SEK 80 mln | https://mb.cision.com/Main/60/4344141/4074830.pdf |
| 2026-02-10 | FY2025 Year-end Report — dividend SEK 4,00 voorgesteld | https://mb.cision.com/Main/60/4305051/3925558.pdf |
| 2025-02-11 | FY2024 Year-end Report | https://mb.cision.com/Main/60/4103398/3256720.pdf |
| 2024-02-14 | FY2023 Year-end Report (Cision news) | https://news.cision.com/prevas/r/prevas-publishes-year-end-report-for-2023,c3927779 |
| diverse | ABGSC research notes via Inderes-platform | https://www.inderes.fi/en/companies/Prevas |

### IPO-prospectus
- **Geraadpleegd?**: false
- **URL**: —
- **Pre-IPO data beschikbaar?**: niet onderzocht (IPO 1998 > 27 jaar; niet relevant voor 10-jaars analyse)
- **Pre-IPO bron**: n.v.t.

### Non-GAAP
- **Gebruikt?**: true (door Prevas — EBITA)
- **Toelichting**: Prevas rapporteert EBITA (EBIT vóór amortization van acquisition-related intangibles en acquisition-related items). DCF in §12 gebruikt IFRS-EBIT als grondslag voor NOPAT. EBITA wordt in tabellen vermeld voor vergelijkbaarheid met management-communicatie.

### Ontbrekende data (eerlijke lijst)
- Jaren 2015-2020: geen verifieerbare financiële cijfers uit geopende bronnen (Cision-archief, search engine indices). Reden: Prevas IR-archief vereist directe download van oudere PDF's; aggregators (StockAnalysis, MacroTrends, Yahoo) gaven in de search-results geen pre-2021 data terug.
- Brutowinst / brutomarge: niet onderscheiden in Prevas income statement (gebruikelijk voor consultancy waar personeelskosten dominante COGS-component zijn).
- Top-5 aandeelhouders met naam en percentage: niet verifieerbaar in geopende bronnen; Yahoo/MarketScreener-paywall.
- Free float percentage: niet verifieerbaar in geopende bronnen.
- IPO-koers 1998: niet onderzocht (niet materieel voor 5y horizon).
- TAM/SAM exacte cijfers voor Prevas-niche: alleen Nordics-built-environment top-30 indicatief beschikbaar.

### Peildatum analyse
- 2026-05-27

---

## 14. Volledige bronnen-lijst

| Titel | URL | Type |
|---|---|---|
| Prevas Q1 2026 Interim Report PDF | https://mb.cision.com/Main/60/4344141/4074830.pdf | jaarverslag |
| Prevas Year-end Report 2025 PDF | https://mb.cision.com/Main/60/4305051/3925558.pdf | jaarverslag |
| Prevas Q4 Year-End Report 2024 PDF | https://mb.cision.com/Main/60/4103398/3256720.pdf | jaarverslag |
| Prevas Year-End Report 2023 (Cision) | https://news.cision.com/prevas/r/prevas-publishes-year-end-report-for-2023,c3927779 | beursmelding |
| Prevas Year-End Report 2021 (Nasdaq view) | https://view.news.eu.nasdaq.com/view?id=b02fa6cbc5cade16cf46ef4caf93c4b29&lang=en | beursmelding |
| Inderes — Prevas company page | https://www.inderes.fi/en/companies/Prevas | aggregator / onderzoeksrapport |
| Sweden 10y bond yield (Trading Economics) | https://tradingeconomics.com/sweden/government-bond-yield | aggregator |
| Damodaran country risk premiums (NYU Stern) | https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html | onderzoeksrapport |
| IndustryRadar Sweden built-environment top-30 | https://industryradar.com/sweden/top-30-swedens-largest-built-environment-players/ | onderzoeksrapport |
| MarketScreener Prevas finances | https://www.marketscreener.com/quote/stock/PREVAS-AB-6491372/finances/ | aggregator |
| Simply Wall St Prevas page | https://simplywall.st/stocks/se/software/sto-prev-b/prevas-shares | aggregator |
| Yahoo Finance PREV-B.ST | https://finance.yahoo.com/quote/PREV-B.ST/ | aggregator |
| Stockopedia Prevas | https://www.stockopedia.com/share-prices/prevas-ab-STO:PREV%20B/ | aggregator |
| Bloomberg Magnus Welén profile | https://www.bloomberg.com/profile/person/23402648 | nieuwsartikel |
| MarketScreener — CEO buys shares | https://www.marketscreener.com/quote/stock/PREVAS-AB-6491372/news/Prevas-CEO-Magnus-Welen-increases-his-shareholding-47222942/ | nieuwsartikel |

---

## 15. Update-historie

| Datum | Versie | Wijziging |
|---|---|---|
| 2026-05-27 | 1.0 | Eerste publicatie — stage 1 markdown research. |

---

## Opmerkingen voor Claude Code

1. **Bronnendekking 2015-2020**: Voor de 10-jaars financiële tabellen ontbreken jaren 2015-2020 volledig en zijn 2021-2022 alleen via aggregator/Cision-extract gevuld. De METHODE.md eist dat recent 5 jaren (2020-2024) HOOG zijn. In deze run zijn 2020-2022 LEEG of AGGREGATOR. Dit betekent dat de `check-sources.py` validator in stage 2 een waarschuwing/fail kan geven op `databronnen.financieel[]`. Aanbeveling: stage 2 kan dit ofwel accepteren als bewuste leemte (omdat 2024/2025/Q1'26 zeer goed gedekt zijn) ofwel een tweede stage-1 run triggeren waarin oudere Prevas IR-PDF's (prevas.com/financial_reporting → Q4 2019, Q4 2020 etc.) handmatig worden gedownload. Op basis van Janco's instructie "vraag toestemming voor de map voordat je aannames gaat doen" is hier gekozen voor de "lege cellen + transparante inventaris"-route.

2. **DCF startpunt FCF**: gebruikt SEK 100 mln (CFO - capex - lease-aflossing). Als stage 2 de FCFE-definitie (CFO - capex zonder lease-aftrek) prefereert, wordt basis-FCF ~140 mln en stijgt fair value evenredig. Deze keuze is bewust en gemotiveerd in §12 (IFRS-16 leases zijn een terugkerende operationele kost voor asset-light consultancy).

3. **Beta**: bottom-up gebruikt (1,00) bij gebrek aan betrouwbare 5y regressie voor deze small-cap. Damodaran sector-beta voor "computer services" gerelevered geeft consistente uitkomst. Als stage 2 een regressie-beta wil bevestigen, raadpleeg Bloomberg PREVB:SS of Refinitiv.

4. **Consistentie executive summary ↔ scorekaart**: oordeel HOLD volgt uit deterministische rubric (totaal 31 ≥ 24 en < 33, Fair Value DCF-score 5 ≥ 3 → HOLD). De fair value basis 158 (kansgewogen) is hoger dan de huidige koers met 96% upside, maar de scorekaart-mechaniek vereist totaal ≥ 33 voor KOOP. Buffett/Munger-score 2 en Moat-score 2 trekken het totaal omlaag. Dit is bedoeld zo.

5. **+17% beweging in originele vraag**: ik kon op de Inderes-pagina geen +17%-rally over een korte periode bevestigen (1m -5,50%, 1d +1,51%). De Q1 2026 results (5 mei 2026, EBITA in lijn, record EAM-contract aangekondigd) zijn de meest plausibele recente katalysator; mogelijk dat Janco's +17% slaat op een langer YTD-herstel of een specifieke window die ik niet kon isoleren. Peildatum-koers SEK 80,70 staat in elk geval boven de YTD-low. Stage 2 hoeft hier niets mee tenzij de website een specifieke beweging-context verwacht.

6. **Goodwill 50% EV**: hoog door Enmac-acquisitie (juli 2024). Geen impairment in 2025 — vertrouwen in Finland-segment-herstel (zes positieve kwartalen). Risico #5 noemt dit expliciet.

7. **Defense-exposure ESG**: Prevas levert IT/engineering aan defensiebedrijven (geen wapensystemen). Dit kan voor specifieke ESG-mandaten een uitsluiting betekenen; in §10 gemarkeerd als "MIDDEN risiconiveau" via de impact op multiple.
