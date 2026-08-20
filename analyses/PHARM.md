# Research: PHARM — Pharming Group N.V.

> Opgesteld volgens `research/METHODE.md` en `research/TEMPLATE.md`.
> Peildatum 17 augustus 2026. Rapportagevaluta van het bedrijf is **US dollar**;
> de notering is in **euro**. Zie de valuta-notitie bij sectie 3.

---

## Bronnen-inventaris (Stap 0.5)

**Overzichtspagina's die ik daadwerkelijk heb geopend**

1. `https://www.pharming.com/investors/financial-documents` — volledige IR-index met alle
   jaarverslagen (2018-2025) en alle kwartaal-/jaarpersberichten vanaf 2018.
2. `https://www.pharming.com/investors/sec-filings` — index van de 20-F's (Pharming is
   Nasdaq-genoteerd via ADS's en dus SEC-filer).
3. `https://www.sec.gov/Archives/edgar/data/1828316/000182831626000015/FilingSummary.xml` —
   R-pagina-index van de FY2025 20-F, gebruikt om de gecontroleerde jaarrekening rechtstreeks
   te lezen (R3 = winst-en-verliesrekening, R5 = balans, R7 = kasstroomoverzicht).

Er is **geen IR-Excel met meerjarige kerncijfers** op de site; wel een externe
"interactive financials"-link (Virtua Research) die geen bruikbare export geeft.
Er is dus geen haallijst-regel voor een spreadsheet.

---

```
Jaar 2025 — HOOG
  Bron: Pharming Group N.V. Form 20-F FY2025 (SEC, geauditeerd) + persbericht 4Q/FY2025
  URL:  https://www.sec.gov/Archives/edgar/data/1828316/000182831626000015/R3.htm
        https://www.sec.gov/Archives/edgar/data/1828316/000182831626000015/R5.htm
        https://www.sec.gov/Archives/edgar/data/1828316/000182831626000015/R7.htm
        https://www.pharming.com/files/pharming-group-reports-4q25-fy25-results-en-12mar26.pdf
  Daadwerkelijk geopend: ja (alle vier)
  Cijfers overgenomen: omzet, omzet per product, kostprijs, brutowinst, R&D, G&A,
                       marketing & sales, EBIT, financieringsresultaat, belasting,
                       nettoresultaat, EPS, totale activa, eigen vermogen, converteerbare
                       obligaties, leaseverplichtingen, voorraden, kas, marketable
                       securities, CFO, capex, SBC, betaalde belasting, afschrijvingen,
                       werkkapitaalmutaties, Abliva-overnamebalans, aandelen ultimo
  Cijfers NIET overgenomen: gewogen gemiddeld aantal aandelen (niet apart getoond in de
                       R-pagina's); segment-/geografie-tabel (R11 en R42 renderen leeg)

Jaar 2024 — HOOG
  Bron: Form 20-F FY2025 (vergelijkende kolom) + persbericht 4Q/FY2025
  URL:  zie 2025
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: volledige W&V, balans en kasstroom 2024
  Cijfers NIET overgenomen: gewogen gemiddeld aantal aandelen

Jaar 2023 — HOOG
  Bron: Form 20-F FY2025 (tweede vergelijkende kolom, herzien) + persbericht 4Q/FY2023
  URL:  https://www.sec.gov/Archives/edgar/data/1828316/000182831626000015/R3.htm
        https://www.pharming.com/files/pharming-group-4q-2023-financial-results-en-14mar2024-final.pdf
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: volledige W&V en kasstroom (20-F-versie), balansposten (persbericht)
  Let op: het 20-F rapporteert 2023 herzien — nettoverlies US$10,5 mln en CFO
         US$(17,3) mln, tegen US$(10,1) mln en US$(17,5) mln in het oorspronkelijke
         persbericht. Ik gebruik consequent de 20-F-cijfers.

Jaar 2022 — HOOG
  Bron: Pharming persbericht 4Q/FY2023 (vergelijkende kolom, geauditeerd overgenomen)
  URL:  https://www.pharming.com/files/pharming-group-4q-2023-financial-results-en-14mar2024-final.pdf
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: omzet, kostprijs, brutowinst, overige baten, R&D, G&A, M&S, EBIT,
                       financieringsresultaat, belasting, nettowinst, EPS, totale activa,
                       eigen vermogen, converteerbare obligaties, leases, kas, CFO, capex
  Cijfers NIET overgenomen: SBC-last 2022 (alleen "share-based compensation settled"
                       US$2,281 mln vermeld, dat is niet de last) — cel blijft leeg

Jaar 2021 — HOOG
  Bron: Pharming persbericht FY2021 + Annual Report 2021 (PDF)
  URL:  https://www.pharming.com/files/pharming-group-financial-results-full-year-2021-final-1.pdf
        https://www.pharming.com/sites/default/files/imce/Public%20Documents/Annual%20report%202022%2006APR22%20compressed.pdf
  Daadwerkelijk geopend: ja (beide)
  Cijfers overgenomen: volledige W&V, balans, kasstroom, SBC
  Bijzonderheid: dit is het eerste boekjaar in US dollar; 2020 is hierin herrekend.

Jaar 2020 — HOOG (dubbel gedekt, twee valuta)
  Bron A (USD): Pharming persbericht FY2021, vergelijkende kolom
  URL:  https://www.pharming.com/files/pharming-group-financial-results-full-year-2021-final-1.pdf
  Bron B (EUR): Pharming persbericht FY2020 (oorspronkelijke rapportage in euro)
  URL:  https://www.pharming.com/files/2020-full-year-results-pr-04mar21-final-pdf.pdf
  Daadwerkelijk geopend: ja (beide)
  Cijfers overgenomen: volledige W&V, balans en kasstroom in BEIDE valuta

Jaar 2019 — HOOG (alleen EUR)
  Bron: Pharming persbericht FY2020 (vergelijkende kolom, herzien) en persbericht FY2019
  URL:  https://www.pharming.com/files/2020-full-year-results-pr-04mar21-final-pdf.pdf
        https://www.pharming.com/files/pr-full-year-results-2019-1300-analyst-call-sem-1.pdf
  Daadwerkelijk geopend: ja (beide)
  Cijfers overgenomen: volledige W&V, balans, kasstroom, gewogen gemiddeld aantal aandelen
  Let op: het FY2020-persbericht herrubriceert 2019 (R&D EUR 28,4 mln i.p.v. 32,9 mln;
         G&A EUR 18,9 mln i.p.v. 14,3 mln; totale operationele kosten identiek) en herziet
         de totale activa naar EUR 228,2 mln en CFO naar EUR 66,5 mln. Ik gebruik de
         herziene cijfers.
  GEEN USD-cijfers beschikbaar: het bedrijf heeft 2019 nooit in dollar herrekend.

Jaar 2018 — HOOG (alleen EUR)
  Bron: Pharming persbericht FY2019 (vergelijkende kolom) en persbericht FY2018
  URL:  https://www.pharming.com/files/pr-full-year-results-2019-1300-analyst-call-sem-1.pdf
        https://www.pharming.com/files/1-pr-full-year-results-2018.pdf
  Daadwerkelijk geopend: ja (beide)
  Cijfers overgenomen: volledige W&V, balans, kasstroom, gewogen gemiddeld aantal aandelen

Jaar 2017 — HOOG (alleen EUR)
  Bron: Pharming persbericht FY2018, vergelijkende kolom
  URL:  https://www.pharming.com/files/1-pr-full-year-results-2018.pdf
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: omzet (product + licentie), kostprijs, brutowinst, overige baten,
                       R&D, G&A, M&S, EBIT, financieringsresultaat, belasting, nettoverlies,
                       EPS, totale activa, eigen vermogen, kas, leningen, CFO, capex
  Cijfers NIET overgenomen: gewogen gemiddeld aantal aandelen (alleen af te leiden, niet
                       gerapporteerd) — cel blijft leeg

Jaar 2016 — HOOG (alleen EUR)
  Bron: Pharming Group Report on Preliminary Financial Results for 2016
  URL:  https://www.pharming.com/sites/default/files/imce/Public%20Documents/2016/PR%20Full%20Year%20Results%202016.pdf
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: volledige W&V, balans, kasstroom, capex
  Cijfers NIET overgenomen: verwaterde EPS, gewogen gemiddeld aantal aandelen, SBC

Jaar 2015 — HOOG (alleen EUR)
  Bron: Pharming Group Report on Preliminary Financial Results for 2016, vergelijkende kolom
  URL:  https://www.pharming.com/sites/default/files/imce/Public%20Documents/2016/PR%20Full%20Year%20Results%202016.pdf
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: volledige W&V, balans, kasstroom, capex
  Cijfers NIET overgenomen: verwaterde EPS, gewogen gemiddeld aantal aandelen, SBC

TTM (1H2026 + FY2025 − 1H2025) — HOOG
  Bron: Pharming persbericht 2Q/1H2026 en presentatie 2Q/1H2026
  URL:  https://www.pharming.com/files/pharming-group-reports-2q-1h26-results-en-30july2026.pdf
        https://www.pharming.com/files/pharming-2q-1h2026-presentation-final-30july2026.pdf
        https://www.pharming.com/files/2q-1h-2026-results-call-transcript-30july26.pdf
  Daadwerkelijk geopend: ja (alle drie)
  Cijfers overgenomen: halfjaaromzet per product, brutowinst, R&D, SG&A, operationeel
                       resultaat, nettoresultaat, EPS, kaspositie, converteerbare
                       obligaties, CFO, capex, aandelen ultimo juni, herziene guidance
  Berekend als TTM = FY2025 − 1H2025 + 1H2026 (alleen voor stroomgrootheden).
```

**Marktgegevens en waarderingsinput**

```
Koers                — Beursgenoten (ShareCompany/BIQH via Euronext), 17-08-2026 17:35,
                       EUR 1,024 (vorige slot 0,994); bevestigd door CentralCharts
                       (EUR 1,0240, +3,04%, volume 6.806.742).
                       https://www.beursgenoten.nl/koersen/euronext-aandelen-amsterdam/pharming/koers
                       https://www.centralcharts.com/en/8697-pharming-group/quotes
EUR/USD              — 1,157 op 14-08-2026 (laatst beschikbare noteringsdag in de
                       gebruikte reeks). Portfolio Dividend Tracker FX-reeks.
Risicovrije rente    — US 10-jaars Treasury 4,63% op 13-08-2026 (FRED, serie DGS10).
                       https://fred.stlouisfed.org/series/DGS10
ERP                  — Damodaran implied ERP 4,23% (jaarultimo 2025, update januari 2026).
                       https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html
Beta (bottom-up)     — Damodaran industry betas, januari 2026: Drugs (Biotechnology)
                       unlevered gecorrigeerd voor kas 1,08 (496 bedrijven); Drugs
                       (Pharmaceutical) 0,92 (228 bedrijven).
                       https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/Betas.html
Sector-WACC          — Damodaran cost of capital VS, januari 2026: Drugs (Biotechnology)
                       8,49%; Drugs (Pharmaceutical) 7,85%.
                       https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/wacc.html
Analistenconsensus   — StockAnalysis (aggregator), 4 analisten, koersdoel US$34,35 per ADS,
                       omzetraming 2026 US$403,5 mln. https://stockanalysis.com/stocks/phar/forecast/
Aandeelhouders       — Simply Wall St (aggregator). Geen AFM-registeruittreksel kunnen
                       openen; zie ontbrekende data.
```

**Zelf-check (regel 6 van Stap 0.5).** Voor elke ingevulde numerieke cel in dit rapport is
hierboven een bron-URL aan te wijzen die ik daadwerkelijk heb geopend. Waar dat niet lukte —
gewogen gemiddeld aantal aandelen per jaar, SBC-last 2015/2016/2022, geografische
omzetsplitsing, bestuurdersbeloning 2025, personeelsaantal 2025, individuele
insidertransacties — blijft de cel **leeg** en staat het punt in sectie 13 onder
ontbrekende data.

**Haallijst — wat ik aan Janco vraag (open, niet blokkerend)**

De analyse is compleet zonder deze bestanden; ze zouden bij de volgende update wel drie
gaten dichten. Als je ze wilt ophalen, zet ze in
`C:\Users\janco\aandelenanalyse\research\_bronnen\PHARM\`:

```
1. PHARM-FY2025-20F.pdf  (dekt 2025 + 2024 + 2023)
   https://www.sec.gov/Archives/edgar/data/1828316/000182831626000015/pharm-20251231.htm
   Nodig voor: Item 6.B beloning bestuur 2025, Item 6.D personeelsaantal,
               Item 7.A grootaandeelhouders, en de omzetsplitsing naar geografie

2. PHARM-AR2025.pdf  (dekt 2025)
   https://annualreport.pharming.com/
   Nodig voor: remuneratierapport 2025 (STI/LTI-criteria, CEO pay ratio)

3. PHARM-Remuneration-2024.pdf
   https://www.pharming.com/sites/default/files/imce/Public%20Documents/2025/2024_Pharming_Remuneration%20Report.pdf
   Nodig voor: beloningsstructuur en aandelenbezitseis CEO
```

Ik heb het 20-F en het online jaarverslag wél geopend, maar de fetch levert bij deze twee
alleen de voorkant en de inhoudsopgave op — de documenten zijn te groot om in één keer door
te lezen. Dat is dus `BRON BESTAAT, NIET OPGEHAALD`, geen `GEEN BRON BESCHIKBAAR`.

---

## Metadata

- **Ticker (bare):** PHARM
- **Yahoo symbol:** PHARM.AS
- **Exchange:** ENXTAM (Euronext Amsterdam); tweede notering Nasdaq (PHAR, 1 ADS = 10 gewone aandelen)
- **Sector (GICS-achtig):** Gezondheidszorg
- **Industrie:** Biofarmacie — weesgeneesmiddelen
- **Land:** Nederland
- **Peildatum analyse:** 2026-08-17
- **Koers op peildatum:** 1,024
- **Valuta:** EUR (rapportagevaluta bedrijf: USD; EUR/USD 1,157)
- **Marktkapitalisatie:** EUR 0,72 mld (US$ 0,84 mld)
- **Marktkap in mln (lokale valuta):** 725
- **Free float pct:** ~98
- **Indexlidmaatschap:** Euronext AMX (Amsterdam Midkap), opgenomen in 2025
- **Domein:** pharming.com

---

## 1. Executive summary

- **Kernthese:**

Pharming Group is een Nederlands biofarmaceutisch bedrijf uit Leiden dat twee goedgekeurde
medicijnen voor zeldzame ziekten verkoopt. RUCONEST behandelt acute aanvallen van erfelijk
angio-oedeem, een zeldzame aandoening waarbij levensbedreigende zwellingen optreden; Joenja
is het enige goedgekeurde middel tegen APDS, een erfelijke afweerstoornis. Beide producten
komen uit een niche waar weinig alternatieven zijn en de prijzen hoog liggen: de brutomarge
bedraagt bijna 88 procent. De omzet groeide van US$212 miljoen in 2020 naar US$376 miljoen
in 2025. Toch verdient het bedrijf nauwelijks geld, omdat het meer dan een kwart van zijn
omzet aan onderzoek uitgeeft en bijna 35 procent aan verkoop en marketing. In 2026 kantelde
het beeld: RUCONEST verliest in de Verenigde Staten terrein aan de eerste pil voor acute
aanvallen, en de omzetverwachting ging in juli met US$30 miljoen omlaag. De hele
beleggingscasus rust nu op Joenja, dat 37 procent groeit, en op twee studie-uitkomsten die
eind 2026 en in 2027 komen.

- **Oordeel:** **PASS**
- **Fair value basis** (basisscenario, EUR): 0,38
- **Fair value kansgewogen**: 0,54
- **EPV per aandeel**: 0,25
- **Upside pct**: −63,3
- **Fair value scenarios:**

| Scenario | Fair value | Upside % | FCF groei % | WACC % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 0,15 | −85,8 | +10,0 | 11,43 | 25 |
| Basis | 0,38 | −63,3 | +22,4 | 10,43 | 50 |
| Optimistisch | 1,25 | +22,2 | +53,5 | 9,93 | 25 |

- **Reverse-DCF impliciete groei pct**: 29,6 (vrije kasstroom, tien jaar lang, daarna 2%)
- **Grootste kans**: een positieve fase-II-uitkomst voor leniolisib bij CVID eind 2026 tilt het
  omzetpotentieel van Joenja volgens het management boven de één miljard dollar.
- **Grootste risico**: RUCONEST is 80% van de omzet en wordt structureel uitgehold door orale
  concurrentie, terwijl de kostenbasis van US$317 miljoen bijna de volledige brutowinst opeet.

---

## 2. Bedrijfsprofiel

- **Beschrijving:**

Pharming Group maakt en verkoopt medicijnen voor zeer zeldzame ziekten. Het bedrijf heeft twee
producten op de markt. RUCONEST (conestat alfa) is een recombinante variant van het menselijke
C1-esteraseremmer-eiwit en wordt gebruikt om acute aanvallen van erfelijk angio-oedeem (HAE) te
stoppen. Bij die aandoening ontbreekt of faalt een eiwit dat zwellingen afremt, waardoor
patiënten onvoorspelbare, soms levensbedreigende zwellingen krijgen in het gezicht, de
luchtwegen of de buik. Het eiwit wordt op een unieke manier gemaakt: in de melk van transgene
konijnen, waarna het wordt gezuiverd en tot injectievloeistof verwerkt. Joenja (leniolisib) is
een tablet die de overactieve PI3K-delta-route remt en het enige middel dat wereldwijd is
goedgekeurd voor APDS, een erfelijke immuunstoornis die naar schatting één op de honderdduizend
mensen treft. Pharming zit in de waardeketen helemaal aan het eind: het ontwikkelt of licentieert
het molecuul, regelt de productie via partners, doet de klinische studies, haalt de goedkeuring
binnen en verkoopt zelf aan artsen en gespecialiseerde apotheken. De omzet ontstaat per
verkochte behandeling en loopt in de Verenigde Staten vrijwel volledig via twee specialty
pharmacies; er zijn geen abonnementen of langlopende contracten. Wat het bedrijf onderscheidt is
niet schaal — het is klein — maar het feit dat het in twee kleine indicaties een product heeft
dat concurrenten niet of nauwelijks kunnen kopiëren.

- **Geschiedenis:**

Pharming werd in 1988 in Leiden opgericht als pionier in transgene biotechnologie. Wereldwijde
bekendheid kwam op 16 december 1990 met de geboorte van Herman, de eerste transgene stier ter
wereld, wiens nakomelingen menselijk lactoferrine in hun melk produceerden. Het bedrijf ging op
16 juni 1999 naar de Amsterdamse beurs. De eerste vijftien jaar waren zwaar: de transgene
technologie werkte, maar er kwam lang geen product uit, en begin deze eeuw kwam Pharming in
ernstige financiële problemen nadat een samenwerkingspartner afhaakte. Het keerpunt was 2010,
toen RUCONEST als eerste door een Nederlands biotechbedrijf zelf ontwikkelde therapie Europese
goedkeuring kreeg; in 2014 volgde de FDA en de Amerikaanse lancering. Commercieel werd het pas
serieus toen Pharming in december 2016 alle Noord-Amerikaanse rechten op RUCONEST terugkocht van
Valeant voor US$60 miljoen vooraf plus tot US$65 miljoen aan mijlpalen — gefinancierd met EUR 104
miljoen aan claimemissie, converteerbare obligaties en een lening tegen 8,25%. Vanaf dat moment
verkocht Pharming zelf in de Verenigde Staten en schoot de omzet omhoog van EUR 16 miljoen (2016)
naar EUR 135 miljoen (2018). In 2019 licentieerde Pharming leniolisib van Novartis, opende het
een Amerikaans hoofdkantoor, en op 23 december 2020 kreeg het een tweede notering op Nasdaq. In
maart 2023 keurde de FDA Joenja goed — het eerste medicijn ooit voor APDS. In 2023 droeg
oprichter-CEO Sijmen de Vries het stokje over aan Fabrice Chouraqui. In 2025 nam Pharming het
Zweedse Abliva over voor circa US$68 miljoen, waarmee napazimone (KL1333) voor mitochondriale
ziekten aan de pijplijn werd toegevoegd, en werd het aandeel opgenomen in de AMX-index. In juli
2026 volgde de eerste echte tegenslag onder de nieuwe CEO: de omzetverwachting ging met US$30
miljoen omlaag door concurrentie voor RUCONEST in de Verenigde Staten.

- **Bedrijfsmodel:**

Pharming verdient geld met de verkoop van twee receptgeneesmiddelen tegen weesgeneesmiddel-
prijzen. Er is geen terugkerende abonnementsomzet: elke verkochte dosis is een losse transactie,
al is de onderliggende patiëntenpopulatie chronisch en dus relatief stabiel. RUCONEST leverde in
2025 US$317,9 miljoen op (84,5% van de omzet), Joenja US$58,2 miljoen (15,5%). De brutomarge ligt
rond 88%, want de productiekosten van een biologisch eiwit zijn laag ten opzichte van de prijs.
De echte kosten zitten aan de commerciële en de onderzoekskant: US$131,0 miljoen marketing en
verkoop en US$100,4 miljoen R&D in 2025. Het verdienmodel is dus: hoge marge per eenheid, kleine
patiëntenpopulatie, en een kostenbasis die vrijwel vast is. Groei komt uit meer patiënten, nieuwe
landen en nieuwe indicaties — niet uit prijsverhoging of volume-efficiëntie.

- **IPO-context:**

Pharming noteert sinds 16 juni 1999 aan Euronext Amsterdam en heeft sindsdien meermalen zwaar
verwaterd om de kas te vullen: rond de terugkoop van de Noord-Amerikaanse RUCONEST-rechten in
2016 kwamen er een claimemissie, twee soorten converteerbare obligaties en 88 miljoen warrants
bij. De tweede notering op Nasdaq (ADS's, 23 december 2020) veranderde de kapitaalstructuur niet
— het waren bestaande aandelen. De laatste grote kapitaalmarktactie was april 2024, toen
converteerbare obligaties met looptijd tot 2029 werden geplaatst (US$104,5 miljoen opbrengst) en
de oude lening voor US$134,9 miljoen werd teruggekocht.

- **Klantprofiel:** B2B, extreem geconcentreerd, chronische patiëntenpopulatie
- **Oprichtingsjaar**: 1988
- **IPO-datum**: 1999-06-16 (Euronext Amsterdam); 2020-12-23 (Nasdaq ADS)
- **IPO-koers** (lokale valuta): —
- **Personeel** (FTE): 415 per ultimo 2023 (laatste geverifieerde stand)
- **Landen actief**: Verenigde Staten (kernmarkt), Duitsland, Verenigd Koninkrijk, Israël, Australië, Japan
- **Klantconcentratie**: twee Amerikaanse specialty pharmacies waren in 2025 samen goed voor
  US$290,9 miljoen ofwel **77% van de totale omzet** (2024: US$227,7 miljoen, eveneens 77%)

### Geografische spreiding (omzet)

| Regio | Omzet % | Valuta-exposure |
|---|---|---|
| Verenigde Staten | — | USD |
| Overig | — | EUR, GBP, JPY, ILS, AUD |

**Toelichting geografie:** De exacte omzetsplitsing naar land staat in de segmentnoot van het
20-F, maar die tabel rendert leeg in de XBRL-viewer en is dus niet geverifieerd; de cellen
blijven daarom leeg. Wat wél hard is: 77% van de omzet loopt via twee Amerikaanse specialty
pharmacies, en Pharming heeft in 2026 de verkoop van RUCONEST buiten de Verenigde Staten
bewust beëindigd. Het bedrijf rapporteert sinds 2021 in dollars juist omdat de kasstromen
overwegend in dollars ontstaan — dat is een natuurlijke hedge tegen de dollarkosten van de
Amerikaanse verkooporganisatie, maar het betekent voor een Europese belegger dat zowel de
omzet als de waarde in dollars luidt terwijl het aandeel in euro's noteert.

### Segmenten

| Naam | Omzet % | Beschrijving |
|---|---|---|
| RUCONEST | 84,5 | Recombinante C1-esteraseremmer voor acute aanvallen van erfelijk angio-oedeem; verkocht sinds 2010 (EU) en 2014 (VS); in 2025 US$317,9 mln. |
| Joenja (leniolisib) | 15,5 | Orale PI3K-delta-remmer, enige goedgekeurde therapie voor APDS; gelicentieerd van Novartis, FDA-goedkeuring maart 2023; in 2025 US$58,2 mln. |

### Aandeelhouders (top 5)

| Naam | Belang % | Type (oprichter / PE / institutioneel / retail) |
|---|---|---|
| BlackRock, Inc. | 3,02 | institutioneel |
| Arrowstreet Capital LP | 2,99 | institutioneel |
| Acadian Asset Management LLC | 2,98 | institutioneel |
| Norges Bank Investment Management | 2,93 | institutioneel |
| Deutsche Asset & Wealth Management | 2,13 | institutioneel |

- **Institutioneel eigendomstrend:** Het aandeelhouderschap is uitzonderlijk versnipperd. Volgens
  Simply Wall St houden instellingen samen 20,8%, individuele insiders 1,58% en het brede publiek
  77,6%; de top-25 heeft samen 26,4%. Er is dus geen controlerende partij en geen ankeraandeelhouder
  die het management corrigeert. Een trendoordeel (stijgend of dalend) kan ik niet onderbouwen
  omdat ik geen tijdreeks uit het AFM-register heb kunnen openen.

---

## 3. Financieel — historische data (10 jaar + TTM)

> **Valuta-breuk — lees dit eerst.** Pharming heeft per 1 januari 2021 de rapportagevaluta
> gewijzigd van euro naar US dollar en heeft daarbij alleen 2020 herrekend. Voor 2015 tot en met
> 2019 bestaan er geen door het bedrijf gepubliceerde dollarcijfers. Ik reken die jaren **niet**
> zelf om — een zelfgekozen wisselkoers is een verzonnen getal. In plaats daarvan staan hieronder
> twee tabellen: de dollarreeks 2020-2025 (plus TTM) en de eurorekening 2015-2020. Boekjaar 2020
> komt in beide voor en dient als brug: EUR 185,7 mln omzet werd US$212,2 mln, een impliciete
> koers van 1,143.

### Resultatenrekening A — zoals gerapporteerd in **US$ mln** (2020-2025 + TTM)

| Jaar | Omzet | Omzetgroei % | Brutowinst | Brutomarge % | EBIT | EBIT-marge % | EBITDA | EBITDA-marge % | Nettowinst | Nettomarge % | EPS | EPS-groei % | Aandelen mln |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2020 | 212,2 | — | 188,6 | 88,9 | 76,3 | 35,9 | — | — | 37,7 | 17,8 | 0,058 | — | 638,8 |
| 2021 | 198,9 | −6,3 | 177,7 | 89,4 | 13,6 | 6,8 | — | — | 16,0 | 8,0 | 0,025 | −56,9 | 648,7 |
| 2022 | 205,6 | +3,4 | 188,1 | 91,5 | 18,2 | 8,9 | — | — | 13,7 | 6,7 | 0,021 | −16,0 | — |
| 2023 | 245,3 | +19,3 | 220,1 | 89,7 | −5,4 | −2,2 | 10,5 | 4,3 | −10,5 | −4,3 | −0,016 | n.v.t. | — |
| 2024 | 297,2 | +21,2 | 261,8 | 88,1 | −8,6 | −2,9 | 7,4 | 2,5 | −11,8 | −4,0 | −0,018 | n.v.t. | — |
| 2025 | 376,1 | +26,6 | 330,6 | 87,9 | 25,8 | 6,9 | 37,1 | 9,9 | 2,5 | 0,7 | 0,004 | n.v.t. | 701,7 |
| TTM | 366,5 | −2,6 | — | — | 18,4 | 5,0 | 29,9 | 8,2 | 9,2 | 2,5 | — | — | 707,8 |

*Afschrijvingen (en dus EBITDA) zijn alleen voor 2023-2025 apart gepubliceerd; voor 2020-2022 is
de post niet uit een geopende bron te halen en blijft de cel leeg. Het gewogen gemiddeld aantal
aandelen wordt niet in de geopende bronnen getoond; de kolom bevat het aantal uitstaande aandelen
ultimo periode waar dat wél is gepubliceerd.*

### Resultatenrekening B — zoals gerapporteerd in **EUR mln** (2015-2020)

| Jaar | Omzet | Omzetgroei % | Brutowinst | Brutomarge % | EBIT | EBIT-marge % | Nettowinst | Nettomarge % | EPS | Aandelen mln (gew. gem.) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2015 | 10,8 | — | 6,0 | 55,7 | −12,8 | −118,5 | −10,0 | −92,0 | −0,024 | — |
| 2016 | 15,9 | +46,6 | 11,2 | 70,5 | −11,5 | −72,7 | −17,5 | −110,5 | −0,042 | — |
| 2017 | 89,6 | +464,6 | 77,2 | 86,1 | 21,9 | 24,4 | −76,2 | −85,1 | −0,152 | — |
| 2018 | 135,1 | +50,8 | 113,0 | 83,6 | 38,0 | 28,1 | 25,0 | 18,5 | 0,041 | 621,5 |
| 2019 | 169,0 | +25,1 | 147,7 | 87,4 | 60,9 | 36,0 | 36,2 | 21,4 | 0,058 | 631,3 |
| 2020 | 185,7 | +9,9 | 165,1 | 88,9 | 67,4 | 36,3 | 32,7 | 17,6 | 0,051 | — |

- **Toelichting resultaten:**

Pharming heeft in tien jaar drie totaal verschillende gedaanten gehad. Tot en met 2016 was het een
verlieslatend onderzoeksbedrijf met minder dan EUR 16 miljoen omzet. Vanaf de terugkoop van de
Amerikaanse RUCONEST-rechten eind 2016 ontplofte de omzet — plus 465% in 2017 — en werd Pharming
kortstondig een uitzonderlijk winstgevend bedrijf: in 2019 en 2020 lag de EBIT-marge boven de 36%.
Het enorme nettoverlies van 2017 was geen operationeel probleem maar een boekhoudkundige
afrekening van EUR 107,6 miljoen aan financieringslasten en derivatenherwaardering op de
financiering van die overname. Sinds 2021 is de derde fase begonnen: de omzet groeide door naar
US$376 miljoen, maar de marge stortte in van 35,9% (2020) naar rond nul in 2023 en 2024, omdat
Pharming zwaar investeerde in de lancering van Joenja en in onderzoek. 2025 was een herstel — EBIT
US$25,8 miljoen — maar het lopende jaar valt alweer terug: de omzet daalde in de eerste helft van
2026 met 6% en het operationele resultaat werd opnieuw negatief.

- **Omzet-CAGR**: +12,1% over 2020-2025 (in US$). Ter vergelijking: +76,5% over 2015-2020 in euro's,
  maar dat cijfer zegt vooral iets over het startpunt bijna nul.

### Kasstromen (US$ mln)

| Jaar | CFO | Capex | FCF | FCF na SBC | FCF/aandeel | FCF-marge % | FCF-groei % | FCF-conversie % | SBC | Dividend totaal | Aandeleninkoop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2020 | 83,6 | 4,66 | 79,0 | 72,4 | 0,113 | 34,1 | — | 209 | 6,54 | 0 | 0 |
| 2021 | 37,8 | 10,74 | 27,1 | 18,0 | 0,028 | 9,1 | −65,7 | 169 | 9,06 | 0 | 0 |
| 2022 | 22,5 | 1,38 | 21,1 | — | — | — | −22,2 | 154 | — | 0 | 0 |
| 2023 | −17,3 | 1,46 | −18,8 | −28,0 | −0,040 | −11,4 | n.v.t. | 178 | 9,25 | 0 | 0 |
| 2024 | −1,8 | 0,80 | −2,6 | −13,8 | −0,020 | −4,7 | n.v.t. | 22 | 11,25 | 0 | 0 |
| 2025 | 54,7 | 0,76 | 54,0 | 40,2 | 0,057 | 10,7 | n.v.t. | 2.126 | 13,77 | 0 | 0 |
| TTM | 35,0 | 0,65 | 34,4 | 20,4 | 0,029 | 5,6 | −36,3 | 372 | 14,0* | 0 | 0 |

*SBC voor de TTM-periode is niet apart gepubliceerd; ik gebruik het jaarcijfer 2025 als
benadering en markeer dat als zodanig — het is geen bronvermelding.*

- **Toelichting kasstromen:**

De kasstroom van Pharming schommelt heftiger dan de winst, en dat vraagt uitleg omdat de swings
groter zijn dan 15% per jaar. In 2023 sloeg de operationele kasstroom om van plus US$22,5 miljoen
naar min US$17,3 miljoen. Twee posten verklaren dat vrijwel volledig: de voorraden namen met
US$14,4 miljoen toe en de vorderingen met US$18,5 miljoen, allebei het gevolg van de Joenja-
lancering — je moet product op de plank leggen en aan apotheken leveren voordat je betaald krijgt.
Dat is dus opbouw van werkkapitaal, geen verslechtering van de onderliggende verdiencapaciteit.
In 2024 bleef de kasstroom licht negatief doordat er US$15,6 miljoen belasting werd betaald tegen
een boekhoudkundige last van maar US$3,3 miljoen. In 2025 draaide alles de goede kant op: de
betaalde belasting viel terug naar US$4,6 miljoen, de schulden aan leveranciers stegen met US$14,1
miljoen en de kasstroom sprong naar US$54,7 miljoen. Wie dat cijfer als de nieuwe norm neemt,
vergist zich: ongeveer US$10 miljoen kwam uit gunstig werkkapitaal en nog eens US$5 miljoen uit
een tijdelijk lage kasbelasting.

**Normalisatie van de kasbelasting (METHODE regel 5).** De betaalde winstbelasting is bij Pharming
structureel losgekoppeld van de belastinglast: 2023 US$0,7 mln betaald tegen een belastingbate;
2024 US$15,6 mln betaald tegen US$3,3 mln last; 2025 US$4,6 mln betaald tegen US$10,3 mln last.
Over drie jaar is er US$20,8 miljoen kas afgedragen op een cumulatief resultaat vóór belasting van
min US$7,7 miljoen. Het bedrijf heeft bovendien US$31,0 miljoen aan actieve belastinglatenties op
de balans staan. Ik reken in de waardering daarom met een **genormaliseerd effectief tarief van
25%** op EBIT in plaats van met de betaalde kasbelasting van enig afzonderlijk jaar. Zonder die
correctie zou 2025 een vrije kasstroom laten zien die ongeveer US$5,7 miljoen te hoog is.

**Signaalcheck.** Twee van de drie waarschuwingssignalen uit METHODE zijn aanwezig: de accruals
werden in 2025 sterk negatief (−11,6%) en de FCF-conversie schoot naar 2.126% doordat de winst bijna
nul was terwijl de kasstroom US$54 miljoen bedroeg. Beide wijzen op hetzelfde: winst en kasstroom
lopen uiteen, met belasting en werkkapitaal als oorzaak. Vandaar de correctie hierboven.

### Balans-ratio's (US$ mln, 2020-2025)

| Jaar | Nettoschuld | Nettoschuld/EBITDA | Eigen vermogen | ROE % | ROIC % | ROCE % | Current ratio | Solvabiliteit % | Immat. activa % van activa | Bruto schuld |
|---|---|---|---|---|---|---|---|---|---|---|
| 2020 | −43,2 | — | 183,4 | 20,6 | 40,8 | — | — | 43,8 | — | 162,0 |
| 2021 | −30,2 | — | 192,9 | 8,3 | 6,3 | — | — | 48,6 | — | 161,8 |
| 2022 | −40,6 | — | 204,6 | 6,7 | 8,3 | — | — | 48,1 | — | 166,7 |
| 2023 | −41,9 | −4,0 | 219,2 | −4,8 | −2,3 | — | — | 47,5 | — | 171,5 |
| 2024 | −55,6 | −7,5 | 221,1 | −5,4 | −3,9 | — | — | 55,3 | 15,3 | 112,3 |
| 2025 | −63,3 | −1,7 | 277,1 | 0,9 | 9,1 | — | 2,59 | 55,4 | 27,1 | 115,8 |

*ROCE is niet ingevuld omdat ik voor de jaren zonder gepubliceerde afschrijvingen geen consistente
noemer kan opbouwen; ROIC is berekend als EBIT × (1 − 25%) gedeeld door eigen vermogen plus bruto
schuld minus kas en marketable securities.*

### Balans-ratio's (EUR mln, 2015-2019)

| Jaar | Totale activa | Eigen vermogen | Kas | Leningen (bruto) | Solvabiliteit % |
|---|---|---|---|---|---|
| 2015 | 57,7 | 23,8 | 31,6 | 14,8 | 41,3 |
| 2016 | 126,8 | 27,5 | 31,9 | 66,5 | 21,7 |
| 2017 | 166,2 | 16,1 | 58,7 | 81,6 | 9,7 |
| 2018 | 214,6 | 61,8 | 80,3 | 72,5 | 28,8 |
| 2019 | 228,2 | 104,7 | 66,3 | 45,6 | 45,9 |

- **Toelichting balans:**

De balans is Pharmings sterkste kaart en tegelijk het bewijs hoe ver het bedrijf is gekomen. In
2017 was de solvabiliteit 9,7% en stond er EUR 81,6 miljoen aan dure leningen tegenover EUR 16,1
miljoen eigen vermogen; eind 2025 is het eigen vermogen US$277,1 miljoen en de solvabiliteit 55,4%.
Belangrijk is het onderscheid tussen bruto en netto: de bruto schuld liep in 2025 licht op van
US$112,3 naar US$115,8 miljoen doordat de converteerbare obligaties aangroeien, maar de
**nettokaspositie verbeterde tegelijk van US$55,6 naar US$63,3 miljoen**. Wie alleen naar de bruto
schuld kijkt, ziet dus ten onrechte een verslechtering. De enige echte schuldpost is de
converteerbare obligatielening die in 2029 vervalt; met een rentedekking van 1,4 keer de EBIT in
2025 is dat geen comfortabele buffer, maar de kaspositie van US$179 miljoen dekt de lening ruim.
De keerzijde: door de Abliva-overname bestaat inmiddels 27,1% van de activa uit immateriële activa,
waarvan US$61,1 miljoen puur het nog niet goedgekeurde middel napazimone is — een afboekingsrisico
als de studie in 2027 faalt.

### Kapitaalstructuur huidig (30 juni 2026, US$ mln)

- **Nettoschuld (huidig)**: −43,9 (nettokaspositie)
- **Bruto schuld**: 115,6
- **Cash & equivalents**: 158,3 (plus marketable securities 1,2; totaal 159,5)
- **Lease-verplichtingen (IFRS-16)**: 17,7 (stand 31-12-2025; per halfjaar niet uitgesplitst)
- **Gemiddelde rente %**: 5,6 (US$5,067 mln betaalde rente 2025 op gemiddelde boekwaarde US$90,3 mln)
- **Rente-dekking (EBIT/rente)**: 1,42× (2025)
- **Schuldvervaldatum**: converteerbare obligaties met einddatum 2029, geplaatst april 2024

### Non-GAAP / aanpassingen
- **Gebruikt?** false
- **Welke aanpassingen**: Pharming rapporteert uitsluitend onder IFRS en publiceert geen adjusted
  earnings. Wel worden in het persbericht "operating expenses" apart geguideerd (exclusief
  kostprijs omzet); dat is een presentatiekeuze, geen alternatieve winstmaatstaf.
- **Waarom**: n.v.t.

### Earnings quality

| Jaar | Accruals ratio % | Non-GAAP verschil % | SBC % van FCF |
|---|---|---|---|
| 2021 | −5,4 | 0 | 33,4 |
| 2022 | −2,1 | 0 | — |
| 2023 | +1,5 | 0 | n.v.t. (FCF negatief) |
| 2024 | −2,3 | 0 | n.v.t. (FCF negatief) |
| 2025 | −11,6 | 0 | 25,5 |

- **Toelichting earnings quality:**

De winstkwaliteit is op het eerste gezicht goed: de accruals-ratio is in vier van de vijf jaren
negatief, wat betekent dat de kasstroom hoger uitkomt dan de boekwinst — conservatief dus. Maar de
−11,6% van 2025 is geen kwaliteitssignaal; die ontstaat doordat de winst bijna nul was terwijl de
kasstroom werd opgeblazen door werkkapitaal en een lage kasbelasting. Pharming rapporteert geen
adjusted winst, wat een pluspunt is: er valt niets weg te definiëren. De belangrijkste
kwaliteitskwestie is de aandelencompensatie. Die bedroeg in 2025 US$13,8 miljoen, ofwel 25,5% van de
vrije kasstroom en 1,6% van de beurswaarde. Dat is een reële kostenpost voor aandeelhouders en ik
trek hem in de hele waardering af.

### Rendementsindicatoren (US$, ROIC vs. WACC)

| Jaar | ROIC % | WACC % (schatting) | Spread (pp) | Oordeel |
|---|---|---|---|---|
| 2020 | 40,8 | 10,4 | +30,4 | uitzonderlijke waardecreatie |
| 2021 | 6,3 | 10,4 | −4,2 | waardevernietiging |
| 2022 | 8,3 | 10,4 | −2,1 | waardevernietiging |
| 2023 | −2,3 | 10,4 | −12,7 | waardevernietiging |
| 2024 | −3,9 | 10,4 | −14,3 | waardevernietiging |
| 2025 | 9,1 | 10,4 | −1,4 | net onder de kostenvoet |

- **Toelichting rendement:**

Dit is de scherpste conclusie van de hele financiële analyse. In 2020 verdiende Pharming ruim
40% op zijn geïnvesteerde kapitaal tegen een kapitaalkostenvoet van ongeveer 10% — dat is precies
het soort spread waar een moat zichtbaar wordt. Sinds 2021 is die spread **vijf jaar op rij
negatief**. Pharming heeft in die periode fors geïnvesteerd in de lancering van Joenja en in
onderzoek, maar het rendement daarop is nog niet zichtbaar: zelfs in het herstelde jaar 2025 komt
het rendement op geïnvesteerd kapitaal met 9,1% net onder de kapitaalkosten uit. Een bedrijf dat
structureel onder zijn kapitaalkosten verdient, vernietigt waarde terwijl het groeit — en dat is
wat de omzetgroei van 12% per jaar sinds 2020 tot dusver heeft opgeleverd. De ROIC-WACC-spread is
in deze analyse dan ook de belangrijkste verklaring voor de lage scores op moat en Buffett.

### Waarderingsratio's (peildatum 17-08-2026)

| Ratio | Waarde | Toelichting |
|---|---|---|
| P/E (TTM) | 90,8 | winst TTM US$9,2 mln |
| P/E forward | — | consensus-EPS niet betrouwbaar te herleiden naar gewoon aandeel |
| P/FCF (na SBC, TTM) | 41,2 | FCF na SBC US$20,4 mln |
| FCF-rendement (na SBC) | 2,43% | |
| EV/EBITDA (TTM) | 26,6 | EBITDA US$29,9 mln |
| EV/Omzet (TTM) | 2,17 | |
| P/B | 3,03 | eigen vermogen per 31-12-2025 |
| PEG | >4 | P/E 90,8 tegen een winstgroei die momenteel negatief is |
| Dividendrendement | 0% | |

- **Toelichting waardering:**

Pharming is niet op elke maatstaf duur. Op 2,2 keer de omzet en 41 keer de vrije kasstroom na
aandelencompensatie betaal je voor een bedrijf met 88% brutomarge geen extreme prijs, en het
aandeel staat 44% onder de top van de afgelopen twaalf maanden. Maar op winst is het aandeel met
een koers-winstverhouding van 91 en een EV/EBITDA van 27 wel degelijk hoog gewaardeerd, en dat is
de eerlijker maatstaf, want de vrije kasstroom van 2025 was opgeblazen door werkkapitaal en een
tijdelijk lage belastingafdracht. Een historisch tienjaarsgemiddelde van deze ratio's heb ik niet
opgenomen: de winst was in vier van de tien jaren negatief, waardoor een gemiddelde P/E geen
betekenis heeft.

### Sector-KPI's

| KPI | Eenheid | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| R&D als % van omzet | % | 35,4 | 25,5 | 28,1 | 28,0 | 26,7 |
| Marketing & sales als % van omzet | % | 29,9 | 41,7 | 50,6 | 40,0 | 34,8 |
| Brutomarge | % | 89,4 | 91,5 | 89,7 | 88,1 | 87,9 |
| RUCONEST-omzet | US$ mln | — | 205,6 | 227,1 | 252,2 | 317,9 |
| Joenja-omzet | US$ mln | 0 | 0 | 18,2 | 45,0 | 58,2 |
| Omzetconcentratie top-2 klanten | % | — | — | — | 77 | 77 |

- **Toelichting sector-KPI's:**

Voor een weesgeneesmiddelenbedrijf zijn drie dingen bepalend: hoeveel je aan onderzoek uitgeeft,
hoe duur het is om artsen en patiënten te bereiken, en hoe afhankelijk je bent van je afzetkanaal.
Op alle drie is Pharming kwetsbaar. De R&D-uitgaven liggen met 27% van de omzet ruim boven wat
gevestigde farmaceuten uitgeven en stijgen in 2026 nog eens met meer dan US$40 miljoen. De
verkoopkosten piekten in 2023 op 50,6% van de omzet — het Joenja-lanceringsjaar — en dalen nu, wat
gunstig is. Het meest verontrustende cijfer is de klantconcentratie: twee Amerikaanse specialty
pharmacies zijn goed voor 77% van de omzet, en juist voorraadafbouw bij die partijen verklaarde
een deel van de omzetdaling in de eerste helft van 2026.

### Dividend

- **Betaalt dividend**: nee
- Pharming heeft nooit dividend uitgekeerd en koopt geen eigen aandelen in. Dat is in dit geval een
  volstrekt logische keuze en geen zwaktesignaal in de klassieke zin: het bedrijf heeft de kas nodig
  voor onderzoek en betaalde tot 2024 nog rente op dure schuld. Het betekent wel dat het volledige
  rendement uit koersstijging moet komen, dat de belegger geen enkele tussentijdse vergoeding krijgt
  voor het gedragen risico, en dat er geen dividenddiscipline is die het management dwingt kritisch
  naar de kostenbasis te kijken. Voor een inkomensbelegger is dit aandeel niet geschikt.
- **Oordeel houdbaarheid**: n.v.t. — er is geen dividend. Gezien de vrije kasstroom van US$20 miljoen
  na aandelencompensatie tegenover een beurswaarde van US$839 miljoen is er ook geen ruimte om er
  binnen de horizon van vijf jaar mee te beginnen.

---

## 4. Moat (concurrentievoordeel)

- **Oordeel**: **NARROW MOAT**
- **Moat-categorieën:**

| Naam | Sterkte | Toelichting |
|---|---|---|
| Immateriële activa | middel | Beide producten leunen op regulatoire bescherming: RUCONEST en Joenja hebben weesgeneesmiddelstatus, en Joenja is wereldwijd het enige goedgekeurde middel voor APDS. Dat is een echte barrière — een concurrent moet een volledige registratiestudie doen in een populatie van enkele honderden patiënten. RUCONEST is bovendien het enige recombinante C1-esteraseremmer-product; de concurrenten zijn plasma-afgeleid of werken via een ander mechanisme. |
| Overstapkosten | zwak | Voor een acuut middel bij een aanval zijn de overstapkosten laag: de arts schrijft een ander middel voor en de patiënt gebruikt het bij de volgende aanval. Precies dat gebeurt nu ook — Pharming meldt dat de actieve patiëntenbasis nog 93% van vorig jaar bedraagt. Bij Joenja liggen ze hoger, omdat het chronisch wordt gebruikt en er geen alternatief bestaat. |
| Netwerkeffecten | geen | Het product wordt niet waardevoller naarmate meer patiënten het gebruiken. Wel bouwt Pharming een voorschrijversnetwerk op — 17 nieuwe voorschrijvers in het tweede kwartaal van 2026 — maar dat is distributiekracht, geen netwerkeffect. |
| Kostenvoordeel | zwak | De productie in transgene konijnenmelk is uniek en moeilijk te kopiëren, maar levert geen aantoonbaar kostenvoordeel op: de brutomarge van 87,9% is niet hoger dan bij plasma-afgeleide concurrenten en daalt al vier jaar op rij. |
| Efficiënte schaal | middel | APDS treft naar schatting één op de honderdduizend mensen. Zo'n markt is te klein om twee spelers rendabel te maken, wat nieuwe toetreders afschrikt. Voor HAE geldt dat niet meer: de markt van US$3,6 miljard is groot genoeg om een half dozijn spelers te dragen, en dat is precies wat er gebeurt. |

- **Kwantitatief bewijs:** De harde toets voor een moat is of het bedrijf structureel meer verdient
  dan zijn kapitaalkosten. Bij Pharming is de spread tussen ROIC en WACC **vijf jaar op rij negatief**:
  −4,2, −2,1, −12,7, −14,3 en −1,4 procentpunt over 2021 tot en met 2025. De brutomarge is gedaald van
  91,5% (2022) naar 87,9% (2025), en het marktaandeel in het belangrijkste product krimpt: de
  RUCONEST-omzet daalde in de eerste helft van 2026 met 12%. Er is dus wel bescherming — je kunt niet
  zomaar een concurrerend APDS-middel op de markt brengen — maar die bescherming vertaalt zich niet in
  bovengemiddelde rendementen.

- **Duurzaamheid:** 5 jaar, en met sterk verschillende houdbaarheid per product. Voor Joenja is de
  bescherming stevig tot minstens 2033: er is geen concurrerend middel in ontwikkeling voor APDS dat
  ver genoeg is. Voor RUCONEST is de duurzaamheid nu al aan het eroderen. De sector is in twee jaar
  veranderd: sinds juli 2025 bestaat er met Ekterly (sebetralstat) een pil die een acute aanval stopt,
  en die is in juni 2026 overgenomen door Chiesi voor US$1,9 miljard — een partij met veel meer
  commerciële slagkracht dan KalVista. Tegelijk verlaagt betere preventie (BioCryst' Orladeyo groeide
  in twaalf maanden 69% naar US$941 miljoen omzet) het aantal aanvallen waarvoor RUCONEST überhaupt
  nodig is. Beide krachten werken tegelijk en zijn structureel, niet cyclisch. Ik reken er in het
  basisscenario dan ook op dat RUCONEST elk jaar 5 tot 7 procent terrein verliest.

- **Erosierisico's:** orale on-demand behandeling (Ekterly/Chiesi), effectievere profylaxe die het
  aantal aanvallen verlaagt, prijsdruk van Amerikaanse zorgverzekeraars bij een groeiend aanbod, en
  het aflopen van de weesgeneesmiddelexclusiviteit op leniolisib.

---

## 5. Management

- **CEO-naam + tenure**: Fabrice Chouraqui, PharmD, MBA — CEO en uitvoerend bestuurder sinds 2023
- **CFO-naam + tenure**: Kenneth Lynard, MSc, EMBA — CFO (aangetreden na Jeroen Wakkerman; exacte
  aantreedmaand niet uit een geopende bron te verifiëren)
- **Oprichter nog betrokken?**: nee. Sijmen de Vries, die Pharming vanaf 2008 leidde en tot de
  commerciële doorbraak bracht, droeg in 2023 over aan Chouraqui en zit sindsdien niet meer in de
  dagelijkse leiding.
- **Insider ownership %**: 1,58 (Simply Wall St, aggregator)
- **Capital allocation track record:**

| Jaar | Dividend totaal | Aandeleninkoop | M&A uitgaven (US$ mln) | Organische capex (US$ mln) |
|---|---|---|---|---|
| 2021 | 0 | 0 | 0 | 10,7 |
| 2022 | 0 | 0 | 0 | 1,4 |
| 2023 | 0 | 0 | 0 | 1,5 |
| 2024 | 0 | 0 | 0 | 0,8 |
| 2025 | 0 | 0 | 68,0 | 0,8 |

- **M&A-track-record**: twee materiële transacties in tien jaar. De terugkoop van de Noord-Amerikaanse
  RUCONEST-rechten van Valeant (december 2016, US$60 miljoen vooraf plus tot US$65 miljoen aan
  mijlpalen) was achteraf een uitstekende deal: de omzet vervijfvoudigde binnen twee jaar. De
  overname van het Zweedse Abliva in 2025 kostte US$60,0 miljoen aan koopsom plus US$7,9 miljoen voor
  het minderheidsbelang en leverde één klinisch middel op — napazimone — dat pas in 2027 een uitkomst
  geeft. Er is US$61,1 miljoen als immaterieel actief geactiveerd en US$2,9 miljoen goodwill; bij een
  negatieve studie-uitkomst gaat dat vrijwel geheel ten laste van het resultaat.

- **Beloning**: De beloningsstructuur heb ik niet kunnen verifiëren — het remuneratierapport 2025 en
  Item 6.B van het 20-F waren niet volledig op te halen. Wat wél hard is: de last voor
  aandelencompensatie steeg van US$9,3 miljoen (2023) via US$11,2 miljoen (2024) naar US$13,8 miljoen
  (2025). Dat is 1,64% van de beurswaarde per jaar, onder de drempel van 3% die METHODE als hoog
  aanmerkt, maar wel 25,5% van de vrije kasstroom van 2025. Het aantal uitstaande aandelen groeide in
  de eerste helft van 2026 met 0,87% naar 707,8 miljoen — een verwatering van ruwweg 1,7% op jaarbasis.

- **Oordeel management**: **NEUTRAAL**
- **Toelichting:**

Het beeld is gemengd en dat is geen diplomatieke formulering maar de conclusie van de cijfers. Aan
de positieve kant staat een balans die in acht jaar van bijna-faillissement naar 55% solvabiliteit
en een nettokaspositie van US$63 miljoen is gebracht, een geslaagde herfinanciering in 2024 en een
lancering van Joenja die de omzet in drie jaar naar US$58 miljoen bracht. Daar staat tegenover dat
het rendement op geïnvesteerd kapitaal vijf jaar op rij onder de kapitaalkosten ligt: de
kostenbasis groeide sneller dan de omzet. De guidance-verlaging van juli 2026 kwam bovendien maar
zes maanden nadat het management in maart een groei van 8 tot 13 procent had afgegeven — dat is een
grote misser op een korte horizon. Het management was er wel open over en verlaagde tegelijk de
kostenverwachting, wat pleit voor transparantie. Insiders bezitten samen 1,58%, onder de drempel van
3 tot 5 procent die voor een small cap als betekenisvol geldt, en er zijn geen open-markt aankopen
door de CEO of CFO gevonden. Controverses of rechtszaken heb ik niet aangetroffen.

**Insidertransacties.** Ik heb geen betrouwbare reeks kunnen samenstellen. De enige aggregator die
transacties toont, geeft voor meerdere posten koersen van EUR 51 tot EUR 102 voor een aandeel dat
rond EUR 1 noteert; die data is intern inconsistent en neem ik daarom niet over. Het enige dat de
bron consistent laat zien, is dat het gaat om verkopen door leden van het uitvoerend comité, en
dat er geen open-markt aankopen zijn geregistreerd. Ik noteer dit als **niet geverifieerd** en zet
het in sectie 13 onder ontbrekende data.

---

## 6. Sector & concurrentie

- **Sector-groeivooruitzicht**: de wereldwijde markt voor therapieën tegen erfelijk angio-oedeem werd
  in 2025 op US$3,6 miljard geraamd en groeit naar verwachting met 10,4% per jaar naar US$8,2 miljard
  in 2033 (Grand View Research). Het on-demand-segment, waar RUCONEST in zit, had in 2025 60,8% van
  de markt, maar het profylaxe-segment groeit sneller (11,6% per jaar) — precies de verschuiving die
  Pharming raakt.

- **Porter five forces:**
  - **Rivaliteit: HOOG.** In de HAE-markt concurreren Takeda, CSL Behring, BioCryst, Ionis en sinds
    2025 KalVista (nu Chiesi) met plasma-producten, remmers en sinds kort een pil. De consolidatie
    versnelt: Chiesi betaalde US$1,9 miljard voor KalVista. In APDS is er geen rivaliteit.
  - **Nieuwe toetreders: MIDDEL.** De drempels zijn hoog — registratiestudies in zeldzame ziekten,
    weesgeneesmiddelexclusiviteit, gespecialiseerde distributie — maar de prijzen zijn zo aantrekkelijk
    dat grote farmaceuten er structureel geld en overnames in stoppen.
  - **Substituten: HOOG.** Voor RUCONEST bestaan directe substituten: icatibant, plasma-afgeleide
    C1-remmers en sinds juli 2025 een oraal middel. Betere profylaxe fungeert bovendien als indirecte
    substituut door het aantal aanvallen te verlagen.
  - **Macht leveranciers: MIDDEL.** De transgene productieketen is uniek en daardoor kwetsbaar voor
    verstoring; Joenja is bovendien gelicentieerd van Novartis, wat royalty-afhankelijkheid meebrengt.
  - **Macht afnemers: HOOG.** Twee Amerikaanse specialty pharmacies verzorgen 77% van de omzet, en hun
    voorraadbeslissingen bewegen de kwartaalcijfers meetbaar. Daarachter zitten zorgverzekeraars die bij
    een groeiend aanbod steeds meer onderhandelingsmacht krijgen.

  **Conclusie Porter: gemiddelde tot onaantrekkelijke sector.** De weesgeneesmiddelenmarkt biedt
  uitzonderlijke marges en regulatoire bescherming, maar juist die aantrekkelijkheid trekt kapitaal
  aan. Voor een subschaal-speler met één kwetsbaar hoofdproduct en twee dominante afnemers is de
  structurele positie ongunstig; de winst lekt weg naar de kant van de distributie en de nieuwe
  toetreders.

- **Concurrenten:**

| Concurrent | Positie | Omzet (recent) | Kerncijfer |
|---|---|---|---|
| BioCryst Pharmaceuticals (BCRX) | orale profylaxe (Orladeyo) | US$941 mln TTM, +68,7% | marktkap US$2,49 mrd (14-08-2026) |
| KalVista → Chiesi Group | eerste orale on-demand (Ekterly) | US$114 mln TTM | overgenomen voor US$1,9 mrd, juni 2026 |
| Takeda | Takhzyro (profylaxe), Firazyr | — | onderdeel van een gediversifieerde farmaceut |
| CSL Behring | Berinert, Haegarda (plasma) | — | onderdeel van CSL Ltd |
| Ionis Pharmaceuticals | donidalorsen (profylaxe) | — | — |

*Voor Takeda, CSL en Ionis heb ik geen segmentcijfers voor het HAE-portfolio uit een geopende bron
kunnen halen; de kwantitatieve kolommen blijven daarom leeg in plaats van geschat.*

- **Positie van het bedrijf**: nichespeler. Pharming is met circa US$318 miljoen RUCONEST-omzet een
  van de kleinere spelers in een HAE-markt van US$3,6 miljard — ruwweg 9 procent — en verliest daarin
  terrein. In APDS is Pharming daarentegen alleenheerser.

- **Positie-toelichting:** De vergelijking met BioCryst is ontnuchterend. Beide bedrijven zijn
  ongeveer even oud in deze markt, maar BioCryst groeit met 69% naar US$941 miljoen omzet terwijl
  Pharming met 2,6% krimpt, en de markt waardeert BioCryst op US$2,5 miljard tegen Pharming op
  US$0,84 miljard. De korting die Pharming krijgt is dus grotendeels verdiend: het verschil zit in
  groei, niet in marge. Tegelijk laat de overname van KalVista voor US$1,9 miljard op US$114 miljoen
  omzet zien dat strategische kopers voor een groeiend HAE-actief bereid zijn zestien keer de omzet te
  betalen — Pharming noteert op 2,2 keer. Dat verschil is de kern van het optimistische scenario en
  meteen de reden dat Pharming zelf geregeld als overnamekandidaat wordt genoemd.

### TAM/SAM/SOM
- **TAM (US$ mln)**: 3.600 (HAE-therapieën wereldwijd, 2025)
- **TAM-groei %**: 10,4
- **SAM (US$ mln)**: 2.189 (on-demand-segment, 60,8% van de TAM in 2025)
- **SAM-groei %**: lager dan de TAM — profylaxe groeit met 11,6% sneller dan de markt
- **Huidige penetratie %**: 8,8 van de TAM (RUCONEST US$317,9 mln op US$3.600 mln); 14,5 van het
  on-demand-segment (US$317,9 mln op US$2.189 mln)
- **Impliciete penetratie na horizon %**: 4,2 van de TAM — RUCONEST komt in het basisscenario in
  2030 uit op circa US$248 mln, tegen een TAM die bij de geraamde 10,4% groei op circa US$5.900 mln
  ligt
- **Groei plausibel?** true — het basisscenario gaat uit van *dalend* marktaandeel, wat de meest
  behoudende aanname is
- **Bron TAM/SAM**: Grand View Research, Hereditary Angioedema Therapeutics Market
- **Toelichting**: De plausibiliteitscheck werkt hier omgekeerd dan gebruikelijk. Ik hoef niet te
  toetsen of een agressieve groeiaanname in de markt past; ik toets of de aanname van krimp niet te
  streng is. Bij een marktgroei van 10,4% per jaar en een RUCONEST-daling van 5% per jaar zakt
  Pharmings aandeel in de totale HAE-markt van 8,8% naar circa 4,2% in vijf jaar. Dat is een
  forse maar realistische aanname gegeven dat er sinds juli 2025 een pil beschikbaar is voor dezelfde
  indicatie en gegeven de al zichtbare daling van 12% in de eerste helft van 2026. Voor Joenja is de
  markt te klein om zinvol als TAM te kwantificeren; het management noemt zelf een potentieel van
  meer dan één miljard dollar, maar dat is voorwaardelijk op een positieve CVID-uitkomst.

---

## 7. Analyse-frameworks (9 frameworks, SCORES 1-5)

### Graham
- **Oordeel**: VOLDOET NIET
- **Graham number**: 0,2931
- **Margin of safety %**: −71,4 (de Graham number ligt 71% onder de koers)
- **Toelichting**: Graham zocht bedrijven met een lange, ononderbroken winstgeschiedenis, een
  koers-winstverhouding onder 15 en een koers-boekwaardeverhouding onder 1,5. Pharming voldoet aan
  geen van die criteria. De winst per aandeel bedraagt over de laatste twaalf maanden EUR 0,0113,
  wat bij een koers van EUR 1,024 uitkomt op een koers-winstverhouding van bijna 91. De
  boekwaarde per aandeel is EUR 0,338, oftewel 3,0 keer de boekwaarde. Bovendien was de winst in
  vier van de laatste tien jaar negatief en is er nooit dividend uitgekeerd — twee harde
  diskwalificaties in Grahams defensieve raamwerk. De Graham number, die winst en boekwaarde
  combineert, komt uit op EUR 0,29: minder dan een derde van de huidige koers. Er is geen enkele
  veiligheidsmarge.
- **Score (1-5)**: **1**

### Buffett / Munger
- **Oordeel**: VOLDOET NIET
- **ROIC structureel boven WACC?** false
- **Toelichting**: Buffett zoekt een uitzonderlijk bedrijf tegen een redelijke prijs, en het eerste
  woord is hier het probleem. Het rendement op geïnvesteerd kapitaal ligt vijf jaar op rij onder de
  kapitaalkosten — in 2025 9,1% tegen een WACC van 10,4%. Voor de begrijpelijkheid geldt een
  waarschuwing: het bedrijf lijkt begrijpelijk (twee producten, twee ziektes) maar de kern van de
  waardering hangt af van klinische uitkomsten die geen belegger buiten de sector kan inschatten. De
  kasstromen zijn allesbehalve voorspelbaar: in vijf jaar liep de vrije kasstroom van plus US$79
  miljoen naar min US$19 miljoen en terug naar plus US$54 miljoen. De prijs is met 41 keer de vrije
  kasstroom na aandelencompensatie evenmin bescheiden. Dit is het tegenovergestelde van een
  Buffett-belegging.
- **Score (1-5)**: **1**

### Peter Lynch
- **Categorie**: Turnaround
- **Oordeel**: ONINTERESSANT
- **PEG-ratio**: >4
- **Toelichting**: Lynch zou Pharming in 2020 als een fast grower hebben geclassificeerd, maar het
  bedrijf hoort na de guidance-verlaging in de categorie turnaround: de omzet krimpt, de marge is
  net hersteld en de casus hangt aan het weer op gang krijgen van groei. Lynch was gek op
  turnarounds, maar wel op turnarounds die je goedkoop kunt kopen — en dat is Pharming niet. Bij
  een koers-winstverhouding van 91 en een winst die momenteel dáált, is de PEG-ratio niet zinvol te
  berekenen maar in elk geval ruim boven 2. Het verhaal is bovendien maar half helder: dat Pharming
  twee zeldzame ziektes behandelt kan iedereen navertellen, maar of de waarde er is, hangt op de
  fase-II-uitkomst voor CVID eind 2026 — en dat is een verhaal dat je niet in twee zinnen aan een
  leek uitlegt.
- **Score (1-5)**: **1**

### Phil Fisher
- **Oordeel**: GEMIDDELD
- **Toelichting**: Fisher keek naar producten met echt groeipotentieel, een innovatiecultuur,
  beschermde marges en integer management. Pharming scoort duidelijk op het eerste punt: de
  R&D-uitgaven bedragen 26,7% van de omzet, ruim boven wat gevestigde farmaceuten uitgeven, en het
  budget groeit door — in 2026 komt er meer dan US$40 miljoen bij. Joenja heeft in APDS een
  onbetwiste positie en de pijplijn is met napazimone verbreed. Op het tweede en derde punt schiet
  het bedrijf tekort: de brutomarge glijdt al vier jaar af, van 91,5% naar 87,9%, en de moat
  beschermt de marge van het hoofdproduct aantoonbaar niet. Het managementoordeel is neutraal, niet
  sterk. Eén van de drie criteria voldaan, met een groeiend onderzoeksbudget.
- **Score (1-5)**: **3**

### Magic Formula (Greenblatt)
- **Oordeel**: ONAANTREKKELIJK
- **Earnings yield %**: 2,32
- **Return on capital %**: 10,70
- **Toelichting**: Greenblatt rangschikt bedrijven op twee assen: hoeveel operationele winst je
  krijgt voor de prijs die je betaalt, en hoe efficiënt het bedrijf zijn tastbare kapitaal inzet.
  Pharming scoort op beide zwak. Het winstrendement — operationele winst gedeeld door
  ondernemingswaarde — bedraagt 2,32%: minder dan de helft van wat je op een Amerikaanse
  staatsobligatie krijgt. Het rendement op kapitaal komt uit op 10,7%, ver onder de 50% die
  Greenblatt bij zijn beste ideeën ziet en zelfs onder de 15%-ondergrens. Dat laatste komt niet
  doordat Pharming kapitaalintensief is — de vaste activa bestaan grotendeels uit gekochte
  immateriële rechten — maar doordat de operationele winst simpelweg te klein is ten opzichte van
  de balans.
- **Score (1-5)**: **1**

### Moat
- **Score (1-5)**: **2** — NARROW MOAT, maar de ROIC-WACC-spread is met −1,4 procentpunt negatief en
  dus kleiner dan de 5 procentpunt die de rubric voor een score van 3 vereist.

### Management
- **Score (1-5)**: **3** — capital allocation gemengd: een uitstekende deal in 2016, een nog niet
  bewezen overname in 2025, en vijf jaar rendement onder de kapitaalkosten.

### Fair Value DCF
- **Score (1-5)**: **1** — fair value basis EUR 0,38 tegen een koers van EUR 1,024: een neerwaarts
  potentieel van 63,3%, ruim boven de drempel van 15%.

### Fair Value IPO-gecorr.
- **Score (1-5)**: **1** — de beursgang was in 1999, ruim tien jaar geleden, dus gelijk aan de
  DCF-basisscore.

### Scorekaart totaal
- **Totaalscore**: 14
- **Max**: 45
- **Eindoordeel**: **PASS**
  - totaal 14 < 24 → PASS; bovendien Fair Value DCF-score = 1 → PASS
- **Samenvatting:**

Pharming is een bedrijf met echte producten, echte patiënten en een echte brutomarge van bijna 88
procent — en toch komt het uit op 14 van de 45 punten. De reden is dat vrijwel elk raamwerk in deze
scorekaart naar hetzelfde kijkt: verdient het bedrijf meer op zijn kapitaal dan dat kapitaal kost,
en betaal je daar een redelijke prijs voor. Op beide vragen is het antwoord nee. Het rendement op
geïnvesteerd kapitaal ligt vijf jaar op rij onder de kapitaalkosten en kwam zelfs in het herstelde
jaar 2025 uit op 9,1 procent tegen een WACC van 10,4 procent. De koers van EUR 1,024 impliceert dat
de vrije kasstroom tien jaar lang met bijna 30 procent per jaar groeit — mijn basisscenario komt op
EUR 0,38 uit en de no-growth-waarde zelfs op EUR 0,25. Het aandeel wordt pas interessant onder EUR
0,26, en dan nog alleen voor wie het binaire risico van de fase-II-uitkomsten wil dragen. De
voornaamste onzekerheid is niet de waardering maar de vraag hoe snel RUCONEST erodeert; die vraag
wordt elk kwartaal opnieuw beantwoord.

---

## 8. Risico's

| # | Omschrijving | Kans | Impact | DCF-aanname geraakt | Toelichting |
|---|---|---|---|---|---|
| 1 | RUCONEST verliest sneller terrein aan orale on-demand behandeling dan aangenomen | HOOG | GROOT | omzetgroei | RUCONEST is 84,5% van de omzet 2025 en daalde in de eerste helft van 2026 al met 12%. Sinds juli 2025 bestaat er met Ekterly een pil voor dezelfde indicatie, sinds juni 2026 in handen van Chiesi met een veel grotere commerciële organisatie. Elke extra procentpunt jaarlijkse daling kost in mijn model ruwweg EUR 0,08 fair value per aandeel: bij 3,5% erosie is de basiswaarde EUR 0,50, bij 6,5% nog EUR 0,26. |
| 2 | Fase-II-uitkomsten voor leniolisib bij CVID en primaire immuundeficiënties vallen tegen | MIDDEN | GROOT | omzetgroei, terminale waarde | De uitkomst komt in het vierde kwartaal van 2026. Het management koppelt hier expliciet het "meer dan één miljard dollar"-potentieel van Joenja aan. Een negatieve uitkomst haalt vrijwel het volledige verschil tussen mijn basis- en optimistische scenario weg. Dit is een binair risico. |
| 3 | Kostenbasis van US$315-320 mln laat geen ruimte voor omzettegenvallers | HOOG | GROOT | EBIT-marge | De operationele kosten bedragen bijna 85% van de omzet. Bij een omzet van US$375 mln en kosten van US$318 mln blijft er nauwelijks winst over; dertig miljoen dollar omzetverlies duwt het bedrijf terug in de rode cijfers zoals in 2023 en 2024. |
| 4 | Extreme klantconcentratie: twee specialty pharmacies zijn 77% van de omzet | MIDDEN | GROOT | omzetgroei, werkkapitaal | Voorraadbeslissingen bij deze twee partijen bewogen de omzet in de eerste helft van 2026 aantoonbaar. Verlies van één relatie of een strengere inkoopvoorwaarde raakt de omzet direct en het werkkapitaal onmiddellijk. |
| 5 | Napazimone (KL1333) faalt in de FALCON-studie | MIDDEN | MIDDEL | terminale waarde, afboeking | Er staat US$61,1 mln als immaterieel actief plus US$2,9 mln goodwill op de balans uit de Abliva-overname. De uitkomst komt in 2027. Bij falen volgt een afboeking van circa 13% van het eigen vermogen en verdwijnt de derde poot uit de groeicasus. |
| 6 | Herfinancieringsrisico op de converteerbare obligaties van 2029 | LAAG | MIDDEL | WACC | US$98,1 mln vervalt in 2029, tegen een kaspositie van US$159,5 mln. De rentedekking was in 2025 slechts 1,4 keer de EBIT. Bij een lagere koers wordt conversie onaantrekkelijk en moet er cash op tafel komen; dat verhoogt de kapitaalkosten. |
| 7 | Verwatering door aandelencompensatie | HOOG | KLEIN | vrije kasstroom per aandeel | De aandelencompensatie liep op naar US$13,8 mln in 2025, 25,5% van de vrije kasstroom. Het aantal aandelen groeide in zes maanden met 0,87%. Dit is structureel en niet eenmalig; ik verwerk het volledig als kosten in de DCF. |
| 8 | Concentratie van de productie in één unieke transgene keten | LAAG | GROOT | omzetgroei | RUCONEST wordt gewonnen uit de melk van transgene konijnen — een productiemethode zonder alternatieve leverancier. Een dier- of kwaliteitsincident legt de omzet stil zonder dat er een tweede bron is. In 2026 was er al US$4,9 mln aan productiegerelateerde voorraadafboekingen. |

**Verplichte check — pre-IPO financial engineering.** Niet van toepassing en niet geconstateerd. De
beursgang was in juni 1999, ruim 27 jaar geleden; er is geen private-equity-eigenaar geweest, geen
dividendrecapitalisatie vóór de notering, en geen schuldopbouw bij gerelateerde partijen die met
beursopbrengsten is afgelost. Er is dus geen IPO-gecorrigeerde fair value nodig: die is gelijk aan
de basiswaarde van EUR 0,38.

---

## 9. These invalide bij

Deze analyse is weerlegd wanneer de RUCONEST-omzet in de tweede helft van 2026 en in 2027 weer
groeit in plaats van te dalen, of wanneer de fase-II-uitkomsten voor leniolisib bij CVID in het
vierde kwartaal van 2026 positief zijn en het management dat vertaalt in een concreet, becijferd
omzetpad. In dat geval verschuift de kansverdeling richting het optimistische scenario en ligt de
kansgewogen waarde boven de huidige koers. Omgekeerd wordt het pessimistische scenario leidend als
de operationele kosten in 2027 niet onder de US$310 miljoen komen terwijl de omzet onder US$370
miljoen blijft — dan verbrandt het bedrijf zijn nettokaspositie in plaats van hem op te bouwen.

---

## 10. ESG

### Materiële factoren (SASB-gebaseerd)

| Factor | SASB-categorie | Risiconiveau | Financiële impact | DCF-impact |
|---|---|---|---|---|
| Dierenwelzijn in de productieketen | Supply Chain Management (Biotechnology & Pharmaceuticals) | Hoog | RUCONEST wordt gewonnen uit de melk van transgene konijnen. Dit is een reputatie- en vergunningsrisico dat vrijwel uniek is voor Pharming en bij een incident of campagne de afzet in bepaalde markten kan raken. | omzetgroei |
| Toegang en betaalbaarheid | Access & Affordability | Midden | Weesgeneesmiddelen behoren tot de duurste therapieën. Amerikaanse prijsdruk en de Inflation Reduction Act raken op termijn de prijszetting. | EBIT-marge |
| Geneesmiddelveiligheid | Drug Safety | Midden | Beide producten worden bij kleine populaties gebruikt; een veiligheidssignaal weegt in zo'n populatie zwaar en kan tot labelbeperking leiden. | omzetgroei, terminale waarde |
| Afhankelijkheid van sleutelpersoneel | Employee Recruitment & Retention | Midden | Recent verloop in het uitvoerend comité en een kleine organisatie maken het bedrijf gevoelig voor het vertrek van enkele mensen. | EBIT-marge |
| Bedrijfsethiek en marketingpraktijken | Business Ethics | Laag | Geen controverses of handhavingsmaatregelen aangetroffen in de geraadpleegde bronnen. | — |

- **Eindoordeel ESG**: **GEMIDDELD RISICO**
- **Toelichting**: Pharming heeft geen zware milieu-voetafdruk en geen governance-schandalen, maar
  wel één factor die je bij vrijwel geen ander beursgenoteerd farmabedrijf tegenkomt: de productie
  van het belangrijkste medicijn loopt via transgene dieren. Dat is voor sommige beleggers een
  uitsluitingscriterium en het maakt de toeleveringsketen kwetsbaar voor maatschappelijke druk. Aan
  de sociale kant is het bedrijf juist sterk: het bedient patiënten die zonder deze middelen geen
  behandeling zouden hebben. De governance is neutraal — een volledig versnipperd aandeelhouderschap
  betekent geen dominante aandeelhouder maar ook geen tegenmacht.

---

## 11. Katalysatoren

| Datum ca. | Omschrijving | Richting | Impact |
|---|---|---|---|
| 2026-10 | PDUFA-datum FDA voor Joenja bij kinderen van 4 tot 11 jaar (sNDA, prioriteitsbeoordeling) | BINAIR | MIDDEL |
| 2026-11 | Derdekwartaalcijfers: eerste toets of RUCONEST daadwerkelijk stabiliseert, zoals het management stelt | NEUTRAAL | GROOT |
| 2026-12 | Fase-II-uitkomsten leniolisib bij CVID en genetische primaire immuundeficiënties | BINAIR | GROOT |
| 2026-Q3/Q4 | Commerciële lancering Joenja in Japan en verdere uitrol in Duitsland | POSITIEF | KLEIN |
| 2027-03 | Jaarcijfers 2026 en guidance 2027 — bepaalt of de kostenbasis daadwerkelijk daalt | NEUTRAAL | GROOT |
| 2027 | Uitkomst FALCON-studie napazimone bij primaire mitochondriale ziekten | BINAIR | GROOT |
| 2029 | Aflossing of conversie van de converteerbare obligaties (US$98,1 mln) | NEGATIEF | MIDDEL |

De twee uitkomsten in het vierde kwartaal van 2026 zijn echt binair en bepalen samen ongeveer het
verschil tussen mijn basisscenario (EUR 0,38) en het optimistische scenario (EUR 1,25). De
kwartaalcijfers zijn gradueel: elk kwartaal levert een nieuw datapunt over de erosiesnelheid van
RUCONEST, en juist die snelheid is de gevoeligste variabele in het model.

---

## 12. Fair value — kwantitatief (DCF)

### DCF-invoeren

```
Basis            fcf=11.7 shares=707.78 net_cash=43.9 gross_debt=115.62 revenue=366.5
                 koers=1.024 ipo_jaar=1999
WACC             rf=4.63 erp=4.23 beta=1.10 crp=0.0 size_premium=2.00
                 cost_of_debt_pretax=5.60 tax_rate=25.80
Pessimistisch    g1=10.0 g2=0.5 gt=0.5 wacc_adj=1.00 kans=25
Basis            g1=22.4 g2=2.0 gt=2.0 wacc_adj=0.00 kans=50
Optimistisch     g1=53.5 g2=2.5 gt=2.5 wacc_adj=-0.50 kans=25
EPV              norm_ebit_margin=5.92 maintenance_capex=11.2 da=11.2
                 norm_ebitda_margin=8.90
Multiples        pe=90.8 pb=3.03 p_fcf=41.2 peg=4.0
Rendement        roic=9.07 earnings_yield=2.32 roc_greenblatt=10.70
Kwalitatief      moat_oordeel=NARROW moat_categorieen_sterk=0
                 management_oordeel=NEUTRAAL
                 capital_allocation=GEMENGD insider_alignment_pct=1.58
                 roic_wacc_spread_5j_plus=false structureel_dividend=false debt_equity=0.42
Eenheid          bedragen in USD mln; percentages als getal (3.05 = 3,05%)
                 fair values omgerekend naar EUR tegen EUR/USD 1,157
```

*Toelichting bij `g1`: de scenario's zijn niet met één groeipercentage gemodelleerd maar met een
volledig omzet- en margepad (zie de projectietabel). De opgegeven `g1` is de resulterende
samengestelde groei van de vrije kasstroom over de vijf projectiejaren, zodat stage 2 dezelfde
uitkomst kan reproduceren.*

### WACC-componenten
- **Risicovrije rente %**: 4,63
- **Bron risicovrije rente**: Amerikaanse 10-jaars staatsobligatie (FRED-serie DGS10), stand 13-08-2026.
  De kasstromen zijn in dollars gemodelleerd, dus is de Treasury de juiste referentie en niet de
  Nederlandse staatslening.
- **Type**: nominaal, spot. *Let op:* de spotrente ligt circa 150 basispunten boven het tienjaars
  gemiddelde. Ik gebruik de spotrente als basis en toon in de gevoeligheidsmatrix expliciet ook
  lagere WACC-varianten (tot 8,5%) die met een genormaliseerde rente overeenkomen.
- **ERP (equity risk premium) %**: 4,23
- **Bron ERP**: Damodaran implied ERP, jaarultimo 2025 (update januari 2026)
- **Beta (adjusted) **: 1,10
- **Bron beta**: bottom-up uit Damodaran industry betas januari 2026 — unlevered beta gecorrigeerd
  voor kas van Drugs (Biotechnology) 1,08 en Drugs (Pharmaceutical) 0,92, gemiddeld 1,00, gerelevered
  naar een D/E van 13,8% bij een belastingtarief van 25,8%.
- **Type beta**: bottom-up. *Motivatie:* Pharming voldoet formeel aan de eis voor een regressiebeta
  (ruim vijf jaar genoteerd, gemiddeld dagvolume ruim boven 100.000 stuks), maar de gepubliceerde
  regressiebeta's lopen extreem uiteen — 0,06 (StockAnalysis), 0,12 en 0,74 bij verschillende
  aanbieders. Die spreiding maakt de regressiebeta onbruikbaar; een bottom-up beta is hier het
  verdedigbare alternatief.
- **Country risk premium %**: 0 (Nederland en de Verenigde Staten kennen geen landenrisicopremie)
- **Size premium %**: 2,00 (marktkapitalisatie EUR 0,72 mrd, ruim onder de drempel van EUR 2 mrd)
- **Cost of equity %**: 11,29
- **Schuldkosten na belasting %**: 4,16 (5,60% vóór belasting × (1 − 25,8%))
- **E/V gewicht %**: 87,9
- **D/V gewicht %**: 12,1
- **WACC %**: **10,43**
- **Sector WACC % (referentie Damodaran)**: 8,49 (Drugs Biotechnology, VS, januari 2026)
- **Illiquiditeitskorting %**: geen — het gemiddelde dagvolume ligt met miljoenen stuks ruim boven
  de drempel van 50.000

### DCF model-specs
- **Model type**: 2-fase (vijf expliciete projectiejaren plus Gordon-terminalwaarde)
- **FCF-definitie**: FCFF (free cash flow to firm), verdisconteerd tegen de WACC
- **Basis FCF**: US$11,7 mln (projectiejaar 2026, opgebouwd uit de bedrijfsguidance)
- **Basis FCF na SBC**: US$11,7 mln — de aandelencompensatie is al volledig in mindering gebracht
- **FCF-type**: opgebouwd uit omzet en marge, niet geëxtrapoleerd uit een historisch jaar
- **Groei fase 1 %**: 22,4 (samengestelde FCF-groei jaar 1-5, basisscenario)
- **Groei fase 2 %**: n.v.t. — na jaar 5 gaat het model direct naar de terminale groeivoet
- **Terminal groei %**: 2,0
- **Terminal methode**: Gordon growth
- **Exit multiple gebruikt**: 5,3× EV/EBITDA (impliciet uit het model, zie hieronder)
- **Bron exit multiple**: afgeleid uit het model, niet uit een peergroep — voor de HAE-sector is geen
  betrouwbare mediaan over een volle cyclus beschikbaar omdat de twee zuivere vergelijkers
  (BioCryst, KalVista) beide verlieslatend zijn of zijn overgenomen
- **Terminal value Gordon growth**: US$319,5 mln nominaal
- **Terminal value % van totaal**: **74%** — net onder de 75% die METHODE als
  geloofwaardigheidsgrens hanteert, maar hoog genoeg om te benoemen: driekwart van de waarde zit in
  wat er ná 2030 gebeurt. In het optimistische scenario loopt dit op tot 82%.
- **Terminal implied EV/EBITDA**: 5,3× op de geprojecteerde EBITDA van 2030 (US$60,8 mln). Dat is
  laag voor de sector en juist geruststellend: het model bouwt geen dure uitstapmultiple in. De
  lage multiple is het gevolg van het gat tussen EBITDA en vrije kasstroom — aandelencompensatie
  en belasting nemen samen ruim de helft weg.
- **Terminal groei consistentie**: een terminale groei van 2% vereist bij een langetermijn-ROIC van
  10% een herinvesteringsvoet van 20%. Voor een farmabedrijf dat 27% van de omzet aan R&D uitgeeft,
  is dat ruimschoots haalbaar — de herinvestering vindt bij Pharming plaats via de winst-en-
  verliesrekening in plaats van via capex. De 2% ligt bovendien onder de nominale BBP-groei van
  zowel de Verenigde Staten als de eurozone, dus de bovengrens uit METHODE wordt gerespecteerd.
- **Mid-year convention**: true
- **Aandelen uitstaand (mln)**: 707,78 (30-06-2026)
- **Nettoschuld huidig**: −43,9 (nettokaspositie, US$ mln)

### DCF-toelichting

Ik heb de vrije kasstroom niet uit een historisch jaar geëxtrapoleerd maar van onderaf opgebouwd,
omdat elk afzonderlijk jaar bij Pharming vertekend is: 2025 door gunstig werkkapitaal en een lage
belastingafdracht, 2023 en 2024 door de opbouw van voorraad voor de Joenja-lancering. Het startpunt
voor 2026 is de eigen guidance van het bedrijf — US$375 tot 395 miljoen omzet en US$315 tot 320
miljoen operationele kosten — waaruit bij de gerealiseerde brutomarge van 88% een bedrijfsresultaat
van ongeveer US$24 miljoen volgt. Daarop pas ik een genormaliseerd belastingtarief van 25% toe in
plaats van de sterk wisselende betaalde belasting, tel ik de afschrijvingen op, en trek ik capex,
werkkapitaalgroei én de volledige aandelencompensatie af. Het bedrijf is niet cyclisch in de zin
van METHODE, dus een mid-cyclus-normalisatie is niet aan de orde; wel is de kasbelasting
genormaliseerd, wat de vrije kasstroom van 2025 met ongeveer US$5,7 miljoen verlaagt. De
leaseverplichtingen behandel ik consequent als schuld — ze zitten in de nettoschuld en de
leasebetalingen zitten daarom niét in de kasstroom, zodat er niets dubbel geteld wordt. Ik gebruik
FCFF met de WACC, disconteer halverwege het jaar, en trek pas aan het eind de nettoschuld af. De
terminale waarde is 74% van het totaal; dat is hoog en het cijfer moet met die wetenschap gelezen
worden.

### 5-jaars projectie (basisscenario, US$ mln)

| Jaar | RUCONEST | Joenja | Omzet | Omzetgroei % | EBIT | EBIT-marge % | NOPAT | Capex | ΔNWC | SBC | FCF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026 | 305,0 | 80,0 | 385,0 | +2,4 | 24,3 | 6,3 | 18,2 | 1,5 | 2 | 15 | 11,7 |
| 2027 | 289,8 | 105,6 | 395,4 | +2,7 | 27,0 | 6,8 | 20,3 | 1,5 | 2 | 15 | 14,8 |
| 2028 | 275,3 | 134,1 | 409,4 | +3,5 | 32,8 | 8,0 | 24,6 | 1,5 | 3 | 16 | 17,1 |
| 2029 | 261,5 | 163,6 | 425,1 | +3,8 | 40,1 | 9,4 | 30,1 | 1,5 | 3 | 17 | 21,6 |
| 2030 | 248,4 | 193,1 | 441,5 | +3,9 | 47,8 | 10,8 | 35,9 | 1,5 | 3 | 18 | 26,4 |

*Opbouw: het startpunt 2026 is de eigen guidance (omzet US$385 mln, operationele kosten US$317,5
mln). RUCONEST daalt daarna met 5% per jaar, Joenja groeit met 32%, 27%, 22% en 18%. De
operationele kosten groeien met 2% per jaar, waardoor er operationele hefboom ontstaat naarmate
Joenja een groter deel van de omzet uitmaakt: de EBIT-marge loopt van 6,3% naar 10,8%. De
brutomarge blijft 88%; afschrijvingen US$12-13 mln, capex US$1,5 mln.*

### Scenarios

| Scenario | FCF-groei % | WACC % | Fair value (EUR) | Upside % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | +10,0 | 11,43 | 0,15 | −85,8 | 25 |
| Basis | +22,4 | 10,43 | 0,38 | −63,3 | 50 |
| Optimistisch | +53,5 | 9,93 | 1,25 | +22,2 | 25 |

- **Kansgewogen fair value**: EUR 0,54 (−47,6% ten opzichte van de koers)

**Scenariodefinities.** Alle drie de scenario's zijn per product opgebouwd, met dezelfde
kostenlogica. In het *pessimistische* scenario daalt RUCONEST met 10% per jaar naar US$197 miljoen
in 2030 en groeit Joenja fors trager (25%, 18%, 12%, 8%); de omzet zakt naar US$331 miljoen en het
management moet de kosten terugbrengen naar US$280 miljoen om überhaupt uit de rode cijfers te
blijven — de vrije kasstroom blijft rond US$9 miljoen steken. In het *basisscenario* daalt RUCONEST
met 5% per jaar en groeit Joenja af van 32% naar 18%. In het *optimistische* scenario slaagt de
CVID-studie: RUCONEST daalt nog maar 3% per jaar, Joenja versnelt naar 40% groei en bereikt in 2030
US$292 miljoen, de omzet komt uit op US$567 miljoen en de EBIT-marge op 25,5%. De kansverdeling van
25/50/25 is bewust symmetrisch gehouden ondanks het binaire karakter van de studie-uitkomsten:
zowel de neerwaartse als de opwaartse tak hangt aan gebeurtenissen met een reële kans.

### Reverse DCF
- **Impliciete groei %**: **29,6** — de vrije kasstroom moet tien jaar lang met bijna 30% per jaar
  groeien, en daarna eeuwig met 2%, om de huidige ondernemingswaarde van US$795 miljoen te
  rechtvaardigen bij een WACC van 10,43% en een startpunt van US$11,7 miljoen.
- **Historische FCF CAGR %**: negatief — de vrije kasstroom na aandelencompensatie ging van US$72,4
  mln (2020) naar US$20,4 mln (TTM)
- **Consensus groei %**: de vier analisten die het aandeel volgen ramen voor 2026 US$403,5 miljoen
  omzet, boven de door het bedrijf zelf afgegeven bandbreedte van US$375-395 miljoen. De ramingen
  zijn dus nog niet aangepast aan de guidance-verlaging van 30 juli 2026 en overschatten het jaar
  met circa 5%.
- **Interpretatie**: Dit is het meest sprekende cijfer van de hele analyse. De markt prijst in dat
  Pharming zijn vrije kasstroom tien jaar lang met bijna 30% per jaar laat groeien. Ter vergelijking:
  de vrije kasstroom na aandelencompensatie is de afgelopen vijf jaar juist met ruim 70 procent gedááld en de omzet groeide met
  12% per jaar. Bijna 30% kasstroomgroei is niet onmogelijk — het is precies wat er gebeurt als de
  CVID-studie slaagt en Joenja richting een miljard dollar gaat — maar het is wel wat je moet
  geloven om vandaag te kopen. Je koopt op deze koers geen ondergewaardeerd bedrijf, je koopt een
  optie op twee klinische uitkomsten.

### EPV (Bruce Greenwald)
- **Genormaliseerde EBIT-marge %**: 5,92 (mediaan van de EBIT-marges 2021-2025 plus TTM: 6,82; 8,87;
  −2,20; −2,90; 6,87; 5,02)
- **Genormaliseerde NOPAT**: US$16,7 mln (genormaliseerde EBIT US$22,3 mln × (1 − 25%))
- **Maintenance capex**: US$11,2 mln — gelijkgesteld aan de afschrijvingen. Pharming heeft nauwelijks
  fysieke capex (US$0,75 mln in 2025), maar zijn werkelijke onderhoudsinvestering is de amortisatie
  van gekochte productrechten; die moet vervangen worden door nieuwe in-licenties. Deze
  gelijkstelling is de conservatieve keuze en laat de aangepaste verdiencapaciteit gelijk aan NOPAT.
- **Adjusted earnings power**: US$16,7 mln
- **EPV per aandeel**: **EUR 0,25** (US$160,2 mln bedrijfswaarde plus US$43,9 mln nettokas, gedeeld
  door 707,78 mln aandelen, omgerekend tegen 1,157)
- **Groeipremie %**: de DCF-basiswaarde ligt 51% boven de EPV; de **koers** ligt 311% boven de EPV,
  ofwel 4,1 keer de waarde zonder enige groei.

### Andere methoden
- **DDM uitgevoerd?** false — Pharming keert geen dividend uit en zal dat binnen de horizon niet doen.
- **SOTP uitgevoerd?** false — geen conglomeraat; de twee producten delen dezelfde verkoop- en
  onderzoeksorganisatie en zijn niet los te waarderen zonder de kostenbasis willekeurig toe te delen.

### Synthese fair value
- **Bandbreedte laag**: 0,15
- **Bandbreedte centraal**: 0,38
- **Bandbreedte hoog**: 1,25
- **Methode-gewichten**: DCF 60%, EPV 30%, multiples 10%
- **Margin of safety vereist %**: 30
- **Koopniveau**: 0,26 (fair value basis EUR 0,38 × 0,70)
- **Synthese-toelichting:**

De drie methoden wijzen dezelfde kant op, maar met een grote spreiding. De DCF geeft EUR 0,38, de
waarde zonder groei EUR 0,25, en de relatieve waardering — 2,2 keer de omzet tegenover 16 keer bij
de overname van KalVista — is de enige die voor het aandeel pleit. Ik weeg de DCF het zwaarst omdat
die de kostenbasis expliciet meeneemt, en de EPV substantieel omdat die laat zien wat je krijgt als
de pijplijn niets oplevert: een kwart van de huidige koers. De multiples krijgen weinig gewicht,
want een omzetmultiple zegt niets over een bedrijf waarvan de kosten 85% van de omzet bedragen. De
markt betaalt vandaag ruim vier keer de no-growth-waarde; dat is de groeipremie waar het om draait.
Gegeven de binaire pijplijn, de negatieve ROIC-WACC-spread en het feit dat 74% van de DCF-waarde
in de terminale waarde zit, is een veiligheidsmarge van 30% het minimum. Onder EUR 0,26 wordt het
aandeel interessant voor wie het klinische risico bewust wil nemen.

### Gevoeligheid (DCF)

Fair value per aandeel in EUR. Rijen: samengestelde groei van de vrije kasstroom in de vijf
projectiejaren. Kolommen: WACC. Terminale groei constant op 2%.

| FCF-groei ↓ / WACC → | 8,5% | 9,5% | 10,5% | 11,5% | 12,5% | 13,5% |
|---|---|---|---|---|---|---|
| −5,0% | 0,229 | 0,207 | 0,190 | 0,177 | 0,166 | 0,157 |
| 0,0% | 0,262 | 0,235 | 0,214 | 0,198 | 0,185 | 0,174 |
| +5,0% | 0,300 | 0,267 | 0,242 | 0,222 | 0,206 | 0,193 |
| +10,0% | 0,343 | 0,304 | 0,274 | 0,250 | 0,231 | 0,215 |
| +15,0% | 0,392 | 0,346 | 0,310 | 0,282 | 0,259 | 0,241 |

Deze matrix varieert alleen de disconteringsvoet en een uniforme kasstroomgroei; het
basisscenario hierboven komt hoger uit (EUR 0,376) omdat de kasstroom daar niet gelijkmatig groeit
maar versnelt naarmate Joenja de omzetmix domineert. De boodschap is dezelfde: zelfs bij een WACC
van 8,5% — de sectorwaarde van Damodaran, dus zonder size premium — en 15% gelijkmatige
kasstroomgroei komt de waarde niet boven EUR 0,40. De huidige koers van EUR
1,024 valt volledig buiten deze matrix. Om daar te komen moet de kasstroom niet met 15% maar met
ongeveer 30% per jaar groeien, en dat is precies wat de reverse DCF laat zien. De waardering is dus
niet gevoelig voor de disconteringsvoet; ze is gevoelig voor één ding: de vraag of de vrije
kasstroom kan verdriedubbelen.

---

## 13. Databronnen

### Bronnen-hiërarchie
- **Jaarverslag PDF / IR-pagina / SEC-filing** → betrouwbaarheid **HOOG**
- **Beursmelding / persbericht met jaarrekening** → betrouwbaarheid **HOOG**
- **Aggregator** (StockAnalysis / MarketScreener / Simply Wall St) → betrouwbaarheid **AGGREGATOR**

### Financiële bronnen (10 jaar historie)

| Jaar | Bron | URL | Betrouwbaarheid |
|---|---|---|---|
| 2016 | Pharming Report on Preliminary Financial Results 2016 | https://www.pharming.com/sites/default/files/imce/Public%20Documents/2016/PR%20Full%20Year%20Results%202016.pdf | HOOG |
| 2017 | Pharming Full Year Results 2018 (vergelijkende kolom) | https://www.pharming.com/files/1-pr-full-year-results-2018.pdf | HOOG |
| 2018 | Pharming Full Year Results 2018 | https://www.pharming.com/files/1-pr-full-year-results-2018.pdf | HOOG |
| 2019 | Pharming Full Year Results 2020 (herziene vergelijkende kolom) | https://www.pharming.com/files/2020-full-year-results-pr-04mar21-final-pdf.pdf | HOOG |
| 2020 | Pharming Full Year Results 2021 (USD) en 2020 (EUR) | https://www.pharming.com/files/pharming-group-financial-results-full-year-2021-final-1.pdf | HOOG |
| 2021 | Pharming Annual Report 2021 + persbericht FY2021 | https://www.pharming.com/sites/default/files/imce/Public%20Documents/Annual%20report%202022%2006APR22%20compressed.pdf | HOOG |
| 2022 | Pharming 4Q/FY2023 persbericht (vergelijkende kolom) | https://www.pharming.com/files/pharming-group-4q-2023-financial-results-en-14mar2024-final.pdf | HOOG |
| 2023 | Form 20-F FY2025, R3/R5/R7 (herzien) | https://www.sec.gov/Archives/edgar/data/1828316/000182831626000015/R3.htm | HOOG |
| 2024 | Form 20-F FY2025, R3/R5/R7 | https://www.sec.gov/Archives/edgar/data/1828316/000182831626000015/R5.htm | HOOG |
| 2025 | Form 20-F FY2025, R3/R5/R7 + persbericht 4Q/FY2025 | https://www.sec.gov/Archives/edgar/data/1828316/000182831626000015/R7.htm | HOOG |

*2015 is eveneens gedekt (vergelijkende kolom in het persbericht over 2016, HOOG), maar valt buiten
het tienjaarsvenster 2016-2025 dat de tabel hanteert.*

**Alle tien jaren zijn HOOG.** Er is geen enkel jaar op een aggregator gebaseerd en er staat geen
enkele cel in de financiële tabellen die niet uit een geopend document komt.

### Jaarverslagen geraadpleegd

| Jaar | Bron | URL |
|---|---|---|
| 2025 | Annual Report 2025 (online) / Form 20-F | https://annualreport.pharming.com/ |
| 2023 | Annual Report 2023 (PDF) | https://www.pharming.com/sites/default/files/imce/Public%20Documents/2024/Pharming%20Annual%20Report%202023_website.pdf |
| 2021 | Annual Report 2021 (PDF) | https://www.pharming.com/sites/default/files/imce/Public%20Documents/Annual%20report%202022%2006APR22%20compressed.pdf |
| 2019 | Annual Report 2019 (PDF) | https://www.pharming.com/sites/default/files/imce/Public%20Documents/Annual%20report%202019%20complete%20copy%2031MAR2020%20SEM.pdf |

### Beursmeldingen geraadpleegd

| Datum | Omschrijving | URL |
|---|---|---|
| 2026-07-30 | 2Q/1H2026-resultaten en verlaging jaarverwachting met US$30 mln | https://www.pharming.com/files/pharming-group-reports-2q-1h26-results-en-30july2026.pdf |
| 2026-07-30 | Presentatie 2Q/1H2026 (pijplijn, mijlpalen, kaspositie) | https://www.pharming.com/files/pharming-2q-1h2026-presentation-final-30july2026.pdf |
| 2026-07-30 | Transcript analistencall 2Q/1H2026 | https://www.pharming.com/files/2q-1h-2026-results-call-transcript-30july26.pdf |
| 2026-03-12 | 4Q/FY2025-resultaten en oorspronkelijke guidance 2026 | https://www.pharming.com/files/pharming-group-reports-4q25-fy25-results-en-12mar26.pdf |
| 2016-12-08 | Afronding overname Noord-Amerikaanse RUCONEST-rechten van Valeant | https://www.pharming.com/files/closing-of-acquisition.pdf |

### IPO-prospectus
- **Geraadpleegd?** false
- **URL**: n.v.t.
- **Pre-IPO data beschikbaar?** false
- **Pre-IPO bron**: n.v.t. — de beursgang dateert van juni 1999 en valt ruim buiten het
  analysevenster. Een IPO-correctie is niet van toepassing.

### Non-GAAP
- **Gebruikt?** false
- **Toelichting**: Pharming publiceert uitsluitend IFRS-cijfers. Alle bedragen in dit rapport zijn
  IFRS zoals gerapporteerd. De enige eigen bewerkingen zijn: (a) de TTM-reeks, berekend als FY2025
  minus 1H2025 plus 1H2026; (b) de aftrek van aandelencompensatie in de vrije-kasstroommaatstaven;
  (c) de normalisatie van de kasbelasting naar 25% in de waardering. Alle drie zijn expliciet
  benoemd op de plaats waar ze gebruikt worden.

### Ontbrekende data (eerlijke lijst)
- **Gewogen gemiddeld aantal aandelen per jaar** — niet zichtbaar in de R-pagina's van het 20-F en
  niet in de persberichten. De kolom "aandelen" bevat waar mogelijk het aantal uitstaande aandelen
  ultimo periode; voor 2022, 2023 en 2024 blijft de cel leeg.
- **Aandelencompensatie 2015, 2016 en 2022** — niet als last gepubliceerd in de geopende bronnen;
  de FCF-na-SBC voor 2022 blijft daarom leeg.
- **Afschrijvingen 2020, 2021 en 2022** — niet apart in de geopende bronnen, waardoor EBITDA en
  nettoschuld/EBITDA voor die jaren leeg blijven.
- **Omzetsplitsing naar geografie** — de segmentnoot van het 20-F (R11, R42) rendert leeg in de
  XBRL-viewer. Alleen de klantconcentratie (77% via twee Amerikaanse specialty pharmacies) is hard.
- **Beloning CEO en CFO over 2025, bonuscriteria, CEO pay ratio, aandelenbezitseis** — Item 6.B van
  het 20-F en het remuneratierapport 2025 waren niet volledig op te halen. Staat op de haallijst.
- **Personeelsaantal 2024 en 2025** — laatste geverifieerde stand is 415 per ultimo 2023.
- **Individuele insidertransacties** — de enige beschikbare aggregator geeft koersen die met een
  factor 50 tot 100 afwijken van de werkelijke beurskoers en is dus intern inconsistent. Niet
  overgenomen.
- **Grootaandeelhouders via het AFM-register** — het register is een JavaScript-toepassing die niet
  te lezen is; de top-5 komt uit een aggregator en is als zodanig gemarkeerd.
- **Dollarcijfers voor 2015 tot en met 2019** — bestaan niet: het bedrijf heeft alleen 2020
  herrekend bij de valutawissel. Niet zelf omgerekend.
- **Historische tienjaarsgemiddelden van de waarderingsratio's** — niet zinvol te berekenen omdat de
  winst in vier van de tien jaren negatief was.
- **Segmentcijfers HAE-portfolio van Takeda, CSL en Ionis** — niet als segment gerapporteerd.
- **IPO-koers 1999** — niet uit een openbare bron te verifiëren.

### Peildatum analyse
- 2026-08-17 (koersdatum; EUR/USD 1,157 per 14-08-2026, laatst beschikbare noteringsdag)

---

## 14. Volledige bronnen-lijst

| Titel | URL | Type |
|---|---|---|
| Pharming Group — Financial documents (IR-index) | https://www.pharming.com/investors/financial-documents | beurswebsite |
| Pharming Group — SEC filings (index) | https://www.pharming.com/investors/sec-filings | beurswebsite |
| Form 20-F FY2025 — FilingSummary | https://www.sec.gov/Archives/edgar/data/1828316/000182831626000015/FilingSummary.xml | jaarverslag |
| Form 20-F FY2025 — geconsolideerde winst-en-verliesrekening (R3) | https://www.sec.gov/Archives/edgar/data/1828316/000182831626000015/R3.htm | jaarverslag |
| Form 20-F FY2025 — geconsolideerde balans (R5) | https://www.sec.gov/Archives/edgar/data/1828316/000182831626000015/R5.htm | jaarverslag |
| Form 20-F FY2025 — geconsolideerd kasstroomoverzicht (R7) | https://www.sec.gov/Archives/edgar/data/1828316/000182831626000015/R7.htm | jaarverslag |
| Form 20-F FY2025 — Abliva-overnamebalans (R41) | https://www.sec.gov/Archives/edgar/data/1828316/000182831626000015/R41.htm | jaarverslag |
| Form 20-F FY2025 (volledig document) | https://www.sec.gov/Archives/edgar/data/1828316/000182831626000015/pharm-20251231.htm | jaarverslag |
| Pharming Annual Report 2025 (online) | https://annualreport.pharming.com/ | jaarverslag |
| Pharming Annual Report 2023 (PDF) | https://www.pharming.com/sites/default/files/imce/Public%20Documents/2024/Pharming%20Annual%20Report%202023_website.pdf | jaarverslag |
| Pharming Annual Report 2021 (PDF) | https://www.pharming.com/sites/default/files/imce/Public%20Documents/Annual%20report%202022%2006APR22%20compressed.pdf | jaarverslag |
| Pharming Annual Report 2019 (PDF) | https://www.pharming.com/sites/default/files/imce/Public%20Documents/Annual%20report%202019%20complete%20copy%2031MAR2020%20SEM.pdf | jaarverslag |
| Pharming 2Q/1H 2026 results (persbericht) | https://www.pharming.com/files/pharming-group-reports-2q-1h26-results-en-30july2026.pdf | beursmelding |
| Pharming 2Q/1H 2026 presentatie | https://www.pharming.com/files/pharming-2q-1h2026-presentation-final-30july2026.pdf | beursmelding |
| Pharming 2Q/1H 2026 analistencall transcript | https://www.pharming.com/files/2q-1h-2026-results-call-transcript-30july26.pdf | beursmelding |
| Pharming 4Q/FY 2025 results (persbericht) | https://www.pharming.com/files/pharming-group-reports-4q25-fy25-results-en-12mar26.pdf | beursmelding |
| Pharming 4Q/FY 2023 results (persbericht) | https://www.pharming.com/files/pharming-group-4q-2023-financial-results-en-14mar2024-final.pdf | beursmelding |
| Pharming FY2021 results (persbericht) | https://www.pharming.com/files/pharming-group-financial-results-full-year-2021-final-1.pdf | beursmelding |
| Pharming FY2020 results (persbericht, EUR) | https://www.pharming.com/files/2020-full-year-results-pr-04mar21-final-pdf.pdf | beursmelding |
| Pharming FY2019 results (persbericht) | https://www.pharming.com/files/pr-full-year-results-2019-1300-analyst-call-sem-1.pdf | beursmelding |
| Pharming FY2018 results (persbericht) | https://www.pharming.com/files/1-pr-full-year-results-2018.pdf | beursmelding |
| Pharming FY2016 results (persbericht) | https://www.pharming.com/sites/default/files/imce/Public%20Documents/2016/PR%20Full%20Year%20Results%202016.pdf | beursmelding |
| Pharming — afronding overname RUCONEST Noord-Amerika van Valeant | https://www.pharming.com/files/closing-of-acquisition.pdf | beursmelding |
| Pharming — About Pharming (bedrijfsgeschiedenis en mijlpalen) | https://www.pharming.com/our-company/about-pharming | beurswebsite |
| Pharming — Leadership team | https://www.pharming.com/our-company/our-leadership | beurswebsite |
| Beursgenoten — koers Pharming (Euronext, 17-08-2026) | https://www.beursgenoten.nl/koersen/euronext-aandelen-amsterdam/pharming/koers | databron |
| CentralCharts — koers en volume Pharming | https://www.centralcharts.com/en/8697-pharming-group/quotes | databron |
| MarketScreener — Pharming quotes en marktkapitalisatie | https://www.marketscreener.com/quote/stock/PHARMING-GROUP-N-V-12738425/quotes/ | aggregator |
| StockAnalysis — Pharming (PHAR) forecast en consensus | https://stockanalysis.com/stocks/phar/forecast/ | aggregator |
| StockAnalysis — BioCryst Pharmaceuticals (BCRX) | https://stockanalysis.com/stocks/bcrx/ | aggregator |
| StockAnalysis — KalVista Pharmaceuticals (KALV) | https://stockanalysis.com/stocks/kalv/ | aggregator |
| Simply Wall St — Pharming ownership | https://simplywall.st/stocks/us/pharmaceuticals-biotech/otc-phgu.f/pharming-group/ownership | aggregator |
| FRED — 10-Year Treasury Constant Maturity Rate (DGS10) | https://fred.stlouisfed.org/series/DGS10 | databron |
| Damodaran — Historical Implied Equity Risk Premiums | https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html | onderzoeksrapport |
| Damodaran — Industry betas (januari 2026) | https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/Betas.html | onderzoeksrapport |
| Damodaran — Cost of capital by industry (VS, januari 2026) | https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/wacc.html | onderzoeksrapport |
| Grand View Research — Hereditary Angioedema Therapeutics Market | https://www.grandviewresearch.com/industry-analysis/hereditary-angioedema-hae-therapeutics-market | onderzoeksrapport |
| Chiesi Group — overname KalVista Pharmaceuticals | https://www.chiesi.com/en/media-hub/press-releases/chiesi-group-to-acquire-kalVista-pharmaceuticals-expanding-its-global-rare-disease-portfolio | nieuwsartikel |
| KalVista — FDA-goedkeuring EKTERLY (sebetralstat) | https://ir.kalvista.com/news-releases/news-release-details/kalvista-pharmaceuticals-announces-fda-approval-ekterlyr | nieuwsartikel |
| Pharming — openbaar bod op Abliva AB | https://www.pharming.com/sites/default/files/imce/Press%20releases/Pharming%20announces%20public%20cash%20offer%20to%20the%20shareholders%20of%20Abliva%20AB_EN_15DEC24.pdf | beursmelding |
| Nasdaq — Pharming Group ADS (1 ADS = 10 gewone aandelen) | https://www.nasdaq.com/market-activity/stocks/phar/financials | databron |

---

## 15. Update-historie

| Datum | Versie | Wijziging |
|---|---|---|
| 2026-08-17 | 1.0 | Eerste publicatie |

---

## Opmerkingen voor Claude Code (stage 2)

1. **Twee resultatentabellen, twee valuta.** De valutawissel per 1-1-2021 maakt één doorlopende
   tienjaarsreeks onmogelijk zonder zelf om te rekenen. Ik heb dat bewust niet gedaan. Bij de
   JSON-conversie: `valuta_label` = "USD mln" en vul `financieel.resultatenrekening[]` met de
   jaren 2020-2025 plus TTM. De eurojaren 2015-2019 horen als toelichtende tabel in
   `toelichting_resultaten` of als aparte noot — niet als numerieke rijen in dezelfde reeks.
2. **De DCF is niet met één groeipercentage gemodelleerd** maar met een volledig omzet- en
   margepad. Het blok "DCF-invoeren" bevat de resulterende samengestelde FCF-groei per scenario,
   zodat `dcf_calculator.py` reproduceerbaar is. Let op: `g1` is een resultante, geen invoer —
   het script zal met een uniform groeipercentage een lagere waarde vinden (zie de
   gevoeligheidsmatrix, rij +15% bij 10,5% WACC geeft EUR 0,310) omdat het versnellende karakter
   van het kasstroompad verloren gaat. Wijkt de scriptuitkomst meer dan 1% af van EUR 0,38 (basis),
   noteer het verschil hier en neem de scriptuitkomst over.
3. **Fair values zijn in EUR**, de kasstromen in USD, omgerekend tegen EUR/USD 1,157 (14-08-2026).
   Zet `fair_value.valuta_kasstromen` op "USD" en let erop dat `executive_summary.fair_value_basis`
   in EUR staat, consistent met `meta.koers`.
4. **Openstaande haallijst** (zie Stap 0.5): 20-F FY2025 volledig, Annual Report 2025 PDF,
   remuneratierapport 2024/2025. Deze dekken de beloningssectie, het personeelsaantal, de
   grootaandeelhouders en de geografische omzetsplitsing.
5. **Wat werkte en wat niet — voor de METHODE-lijst:**
   - *Betrouwbaar:* SEC R-pagina's van het 20-F (R3/R5/R7/R41) geven de volledige geauditeerde
     jaarrekening in kleine, goed leesbare stukken. Voor elke SEC-filer is dit sneller en
     betrouwbaarder dan het hoofddocument. De IR-index van Pharming zelf is uitzonderlijk compleet
     en gaat terug tot 2016.
   - *Werkt niet:* de segmentpagina's van het 20-F (R11, R42) renderen leeg — de XBRL-viewer toont
     alleen puntjes. Yahoo Finance en StockAnalysis geven voor PHARM.AS structureel verouderde
     koersen (weken tot maanden achter). live.euronext.com levert geen koersdata via fetch.
     Beursduivel, IEX en de FT blokkeren via robots.txt. Voor een actuele Euronext-koers werkte
     **beursgenoten.nl** wel, bevestigd door centralcharts.com.
   - *Let op bij Nederlandse dubbelnoteringen:* Pharming's Nasdaq-notering is een ADS die tien
     gewone aandelen vertegenwoordigt. Analistenkoersdoelen in dollars (US$34,35) moeten door tien
     én door de wisselkoers voordat ze met de Amsterdamse koers vergelijkbaar zijn.
