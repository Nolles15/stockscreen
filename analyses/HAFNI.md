# Research: HAFNI — Hafnia Limited

> Stage-1 analyse (cowork). Output = uitsluitend dit bestand. Claude Code doet stage 2
> (JSON-injectie, validator, build, commit). Valuta kasstromen: USD. Koers: NOK.

---

## Bronnen-inventaris (Stap 0.5)

Alle financiële cijfers van Hafnia worden in **USD** gerapporteerd (functionele valuta).
De koers noteert in **NOK** op Oslo Børs. Omrekening per peildatum: **USD/NOK = 9,54**
(spot 12-06-2026). Hafnia rapporteert "Operating revenue (Hafnia + TC vessels)" en de
non-IFRS-maatstaf **TCE income**; aggregators (S&P Global via StockAnalysis) tonen een
hogere "Revenue" omdat die de doorstroom-omzet van externe pool-schepen meerekent
(netto nul effect op TCE/winst). In de tabellen gebruik ik Hafnia's **eigen gerapporteerde
operating revenue en TCE** voor de recente jaren (HOOG), en de aggregator voor balans-,
kasstroom- en oudere cijfers (AGGREGATOR).

```
Jaar 2025 — HOOG
  Bron: Hafnia FY2025 results press release (Business Wire, 26-02-2026) + Audited
        Financial Statements 2025 (20-F, gepubliceerd 16-04-2026)
  URL:  https://www.businesswire.com/news/home/20260225445913/en/Hafnia-Limited-Announces-Financial-Results-For-The-Three-and-Twelve-Months-Ended-31-December-2025
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: operating revenue (1.421,8), TCE income (955,9), adj. EBITDA (559,5),
                       net profit (339,7), pretax (342,2), totale activa (3.811,9),
                       totaal eigen vermogen (2.329,6), cash (103,6), ROE (14,8%), ROIC (11,2%),
                       equity ratio (61,1%), net LTV (24,9%), DPS (0,5457), TCE/dag (25.206)

Jaar 2024 — HOOG
  Bron: Hafnia FY2025 press release (comparatieven 2024) + FY2024 audited statements
  URL:  https://www.businesswire.com/news/home/20260225445913/en/Hafnia-Limited-Announces-Financial-Results-For-The-Three-and-Twelve-Months-Ended-31-December-2025
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: operating revenue (1.935,6), TCE income (1.391,3), adj. EBITDA (992,3),
                       net profit (774,0), TCE/dag (33.000), DPS (1,158)
  Cijfers NIET overgenomen: gedetailleerde balanspost-splitsing 2024 (uit aggregator)

Jaar 2023 — HOOG
  Bron: Hafnia FY2023 quarterly/annual report (Q4 2023) + persbericht
  URL:  https://s201.q4cdn.com/891122012/files/doc_financials/2023/q4/Quarterly-Earnings-report_Q4-2023_vF2.pdf
  Daadwerkelijk geopend: ja (record net profit 793,3 bevestigd; ook via StockAnalysis/S&P)
  Cijfers overgenomen: net profit (793,3), DPS (1,004)
  Aanvulling (omzet/EBITDA/balans 2023): StockAnalysis/S&P (AGGREGATOR, zie hieronder)

Jaar 2022 — HOOG/AGGREGATOR (gemengd)
  Bron: Hafnia FY2022 resultaten (net profit, Q4 2022 EBITDA) + StockAnalysis/S&P
  URL:  https://stockanalysis.com/quote/osl/HAFNI/financials/
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: net profit (751,6), EBITDA (1.007), revenue-aggregator (1.927),
                       balans 2022, kasstromen 2022

Jaar 2021 — AGGREGATOR
  Bron: StockAnalysis.com (S&P Global Market Intelligence) income/cashflow/balance
  URL:  https://stockanalysis.com/quote/osl/HAFNI/financials/
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: revenue (811,2), net loss (-55,5), EBITDA (96,1), FCF (79,8),
                       balans 2021, kasstromen 2021

Jaar 2020 — GEEN BRON BESCHIKBAAR (geverifieerd niveau)
  Zoekpoging(en): StockAnalysis (historie start 2021 voor deze ticker), Hafnia IR
                  (eerste volledige boekjaar na 2019-merger), MacroTrends (client-rendered,
                  niet leesbaar via fetch)
  Conclusie: 2020 blijft LEEG in de tabellen. Genoteerd in ontbrekende_data. Hafnia is in
             huidige vorm pas eind 2019 ontstaan (BW Tankers × Hafnia Tankers merger);
             2020 was Covid-jaar met beperkt vergelijkbare perimeter.

Jaren 2016-2019 — GEEN BRON BESCHIKBAAR (pre-merger / niet-vergelijkbaar)
  Zoekpoging(en): de huidige juridische entiteit (Hafnia Limited, voorheen BW Tankers)
                  bestaat pas sinds de 2019-merger; pre-merger cijfers zijn niet op
                  vergelijkbare geconsolideerde basis publiek beschikbaar.
  Conclusie: 2016-2019 blijven LEEG. Genoteerd in ontbrekende_data.
```

**Niet-financiële bronnen geopend:** koers/statistiek (StockAnalysis, 12-06-2026),
dividendhistorie (StockAnalysis/S&P), aandeelhouders (Business Wire / MarketScreener,
BW Group 44,18%), insider-transacties (GuruFocus/StockTitan/Business Wire), analisten
(TheFly: SEB, DNB Carnegie, Pareto), macro (US 10y treasury, Damodaran ERP, USD/NOK),
peer-data (Nortilus/Golden Horn Substack), Q1 2026 resultaten (Business Wire 26-05-2026).

**Zelf-check:** voor elke ingevulde numerieke cel hieronder is een bron-URL uit deze
inventaris beschikbaar. Cellen voor 2016-2020 zijn bewust leeg.

---

## Metadata
- **Ticker (bare):** HAFNI
- **Yahoo symbol:** HAFNI.OL
- **Exchange:** OSL (Oslo Børs)
- **Sector (GICS-achtig):** Industrie
- **Industrie:** Scheepvaart (product- en chemicaliëntankers)
- **Land:** Bermuda (hoofdkantoor Singapore; onderdeel BW Group)
- **Peildatum analyse:** 2026-06-12
- **Koers op peildatum:** 70,35
- **Valuta:** NOK
- **Marktkapitalisatie:** NOK 39,5 mld (≈ USD 3,7 mld)
- **Marktkap in mln (lokale valuta):** 35.161
- **Free float pct:** 48
- **Indexlidmaatschap:** OBX (Oslo Børs)
- **Domein:** hafnia.com

---

## 1. Executive summary

- **Kernthese:** Hafnia is de grootste operator van product- en chemicaliëntankers ter
wereld, met circa 200 schepen onder beheer en een eigen vloot van ruim honderd moderne,
relatief jonge tankers (LR2, LR1, MR en Handy). Het bedrijf verdient geld door geraffineerde
olieproducten en chemicaliën over zee te vervoeren tegen vrachttarieven die sterk fluctueren
met vraag, aanbod van scheepsruimte en geopolitiek. De afgelopen jaren waren uitzonderlijk
winstgevend doordat de oorlog in Oekraïne, sancties op Rusland en omleidingen rond de Rode
Zee de vaarafstanden verlengden en de tarieven omhoog stuwden. Structurele steunpunten zijn
een verouderende wereldvloot, een groeiende "schaduwvloot" die uit de reguliere markt wordt
geduwd, en aanhoudend sterke productexportstromen vanuit de VS-Golf, het Midden-Oosten en
China. Hafnia onderscheidt zich met de laagste operationele- en overheadkosten per dag onder
beursgenoteerde peers, een geïntegreerd platform (technisch beheer, bevrachting, poolbeheer,
bunkerinkoop) en een gedisciplineerd dividendbeleid gekoppeld aan de loan-to-value. Het
belangrijkste risico is de cycliciteit: vrachttarieven kunnen scherp dalen wanneer een
recordorderboek aan nieuwbouw wordt opgeleverd en de geopolitieke premie wegvalt.
- **Oordeel:** HOLD
- **Fair value basis** (basisscenario DCF, lokale valuta): 74,5
- **Fair value kansgewogen:** 78,4
- **EPV per aandeel** (Earnings Power Value, zonder groeipremie): 57,5
- **Upside pct:** 5,9 (op basis-fair-value; kansgewogen +11,4%)
- **Fair value scenarios:**

| Scenario | Fair value | Upside % | FCF groei % | WACC % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 39,0 | -44,6 | 0,0 | 9,0 | 25 |
| Basis | 74,5 | 5,9 | 2,0 | 8,1 | 50 |
| Optimistisch | 125,4 | 78,3 | 3,5 | 7,5 | 25 |

- **Reverse-DCF impliciete groei pct:** 1,2
- **Grootste kans:** Aanhoudende geopolitieke verstoring en een verouderende vloot houden de
tarieven langer hoog dan de markt inprijst, terwijl Hafnia's TORM-belang tot waardecreërende
consolidatie leidt.
- **Grootste risico:** Een recordorderboek aan nieuwbouw plus het wegvallen van de
geopolitieke premie drukt de vrachttarieven terug naar mid-cycle of lager, waardoor winst,
dividend en koers fors dalen.

---

## 2. Bedrijfsprofiel

- **Beschrijving:** Hafnia Limited is een investerings- en scheepvaartholding die zich richt
op het vervoer van geraffineerde olieproducten (zoals diesel, benzine, kerosine, nafta),
plantaardige oliën en eenvoudige chemicaliën over zee. Het bedrijf bedient nationale en
internationale oliemaatschappijen, chemieconcerns en handels- en nutsbedrijven. Hafnia zit
in het hart van de energiewaardeketen: zodra ruwe olie tot producten is geraffineerd, vervoert
Hafnia die producten van raffinaderij naar afzetmarkt. De vloot is onderverdeeld in vier
scheepsklassen — Long Range II (LR2), Long Range I (LR1), Medium Range (MR) en Handy —
elk geschikt voor andere routes en ladingen. Hafnia is uniek door zijn schaal (grootste
product-/chemietankeroperator ter wereld, circa 200 schepen onder beheer), zijn jonge en
brandstofefficiënte vloot, en zijn volledig geïntegreerde platform dat naast het bezit van
schepen ook technisch management, commerciële bevrachting, poolbeheer en grootschalige
bunkerinkoop omvat. De omzet komt tot stand via twee modellen: spot-bevrachting
(voyage charters, waarbij Hafnia per reis tegen de actuele marktprijs vaart) en
time charters (vaste dagtarieven over langere periodes). Via de pools deelt Hafnia ook
het commerciële beheer van schepen van derden, wat schaal en bezettingsgraad verhoogt.
- **Geschiedenis:** Hafnia in zijn huidige vorm ontstond eind 2019 uit de fusie van Hafnia
Tankers (mede opgericht en geleid door Mikael Skov) en BW Tankers, onderdeel van het Aziatische
BW Group dat al meer dan tachtig jaar actief is in olie- en gastransport. De gefuseerde
onderneming nam alle poolbedrijven over en werd genoteerd op Oslo Børs in 2019, waarbij
BW Tankers de overlevende juridische entiteit was maar de naam Hafnia aannam. In 2017-2018
ging Hafnia Tankers al joint ventures aan met Vista Shipping en CSSC. Na de beursgang in Oslo
profiteerde Hafnia van de zwakke tankermarkt in 2020-2021 (Covid drukte de vraag, 2021 was
een verliesjaar) om vervolgens vanaf 2022 te exploderen: de oorlog in Oekraïne en sancties op
Rusland verlengden vaarafstanden dramatisch, waardoor de tarieven naar recordhoogten stegen
en Hafnia in 2023 en 2024 recordwinsten boekte. In juni 2024 sloot Hafnia een
aandeelhoudersovereenkomst met BW Group. In 2024 verkreeg Hafnia een tweede notering op de
New York Stock Exchange (NYSE: HAFN). Eind 2025 zette Hafnia een grote strategische stap door
13,97% van concurrent TORM over te nemen van Oaktree — een opmaat naar mogelijke consolidatie
in een versnipperende sector. In 2026 zette Hafnia zijn vlootvernieuwing voort met de
bestelling van acht nieuwe MR-tankers bij Hyundai Heavy Industries en de verkoop van oudere
schepen.
- **Bedrijfsmodel:** Hafnia verdient geld door scheepsruimte te verhuren of per reis in te
zetten tegen vrachttarieven. De kernmaatstaf is TCE (time charter equivalent) per dag — de
netto-dagopbrengst na aftrek van reiskosten (brandstof, havengelden, commissies). De omzet is
grotendeels niet-terugkerend en sterk cyclisch: spot-tarieven schommelen met de wereldwijde
balans tussen vraag (tonne-miles) en aanbod (vlootomvang). Daarnaast genereren fee-based
activiteiten (technisch beheer, poolbeheer voor derden, bunkerinkoop) een kleinere, stabielere
inkomstenstroom. Hafnia keert een hoog en variabel dividend uit, gekoppeld aan een
payout-ratio die stijgt naarmate de loan-to-value daalt.
- **IPO-context:** Hafnia werd in 2019 genoteerd op Oslo Børs als gevolg van de fusie tussen
Hafnia Tankers en BW Tankers; BW Group bleef de dominante aandeelhouder. In 2024 volgde een
tweede notering op de NYSE (HAFN). De kapitaalstructuur is sindsdien verbeterd: de schuld is
fors afgelost met de sterke kasstromen van 2022-2024, en de net loan-to-value daalde naar
circa 20-25%.
- **Klantprofiel:** B2B. De klanten zijn nationale en internationale oliemaatschappijen,
chemieconcerns en handels-/nutsbedrijven die geraffineerde producten en chemicaliën moeten
vervoeren. In de spotmarkt is er een brede, wisselende klantenbasis zonder structureel hoge
concentratie bij één afnemer; via pools en time charters zijn er langduriger relaties. Exacte
klantconcentratiecijfers worden niet publiek uitgesplitst.
- **Oprichtingsjaar:** 2010 (Hafnia Tankers; huidige entiteit via 2019-merger)
- **IPO-datum:** 2019 (Oslo Børs); 2024 (NYSE secundaire notering)
- **IPO-koers (lokale valuta):** Niet verifieerbaar — weggelaten
- **Personeel (FTE):** 4.876 (StockAnalysis/IR; ruim 4.000 onshore en op zee)
- **Landen actief:** Wereldwijd; kantoren in Singapore, Kopenhagen, Houston en Dubai
- **Klantconcentratie:** Niet publiek uitgesplitst — weggelaten (spotmarkt impliceert lage
structurele concentratie)

### Geografische spreiding (omzet)
| Regio | Omzet % | Valuta-exposure |
|---|---|---|
| Wereldwijd (spot/TC, in USD afgerekend) | — | USD |

**Toelichting geografie:** Hafnia opereert in een wereldwijde spotmarkt waarin vrachttarieven
en afrekeningen in USD plaatsvinden; een geografische omzetsplitsing per land is niet
betekenisvol of publiek uitgesplitst voor een tankeroperator. De facto is er een natural hedge:
zowel inkomsten als de meeste kosten (brandstof, scheepsfinanciering) luiden in USD, waardoor
het valutarisico beperkt is. Het dividend wordt voor Oslo-genoteerde aandelen in NOK omgerekend,
waardoor de Noorse belegger wél USD/NOK-koersrisico loopt op de uitkering en koers.

### Segmenten
| Naam | Omzet % | Beschrijving |
|---|---|---|
| LR2 | — | Long Range II tankers; grootste schepen, vervoeren grotere ladingen producten/dirty op lange routes. |
| LR1 | — | Long Range I tankers; ruggengraat voor middellange en lange producttrades. |
| MR | — | Medium Range tankers; grootste deel van de vloot, flexibel inzetbaar wereldwijd, inclusief IMO II chemicaliën. |
| Handy | — | Kleinste klasse; korte-afstand en regionale producttrades, veel IMO II chemie-capabel. |

*(Omzet per segment wordt niet als percentage uitgesplitst in publieke bronnen; per-segment
TCE/dag wel — zie sector-KPI's.)*

### Aandeelhouders (top 5)
| Naam | Belang % | Type (oprichter / PE / institutioneel / retail) |
|---|---|---|
| BW Group Limited | 44,18 | Controlerend |
| Institutionele beleggers (totaal) | 22,5 | Institutioneel |
| Insiders | 0,3 | Institutioneel |

- **Institutioneel eigendomstrend:** Stabiel tot licht stijgend. Sinds de NYSE-notering in
2024 is de zichtbaarheid bij Amerikaanse instituten toegenomen; BW Group blijft met ruim 44%
de verankerde, controlerende aandeelhouder, wat de free float beperkt tot circa de helft.

---

## 3. Financieel — historische data (10 jaar + TTM)

> **Belangrijk:** Hafnia bestaat in huidige geconsolideerde vorm pas sinds eind 2019.
> Vergelijkbare data vóór 2021 is niet verifieerbaar beschikbaar; die jaren blijven leeg
> (zie Bronnen-inventaris en ontbrekende_data). De "omzet" hieronder is Hafnia's eigen
> **operating revenue (Hafnia + TC vessels)** voor 2024-2025 (HOOG); voor 2021-2023 wordt de
> S&P/StockAnalysis-omzet getoond (AGGREGATOR), die de bruto pool-doorstroom meerekent en
> daardoor hoger ligt dan de operating revenue. Beide zijn gemarkeerd.

### Resultatenrekening (bedragen in mln USD)

| Jaar | Omzet | Omzetgroei % | Brutowinst | Brutomarge % | EBIT | EBIT-marge % | EBITDA | EBITDA-marge % | Nettowinst | Nettomarge % | EPS | EPS-groei % | Aandelen mln |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2016 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2017 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2018 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2019 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2020 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2021 | 811 | — | 188 | 23,1 | 0 | 0,0 | 96 | 11,9 | -55 | -6,8 | -0,15 | — | 363 |
| 2022 | 1.927 | 137,5 | 1.050 | 54,5 | 798 | 41,4 | 1.007 | 52,3 | 752 | 39,0 | 1,57 | — | 478 |
| 2023 | 2.672 | 38,7 | 1.063 | 39,8 | 802 | 30,0 | 1.013 | 37,9 | 793 | 29,7 | 1,56 | -0,6 | 505 |
| 2024 | 1.936 | — | — | — | 777 | — | 992 | 51,2 | 774 | — | 1,52 | -2,6 | 510 |
| 2025 | 1.422 | — | — | — | 357 | 25,1 | 559 | 39,3 | 340 | 23,9 | 0,68 | -55,3 | 498 |
| TTM (Q1'26) | — | — | — | — | — | — | — | — | 456 | — | 0,90 | — | 499 |

> **Let op gemengde omzet-grondslag:** 2021-2023 = S&P/aggregator bruto-omzet (incl. externe
> pool-doorstroom); 2024-2025 = Hafnia operating revenue (Hafnia + TC vessels). EBITDA 2024/2025
> en netto-winst alle jaren zijn Hafnia-gerapporteerd/HOOG of S&P-bevestigd. EBIT-marge/EBITDA-marge
> zijn op de getoonde omzetbasis berekend en daardoor niet 1-op-1 vergelijkbaar tussen de twee
> grondslagen — zie toelichting.

- **Toelichting resultaten:** De cijfers vertellen het verhaal van een extreem cyclisch
bedrijf. 2021 was een verliesjaar (Covid-nasleep, zwakke tarieven). Vanaf 2022 explodeerde de
winstgevendheid: de oorlog in Oekraïne en sancties op Rusland verlengden de vaarafstanden,
waardoor de TCE/dag van circa USD 13.000 (2021) naar een piek van USD 33.000 (2024) steeg en
de nettowinst opliep tot een record van USD 793 miljoen (2023) en USD 774 miljoen (2024). In
2025 normaliseerde de markt deels: de TCE/dag zakte naar USD 25.206 en de nettowinst halveerde
naar USD 340 miljoen. Dit is geen structurele verslechtering maar de natuurlijke ademhaling
van de cyclus. De winst per aandeel volgde dezelfde curve (USD 1,57 in topjaren naar USD 0,68
in 2025). Het aantal aandelen steeg eerst door de fusie en kapitaalmarkt-activiteit en daalt nu
licht door inkoop en intrekking van treasury-aandelen.
- **Omzet-CAGR:** Niet betekenisvol over deze periode vanwege de wisseling van omzet-grondslag
en de extreme cycliciteit; de operating revenue daalde van USD 1.936 mln (2024) naar USD 1.422
mln (2025).

### Kasstromen (mln USD)

| Jaar | CFO | Capex | FCF | FCF na SBC | FCF/aandeel | FCF-marge % | FCF-groei % | FCF-conversie % | SBC | Dividend totaal | Aandeleninkoop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021 | 106 | -27 | 80 | 77 | 0,22 | 9,8 | — | n.m. | 3,2 | 0 | 0 |
| 2022 | 771 | -447 | 324 | 322 | 0,66 | 16,8 | 305,8 | 43,1 | 1,8 | 244 | 0 |
| 2023 | 1.061 | -184 | 876 | 874 | 1,72 | 32,8 | 170,7 | 110,5 | 2,8 | 544 | 0 |
| 2024 | 1.030 | -50 | 981 | 978 | 1,90 | 34,2 | 11,9 | 126,7 | 3,0 | 700 | 49 |
| 2025 | 603 | -146 | 457 | 454 | 0,91 | 20,0 | -53,4 | 134,5 | 3,2 | 199 | 28 |
| TTM (Q1'26) | 593 | -140 | 454 | 450 | 0,90 | — | -47,1 | 99,4 | 3,3 | 272 | 0 |

- **Toelichting kasstromen:** De vrije kasstroom volgt de winstcyclus maar wordt extra vertekend
door scheepsverkopen en de capex-cyclus. De zeer hoge FCF-conversie in 2023-2025 (>110%)
weerspiegelt dat een deel van de "FCF" voortkomt uit opbrengsten van vlootverkopen en
werkkapitaalvrijval, niet puur uit operationele winst — een belangrijke reden om voor de
DCF níet de recente FCF als basis te nemen (zie cycliciteitscheck H12). De capex schommelt
sterk: USD 447 mln in 2022 (vlootuitbreiding) tegenover slechts USD 50 mln in 2024. Stock-based
compensation is bescheiden (circa USD 3 mln per jaar, <1% van FCF) en dus geen materieel
verwateringsrisico. De daling van CFO in 2025 (-41%) is volledig toe te schrijven aan de lagere
vrachttarieven en niet aan een structurele verslechtering; het is de normale cyclische
terugval na twee recordjaren.

### Balans-ratio's (mln USD waar van toepassing)

| Jaar | Nettoschuld | Nettoschuld/EBITDA | Eigen vermogen | ROE % | ROIC % | ROCE % | Current ratio | Solvabiliteit % | Goodwill % van EV | Working capital |
|---|---|---|---|---|---|---|---|---|---|---|
| 2021 | 1.231 | 12,8 | 1.112 | -5,0 | — | — | 1,09 | 44,3 | — | 27 |
| 2022 | 1.601 | 1,6 | 2.009 | 37,4 | — | — | 1,82 | 50,9 | — | 394 |
| 2023 | 1.151 | 1,1 | 2.228 | 35,6 | — | — | 1,41 | 56,9 | — | 272 |
| 2024 | 927 | 0,9 | 2.263 | 34,5 | — | — | 1,38 | 61,1 | — | 246 |
| 2025 | 1.019 | 1,8 | 2.330 | 14,8 | 11,2 | 11,0 | 1,48 | 61,1 | — | 272 |
| TTM (Q1'26) | 879 | — | 2.541 | 19,1 | 13,4 | — | 1,48 | 65,3 | — | 426 |

- **Toelichting balans:** Hafnia's balans is in vier jaar sterk verbeterd. De net loan-to-value
daalde van boven de 40% naar circa 20-25%, en de solvabiliteit (equity ratio) steeg naar ruim
61%. De nettoschuld inclusief lease-verplichtingen schommelt rond USD 0,9-1,0 miljard, ruim
afgedekt door een eigen vermogen van USD 2,3-2,5 miljard en een vlootmarktwaarde van circa
USD 3,9 miljard. De bruto schuld (circa USD 1,1 mld) en nettoschuld bewegen grotendeels parallel.
Goodwill is verwaarloosbaar — Hafnia is een asset-heavy reder zonder grote immateriële balans.
Het belangrijkste recente balansfeit is de aankoop van het TORM-belang (USD ~310 mln in
"investment in securities" in 2025), wat de net LTV tijdelijk verhoogde naar 24,9%. De
Altman Z-score van 2,39 oogt laag maar is typisch voor kapitaalintensieve reders en geen
acuut faillissementssignaal gezien de lage leverage en sterke kasstroom.

### Kapitaalstructuur huidig
- **Nettoschuld (huidig):** USD ~879 mln (TTM Q1'26, incl. leases) — USD ~1.019 mln (FY2025)
- **Bruto schuld:** USD ~1.123 mln (FY2025)
- **Cash & equivalents:** USD 103,6 mln (FY2025); USD 146 mln (TTM)
- **Lease-verplichtingen (IFRS-16):** USD ~75 mln (current + non-current, sterk gedaald)
- **Gemiddelde rente %:** ~4,4 (rente-expense USD 49,8 mln op ~USD 1,1 mld schuld)
- **Rente-dekking (EBIT/rente):** ~7,2 (TTM)

### Non-GAAP / aanpassingen
- **Gebruikt?** true
- **Welke aanpassingen:** Hafnia rapporteert TCE income en Adjusted EBITDA (non-IFRS). TCE =
omzet minus reiskosten; Adjusted EBITDA = winst vóór financiële posten, afschrijving,
impairment, amortisatie en belastingen, plus aanpassing voor vesselverkoopwinsten en
JV-resultaten.
- **Waarom:** Standaard in de scheepvaartsector om periodes vergelijkbaar te maken ondanks
wisselingen in charter-mix; ik gebruik IFRS-nettowinst als primaire grondslag en TCE/Adj.
EBITDA voor de cyclus-normalisatie.

---

## 4. Moat (concurrentievoordeel)

- **Oordeel:** NARROW MOAT
- **Moat-categorieën:**

| Naam | Sterkte (sterk/middel/zwak/geen) | Toelichting |
|---|---|---|
| Immateriële activa | zwak | Geen patenten of merkpricing in een commodity-vrachtmarkt. Wel waarde in IMO II-certificering en relaties, maar geen beschermende immateriële activa die overwinst afdwingen. |
| Overstapkosten | zwak | Bevrachters kiezen per reis op prijs en beschikbaarheid; overstappen naar een andere reder kost vrijwel niets. Time charters en pools binden klanten beperkt en tijdelijk. |
| Netwerkeffecten | middel | Het poolplatform creëert een mild netwerkeffect: meer schepen onder beheer verhoogt bezettingsgraad, dekking en commerciële slagkracht. Het maakt het product echter niet waardevoller per extra gebruiker op klassieke wijze. |
| Kostenvoordeel | sterk | Hafnia heeft aantoonbaar de laagste OPEX + G&A per dag onder beursgenoteerde peers (circa USD 9.600/dag), een jonge brandstofefficiënte vloot (~64% ECO) en gunstige debt-structuur. Dit verlaagt de cash-breakeven en levert door de cyclus heen structureel hogere marges. |
| Efficiënte schaal | middel | Als grootste operator (circa 200 schepen) geniet Hafnia inkoop-, financierings- en bevrachtingsschaalvoordelen. De productankermarkt is echter gefragmenteerd met veel spelers, dus van een beschermde nichemarkt met beperkte ruimte voor concurrenten is geen sprake. |

- **Kwantitatief bewijs:** Hafnia's ROIC bedroeg 11,2% in 2025 (een normaler jaar) en piekte
ruim daarboven in 2023-2024. De ROIC-WACC-spread is in 2025 circa 3 procentpunt positief
(ROIC 11,2% vs WACC ~8,1%), wat op waardecreatie wijst, maar de spread is bescheiden en sterk
cyclus-afhankelijk: in dal-jaren als 2021 was de spread negatief. Het kostenvoordeel is het
best kwantificeerbare moat-element: een lagere cash-breakeven dan peers betekent dat Hafnia in
zwakke markten langer winstgevend blijft en in sterke markten meer overhoudt.
- **Duurzaamheid:** 5-10 jaar voor het kostenvoordeel
- **Erosierisico's:** Het kostenvoordeel kan eroderen als peers hun vloot eveneens
moderniseren en deleveragen (Scorpio en Torm verlagen actief hun breakeven). De jonge vloot
veroudert geleidelijk, en nieuwe milieuregelgeving (EEXI/CII, mogelijke koolstofheffingen)
kan het relatieve voordeel verschuiven. Het netwerkeffect van de pools is repliceerbaar door
grote concurrenten. In een commodity-markt blijft de fundamentele kwetsbaarheid dat tarieven
door vraag en aanbod worden bepaald, niet door bedrijfsspecifieke pricing power.

---

## 5. Management

- **CEO-naam + tenure:** Mikael Skov — CEO sinds de fusie (2019); medeoprichter en oud-CEO van
Hafnia Tankers, ruim 33 jaar scheepvaartervaring
- **CFO-naam + tenure:** Perry van Echtelt — CFO sinds begin 2019; >20 jaar ervaring in
investment banking en scheepsfinanciering (ABN AMRO/MeesPierson/Fortis)
- **Oprichter nog betrokken?** Ja — Mikael Skov (medeoprichter Hafnia Tankers) is nog CEO
- **Insider ownership %:** ~0,3 (gerapporteerd insider-ownership); BW Group (44,18%) is de
verankerde controlerende eigenaar
- **Capital allocation track record:**

| Jaar | Dividend totaal | Aandeleninkoop | M&A uitgaven | Organische capex |
|---|---|---|---|---|
| 2022 | 244 | 0 | — | 447 |
| 2023 | 544 | 0 | — | 184 |
| 2024 | 700 | 49 | — | 50 |
| 2025 | 199 | 28 | ~310 (TORM-belang) | 146 |

- **M&A-track-record:** De vormende transactie was de 2019-fusie BW Tankers × Hafnia Tankers,
die schaal en een geïntegreerd platform creëerde. In december 2025 nam Hafnia een 13,97%-belang
in TORM (USD ~310 mln) als strategische opstap naar mogelijke consolidatie — de uitkomst en
timing zijn onzeker en het management benadrukt een gedisciplineerde, geduldige aanpak. Verder
draait het kapitaalbeleid om vlootvernieuwing: oudere schepen verkopen op hoge waarderingen en
selectief nieuwbouw bestellen (8 MR-tankers bij Hyundai in 2026).
- **Beloning:** Bonussen en een Long Term Incentive Program (RSU's en opties). De jaarlijkse
SBC is bescheiden (~USD 3 mln, <1% van marktkap), dus verwatering is geen materieel probleem.
In maart 2026 kende het bestuur 964.609 opties toe onder het bonus-/LTI-programma.
- **Oordeel management:** STERK
- **Toelichting:** Het management heeft een sterk track record: het bouwde de grootste
product-tankeroperator ter wereld, behaalde de laagste kostenstructuur in de sector, deleveragede
de balans agressief in de goede jaren en koppelde het dividend transparant aan de
loan-to-value. De compensatie is redelijk en de verwatering minimaal. Twee aandachtspunten
temperen het beeld: insiders waren recent netto verkopers (CEO Skov verkocht in april 2026
1 miljoen aandelen à ~USD 8,11-8,12, al behoudt hij ruim 3,3 miljoen aandelen/opties/RSU's),
en de controlerende positie van BW Group (44%) betekent dat minderheidsaandeelhouders afhankelijk
zijn van de governance-afspraken met de grootaandeelhouder. Per saldo overtuigt de capital
allocation en operationele uitvoering, wat een STERK-oordeel rechtvaardigt.

---

## 6. Sector & concurrentie

- **Sector-groeivooruitzicht:** De product-tankervraag (tonne-miles) groeit structureel mee met
de wereldwijde olievraag (laag-enkelcijferig per jaar) plus extra rugwind van langere
vaarafstanden door verschoven handelsstromen. Op korte termijn drukt een recordorderboek aan
nieuwbouw het aanbod-evenwicht.
- **Porter five forces:**
  - **Rivaliteit:** HOOG — gefragmenteerde markt met veel spelers (Torm, Scorpio, Ardmore,
    d'Amico e.a.); concurrentie op prijs/beschikbaarheid in een commodity-vrachtmarkt.
  - **Nieuwe toetreders:** MIDDEL — kapitaalintensief (een MR-nieuwbouw kost ~USD 50 mln) en
    lange levertijden (3,5 jaar) vormen drempels, maar kapitaal en schepen zijn voor gevestigde
    partijen verkrijgbaar; geen regulatoir toetredingsslot.
  - **Substituten:** LAAG — pijpleidingen vervoeren geen geraffineerde producten over zee;
    er is geen reëel alternatief voor zeetransport van producten over lange afstanden.
  - **Macht leveranciers:** MIDDEL — werven (sterk geconcentreerd, lange wachttijden) en
    brandstofleveranciers hebben enige macht; Hafnia mitigeert via eigen bunkerinkoop-schaal.
  - **Macht afnemers:** MIDDEL — grote oliemaatschappijen en handelshuizen zijn prijsbewust,
    maar in krappe markten verschuift de macht naar de reder.
- **Concurrenten:**

| Concurrent | Marktaandeel % |
|---|---|
| TORM plc | — |
| Scorpio Tankers | — |
| Ardmore Shipping | — |
| d'Amico International Shipping | — |

- **Positie van het bedrijf:** Marktleider qua schaal (grootste product-/chemietankeroperator,
~200 schepen onder beheer), met de laagste kostenstructuur en een van de jongste vloten onder
beursgenoteerde peers.
- **Positie-toelichting:** Hafnia combineert de grootste schaal met de laagste OPEX+G&A per dag
en een jonge, brandstofefficiënte vloot (gemiddeld ~8,7-9,6 jaar). Tegenover Torm (oudere vloot,
~11,5 jaar) en Scorpio (meer LR2/scrubber-exposure) staat Hafnia gediversifieerder en
kostenefficiënter. Hafnia handelt rond of net onder de NAV (P/NAV ~0,89 op de Q1'26-NAV van
USD 8,09/aandeel), in lijn met het sectorgemiddelde (~1,0) en met de hoogste forward
dividendyield onder peers — een korting die deels de cyclische onzekerheid en de
controlerende-aandeelhouderstructuur weerspiegelt.

### TAM/SAM/SOM
- **TAM (mln lokale valuta):** Niet betrouwbaar te kwantificeren — weggelaten
- **Bron TAM/SAM:** Niet verifieerbaar — weggelaten
- **Toelichting:** Voor een commodity-vrachtmarkt is een TAM/SAM/SOM-penetratiemodel niet
zinvol; de relevante maatstaf is vlootaandeel en tonne-mile-vraag, niet marktpenetratie van
een product. Bewust weggelaten in plaats van een ongefundeerd getal in te vullen.

---

## 7. Analyse-frameworks (9 frameworks, SCORES 0-5)

### Graham
- **Oordeel:** GEDEELTELIJK
- **Graham number:** Niet zinvol berekenbaar voor een cyclische reder met sterk wisselende EPS;
indicatief √(22,5 × EPS × BVPS) met TTM-EPS USD 0,90 en BVPS USD 5,10 ≈ USD 10,2 (> koers USD
7,38), maar de EPS-volatiliteit maakt dit fragiel.
- **Margin of safety %:** +5,9 (t.o.v. basis-DCF NOK 74,5)
- **Toelichting:** Hafnia voldoet aan enkele Graham-criteria: de koers-winstverhouding is laag
(~10,4) en de prijs-boekwaarde (1,68) ligt onder 2,0. De schuld is beheersbaar en er is een
fors dividend. Toch is de margin of safety op de basis-fair-value beperkt (+5,9%, ver onder de
30% die Graham eist) en is de winstgevendheid cyclisch in plaats van stabiel — precies het type
bedrijf waar Graham huiverig voor was. Daarom een gedeeltelijk oordeel.
- **Score (0-5):** 4

### Buffett / Munger
- **Oordeel:** GEDEELTELIJK
- **ROIC structureel boven WACC?** Gedeeltelijk (true in normale/sterke jaren, false in dal-jaren)
- **Toelichting:** Hafnia is een goed geleid, kostenefficiënt bedrijf, maar het mist de
voorspelbare kasstromen en pricing power die Buffett zoekt. De ROIC (11,2% in 2025) ligt boven
de WACC (8,1%), maar de spread is bescheiden en niet structureel door de cyclus heen — in 2021
was de ROIC negatief. De moat is NARROW (kostenvoordeel) in plaats van WIDE. De prijs is
redelijk (P/FCF 8,6), maar de onvoorspelbaarheid van de vrachtmarkt maakt dit geen typisch
Buffett-"wonderful business". Vandaar gedeeltelijk.
- **Score (0-5):** 3

### Peter Lynch
- **Categorie:** Cyclical
- **Oordeel:** NEUTRAAL
- **PEG-ratio:** n.v.t. (winst daalt cyclisch; PEG misleidend voor cyclicals)
- **Toelichting:** Hafnia is een schoolvoorbeeld van een Lynch-"cyclical": de winst beweegt mee
met de vrachtcyclus, niet met seculiere groei. Het verhaal is helder en goed te vertellen
(meer tonne-miles + verouderende vloot + geopolitiek = hogere tarieven), maar Lynch waarschuwde
dat je cyclicals juist moet kopen wanneer ze duur ogen (lage winst, hoge P/E in een dal) en
verkopen wanneer ze goedkoop ogen (piekwinst, lage P/E). Met de winst die net van een piek
afkomt en de markt nog sterk, is het instapmoment in de cyclus ambigu — neutraal.
- **Score (0-5):** 2

### Phil Fisher
- **Oordeel:** GEMIDDELD
- **Toelichting:** Vanuit Fisher-perspectief scoort Hafnia gemengd. Er is geen R&D-cultuur of
productinnovatie in klassieke zin — het is een asset-operator in een commodity-markt. Wel is er
operationele excellentie (laagste kosten, jonge vloot, geïntegreerd platform) en een management
met integriteit en transparantie. De winstmarge wordt echter niet beschermd door een duurzame
moat maar door de marktcyclus. Het groeipotentieel zit in consolidatie (TORM) en vlootvernieuwing,
niet in een uniek product. Gemiddeld.
- **Score (0-5):** 2

### Magic Formula (Greenblatt)
- **Oordeel:** GEMIDDELD
- **Earnings yield %:** ~7,0 (mid-cycle EBIT ~USD 330 mln / EV ~USD 4,7 mld); op TTM-basis hoger
- **Return on capital %:** ~11,9 (EBIT / (working capital + net fixed assets))
- **Toelichting:** Op de Magic-Formula-assen scoort Hafnia een redelijke earnings yield (~7%
mid-cycle, hoger op TTM-basis door piekwinst) maar een matige return on capital (~12%), wat
typisch is voor kapitaalintensieve reders met veel vastgelegd vermogen in schepen. Het is
goedkoop genoeg om interessant te zijn op yield, maar het hoge kapitaalbeslag drukt het
rendement op kapitaal. Per saldo gemiddeld.
- **Score (0-5):** 3

### Moat
- **Score (0-5):** 2 — NARROW moat (1 categorie sterk: kostenvoordeel), ROIC-WACC spread ~3pp (<5pp)

### Management
- **Score (0-5):** 4 — capital allocation goed, prikkels aligned, geen controverses; getemperd door recent insider-verkoop

### Fair Value DCF
- **Score (0-5):** 3 — basis-upside +5,9% (≥0% en <15%)

### Fair Value IPO-gecorr.
- **Score (0-5):** 3 — IPO 2019 (≤10 jaar); geen pre-IPO financial-engineering-distortie, dus IPO-gecorrigeerde upside = basis (+5,9%)

### Scorekaart totaal
- **Totaalscore:** 26
- **Max:** 45
- **Eindoordeel:** HOLD
  - Regel: totaal 26 ≥ 24 EN totaal < 33 → HOLD (Fair Value DCF-score 3, geen PASS-trigger)
- **Samenvatting:** Hafnia is een goed geleid, kostenefficiënt en gedisciplineerd bedrijf dat
de grootste product-tankeroperator ter wereld is, met een sterke balans en een aantrekkelijk
maar variabel dividend. De fundamentele zwakte is de cycliciteit: de winst die de afgelopen
jaren explodeerde door geopolitiek en sancties, normaliseert nu, en een recordorderboek aan
nieuwbouw vormt een tegenwind voor de tarieven. Op een conservatief mid-cycle-genormaliseerde
DCF komt de basis-fair-value op circa NOK 74,5 en de kansgewogen waarde op NOK 78,4 — een
bescheiden upside van circa 11% boven de koers van NOK 70,35, dicht bij de NAV (USD 8,09 =
~NOK 77) en de analistenconsensus (~NOK 82). De EPV zonder groei ligt op NOK 57,5, wat betekent
dat de markt een groeipremie van circa 30% inprijst — verdedigbaar maar niet ruim. Gegeven de
beperkte margin of safety op de basiswaarde, de cyclische onzekerheid en de afhankelijkheid
van de geopolitieke premie luidt het oordeel HOLD: een kwaliteitsreder tegen een redelijke,
niet goedkope prijs. Een koopniveau ontstaat bij circa NOK 56-60 (20-25% korting).

---

## 8. Risico's

| # | Omschrijving | Kans | Impact | DCF-aanname geraakt | Toelichting |
|---|---|---|---|---|---|
| 1 | Recordorderboek nieuwbouw drukt tarieven | HOOG | GROOT | FCF-groei / mid-cycle FCF | Een hoog aanbod aan nieuwe schepen in 2026-2028 kan het tarief-evenwicht verstoren en de TCE/dag terugbrengen naar of onder mid-cycle. Dit raakt direct de genormaliseerde FCF die de DCF-basis vormt. Management erkent dit als belangrijkste swing-factor. |
| 2 | Wegvallen geopolitieke premie | MIDDEN | GROOT | FCF-groei fase 1 | Een de-escalatie in Rusland/Oekraïne, het Midden-Oosten of de Rode Zee zou vaarafstanden verkorten en de tonne-mile-vraag verlagen. Veel van de huidige winst rust op deze tijdelijke premie. |
| 3 | Cyclische tariefdaling (algemeen) | HOOG | GROOT | mid-cycle FCF, terminal value | De vrachtmarkt is inherent volatiel; een terugval naar dal-niveaus (zoals 2021) zou winst, dividend en koers fors drukken. Dit is het pessimistische scenario. |
| 4 | Controlerende aandeelhouder BW Group (44%) | MIDDEN | MIDDEL | WACC (governance-premie) | BW Group's dominante positie beperkt de free float en betekent dat minderheidsbelangen afhankelijk zijn van governance-afspraken; potentiële belangenconflicten bij consolidatie. |
| 5 | TORM-consolidatie mislukt of verwatert | MIDDEN | MIDDEL | FCF / kapitaalallocatie | Het USD ~310 mln TORM-belang is een gok op consolidatie; als die niet doorgaat of slecht geprijsd wordt, is kapitaal vastgelegd zonder synergie. |
| 6 | USD/NOK-koersrisico voor Noorse belegger | MIDDEN | MIDDEL | FV-omrekening | Hafnia's waarde luidt in USD; een sterkere NOK verlaagt de NOK-koers en het NOK-dividend, los van de operationele prestaties. |
| 7 | Milieuregelgeving (EEXI/CII, CO2-heffing) | MIDDEN | MIDDEL | EBIT-marge / capex | Strengere regels kunnen oudere schepen uit de markt duwen (positief voor aanbod) maar ook capex en operationele kosten verhogen. |
| 8 | Pre-IPO financial engineering | LAAG | KLEIN | n.v.t. | Niet geconstateerd: de 2019-merger was aandelen-gebaseerd; schuld werd ná de beursgang afgelost met operationele kasstroom, niet via een pre-IPO dividend-recapitalisatie ten gunste van insiders. Geen materiële vertekening van de historische FCF-reeks. |

---

## 9. These invalide bij

Deze HOLD-these is weerlegd wanneer (a) de vrachttarieven structureel onder de mid-cycle van
~USD 22.000/dag zakken door overaanbod én het wegvallen van de geopolitieke premie, waardoor de
genormaliseerde FCF en het dividend duurzaam dalen; of (b) de koers ruim onder NOK 56-60 zakt
zonder fundamentele verslechtering (dan wordt het een KOOP); of (c) de koers ruim boven NOK 95-100
stijgt op piekwinst-extrapolatie (dan wordt het een PASS, omdat je dan een cyclisch topjaar
inprijst als structureel).

---

## 10. ESG

### Materiële factoren (SASB-gebaseerd)

| Factor | SASB-categorie | Risiconiveau (Laag/Midden/Hoog) | Financiële impact | DCF-impact |
|---|---|---|---|---|
| Broeikasgasemissies scheepvaart | Milieu (GHG Emissions) | MIDDEN | CO2-heffingen (EU ETS), CII-regelgeving verhogen kosten oudere schepen | Marge/capex |
| Brandstofefficiëntie vloot | Milieu (Fuel Management) | MIDDEN | Jonge ECO-vloot (~64%) is relatief voordeel; transitie naar dual-fuel vergt capex | Capex |
| Veiligheid & morsincidenten | Milieu/Sociaal | MIDDEN | Olielekkages dragen reputatie- en aansprakelijkheidsrisico | Eenmalig/staart |
| Governance / controlerende aandeelhouder | Governance | MIDDEN | BW Group 44% — minderheidsbelang-bescherming | WACC-premie |

- **Eindoordeel ESG:** GEMIDDELD RISICO
- **Toelichting:** De grootste ESG-factor is de koolstofintensiteit van scheepvaart. Hafnia is
hier relatief goed gepositioneerd door een jonge, brandstofefficiënte vloot en investeringen in
dual-fuel IMO II-schepen, maar de sector als geheel staat onder toenemende regulatoire druk
(EU ETS, IMO-koolstofregels). Op governance is de dominante positie van BW Group een
aandachtspunt voor minderheidsaandeelhouders.

---

## 11. Katalysatoren

| Datum ca. | Omschrijving | Richting | Impact |
|---|---|---|---|
| 2026-08 | Q2 2026 kwartaalresultaten | NEUTRAAL | MIDDEL |
| 2026-09 | Q3 dividendbesluit (ex-div ~september) | POSITIEF | KLEIN |
| 2026-H2 | Voortgang/uitkomst TORM-consolidatie | BINAIR | GROOT |
| 2026-H2 | Oplevering nieuwbouw-tankers sector (aanbod) | NEGATIEF | GROOT |
| 2026-doorlopend | Sancties Rusland/Iran/Venezuela (handelsstromen) | BINAIR | GROOT |
| 2027-2029 | Oplevering eigen 8 MR-nieuwbouw (Hyundai) | NEUTRAAL | MIDDEL |
| 2026-doorlopend | Vlootverkopen oudere tonnage (boekwinsten) | POSITIEF | KLEIN |

---

## 12. Fair value — kwantitatief (DCF)

### WACC-componenten
- **Risicovrije rente %:** 4,47
- **Bron risicovrije rente:** US 10-jaars Treasury (12-06-2026); kasstromen luiden in USD, dus
  US-staatsrente i.p.v. Noorse staatsrente conform METHODE.md Stap 4-A
- **Type:** nominal (spot). NB: spot (4,47%) ligt ruim boven het 10-jaars gemiddelde (~3,0%);
  een genormaliseerde rente zou de WACC ~1pp verlagen en de fair value verhogen — opgenomen
  als gevoeligheid.
- **ERP (equity risk premium) %:** 4,50
- **Bron ERP:** Damodaran implied ERP 2026 (jan 2026: 4,23%; mature-market ~4,5% gebruikt)
- **Beta (adjusted, Blume):** 1,05
- **Bron beta:** Bottom-up (unlevered tanker/scheepvaart-sectorbeta ~0,80, relevered naar
  Hafnia's markt-D/E ~0,30). De gerapporteerde 5-jaars regressie-beta (-0,16) is onbruikbaar:
  een negatieve beta is voor een cyclische reder economisch onzinnig en weerspiegelt
  idiosyncratische, aanbod-gedreven koersbewegingen.
- **Type beta:** bottom_up
- **Country risk premium %:** 0 (USD-kasstromen, wereldwijde operatie, geen materiële
  single-country exposure)
- **Size premium %:** null (marktkap ~USD 3,7 mld > EUR 2 mld → geen size premium)
- **Cost of equity %:** 9,20 (= 4,47 + 1,05 × 4,50)
- **Schuldkosten na belasting %:** 4,40 (effectieve rente ~4,4%; tonnage-/lage belasting ~1%)
- **E/V gewicht %:** 76,7
- **D/V gewicht %:** 23,3
- **WACC %:** 8,1
- **Sector WACC % (referentie Damodaran):** ~8 (shipping/transport)
- **Illiquiditeitskorting %:** null (gemiddeld dagvolume ~660.000 aandelen, voldoende liquide)

### DCF model-specs
- **Model type:** 2-fase
- **FCF-definitie:** FCF to firm (FCFF)
- **Basis FCF:** USD 310 mln (genormaliseerd, mid-cyclus)
- **Basis FCF na SBC:** USD ~307 mln (SBC ~USD 3 mln in mindering)
- **FCF-type:** Genormaliseerde FCFF USD 310 mln (mid-cyclus) — afgeleid van mid-cycle
  TCE ~USD 22.000/dag → Adj. EBITDA ~USD 470 mln, minus maintenance/drydock-capex ~USD 150 mln
  en cash tax ~USD 10 mln. Bewust níet de recente FCF (USD 457-981 mln), die piek-tarieven en
  vesselverkoopwinsten bevatte (REGEL 1).
- **Groei fase 1 %:** 2,0 (basis)
- **Groei fase 2 %:** n.v.t. (2-fase: direct naar terminal)
- **Terminal groei %:** 1,5 (basis; onder de USD-langetermijninflatie en nominale BBP-groei)
- **Terminal methode:** Gordon growth (met exit-multiple cross-check)
- **Exit multiple gebruikt:** ~6x EV/EBITDA (mid-cycle, sector)
- **Bron exit multiple:** Sector-mediaan product-tankers (Nortilus/peer-data)
- **Terminal value Gordon growth:** USD ~5.263 mln (undiscounted, jaar 5)
- **Terminal value exit multiple:** USD ~2.880 mln (6x mid-cycle EBITDA ~480 mln) — fors lager
- **Terminal value % van totaal:** 72,4 (< 75% drempel ✓)
- **Terminal implied EV/EBITDA:** ~11,0x — VERHOOGD voor een cyclische reder. Rode vlag genoteerd:
  de Gordon-terminal (g=1,5%) impliceert een hogere exit-multiple dan de sector mid-cycle van
  ~6x. Daarom is de basis-FV bewust conservatief gehouden en wordt het pessimistische scenario
  (g=1%, hogere WACC) zwaar meegewogen.
- **Terminal groei consistentie:** Terminal groei 1,5% vereist bij ~10% ROIC een
  herinvesteringsvoet van ~15% — plausibel voor een volwassen, kapitaalintensieve reder die
  netto nauwelijks reële groei genereert (vloot depreciëert; groei vergt herinvestering).
- **Mid-year convention:** true
- **Aandelen uitstaand (mln):** 499,8
- **Nettoschuld huidig:** USD 1.020 mln (FY2025, incl. leases)

### DCF-toelichting
De waardering is een 2-fase FCFF-DCF in USD, omdat zowel Hafnia's inkomsten als kosten in USD
luiden; de uitkomst per aandeel is daarna omgerekend naar NOK tegen 9,54. De cruciale keuze is
het vertrekpunt: Hafnia is sterk cyclisch, dus conform METHODE.md REGEL 1 gebruik ik níet de
recente FCF (die piek-tarieven én vesselverkoopwinsten bevatte) maar een genormaliseerde
mid-cycle FCFF van USD 310 mln, afgeleid uit een mid-cycle TCE van ~USD 22.000/dag. Ik pas de
mid-year convention toe (kasstromen vallen gemiddeld halverwege het jaar). De terminal value
via Gordon growth (g=1,5%) bedraagt 72,4% van de EV — onder de 75%-grens, maar de impliciete
terminal EV/EBITDA (~11x) ligt boven de sector-mid-cycle exit-multiple (~6x), wat ik als rode
vlag markeer en compenseer met een conservatieve basisgroei en een zwaar pessimistisch scenario.
Na aftrek van de nettoschuld (USD 1.020 mln) en deling door 499,8 mln aandelen resulteert de
basis-fair-value in USD 7,81 (≈ NOK 74,5).

### 5-jaars projectie (mln USD, basisscenario)

| Jaar | Omzet | Omzetgroei % | EBIT | EBIT-marge % | NOPAT | Capex | ΔNWC | SBC | FCF |
|---|---|---|---|---|---|---|---|---|---|
| 2026 | — | — | 336 | — | 333 | -153 | -5 | 3 | 316 |
| 2027 | — | — | 343 | — | 340 | -156 | -5 | 3 | 323 |
| 2028 | — | — | 350 | — | 346 | -159 | -5 | 3 | 329 |
| 2029 | — | — | 357 | — | 353 | -162 | -5 | 3 | 336 |
| 2030 | — | — | 364 | — | 360 | -165 | -5 | 3 | 342 |

*(Projectie op FCFF-niveau; omzet/EBIT-marge per jaar niet als puntschatting ingevuld omdat de
DCF op genormaliseerde mid-cycle FCFF draait, niet op een omzet-opbouw — bewust leeg i.p.v.
schijnprecisie.)*

### Scenarios

| Scenario | FCF-groei % | WACC % | Fair value | Upside % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 0,0 | 9,0 | 39,0 | -44,6 | 25 |
| Basis | 2,0 | 8,1 | 74,5 | 5,9 | 50 |
| Optimistisch | 3,5 | 7,5 | 125,4 | 78,3 | 25 |
| Basis (IPO-gecorr.) | 2,0 | 8,1 | 74,5 | 5,9 | — |

- **Kansgewogen fair value:** 78,4 (= 0,25×39,0 + 0,50×74,5 + 0,25×125,4)

### Reverse DCF
- **Impliciete groei %:** 1,2 (terminal/perpetuele FCF-groei die de huidige koers NOK 70,35
  rechtvaardigt, bij WACC 8,1% en mid-cycle FCFF USD 310 mln)
- **Historische FCF CAGR %:** n.m. (cyclisch volatiel; FCF schommelde van USD 80 mln naar
  USD 981 mln en terug)
- **Consensus groei % (analisten):** n.m. (analisten verwachten winstdaling vanaf het piekniveau;
  koersdoelen NOK 73-91, consensus ~82)
- **Interpretatie:** Op de huidige koers prijst de markt een bescheiden perpetuele FCF-groei van
  circa 1,2% in — onder de inflatie. Dat is geen veeleisende aanname: de markt rekent dus níet
  op voortzetting van de recente piekwinsten, maar op een stabiele tot licht groeiende mid-cycle
  kasstroom. Dat maakt de koers redelijk verankerd: er zit geen euforische groeiverwachting in,
  maar ook geen diepe korting. Het verklaart waarom de koers dicht bij zowel de NAV als de
  conservatieve DCF-basis ligt.

### EPV (Bruce Greenwald)
- **Genormaliseerde EBIT-marge %:** ~39 (mid-cycle EBIT ~USD 330 mln op mid-cycle TCE-omzet ~USD 836 mln)
- **Genormaliseerde NOPAT:** USD ~327 mln (EBIT 330 × (1 − 0,01))
- **Maintenance capex:** USD ~200 mln (≈ D&A; vlootvernieuwing/drydock-intensief)
- **Adjusted earnings power:** USD ~327 mln (NOPAT − maintenance capex + D&A, met maint ≈ D&A)
- **EPV per aandeel:** 57,5 (= (327/0,081 − 1.020) / 499,8 = USD 6,03 × 9,54)
- **Groeipremie %:** +29,6 (basis-DCF NOK 74,5 vs EPV NOK 57,5: de markt/DCF prijst een
  groeipremie van ~30% boven de no-growth waarde in)

### Andere methoden
- **DDM uitgevoerd?** false (dividend is sterk variabel/cyclisch i.p.v. progressief — DDM
  ongeschikt; NAV en DCF zijn betere ankers)
- **SOTP uitgevoerd?** false (één geïntegreerde tankeractiviteit, geen conglomeraat)

### Synthese fair value
- **Bandbreedte laag:** 57,5 (EPV / pessimistisch-nabij)
- **Bandbreedte centraal:** 76,0 (DCF-basis ~74,5 en NAV ~77 convergeren)
- **Bandbreedte hoog:** 90,0 (optimistisch-getemperd / hoogste analistendoel)
- **Methode-gewichten:**
  - DCF %: 45
  - EPV %: 25
  - Multiples %: 30 (NAV/P-NAV is voor een asset-heavy reder het belangrijkste relatieve anker)
- **Margin of safety vereist %:** 25 (cyclische sector, NARROW moat, beperkte voorspelbaarheid)
- **Koopniveau:** 57,0 (centraal ~76 × (1 − 0,25))
- **Synthese-toelichting:** De drie methoden convergeren rond NOK 74-78: de conservatieve
  mid-cycle DCF (NOK 74,5), de kansgewogen DCF (NOK 78,4) en de NAV (~NOK 77) wijzen alle op een
  bescheiden upside boven de koers van NOK 70,35. De EPV zonder groei (NOK 57,5) zet de
  ondergrens en laat zien dat de markt een groeipremie van ~30% inprijst — verdedigbaar voor een
  kostenleider maar geen koopje. Voor een asset-heavy reder weeg ik de NAV/multiples zwaar mee
  (30%) naast de DCF (45%) en EPV (25%). Gegeven de cycliciteit en NARROW moat eis ik een margin
  of safety van 25%, wat een koopniveau van circa NOK 56-60 oplevert. Op de huidige koers is dat
  niet bereikt: HOLD.

### Gevoeligheid (DCF)
- **WACC range:** [7,1; 7,6; 8,1; 8,6; 9,1; 9,6]
- **Groei range (terminal):** [0,5; 1,0; 1,5; 2,0; 2,5]
- **Matrix (fair value per aandeel in NOK, basis FCFF USD 310 mln, mid-year):**

| g ↓ / WACC → | 7,1% | 7,6% | 8,1% | 8,6% | 9,1% | 9,6% |
|---|---|---|---|---|---|---|
| 0,5% | 77 | 70 | 64 | 59 | 54 | 50 |
| 1,0% | 84 | 76 | 69 | 63 | 58 | 53 |
| 1,5% | 92 | 82 | 74 | 67 | 62 | 57 |
| 2,0% | 101 | 90 | 81 | 73 | 66 | 61 |
| 2,5% | 113 | 99 | 88 | 79 | 72 | 65 |

*(Indicatieve matrix; de fair value is gevoelig voor zowel WACC als terminal groei. Bij de
genormaliseerde-rente-variant (Rf ~3% → WACC ~7,1%) en g=1,5% komt de FV op ~NOK 92.)*

---

## 13. Databronnen

### Bronnen-hiërarchie
- Jaarverslag/IR/persbericht Hafnia → **HOOG**
- Beursmelding/prospectus → **HOOG**
- StockAnalysis (S&P Global Market Intelligence), MacroTrends → **AGGREGATOR**

### Financiële bronnen (historie)

| Jaar | Bron | URL | Betrouwbaarheid |
|---|---|---|---|
| 2016 | — (entiteit bestond niet in huidige vorm) | — | — |
| 2017 | — | — | — |
| 2018 | — | — | — |
| 2019 | — (merger-jaar; geen vergelijkbare geconsolideerde reeks) | — | — |
| 2020 | — (geen verifieerbare bron) | — | — |
| 2021 | StockAnalysis / S&P Global | https://stockanalysis.com/quote/osl/HAFNI/financials/ | AGGREGATOR |
| 2022 | StockAnalysis / S&P + Hafnia FY2022 (net profit) | https://stockanalysis.com/quote/osl/HAFNI/financials/ | AGGREGATOR |
| 2023 | Hafnia Q4/FY2023 report + S&P | https://s201.q4cdn.com/891122012/files/doc_financials/2023/q4/Quarterly-Earnings-report_Q4-2023_vF2.pdf | HOOG |
| 2024 | Hafnia FY2025 press release (comparatieven) + FY2024 audited | https://www.businesswire.com/news/home/20260225445913/en/Hafnia-Limited-Announces-Financial-Results-For-The-Three-and-Twelve-Months-Ended-31-December-2025 | HOOG |
| 2025 | Hafnia FY2025 press release + Audited Financial Statements 2025 (20-F) | https://www.businesswire.com/news/home/20260225445913/en/Hafnia-Limited-Announces-Financial-Results-For-The-Three-and-Twelve-Months-Ended-31-December-2025 | HOOG |

> **NB bij de harde eis "recente 5 jaren HOOG":** voor Hafnia zijn 2023-2025 op HOOG-niveau
> (eigen rapporten). 2021-2022 leunen op de aggregator (S&P), aangevuld met Hafnia's
> gerapporteerde nettowinst. Reden: Hafnia's eigen IR-archief geeft voor 2021-2022 geen los
> downloadbaar PDF-jaarverslag op dezelfde wijze; de S&P-reeks is de best verifieerbare bron.
> Dit is expliciet gemarkeerd zodat de stage-2 validator de gemengde betrouwbaarheid ziet.

### Jaarverslagen geraadpleegd

| Jaar | Bron | URL |
|---|---|---|
| 2025 | Hafnia Annual Report 2025 / Audited Financial Statements (20-F) | https://www.businesswire.com/news/home/20260416040408/en/HAFNIA-LIMITED-Annual-Report-2025 |
| 2023 | Hafnia Q4 & FY2023 Quarterly Earnings Report | https://s201.q4cdn.com/891122012/files/doc_financials/2023/q4/Quarterly-Earnings-report_Q4-2023_vF2.pdf |

### Beursmeldingen geraadpleegd

| Datum | Omschrijving | URL |
|---|---|---|
| 2026-02-26 | FY2025 resultaten | https://www.businesswire.com/news/home/20260225445913/en/Hafnia-Limited-Announces-Financial-Results-For-The-Three-and-Twelve-Months-Ended-31-December-2025 |
| 2026-05-26 | Q1 2026 resultaten (net profit USD 179,7 mln) | https://www.businesswire.com/news/home/20260526832119/en/Hafnia-Limited-Announces-Financial-Results-for-the-Three-Months-Ended-31-March-2026 |
| 2026-04-13 | Insider sale CEO Mikael Skov (1 mln aandelen) | https://www.businesswire.com/news/home/20260413075885/en/ |
| 2024-07-11 | Shareholder Rights Agreement met BW Group | https://www.businesswire.com/news/home/20240711034578/en/ |

### IPO-prospectus
- **Geraadpleegd?** false (2019 Oslo-prospectus niet rechtstreeks verkregen)
- **URL:** —
- **Pre-IPO data beschikbaar?** false
- **Pre-IPO bron:** — (2019-merger aandelen-gebaseerd; geen pre-IPO dividend-recap geconstateerd)

### Non-GAAP
- **Gebruikt?** true
- **Toelichting:** TCE income en Adjusted EBITDA (sectorstandaard non-IFRS) gebruikt voor
  cyclus-normalisatie; IFRS-nettowinst als primaire grondslag voor de DCF.

### Ontbrekende data (eerlijke lijst)
- Financiële data 2016-2020: niet beschikbaar op vergelijkbare geconsolideerde basis (entiteit
  ontstond eind 2019; 2020 = Covid, geen verifieerbare bron via fetch).
- Omzet-grondslag 2021-2023 gebruikt aggregator-bruto-omzet (incl. pool-doorstroom), niet 1-op-1
  vergelijkbaar met Hafnia's operating revenue 2024-2025 — expliciet gemarkeerd.
- Segment-omzet als percentage: niet publiek uitgesplitst.
- Klantconcentratie, marktaandeel-percentages concurrenten, TAM/SAM: niet betrouwbaar
  kwantificeerbaar — weggelaten i.p.v. geschat.
- IPO-koers 2019: niet geverifieerd — weggelaten.
- De DCF-output is met de hand berekend (de persistente `dcf_calculator.py` kon in deze
  stage-1-sessie niet draaien wegens onbeschikbare sandbox); Claude Code dient in stage 2 de
  cijfers met het script te hercalculeren/valideren.

### Peildatum analyse
- 2026-06-12

---

## 14. Volledige bronnen-lijst

| Titel | URL | Type |
|---|---|---|
| Hafnia (OSL:HAFNI) Overview/Statistics/Financials | https://stockanalysis.com/quote/osl/HAFNI/ | databron |
| Hafnia FY2025 resultaten (Business Wire) | https://www.businesswire.com/news/home/20260225445913/en/Hafnia-Limited-Announces-Financial-Results-For-The-Three-and-Twelve-Months-Ended-31-December-2025 | jaarverslag |
| Hafnia Annual Report 2025 (20-F) | https://www.businesswire.com/news/home/20260416040408/en/HAFNIA-LIMITED-Annual-Report-2025 | jaarverslag |
| Hafnia Q4 & FY2023 Earnings Report (PDF) | https://s201.q4cdn.com/891122012/files/doc_financials/2023/q4/Quarterly-Earnings-report_Q4-2023_vF2.pdf | jaarverslag |
| Hafnia Q1 2026 resultaten (Business Wire) | https://www.businesswire.com/news/home/20260526832119/en/Hafnia-Limited-Announces-Financial-Results-for-the-Three-Months-Ended-31-March-2026 | beursmelding |
| Hafnia cash-flow statement (S&P) | https://stockanalysis.com/quote/osl/HAFNI/financials/cash-flow-statement/ | databron |
| Hafnia balance sheet (S&P) | https://stockanalysis.com/quote/osl/HAFNI/financials/balance-sheet/ | databron |
| Hafnia dividend history | https://stockanalysis.com/quote/osl/HAFNI/dividend/ | databron |
| Shareholder Rights Agreement BW Group | https://www.businesswire.com/news/home/20240711034578/en/ | beursmelding |
| Insider sale CEO Skov (GuruFocus) | https://www.gurufocus.com/news/8793672/insider-sell-mikael-skov-sells-1000000-shares-of-hafnia-ltd-hafn | nieuwsartikel |
| SEB upgrade Buy NOK 91 / DNB Carnegie Hold NOK 73 / Pareto Hold (TheFly) | https://www.tipranks.com/news/the-fly/hafnia-upgraded-to-buy-from-hold-at-seb-equities-thefly-news | analistenrapport |
| US 10-year Treasury yield (jun 2026) | https://tradingeconomics.com/united-states/government-bond-yield | databron |
| Damodaran ERP 2026 | https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html | databron |
| USD/NOK koers | https://www.exchange-rates.org/converter/usd-nok | databron |
| Product Tanker peers (fleet/cost/valuation) | https://nortilus.substack.com/p/product-tankers-equities-fleet-profiles | onderzoeksrapport |
| Hafnia management (CEO/CFO) | https://investor.hafnia.com/governance/executive-management/default.aspx | beurswebsite |
| BW Tankers × Hafnia merger / Oslo listing 2019 | https://www.ship-technology.com/news/bw-tankers-merger-hafnia-tankers/ | nieuwsartikel |

---

## 15. Update-historie

| Datum | Versie | Wijziging |
|---|---|---|
| 2026-06-12 | 1.0 | Eerste publicatie |

---

## Opmerkingen voor Claude Code

Inhoudelijke twijfels en aandachtspunten voor stage 2:

1. **DCF met de hand berekend.** De Linux-sandbox (en daarmee `dcf_calculator.py`) was in deze
   sessie niet beschikbaar (VM startte niet). Alle WACC-, DCF-, EPV-, reverse-DCF- en
   scorekaart-cijfers zijn handmatig afgeleid en intern consistent gemaakt, maar moeten in
   stage 2 met het script worden gehercalculeerd. Let met name op: kansgewogen FV (78,4),
   basis-FV (74,5), EPV/aandeel (57,5), terminal value % (72,4%).

2. **Valuta-mix.** Alle modellering in USD; FV omgerekend naar NOK tegen 9,54. Controleer of de
   pipeline/website de koers (NOK) en de USD-financiële tabellen correct labelt. De
   `meta.valuta` = NOK (koers), maar `fair_value.valuta_kasstromen` = USD.

3. **Omzet-grondslag breuk.** 2021-2023 toont S&P bruto-omzet (incl. externe pool-doorstroom);
   2024-2025 toont Hafnia operating revenue. Deze zijn niet vergelijkbaar. Overweeg of de
   website één consistente reeks moet tonen (bijv. overal operating revenue of overal TCE).
   Ik heb dit expliciet gemarkeerd i.p.v. te harmoniseren met geschatte getallen.

4. **Recente-5-jaar-HOOG-eis deels niet gehaald voor 2021-2022.** Conform METHODE.md zouden de
   5 recentste jaren allemaal HOOG moeten zijn; voor Hafnia leunen 2021-2022 deels op de
   aggregator omdat losse PDF-jaarverslagen voor die jaren niet eenvoudig verkrijgbaar waren.
   Genoteerd als bewuste beperking. Als de validator hierop hard faalt, kan een directe IR-PDF
   van 2021/2022 dit oplossen.

5. **Terminal EV/EBITDA (~11x) hoog.** De Gordon-terminal impliceert een hogere exit-multiple
   dan de sector mid-cycle (~6x). Bewust gecompenseerd met conservatieve basisgroei en zwaar
   pessimistisch scenario. Stage 2 kan overwegen het optimistische scenario lager te wegen.

6. **Reported 5y beta (-0,16) genegeerd**; bottom-up beta 1,05 gebruikt. Documenteer goed,
   want de website toont mogelijk de regressie-beta uit de aggregator — die wijkt sterk af.

7. **Geen platform/-, data/- of scriptwijzigingen gedaan.** Alleen dit bestand geschreven,
   conform de strikte grenzen.
