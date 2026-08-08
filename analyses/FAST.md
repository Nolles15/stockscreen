# Research: FAST — Fastenal Company

> Opgesteld volgens `research/METHODE.md` en `research/TEMPLATE.md`.
> Peildatum 7 augustus 2026 (slotkoers NASDAQ). Alle bedragen in USD mln
> tenzij anders vermeld; per-aandeel-getallen in USD.

---

## Bronnen-inventaris (Stap 0.5)

Fastenal is een Amerikaanse 10-K-indiener. De complete meerjarige reeks is
opgehaald uit de XBRL-feed van de jaarrekeningen zelf
(`data.sec.gov/api/xbrl/companyconcept/CIK0000815556/...`), aangevuld met de
R-pagina's van de FY2025-10-K en de kwartaalpersberichten op de IR-site. Er is
géén aggregator gebruikt voor enig cijfer in de financiële tabellen. Elke reeks
is verankerd aan een extern gecontroleerd punt (zie "Ankercontroles" onderaan
deze sectie).

**Overzichtspagina's die daadwerkelijk zijn geopend:**
- SEC EDGAR-filinglijst 10-K: `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000815556&type=10-K`
- Filing-index FY2025-10-K: `https://www.sec.gov/Archives/edgar/data/815556/000081555626000009/0000815556-26-000009-index.htm`
- `FilingSummary.xml` van diezelfde filing (voor de R-nummering van balans, kasstroom en segmentnoten)

```
Jaar 2025 — HOOG
  Bron: Fastenal Company Form 10-K FY2025 (XBRL + R-pagina's) en het
        persbericht "Fastenal Company Reports 2025 Annual and Fourth
        Quarter Earnings"
  URL:  https://www.sec.gov/Archives/edgar/data/815556/000081555626000009/fast-20251231.htm
        https://www.sec.gov/Archives/edgar/data/815556/000081555626000009/R3.htm   (balans)
        https://www.sec.gov/Archives/edgar/data/815556/000081555626000009/R8.htm   (kasstroom)
        https://s23.q4cdn.com/591718779/files/doc_financials/2025/Q4/EX_99-1-12-31-2025-Earnings-Release-1-19-R8_FINAL.pdf
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: omzet, brutowinst, EBIT, nettowinst, EPS, verwaterde
                       aandelen, CFO, capex, SBC, afschrijving en amortisatie,
                       totale activa, eigen vermogen, kas, bruto schuld,
                       vlottende activa/passiva, voorraad, belastinglast,
                       betaalde belasting, dividend, geografische omzetsplitsing,
                       productmix, eindmarktmix, personeel, vestigingen, FMI
  Cijfers NIET overgenomen: (geen)

Jaar 2024 — HOOG
  Bron: Fastenal Form 10-K FY2024 en FY2025 (vergelijkingskolom)
  URL:  https://www.sec.gov/Archives/edgar/data/815556/000081555625000065/0000815556-25-000065-index.htm
        (XBRL: https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/NetIncomeLoss.json e.a.)
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: volledige W&V, balans en kasstroom
  Cijfers NIET overgenomen: (geen)

Jaar 2023 — HOOG
  Bron: Fastenal Form 10-K FY2023 (XBRL) + vergelijkingskolommen FY2025-10-K
  URL:  https://www.sec.gov/Archives/edgar/data/815556/000081555623000009/0000815556-23-000009-index.htm
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: volledige W&V, balans en kasstroom
  Cijfers NIET overgenomen: (geen)

Jaar 2022 — HOOG
  Bron: Fastenal Form 10-K FY2022 (XBRL)
  URL:  https://www.sec.gov/Archives/edgar/data/815556/000081555622000009/0000815556-22-000009-index.htm
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: volledige W&V, balans en kasstroom
  Cijfers NIET overgenomen: (geen)

Jaar 2021 — HOOG
  Bron: Fastenal Form 10-K FY2021 (XBRL companyconcept, fy=2021, form=10-K)
  URL:  https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/NetCashProvidedByUsedInOperatingActivities.json
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: volledige W&V, balans en kasstroom
  Cijfers NIET overgenomen: (geen)

Jaren 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013 — HOOG
  Bron: de betreffende Form 10-K's van Fastenal, via de XBRL-companyconcept-
        feed (elk cijfer draagt form=10-K en het bijbehorende fy-label).
  URL:  https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/
        {SalesRevenueNet | RevenueFromContractWithCustomerExcludingAssessedTax |
         GrossProfit | OperatingIncomeLoss | NetIncomeLoss |
         NetCashProvidedByUsedInOperatingActivities |
         PaymentsToAcquirePropertyPlantAndEquipment | ShareBasedCompensation |
         Depreciation | Assets | StockholdersEquity |
         CashAndCashEquivalentsAtCarryingValue | LongTermDebt | AssetsCurrent |
         LiabilitiesCurrent | InventoryNet | IncomeTaxExpenseBenefit |
         IncomeTaxesPaidNet | WeightedAverageNumberOfDilutedSharesOutstanding |
         CommonStockDividendsPerShareCashPaid | PaymentsOfDividends |
         PaymentsForRepurchaseOfCommonStock}.json
  Daadwerkelijk geopend: ja (elk van deze endpoints afzonderlijk)
  Cijfers overgenomen: omzet, brutowinst, EBIT, nettowinst, CFO, capex, SBC,
                       afschrijving, totale activa, eigen vermogen, kas,
                       schuld, vlottende activa/passiva, voorraad,
                       belastinglast, betaalde belasting, aandelen, dividend
  Cijfers NIET overgenomen: amortisatie immateriële activa vóór 2023 (staat
                       niet als aparte XBRL-post in de oudere filings; EBITDA
                       vóór 2023 is daarom EBIT + afschrijving, zonder de
                       ~USD 10,7 mln amortisatie — het effect op de
                       EBITDA-marge is minder dan 0,2 procentpunt)

TTM (Q3 2025 t/m Q2 2026) — HOOG
  Bron: FY2025-jaarcijfers minus H1-2025 plus H1-2026, uit de officiële
        kwartaalpersberichten
  URL:  https://investor.fastenal.com/news-releases/news-details/2025/Fastenal-Company-Reports-2025-Second-Quarter-Earnings/default.aspx
        https://investor.fastenal.com/news-releases/news-details/2026/Fastenal-Company-Reports-2026-Second-Quarter-Earnings/default.aspx
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: omzet, EBIT, nettowinst, EPS, kas, bruto schuld,
                       uitstaande aandelen, brutomarge, FMI/digitale KPI's

Macro- en waarderingsinvoeren — HOOG
  Risicovrije rente: FRED DGS10, 4,69% per 2026-08-06
    https://fred.stlouisfed.org/series/DGS10
  Equity risk premium: Damodaran implied ERP 4,28% per 2026-08-01
    https://pages.stern.nyu.edu/~adamodar/New_Home_Page/home.htm
  Sector-WACC ter referentie: Damodaran "Retail (Distributors)" 7,22% (jan-2026)
    https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/wacc.html

Marktdata en peers — AGGREGATOR (uitsluitend voor niet-jaarrekeningcijfers)
  Bron: StockAnalysis.com — slotkoers, beta, analistenconsensus, peer-multiples
  URL:  https://stockanalysis.com/stocks/fast/  |  /statistics/  |  /forecast/
        https://stockanalysis.com/stocks/gww/statistics/
        https://stockanalysis.com/stocks/msm/statistics/
        https://stockanalysis.com/stocks/ait/statistics/
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: slotkoers 51,84 (7-8-2026), 5-jaars beta 0,71,
                       consensus (18 analisten, koersdoel 47,84),
                       peer-multiples GWW/MSM/AIT
  Cijfers NIET overgenomen: geen enkel cijfer in de financiële tabellen van
                       hoofdstuk 3 — die komen allemaal uit de 10-K's.
  Let op: StockAnalysis rapporteert "total debt" van USD 441,5 mln; dat is
  inclusief operationele leaseverplichtingen. Fastenal rapporteert zelf
  USD 120,0 mln rentedragende schuld per 30-6-2026. Ik gebruik consequent de
  door het bedrijf gerapporteerde rentedragende schuld (zie IFRS-16/ASC-842-nota
  in sectie 3).

Historische koers/winst-verhouding — AGGREGATOR
  Bron: MacroTrends P/E-historie (jaareinden 2012-2025)
  URL:  https://www.macrotrends.net/stocks/charts/FAST/fastenal/pe-ratio
  Daadwerkelijk geopend: ja
  Gebruikt voor: uitsluitend het 10-jaars gemiddelde P/E (27,01) in de
  relatieve waardering. De onderliggende EPS-reeks van deze bron is
  gekruiscontroleerd tegen mijn eigen split-gecorrigeerde reeks uit de 10-K's
  en komt tot op één cent overeen voor elk van de dertien jaren.
```

**Ankercontroles (verplicht bij XBRL-gebruik, zie geheugennotitie `fa-sec-xbrl-bron`):**

| Controle | Uit mijn reeks | Extern anker | Verschil |
|---|---|---|---|
| Verwaterde EPS FY2025 | 1.258,4 / 1.150,3 = 1,094 | 1,09 (persbericht FY2025) | 0,4% (afronding) |
| EBIT-marge FY2025 | 1.655,7 / 8.200,5 = 20,19% | 20,2% (persbericht FY2025) | — |
| Kas per 31-12-2025 | 276,8 | 276,8 (persbericht FY2025) | 0 |
| Omzet TTM | 8.749,4 | 8,75 mrd (StockAnalysis) | — |
| EPS TTM | 1,17 | 1,17 (StockAnalysis) | 0 |
| Dividend betaald 2025 / aandelen | 1.004,2 / 1.150,3 = 0,873 | DPS 0,875 (XBRL) | 0,2% |
| Split-correctie | mijn EPS 2013-2025 | MacroTrends EPS-reeks | identiek |

**Split-behandeling.** Fastenal splitste 2-voor-1 in mei 2019 en opnieuw
2-voor-1 in mei 2025. De XBRL-feed bewijst dit: het FY2019-10-K verdubbelt de
aandelenreeks van 2017-2018 en het FY2025-10-K verdubbelt die van 2023-2024.
Alle per-aandeel-getallen in dit rapport (EPS, DPS, boekwaarde, aandelen) zijn
herrekend naar de huidige, post-mei-2025 basis: reeksen vóór 2019 zijn met vier
vermenigvuldigd, 2019 t/m 2024 met twee. De absolute bedragen (omzet, winst,
kasstroom) zijn niet aangeraakt.

**Haallijst: leeg.** Er is geen enkele bron die ik niet kon openen. De volledige
tienjaarshistorie staat op HOOG. Aan Janco hoeft niets gevraagd te worden.

---

## Metadata
- **Ticker (bare):** FAST
- **Yahoo symbol:** FAST
- **Exchange:** NASDAQ (Nasdaq Global Select Market)
- **Sector (GICS-achtig):** Industrie
- **Industrie:** Handel en distributie van industriële producten (fasteners en MRO)
- **Land:** Verenigde Staten
- **Peildatum analyse:** 2026-08-07
- **Koers op peildatum:** 51,84
- **Valuta:** USD
- **Marktkapitalisatie:** USD 59,5 mld
- **Marktkap in mln (lokale valuta):** 59.486
- **Free float pct:** ca. 99 (insiders houden minder dan 1%)
- **Indexlidmaatschap:** S&P 500, Nasdaq-100
- **Domein:** fastenal.com

---

## 1. Executive summary

- **Kernthese:**

Fastenal verkoopt de kleinste onderdelen van de industrie: bouten, moeren en
handschoenen die een fabriek draaiende houden. Het bedrijf uit Winona is met 8,2
miljard dollar omzet de op één na grootste industriële distributeur van
Noord-Amerika. De kern van het model is nabijheid: de voorraad
staat bij de klant zelf, in een filiaal om de hoek of in een automaat die zelf
bijbestelt. Die 140.789 apparaten leveren bijna de helft van de omzet, en
driekwart loopt onder contract. De groei komt uit consolidatie van inkoop bij
grote klanten. Het rendement op geïnvesteerd kapitaal ligt al tien jaar boven 23
procent, bij een bedrijfsmarge die nooit buiten 19,8 tot 20,8 procent kwam. Het
grootste risico zit in de prijs.

- **Oordeel:** **PASS**
- **Fair value basis** (basisscenario, USD): 21,98
- **Fair value kansgewogen**: 22,04
- **EPV per aandeel** (Earnings Power Value, zonder groeipremie): 14,51
- **Upside pct**: −57,6
- **Fair value scenarios:**

| Scenario | Fair value | Upside % | FCF groei % | WACC % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 14,19 | −72,63 | 4,0 (fase 1) / 2,0 (fase 2) | 8,88 | 25 |
| Basis | 21,98 | −57,60 | 8,0 (fase 1) / 5,0 (fase 2) | 8,13 | 50 |
| Optimistisch | 30,03 | −42,08 | 11,0 (fase 1) / 7,0 (fase 2) | 7,88 | 25 |

- **Reverse-DCF impliciete groei pct**: 21,63 (vrije kasstroom, jaar 1-5; daarna 13,52% in jaar 6-10)
- **Grootste kans**: verdere consolidatie van industriële inkoop bij grote klanten, waarbij Fastenal met FMI-apparatuur op de werkvloer structureel marktaandeel wint van kleine regionale distributeurs.
- **Grootste risico**: de waardering — bij 44 keer de winst is elke tegenvaller in groei of marge onmiddellijk een koersdaling van tientallen procenten, zonder dat er iets mis is met het bedrijf.

---

## 2. Bedrijfsprofiel

- **Beschrijving:**

Fastenal is een groothandel in industriële en bouwbenodigdheden. De naam komt
van "fasteners" — bevestigingsmiddelen — en dat is nog steeds de grootste
productgroep met 30,5% van de omzet in 2025. Daarnaast verkoopt het bedrijf
acht andere hoofdgroepen: veiligheidsartikelen (22,2%), schoonmaakmiddelen
(9,0%), handgereedschap (8,3%), hydrauliek en pneumatiek (6,9%),
materiaalhandling (5,7%), snijgereedschap (5,2%), elektra (4,7%) en
lasbenodigdheden (4,3%). De klant is vrijwel altijd een bedrijf, geen
consument: 75,9% van de omzet komt uit de maakindustrie, 8,1% uit
niet-residentiële bouw en 16,0% uit een restcategorie van vervoerders,
overheden en handelaren. Fastenal zit in de waardeketen tussen honderden
fabrikanten en tienduizenden industriële eindgebruikers, en lost daarmee een
probleem op dat weinig aandacht krijgt maar veel geld kost: een fabriek heeft
duizenden verschillende artikelen nodig in kleine hoeveelheden, en het
administratief afhandelen van een bestelling kost al snel meer dan de bout
zelf. Fastenal neemt dat werk over door de voorraad fysiek bij de klant neer
te zetten en het bijvullen te automatiseren. De omzet ontstaat op drie
manieren: gewone verkoop via een filiaal, verkoop via een "Onsite" — een
Fastenal-magazijn met eigen personeel binnen de fabrieksmuren van de klant —
en verkoop via FMI-apparatuur (FASTStock, FASTBin, FASTVend) die bij de klant
staat opgesteld. Ongeveer 74% van de omzet valt onder een contractuele
afspraak, waarvan nationale raamcontracten met 65% van de totale omzet het
grootste deel vormen.

- **Geschiedenis:**

Fastenal werd in 1967 opgericht in Winona, Minnesota door Bob Kierlin samen met
zijn stadsgenoten Jack Remick, Van McConnon en Steve Slaggie. Het
oorspronkelijke idee was radicaal voor die tijd: bouten en moeren verkopen uit
automaten. Dat werkte technisch nog niet, en de vier vielen terug op een gewone
winkel — een beslissing die het bedrijfsmodel voor de volgende vijftig jaar
bepaalde en waarvan de ironie is dat de automaat uiteindelijk tóch de kern werd,
alleen vier decennia later en elektronisch. Kierlin leidde het bedrijf van 1968
tot 2002 als CEO en bleef tot zijn vertrek uit de raad in 2014 voorzitter. De
groeiformule was simpel en werd decennialang herhaald: open een klein filiaal in
een stad waar de industrie zit, geef de filiaalleider ruime bevoegdheid en laat
hem de omgeving bewerken. In 1987 ging het bedrijf naar de beurs in Nasdaq, wat
het kapitaal opleverde om die formule te versnellen. De internationale
uitbreiding kwam laat en voorzichtig: Canada in 1994, Mexico in 1999, Singapore
in 2001 als eerste vestiging buiten Noord-Amerika. In 2008 werd Fastenal
opgenomen in de S&P 500; in 2009, midden in de financiële crisis, kocht het
delen van fabrikant Holo-Krome — een van de zeer weinige overnames in de
geschiedenis van het bedrijf. Het keerpunt in het verdienmodel kwam rond 2011,
toen Fastenal industriële automaten (FASTVend) begon uit te rollen en later
FASTBin toevoegde: bakken met gewichts- of infraroodsensoren die zelf
signaleren dat ze bijgevuld moeten worden. Vanaf 2015 verschoof de nadruk van
"meer filialen" naar "meer aanwezigheid bij de klant": het aantal filialen daalde
van ruim 2.600 naar 1.595, terwijl het aantal Onsite-locaties en FMI-apparaten
sterk toenam. Dan Florness volgde Kierlins opvolger op als CEO in januari 2016
en bracht het bedrijf van 3,96 naar 8,20 miljard dollar omzet. Op 16 juli 2026
droeg hij het stokje over aan Jeff Watts, die net als Florness in 1996 bij
Fastenal begon.

- **Bedrijfsmodel:** Fastenal verdient aan het verschil tussen inkoop- en
verkoopprijs van industriële artikelen, met een brutomarge die in 2025 op 45,0%
lag. Het onderscheidende zit niet in het product maar in de logistiek: door
voorraad fysiek bij de klant te plaatsen — in een filiaal om de hoek, een Onsite
binnen de fabrieksmuren, of in FMI-apparatuur op de werkvloer — verlaagt
Fastenal de totale inkoopkosten van de klant, niet alleen de stukprijs. Die
apparatuur maakt de omzet bovendien terugkerend: wie een FASTVend-automaat in
zijn productiehal heeft staan, bestelt daar automatisch bij. 61,6% van de omzet
liep in het tweede kwartaal van 2026 via deze "digital footprint", en 75,8% van
de omzet valt onder contract. Ongeveer 11% van de omzet bestaat uit eigen merken
(Body Guard, ORMADUS), waarop de marge hoger ligt.

- **IPO-context:** Fastenal ging in 1987 naar de Nasdaq, bijna veertig jaar
geleden. Sindsdien is er nooit een kapitaalverhoging van betekenis geweest: het
aantal verwaterde aandelen daalde tussen 2013 en 2025 zelfs licht, van 1.190,7
naar 1.150,3 miljoen op de huidige split-basis. Het bedrijf financiert zichzelf
volledig uit eigen kasstroom en een kleine kredietfaciliteit. Omdat de
beursgang meer dan tien jaar geleden plaatsvond, is er geen IPO-correctie van
toepassing (zie sectie 7 en 8).

- **Klantprofiel:** B2B, sterk gefragmenteerd. In 2025 was er geen enkele klant
goed voor 5% of meer van de omzet. Fastenal bediende in 2025 gemiddeld 98.361
actieve klantlocaties per maand en 250.845 unieke locaties over het jaar. De
retentie is hoog omdat de klantrelatie fysiek verankerd is in geïnstalleerde
apparatuur en, bij Onsites, in Fastenal-personeel binnen de fabriek van de klant.
Het zwaartepunt schuift naar grotere klanten: het aantal locaties dat meer dan
USD 50.000 per maand besteedt groeide in Q2 2026 met 16,5% naar 3.125, terwijl
het totale aantal actieve locaties daalde van 101.440 naar 93.283 — kleinere
klanten vallen af, grotere groeien harder.
- **Oprichtingsjaar**: 1967
- **IPO-datum**: 1987 (exacte dag niet verifieerbaar in de geraadpleegde bronnen — weggelaten)
- **IPO-koers** (lokale valuta): niet verifieerbaar — weggelaten
- **Personeel** (FTE): 22.230 per 30-06-2026; 24.489 medewerkers absoluut per 31-12-2025
- **Landen actief**: 25
- **Klantconcentratie**: In 2025 was geen enkele klant goed voor 5% of meer van
de geconsolideerde omzet — een zeldzaam lage concentratie voor een B2B-bedrijf
van deze omvang. Nationale raamcontracten vormen samen wel 65% van de omzet, dus
de concentratie zit niet bij één klant maar bij één klantsoort: grote,
professioneel inkopende ondernemingen met meerdere vestigingen. Dat verlaagt het
risico dat één opzegging pijn doet, maar verhoogt de collectieve
onderhandelingsmacht van de afnemers en verklaart de aanhoudende druk op de
brutomarge.

### Geografische spreiding (omzet)
| Regio | Omzet % | Valuta-exposure |
|---|---|---|
| Verenigde Staten | 83,2 | USD |
| Canada en Mexico | 13,5 | CAD / MXN |
| Overige landen | 3,3 | diverse |

**Toelichting geografie:** Fastenal is in essentie een Amerikaans bedrijf: 83,2%
van de omzet kwam in 2025 uit de Verenigde Staten, een aandeel dat sinds 2023
constant is. Het valutarisico is beperkt en natuurlijk afgedekt, omdat Fastenal
in Canada en Mexico ook inkoopt en distributiecentra exploiteert in dezelfde
valuta als waarin het verkoopt. Het effect loopt via de omrekening: de post
koersverschillen bedroeg plus USD 11,0 mln (2025) en min 10,6 mln (2024).

### Segmenten
| Naam | Omzet % | Beschrijving |
|---|---|---|
| Fasteners | 30,5 | Bouten, moeren, schroeven, draadstang en aanverwant bevestigingsmateriaal — het historische hart van het bedrijf en de productgroep met de sterkste concurrentiepositie door schaal en eigen productie. |
| Veiligheidsartikelen | 22,2 | Handschoenen, brillen, gehoorbescherming, valbeveiliging. Snelst groeiende groep, sterk verbruiksgoed en daardoor bij uitstek geschikt voor verkoop via automaten. |
| Schoonmaakartikelen | 9,0 | Reinigingsmiddelen en -materialen voor productieomgevingen; typisch herhaalaankopen met lage stukprijs. |
| Gereedschap | 8,3 | Hand- en elektrisch gereedschap; hogere stukprijs, lagere frequentie. |
| Hydrauliek en pneumatiek | 6,9 | Slangen, koppelingen, cilinders en fittingen voor onderhoud aan machines. |
| Materiaalhandling | 5,7 | Opslag-, transport- en hijsmiddelen binnen de fabriek. |
| Snijgereedschap | 5,2 | Boren, frezen en slijpschijven; slijtdelen met hoge herhaalfrequentie. |
| Elektra | 4,7 | Kabel, schakelmateriaal en verlichting voor industriële toepassing. |
| Lasbenodigdheden | 4,3 | Draad, elektroden, gassen en toebehoren. |
| Overig | 3,2 | Restcategorie. |

### Aandeelhouders (top 5)
| Naam | Belang % | Type (oprichter / PE / institutioneel / retail) |
|---|---|---|
| The Vanguard Group, Inc. | 13,0 | institutioneel |
| BlackRock, Inc. | 9,1 | institutioneel |
| State Street Global Advisors, Inc. | 4,7 | institutioneel |
| Bank of New York Mellon Corp. | 2,4 | institutioneel |
| Norges Bank | 1,4 | institutioneel |

- **Institutioneel eigendomstrend:** stabiel en hoog. Per februari 2026 zat 86%
bij instellingen en minder dan 1% bij insiders; de top negentien houders bezit
samen de helft. Er is geen controlerende partij en geen meervoudig stemrecht. De
belangen van Vanguard, BlackRock en State Street zijn grotendeels indexbezit en
bewegen daardoor nauwelijks.

*Bank of New York Mellon (27,66 mln aandelen) en Norges Bank (16,10 mln) zijn
omgerekend op 1.147,5 mln uitstaande aandelen.*

---

## 3. Financieel — historische data (10 jaar + TTM)

Alle jaren 2013 t/m 2025 komen rechtstreeks uit de Form 10-K's (betrouwbaarheid
HOOG). Er zijn geen lege cellen: de volledige dertienjaarshistorie is
beschikbaar. Per-aandeel-getallen zijn herrekend naar de huidige split-basis.

### Resultatenrekening (bedragen in USD mln)

| Jaar | Omzet | Omzetgroei % | Brutowinst | Brutomarge % | EBIT | EBIT-marge % | EBITDA | EBITDA-marge % | Nettowinst | Nettomarge % | EPS | EPS-groei % | Aandelen mln |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2013 | 3.326,1 | — | 1.719,4 | 51,70 | 712,7 | 21,43 | 776,4 | 23,34 | 448,6 | 13,49 | 0,38 | — | 1.190,7 |
| 2014 | 3.733,5 | 12,25 | 1.897,4 | 50,82 | 787,6 | 21,10 | 859,7 | 23,03 | 494,1 | 13,24 | 0,42 | 10,28 | 1.189,3 |
| 2015 | 3.869,2 | 3,63 | 1.948,9 | 50,37 | 828,8 | 21,42 | 914,8 | 23,64 | 516,4 | 13,35 | 0,44 | 6,38 | 1.168,2 |
| 2016 | 3.962,0 | 2,40 | 1.964,8 | 49,59 | 795,8 | 20,09 | 899,4 | 22,70 | 499,5 | 12,61 | 0,43 | −2,30 | 1.156,6 |
| 2017 | 4.390,5 | 10,81 | 2.163,6 | 49,28 | 881,8 | 20,08 | 1.005,4 | 22,90 | 578,6 | 13,18 | 0,50 | 16,17 | 1.153,4 |
| 2018 | 4.965,1 | 13,09 | 2.398,9 | 48,32 | 999,2 | 20,12 | 1.133,3 | 22,83 | 751,9 | 15,14 | 0,65 | 30,49 | 1.148,7 |
| 2019 | 5.333,7 | 7,42 | 2.515,4 | 47,16 | 1.057,2 | 19,82 | 1.201,8 | 22,53 | 790,9 | 14,83 | 0,69 | 5,17 | 1.148,9 |
| 2020 | 5.647,3 | 5,88 | 2.567,8 | 45,47 | 1.141,8 | 20,22 | 1.295,1 | 22,93 | 859,1 | 15,21 | 0,75 | 8,39 | 1.151,3 |
| 2021 | 6.010,9 | 6,44 | 2.777,2 | 46,20 | 1.217,4 | 20,25 | 1.377,3 | 22,91 | 925,0 | 15,39 | 0,80 | 7,40 | 1.154,2 |
| 2022 | 6.980,6 | 16,13 | 3.215,8 | 46,07 | 1.453,6 | 20,82 | 1.630,2 | 23,35 | 1.086,9 | 15,57 | 0,94 | 17,81 | 1.151,2 |
| 2023 | 7.346,7 | 5,24 | 3.354,5 | 45,66 | 1.528,7 | 20,81 | 1.706,0 | 23,22 | 1.155,0 | 15,72 | 1,01 | 6,75 | 1.146,0 |
| 2024 | 7.546,0 | 2,71 | 3.401,9 | 45,08 | 1.510,0 | 20,01 | 1.685,4 | 22,34 | 1.150,6 | 15,25 | 1,00 | −0,60 | 1.148,6 |
| 2025 | 8.200,5 | 8,67 | 3.691,2 | 45,01 | 1.655,7 | 20,19 | 1.834,9 | 22,38 | 1.258,4 | 15,35 | 1,09 | 9,20 | 1.150,3 |
| TTM | 8.749,4 | — | — | — | 1.775,1 | 20,29 | 1.954,3 | 22,34 | 1.352,1 | 15,45 | 1,17 | — | 1.147,5 |

- **Toelichting resultaten:** Twee dingen springen eruit. De brutomarge daalde in
twaalf jaar gestaag van 51,7% naar 45,0%, omdat Fastenal steeds meer zaken doet
met grote klanten die scherper inkopen. Tegelijk bewoog de bedrijfsmarge
nauwelijks: sinds 2016 elk jaar tussen 19,8% en 20,8%. De margedruk aan de
inkoopkant is dus volledig gecompenseerd met kostenbeheersing — het
personeelsbestand groeide veel langzamer dan de omzet. Dat is de belangrijkste
kwaliteitsindicator in deze tabel: een distributeur die tien jaar lang precies
één vijfde van elke omzetdollar overhoudt terwijl zijn markt consolideert, heeft
prijszettingsvermogen. De omzetgroei is onregelmatig maar structureel positief:
2016 en 2024 waren zwakke industriejaren, 2022 en 2026 sterke.
- **Omzet-CAGR** (2015-2025): 7,80%. Nettowinst-CAGR over dezelfde periode: 9,32%.

### Kasstromen (USD mln)

| Jaar | CFO | Capex | FCF | FCF na SBC | FCF/aandeel | FCF-marge % | FCF-groei % | FCF-conversie % | SBC | Dividend totaal | Aandeleninkoop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2013 | 416,1 | 206,5 | 209,6 | 204,2 | 0,17 | 6,30 | — | 46,7 | 5,4 | 237,5 | 9,1 |
| 2014 | 499,4 | 189,5 | 309,9 | 302,9 | 0,25 | 8,30 | 47,88 | 62,7 | 7,0 | 296,6 | 52,9 |
| 2015 | 546,9 | 155,2 | 391,8 | 385,9 | 0,33 | 10,13 | 26,41 | 75,9 | 5,8 | 327,1 | 293,0 |
| 2016 | 514,0 | 189,5 | 324,5 | 320,4 | 0,28 | 8,19 | −17,16 | 65,0 | 4,1 | 346,6 | 59,4 |
| 2017 | 585,2 | 119,9 | 465,3 | 460,1 | 0,40 | 10,60 | 43,37 | 80,4 | 5,2 | 369,1 | 82,6 |
| 2018 | 674,2 | 176,3 | 497,9 | 492,8 | 0,43 | 10,03 | 7,01 | 66,2 | 5,1 | 441,9 | 103,0 |
| 2019 | 842,7 | 246,4 | 596,3 | 590,6 | 0,51 | 11,18 | 19,76 | 75,4 | 5,7 | 498,6 | 0,0 |
| 2020 | 1.101,8 | 168,1 | 933,7 | 928,0 | 0,81 | 16,53 | 56,58 | 108,7 | 5,7 | 803,4 | 52,0 |
| 2021 | 770,1 | 156,6 | 613,5 | 607,9 | 0,53 | 10,21 | −34,29 | 66,3 | 5,6 | 643,7 | 0,0 |
| 2022 | 941,0 | 173,8 | 767,2 | 760,0 | 0,66 | 10,99 | 25,05 | 70,6 | 7,2 | 711,3 | 237,8 |
| 2023 | 1.432,7 | 172,8 | 1.259,9 | 1.252,6 | 1,09 | 17,15 | 64,22 | 109,1 | 7,3 | 1.016,8 | 0,0 |
| 2024 | 1.173,3 | 226,5 | 946,8 | 938,8 | 0,82 | 12,55 | −24,85 | 82,3 | 8,0 | 893,3 | 0,0 |
| 2025 | 1.295,9 | 245,3 | 1.050,6 | 1.042,2 | 0,91 | 12,81 | 10,96 | 83,5 | 8,4 | 1.004,2 | 0,0 |

- **Toelichting kasstromen:** De vrije kasstroom beweegt grilliger dan de winst,
en de oorzaak is vrijwel altijd werkkapitaal. In 2023 kwam er USD 189,1 mln uit
de voorraad vrij nadat de leveringsketens normaliseerden: de FCF-conversie sprong
naar 109% en de FCF met 64% omhoog. In 2024 en 2025 ging dat geld er weer in —
voorraad plus 133,9 en 89,2 mln, debiteuren plus 31,9 en 130,1 mln — waardoor de
FCF in 2024 met 24,9% daalde terwijl de winst vlak bleef. Die daling is dus een
groei-investering, geen verslechtering. Beide bewegingen overschrijden de
15%-grens uit de methodiek en zijn hiermee verklaard. De aandelencompensatie is
met USD 8,4 mln laag, zodat FCF en FCF-na-SBC nauwelijks verschillen.

### Balans-ratio's

| Jaar | Nettoschuld | Nettoschuld/EBITDA | Eigen vermogen | ROE % | ROIC % | ROCE % | Current ratio | Solvabiliteit % | Goodwill % van activa | Working capital |
|---|---|---|---|---|---|---|---|---|---|---|
| 2013 | −58,5 | −0,08 | 1.772,7 | 25,31 | 26,14 | 40,2 | 5,87 | 85,40 | 0,0 | 1.168,6 |
| 2014 | −24,5 | −0,03 | 1.915,2 | 26,80 | 27,42 | 39,3 | 4,22 | 81,18 | 0,0 | 1.207,9 |
| 2015 | 236,0 | 0,26 | 1.801,3 | 27,79 | 26,38 | 38,3 | 4,46 | 71,13 | 0,0 | 1.291,6 |
| 2016 | 277,3 | 0,31 | 1.933,1 | 26,75 | 23,70 | 33,4 | 6,24 | 72,43 | 0,0 | 1.445,1 |
| 2017 | 298,1 | 0,30 | 2.096,9 | 28,71 | 25,38 | 34,9 | 5,51 | 72,05 | 0,0 | 1.584,7 |
| 2018 | 332,8 | 0,29 | 2.302,7 | 34,18 | 30,26 | 38,4 | 5,30 | 69,33 | 0,0 | 1.878,8 |
| 2019 | 170,1 | 0,14 | 2.665,6 | 31,84 | 29,29 | 35,9 | 4,51 | 70,15 | 0,0 | 1.912,5 |
| 2020 | 159,3 | 0,12 | 2.733,2 | 31,83 | 30,24 | 37,3 | 4,08 | 68,94 | 0,0 | 1.886,9 |
| 2021 | 153,8 | 0,11 | 3.042,2 | 32,03 | 30,63 | 36,3 | 4,19 | 70,77 | 0,0 | 2.174,4 |
| 2022 | 324,9 | 0,20 | 3.163,2 | 35,03 | 32,83 | 39,7 | 3,96 | 69,54 | 0,0 | 2.335,0 |
| 2023 | 38,7 | 0,02 | 3.348,8 | 35,47 | 33,74 | 42,8 | 4,57 | 75,04 | 0,0 | 2.359,6 |
| 2024 | −55,8 | −0,03 | 3.616,3 | 33,04 | 33,16 | 39,9 | 4,67 | 76,98 | 0,0 | 2.524,8 |
| 2025 | −151,8 | −0,08 | 3.943,6 | 33,29 | 34,25 | 40,7 | 4,85 | 78,05 | 0,0 | 2.756,5 |

*ROCE = EBIT / (eigen vermogen + bruto schuld − kas), jaareindestanden.
Quick ratio 2025: 2,41 (2024: 2,28; 2023: 2,27).*

- **Toelichting balans:** Dit is een balans zonder verhaal, en dat ís het
verhaal. Fastenal deed nooit een overname van betekenis en draagt daardoor geen
goodwill: nul procent van de activa. De solvabiliteit ligt dertien jaar boven de
68%, de current ratio tussen 4 en 6. De rentedragende schuld — een doorlopende
kredietfaciliteit — werd afgebouwd van 555 (2022) naar 125 mln (2025) en 120 mln
per 30 juni 2026. Let op het onderscheid tussen bruto en netto: in 2015-2022
stond er bruto schuld op de balans terwijl de nettoschuld nooit boven 0,31×
EBITDA kwam, en sinds 2024 is er weer netto kas. Herfinancieringsrisico bestaat
feitelijk niet: de rentedekking bedraagt 251×.

### Kapitaalstructuur huidig (per 30-06-2026)
- **Nettoschuld (huidig)**: −84,7 (nettokaspositie)
- **Bruto schuld**: 120,0
- **Cash & equivalents**: 204,7
- **Lease-verplichtingen (ASC 842, per 31-12-2025)**: 316,9 (106,1 kortlopend + 210,8 langlopend)
- **Gemiddelde rente %**: 4,06 (rente betaald 2025 van 6,6 gedeeld door de gemiddelde uitstaande schuld van 162,5 over 2024-2025)
- **Rente-dekking (EBIT/rente)**: 251×

**Behandeling van leases (ASC 842, de Amerikaanse tegenhanger van IFRS 16).**
Fastenal huurt zijn filialen en verwerkt de huur als operationele kosten in de
bedrijfslasten; de gerapporteerde EBIT van 1.655,7 mln is dus al ná huur. In lijn
met de vaste werkwijze behandel ik leases daarom consequent als operationele
kosten en tel ik de leaseverplichting van USD 316,9 mln **niet** op bij de
nettoschuld. Wie dat wél zou doen, telt de huur twee keer mee: één keer in de
lagere EBIT en één keer in een hogere schuld. De aggregator StockAnalysis doet
dat laatste (441,5 mln "total debt") — dat cijfer wordt in dit rapport niet
gebruikt.

### Earnings quality

| Jaar | Accruals ratio % | FCF-conversie % | SBC als % van FCF | SBC als % van marktkap |
|---|---|---|---|---|
| 2021 | 3,75 | 66,3 | 0,91 | — |
| 2022 | 3,30 | 70,6 | 0,94 | — |
| 2023 | −6,16 | 109,1 | 0,58 | — |
| 2024 | −0,50 | 82,3 | 0,84 | — |
| 2025 | −0,77 | 83,5 | 0,80 | 0,014 |

*Accruals ratio = (nettowinst − operationele kasstroom) / gemiddeld totaal
activa. Negatief is conservatief.*

- **Toelichting earnings quality:** De accruals ratio schommelt rond nul en is in
drie van de vijf jaar negatief — de kasstroom komt dus gemiddeld eerder binnen
dan de winst wordt geboekt, wat conservatief is. De uitschieters zijn allebei
werkkapitaal: 2023 (voorraadafbouw) en 2021-2022 (voorraadopbouw tijdens de
leveringsketencrisis). Er is geen trend van oplopende accruals, het klassieke
waarschuwingssignaal. Fastenal rapporteert géén adjusted of underlying winst:
geen non-GAAP-brug, geen jaarlijks terugkerende herstructureringspost, geen
uitgesloten acquisitie-afschrijving. Dat maakt de winstkwaliteit ongewoon
eenvoudig te beoordelen.

### Rendementsindicatoren

| Jaar | ROE % | ROIC % | ROA % | Asset turnover |
|---|---|---|---|---|
| 2016 | 26,75 | 23,70 | 19,21 | 1,52 |
| 2017 | 28,71 | 25,38 | 20,74 | 1,57 |
| 2018 | 34,18 | 30,26 | 24,13 | 1,59 |
| 2019 | 31,84 | 29,29 | 22,21 | 1,50 |
| 2020 | 31,83 | 30,24 | 22,13 | 1,45 |
| 2021 | 32,03 | 30,63 | 22,39 | 1,45 |
| 2022 | 35,03 | 32,83 | 24,57 | 1,58 |
| 2023 | 35,47 | 33,74 | 25,63 | 1,63 |
| 2024 | 33,04 | 33,16 | 25,12 | 1,65 |
| 2025 | 33,29 | 34,25 | 25,81 | 1,68 |

- **Toelichting rendement:** Dit is de tabel die het meest over de kwaliteit van
Fastenal zegt. Het rendement op geïnvesteerd kapitaal steeg over tien jaar van
23,7% naar 34,3% — kapitaal wordt dus elk jaar iets beter ingezet. De motor staat
in de laatste kolom: de omloopsnelheid van de activa liep op van 1,45 naar 1,68,
dus Fastenal haalt steeds meer omzet uit dezelfde balans. Dat komt door de
verschuiving van eigen filialen naar apparatuur bij de klant. Het rendement op
eigen vermogen van 33% wordt bovendien niet met schuld opgeklopt.

### ROIC vs. WACC spread

| Jaar | ROIC % | WACC (schatting) % | Spread (pp) | Oordeel |
|---|---|---|---|---|
| 2016 | 23,70 | 8,13 | 15,57 | waardecreatie |
| 2017 | 25,38 | 8,13 | 17,25 | waardecreatie |
| 2018 | 30,26 | 8,13 | 22,13 | waardecreatie |
| 2019 | 29,29 | 8,13 | 21,16 | waardecreatie |
| 2020 | 30,24 | 8,13 | 22,11 | waardecreatie |
| 2021 | 30,63 | 8,13 | 22,50 | waardecreatie |
| 2022 | 32,83 | 8,13 | 24,70 | waardecreatie |
| 2023 | 33,74 | 8,13 | 25,61 | waardecreatie |
| 2024 | 33,16 | 8,13 | 25,03 | waardecreatie |
| 2025 | 34,25 | 8,13 | 26,12 | waardecreatie |

**Verantwoording van de WACC-kolom.** Ik gebruik hier één WACC — de actuele
8,13% uit sectie 12 — voor de hele reeks, in plaats van per jaar een eigen WACC
te construeren. Reden: een geloofwaardige jaar-WACC vereist per jaar een
gedocumenteerde risicovrije rente, ERP en beta, en die heb ik voor de oudere
jaren niet uit een primaire bron. Een jaarreeks verzinnen zou precies de fout
zijn die de methodiek verbiedt. De conclusie is bovendien ongevoelig voor de
keuze: bij elke plausibele WACC tussen 6,5% en 9,5% blijft de spread boven de
14 procentpunt in elk jaar. De spread is niet alleen positief maar ook
stijgend, en die stijging is structureel: hij komt uit een dalende
kapitaalintensiteit (asset turnover van 1,45 naar 1,68) bij een stabiele
bedrijfsmarge, niet uit tijdelijk hoge prijzen.

### Waarderingsratio's

| Ratio | Huidig (7-8-2026) | 10-jaars gemiddelde (2016-2025) |
|---|---|---|
| P/E (TTM) | 44,31 | 27,01 |
| P/E forward (consensus 2026: EPS 1,26) | 41,14 | — |
| P/FCF (na SBC, FY2025) | 57,08 | — |
| EV/EBITDA (TTM) | 30,40 | — |
| EV/Sales (TTM) | 6,79 | — |
| P/B (eigen vermogen 31-12-2025) | 15,08 | — |
| Dividendrendement (2026 run-rate 1,00) | 1,93% | — |
| PEG (P/E TTM ÷ consensus EPS-groei 15,77%) | 2,81 | — |

- **Toelichting waardering:** De koers-winstverhouding van 44,31 ligt 64% boven
het tienjaars gemiddelde van 27,01 en boven élk afzonderlijk jaareinde in die
periode — het vorige record was 36,22 eind 2025 en daarvóór 35,67 eind 2021.
Dat is de kern van deze analyse: er is niets aan de resultaten van Fastenal dat
in 2026 fundamenteel beter is dan in 2018, toen het aandeel op 16,4 keer de
winst stond, maar beleggers betalen er bijna drie keer zoveel voor. De
EV/EBITDA van 30,4 tegenover 20,8 voor Grainger en 17,1 voor MSC Industrial
bevestigt dat het premie-effect bedrijfsspecifiek is en niet sectorbreed.

### Sector-specifieke KPI's (industriële distributie)

| KPI | 2023 | 2024 | 2025 | Q2 2026 |
|---|---|---|---|---|
| Aantal filialen (jaareinde) | — | — | 1.595 | — |
| Gewogen FASTBin/FASTVend geïnstalleerd (MEU) | — | 126.997* | 136.638 | 140.789 |
| Gewogen FMI-ondertekeningen (MEU, per jaar) | — | — | 25.892 | 6.993 (kwartaal) |
| Digitale voetafdruk (% van omzet) | — | — | 61,4 | 61,6 |
| Contractomzet (% van omzet) | — | 74 (2025) | 74 | 75,8 |
| Klantlocaties met > USD 50k/maand | — | 2.331* | 2.657 | 3.125 |
| Klantlocaties met > USD 10k/maand | — | 10.834* | 11.712 | 12.865 |
| Totaal actieve klantlocaties (maandgemiddelde) | — | — | 98.361 | 93.283 (kwartaaleinde) |
| Personeel absoluut (jaareinde) | — | 23.707* | 24.489 | — |
| Personeel FTE | — | 20.951* | 21.602 | 22.230 |

*\* Afgeleid uit het gerapporteerde groeipercentage in het FY2025-persbericht
(bijvoorbeeld 136.638 ÷ 1,076 voor de MEU's). Waar het percentage op één
decimaal is gerapporteerd, is het afgeleide cijfer op één procent nauwkeurig.*

- **Toelichting sector-KPI's:** Het aantal geïnstalleerde FMI-eenheden is de
belangrijkste voorlopende indicator: elke automaat die erbij komt genereert
jarenlang omzet zonder dat er een verkoper aan te pas komt. Die teller loopt
gestaag door (+7,6% in 2025, +6,5% in Q2 2026) maar langzamer dan de omzet — de
groei komt tegenwoordig dus meer uit prijs en uit grotere klanten dan uit nieuwe
apparatuur. De tweede indicator is verschuivend: het aantal locaties boven USD
50k per maand groeide 16,5%, terwijl het totale aantal actieve locaties daalde
van 101.440 naar 93.283. Fastenal ruilt kleine klanten in voor grote — goed voor
de omzet, ongunstig voor de brutomarge.

### 2.8 Dividendanalyse

| Jaar | DPS | Groei YoY % | Payout (EPS) % | FCF-payout % | FCF-dekking | Bijzonderheden |
|---|---|---|---|---|---|---|
| 2013 | 0,200 | — | 53,1 | 113,3 | 0,88 | |
| 2014 | 0,250 | 25,0 | 60,2 | 95,7 | 1,05 | |
| 2015 | 0,280 | 12,0 | 63,3 | 83,5 | 1,20 | |
| 2016 | 0,300 | 7,1 | 69,5 | 106,8 | 0,94 | |
| 2017 | 0,320 | 6,7 | 63,8 | 79,3 | 1,26 | |
| 2018 | 0,385 | 20,3 | 58,8 | 88,8 | 1,13 | |
| 2019 | 0,435 | 13,0 | 63,2 | 83,6 | 1,20 | |
| 2020 | 0,700 | 60,9 | 93,8 | 86,0 | 1,16 | incl. speciaal dividend |
| 2021 | 0,560 | −20,0 | 69,9 | 104,9 | 0,95 | terugval na speciaal dividend 2020 |
| 2022 | 0,620 | 10,7 | 65,7 | 92,7 | 1,08 | |
| 2023 | 0,890 | 43,5 | 88,3 | 80,7 | 1,24 | incl. speciaal dividend |
| 2024 | 0,780 | −12,4 | 77,9 | 94,3 | 1,06 | terugval na speciaal dividend 2023 |
| 2025 | 0,875 | 12,2 | 80,0 | 95,6 | 1,05 | |
| 2026 (run-rate) | 1,000 | 14,3 | — | — | — | Q1 0,24 / Q2 0,24 / Q3 0,26 verklaard |

*DPS op kasbasis, herrekend naar de huidige split-basis. De dalingen in 2021 en
2024 zijn géén dividendverlagingen: het reguliere kwartaaldividend steeg elk van
die jaren. Ze weerspiegelen het wegvallen van de speciale uitkeringen die in
2020 en 2023 werden betaald.*

- **Dividend-CAGR 2015-2025:** 12,07% per jaar — ruim boven de Amerikaanse
inflatie over die periode, dus reële dividendgroei. Wie in 2015 kocht op de
koers van 7,70 (split-gecorrigeerd) ontvangt over 2025 een dividend van 0,875,
oftewel een *yield on cost* van 11,4%. Wie in 2020 kocht op 21,29 zit op 4,1%.
- **Dividendsoorten:** regulier kwartaaldividend, aangevuld met incidentele
speciale uitkeringen (2020 en 2023). Geen stockdividend.
- **Dividendbeleid:** Fastenal publiceert geen formele payout-doelstelling of
-bandbreedte en heeft geen expliciet "progressive dividend"-beleid, maar heeft
het reguliere dividend in geen enkel jaar van de onderzochte periode verlaagd —
ook niet in 2020, het jaar waarin het bovendien een extra uitkering deed. Het
dividend werd voor het laatst verhoogd naar USD 0,26 per kwartaal, betaalbaar
25 augustus 2026 aan aandeelhouders van 28 juli 2026.
- **Rendement:** op de run-rate van 2026 (USD 1,00) bedraagt het
dividendrendement 1,93%. Dat ligt onder het rendement op de Amerikaanse
tienjaarslening van 4,69% — als obligatiealternatief is dit aandeel dus niet
aantrekkelijk, en de belegger moet het verschil volledig uit koerswinst en
dividendgroei halen.

- **Toelichting dividend:** Fastenal keert sinds jaar en dag een groeiend
kwartaaldividend uit en heeft dat in de hele onderzochte periode niet één keer
verlaagd — ook niet in 2020, het jaar waarin het bovendien een extra uitkering
deed. De schijnbare dalingen in 2021 en 2024 zijn optisch: in 2020 en 2023 werd
een speciaal dividend betaald, en het jaar daarna viel dat weg. Het dividend
groeide met 12,1% per jaar over 2015-2025, ruim boven de inflatie. Opvallend is
niet de groei maar de omvang: met een FCF-payout van 95,6% gaat vrijwel de hele
vrije kasstroom naar de aandeelhouder.

- **Eindoordeel dividend:** Houdbaar maar niet ruim gedekt. De FCF-dekking
bedroeg in 2025 1,05× — ver onder de 1,5× die comfortabel heet — en de
FCF-payout van 95,6% ligt boven de waarschuwingsgrens van 80%. Er is dus geen
buffer voor een tegenvallend jaar. Het dividend is groeiend en bestendig, maar
draagt bij 1,93% rendement weinig bij aan de beleggingsthese.

### Non-GAAP / aanpassingen
- **Gebruikt?** false
- **Welke aanpassingen**: geen. Fastenal rapporteert uitsluitend op US-GAAP-grondslag en publiceert geen adjusted earnings, adjusted EBITDA of underlying result.
- **Waarom**: n.v.t.

---

## 4. Moat (concurrentievoordeel)

- **Oordeel:** **NARROW MOAT**
- **Moat-categorieën:**

| Naam | Sterkte | Toelichting |
|---|---|---|
| Immateriële activa | zwak | Fastenal heeft geen patenten of merken van betekenis; een bout is een bout. De eigen merken Body Guard en ORMADUS zijn samen ongeveer 11% van de omzet en leveren margevoordeel op, maar geen prijszettingsmacht bij de klant. |
| Overstapkosten | sterk | Dit is het hart van de moat. Een klant met FASTVend-automaten en FASTBin-sensoren in zijn productiehal heeft Fastenal fysiek en digitaal in zijn proces ingebouwd; bij een Onsite staat er zelfs Fastenal-personeel binnen de fabrieksmuren. Overstappen betekent apparatuur laten weghalen, artikelbestanden opnieuw koppelen en inkoopprocessen herbouwen. 75,8% van de omzet loopt onder contract. |
| Netwerkeffecten | geen | De waarde van Fastenal voor klant A neemt niet toe doordat klant B ook klant is. Er is geen platform, geen marktplaats, geen tweezijdige markt. |
| Kostenvoordeel | sterk | Met 8,7 miljard dollar inkoopvolume, 19 distributiecentra, 5,3 miljoen vierkante voet opslag, twaalf geautomatiseerde magazijnen en een eigen wagenpark van circa 590 vrachtwagens en 9.200 bestelvoertuigen bedient Fastenal een order tegen kosten die een regionale distributeur niet kan evenaren. De bedrijfsmarge van 20% tegenover 8,8% bij MSC Industrial en 10,9% bij Applied Industrial kwantificeert dat verschil. |
| Efficiënte schaal | middel | Binnen één klantfabriek is er ruimte voor precies één Onsite-partner, en in kleine industriesteden is er zelden plaats voor twee distributeurs met een fysiek filiaal. Maar de markt als geheel is groot en gefragmenteerd — Grainger, Ferguson, MSC en duizenden lokale spelers zijn actief — dus dit is een lokaal en geen structureel voordeel. |

- **Kwantitatief bewijs:** Het rendement op geïnvesteerd kapitaal lag tien jaar
achtereen tussen 23,7% en 34,3%, met een spread ten opzichte van de
kapitaalkosten van 15,6 tot 26,1 procentpunt, en die spread is over die periode
gestegen in plaats van geërodeerd. De bedrijfsmarge bewoog in diezelfde tien jaar
tussen 19,8% en 20,8% — een bandbreedte van één procentpunt, inclusief een
pandemiejaar en twee industriële recessies. Een bedrijf zonder
concurrentievoordeel laat zulke cijfers niet zien; concurrenten zouden de
overwinst weggeconcurreerd hebben. Het derde bewijsstuk staat in de
vergelijkingstabel van sectie 6: Fastenal haalt een bedrijfsmarge van 20,3% waar
MSC Industrial op 8,8% en Applied Industrial op 10,9% blijven steken, bij
dezelfde klanten en grotendeels dezelfde producten. Dat verschil van meer dan
tien procentpunt is niet met betere inkoop te verklaren maar met een structureel
lagere kostprijs per uitgeleverde order.
- **Duurzaamheid:** houdbaar op tien jaar, onzeker op twintig. Het
kostenvoordeel is fysiek en cumulatief — elk nieuw distributiecentrum en elke
extra geïnstalleerde automaat maakt de positie iets steviger — en zulke
voordelen verdwijnen niet snel; het kost een uitdager jaren en honderden
miljoenen om hetzelfde netwerk neer te zetten. De overstapkosten zijn per klant
hoog maar niet onoverkomelijk: een grote klant die zijn hele MRO-inkoop
heroverweegt, kán overstappen, het kost hem een half jaar rompslomp. Op twintig
jaar is de grootste onzekerheid niet de concurrent maar de klant zelf, die
steeds professioneler inkoopt.
- **Erosierisico's:** Amazon Business is de meest genoemde dreiging en is
serieus voor het gestandaardiseerde, laagfrequente deel van het assortiment,
maar heeft geen antwoord op de automaat in de productiehal. Grainger investeert
zwaar in hetzelfde digitale terrein en is ruim twee keer zo groot in omzet.
De reële erosie zal waarschijnlijk sluipend zijn: verdere brutomargedruk
naarmate het klantenbestand verder naar grote, professioneel inkopende partijen
verschuift — precies de daling van 51,7% naar 45,0% die de tabel in sectie 3 al
dertien jaar laat zien.

---

## 5. Management

- **CEO-naam + tenure:** Jeffery M. (Jeff) Watts, CEO sinds 16 juli 2026. Watts
begon in 1996 bij Fastenal en werkte zich op door de verkooporganisatie; in
augustus 2024 werd hij President en Chief Sales Officer, de rol waarin hij twee
jaar op het CEO-schap werd voorbereid. Zijn voorganger Daniel L. Florness was CEO
van januari 2016 tot juli 2026, kwam eveneens in 1996 binnen als CFO en bracht de
omzet van 3,96 naar 8,20 miljard dollar.
- **CFO-naam + tenure:** Max Tunnicliff, CFO sinds 10 november 2025 en de enige
recente benoeming van buiten. Hij was daarvoor CFO van Beko Europe en CFO
Azië-Pacific bij Whirlpool, waar hij ook hoofd interne audit en VP Strategie was
en teams leidde voor financiële verslaggeving, categorie-winstgevendheid en
supply-chain-finance. Zijn voorganger Holden Lewis vertrok begin 2025, waarna
Sheryl Lisowski — tevens Chief Accounting Officer en Treasurer — de functie
enkele maanden waarnam. De internationale ervaring van Tunnicliff sluit aan op de
groeiambitie buiten Noord-Amerika.
- **Oprichter nog betrokken?** Nee. Bob Kierlin was CEO van 1968 tot 2002 en
bleef voorzitter tot zijn vertrek uit de raad van bestuur in 2014.
- **Insider ownership %:** minder dan 1% (peildatum februari 2026).
- **Capital allocation (samenvatting):** Alle groei is organisch en uit eigen
kasstroom gefinancierd; er is geen overname van betekenis gedaan en er staat nul
goodwill op de balans. De vrije kasstroom gaat vrijwel volledig naar een
groeiend dividend, met incidenteel een bescheiden aandeleninkoop. Het rendement
op geïnvesteerd kapitaal steeg over tien jaar van 23,7% naar 34,3%.

- **Capital allocation — detail:**

| Jaar | Dividend totaal | Aandeleninkoop | M&A-uitgaven | Organische capex |
|---|---|---|---|---|
| 2019 | 498,6 | 0,0 | 0,0 | 246,4 |
| 2020 | 803,4 | 52,0 | 0,0 | 168,1 |
| 2021 | 643,7 | 0,0 | 0,0 | 156,6 |
| 2022 | 711,3 | 237,8 | 0,0 | 173,8 |
| 2023 | 1.016,8 | 0,0 | 0,0 | 172,8 |
| 2024 | 893,3 | 0,0 | 0,0 | 226,5 |
| 2025 | 1.004,2 | 0,0 | 0,0 | 245,3 |
| 2026 (H1) | ca. 551 | 49,8 | 0,0 | ca. 160 |

- **M&A-track-record:** vrijwel afwezig, en dat is een compliment. Fastenal
heeft in de gehele onderzochte periode geen enkele overname van materiële
omvang gedaan; de balans draagt nul goodwill. Er zijn dus ook geen
goodwill-afschrijvingen, geen niet-gerealiseerde synergieën en geen
integratieproblemen. De laatste noemenswaardige transactie was de aankoop van
delen van fastenerfabrikant Holo-Krome in 2009. Alle groei is organisch
gefinancierd uit eigen kasstroom.
- **Timing van aandeleninkopen:** matig. De twee grootste inkoopjaren van het
afgelopen decennium waren 2022 (USD 237,8 mln bij een jaareinde-P/E van 22,85)
en 2015 (USD 293,0 mln bij een P/E van 17,39) — beide relatief goedkope
momenten, dus dat is verstandig getimed. Maar het bedrag is klein en het
programma ligt sindsdien grotendeels stil: in 2023, 2024 en 2025 werd niets
ingekocht, en in de eerste helft van 2026 slechts 1,075 mln aandelen voor USD
49,8 mln tegen gemiddeld 46,33 — bij een koers die 64% boven het historische
gemiddelde P/E ligt. Van de autorisatie uit juli 2022 resteert nog circa 10,25
mln aandelen. Dat het management juist nú weer begint in te kopen op de duurste
waardering in de geschiedenis van het aandeel, is de zwakste stip in een verder
sterk kapitaalallocatiebeeld.
- **Beloning:** Fastenal hanteert een ongewoon sober beloningsmodel. De totale
aandelencompensatie bedroeg in 2025 USD 8,4 mln — 0,014% van de marktwaarde,
tegenover een sectorgemiddelde dat vaak boven de 1% ligt en een grens van 3%
die als hoog geldt. De verwatering is negatief: het aantal verwaterde aandelen
daalde van 1.190,7 mln (2013) naar 1.150,3 mln (2025). Prikkels worden vooral
gegeven via aandelenopties met een uitoefenprijs die jaren geleden is
vastgelegd, wat de directie belang geeft bij koersstijging op meerjarige
termijn.
- **Insider transactions (laatste 24 maanden):**

| Datum | Persoon | Functie | Type | Aantal | Koers |
|---|---|---|---|---|---|
| 2026-08-05 | Rita J. Heise | commissaris | verkoop na optie-uitoefening | 34.964 | 50,05 |
| 2026-07-28 | Michael J. Ancius | — | verkoop na optie-uitoefening | 3.000 | 49,00 |
| 2026-03-05 | Reyne K. Wisecup | commissaris | verkoop na optie-uitoefening | 36.920 | 47,34 |
| 2026-01-23 | Scott Satterlee | voorzitter RvB | verkoop na optie-uitoefening | 15.964 | 44,19 |
| 2025-11-19 | Sarah N. Nielsen | commissaris | **open-markt aankoop** | 1.000 | 39,60 |
| 2025-11-17 | Daniel L. Johnson | commissaris | **open-markt aankoop** | 1.000 | 40,44 |
| 2025-11-13 | Hsenghung Sam Hsu | commissaris | **open-markt aankoop** | 1.000 | 49,58 |
| 2025-11-12 | Stephen L. Eastman | commissaris | **open-markt aankoop** | 1.000 | 40,82 |
| 2025-10-16 | Hsenghung Sam Hsu | commissaris | **open-markt aankoop** | 1.000 | 42,45 |
| 2025-08-12 | Daniel L. Florness | CEO | verkoop na optie-uitoefening | 84.612 | 48,44 |
| 2025-08-08 | Jeffery M. Watts | President/CSO | verkoop na optie-uitoefening | 48.724 | 48,05 |
| 2025-08-08 | John L. Soderberg | Senior EVP-IT | verkoop na optie-uitoefening | 34.612 | 48,03 |
| 2025-07-24 | Anthony P. Broersma | EVP-Operations | verkoop na optie-uitoefening | 13.582 | 47,93 |
| 2025-07-17 | Sheryl A. Lisowski | interim-CFO | verkoop na optie-uitoefening | 21.052 | 45,21 |
| 2025-04-24 | Daniel L. Florness | CEO | verkoop na optie-uitoefening | 50.000 | 82,12 (pre-split) |
| 2025-03-14 | Holden Lewis | CFO | verkoop na optie-uitoefening | 68.664 | 74,96 (pre-split) |

*Koersen vóór mei 2025 staan op pre-split basis (de 2-voor-1 split vond plaats
in mei 2025). Aankopen van 20 aandelen door Charles S. Miller (april 2025)
zijn weggelaten wegens verwaarloosbare omvang.*

Netto is er in de afgelopen 24 maanden fors méér verkocht dan gekocht, maar de
verkopen zijn vrijwel zonder uitzondering directe doorverkoop na
optie-uitoefening — de klassieke, weinig informatieve vorm. De informatieve
transacties zijn de vijf open-markt aankopen door commissarissen in oktober en
november 2025, tegen koersen van 39,60 tot 49,58. Dat zijn kleine bedragen
(circa USD 40.000 elk) maar het zijn wél echte aankopen met eigen geld, en ze
vonden plaats toen het aandeel op zijn laagste niveau van het afgelopen jaar
stond. Sinds die aankopen is de koers ruim 25% gestegen; er zijn daarna geen
open-markt aankopen meer gemeld.

- **Integriteit en transparantie:** Er is geen pre-IPO-schuldopbouw geweest en
het bedrijf heeft nooit een dividendrecapitalisatie of leveraged structuur
gekend. In de geraadpleegde bronnen zijn geen controverses, materiële
rechtszaken of regulatoire maatregelen aangetroffen. Woord en daad komen
overeen: de doelstelling voor de digitale voetafdruk werd in Q2 2026 openlijk
bijgesteld van 66% naar 63-64% mét opgave van reden, en de CFO gaf op dezelfde
call toe dat de prijsdoorberekening bij tarieven "niet snel genoeg" ging en 0,4%
marge kostte. Een directie die een gemiste doelstelling zelf benoemt vóór
analisten ernaar vragen, scoort hoog op downside transparency.
- **Oordeel management:** **STERK**
- **Toelichting:** Het beeld is consistent en zeldzaam degelijk. Fastenal groeit
al decennia volledig organisch, koopt geen bedrijven, draagt geen goodwill en
financiert alles uit eigen kasstroom. De aandelencompensatie is met 0,014% van de
marktwaarde te verwaarlozen en het aandelenaantal daalde per saldo, terwijl
verwatering bij vergelijkbare Amerikaanse bedrijven eerder regel dan uitzondering
is. Het rendement op geïnvesteerd kapitaal steeg over tien jaar van 23,7% naar
34,3%. Twee punten houden het oordeel af van onbetwist: het insiderbelang is met
minder dan 1% laag, en de hervatte aandeleninkoop in 2026 gebeurt tegen de
duurste waardering uit de bedrijfsgeschiedenis.

---

## 6. Sector & concurrentie

- **Sector-groeivooruitzicht:** De Noord-Amerikaanse markt voor industriële
distributie groeit structureel ongeveer in lijn met de industriële productie
plus inflatie, historisch enkele procenten per jaar. Een geverifieerd
groeicijfer voor de totale markt met bron heb ik niet kunnen vinden en wordt
daarom niet ingevuld (zie ontbrekende data, sectie 13). Wat wél verifieerbaar
is: de Amerikaanse inkoopmanagersindex stond in het tweede kwartaal van 2026
gemiddeld net boven de 53, wat op bescheiden groei duidt, en Fastenal groeide in
datzelfde kwartaal met 14,7% — dus veruit het grootste deel van de groei is
marktaandeelwinst en prijs, niet marktgroei.
- **Porter five forces:**
  - **Rivaliteit — hoog.** Grainger is met USD 18,9 mrd omzet ruim twee keer zo
    groot, Ferguson en Wesco opereren in aangrenzende segmenten, MSC Industrial
    en Applied Industrial vechten om dezelfde fabrieksklant, en daaronder zitten
    duizenden regionale distributeurs. Concurrentie gaat op service en totale
    kosten, niet uitsluitend op prijs, maar de druk op de brutomarge is
    zichtbaar en aanhoudend.
  - **Nieuwe toetreders — laag.** Een landelijk netwerk van 1.595 vestigingen,
    19 distributiecentra en 140.789 geïnstalleerde apparaten is niet in enkele
    jaren na te bouwen; de benodigde investering en de tijd om klantvertrouwen op
    te bouwen vormen een reële drempel.
  - **Substituten — middel.** Rechtstreekse inkoop bij de fabrikant en
    e-commerceplatforms zoals Amazon Business zijn echte alternatieven voor
    gestandaardiseerde artikelen met een hoge stukprijs, maar geen van beide
    levert de voorraad ter plaatse met automatisch bijvullen — en juist dat is
    waar de klant zijn proceskosten mee verlaagt.
  - **Macht leveranciers — laag.** Fastenal koopt bij honderden fabrikanten,
    produceert een deel zelf en heeft met 8,7 mrd inkoopvolume een sterke
    onderhandelingspositie. Geen enkele leverancier is onmisbaar, en waar een
    fabrikant toch te veeleisend wordt, kan het bedrijf terugvallen op zijn eigen
    merken Body Guard en ORMADUS.
  - **Macht afnemers — middel tot hoog en toenemend.** Geen enkele klant is 5%
    of meer van de omzet, maar het zwaartepunt verschuift naar grote nationale
    accounts die professioneel inkopen. Die verschuiving verklaart de
    brutomargedaling van 51,7% naar 45,0% sinds 2013.

  **Conclusie Porter:** gemiddeld aantrekkelijke sector. De toetredingsdrempels
  en de zwakke leveranciersmacht beschermen de winstgevendheid, maar de
  toenemende macht van grote afnemers zorgt voor structurele, geleidelijke
  margedruk die alleen met kostenbesparing te compenseren is.

- **Concurrenten:**

| Concurrent | Omzet TTM (USD mrd) | EBIT-marge % | ROIC % | Schuld/EBITDA | EV/EBITDA | P/FCF | P/E |
|---|---|---|---|---|---|---|---|
| **Fastenal (FAST)** | **8,75** | **20,3** | **34,3** | **0,06** | **30,4** | **57,1** | **44,3** |
| W.W. Grainger (GWW) | 18,85 | 15,7 | 33,2 | 0,93 | 20,8 | 39,9 | 32,6 |
| Applied Industrial (AIT) | 4,84 | 10,9 | 19,9 | 0,62 | 22,7 | 30,2 | 33,9 |
| MSC Industrial (MSM) | 3,91 | 8,8 | 13,6 | 1,28 | 17,1 | 31,4 | 29,7 |

*Marktaandelen per speler zijn niet uit een verifieerbare bron beschikbaar en
worden daarom niet ingevuld.*

- **Positie van het bedrijf:** koploper op kwaliteit, uitdager op omvang.
Fastenal is qua omzet de nummer twee achter Grainger, maar de nummer één op elke
rendementsmaatstaf: de hoogste bedrijfsmarge (20,3% tegenover 15,7%, 10,9% en
8,8%), het hoogste rendement op geïnvesteerd kapitaal, en verreweg de sterkste
balans (0,06× schuld/EBITDA tegen 0,62 tot 1,28 bij de anderen). Het
onderscheidend vermogen zit in het FMI-model, dat geen van de drie
vergelijkingsgroepen in dezelfde schaal heeft. De keerzijde staat in dezelfde
tabel: de markt wéét dit, en rekent er een EV/EBITDA van 30,4 voor, tegenover
17 tot 23 voor de rest.

### TAM/SAM/SOM
- **TAM:** niet verifieerbaar uit een primaire of onafhankelijke bron — weggelaten
- **SAM:** niet verifieerbaar — weggelaten
- **Bron TAM/SAM:** geen
- **Toelichting:** Een geloofwaardige marktomvang voor Noord-Amerikaanse
MRO-distributie kon ik niet uit een bron halen die aan de bronneneisen voldoet;
de circulerende cijfers komen uit betaalde marktrapporten die ik niet inzag. In
plaats van te schatten blijven deze velden leeg. Over de plausibiliteit valt wél
iets te zeggen: Fastenal zette in 2025 8,2 mrd om in een markt waarin de vier
grootste spelers samen ruim 36 mrd omzetten. De vraag is niet of de markt groot
genoeg is, maar of de brutomarge overeind blijft.

---

## 7. Analyse-frameworks

### Graham
- **Oordeel**: VOLDOET NIET
- **Graham number**: 9,51
- **Margin of safety %**: −81,7 (de koers ligt 445% boven het Graham-getal)
- **Toelichting:** Graham zoekt een defensief aandeel met een P/E onder 15, een
P/B onder 1,5 en een prijs onder het Graham-getal (de wortel van 22,5 × winst ×
boekwaarde). Fastenal faalt op alle drie: P/E 44,31, P/B 15,08 en een Graham-getal
van 9,51 tegenover een koers van 51,84. Op de criteria die niets met prijs te
maken hebben scoort het bedrijf juist voorbeeldig — schuldratio 0,03,
onafgebroken winstgroei, dertien jaar ononderbroken dividend — maar Graham is
bovenal een prijsdiscipline, en die staat hier op rood.
- **Cascade**: regel 5 (P/E ≤ 15 én P/B ≤ 1,5) nee → regel 4 (P/E ≤ 15) nee →
  regel 3 (P/E ≤ 20) nee → regel 2 (P/E ≤ 25) nee → regel 1 (P/E > 25 óf
  P/B > 3,0) ja. Geen botsing tussen regels.
- **Score (1-5)**: **1**

### Buffett / Munger
- **Oordeel**: GEDEELTELIJK
- **ROIC structureel boven WACC?** true — tien jaar achtereen, met een spread die
  opliep van 15,6 naar 26,1 procentpunt
- **Toelichting:** Buffett vraagt om een uitzonderlijk bedrijf tegen een
redelijke prijs. Fastenal levert het eerste deel volledig: het bedrijfsmodel is
in tien minuten uit te leggen, de kasstromen zijn voorspelbaar, het rendement op
geïnvesteerd kapitaal is meer dan vier keer de kapitaalkosten en de directie is
degelijk. Het tweede deel ontbreekt: bij 57 keer de vrije kasstroom is de prijs
niet redelijk. Dit is het framework waarin de spanning tussen kwaliteit en
waardering het scherpst zichtbaar wordt.
- **Cascade — botsing, expliciet gemeld**: regel 5 en 4 vereisen P/FCF ≤ 20
  respectievelijk een WIDE moat — beide nee. Regel 3 vereist ROIC > WACC
  structureel (ja) én moat NARROW+ (ja) én P/FCF ≤ 30 (nee, 57,08). Regel 1
  (ROIC < WACC structureel) is aantoonbaar onwaar. De score valt daarmee terug op
  regel 2 als restcategorie: te goed voor een 1, te duur voor een 3.
- **Score (1-5)**: **2**

### Peter Lynch
- **Categorie**: Stalwart — een groot, gevestigd bedrijf dat met circa 8% per
  jaar groeit, betrouwbaar maar niet spectaculair
- **Oordeel**: ONINTERESSANT (op deze prijs)
- **PEG-ratio**: 2,81 (P/E 44,31 gedeeld door de consensusverwachting van 15,77%
  winstgroei voor 2026)
- **Toelichting:** Lynch kocht stalwarts alleen met korting, en zijn vuistregel
was dat de P/E niet ver boven de groeivoet mag liggen. Hier is de P/E bijna drie
keer de groeivoet. Het verhaal is wél zo helder als Lynch het graag zag: Fastenal
verkoopt schroeven en handschoenen uit automaten die in fabrieken staan. De
categorie-indeling verdient toelichting — met 14,7% omzetgroei in het laatste
kwartaal oogt Fastenal als fast grower, maar de tienjaars omzet-CAGR van 7,8% en
de EPS-CAGR van 9,3% wijzen op een stalwart met een goed jaar.
- **Cascade**: regel 5 t/m 2 vereisen alle een PEG van 2,0 of lager; met 2,81
  treft regel 1. Geen botsing.
- **Score (1-5)**: **1**

### Phil Fisher
- **Oordeel**: STERK
- **Toelichting:** Fisher keek naar de kwaliteit van de onderneming en negeerde
de prijs grotendeels. Margebescherming door de moat is aantoonbaar: de
bedrijfsmarge bewoog tien jaar tussen 19,8% en 20,8% terwijl de brutomarge zes
procentpunt daalde. Managementintegriteit is sterk, zoals onderbouwd in sectie 5.
De derde toetssteen — R&D boven sectorgemiddelde — wordt formeel niet gehaald
omdat een distributeur geen R&D rapporteert, terwijl Fastenal wél eigen
technologie ontwikkelt (FASTBin-sensoren, FASTVend, FAST360°, geautomatiseerde
magazijnen) en die als capex boekt.
- **Cascade**: regel 5 vereist alle drie criteria — nee. Regel 4 vereist er twee:
  margebescherming en integriteit zijn beide vervuld, dus regel 4 treft.
- **Score (1-5)**: **4**

### Magic Formula (Greenblatt)
- **Oordeel**: ONAANTREKKELIJK
- **Earnings yield %**: 2,99 (EBIT TTM 1.775,1 gedeeld door de
  ondernemingswaarde van 59.401,7)
- **Return on capital %**: 45,53 (EBIT 2025 van 1.655,7 gedeeld door
  netto werkkapitaal 2.504,7 plus netto vaste activa 1.131,6)
- **Toelichting:** Greenblatt zoekt bedrijven die veel verdienen op hun kapitaal
én goedkoop zijn. Fastenal scoort uitzonderlijk op de eerste as — 45,5% rendement
op kapitaal plaatst het in de bovenste regionen van elk universum — en zwak op de
tweede. Het winstrendement van 2,99% ligt onder wat een Amerikaanse
tienjaarslening oplevert (4,69%): een koper van het hele bedrijf verdient minder
dan een koper van staatsschuld, zonder de zekerheid.
- **Cascade — marginale uitkomst, expliciet gemeld**: regel 5 en 4 vereisen 10%
  respectievelijk 7% winstrendement — nee. Regel 3 vraagt 5% winstrendement óf
  50% rendement op kapitaal — beide net niet (2,99% en 45,53%). Regel 2 vraagt
  minstens 3% winstrendement en 2,99% mist dat met één honderdste procentpunt,
  zodat regel 1 treft. De uitkomst is robuust: op de FY2025-EBIT is het
  winstrendement 2,79%, dus bij elke redelijke definitie blijft het onder 3%.
- **Score (1-5)**: **1**

### Moat
- **Score (1-5)**: **3** — NARROW MOAT (twee van de vijf categorieën sterk) met
  een ROIC-WACC spread van 26,1 procentpunt. Regel 5 vereist een monopolie of
  duopolie, wat in een markt met Grainger, Ferguson, MSC, Applied Industrial en
  duizenden regionale spelers niet aan de orde is. Regel 4 vereist een WIDE moat,
  dus minstens drie sterke categorieën; er zijn er twee. Regel 3 (NARROW moat en
  spread boven 5 procentpunt) treft.

### Management
- **Score (1-5)**: **4** — kapitaalallocatie goed (organische groei, nul
  goodwill, stijgende ROIC), prikkels in lijn met aandeelhouders (SBC 0,014% van
  de marktwaarde, negatieve verwatering), geen controverses aangetroffen. Regel 5
  vereist naast dit alles een insiderbelang boven 1%; dat is minder dan 1% en
  blokkeert de hoogste score.

### Fair Value DCF
- **Score (1-5)**: **1** — de fair value in het basisscenario bedraagt 21,98
  tegenover een koers van 51,84, een neerwaarts potentieel van 57,6%. Regel 1
  (downside groter dan 15%) treft.

### Fair Value IPO-gecorr.
- **Score (1-5)**: **1** — Fastenal ging in 1987 naar de beurs, bijna veertig
  jaar geleden. Volgens de rubric is de score in dat geval gelijk aan die van
  Fair Value DCF basis. Er is geen IPO-correctie van toepassing.

### Scorekaart totaal

| Framework | Score (1-5) | Oordeel |
|---|---|---|
| Graham | 1 / 5 | P/E 44,31 en P/B 15,08 — regel 1 |
| Buffett / Munger | 2 / 5 | uitzonderlijk bedrijf, P/FCF 57 blokkeert regel 3 |
| Peter Lynch | 1 / 5 | PEG 2,81 — regel 1 |
| Phil Fisher | 4 / 5 | twee van drie criteria voldaan |
| Magic Formula | 1 / 5 | winstrendement 2,99% — regel 1 |
| Moat | 3 / 5 | NARROW moat, spread 26,1 pp |
| Management | 4 / 5 | goede allocatie, insiderbelang < 1% |
| Fair Value DCF (basis) | 1 / 5 | downside 57,6% |
| Fair Value IPO-gecorr. | 1 / 5 | gelijk aan DCF basis (IPO 1987) |
| **TOTAALSCORE** | **18 / 45** | **PASS** |

- **Totaalscore**: 18
- **Max**: 45
- **Eindoordeel**: **PASS** — totaal 18 ligt onder de drempel van 24, én de
  score op Fair Value DCF is 1. Beide criteria leiden onafhankelijk van elkaar
  tot PASS.
- **Samenvatting:** Fastenal is een van de best geleide industriële
distributeurs ter wereld. Het verdient al tien jaar meer dan 23% op zijn
geïnvesteerde kapitaal, houdt de bedrijfsmarge binnen één procentpunt constant,
draagt geen goodwill en geen schuld, verwatert zijn aandeelhouders niet en heeft
zojuist een twee jaar voorbereide CEO-wissel afgerond. Op elk kwalitatief
onderdeel scoort het hoog: moat 3, Fisher 4, management 4. Maar de scorekaart is
bewust óók een prijsdiscipline, en daar staat alles op rood. Bij 44 keer de
winst, 57 keer de vrije kasstroom en 30 keer de EBITDA moet de kasstroom tien
jaar lang met dubbele cijfers groeien om de koers te rechtvaardigen, terwijl de
tienjaarsgroei 10,4% bedroeg. De fair value in het basisscenario is 21,98 en
zelfs het optimistische scenario blijft op 30,03. De voornaamste onzekerheid is
niet operationeel maar multiple-gedreven: terugkeer naar de gemiddelde P/E van 27
kost 39% koers zonder dat één cijfer tegenvalt. De katalysatorkalender biedt geen
steun. Gegeven de uitstekende datakwaliteit volstaat een marge van 20%, wat het
koopniveau op 19,46 brengt.

---

## 8. Risico's

| # | Omschrijving | Kans | Impact | DCF-aanname geraakt | Toelichting |
|---|---|---|---|---|---|
| 1 | Multiple-compressie: de koers-winstverhouding keert terug naar het historische gemiddelde | HOOG | GROOT | geen (waarderingsrisico, geen kasstroomrisico) | Bij de huidige P/E van 44,31 tegenover een tienjaars gemiddelde van 27,01 kost een terugkeer naar dat gemiddelde 39% koers zonder dat er één operationeel cijfer tegenvalt. Dit is het dominante risico en het staat volledig los van de kwaliteit van de onderneming. |
| 2 | Aanhoudende brutomargedruk door verschuiving naar grote klanten | HOOG | MIDDEL | EBIT-marge in de EPV en FCF-marge in de DCF | De brutomarge daalde van 51,7% (2013) naar 45,0% (2025) en in Q2 2026 opnieuw met 75 basispunten. Zolang Fastenal groeit door grote nationale accounts te winnen, gaat die daling door. Tot nu toe volledig gecompenseerd met kostenbeheersing; als dat stopt, daalt de FCF-marge onder de genormaliseerde 10,98%. |
| 3 | Industriële recessie in Noord-Amerika | MIDDEN | GROOT | FCF-groei fase 1 | 75,9% van de omzet komt uit de maakindustrie en 8,1% uit niet-residentiële bouw. In de zwakke industriejaren 2016 en 2024 groeide de omzet met slechts 2,4% respectievelijk 2,7%; in 2009 kromp zij met 17,5%. Een echte recessie zou het pessimistische scenario van 4% kasstroomgroei realistisch maken of zelfs onderschrijden. |
| 4 | Tarieven en inputkosten die sneller stijgen dan de doorberekening | MIDDEN | MIDDEL | brutomarge, dus FCF-marge | De CFO meldde over Q1 2026 dat tariefkosten "sneller door de winst- en verliesrekening bewogen dan onze prijzen", goed voor 0,4% margeverlies, en in Q2 2026 een netto prijs-kosteneffect van min 40 basispunten. Herhaalt zich dit meerdere kwartalen, dan wordt het structureel. |
| 5 | Kapitaalintensiteit loopt op | MIDDEN | MIDDEL | capex in de FCF-basis | Management stuurt op netto capex van circa USD 320 mln voor 2026, oftewel 3,5% van de omzet, tegenover een historische mediaan van 2,98%. Elk procentpunt extra capex kost bij de huidige omzet ongeveer USD 87 mln vrije kasstroom per jaar. |
| 6 | Dividend verbruikt vrijwel de hele vrije kasstroom | MIDDEN | KLEIN | geen directe DCF-aanname; wel flexibiliteit | De FCF-payout bedroeg in 2025 95,6% en de FCF-dekking 1,05× — ver onder de 1,5× die comfortabel heet. Er is dus geen buffer voor een tegenvallend jaar; dat moet dan uit de kredietfaciliteit worden overbrugd, of het dividend groeit dat jaar niet. Het risico is klein maar beperkt de manoeuvreerruimte van de nieuwe directie. |
| 7 | Overgang naar een nieuwe CEO | LAAG | MIDDEL | FCF-groei fase 1 | Jeff Watts nam op 16 juli 2026 over van Daniel Florness. De opvolging is bijna twee jaar voorbereid en beide mannen werken sinds 1996 bij het bedrijf, dus het risico is beperkt — maar elke wisseling aan de top brengt uitvoeringsrisico, zeker in combinatie met een CFO die pas sinds november 2025 in functie is. |
| 8 | Digitale concurrentie van Amazon Business en van Grainger | LAAG | MIDDEL | eindwaarde-groei | Voor het gestandaardiseerde deel van het assortiment zijn e-commerceplatforms een reëel alternatief, en Grainger investeert zwaar in hetzelfde digitale terrein. Het FMI-model beschermt de kern, maar de randen van het assortiment kunnen langzaam wegsijpelen — met effect op de eindwaarde, niet op de eerste jaren. |

**Verplicht risico-item — pre-IPO financial engineering:** niet geconstateerd,
en niet van toepassing. Fastenal ging in 1987 naar de beurs, bijna veertig jaar
geleden. Er zijn geen pre-IPO-schulden bij gerelateerde partijen, geen
IPO-opbrengsten die naar aflossing aan insiders gingen en geen
dividendrecapitalisatie geweest; het bedrijf heeft in de gehele onderzochte
periode een nettokaspositie of een verwaarloosbare nettoschuld gehad. Er is
daarom geen gecorrigeerde fair value: de IPO-gecorrigeerde waarde is gelijk aan
de basiswaarde van 21,98.

---

## 9. These invalide bij

Deze analyse concludeert dat Fastenal een uitstekend bedrijf tegen een te hoge
prijs is. Die conclusie is weerlegd wanneer de omzetgroei structureel — vier
kwartalen of langer — boven de 12% blijft **en** de bedrijfsmarge daarbij boven
20% blijft, want dan groeit het bedrijf de waardering geleidelijk in. Zij is
eveneens weerlegd wanneer de vrije kasstroom binnen twee jaar boven USD 1,6 mrd
uitkomt zonder eenmalige werkkapitaalmeevaller, of wanneer de koers onder 26
zakt, want dan wordt het optimistische scenario met een marge van 15% bereikbaar.
Omgekeerd wordt de these bevestigd zodra de bedrijfsmarge twee opeenvolgende
kwartalen onder 19% duikt.

---

## 10. ESG

### Materiële factoren (SASB-gebaseerd)

| Factor | SASB-categorie | Risiconiveau | Financiële impact | DCF-impact |
|---|---|---|---|---|
| Brandstof- en transportemissies eigen wagenpark | Milieu — Vlootbrandstofbeheer | Midden | Circa 590 vrachtwagens en 9.200 bestelvoertuigen; brandstofkosten en toekomstige emissieregels raken de bedrijfslasten | klein negatief op EBIT-marge |
| Arbeidsveiligheid en -omstandigheden distributiecentra | Sociaal — Werknemersveiligheid | Midden | 4.057 medewerkers in distributie en transport, 1.046 in productie; ongevallen leiden tot claims en verzuim | verwaarloosbaar |
| Personeelsverloop en talentbehoud verkooporganisatie | Sociaal — Werving en behoud | Midden | 17.166 van de 24.489 medewerkers zitten in verkoop; het model draait op lokale relaties | FCF-groei fase 1 |
| Leveranciersketen en herkomst van producten | Milieu/Sociaal — Ketenbeheer | Midden | Aanzienlijk deel van het assortiment komt uit Azië; tarieven en arbeidsomstandigheden zijn beide relevant | brutomarge |
| Productveiligheid en aansprakelijkheid | Sociaal — Productkwaliteit | Laag | Genoemd als risicofactor in het 10-K; bevestigingsmiddelen in kritische toepassingen | verwaarloosbaar |
| Cyberveiligheid en dataprivacy | Governance — Datasecurity | Midden | Genoemd als risicofactor; het FMI-netwerk is een groot, verbonden apparatenpark | verwaarloosbaar tenzij incident |

- **Eindoordeel ESG:** **LAAG RISICO**
- **Toelichting:** Fastenal is een distributeur zonder zware industriële
processen, zonder mijnbouw, zonder fossiele reserves en zonder noemenswaardige
regulatoire blootstelling. De materiële ESG-thema's zijn de gebruikelijke voor
een logistiek bedrijf: transportemissies, magazijnveiligheid en ketenbeheer.
Geen daarvan is groot genoeg om de waardering te beïnvloeden. Ik heb geen
ESG-ratings van derden overgenomen omdat die niet uit een primaire bron
verifieerbaar waren; het oordeel hierboven is gebaseerd op de risicofactoren en
de personeelsopbouw zoals gerapporteerd in het 10-K over 2025.

---

## 11. Katalysatoren

| Datum ca. | Omschrijving | Richting | Impact |
|---|---|---|---|
| 25-08-2026 | Betaling kwartaaldividend USD 0,26 (registratiedatum 28-07-2026 verstreken) | NEUTRAAL | KLEIN |
| okt-2026 | Kwartaalcijfers Q3 2026 — het eerste volledige kwartaal onder CEO Jeff Watts; de markt let op of de brutomargedruk van 75 basispunten uit Q2 aanhoudt | BINAIR | GROOT |
| okt-2026 | Verklaring kwartaaldividend Q4 2026 en opgave van de inkoopactiviteit over Q3 | NEUTRAAL | KLEIN |
| dec-2026 | Eventueel speciaal dividend — Fastenal deed dit in 2020 en 2023, telkens aangekondigd in het vierde kwartaal | POSITIEF | KLEIN |
| jan-2027 | Jaarcijfers 2026 plus de capexdoelstelling voor 2027; bevestiging of de netto capex boven 3,5% van de omzet blijft | BINAIR | MIDDEL |
| doorlopend 2026-2027 | Ontwikkeling Amerikaanse importtarieven en de snelheid waarmee Fastenal die doorberekent | NEGATIEF | MIDDEL |
| doorlopend 2026-2027 | Amerikaanse inkoopmanagersindex — stond in Q2 2026 gemiddeld net boven 53; een terugval onder 50 raakt direct de volumegroei | NEGATIEF | GROOT |
| feb-2027 | Publicatie Form 10-K 2026 met de nieuwe cijfers over FMI-installaties, contractomzet en klantlocaties | NEUTRAAL | KLEIN |

De kwartaalcijfers van oktober 2026 en januari 2027 zijn binaire gebeurtenissen
in de zin dat de marktreactie afhangt van één cijfer: de brutomarge. Bij een
aandeel op 44 keer de winst is de koersgevoeligheid voor een marge-afwijking van
een half procentpunt aanzienlijk. De tarief- en PMI-ontwikkelingen zijn
gradueel en werken via de groei-aanname van fase 1 in de DCF.

---

## 12. Fair value — kwantitatief (DCF)

### DCF-invoeren

```
Basis            fcf=960.7  shares=1147.5  net_cash=84.7  gross_debt=120.0
                 revenue=8749.4  koers=51.84  ipo_jaar=1987
WACC             rf=4.69  erp=4.28  beta=0.806  crp=0.0  size_premium=0.0
                 cost_of_debt_pretax=4.06  tax_rate=23.96
Pessimistisch    g1=4.0   g2=2.0  gt=2.00  wacc_adj=0.75   kans=25
Basis            g1=8.0   g2=5.0  gt=2.50  wacc_adj=0.00   kans=50
Optimistisch     g1=11.0  g2=7.0  gt=3.00  wacc_adj=-0.25  kans=25
EPV              norm_ebit_margin=20.24  maintenance_capex=179.2  da=179.2
                 norm_ebitda_margin=22.29
Multiples        pe=44.31  pb=15.08  p_fcf=57.08  peg=2.81
Rendement        roic=34.25  earnings_yield=2.99  roc_greenblatt=45.53
Kwalitatief      moat_oordeel=NARROW  moat_categorieen_sterk=2
                 management_oordeel=STERK  capital_allocation=GOED
                 insider_alignment_pct=0.9  roic_wacc_spread_5j_plus=true
                 structureel_dividend=true  debt_equity=0.0304
Eenheid          bedragen in USD mln; percentages als getal (3.05 = 3,05%)
```

*Herleidbaarheid van elke invoer: `rf` uit FRED DGS10 (6-8-2026); `erp` uit
Damodaran's implied ERP (1-8-2026); `beta` is de Blume-correctie
0,67 × 0,71 + 0,33 op de 5-jaars beta van StockAnalysis; `crp` is nul omdat
Fastenal in de Verenigde Staten gevestigd is en Damodaran de Amerikaanse
country risk premium op nul stelt; `size_premium` is nul bij een marktwaarde van
USD 59,5 mrd; `cost_of_debt_pretax` volgt uit de betaalde rente van 6,6 mln in
2025 gedeeld door de gemiddelde uitstaande schuld van 162,5 mln;
`tax_rate` is de effectieve belastingdruk over 2025 (396,6 / 1.655,0). Eén
valuta: alles in USD, de rapportage- én noteringsvaluta, dus er is geen
wisselkoers in het spel.*

### WACC-componenten
- **Risicovrije rente %**: 4,69 — Amerikaanse staatsobligatie 10 jaar
- **Bron risicovrije rente**: FRED, serie DGS10, waarde per 6 augustus 2026
- **Type**: spot, nominaal
- **ERP %**: 4,28
- **Bron ERP**: Aswath Damodaran, implied equity risk premium S&P 500 per 1 augustus 2026
- **Beta (adjusted, Blume)**: 0,81
- **Bron beta**: StockAnalysis 5-jaars beta 0,71, gecorrigeerd volgens Blume (0,67 × raw + 0,33)
- **Type beta**: 5 jaar maandelijks, regressie — toegestaan omdat Fastenal 39 jaar genoteerd is en met een gemiddeld dagvolume van 10,8 mln aandelen ruimschoots liquide is
- **Country risk premium %**: 0,00
- **Size premium %**: 0,00
- **Cost of equity %**: 8,14
- **Schuldkosten na belasting %**: 3,09
- **E/V gewicht %**: 99,80
- **D/V gewicht %**: 0,20
- **WACC %**: **8,13**
- **Sector WACC % (Damodaran, "Retail (Distributors)", januari 2026)**: 7,22
- **Illiquiditeitskorting %**: null (niet van toepassing)

**Spot versus genormaliseerde rente.** De Amerikaanse tienjaarsrente van 4,69%
ligt duidelijk boven het gemiddelde van het afgelopen decennium, dat door de
jaren 2016-2021 met rentes rond 2% flink omlaag wordt getrokken. Ik gebruik
niettemin de spotrente, om twee redenen: het is de rente waartegen een belegger
vandaag daadwerkelijk kan beleggen, en Damodaran's ERP van 4,28% is
mét die rente geconsistificeerd — de twee horen bij elkaar en mogen niet
gemengd worden met een historisch gemiddelde. De gevoeligheidsmatrix hieronder
loopt door tot een WACC van 6,5%, wat een genormaliseerde rente ruimschoots
dekt; ook daar blijft de fair value ver onder de koers.

### DCF model-specs
- **Model type**: 2-fase (5 + 5 jaar) met Gordon-eindwaarde
- **FCF-definitie**: FCFF — vrije kasstroom naar de onderneming, verdisconteerd tegen de WACC, waarna de nettokaspositie wordt opgeteld
- **Basis FCF**: 960,7 (genormaliseerd, mid-cyclus)
- **Basis FCF na SBC**: 960,7 (de normalisatie is uitgevoerd op de reeks ná SBC)
- **FCF-type**: "Genormaliseerde FCF 960,7 mln (mid-cyclus)" — mediane FCF-marge na SBC over 2016-2025 van 10,98% toegepast op de TTM-omzet van 8.749,4
- **Groei fase 1 %** (jaar 1-5): 8,0
- **Groei fase 2 %** (jaar 6-10): 5,0
- **Terminal groei %**: 2,5
- **Terminal methode**: Gordon growth, met kruiscontrole via exit multiple
- **Exit multiple gebruikt (EV/EBITDA)**: 15,0
- **Bron exit multiple**: mediaan van de peer-groep over de huidige cyclus — Grainger 20,78, Applied Industrial 22,70, MSC Industrial 17,06; ik gebruik bewust 15,0 en niet de mediaan van 20,78, omdat de peers vandaag zelf op historisch hoge multiples staan en een eindwaarde over tien jaar op een cyclisch topniveau ongeloofwaardig is
- **Terminal value Gordon growth**: 32.802,4
- **Terminal value exit multiple**: 54.972,8 (15,0 × geprojecteerde EBITDA jaar 10 van 3.664,9)
- **Terminal value % van totaal**: 62,1 (ruim onder de grens van 75%)
- **Terminal implied EV/EBITDA**: 8,95
- **Terminal groei consistentie**: 2,5% eindwaardegroei vereist bij een
  volwassen ROIC van 20% een herinvesteringsvoet van 12,5% — plausibel voor een
  distributeur met een historische kapitaalintensiteit van circa 3% van de
  omzet. De 2,5% ligt bovendien ruim onder de Amerikaanse nominale
  BBP-groei op lange termijn, zoals de methodiek voorschrijft.
- **Mid-year convention**: true
- **Aandelen uitstaand (mln)**: 1.147,5
- **Nettoschuld huidig**: −84,7 (nettokaspositie)

### Cycliciteitscheck (verplicht vóór de FCF-keuze)

**Is Fastenal cyclisch?** Gedeeltelijk, en dat verdient een genuanceerd
antwoord. De eindmarkten zijn onmiskenbaar cyclisch: 75,9% van de omzet komt uit
de maakindustrie en 8,1% uit niet-residentiële bouw, allebei sectoren die de
methodiek expliciet als cyclisch aanmerkt. De omzet krompt in echte recessies —
in 2009 met 17,5% — en groeide in de zwakke jaren 2016 en 2024 met slechts 2,4%
en 2,7%. Maar de winstgevendheid gedraagt zich niet cyclisch: de bedrijfsmarge
bleef tien jaar lang tussen 19,8% en 20,8%, inclusief een pandemie en twee
industriële inzinkingen. De volatiliteit in de vrije kasstroom komt vrijwel
volledig uit werkkapitaal, niet uit marges.

Ik pas daarom de normalisatieregels toe, met de aantekening dat het verschil
klein is:
- **Piek-FCF na SBC (2016-2025)**: 1.252,6 (2023, voorraadafbouw)
- **Dal-FCF na SBC**: 320,4 (2016)
- **Gemiddelde FCF na SBC**: 739,3
- **Mediane FCF-marge na SBC**: 10,98%
- **Gekozen genormaliseerde FCF**: 960,7 (10,98% × TTM-omzet 8.749,4)
- **Afwijking ten opzichte van de meest recente FCF (2025: 1.042,2)**: −7,8%

Die afwijking blijft ruim onder de 20% die een rodevlagmelding vereist, en de
margecontrole uit regel 4 is per constructie sluitend omdat de genormaliseerde
FCF juist als mediane marge maal huidige omzet is berekend. Het startpunt is dus
niet gemanipuleerd: het ligt 7,8% onder het laatst gerapporteerde jaar, wat
gerechtvaardigd is omdat de capexdoelstelling voor 2026 (3,5% van de omzet)
boven de historische mediaan van 2,98% ligt.

### Normalisatie van de kasbelasting (REGEL 5, verplicht)

Twee van de drie signalen die de methodiek noemt, treden bij Fastenal op: de
FCF-conversie kwam in 2020 en 2023 boven 100% uit (108,7% en 109,1%) en de
accruals werden in 2023 sterk negatief (−6,2%). De controle is daarom uitgevoerd
en niet optioneel.

| Jaar | Belastinglast W&V | Betaalde winstbelasting | Verschil % |
|---|---|---|---|
| 2021 | 282,8 | 294,0 | +4,0 |
| 2022 | 353,1 | 354,1 | +0,3 |
| 2023 | 367,0 | 383,0 | +4,4 |
| 2024 | 357,5 | 356,5 | −0,3 |
| 2025 | 396,6 | 398,8 | +0,6 |
| **2021-2025 cumulatief** | **1.757,0** | **1.786,4** | **+1,7** |

**Conclusie: geen correctie nodig.** Fastenal betaalt cumulatief 1,7% méér
belasting dan het als last boekt — het tegenovergestelde van een
belastingvakantie. Er zijn geen voorwaartse verliezen, geen versnelde
afschrijvingsvoordelen en geen eenmalige teruggaven die de vrije kasstroom
opblazen. De FCF-basis van 960,7 is dus niet naar boven vertekend door een
tijdelijk lage kasbelasting, en de oorzaak van de hoge FCF-conversie in 2020 en
2023 ligt volledig bij werkkapitaal (respectievelijk voorraadbeheersing tijdens
de pandemie en voorraadafbouw daarna). De omvang van de correctie is nul en dat
staat hiermee vermeld, zoals de regel vereist.

### DCF-toelichting

Ik reken met de vrije kasstroom naar de onderneming — operationele kasstroom min
investeringen min aandelencompensatie — en verdisconteer die tegen de WACC van
8,13%, waarna de nettokaspositie van USD 84,7 mln erbij komt en het geheel door
1.147,5 mln aandelen wordt gedeeld. Die combinatie is methodisch de enige
correcte; FCFE tegen WACC zou de uitkomst systematisch scheeftrekken. De
mid-year convention is toegepast omdat kasstromen gemiddeld halverwege het jaar
binnenkomen, wat de uitkomst met ongeveer 4% verhoogt. Het startpunt is niet de
meest recente kasstroom maar de mediane FCF-marge na SBC over tien jaar,
toegepast op de omzet van de laatste twaalf maanden — 7,8% onder 2025. De
eindwaarde is 62,1% van de ondernemingswaarde: hoog, maar onder de grens van 75%
en normaal voor een bedrijf met lage kapitaalintensiteit. De impliciete
EV/EBITDA die uit het Gordon-model rolt is 8,95 in jaar tien, tegenover de 30,4
waarop het aandeel nu handelt.

### 5-jaars projectie (basisscenario)

| Jaar | FCF (USD mln) | Contante waarde |
|---|---|---|
| 2027 | 1.037,6 | 997,8 |
| 2028 | 1.120,6 | 996,6 |
| 2029 | 1.210,2 | 995,4 |
| 2030 | 1.307,0 | 994,2 |
| 2031 | 1.411,6 | 993,0 |
| 2032 | 1.482,2 | 964,3 |
| 2033 | 1.556,3 | 936,4 |
| 2034 | 1.634,1 | 909,3 |
| 2035 | 1.715,8 | 883,0 |
| 2036 | 1.801,6 | 857,4 |

*De projectie is op kasstroomniveau opgesteld en niet op omzet- en EBIT-niveau,
omdat het DCF-model met een genormaliseerde FCF-marge werkt. Een aparte
omzet-, NOPAT- en werkkapitaalprojectie zou dezelfde uitkomst met meer
schijnprecisie presenteren.*

### Scenarios

| Scenario | FCF-groei % (fase 1 / fase 2 / eindwaarde) | WACC % | Fair value | Upside % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 4,0 / 2,0 / 2,00 | 8,88 | 14,19 | −72,63 | 25 |
| Basis | 8,0 / 5,0 / 2,50 | 8,13 | 21,98 | −57,60 | 50 |
| Optimistisch | 11,0 / 7,0 / 3,00 | 7,88 | 30,03 | −42,08 | 25 |

- **Kansgewogen fair value**: **22,04**

**Onderbouwing van de kansverdeling.** De standaardverdeling 25/50/25 is
aangehouden. Er zijn geen binaire katalysatoren die de verdeling asymmetrisch
maken, de datakwaliteit is uitstekend (volledige tienjaarshistorie uit
jaarrekeningen) en Fastenal bevindt zich noch aan de top noch aan de bodem van
de industriële cyclus — de PMI stond in Q2 2026 net boven 53. Het pessimistische
scenario veronderstelt een industriële inzinking waarin de kasstroomgroei
terugvalt naar 4% en de risicopremie oploopt; het optimistische scenario
extrapoleert de huidige 14,7% omzetgroei naar 11% kasstroomgroei over vijf jaar,
wat boven de historische tienjaars-FCF-CAGR van 10,4% ligt.

### Reverse DCF
- **Impliciete groei %**: 21,63 in jaar 1-5, gevolgd door 13,52% in jaar 6-10 (bij een eindwaardegroei van 2,5% en een WACC van 8,13%)
- **Historische FCF CAGR %**: 10,44 (2015-2025, na SBC)
- **Consensus groei % (analisten)**: 15,77 winstgroei voor 2026 (18 analisten)
- **Interpretatie:** Om de koers van 51,84 te rechtvaardigen moet de vrije
kasstroom vijf jaar lang met 21,6% per jaar groeien en daarna nog vijf jaar met
13,5%. Fastenal deed dat de afgelopen tien jaar niet: de kasstroom groeide met
10,4% per jaar, de omzet met 7,8%. Zelfs de analistenconsensus blijft met 15,77%
winstgroei ver onder wat de koers inprijst — en dat is één jaar, geen tien. Er is
een tweede lezing van dezelfde boodschap: bij deze koers en een discontovoet van
8,13% zou de vrije kasstroom eeuwig met 6,4% per jaar moeten groeien, sneller dan
de Amerikaanse economie in nominale termen.

### EPV (Bruce Greenwald)
- **Genormaliseerde EBIT-marge %**: 20,24 (gemiddelde over 2016-2025)
- **Genormaliseerde EBIT**: 1.771,1
- **Genormaliseerde NOPAT**: 1.346,6
- **Maintenance capex**: 179,2 (gelijkgesteld aan de afschrijving en amortisatie van 2025)
- **Adjusted earnings power**: 1.346,6
- **EPV**: 16.564,9
- **EPV per aandeel**: **14,51**
- **Groeipremie %**: 51,5 (de DCF-basiswaarde van 21,98 ligt 51,5% boven de EPV)
- **Koers ten opzichte van EPV**: +257,3%

De EPV is de waarde van Fastenal zonder énige groei: puur de huidige
verdiencapaciteit, eeuwigdurend, tegen de kapitaalkosten. Die komt uit op 14,51
per aandeel. De koers van 51,84 ligt daar 257% boven, wat betekent dat ruim
zeventig procent van de beurswaarde uitsluitend berust op groei die nog moet
komen. Dat is op zichzelf niet onredelijk voor een bedrijf met een
ROIC-WACC-spread van 26 procentpunt — groei is bij zo'n spread echt veel waard —
maar het maakt de belegger volledig afhankelijk van het uitkomen van die groei.
De DCF-basiswaarde van 21,98 kent een groeipremie van 51,5% boven de EPV; de
markt kent er een van 257% toe.

### Andere methoden
- **DDM uitgevoerd?** false — Fastenal keert weliswaar structureel dividend uit,
maar met een FCF-payout van 95,6% en een dividendrendement van 1,93% zou een
dividendgroeimodel dezelfde informatie geven als de DCF, met meer aannames.
- **SOTP uitgevoerd?** false — één segment, geen conglomeraat.

### Synthese fair value

| Methode | Uitkomst per aandeel | Gewicht |
|---|---|---|
| DCF basisscenario | 21,98 | 55% |
| EPV (no-growth) | 14,51 | 15% |
| Relatieve waardering (gemiddelde van 31,60 en 35,46) | 33,53 | 30% |

*Relatieve waardering, twee ankers: (a) het tienjaars gemiddelde P/E van 27,01
maal de TTM-winst per aandeel van 1,17 geeft 31,60; (b) de peer-mediaan
EV/EBITDA van 20,78 (Grainger) maal de TTM-EBITDA van 1.954,3 plus de
nettokaspositie, gedeeld door 1.147,5 aandelen, geeft 35,46.*

- **Bandbreedte laag**: 14,19 (pessimistisch DCF)
- **Bandbreedte centraal**: **24,33**
- **Bandbreedte hoog**: 30,03 (optimistisch DCF)
- **Methode-gewichten**: DCF 55% / EPV 15% / Multiples 30%
- **Margin of safety vereist %**: 20
- **Koopniveau** (24,33 × 0,80): **19,46**
- **Synthese-toelichting:** De drie methoden wijzen dezelfde kant op, met
verschillende afstanden. De kasstroommodellen zijn het strengst: de DCF komt op
21,98 en de EPV op 14,51, waarmee ruim zeventig procent van de beurswaarde op
toekomstige groei berust. De relatieve waardering is milder omdat zij het aandeel
tegen zijn eigen verleden en zijn concurrenten afzet: 31,60 op het tienjaars
gemiddelde P/E, 35,46 op de peer-multiple. Ik weeg de DCF zwaar (55%) wegens de
uitstekende datakwaliteit, de EPV licht (15%) omdat zij bij deze ROIC-spread te
streng is, en de multiples met 30%. Dat geeft 24,33 centraal. Het aandeel wordt
interessant onder 19,46 — een niveau waarop het sinds 2019 niet heeft gehandeld.

### Gevoeligheid (DCF)

Fair value per aandeel bij variatie in groei fase 1 (rijen; fase 2 telkens
62,5% daarvan) en WACC (kolommen), eindwaardegroei constant op 2,5%:

| Groei fase 1 \ WACC | 6,5% | 7,0% | 7,5% | 8,13% | 8,75% | 9,5% |
|---|---|---|---|---|---|---|
| 4% | 23,76 | 21,16 | 19,09 | 16,94 | 15,35 | 13,75 |
| 6% | 27,21 | 24,19 | 21,77 | 19,26 | 17,42 | 15,56 |
| 8% | 31,14 | 27,62 | 24,81 | **21,90** | 19,76 | 17,60 |
| 10% | 35,60 | 31,52 | 28,26 | 24,89 | 22,40 | 19,90 |
| 12% | 40,67 | 35,94 | 32,16 | 28,26 | 25,39 | 22,50 |

*De vetgedrukte cel is het basisscenario. Het kleine verschil met de 21,98 uit
de scenariotabel komt doordat in de matrix fase 2 mechanisch op 62,5% van fase 1
is gezet (dus 5,0% bij 8%) terwijl het basisscenario 5,0% expliciet vastlegt;
het effect is 0,08 per aandeel.*

De matrix maakt het punt scherper dan welke toelichting ook: zelfs in de
gunstigste hoek — 12% kasstroomgroei vijf jaar lang bij een WACC van 6,5%, dus
een groei boven het historische record én een rente die ver onder de huidige
ligt — komt de fair value niet boven 40,67 uit. De koers van 51,84 valt buiten
de hele matrix.

---

## 13. Databronnen

### Bronnen-hiërarchie
- **Jaarverslag / SEC-filing (10-K, inclusief XBRL en R-pagina's)** → **HOOG**
- **Beursmelding / persbericht kwartaalcijfers op de IR-site** → **HOOG**
- **Aggregator (StockAnalysis, MacroTrends)** → **AGGREGATOR**

### Financiële bronnen (10 jaar historie)

| Jaar | Bron | URL | Betrouwbaarheid |
|---|---|---|---|
| 2016 | Fastenal Form 10-K FY2016 (via XBRL companyconcept, form=10-K, fy=2016) | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/OperatingIncomeLoss.json | HOOG |
| 2017 | Fastenal Form 10-K FY2017 (via XBRL companyconcept) | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/OperatingIncomeLoss.json | HOOG |
| 2018 | Fastenal Form 10-K FY2018 (via XBRL companyconcept) | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/NetIncomeLoss.json | HOOG |
| 2019 | Fastenal Form 10-K FY2019 (via XBRL companyconcept) | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/NetCashProvidedByUsedInOperatingActivities.json | HOOG |
| 2020 | Fastenal Form 10-K FY2020 (via XBRL companyconcept) | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/Assets.json | HOOG |
| 2021 | Fastenal Form 10-K FY2021 (via XBRL companyconcept) | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/StockholdersEquity.json | HOOG |
| 2022 | Fastenal Form 10-K FY2022 | https://www.sec.gov/Archives/edgar/data/815556/000081555622000009/0000815556-22-000009-index.htm | HOOG |
| 2023 | Fastenal Form 10-K FY2023 | https://www.sec.gov/Archives/edgar/data/815556/000081555623000009/0000815556-23-000009-index.htm | HOOG |
| 2024 | Fastenal Form 10-K FY2024 | https://www.sec.gov/Archives/edgar/data/815556/000081555625000065/0000815556-25-000065-index.htm | HOOG |
| 2025 | Fastenal Form 10-K FY2025 | https://www.sec.gov/Archives/edgar/data/815556/000081555626000009/fast-20251231.htm | HOOG |

*Aanvullend zijn 2013, 2014 en 2015 op HOOG gedekt via dezelfde
XBRL-companyconcept-endpoints (form=10-K), zodat de reeks dertien jaar beslaat.
Alle vijf meest recente jaren (2021-2025) staan op HOOG; er is geen enkel jaar
op AGGREGATOR gezet en er is geen openstaande haallijst.*

### Jaarverslagen geraadpleegd

| Jaar | Bron | URL |
|---|---|---|
| 2025 | Form 10-K, hoofddocument (bedrijfsprofiel, personeel, klantconcentratie, concurrentie, risicofactoren) | https://www.sec.gov/Archives/edgar/data/815556/000081555626000009/fast-20251231.htm |
| 2025 | Form 10-K, R3 — geconsolideerde balans | https://www.sec.gov/Archives/edgar/data/815556/000081555626000009/R3.htm |
| 2025 | Form 10-K, R8 — geconsolideerd kasstroomoverzicht | https://www.sec.gov/Archives/edgar/data/815556/000081555626000009/R8.htm |
| 2025 | Form 10-K, R34 — omzet per geografisch gebied | https://www.sec.gov/Archives/edgar/data/815556/000081555626000009/R34.htm |
| 2025 | Form 10-K, R35 — omzetaandeel per eindmarkt | https://www.sec.gov/Archives/edgar/data/815556/000081555626000009/R35.htm |
| 2025 | Form 10-K, R36 — omzetaandeel per productlijn | https://www.sec.gov/Archives/edgar/data/815556/000081555626000009/R36.htm |
| 2024-2022 | Form 10-K's FY2024, FY2023, FY2022 (filing-indexen) | zie tabel hierboven |

### Beursmeldingen geraadpleegd

| Datum | Omschrijving | URL |
|---|---|---|
| 2026-07-14 | Fastenal Company Reports 2026 Second Quarter Earnings | https://investor.fastenal.com/news-releases/news-details/2026/Fastenal-Company-Reports-2026-Second-Quarter-Earnings/default.aspx |
| 2026-07 | Verklaring kwartaaldividend USD 0,26 en inkoopactiviteit Q2 2026 | https://www.stocktitan.net/news/FAST/fastenal-company-announces-cash-dividend-and-share-repurchase-h895l3b5hlz8.html |
| 2026-07-21 | Transcript conference call Q2 2026 (tarieven, capex-guidance, digitale doelstelling) | https://www.fool.com/earnings/call-transcripts/2026/07/21/fastenal-fast-q2-2026-earnings-call-transcript/ |
| 2026-01-19 | Fastenal Company Reports 2025 Annual and Fourth Quarter Earnings | https://s23.q4cdn.com/591718779/files/doc_financials/2025/Q4/EX_99-1-12-31-2025-Earnings-Release-1-19-R8_FINAL.pdf |
| 2025-11-03 | Fastenal Names New Chief Financial Officer (Max Tunnicliff) | https://s23.q4cdn.com/591718779/files/doc_financials/2025/Q4/EX_99-1-10-30-2025-CFO_Final-11-03-2025-9-02am.pdf |
| 2025 | Fastenal Announces CEO Transition (Watts volgt Florness per 16-07-2026) | https://investor.fastenal.com/news-releases/news-details/2025/Fastenal-Announces-CEO-Transition/default.aspx |
| 2025-07-14 | Fastenal Company Reports 2025 Second Quarter Earnings | https://investor.fastenal.com/news-releases/news-details/2025/Fastenal-Company-Reports-2025-Second-Quarter-Earnings/default.aspx |
| 2026-04-14 | Verslag Q1 2026-call over tarieven en prijszetting | https://www.digitalcommerce360.com/2026/04/14/fastenal-tariffs-war-impact-q1-fy26/ |

### IPO-prospectus
- **Geraadpleegd?** false
- **URL**: n.v.t. — de beursgang dateert van 1987 en valt ruim buiten het
  elektronische EDGAR-archief, dat in 1993-1996 begon.
- **Pre-IPO data beschikbaar?** false
- **Pre-IPO bron**: geen. Dit is geen tekortkoming: de methodiek eist
  pre-IPO-analyse voor bedrijven die korter dan vijf jaar genoteerd zijn.
  Fastenal is 39 jaar genoteerd en er is een ononderbroken reeks van
  gecontroleerde jaarrekeningen beschikbaar.

### Non-GAAP
- **Gebruikt?** false
- **Toelichting**: Fastenal publiceert geen adjusted of underlying cijfers. Alle
  bedragen in dit rapport zijn US-GAAP zoals gerapporteerd. De enige bewerkingen
  die ik zelf heb uitgevoerd zijn (a) de split-correctie van per-aandeel-getallen
  naar de huidige basis en (b) de aftrek van SBC van de vrije kasstroom, beide
  expliciet benoemd en herleidbaar.

### Ontbrekende data
- **Amortisatie van immateriële activa vóór 2023** is niet als aparte XBRL-post
  beschikbaar in de oudere 10-K's. De EBITDA-reeks vóór 2023 bestaat daarom uit
  EBIT plus afschrijving zonder de circa USD 10,7 mln amortisatie; het effect op
  de EBITDA-marge is minder dan 0,2 procentpunt en heeft geen invloed op enige
  conclusie.
- **Onsite-locaties**: Fastenal rapporteert het aantal actieve
  Onsite-vestigingen niet meer als afzonderlijk kengetal in het 10-K over 2025
  of in het Q2-2026-persbericht; het gaat op in de rapportage per
  klantbestedingscategorie. Ik vul daarom geen Onsite-aantal in.
- **TAM en SAM** voor Noord-Amerikaanse MRO-distributie: geen bron die aan de
  bronneneisen voldoet. Velden leeg gelaten.
- **Marktaandelen** van Fastenal en zijn concurrenten: geen verifieerbare bron.
  Kolom weggelaten uit de concurrentietabel.
- **IPO-datum en -koers 1987**: het jaar is verifieerbaar, de exacte datum en de
  introductiekoers niet uit een bron die aan de eisen voldoet. Weggelaten.
- **Splitsing Canada versus Mexico** in de omzettabel: het 10-K rapporteert deze
  twee landen als één regio. Een aparte country risk premium per land is daarom
  niet te wegen; de CRP is op nul gezet met de motivering in sectie 12.
- **CEO-beloning in absolute bedragen en de CEO pay ratio**: staan in de
  proxy statement (DEF 14A), die voor deze analyse niet is geopend. De
  beoordeling van de beloningsstructuur steunt daarom op de wél verifieerbare
  SBC-omvang en de verwateringsontwikkeling, niet op de bedragen per persoon.
- **Werkelijk gemiddelde van de Amerikaanse tienjaarsrente over 2016-2026**: niet
  als één gepubliceerd cijfer opgehaald; de keuze voor de spotrente is in
  sectie 12 gemotiveerd en de gevoeligheidsmatrix dekt lagere rentes af.

### Peildatum analyse
- 2026-08-07 (slotkoers NASDAQ USD 51,84)

---

## 14. Volledige bronnen-lijst

| Titel | URL | Type |
|---|---|---|
| Fastenal Company Form 10-K 2025 | https://www.sec.gov/Archives/edgar/data/815556/000081555626000009/fast-20251231.htm | jaarverslag |
| Fastenal 10-K 2025 — geconsolideerde balans (R3) | https://www.sec.gov/Archives/edgar/data/815556/000081555626000009/R3.htm | jaarverslag |
| Fastenal 10-K 2025 — kasstroomoverzicht (R8) | https://www.sec.gov/Archives/edgar/data/815556/000081555626000009/R8.htm | jaarverslag |
| Fastenal 10-K 2025 — omzet per regio (R34) | https://www.sec.gov/Archives/edgar/data/815556/000081555626000009/R34.htm | jaarverslag |
| Fastenal 10-K 2025 — eindmarktmix (R35) | https://www.sec.gov/Archives/edgar/data/815556/000081555626000009/R35.htm | jaarverslag |
| Fastenal 10-K 2025 — productmix (R36) | https://www.sec.gov/Archives/edgar/data/815556/000081555626000009/R36.htm | jaarverslag |
| SEC EDGAR — overzicht 10-K-filings Fastenal (CIK 0000815556) | https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000815556&type=10-K | beurswebsite |
| SEC XBRL companyconcept — omzet (RevenueFromContractWithCustomer) | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/RevenueFromContractWithCustomerExcludingAssessedTax.json | jaarverslag |
| SEC XBRL companyconcept — omzet (SalesRevenueNet, oudere jaren) | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/SalesRevenueNet.json | jaarverslag |
| SEC XBRL companyconcept — brutowinst | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/GrossProfit.json | jaarverslag |
| SEC XBRL companyconcept — bedrijfsresultaat | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/OperatingIncomeLoss.json | jaarverslag |
| SEC XBRL companyconcept — nettowinst | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/NetIncomeLoss.json | jaarverslag |
| SEC XBRL companyconcept — operationele kasstroom | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/NetCashProvidedByUsedInOperatingActivities.json | jaarverslag |
| SEC XBRL companyconcept — investeringen in materiële vaste activa | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/PaymentsToAcquirePropertyPlantAndEquipment.json | jaarverslag |
| SEC XBRL companyconcept — aandelencompensatie | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/ShareBasedCompensation.json | jaarverslag |
| SEC XBRL companyconcept — afschrijving | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/Depreciation.json | jaarverslag |
| SEC XBRL companyconcept — afschrijving en amortisatie | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/DepreciationDepletionAndAmortization.json | jaarverslag |
| SEC XBRL companyconcept — totale activa | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/Assets.json | jaarverslag |
| SEC XBRL companyconcept — eigen vermogen | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/StockholdersEquity.json | jaarverslag |
| SEC XBRL companyconcept — liquide middelen | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/CashAndCashEquivalentsAtCarryingValue.json | jaarverslag |
| SEC XBRL companyconcept — schuld | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/LongTermDebt.json | jaarverslag |
| SEC XBRL companyconcept — vlottende activa | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/AssetsCurrent.json | jaarverslag |
| SEC XBRL companyconcept — kortlopende verplichtingen | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/LiabilitiesCurrent.json | jaarverslag |
| SEC XBRL companyconcept — voorraad | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/InventoryNet.json | jaarverslag |
| SEC XBRL companyconcept — belastinglast | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/IncomeTaxExpenseBenefit.json | jaarverslag |
| SEC XBRL companyconcept — betaalde winstbelasting | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/IncomeTaxesPaidNet.json | jaarverslag |
| SEC XBRL companyconcept — verwaterde aandelen | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/WeightedAverageNumberOfDilutedSharesOutstanding.json | jaarverslag |
| SEC XBRL companyconcept — dividend per aandeel (kas) | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/CommonStockDividendsPerShareCashPaid.json | jaarverslag |
| SEC XBRL companyconcept — betaald dividend | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/PaymentsOfDividends.json | jaarverslag |
| SEC XBRL companyconcept — inkoop eigen aandelen | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/PaymentsForRepurchaseOfCommonStock.json | jaarverslag |
| SEC XBRL companyconcept — operationele leaseverplichting | https://data.sec.gov/api/xbrl/companyconcept/CIK0000815556/us-gaap/OperatingLeaseLiability.json | jaarverslag |
| Fastenal Company Reports 2026 Second Quarter Earnings | https://investor.fastenal.com/news-releases/news-details/2026/Fastenal-Company-Reports-2026-Second-Quarter-Earnings/default.aspx | beursmelding |
| Fastenal Company Reports 2025 Annual and Fourth Quarter Earnings | https://s23.q4cdn.com/591718779/files/doc_financials/2025/Q4/EX_99-1-12-31-2025-Earnings-Release-1-19-R8_FINAL.pdf | beursmelding |
| Fastenal Company Reports 2025 Second Quarter Earnings | https://investor.fastenal.com/news-releases/news-details/2025/Fastenal-Company-Reports-2025-Second-Quarter-Earnings/default.aspx | beursmelding |
| Fastenal Company Announces Cash Dividend and Share Repurchase Activity (juli 2026) | https://www.stocktitan.net/news/FAST/fastenal-company-announces-cash-dividend-and-share-repurchase-h895l3b5hlz8.html | beursmelding |
| Fastenal Announces CEO Transition | https://investor.fastenal.com/news-releases/news-details/2025/Fastenal-Announces-CEO-Transition/default.aspx | beursmelding |
| Fastenal Names New Chief Financial Officer (Max Tunnicliff) | https://s23.q4cdn.com/591718779/files/doc_financials/2025/Q4/EX_99-1-10-30-2025-CFO_Final-11-03-2025-9-02am.pdf | beursmelding |
| Fastenal (FAST) Q2 2026 Earnings Call Transcript | https://www.fool.com/earnings/call-transcripts/2026/07/21/fastenal-fast-q2-2026-earnings-call-transcript/ | nieuwsartikel |
| Tariffs, war impact Fastenal growth in Q1 (Digital Commerce 360) | https://www.digitalcommerce360.com/2026/04/14/fastenal-tariffs-war-impact-q1-fy26/ | nieuwsartikel |
| Fastenal founder Bob Kierlin dies at 85 (Modern Distribution Management) | https://www.mdm.com/news/top-distributor-sectors/contractor/fastenal-founder-bob-kierlin-dies-at-85/ | nieuwsartikel |
| Fastenal — Wikipedia (bedrijfsgeschiedenis, mijlpalen) | https://en.wikipedia.org/wiki/Fastenal | nieuwsartikel |
| With 86% ownership in Fastenal, institutional investors have a lot riding on the business | https://finance.yahoo.com/news/86-ownership-fastenal-company-nasdaq-130019227.html | nieuwsartikel |
| Fastenal insider transactions (Insider Monitor, Form 4-samenvatting) | https://www.insider-monitor.com/trading/cik815556.html | databron |
| Fastenal institutionele aandeelhouders (Fintel) | https://fintel.io/so/us/fast | databron |
| StockAnalysis — Fastenal overzicht (koers, marktkap, beta) | https://stockanalysis.com/stocks/fast/ | databron |
| StockAnalysis — Fastenal statistieken | https://stockanalysis.com/stocks/fast/statistics/ | databron |
| StockAnalysis — Fastenal analistenverwachtingen | https://stockanalysis.com/stocks/fast/forecast/ | analistenrapport |
| StockAnalysis — W.W. Grainger statistieken | https://stockanalysis.com/stocks/gww/statistics/ | databron |
| StockAnalysis — MSC Industrial statistieken | https://stockanalysis.com/stocks/msm/statistics/ | databron |
| StockAnalysis — Applied Industrial Technologies statistieken | https://stockanalysis.com/stocks/ait/statistics/ | databron |
| MacroTrends — Fastenal historische koers-winstverhouding | https://www.macrotrends.net/stocks/charts/FAST/fastenal/pe-ratio | databron |
| FRED — 10-jaars Amerikaanse staatsobligatie (DGS10) | https://fred.stlouisfed.org/series/DGS10 | databron |
| Damodaran — implied equity risk premium (1 augustus 2026) | https://pages.stern.nyu.edu/~adamodar/New_Home_Page/home.htm | onderzoeksrapport |
| Damodaran — kapitaalkosten per sector (januari 2026) | https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/wacc.html | onderzoeksrapport |

---

## 15. Update-historie

| Datum | Versie | Wijziging |
|---|---|---|
| 2026-08-08 | 1.0 | Eerste publicatie. Volledige dertienjaarshistorie (2013-2025) uit SEC-jaarrekeningen, geen enkel jaar op aggregator. |

---

## Opmerkingen voor Claude Code

1. **Bron-dekking is volledig HOOG.** Alle dertien jaren komen uit 10-K's via de
   XBRL-companyconcept-feed. Er staat niets op de haallijst en er is geen enkele
   cel geschat. De URL's in sectie 13 bevatten "sec.gov" en eindigen op ".json"
   of ".htm" — als `check-sources.py` op de tekenreeksen
   "jaarverslag"/"annual"/"investor" of op ".pdf" test, zullen de
   XBRL-URL's mogelijk niet matchen terwijl ze wel degelijk HOOG zijn. Overweeg
   "sec.gov/Archives" en "data.sec.gov/api/xbrl" aan het patroon toe te voegen.
2. **Split-correctie.** Elk per-aandeel-getal in dit rapport staat op de
   post-mei-2025 basis (2-voor-1 in mei 2019 én mei 2025). De omrekening is in
   de bronnen-inventaris toegelicht en gekruiscontroleerd tegen de MacroTrends
   EPS-reeks, die tot op de cent overeenkomt. Neem de reeks niet opnieuw uit een
   aggregator over zonder die controle.
3. **Schulddefinitie.** Gebruik de door Fastenal gerapporteerde rentedragende
   schuld (USD 120,0 mln per 30-6-2026), niet de USD 441,5 mln van
   StockAnalysis: die laatste telt operationele leases mee terwijl de huur al in
   de EBIT zit.
4. **Twee marginale rubric-uitkomsten, expliciet gemeld:**
   - *Magic Formula*: winstrendement 2,99% mist de 3%-drempel van regel 2 met
     een honderdste procentpunt en valt daardoor op 1. Bij de FY2025-EBIT is het
     2,79%, dus de uitkomst is niet gevoelig voor de definitie.
   - *Buffett / Munger*: regel 3 faalt uitsluitend op P/FCF (57,08 tegen een
     drempel van 30) terwijl de kwaliteitsvoorwaarden wél zijn vervuld, en regel 1
     is aantoonbaar onwaar. Conform de cascadeafspraak is 2 toegekend als
     restcategorie.
5. **Eindoordeel is dubbel geborgd.** Totaal 18 (< 24) én Fair Value DCF-score 1
   leiden onafhankelijk van elkaar tot PASS. Er is geen scenario waarin een
   herrekening van één framework het oordeel kantelt.
6. **Voor het rekenscript.** Het blok "DCF-invoeren" is compleet ingevuld. Mijn
   eigen uitkomst (basis 21,98; kansgewogen 22,04; EPV 14,51) is een
   tussenresultaat — als `dcf_calculator.py` meer dan 1% afwijkt, wint het
   script en hoort het verschil hier onder update-historie te komen.
