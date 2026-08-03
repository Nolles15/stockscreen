# Research: TMV — TeamViewer SE

> Stage-1 analyse (cowork). Output = uitsluitend dit bestand. JSON-conversie, validatie en deploy doet Claude Code (stage 2).

---

## Bronnen-inventaris (Stap 0.5)

**Belangrijke kanttekening vooraf (zie ook §13 en Opmerkingen voor Claude Code):**
Ik heb de officiële jaarverslag-PDF's van TeamViewer **niet** rechtstreeks kunnen openen in deze sessie (de PDF's en de meeste IR-detailpagina's konden niet betrouwbaar worden opgehaald; de sandbox-omgeving viel bovendien uit). De financiële tabellen 2021–2025 zijn daarom gebouwd op **S&P Global Market Intelligence** (via StockAnalysis.com), **kruisgecontroleerd** tegen TeamViewer's eigen persberichten (FY2024-omzet €671,4 mln, FY2025-omzet €747 mln, FY2025 nettowinst, adj. EBITDA, nettoschuld €901 mln — die matchen). Volgens METHODE.md tellen aggregator-bronnen voor de recente 5 jaren **niet** als HOOG. Conform de projectregel "liever aangeven dan een leeg rapport" vul ik de tabellen wél in op AGGREGATOR-niveau en markeer ik dat expliciet. **Stage 2 zou deze cijfers idealiter tegen de PDF-jaarverslagen moeten verifiëren voordat het verdict definitief wordt.**

```
Jaar 2025 — AGGREGATOR (kruisgecheckt met persbericht)
  Bron: S&P Global via StockAnalysis.com (income/balance/cashflow), bevestigd door
        TeamViewer FY2025-persbericht (10-2-2026) en HGB-jaarrekening 18-3-2026.
  URL:  https://stockanalysis.com/quote/etr/TMV/financials/
        https://www.eqs-news.com/news/corporate/teamviewer-delivers-on-fy-2025-pro-forma-topline-prognose...
  Daadwerkelijk geopend: ja (StockAnalysis); persbericht via web search bevestigd
  Cijfers overgenomen: omzet, brutowinst, EBIT, EBITDA, nettowinst, EPS, aandelen,
                       CFO, capex, FCF, SBC, nettoschuld, bruto schuld, goodwill, EV-componenten
  Cijfers NIET overgenomen: geen (PDF-verificatie ontbreekt — zie kanttekening)

Jaar 2024 — AGGREGATOR (kruisgecheckt met persbericht; omzet €671,4 mln bevestigd)
  Bron: S&P Global via StockAnalysis.com
  URL:  https://stockanalysis.com/quote/etr/TMV/financials/
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: idem 2025

Jaar 2023 — AGGREGATOR
  Bron: S&P Global via StockAnalysis.com
  URL:  https://stockanalysis.com/quote/etr/TMV/financials/
  Cijfers overgenomen: idem

Jaar 2022 — AGGREGATOR
  Bron: S&P Global via StockAnalysis.com
  URL:  https://stockanalysis.com/quote/etr/TMV/financials/
  Cijfers overgenomen: idem

Jaar 2021 — AGGREGATOR
  Bron: S&P Global via StockAnalysis.com
  URL:  https://stockanalysis.com/quote/etr/TMV/financials/
  Cijfers overgenomen: omzet, brutowinst, EBIT, EBITDA, nettowinst, EPS, aandelen,
                       CFO, capex, FCF, SBC, balans

Jaar 2020 — AGGREGATOR (alleen omzet betrouwbaar)
  Bron: Statista / companiesmarketcap (afgeleid van jaarverslag 2020)
  URL:  https://www.statista.com/statistics/1543342/teamviewer-revenue/
  Cijfers overgenomen: omzet (€455,6 mln). 
  Cijfers NIET overgenomen: volledige winst/kasstroom/balans-detail (niet betrouwbaar in één bron)

Jaar 2019 — AGGREGATOR (alleen omzet) — IPO-jaar (sep 2019)
  Bron: Statista
  URL:  https://www.statista.com/statistics/1543342/teamviewer-revenue/
  Cijfers overgenomen: omzet (€390,2 mln). Pre-/rond-IPO; balans vertekend.
  Cijfers NIET overgenomen: winst/kasstroom/balans (PE-gedreven kapitaalstructuur, niet vergelijkbaar)

Jaren 2015–2018 — GEEN BRON BESCHIKBAAR
  Zoekpoging(en): aggregators starten effectief bij 2019/2020 (IPO sep-2019);
                  pre-IPO was TeamViewer privaat onder Permira (overgenomen 2014).
  Conclusie: 2015–2018 blijven LEEG in alle tabellen. Genoteerd in ontbrekende_data.
```

---

## Metadata
- **Ticker (bare):** TMV
- **Yahoo symbol:** TMV.DE
- **Exchange:** XETRA (Frankfurt)
- **Sector (GICS-achtig):** Technologie
- **Industrie:** Software (remote connectivity / digital workplace)
- **Land:** Duitsland
- **Peildatum analyse:** 2026-06-11
- **Koers op peildatum:** 5,36
- **Valuta:** EUR
- **Marktkapitalisatie:** EUR 0,84 mld
- **Marktkap in mln (lokale valuta):** 842
- **Free float pct:** ~78
- **Indexlidmaatschap:** MDAX
- **Domein:** teamviewer.com

---

## 1. Executive summary

- **Kernthese:** TeamViewer is een Duits softwarebedrijf dat tot de wereldwijde standaard is uitgegroeid voor remote-access en remote-support: vanaf één apparaat veilig inloggen op en besturen van een ander apparaat, waar ook ter wereld. Het bedrijf bedient zowel kleine gebruikers (SMB) als grote ondernemingen, en breidt zijn platform uit richting "digital workplace" en augmented-reality-ondersteuning voor fabrieks- en buitendienstmedewerkers. De economische motor is hoogwaardig: een vrijwel volledig op abonnementen gebaseerd model met brutomarges rond 86% en vrije-kasstroommarges van 30–37%. De structurele groeidrivers zijn de verschuiving van losse remote-support naar geïntegreerde enterprise-platformen (TeamViewer One), hybride werken, en de eind 2024 aangekondigde overname van het Britse 1E (digital employee experience), die de aanwezigheid in Noord-Amerika en het enterprise-segment vergroot. Daar staat tegenover dat de groei is afgevlakt: het grote SMB-fundament groeit nauwelijks nog en kende eenmalige klant-churn, terwijl de enterprise-tak de motor moet worden. Het belangrijkste risico is de balans: de 1E-deal is met schuld gefinancierd, waardoor de nettoschuld sprong naar circa €901 mln en de nettowinst onder druk kwam door hogere rentelasten en valuta-effecten.
- **Oordeel:** HOLD
- **Fair value basis (kansgewogen, EUR):** 15,24
- **Fair value kansgewogen:** 15,24
- **EPV per aandeel (zonder groeipremie):** 5,61
- **Upside pct:** +184
- **Fair value scenarios:**

| Scenario | Fair value | Upside % | FCF groei % | WACC % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 10,50 | +96 | 0% → −1% | 9,0 | 25 |
| Basis | 14,49 | +170 | 2% → 1% | 8,5 | 50 |
| Optimistisch | 23,50 | +338 | 4% → 3% | 8,0 | 25 |

- **Reverse-DCF impliciete groei pct:** −3,1 (de markt prijst FCF-krimp in)
- **Grootste kans:** Als de enterprise-/1E-strategie aanslaat en de schuld wordt afgebouwd, herwaardeert een bedrijf met ~25% FCF-rendement fors.
- **Grootste risico:** Structurele SMB-erosie plus een met schuld beladen balans, waardoor de vrije kasstroom jarenlang naar rente en aflossing gaat in plaats van naar aandeelhouders.

---

## 2. Bedrijfsprofiel

- **Beschrijving:** TeamViewer SE maakt software waarmee gebruikers op afstand verbinding maken met en controle nemen over computers, servers, mobiele apparaten en industriële machines. Het kernproduct laat een IT-medewerker, helpdesk of monteur veilig inloggen op een apparaat elders ter wereld om het te bedienen, problemen op te lossen of onderhoud uit te voeren — zonder fysiek aanwezig te zijn. De software draait op vrijwel elk platform en is wereldwijd op meer dan twee miljard apparaten geïnstalleerd geweest. Het bedrijf zit aan het eind van de IT-waardeketen: het levert geen hardware of netwerk, maar de verbindings- en beheerlaag daarbovenop. Wat TeamViewer onderscheidt is de combinatie van gebruiksgemak, snelheid en beveiliging, plus een gigantische geïnstalleerde basis die in de loop der jaren is opgebouwd via gratis privégebruik dat als marketingtrechter naar betaalde zakelijke licenties fungeert. Het oplost een concreet probleem: organisaties moeten apparaten en mensen ondersteunen die overal verspreid zitten, en fysieke aanwezigheid is duur en traag. De omzet komt nagenoeg volledig uit terugkerende abonnementen (subscriptions), aangevuld met enterprise-contracten en, sinds de 1E-overname, digital-employee-experience-software die de prestaties en tevredenheid van werkplekken meet en automatisch verbetert.
- **Geschiedenis:** TeamViewer werd in 2005 opgericht in Göppingen (Duitsland); de eerste versie van de remote-software verscheen datzelfde jaar en werd snel populair doordat privégebruik gratis was, wat een enorme installed base creëerde. In 2014 nam de Britse private-equityfirma Permira het bedrijf over voor circa €870 mln. Onder Permira werd het bedrijf omgevormd van een licentiemodel naar een abonnementsmodel (subscription), wat de voorspelbaarheid van de omzet sterk verhoogde. Het keerpunt was de beursgang: op 25 september 2019 ging TeamViewer naar de beurs in Frankfurt in de grootste Europese IPO van dat jaar, met een waardering tot circa €5,5 mld — destijds meer dan 17× de billings. Permira verkocht in tranches en is inmiddels (najaar 2025) nagenoeg volledig uitgestapt (van >13% naar ~2,9%). De jaren na de IPO kenden tegenslagen: een veelbesproken en grotendeels mislukte sponsorstrategie in de sport (onder meer Manchester United en Mercedes F1) die later werd teruggeschroefd, plus een aandelenkoers die ver onder de IPO-prijs (€26,25) zakte. Het bedrijf reageerde met fors aandeleninkoopprogramma's, waarmee het aantal uitstaande aandelen daalde van ~201 mln (2021) naar ~157 mln (2025). De grootste strategische zet in de recente geschiedenis is de overname van het Britse 1E (digital employee experience), aangekondigd 10 december 2024 voor een ondernemingswaarde van USD 720 mln en begin 2025 afgerond — TeamViewers grootste acquisitie ooit, gefinancierd met schuld, gericht op het versterken van de enterprise-tak en de Noord-Amerikaanse aanwezigheid.
- **Bedrijfsmodel:** TeamViewer verdient vrijwel uitsluitend aan abonnementen (subscriptions) op zijn connectivity- en digital-workplace-software. Klanten betalen periodiek voor licenties die schalen met het aantal gebruikers, apparaten en functionaliteit. De omzet is daardoor grotendeels terugkerend (recurring), gemeten als Annual Recurring Revenue (ARR, ~€737 mln). Het model splitst grofweg in SMB (groot volume, klein bedrag per klant, hoog volume) en Enterprise (lagere aantallen, hogere contractwaarde, lagere churn). 1E voegt DEX-software toe (eveneens abonnement, vrijwel volledig enterprise).
- **IPO-context:** Beursgang 25 september 2019 in Frankfurt tegen €26,25 per aandeel, waardering tot ~€5,5 mld. Het was een exit-vehikel voor Permira: de IPO en latere tranches dienden om de PE-eigenaar te laten uitstappen, niet om vers kapitaal voor het bedrijf op te halen. De kapitaalstructuur kende bij IPO nog aanzienlijke schuld uit de Permira-overname; die is in de jaren erna deels afgebouwd, om met de 1E-deal in 2024/2025 weer fors op te lopen.
- **Klantprofiel:** Overwegend B2B, van zzp'ers en kleine IT-dienstverleners (SMB) tot grote multinationals (Enterprise). De klantenbasis is zeer breed en gefragmenteerd; er is geen sprake van afhankelijkheid van enkele grote klanten. Enterprise kent hogere retentie en hogere contractwaarde; SMB is volumineuzer maar gevoeliger voor prijs en macro.
- **Oprichtingsjaar:** 2005
- **IPO-datum:** 2019-09-25
- **IPO-koers (EUR):** 26,25
- **Personeel (FTE):** ~1.700 (TeamViewer) + ~300 (1E) — orde van grootte, niet PDF-geverifieerd
- **Landen actief:** wereldwijd; hoofdkantoor Göppingen (DE), sterke aanwezigheid VS, EMEA, APAC
- **Klantconcentratie:** Zeer laag — gefragmenteerde basis van honderdduizenden betalende abonnees; geen enkele klant materieel.

### Geografische spreiding (omzet)
| Regio | Omzet % | Valuta-exposure |
|---|---|---|
| EMEA | — | EUR (hoofdvaluta) |
| Americas | — | USD |
| APAC | — | diverse |

**Toelichting geografie:** TeamViewer rapporteert in EUR maar genereert een aanzienlijk deel van de omzet in USD en andere valuta; in Q1 2026 was er een FX-tegenwind van ~3,3 procentpunt op de gerapporteerde omzet. De exacte regioverdeling kon ik niet uit een geverifieerde bron halen (PDF ontbreekt) en laat ik daarom leeg in plaats van te schatten. De USD-exposure is door de 1E-overname (Noord-Amerika) toegenomen, wat zowel een natuurlijke hedge op de USD-schuld als een gevoeligheid voor de EUR/USD-koers met zich meebrengt.

### Segmenten
| Naam | Omzet % | Beschrijving |
|---|---|---|
| Enterprise | — | Grote klanten, hogere contractwaarde, lagere churn; ARR +8% cc in Q1 2026. Strategische groeimotor. |
| SMB | — | Groot volume kleine klanten; nauwelijks groei, gevoelig voor prijs/macro, eenmalige churn-effecten. |

*(Exacte omzet-% per segment niet PDF-geverifieerd → leeg gelaten i.p.v. geschat.)*

### Aandeelhouders (top 5)
| Naam | Belang % | Type (oprichter / PE / institutioneel / retail) |
|---|---|---|
| BlackRock | 5,17 | institutioneel |
| Norges Bank | 3,33 | institutioneel |
| Permira (TigerLuxOne / restant) | ~2,9 | PE (vrijwel uitgestapt) |
| Management board (collectief) | 1,84 | insiders |
| Free float (overig) | ~78 | retail + institutioneel |

- **Institutioneel eigendomstrend:** Permira is van >13% naar ~2,9% gezakt (najaar 2025) — de jarenlange PE-overhang is daarmee vrijwel verdwenen, wat het free float verhoogt en de aandelenstroom uit gedwongen verkopen wegneemt. Reguliere institutionele namen (BlackRock, Norges) zijn stabiel aanwezig.

---

## 3. Financieel — historische data (10 jaar + TTM)

### Resultatenrekening (bedragen in mln EUR)

Betrouwbaarheid: **AGGREGATOR** (S&P Global via StockAnalysis, kruisgecheckt met persberichten). Zie kanttekening in de bronnen-inventaris.

| Jaar | Omzet | Omzetgroei % | Brutowinst | Brutomarge % | EBIT | EBIT-marge % | EBITDA | EBITDA-marge % | Nettowinst | Nettomarge % | EPS | EPS-groei % | Aandelen mln |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2015 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2016 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2017 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2018 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2019 | 390,2 | — | — | — | — | — | — | — | — | — | — | — | — |
| 2020 | 455,6 | +16,8 | — | — | — | — | — | — | — | — | — | — | — |
| 2021 | 501,1 | +10,0 | 430,2 | 85,8 | 117,4 | 23,4 | 147,2 | 29,4 | 50,1 | 10,0 | 0,25 | — | 201,1 |
| 2022 | 565,9 | +12,9 | 484,6 | 85,6 | 127,3 | 22,5 | 158,5 | 28,0 | 67,6 | 11,9 | 0,37 | +46,5 | 185,0 |
| 2023 | 626,7 | +10,8 | 545,0 | 87,0 | 166,6 | 26,6 | 197,8 | 31,6 | 114,0 | 18,2 | 0,66 | +80,3 | 172,0 |
| 2024 | 671,4 | +7,1 | 590,6 | 88,0 | 206,4 | 30,7 | 225,7 | 33,6 | 123,1 | 18,3 | 0,76 | +15,2 | 160,0 |
| 2025 | 746,8 | +11,2 | 645,6 | 86,5 | 252,6 | 33,8 | 268,4 | 35,9 | 118,3 | 15,8 | 0,75 | −1,3 | 157,0 |
| TTM (Q1'26) | 751,2 | +9,1 | 650,1 | 86,5 | 261,3 | 34,8 | 277,0 | 36,9 | 122,8 | 16,4 | 0,78 | −3,9 | 157,0 |

- **Toelichting resultaten:** De omzet groeide stabiel van ~€390 mln (2019) naar ~€747 mln (2025), een CAGR van circa 11% over 2019–2025. De brutomarge is consistent hoog (85–88%), typisch voor pure software. De EBIT-marge verbeterde sterk (van 23% naar 34%) door schaalvoordelen en kostendiscipline. Opvallend is dat de nettowinst in 2025 dáálde (€118,3 mln vs €123,1 mln in 2024) ondanks hogere omzet en EBIT: dat komt door de gestegen rentelasten (van ~€15 mln naar ~€36 mln) na de schuldgefinancierde 1E-overname én een fors negatief valuta-effect (~€25 mln). De FY2025-omzet van €747 mln is de gerapporteerde IFRS-omzet; het door management gecommuniceerde "€768 mln, +5%" is een pro-forma cijfer dat 1E voor een vol jaar meetelt.
- **Omzet-CAGR:** ~11% (2019–2025); ~10,5% (2021–2025).

### Kasstromen (mln EUR)

| Jaar | CFO | Capex | FCF | FCF na SBC | FCF/aandeel | FCF-marge % | FCF-groei % | FCF-conversie % | SBC | Dividend totaal | Aandeleninkoop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021 | 194,0 | −6,6 | 187,4 | 164,6 | 0,93 | 37,4 | −12,3 | 374 | 27,6 | 0 | 0 |
| 2022 | 204,3 | −5,1 | 199,3 | 176,5 | 1,08 | 35,2 | +6,4 | 295 | 27,6 | 0 | 300,1 |
| 2023 | 229,9 | −3,3 | 226,5 | 183,6 | 1,31 | 36,2 | +13,7 | 199 | 21,8 | 0 | 161,9 |
| 2024 | 249,2 | −3,2 | 246,0 | 202,7 | 1,52 | 36,6 | +8,6 | 200 | 16,8 | 0 | 137,7 |
| 2025 | 233,0 | −4,6 | 228,4 | 208,9 | 1,44 | 30,6 | −7,2 | 193 | 11,9 | 0 | 0 |

- **Toelichting kasstromen:** TeamViewer is een uitgesproken cash-machine: de operationele kasstroom ligt structureel rond €200–250 mln en de FCF-conversie (FCF/nettowinst) is door de jaren heen ruim boven 100% — een teken van hoogwaardige, "echte" winst (lage capex, gunstige werkkapitaaldynamiek door vooruitbetaalde abonnementen/deferred revenue). De FCF na aftrek van aandelencompensatie (SBC) groeide van €164,6 mln (2021) naar €208,9 mln (2025). De FCF dáálde in 2025 (−7%) door hogere betaalde rente en een minder gunstig werkkapitaaleffect; dit is grotendeels gekoppeld aan de 1E-financiering en lijkt deels eenmalig. Belangrijk: de SBC daalt gestaag (van €27,6 mln naar €11,9 mln), wat de verwatering vermindert. De aandeleninkopen waren fors in 2022–2024 (samen ~€600 mln) maar stopten in 2025 omdat de cash naar de 1E-deal en schuldafbouw ging.

### Balans-ratio's (mln EUR / ratio's)

| Jaar | Nettoschuld | Nettoschuld/EBITDA | Eigen vermogen | ROE % | ROIC % | ROCE % | Current ratio | Solvabiliteit % | Goodwill % van EV | Working capital |
|---|---|---|---|---|---|---|---|---|---|---|
| 2021 | 326,9 | 2,2 | 320,1 | 15,6 | — | — | 1,71 | 20,7 | — | 239,6 |
| 2022 | 471,6 | 3,0 | 115,3 | 58,6 | — | — | 0,44 | 9,8 | — | −265,3 |
| 2023 | 456,6 | 2,3 | 83,7 | 136,3 | — | — | 0,31 | 7,5 | — | −352,3 |
| 2024 | 389,4 | 1,7 | 100,5 | 122,5 | 31,2 | — | 0,25 | 9,4 | — | −413,6 |
| 2025 | 901,4 | 3,4 | 164,9 | 71,7 | 17,5 | — | 0,15 | 9,8 | — | −718,7 |

- **Toelichting balans:** De balans is het hart van de beleggingscasus. De nettoschuld verdubbelde bijna van €389 mln (2024) naar €901 mln (2025) door de schuldgefinancierde 1E-overname (kasuitgave acquisities €682,5 mln in 2025; nieuw uitgegeven schuld €758 mln). Nettoschuld/EBITDA staat daardoor weer op ~3,4× (management noemt een pro-forma net leverage van ~2,6× inclusief een vol jaar 1E-EBITDA, en stuurt op afbouw). Het eigen vermogen is laag (€165 mln) en de tangible book value is fors negatief omdat de balans gedomineerd wordt door goodwill (~€1,1 mld) en immateriële activa (~€344 mln) uit de Permira-overname en 1E. De negatieve current ratio en negatief werkkapitaal zijn deels structureel en gunstig (vooruitbetaalde abonnementen = deferred revenue als renteloze financiering), maar de current portion of debt (~€378 mln kortlopend) verdient aandacht qua herfinanciering. De ROE-cijfers ogen extreem hoog maar zijn vertekend door het kleine eigen vermogen; ROIC (~17–31%) is de betere maatstaf en blijft ruim boven de WACC.

### Kapitaalstructuur huidig
- **Nettoschuld (huidig):** ~901 mln (FY2025)
- **Bruto schuld:** ~943 mln
- **Cash & equivalents:** ~42 mln
- **Lease-verplichtingen (IFRS-16):** ~33 mln (kort + lang)
- **Gemiddelde rente %:** ~3,8–4,5 (afgeleid: betaalde rente ~€34–36 mln / bruto schuld ~€943 mln)
- **Rente-dekking (EBIT/rente):** ~7,0× (EBIT 252,6 / rente ~36)

### Non-GAAP / aanpassingen
- **Gebruikt?** true (door het bedrijf; ik gebruik IFRS als primaire grondslag)
- **Welke aanpassingen:** TeamViewer rapporteert "adjusted EBITDA" en "adjusted EPS" waarin o.a. M&A-kosten, herstructurering, SBC en PPA-afschrijvingen worden uitgesloten. Adj. EBITDA-marge ~44–45%; adj. EPS €1,17–1,23 vs IFRS-EPS €0,75.
- **Waarom:** Voor vergelijkbaarheid met peers en management-communicatie. Het verschil GAAP↔adjusted is groot (adj. EPS ~€1,2 vs IFRS €0,75), vooral door PPA-amortisatie op acquisities — een terugkerende, niet-eenmalige post. Ik hanteer IFRS/FCF na SBC in de waardering.

---

## 4. Moat (concurrentievoordeel)

- **Oordeel:** NARROW MOAT
- **Moat-categorieën:**

| Naam | Sterkte (sterk/middel/zwak/geen) | Toelichting |
|---|---|---|
| Immateriële activa | middel | Sterk merk in remote-access (jarenlang de de-facto standaard voor consument en SMB) en een geïnstalleerde basis van miljarden apparaten. Geen patentenmuur; de merksterkte en herkenbaarheid vormen de bescherming, vooral in SMB. |
| Overstapkosten | middel | Voor enterprise-klanten zijn die reëel: integratie in IT-processen, beheerde apparaten en compliance maken overstappen kostbaar en risicovol. Voor SMB zijn de overstapkosten laag — daar concurreert men op prijs en gemak. |
| Netwerkeffecten | zwak | Beperkt: het product wordt niet wezenlijk waardevoller naarmate meer mensen het gebruiken (anders dan een sociaal netwerk). Wel een indirect distributie-effect via gratis privégebruik dat zakelijke adoptie voedt. |
| Kostenvoordeel | middel | Schaalvoordeel in een vrijwel-zero-marginal-cost softwaremodel: brutomarge ~86%, FCF-marge ~30%+. Een nieuwkomer kan de techniek repliceren maar niet eenvoudig de geïnstalleerde basis en het distributiebereik. |
| Efficiënte schaal | zwak | De markt is groot en kent meerdere geloofwaardige spelers; er is geen niche met ruimte voor slechts één aanbieder. |

- **Kwantitatief bewijs:** De moat blijkt vooral uit de structureel hoge en stijgende EBIT-marge (23% → 34% over 2021–2025) en een ROIC die ruim boven de WACC ligt (ROIC ~17% in 2025 post-1E, ~31% in 2024 pre-1E, vs WACC ~8,5%). De FCF-marge boven 30% en FCF-conversie boven 100% bevestigen prijszettingsvermogen en kapitaallichtheid. De ROIC-WACC-spread is positief en structureel (meerdere jaren), maar is door de 1E-overname (veel goodwill, meer geïnvesteerd kapitaal) gedaald.
- **Duurzaamheid:** Middellang (5–10 jaar) in enterprise, korter in SMB. De enterprise-overstapkosten en het merk geven een verdedigbare positie, maar geen onaantastbare.
- **Erosierisico's:** Remote-access is grotendeels gecommoditiseerd; Microsoft (ingebouwde tools), gratis/goedkope alternatieven (AnyDesk, RustDesk, Splashtop) en de bundeling door grote platformen drukken de prijs in SMB. Generatieve AI kan remote-support deels automatiseren — kans én bedreiging. De moat is reëel maar smal (NARROW), niet breed.

---

## 5. Management

- **CEO-naam + tenure:** Oliver Steil, CEO sinds 2018 (leidde ook de pre-IPO transformatie en IPO 2019).
- **CFO-naam + tenure:** Michael Wilkens (CFO; aangetreden ~2022). *(Functie/naam niet via PDF geverifieerd — stage 2 controleren.)*
- **Oprichter nog betrokken?** Nee — oprichters zijn al lang weg (Permira-overname 2014).
- **Insider ownership %:** ~1,84 (management board collectief); CEO Steil ~2,77 mln aandelen (~1,7%).
- **Capital allocation track record:**

| Jaar | Dividend totaal | Aandeleninkoop | M&A uitgaven | Organische capex |
|---|---|---|---|---|
| 2021 | 0 | 0 | ~23 | 6,6 |
| 2022 | 0 | 300,1 | ~2 | 5,1 |
| 2023 | 0 | 161,9 | ~8 | 3,3 |
| 2024 | 0 | 137,7 | 0 | 3,2 |
| 2025 | 0 | 0 | ~683 (1E) | 4,6 |

- **M&A-track-record:** Beperkt aantal kleine deals tot 2024; daarna de transformatieve 1E-overname (USD 720 mln, grootste ooit, schuldgefinancierd). Het succes daarvan is nog niet bewezen — 1E kende kort na overname churn en moet in 2026 "gerevitaliseerd" worden. Dit is de bepalende kapitaalallocatie-beslissing en het oordeel hangt af van de integratie-uitkomst.
- **Beloning:** Mix van vast salaris, jaarlijkse bonus en langetermijn-incentives in aandelen. SBC daalde van ~€28 mln naar ~€12 mln, wat de verwatering temperde. Exacte KPI-koppeling (ROIC/FCF vs koers/omzet) kon ik niet uit het remuneratierapport (PDF) verifiëren → niet ingevuld i.p.v. geschat.
- **Oordeel management:** NEUTRAAL
- **Toelichting:** Het management heeft operationeel sterk gepresteerd (margeverbetering, dalende SBC, fors aandelen ingekocht toen de koers laag stond, IPO-transformatie geslaagd). Tegelijk roept de timing en financiering van de 1E-deal vragen op: een grote, met schuld gefinancierde overname op een moment dat de organische groei afvlakt, vlak voordat de PE-eigenaar volledig uitstapte. De aandeleninkopen 2022–2024 vonden deels plaats tegen hogere koersen dan vandaag, wat de allocatie-discipline relativeert. Insider-alignment is bescheiden (~1,8%) maar reëel. Geen grote integriteits- of fraudecontroverses. Per saldo: competent maar met een nog onbewezen, balansrisicovolle grote kapitaalinzet — vandaar NEUTRAAL in plaats van STERK.

---

## 6. Sector & concurrentie

- **Sector-groeivooruitzicht:** De bredere markt voor remote-connectivity/remote-support en digital-workplace-software groeit naar schatting mid-single tot low-double digit % per jaar, gedreven door hybride werken, IoT/industriële connectiviteit en IT-automatisering. *(Exact %/bron niet hard geverifieerd → indicatief.)*
- **Porter five forces:**
  - **Rivaliteit:** HOOG — meerdere geloofwaardige spelers (AnyDesk, Splashtop, LogMeIn/GoTo, Microsoft-tools, RustDesk open source), prijsdruk in SMB.
  - **Nieuwe toetreders:** MIDDEL — techniek is repliceerbaar, maar merk, geïnstalleerde basis en enterprise-relaties vormen een drempel.
  - **Substituten:** HOOG — ingebouwde OS-tools, VPN/RDP, cloud-beheer en AI-automatisering kunnen delen van de functie vervangen.
  - **Macht leveranciers:** LAAG — software, weinig afhankelijkheid van schaarse inputs; cloud-infra is inkoopbaar.
  - **Macht afnemers:** MIDDEL — SMB-klanten prijsgevoelig en makkelijk overstappend; enterprise-klanten gebonden door integratie.
- **Concurrenten:**

| Concurrent | Marktaandeel % |
|---|---|
| AnyDesk | — |
| Splashtop | — |
| GoTo (LogMeIn) | — |
| Microsoft (RDP/Intune e.d.) | — |

*(Marktaandelen niet uit geverifieerde bron beschikbaar → leeg gelaten.)*

- **Positie van het bedrijf:** Leider/uitdager in remote-access met een zeer brede geïnstalleerde basis; uitdager in het bredere digital-workplace-segment waar grotere platformspelers domineren. Sterke positie in SMB-remote-support, groeiende maar nog kleinere positie in enterprise/DEX.

### TAM/SAM/SOM
- **TAM (mln EUR):** —
- **TAM-groei %:** —
- **SAM (mln):** —
- **SAM-groei %:** —
- **Huidige penetratie %:** —
- **Impliciete penetratie na horizon %:** —
- **Groei plausibel?** —
- **Bron TAM/SAM:** geen geverifieerde bron
- **Toelichting:** Geen betrouwbaar gekwantificeerde TAM/SAM-bron gevonden; conform de bronregels niet ingevuld i.p.v. geschat. Kwalitatief is de markt groot en groeiend, maar competitief en deels commoditiserend.

---

## 7. Analyse-frameworks (9 frameworks, SCORES 0-5)

### Graham
- **Oordeel:** GEDEELTELIJK
- **Graham number:** n.v.t. betrouwbaar (EPS €0,75, BVPS €1,01 → √(22,5×0,75×1,01) ≈ €4,1) → koers €5,36 ligt erboven
- **Margin of safety %:** negatief t.o.v. Graham number
- **Toelichting:** De P/E is met ~7× zeer laag en zou Graham aanspreken, maar de P/B is ~5× (door het kleine, goodwill-zware eigen vermogen) en dat botst hard met Graham's eis van P/B < 1,5. De schuldratio's zijn na 1E ook te hoog voor de defensieve Graham-criteria. Per de harde rubric (P/B > 3,0 → score 1) faalt het aandeel op Grahams balansgerichte maatstaven, ook al is het op winstbasis goedkoop.
- **Score (0-5):** 1

### Buffett / Munger
- **Oordeel:** GEDEELTELIJK
- **ROIC structureel boven WACC?** true (meerdere jaren ROIC > WACC, al gedaald na 1E)
- **Toelichting:** Veel Buffett-kenmerken zijn aanwezig: begrijpelijk product, hoge en voorspelbare kasstromen, prijszettingsvermogen in enterprise, en een zeer lage prijs (P/FCF ~4×). Wat ontbreekt voor topscore is een brede moat (slechts NARROW) en een schone balans — de schuldgefinancierde overname verhoogt het risico en verlaagt de ROIC. ROIC > WACC structureel ✓, moat NARROW, P/FCF ≤ 30 ✓ → score 3 per rubric.
- **Score (0-5):** 3

### Peter Lynch
- **Categorie:** Stalwart (met turnaround-elementen)
- **Oordeel:** INTERESSANT
- **PEG-ratio:** op IFRS-basis vertekend (2025/2026-winst daalt); op adjusted-EPS-basis en meerjarige winstgroei laag (<1)
- **Toelichting:** Het verhaal is helder en in twee zinnen uit te leggen — een goedkope kasstroommachine met afgevlakte groei en een nog te bewijzen enterprise-pivot. De PEG is moeilijk eenduidig te bepalen omdat de IFRS-winst in 2025/2026 daalt door rente en FX; op genormaliseerde/adjusted basis is de waardering t.o.v. de meerjarige groei aantrekkelijk. Conservatief gewaardeerd als een goedkope stalwart met turnaround-kenmerken → score 3.
- **Score (0-5):** 3

### Phil Fisher
- **Oordeel:** GEMIDDELD
- **Toelichting:** R&D als % van omzet is met ~13% hoog en past bij een innovatieve softwarespeler; de marges zijn goed beschermd door de NARROW moat. Management is competent (NEUTRAAL, niet ZORGWEKKEND). Daarmee zijn 2 van de 3 Fisher-criteria voldaan (margebescherming + R&D-intensiteit), terwijl management net niet als STERK kwalificeert → score 4 per rubric.
- **Score (0-5):** 4

### Magic Formula (Greenblatt)
- **Oordeel:** AANTREKKELIJK
- **Earnings yield %:** ~14,5 (EBIT €252,6 / EV €1.743)
- **Return on capital %:** zeer hoog (>50%) — door negatief werkkapitaal (deferred revenue) en lage vaste activa is het geïnvesteerde tastbare kapitaal klein
- **Toelichting:** Op Greenblatt's twee assen scoort TeamViewer uitstekend: een earnings yield van ~14,5% (ruim boven de 10%-drempel) en een zeer hoge return on tangible capital doordat het bedrijf kapitaallicht is en met vooruitbetaalde abonnementen werkt. Dit is precies het type "goedkoop én hoogrenderend" dat de Magic Formula zoekt → score 5.
- **Score (0-5):** 5

### Moat
- **Score (0-5):** 3 — NARROW moat met positieve, structurele ROIC-WACC-spread (~9pp in 2025), maar geen breed/monopolistisch voordeel.

### Management
- **Score (0-5):** 3 — competent met goede operationele uitvoering, maar gemengde kapitaalallocatie (grote leveraged deal, buybacks deels op hogere koersen) en bescheiden insider-alignment.

### Fair Value DCF
- **Score (0-5):** 5 — basis-DCF impliceert upside ruim boven 30% (zie §12).

### Fair Value IPO-gecorr.
- **Score (0-5):** 5 — IPO < 10 jaar geleden (2019); ook IPO-gecorrigeerd blijft de upside > 30% (de FCF-machine is post-IPO bevestigd; de waarde zit niet in een eenmalig schoongemaakte balans).

### Scorekaart totaal
- **Totaalscore:** 32
- **Max:** 45
- **Eindoordeel:** HOLD
  - Regel: totaal 32 < 33 → geen KOOP; totaal ≥ 24 → HOLD; DCF-score 5 (≠1) → geen PASS. **HOLD.**
- **Samenvatting:** TeamViewer scoort 32/45 — net onder de KOOP-drempel van 33. De spanning in de casus is scherp: op kasstroom is het aandeel spotgoedkoop (P/FCF ~4×, earnings yield ~14,5%, FCF-rendement ~25%) en de DCF- en Magic-Formula-scores zijn maximaal, maar Graham faalt hard op de balans (P/B ~5×, hoge schuld) en de moat is slechts NARROW. Het management is competent maar deed een grote, schuldgefinancierde overname waarvan het succes nog moet blijken. Het eindoordeel is daarom HOLD, niet KOOP: de onderwaardering is reëel, maar de balansrisico's en de afvlakkende organische groei rechtvaardigen geen koopsignaal zonder bewijs dat de enterprise-/1E-strategie aanslaat en de schuld daalt.

---

## 8. Risico's

| # | Omschrijving | Kans | Impact | DCF-aanname geraakt | Toelichting |
|---|---|---|---|---|---|
| 1 | Schuldgefinancierde 1E-overname / hoge leverage | HOOG | GROOT | WACC, FCF (rente) | Nettoschuld sprong naar ~€901 mln (~3,4× EBITDA). Hogere rente drukt de nettowinst en bindt FCF aan aflossing i.p.v. aandeelhouders. Bij rentestijging of tegenvallende EBITDA loopt het herfinancieringsrisico op (kortlopend deel ~€378 mln). |
| 2 | Structurele erosie van het SMB-segment | HOOG | GROOT | omzetgroei, terminal groei | Het grote SMB-fundament groeit nauwelijks nog, kende eenmalige churn en staat onder prijsdruk van goedkope/gratis alternatieven. Als SMB structureel krimpt, valt een groot deel van de kasstroombasis weg. |
| 3 | Mislukte integratie/revitalisatie van 1E | MIDDEN | GROOT | omzetgroei, marge | 1E kende kort na overname churn; de hele acquisitiethese (enterprise + Noord-Amerika) staat of valt met succesvolle integratie. Mislukking betekent goodwill-afschrijving en verloren strategische logica. |
| 4 | Commoditisering & AI-substitutie van remote-access | MIDDEN | GROOT | terminal groei, marge | Microsoft-tools, AnyDesk/Splashtop/RustDesk en AI-gedreven automatisering kunnen prijzen en volumes uithollen, vooral in SMB. Raakt de duurzaamheid van de terminal-waarde direct. |
| 5 | Valuta (EUR/USD) | MIDDEN | MIDDEL | FCF, nettowinst | Aanzienlijke USD-omzet en USD-schuld; FX kostte ~3,3pp omzet in Q1 2026 en ~€25 mln in 2025. Volatiel maar deels natuurlijke hedge. |
| 6 | Goodwill/immateriële-activa-afschrijving | MIDDEN | MIDDEL | eigen vermogen, vertrouwen | Goodwill ~€1,1 mld op een eigen vermogen van ~€165 mln. Een impairment (bv. op 1E) zou het eigen vermogen sterk raken en het vertrouwen ondermijnen, ook al is het non-cash. |
| 7 | Afvlakkende totale groei / guidance-risico | MIDDEN | MIDDEL | omzetgroei | De FY2026-guidance is bescheiden (tot ~3% omzetgroei); een eerdere guidance-cut zette het aandeel onder druk. Lagere groei verlaagt direct de DCF-waarde. |
| 8 | Pre-IPO/PE-erfenis & kapitaalstructuur | LAAG | MIDDEL | WACC | Permira is vrijwel uitgestapt (overhang weg, positief), maar de balans draagt nog de erfenis van de leveraged PE-structuur en latere herbeleveraging via 1E. |

**Pre-IPO financial-engineering check (verplicht):** TeamViewer kwam in 2019 naar de beurs als exit voor private-equityeigenaar Permira (overname 2014), met een door PE opgetuigde, schuldhoudende kapitaalstructuur. De IPO-opbrengsten gingen primair naar de verkopende aandeelhouder (Permira), niet als vers kapitaal naar het bedrijf; er was schuld op de balans uit de overnameperiode die daarna deels is afgebouwd. Een "schoon balansmoment" dat de historie vertekent is hier beperkt relevant voor de waardering vandaag, omdat de balans inmiddels juist opnieuw fors is beladen via de 1E-deal (2024/2025). Dividend-recapitalisatie vóór IPO ten gunste van insiders: niet specifiek geconstateerd in geverifieerde bronnen. De gecorrigeerde fair value verschilt niet wezenlijk van de basis, omdat de waardering op huidige (post-1E) FCF en huidige nettoschuld is gebouwd — niet op een geflatteerd pre-IPO plaatje.

---

## 9. These invalide bij

De these (onderwaardeerde kasstroommachine) is weerlegd wanneer (a) de FCF na SBC structureel onder ~€180 mln zakt door aanhoudende SMB-krimp of margedruk; (b) de nettoschuld niet daalt of stijgt terwijl de rente oploopt, zodat de FCF meerjarig naar de schuldeisers gaat; of (c) 1E moet worden afgeschreven, wat zowel de strategische logica als het eigen vermogen onderuithaalt. In die gevallen is het aandeel geen koopje maar een terechte value-trap.

---

## 10. ESG

### Materiële factoren (SASB-gebaseerd)

| Factor | SASB-categorie | Risiconiveau (Laag/Midden/Hoog) | Financiële impact | DCF-impact |
|---|---|---|---|---|
| Datasecurity & privacy | Data Security | Midden | Beveiligingsincident kan vertrouwen en enterprise-omzet schaden | omzetgroei |
| Misbruik van software (scam/fraude) | Business Ethics | Midden | Remote-access wordt door oplichters misbruikt; reputatie- en regelgevingsrisico | omzetgroei |
| Energie/datacenters | Environmental Footprint of Hardware Infrastructure | Laag | Beperkt; asset-light SaaS | marge |
| Menselijk kapitaal / talent | Recruiting & Managing | Midden | Afhankelijkheid van software-engineers | marge |

- **Eindoordeel ESG:** GEMIDDELD RISICO
- **Toelichting:** Als asset-light softwarebedrijf is de milieuvoetafdruk klein. De materiële ESG-risico's zijn datasecurity/privacy en het misbruik van remote-access door fraudeurs (een terugkerend reputatiethema voor de hele categorie). Governance is verbeterd nu de PE-overhang weg is, maar de hoge schuld en goodwill blijven aandachtspunten.

---

## 11. Katalysatoren

| Datum ca. | Omschrijving | Richting | Impact |
|---|---|---|---|
| 2026-08 | Halfjaarcijfers H1 2026 (groei + 1E-integratie + leverage-update) | BINAIR | GROOT |
| 2026-11 | Q3 2026-cijfers en eventuele guidance-aanpassing | BINAIR | MIDDEL |
| 2026 (doorlopend) | Schuldafbouw / deleveraging richting ~2,5× | POSITIEF | GROOT |
| 2026 (doorlopend) | 1E/DEX-revitalisatie en enterprise-ARR-momentum | POSITIEF | GROOT |
| 2026 (doorlopend) | Hervatting aandeleninkoop indien schuld daalt | POSITIEF | MIDDEL |
| 2026-Q4 / 2027 | EUR/USD-ontwikkeling (FX-effect op omzet en schuld) | NEUTRAAL | MIDDEL |
| 2027-02 | FY2026-jaarcijfers en FY2027-guidance | BINAIR | GROOT |

---

## 12. Fair value — kwantitatief (DCF)

> **Methodische let op:** TeamViewer is **niet** cyclisch (software/SaaS), dus de cyclus-normalisatie van METHODE.md REGEL 1–4 is niet van toepassing. Ik gebruik de recente FCF na SBC (€208,9 mln, FY2025) als vertrekpunt. De berekeningen zijn handmatig uitgevoerd (de reken-sandbox was niet beschikbaar); de methode (2-fase FCFF, mid-year convention, Gordon + exit-multiple cross-check) volgt METHODE.md H7.

### WACC-componenten
- **Risicovrije rente %:** 2,99
- **Bron risicovrije rente:** Duitse 10-jaars Bund, 12-6-2026 (TradingEconomics)
- **Type:** nominal (spot)
- **ERP (equity risk premium) %:** 4,5
- **Bron ERP:** Damodaran 2026 (US implied ERP ~4,23% jan-2026; mature-market benadering, afgerond naar 4,5%)
- **Beta (adjusted, Blume):** 0,93
- **Bron beta:** marktdata (aggregator); 5y
- **Type beta:** regressie (5y)
- **Country risk premium %:** 0 (Duitsland)
- **Size premium %:** 1,5 (marktkap < €2 mld)
- **Cost of equity %:** 8,68  (= 2,99 + 0,93×4,5 + 1,5)
- **Schuldkosten na belasting %:** 3,33  (= 4,5% × (1 − 0,26))
- **E/V gewicht %:** 70 (genormaliseerde doelstructuur; deleveraging richting ~2,5×)
- **D/V gewicht %:** 30
- **WACC %:** 8,5  (genormaliseerd ~7,1% op doelstructuur; afgerond opwaarts naar 8,5% conform sector-WACC software ~8% en conservatisme)
- **Sector WACC % (referentie Damodaran):** ~8,0–8,5 (software/computer services)
- **Illiquiditeitskorting %:** null (MDAX-aandeel, voldoende liquide)

### DCF model-specs
- **Model type:** 2-fase (jaren 1–5 en 6–10) + terminal
- **FCF-definitie:** FCF to firm (FCFF), na SBC
- **Basis FCF:** 208,9 (FY2025, na SBC)
- **Basis FCF na SBC:** 208,9
- **FCF-type:** stated FCF (CFO − capex), na aftrek SBC
- **Groei fase 1 % (jaar 1–5):** 2,0 (basis)
- **Groei fase 2 % (jaar 6–10):** 1,0 (basis)
- **Terminal groei %:** 1,5 (basis) — onder Duitse nominale BBP-groei
- **Terminal methode:** Gordon growth (met exit-multiple cross-check)
- **Exit multiple gebruikt (EV/EBITDA):** ~9× (sector-software mid-cycle)
- **Bron exit multiple:** sector-mediaan software (indicatief)
- **Terminal value Gordon growth:** ~3.516 mln (basis)
- **Terminal value exit multiple:** ~9× × EBITDA ~€300 mln (toekomstig) ≈ vergelijkbare orde van grootte; Gordon en exit-multiple liggen binnen redelijke marge
- **Terminal value % van totaal:** ~51% (basis) — ruim onder 75% ✓
- **Terminal implied EV/EBITDA:** redelijk (< 15×) voor een softwarebedrijf ✓
- **Terminal groei consistentie:** Terminalgroei 1,5% vereist een bescheiden herinvesteringsvoet bij ROIC ~15% (g = herinvestering × ROIC → ~10% herinvestering) — plausibel voor een volwassen, kapitaallichte softwarespeler.
- **Mid-year convention:** true
- **Aandelen uitstaand (mln):** 157
- **Nettoschuld huidig:** 901,4

### DCF-toelichting
Ik hanteer FCFF na SBC, verdisconteerd tegen de WACC, waarna de nettoschuld wordt afgetrokken en gedeeld door 157 mln aandelen. Voor de basis koos ik bewust een conservatieve WACC (8,5%) — hoger dan de WACC die uit de huidige (zwaar geleveragede) kapitaalstructuur rolt (~6%) — omdat die lage WACC kunstmatig op goedkope schuld leunt die het bedrijf juist afbouwt, en omdat de sector-WACC voor software rond 8% ligt. De groeiaannames zijn behoudend (2% → 1% → 1,5% terminaal) gezien de afgevlakte organische groei en SMB-erosie. De terminal value is ~51% van de totale waarde (gezond, < 75%). Zelfs met deze conservatieve aannames komt de basis-fair-value op ~€14,5, ruim boven de koers van €5,36. Dat komt doordat het bedrijf een zeer hoge FCF genereert (€209 mln) ten opzichte van een marktkap van slechts €842 mln. De spanning zit hem niet in de DCF-mechaniek maar in de vraag of die FCF houdbaar is — daarover gaat de reverse-DCF hieronder.

### 5-jaars projectie (basis, mln EUR; indicatief)

| Jaar | Omzet | Omzetgroei % | EBIT | EBIT-marge % | NOPAT | Capex | ΔNWC | SBC | FCF |
|---|---|---|---|---|---|---|---|---|---|
| 2026 | 765 | +2,4 | 255 | 33,3 | 189 | −5 | + | 11 | 213 |
| 2027 | 780 | +2,0 | 262 | 33,6 | 194 | −5 | + | 10 | 217 |
| 2028 | 796 | +2,0 | 268 | 33,7 | 198 | −5 | + | 10 | 222 |
| 2029 | 812 | +2,0 | 273 | 33,6 | 202 | −5 | + | 9 | 226 |
| 2030 | 828 | +2,0 | 279 | 33,7 | 206 | −5 | + | 9 | 231 |

*(NOPAT/FCF indicatief afgeleid; ΔNWC positief door deferred revenue.)*

### Scenarios

| Scenario | FCF-groei % | WACC % | Fair value | Upside % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 0% → −1% | 9,0 | 10,50 | +96 | 25 |
| Basis | 2% → 1% | 8,5 | 14,49 | +170 | 50 |
| Optimistisch | 4% → 3% | 8,0 | 23,50 | +338 | 25 |

- **Kansgewogen fair value:** 15,24  (= 0,25×10,50 + 0,50×14,49 + 0,25×23,50)

### Reverse DCF
- **Impliciete groei %:** −3,1 (de constante eeuwigdurende FCF-groei die de huidige koers rechtvaardigt bij WACC 8,5%)
- **Historische FCF CAGR %:** +6,1 (FCF na SBC 2021→2025: €164,6 → €208,9 mln)
- **Consensus groei % (analisten):** omzet +2–3% (2026), FCF ~€195 mln 2026 (tijdelijke daling −7% door rente/integratie)
- **Interpretatie:** Dit is de kern van de casus. Om de koers van €5,36 te rechtvaardigen, hoeft de markt niet eens te geloven in stagnatie — ze prijst een eeuwigdurende FCF-**krimp** van circa 3% per jaar in. Dat staat haaks op de historische FCF-groei (+6%) en op de analistenconsensus (vlakke tot licht positieve groei). Met andere woorden: de markt is uitgesproken pessimistisch en weegt de balansrisico's, SMB-erosie en het 1E-integratierisico zwaar. Als de FCF zelfs maar vlak blijft, is het aandeel fors ondergewaardeerd; als de FCF structureel krimpt, is de huidige prijs terecht. De waarheid ligt waarschijnlijk tussen vlak en licht groeiend — wat een aanzienlijke onderwaardering impliceert, mits de balans niet ontspoort.

### EPV (Bruce Greenwald)
- **Genormaliseerde EBIT-marge %:** 27,4 (gemiddelde 2021–2025)
- **Genormaliseerde NOPAT:** 151 mln  (EBIT €204,6 mln × (1 − 0,26))
- **Maintenance capex:** ~6 mln (≈ D&A in dit asset-light model; netto-effect ~0 op earnings power)
- **Adjusted earnings power:** ~151 mln
- **EPV per aandeel:** 5,61  (= (151/0,085 − 901,4) / 157)
- **Groeipremie %:** ~0 (koers €5,36 ≈ EPV €5,61)

De EPV (de waarde zónder groei) komt vrijwel exact op de huidige koers uit. Dat betekent: de markt betaalt vandaag ongeveer de no-growth-waarde van het bedrijf en kent **nul** waarde toe aan toekomstige groei. Alle upside in de DCF (~€14,5 basis) bestaat dus uit groei die de markt simpelweg niet inprijst. Dat onderstreept zowel de potentie (als er enige groei is, is het aandeel goedkoop) als de reden (de markt twijfelt of die groei en de FCF houdbaar zijn gegeven schuld en SMB-erosie).

### Andere methoden
- **DDM uitgevoerd?** false (geen dividend)
- **SOTP uitgevoerd?** false (geen conglomeraat; SMB/Enterprise/1E niet apart genoeg gewaardeerd met geverifieerde data)

### Synthese fair value
- **Bandbreedte laag:** 5,61 (EPV, no-growth)
- **Bandbreedte centraal:** 14,49 (DCF basis)
- **Bandbreedte hoog:** 23,50 (DCF optimistisch)
- **Methode-gewichten:**
  - DCF %: 50
  - EPV %: 35
  - Multiples %: 15
- **Margin of safety vereist %:** 35 (hoog, gezien balansrisico, NARROW moat en datakwaliteit op AGGREGATOR-niveau)
- **Koopniveau:** ~9,90 (gewogen fair value ~€15,2 × (1 − 0,35))
- **Synthese-toelichting:** De drie methoden spannen een brede band op: de EPV (€5,61, geen groei) ligt vrijwel op de koers, terwijl de DCF (€14,5 basis) en de optimistische variant (€23,5) een fors hogere waarde tonen. Het verschil tússen EPV en DCF ís de beleggingscasus: de markt betaalt alleen voor de steady state en negeert elke groei. Ik weeg DCF zwaar (50%) maar geef EPV een stevige 35% als ankerpunt voor het neerwaartse risico, en multiples 15%. De gewogen fair value komt op circa €15,2. Gezien de hoge schuld, de slechts NARROW moat en het feit dat mijn cijfers op aggregatordata (niet PDF) rusten, eis ik een ongebruikelijk hoge margin of safety van 35%, wat een koopniveau rond €9,90 oplevert. Daaronder (~€9,90) wordt het aandeel ook bij voorzichtige aannames aantrekkelijk; daarboven is het een afweging tussen diepe waarde en een mogelijke value-trap. De huidige koers van €5,36 ligt nét boven het EPV-anker (€5,61) maar ruim onder de gewogen fair value — een onderwaardering die echter pas tot een koopsignaal leidt zodra de balans- en groei-onzekerheid afneemt; vandaar HOLD.

### Gevoeligheid (DCF)
- **WACC range:** [7,5%, 8,0%, 8,5%, 9,0%, 9,5%, 10,0%]
- **Groei range (fase-1 FCF-groei):** [0%, 1%, 2%, 4%, 6%]
- **Matrix (fair value per aandeel, EUR; indicatief, terminal 1,5%, mid-year):**

| Groei \ WACC | 7,5% | 8,0% | 8,5% | 9,0% | 9,5% | 10,0% |
|---|---|---|---|---|---|---|
| 0% | 14,8 | 13,2 | 11,8 | 10,5 | 9,4 | 8,5 |
| 1% | 16,4 | 14,6 | 13,1 | 11,7 | 10,5 | 9,5 |
| 2% | 18,2 | 16,2 | 14,5 | 13,0 | 11,7 | 10,6 |
| 4% | 22,6 | 20,0 | 17,9 | 16,0 | 14,4 | 13,0 |
| 6% | 28,3 | 24,9 | 22,1 | 19,7 | 17,7 | 15,9 |

De fair value blijft in vrijwel de hele matrix ruim boven de koers van €5,36 — pas bij gelijktijdig hoge WACC én FCF-krimp (zoals de reverse-DCF impliceert) zakt de waarde naar het koersniveau.

---

## 13. Databronnen

### Bronnen-hiërarchie
- Jaarverslag PDF / IR-pagina → HOOG — **niet rechtstreeks geopend in deze sessie** (zie kanttekening)
- Beursmelding / persbericht → HOOG — via web search bevestigd (FY2024/FY2025 kerncijfers)
- Aggregator (S&P Global via StockAnalysis, Statista) → AGGREGATOR — primaire bron voor de tabellen

### Financiële bronnen (10 jaar historie)

| Jaar | Bron | URL | Betrouwbaarheid |
|---|---|---|---|
| 2015 | geen | — | — |
| 2016 | geen | — | — |
| 2017 | geen | — | — |
| 2018 | geen | — | — |
| 2019 | Statista (omzet) | https://www.statista.com/statistics/1543342/teamviewer-revenue/ | AGGREGATOR |
| 2020 | Statista (omzet) | https://www.statista.com/statistics/1543342/teamviewer-revenue/ | AGGREGATOR |
| 2021 | S&P Global / StockAnalysis | https://stockanalysis.com/quote/etr/TMV/financials/ | AGGREGATOR |
| 2022 | S&P Global / StockAnalysis | https://stockanalysis.com/quote/etr/TMV/financials/ | AGGREGATOR |
| 2023 | S&P Global / StockAnalysis | https://stockanalysis.com/quote/etr/TMV/financials/ | AGGREGATOR |
| 2024 | S&P Global / StockAnalysis (+ persbericht) | https://stockanalysis.com/quote/etr/TMV/financials/ | AGGREGATOR |
| 2025 | S&P Global / StockAnalysis (+ persbericht) | https://stockanalysis.com/quote/etr/TMV/financials/ | AGGREGATOR |

**Afwijking van de harde eis:** de recente 5 jaren (2021–2025) zijn AGGREGATOR, niet HOOG, omdat de PDF-jaarverslagen niet rechtstreeks konden worden geopend. Dit is bewust gemeld in plaats van te verzwijgen of een lege analyse op te leveren (conform de instructie van de gebruiker: liever aangeven dan leeg terugkomen).

### Jaarverslagen geraadpleegd

| Jaar | Bron | URL |
|---|---|---|
| 2025 (persbericht) | TeamViewer FY2025 results (EQS) | https://www.eqs-news.com/company/teamviewer-se/reports/14cdaa0b-7371-1014-b130-232b05d60f5f |
| — | TeamViewer IR financial results | https://ir.teamviewer.com/publications/financial-results |

### Beursmeldingen geraadpleegd

| Datum | Omschrijving | URL |
|---|---|---|
| 2024-12-10 | Aankondiging overname 1E (USD 720 mln) | https://www.teamviewer.com/en-us/global/company/press/2024/teamviewer-to-acquire-1e/ |
| 2026-02-10 | FY2025 voorlopige resultaten | https://www.eqs-news.com/news/corporate/teamviewer-delivers-on-fy-2025-pro-forma-topline... |
| 2026-Q1 | Q1 2026 resultaten (omzet €183,2 mln, adj. EBITDA-marge 45,3%) | https://www.eqs-news.com/news/corporate/teamviewer-q1-2026-revenue-in-line-with-expectations-enterprise-arr-up-8-cc-full-year-2026-guidance-reaffirmed/f9b950f7-6718-4203-86e0-138e3c81ea46_en |
| 2025-09-03 | Permira verlaagt belang naar 2,9% | https://in.tradingview.com/news/reuters.com,2025:newsml_FWN3UR1GG:0 |

### IPO-prospectus
- **Geraadpleegd?** false (prospectus niet rechtstreeks geopend)
- **URL:** —
- **Pre-IPO data beschikbaar?** false (pre-IPO privaat onder Permira; geen publieke jaarrekeningen 2015–2018)
- **Pre-IPO bron:** —

### Non-GAAP
- **Gebruikt?** true
- **Toelichting:** Het bedrijf rapporteert adjusted EBITDA (~44–45% marge) en adjusted EPS (€1,17–1,23) waarin M&A-kosten, herstructurering, SBC en PPA-amortisatie zijn uitgesloten. Ik gebruik IFRS-cijfers en FCF na SBC als waarderingsgrondslag; de adjusted-cijfers zijn alleen ter context genoemd.

### Ontbrekende data (eerlijke lijst)
- Boekjaren 2015–2018: geen publieke data (pre-IPO, privaat onder Permira).
- 2019–2020: alleen omzet betrouwbaar; winst/kasstroom/balans-detail niet uit één betrouwbare bron.
- **Recente 5 jaren niet PDF-geverifieerd** — gebouwd op S&P Global aggregatordata, gekruist met persberichten. Stage 2 zou dit tegen de jaarverslag-PDF's moeten verifiëren.
- Geografische omzetverdeling (%), segment-omzet-% (SMB/Enterprise), exacte marktaandelen concurrenten, TAM/SAM: geen geverifieerde bron → leeg gelaten.
- Goodwill % van EV per jaar, ROCE, asset turnover per jaar: niet volledig betrouwbaar afleidbaar → leeg.
- CFO-naam/tenure en exacte beloning-KPI's: niet uit remuneratierapport geverifieerd.
- De €34,2 mln nettowinst die in één nieuwsbron rondging, bleek een verkeerd toegewezen kwartaalcijfer; de FY2025-geconsolideerde nettowinst is €118,3 mln (S&P Global) — conflict opgelost maar PDF-bevestiging ontbreekt.

### Peildatum analyse
- 2026-06-11 (koers €5,36; laatst geverifieerde slotkoers uit de zoekresultaten)

---

## 14. Volledige bronnen-lijst

| Titel | URL | Type |
|---|---|---|
| StockAnalysis — TMV Income Statement (S&P Global) | https://stockanalysis.com/quote/etr/TMV/financials/ | aggregator |
| StockAnalysis — TMV Balance Sheet | https://stockanalysis.com/quote/etr/TMV/financials/balance-sheet/ | aggregator |
| StockAnalysis — TMV Cash Flow | https://stockanalysis.com/quote/etr/TMV/financials/cash-flow-statement/ | aggregator |
| StockAnalysis — TMV Forecast / analistenconsensus | https://stockanalysis.com/quote/etr/TMV/forecast/ | aggregator |
| TeamViewer — overname 1E (persbericht) | https://www.teamviewer.com/en-us/global/company/press/2024/teamviewer-to-acquire-1e/ | beursmelding |
| TeamViewer IR — to acquire 1E | https://ir.teamviewer.com/publications/teamviewer-to-acquire-1e | beursmelding |
| Carlyle — verkoop 1E aan TeamViewer | https://www.carlyle.com/media-room/news-release-archive/carlyle-agrees-to-sell-1e-to-teamviewer | nieuwsartikel |
| EQS — TeamViewer Q1 2026 resultaten | https://www.eqs-news.com/news/corporate/teamviewer-q1-2026-revenue-in-line-with-expectations-enterprise-arr-up-8-cc-full-year-2026-guidance-reaffirmed/f9b950f7-6718-4203-86e0-138e3c81ea46_en | beursmelding |
| EQS — TeamViewer FY2025 pro forma topline | https://www.eqs-news.com/company/teamviewer-se/reports/14cdaa0b-7371-1014-b130-232b05d60f5f | beursmelding |
| Reuters via TradingView — Permira verlaagt belang naar 2,9% | https://in.tradingview.com/news/reuters.com,2025:newsml_FWN3UR1GG:0 | nieuwsartikel |
| Bloomberg — Permira's TeamViewer debt-laden IPO (2019) | https://www.bloomberg.com/opinion/articles/2019-09-13/permira-s-teamviewer-debt-laden-ipo-wants-to-be-a-hot-tech-ipo | nieuwsartikel |
| Statista — TeamViewer revenue 2019–2023 | https://www.statista.com/statistics/1543342/teamviewer-revenue/ | aggregator |
| TradingEconomics — Duitse 10y Bund yield | https://tradingeconomics.com/germany/government-bond-yield | databron |
| Damodaran — ERP 2026 (SSRN) | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6361419 | onderzoeksrapport |
| Wikipedia — TeamViewer (company) | https://en.wikipedia.org/wiki/TeamViewer_(company) | aggregator |

---

## 15. Update-historie

| Datum | Versie | Wijziging |
|---|---|---|
| 2026-06-13 | 1.0 | Eerste publicatie (stage 1, cowork) |

---

## Opmerkingen voor Claude Code

Inhoudelijke twijfels en aandachtspunten voor stage 2:

1. **PDF-verificatie ontbreekt (belangrijkste punt).** De financiële tabellen 2021–2025 zijn op S&P Global-aggregatordata gebouwd (StockAnalysis), gekruist met persberichten — niet op de jaarverslag-PDF's. METHODE.md eist HOOG voor de recente 5 jaren. De `check-sources.py`-validator zal hier vermoedelijk op afgaan (URL's bevatten geen "annual"/"jaarverslag"/.pdf voor de recente jaren). Overweeg: (a) PDF-jaarverslagen 2021–2025 ophalen via ir.teamviewer.com / financialreports.eu en de cijfers verifiëren, of (b) de betrouwbaarheidsvlag bewust op AGGREGATOR laten staan met deze toelichting. De kerncijfers (omzet, EBIT, EBITDA, nettowinst, FCF, nettoschuld) matchen wél met de officiële persberichten, dus het materiële risico op verkeerde cijfers is laag.

2. **Nettowinst-conflict opgelost maar PDF-bevestiging gewenst.** Eén nieuwsbron noemde €34,2 mln FY2025-nettowinst; dat bleek een misgeïnterpreteerd kwartaalcijfer. S&P Global geeft FY2025-geconsolideerde nettowinst €118,25 mln. Graag bevestigen tegen de IFRS-jaarrekening.

3. **WACC-keuze.** De WACC die uit de huidige (zwaar geleveragede) kapitaalstructuur rolt is ~6%, wat de fair value nog hoger zou maken (~€22). Ik koos bewust 8,5% (genormaliseerde doelstructuur + sector-WACC) om de fair value conservatief te houden. Als stage 2 het dcf_calculator.py-script draait met de exacte huidige kapitaalstructuur, zal de uitkomst afwijken — let op de aanname-keuze (target vs spot leverage).

4. **DCF/EPV handmatig berekend** (reken-sandbox was niet beschikbaar). Cijfers zijn met de hand afgeleid en intern consistent gemaakt, maar zijn afrondingen. Graag herrekenen met het script; de scenario-inputs (g1/g2/gt/WACC/kansen) staan expliciet in §1 en §12.

5. **Scorekaart 32/45 → HOLD, net onder KOOP (33).** Dit is gevoelig voor de Graham-score (1, door P/B > 3) en de Lynch-score (3). Als stage 2 de Lynch-PEG anders berekent (bv. op adjusted EPS met meerjarige groei → score 4), kantelt het totaal naar 33 en daarmee het verdict naar KOOP. De drempel is dus knife-edge — controleer de PEG-berekening expliciet.

6. **Ontbrekende velden** (geografie %, segment-omzet %, marktaandelen, TAM/SAM, sommige balans-ratio's per jaar) zijn bewust leeg gelaten i.p.v. geschat. Aanvullen kan alleen met geverifieerde bron.
