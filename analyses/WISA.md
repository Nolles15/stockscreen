# Research: WISA — WISA Group Oyj

> Versie 1.0, peildatum 2026-11-10. Opgesteld volgens METHODE.md (stage 1). Op verzoek van Janco is deze analyse niet bedoeld voor de website: de inhoudelijke regels zijn gevolgd, de woordentellingen voor JSON-velden niet.

## Bronnen-inventaris (Stap 0.5)

Perimeter: WISA is de voormalige business area UPM Plywood. Tot en met 2022 rapporteerde UPM het segment inclusief de Russische fabriek Chudovo (verkocht in 2023) en tot en met 2020 inclusief Jyväskylä (gesloten in 2020). Vanaf 2023 zijn het de geauditeerde carve-outcijfers van WISA zelf, zonder die twee fabrieken. De reeks is dus niet volledig vergelijkbaar; zie sectie 3.

| Jaar | Bron (URL) | Betrouwbaarheid | Cijfers uit deze bron |
|---|---|---|---|
| 2015, 2016 | https://www.upm.com/siteassets/asset/investors/2016/upm-results-2016-en.pdf | HOOG | Omzet, vergelijkbare EBITDA, afschrijvingen, bedrijfsresultaat, vergelijkbare EBIT, gemiddeld geïnvesteerd vermogen, ROCE, leveringen |
| 2017, 2018 | https://www.upm.com/siteassets/asset/investors/2018/upm-results-2018-en.pdf | HOOG | idem |
| 2019, 2020 | https://www.upm.com/siteassets/asset/investors/2020/upm-results-2020-en.pdf | HOOG | idem |
| 2021, 2022 | https://www.upm.com/siteassets/asset/investors/2022/upm-financial-statements-release-2022-en.pdf | HOOG | idem |
| 2023, 2024, 2025, H1 2025, H1 2026 | https://www.upm.com/contentassets/7b217c3378a04f0a9dbc998aafdabd77/wisa-group-plc-supplement-to-the-prospectus-17-august-2026.pdf | HOOG (FY geauditeerd, H1 niet) | Volledige W&V, balans, kasstroomoverzicht incl. betaalde belasting, kerncijfers, pro-formabalans 30-6-2026, nettoschuld, aantal aandelen, guidance 2026 |
| Q3 2026, 9M 2026, Q4 2025 | https://mb.cision.com/Main/23100/4416385/4342561.pdf (UPM Interim Report Q3 2026) | HOOG | Segmentcijfers Plywood per kwartaal |
| Koers | https://query1.finance.yahoo.com/v8/finance/chart/WISA.HE | AGGREGATOR | Slotkoersen en volumes 2 t/m 10 november 2026 |

GEEN BRON BESCHIKBAAR voor 2015-2022: kasstroomoverzicht, betaalde belasting, capex en balans op segmentniveau. UPM publiceert die niet per business area. Die cellen blijven leeg. De FCF-normalisatie loopt daarom via de margeroute (11 jaar EBIT-marge) met de FCF-marge 2023-2025 als controle.

Haallijst: niet nodig. Alle recente vijf jaar staan op HOOG. Wel nuttig maar niet gevonden: het volledige prospectus van 16 juli 2026, met een grafiek van omzet en EBITDA 2015-2025 zónder Chudovo en Jyväskylä. Zie Opmerkingen.

---

## Metadata
- **Ticker (bare):** WISA
- **Yahoo symbol:** WISA.HE
- **Exchange:** HEX
- **Sector (GICS-achtig):** Materialen
- **Industrie:** Houtproducten (triplex en fineer)
- **Land:** Finland
- **Peildatum analyse:** 2026-11-10
- **Koers op peildatum:** 0,529
- **Valuta:** EUR
- **Marktkapitalisatie:** EUR 279 mln
- **Marktkap in mln (lokale valuta):** 279
- **Free float pct:** onbekend; aandelen zijn 1-op-1 uitgekeerd aan UPM-aandeelhouders, dus vrijwel volledig vrij verhandelbaar
- **Indexlidmaatschap:** onbekend (niet geverifieerd)
- **Domein:** wisa.com

---

## 1. Executive summary

**Oordeel: HOLD** (scorekaart 29/45).

WISA maakt premium berken- en vurentriplex in zeven fabrieken in Finland en Estland. UPM heeft het bedrijf op 31 oktober 2026 afgesplitst en sinds 2 november 2026 noteert het in Helsinki. Het bedrijf is cyclisch: de vergelijkbare EBIT-marge schommelde tussen 2015 en 2025 van 7,7% tot 20,2%.

De waardering is laag. Tegen € 0,529 betaal je 5,5 keer de EBITDA van de laatste twaalf maanden en 9 keer de genormaliseerde winst. De vrije kasstroom is ongeveer 11% van de beurswaarde. Mijn basis-DCF komt op € 1,01. De synthese van DCF, multiples en EPV (de waarde zonder groei) komt op € 0,79. Ook de EPV van € 0,71 ligt boven de koers: je betaalt dus niets voor groei.

Dat het toch HOLD wordt en geen KOOP, komt door de kwaliteitsframeworks. Het rendement op geïnvesteerd vermogen ligt elk jaar boven de WACC, maar niet ruim genoeg om als uitzonderlijk te tellen. De moat is smal (alleen in LNG-triplex echt sterk) en het management heeft als zelfstandig bedrijf nog geen trackrecord. Waarderingsscores van 5 kunnen dat niet compenseren tot de grens van 33.

Belangrijkste onzekerheid: wat een normale marge is zonder de goedkope Russische fabriek en mét Russisch triplex dat na een eventueel einde van de oorlog terug op de Europese markt komt.

---

## 2. Bedrijfsprofiel

WISA Group verkoopt triplex en fineer onder het merk WISA voor drie hoofdtoepassingen. De marktaandelen komen uit de CMD van 23 september 2026 (Quartr-samenvatting, niet in een primaire bron gecontroleerd):
- **Isolatie van LNG-tankers**, ongeveer 22% marktaandeel. Volgens de CMD is WISA een van de weinige leveranciers wereldwijd. De klanten zijn Aziatische scheepswerven; in 2022 tekende UPM Plywood een groot contract met een Koreaanse werf (UPM FS 2022).
- **Vloeren van bedrijfswagens en trailers**, ongeveer 34%.
- **Groothandel in panelen en de bouw**, ongeveer 19%.

Kerncijfers: omzet € 409 mln in 2025, 1.519 medewerkers op 30 juni 2026. CEO Tuija Suur-Hamari kwam uit UPM's Group Executive Team; CFO is Lasse von Hertzen.

Geschiedenis in het kort: UPM sloot de fabriek in Jyväskylä in 2020. Na de Russische inval schreef UPM in 2022 € 63 mln af op Chudovo en verkocht die fabriek in 2023. In het voorjaar van 2025 lag de Finse productie 10 weken stil door een cao-staking. Op 29 april 2026 keurde het bestuur van UPM de afsplitsing goed.

### Geografische spreiding (omzet)
Niet opgehaald. De CMD noemt Europa en de wereldwijde LNG-markt als kernmarkten.

### Segmenten
Eén segment. Eindmarkten zie hierboven.

### Aandeelhouders (top 5)
Niet geverifieerd. De aandeelhouders zijn dezelfde als die van UPM op de registratiedatum (1 WISA-aandeel per UPM-aandeel). Op 3 november 2026 volgde een eerste melding van gewijzigde zeggenschap; de inhoud is niet opgehaald.

---

## 3. Financieel — historische data (10 jaar + TTM)

### Resultatenrekening (bedragen in EUR mln)

| Jaar | Perimeter | Omzet | Vergl. EBITDA | Vergl. EBIT | EBIT-marge % | Bedrijfsresultaat | Nettowinst | Leveringen (1000 m³) |
|---|---|---|---|---|---|---|---|---|
| 2015 | UPM-segment | 439 | 78 | 55 | 12,5 | 53 | | 740 |
| 2016 | UPM-segment | 444 | 80 | 58 | 13,1 | 58 | | 764 |
| 2017 | UPM-segment | 484 | 85 | 62 | 12,8 | 62 | | 811 |
| 2018 | UPM-segment | 480 | 75 | 52 | 10,8 | 52 | | 791 |
| 2019 | UPM-segment | 450 | 61 | 36 | 8,0 | 36 | | 739 |
| 2020 | UPM-segment | 405 | 59 | 33 | 8,1 | 10 | | 683 |
| 2021 | UPM-segment | 492 | 99 | 72 | 14,6 | 80 | | 738 |
| 2022 | UPM-segment | 539 | 133 | 109 | 20,2 | 44 | | 616 |
| 2023 | Carve-out | 421,8 | 73,4 | 52,8 | 12,5 | 46,6 | 41,7 | |
| 2024 | Carve-out | 429,9 | 60,6 | 38,4 | 8,9 | 38,4 | 34,2 | |
| 2025 | Carve-out | 409,1 | 50,7 | 31,3 | 7,7 | 26,9 | 24,6 | |
| H1 2026 | Carve-out | 240,1 | 33,2 | 24,0 | 10,0 | 13,1 | 8,7 | |
| 9M 2026 | UPM-segment | 345 | 44 | 37* | 10,9* | 23* | | 385 |

\* Door IFRS 5 is vanaf mei 2026 niet meer afgeschreven. Bij een normale afschrijving van ongeveer € 5 mln per kwartaal ligt de onderliggende EBIT over 9M 2026 rond € 26 tot 29 mln. Q3 2026 alleen: omzet 105, vergelijkbare EBIT 9, onderliggend ongeveer 3 à 4.

TTM tot en met Q2 2026 (carve-out, prospectus): omzet 457, vergelijkbare EBITDA 67,2, vergelijkbare EBIT 49,3.

### Kasstromen

| Jaar | CFO | waarvan ontvangen rente | Betaalde belasting | Belastinglast W&V | Capex | FCF (CFO − rente − capex) | FCF-marge % |
|---|---|---|---|---|---|---|---|
| 2023 | 48,8 | 4,7 | 7,0 | 10,0 | 16,2 | 27,9 | 6,6 |
| 2024 | 68,0 | 5,4 | 8,0 | 9,3 | 16,5 | 46,1 | 10,7 |
| 2025 | 44,3 | 3,7 | 6,9 | 6,0 | 8,2 | 32,4 | 7,9 |
| H1 2026 | −1,3 | 0,6 | 5,3 | 5,9 | 2,5 | −4,4 | |

De ontvangen rente kwam uit de cashpool bij UPM. Die verdwijnt na de afsplitsing en is daarom van de CFO afgetrokken. Leasebetalingen zijn € 0,9 à 1,3 mln per jaar.

### Balans-ratio's (10 jaar)

| Jaar | Gem. geïnvesteerd vermogen | Vergl. ROCE % (voor belasting) | ROIC % (na 20% belasting) |
|---|---|---|---|
| 2015 | 263 | 20,9 | 16,7 |
| 2016 | 259 | 22,6 | 17,9 |
| 2017 | 267 | 23,1 | 18,6 |
| 2018 | 283 | 18,4 | 14,7 |
| 2019 | 329 | 11,0 | 8,8 |
| 2020 | 292 | 11,2 | 9,0 |
| 2021 | 286 | 25,1 | 20,1 |
| 2022 | 247 | 44,3 | 35,3 |
| 2023 | 233,5 | 22,6 | 18,1 |
| 2024 | 222,2 | 17,3 | 13,8 |
| 2025 | 209,5 | 14,9 | 12,0 |

Gemiddelde ROIC 16,8%, mediaan 16,7%, laagste 8,8% (2019). Alle elf jaren boven de WACC van 8,3%.

### Kapitaalstructuur huidig
Pro forma per 30 juni 2026 (prospectussupplement):
- Termijnlening € 130 mln, waarvan € 59,9 mln de overname is van bestaande UPM-schuld. Verwachte rentelast € 4,7 mln per jaar (ongeveer 3,6%).
- Kredietfaciliteit € 50 mln, niet opgenomen.
- Bruto schuld € 135,9 mln (inclusief leases), kas € 42,2 mln, **nettoschuld € 93,7 mln**. Dat is 1,4 keer de vergelijkbare EBITDA.
- Eigen vermogen pro forma € 100,6 mln (€ 0,191 per aandeel).
- Voorzieningen € 14,3 mln, onder meer voor milieu. Behandeld als schuldachtig in de IPO-correctie.
- Aandelen: 527.324.046.

### Non-GAAP / aanpassingen
Vergelijkbare EBIT is exclusief bijzondere posten: kosten van afsplitsing en notering (€ 10,8 mln in H1 2026, € 23 mln in totaal), herstructurering (€ 4,1 mln in 2025) en verkoopresultaat Chudovo (−€ 5,3 mln in 2023). Volgens de pro forma liggen de structurele kosten van een zelfstandig bedrijf al grotendeels in de carve-out: de pro forma vergelijkbare EBIT 2025 is € 31,6 mln tegenover € 31,3 mln in de carve-out.

---

## 4. Moat (concurrentievoordeel)

**Oordeel: NARROW MOAT**

| Categorie | Oordeel | Onderbouwing |
|---|---|---|
| Immateriële activa | STERK | Het merk WISA en de certificeringen voor LNG-isolatie. Volgens de CMD zijn er maar weinig gekwalificeerde leveranciers. |
| Overstapkosten | MATIG | Werven en trailerbouwers ontwerpen op een gekwalificeerd product. Buiten LNG is triplex grotendeels uitwisselbaar. |
| Netwerkeffecten | GEEN | Niet van toepassing. |
| Kostenvoordeel | ZWAK | Fins en Baltisch berkenhout is duurder dan Russisch hout. Met Chudovo verdween het goedkoopste deel van de productie. |
| Efficiënte schaal | MATIG | De markt voor LNG-triplex is klein en gespecialiseerd, dus voor een nieuwkomer onaantrekkelijk. In de bouw speelt dit niet. |

Kwantitatief: ROIC elf jaar op rij boven de WACC, gemiddeld 8,5 procentpunt erboven. Maar de jaren na Chudovo zitten lager: 18,1%, 13,8% en 12,0% (2025 door de staking). De marge van 20% in 2022 kwam vooral door de EU-sancties op Russisch berkentriplex en is niet structureel.

---

## 5. Management

**Oordeel: NEUTRAAL**

- CEO Tuija Suur-Hamari en het directieteam komen uit UPM. Als zelfstandig beursbedrijf hebben ze nog geen trackrecord.
- De directie en de raad hielden op 13 augustus 2026 samen 267.430 UPM-aandelen, ongeveer 0,05%. Dat zijn nu evenveel WISA-aandelen, ter waarde van circa € 0,14 mln.
- Op 1 november 2026 besloot de raad tot een langetermijnbeloningsplan met een eigendomsrichtlijn voor de directie. Het plan gebruikt bestaande aandelen, met een plafond van € 2 mln (volgens het Inderes-forum, niet primair gecontroleerd).
- Insidertransacties op de open markt na de notering: geen gevonden.
- Kapitaalallocatie onder UPM: weinig investeringen (capex ongeveer 2% à 4% van de omzet), sluiting van Jyväskylä, gedwongen verkoop van Chudovo. UPM liet tussen 2023 en H1 2026 € 235 mln dividend uitkeren en haalde de cashpoolvordering van € 52,4 mln terug. Dat was het besluit van UPM, niet van dit management.
- Uitkeringsbeleid: ongeveer 50% van de winst.

---

## 6. Sector & concurrentie

- Triplex is een cyclische markt die meebeweegt met de bouw, de transportsector en de bouw van LNG-tankers.
- Volgens de CMD waren de kernmarkten in 2025 € 3,1 mrd waard, met een verwachte groei van 5,3% per jaar tot 2030. Dat is een eigen schatting van het bedrijf.
- Bekende concurrenten in berkentriplex zijn Metsä Wood, Latvijas Finieris, Koskisen en Sveza uit Rusland. Russisch berkentriplex mag sinds 2022 de EU niet meer in. Marktaandelen van deze concurrenten zijn niet opgehaald.
- Q3 2026 volgens UPM: de vraag naar LNG-toepassingen was sterk en naar bedrijfswagenvloeren goed. Bij paneelhandel en bouw werd de markt lastiger. De variabele kosten stegen.

### TAM/SAM/SOM
TAM volgens het bedrijf € 3,1 mrd (2025). Met een omzet van ongeveer € 450 mln is het aandeel ongeveer 15%.

---

## 7. Analyse-frameworks (9 frameworks, SCORES 1-5)

### Graham
P/E 9,0 (genormaliseerd), P/B 2,77. Score 5 en 4 vallen af op P/B > 2,0. Score 3: P/E ≤ 20 klopt. **3/5**

### Buffett / Munger
ROIC elf jaar boven de WACC van 8,3%, maar niet structureel boven het dubbele (16,6%): dat lukte in 6 van de 11 jaren en na Chudovo in 1 van de 3. Moat NARROW, P/FCF 9,0. **3/5**

### Peter Lynch
P/E op het midden van de guidance 2026 is 9,6. Verwachte winstgroei per aandeel in het basisscenario: 5,1% per jaar. PEG 1,89. **2/5**
Wie de doelen voor 2030 als groei neemt, komt op een PEG van ongeveer 0,6. Daar is nu nog geen bewijs voor.

### Phil Fisher
Uitgaven aan R&D zijn niet gerapporteerd. Het management is als zelfstandig bedrijf nog niet te beoordelen. De moat beschermt de marge alleen in LNG; de marge zelf schommelt tussen 7,7% en 20%. **2/5**

### Magic Formula
Earnings yield (EBIT/EV) 11,7%. Return on capital (EBIT gedeeld door werkkapitaal plus materiële vaste activa, 181) 24,1%. Score 4 vraagt een return on capital van minstens 30%. Score 3: earnings yield ≥ 5%. **3/5**

### Moat
NARROW (1 categorie STERK) en een gemiddelde ROIC-WACC-spread van 8,5 procentpunt (> 5). **3/5**

### Management
Kapitaalallocatie als zelfstandig bedrijf: nog niet te beoordelen. Prikkels: deels in lijn met aandeelhouders. **3/5**

### Fair Value DCF
Basis € 1,009 tegen € 0,529: 91% opwaarts. **5/5**

### Fair Value IPO-gecorr.
Afsplitsing ≤ 10 jaar geleden. Gecorrigeerde basis € 0,982 (zie sectie 8): 86% opwaarts. **5/5**

### Scorekaart totaal

| Framework | Score | Oordeel |
|---|---|---|
| Graham | 3 / 5 | P/E ≤ 20, P/B 2,77 te hoog voor 4 |
| Buffett / Munger | 3 / 5 | ROIC > WACC structureel, niet > 2×WACC |
| Peter Lynch | 2 / 5 | PEG 1,89 |
| Phil Fisher | 2 / 5 | Geen aantoonbare R&D-voorsprong, marge volatiel |
| Magic Formula | 3 / 5 | EY 11,7%, ROC 24% |
| Moat | 3 / 5 | NARROW, spread 8,5 pp |
| Management | 3 / 5 | Nieuw als beursbedrijf, prikkels deels in lijn |
| Fair Value DCF | 5 / 5 | +91% |
| Fair Value IPO-gecorr. | 5 / 5 | +86% |
| **Totaal** | **29 / 45** | **HOLD** |

Hoe robuust is dit oordeel? De DCF-score blijft 5 tot een WACC van 10,3% (basis € 0,73, nog 38% opwaarts). PASS komt pas in beeld als de basiswaarde onder € 0,46 zakt. Dat vraagt een structurele marge van ongeveer 7% én een WACC boven 10%. KOOP (33) vraagt vier punten meer. Die moeten uit de kwaliteitsframeworks komen: Graham 4 bij een P/B onder 2,0 (koers ≤ € 0,38), Buffett 4 bij aantoonbaar ROIC > 16,6% na Chudovo, Lynch 3 bij een aannemelijk winstgroeipad van ruim 6% per jaar.

Eindalinea: het oordeel is HOLD, maar het profiel wijkt af van een gewone HOLD. De waarderingsframeworks staan op het maximum en de kwaliteitsframeworks in het midden. Wat je moet volgen is de normale marge na Chudovo: 2023 haalde 12,5%, 2024 8,9%, en 2025 en 2026 zijn vertekend door de staking en de afsplitsing. Op korte termijn drukt verkoop door indexfondsen en UPM-aandeelhouders mogelijk de koers. Positief is dat het eerste dividendvoorstel en de eerste zelfstandige jaarcijfers in februari 2027 komen. Omdat de bronnen goed zijn maar de perimeter is veranderd, is een veiligheidsmarge van 30% gepast: koopniveau € 0,56 op de synthesewaarde. De huidige koers ligt daar net onder.

---

## 8. Risico's (minimaal 5-8 stuks)

| # | Risico | Kans | Impact | DCF-aanname |
|---|---|---|---|---|
| 1 | Einde oorlog in Oekraïne, Russisch berkentriplex komt terug op de EU-markt en drukt prijzen | MIDDEN | GROOT | Genormaliseerde EBIT-marge (10% → 8%: basis € 0,78) |
| 2 | Aanhoudende zwakte in Europese bouw en paneelhandel | HOOG | MIDDEL | Groei fase 1, marge |
| 3 | Cyclus van LNG-tankers keert na de huidige orderpiek | MIDDEN | MIDDEL | Groei fase 1 en 2 |
| 4 | Nieuwe stakingen in Finland (2025: 10 weken stil) | MIDDEN | MIDDEL | Normalisatie, kasstroom per jaar |
| 5 | Stijgende houtprijzen (berkenstammen) | HOOG | MIDDEL | Marge |
| 6 | Extra kosten als zelfstandig bedrijf en IT-scheiding | MIDDEN | KLEIN | Marge (−0,5 pp ≈ −€ 0,04) |
| 7 | Verkoopdruk na de afsplitsing en lage belangstelling van analisten | HOOG | KLEIN | Geen; koers op korte termijn |
| 8 | Pre-IPO financial engineering (zie onder) | MIDDEN | MIDDEL | Nettoschuld |

**Verplicht: pre-IPO financial engineering**
- Schulden geladen? Ja. WISA kreeg een termijnlening van € 130 mln, waarvan € 59,9 mln de overname van UPM-schuld is, en betaalde € 11,7 mln aan UPM voor een milieuverplichting.
- Opbrengsten naar insiders? Er was geen emissie. Wel keerde WISA in 2023 t/m H1 2026 € 235 mln dividend uit aan UPM en ging de cashpoolvordering van € 52,4 mln terug naar UPM.
- Dividend-recapitalisatie? In feite wel: het bedrijf ging van nettokas naar 1,4 keer nettoschuld/EBITDA.
- Correctie: de nettoschuld van € 93,7 mln zit al in de DCF. Daarnaast trek ik de voorzieningen van € 14,3 mln af als schuldachtig. IPO-gecorrigeerde basis: € 0,982.

---

## 9. These invalide bij
- Een vergelijkbare EBIT-marge die twee jaar op rij onder 8% blijft, terwijl de LNG-markt goed loopt.
- Nettoschuld/EBITDA boven 2,5 keer.
- EU-sancties op Russisch triplex vervallen zonder vervangende handelsmaatregelen.
- Een grote overname tegen een hoge prijs in de eerste twee jaar.

---

## 10. ESG

### Materiële factoren (SASB-gebaseerd)
- Duurzaam bosbeheer en traceerbare houtketen: het bedrijf noemt dit een kern van de strategie (CMD). Het risico zit in hout uit sanctiegebieden dat via derde landen binnenkomt bij concurrenten.
- Energie in de fabrieken (drogen en persen) en veiligheid in de productie: niet opgehaald.
- Milieuvoorzieningen voor eerder gesloten fabrieken: € 14,3 mln aan voorzieningen per 30-6-2026.

---

## 11. Katalysatoren (5-8 stuks, chronologisch)

| Datum (ca.) | Katalysator | Richting | Impact |
|---|---|---|---|
| nov-dec 2026 | Verkoop door indexfondsen en UPM-aandeelhouders; eerste analistendekking | NEGATIEF | MIDDEL |
| feb 2027 | Eerste zelfstandige jaarcijfers 2026 en eerste dividendvoorstel (datum niet geverifieerd) | BINAIR | MIDDEL |
| voorjaar 2027 | Ontwikkeling bouwmarkt in Europa | NEUTRAAL | MIDDEL |
| apr 2027 | Q1 2027: eerste kwartaal met volledige zelfstandige kosten en afschrijving | BINAIR | MIDDEL |
| 2027 | Ontwikkeling orderboek van LNG-tankers | POSITIEF | MIDDEL |
| onbepaald | Wapenstilstand in Oekraïne en verlenging of afschaffing van EU-sancties | BINAIR | GROOT |
| 2027-2028 | Overnames als groeimogelijkheid (genoemd op de CMD) | NEUTRAAL | KLEIN |

---

## 12. Fair value — kwantitatief (DCF)

### DCF-invoeren

```
Basis            fcf=35.0  shares=527.324  net_cash=-93.7  gross_debt=135.9  revenue=435  koers=0.529  ipo_jaar=2026
WACC             rf=3.27  erp=4.41  beta=1.09  crp=0.36  size_premium=2.5  cost_of_debt_pretax=3.6  tax_rate=20
Pessimistisch    g1=0.0  g2=1.0  gt=1.5  wacc_adj=1.0  kans=30   (basis_fcf=28.3)
Basis            g1=3.0  g2=2.5  gt=2.0  wacc_adj=0.0  kans=50
Optimistisch     g1=8.0  g2=3.0  gt=2.5  wacc_adj=0.0  kans=20
EPV              norm_ebit_margin=10.0  maintenance_capex=16  da=20  norm_ebitda_margin=14.6
Multiples        pe=9.0  pb=2.77  p_fcf=9.0  peg=1.89
Rendement        roic=16.8  earnings_yield=11.7  roc_greenblatt=24.1
Kwalitatief      moat_oordeel=NARROW MOAT  moat_categorieen_sterk=1  management_oordeel=NEUTRAAL
                 capital_allocation=GEMENGD  insider_alignment_pct=0.05
                 roic_wacc_spread_5j_plus=ja (gem. 8,5 pp)  structureel_dividend=nee (nog geen historie)  debt_equity=1.35
Eenheid          bedragen in EUR mln
```

Fase 1 = jaar 1-5, fase 2 = jaar 6-10, daarna Gordon. Mid-year convention. Het pessimistische scenario start op een lagere basis-FCF (8% marge).

### WACC-componenten
- **Risicovrije rente %**: 3,27
- **Bron**: Duitse 10-jaarsrente (Bund), Trading Economics, 9-11-2026. Kasstromen in EUR, dus volgens H7-4A de Bund. Finse 10-jaars: 3,65%.
- **Type**: nominaal, spot. De rente ligt meer dan 150 bp boven het 10-jaarsgemiddelde; de variant met WACC −1 pp staat in de gevoeligheid (basis € 1,23).
- **ERP %**: 4,41
- **Bron ERP**: Damodaran, implied ERP per 1 november 2026 (trailing 12m, adjusted payout)
- **Beta**: 1,09 (relevered)
- **Bron beta**: bottom-up. Gemiddelde van de unlevered bèta van Global Paper/Forest Products, gecorrigeerd voor kas (0,61) en US Building Materials (0,96): 0,785. Relevered met D/E 0,487 op marktwaarde en 20% belasting.
- **Type beta**: bottom-up (er is geen koershistorie)
- **Country risk premium %**: 0,36 (Damodaran, Finland Aa1)
- **Size premium %**: 2,5 (beurswaarde € 279 mln)
- **Cost of equity %**: 10,94
- **Schuldkosten na belasting %**: 2,88 (3,6% voor belasting)
- **E/V gewicht %**: 67,2
- **D/V gewicht %**: 32,8
- **WACC %**: 8,30
- **Sector WACC %**: niet opgehaald
- **Illiquiditeitskorting %**: null (dagvolume 13 tot 65 mln aandelen)

### DCF model-specs
- **Model type**: 2-fase plus Gordon
- **FCF-definitie**: FCFF
- **Basis FCF**: € 35,0 mln, genormaliseerd mid-cycle
- **Basis FCF na SBC**: ja, € 1 mln SBC afgetrokken
- **FCF-type**: adjusted (opgebouwd uit componenten)
- **Groei fase 1 %**: 3,0
- **Groei fase 2 %**: 2,5
- **Terminal groei %**: 2,0
- **Terminal methode**: Gordon, met controle via exit-multiple
- **Exit multiple gebruikt**: 7,0x EV/EBITDA
- **Bron exit multiple**: Damodaran US Paper/Forest Products EV/EBITDA 8,2x (trailing, jan 2026), naar beneden bijgesteld voor Europese smallcap en een gemiddelde over de cyclus
- **Terminal value Gordon growth**: € 743 mln
- **Terminal value exit multiple**: € 583 mln (basiswaarde dan € 0,87)
- **Terminal value % van totaal**: 56%
- **Terminal implied EV/EBITDA**: 8,7x
- **Terminal groei consistentie**: 2% groei bij een ROIC van ongeveer 16% vraagt een herinvestering van 12,5% van de NOPAT. Dat past bij de capex-historie (capex boven onderhoud plus groei van het werkkapitaal). Lager dan de nominale groei van de economie in de eurozone.
- **Mid-year convention**: true
- **Aandelen uitstaand (mln)**: 527,324
- **Nettoschuld huidig**: € 93,7 mln (pro forma 30-6-2026)

### DCF-toelichting
WISA is cyclisch, dus REGEL 1 tot en met 4 zijn toegepast. Ik ga uit van een genormaliseerde omzet van € 435 mln (2023-2025 en TTM liggen tussen 409 en 457) en een EBIT-marge van 10,0%. De vergelijkbare EBIT-marge over 2015-2025 was gemiddeld 11,8% en in de mediaan 12,5%; zonder piekjaar 2022 gemiddeld 10,9%, en sinds het vertrek van Chudovo gemiddeld 9,7%. Dat laatste is inclusief stakingsjaar 2025. Het midden van de guidance 2026 (€ 41 mln) is 9,1%. 10% weegt de kleinere perimeter zwaarder dan de lange historie.

Opbouw FCFF: NOPAT 34,8 + afschrijving 20 − capex 16 − groei werkkapitaal 1,3 − leases 1,2 − SBC 1,0 = 35,3, afgerond 35,0. Controle REGEL 4: mediane FCF-marge 2023-2025 (7,9%) × 435 = 34,5, een afwijking van 1,5%. REGEL 3: piek-FCF 46,1 (2024), dal 27,9 (2023), gemiddelde 35,5. Gekozen 35,0 ligt 8% boven de meest recente FCF (2025: 32,4), dus geen rode vlag.

REGEL 5: de betaalde belasting was in 2023 en 2024 lager dan de last (€ 15,0 mln tegen € 19,3 mln) en in 2025 hoger (6,9 tegen 6,0). Cumulatief betaalde WISA 87% van de last. Geen structurele afwijking, dus gerekend met het wettelijke Finse tarief van 20%.

Er is geen FCF van vóór 2023 beschikbaar op segmentniveau. Daarom loopt de normalisatie via de marge en niet via een FCF-gemiddelde over 10 jaar.

### 5-jaars projectie (basis)

| Jaar | Omzet | Omzetgroei % | EBIT | EBIT-marge % | NOPAT | Capex | ΔNWC | SBC | FCF |
|---|---|---|---|---|---|---|---|---|---|
| 2026 (genorm.) | 435,0 | | 43,5 | 10,0 | 34,8 | 16 | 1,3 | 1,0 | 35,0 |
| 2027 | 448,1 | 3,0 | 44,8 | 10,0 | 35,8 | | | | 36,1 |
| 2028 | 461,5 | 3,0 | 46,1 | 10,0 | 36,9 | | | | 37,1 |
| 2029 | 475,3 | 3,0 | 47,5 | 10,0 | 38,0 | | | | 38,2 |
| 2030 | 489,6 | 3,0 | 49,0 | 10,0 | 39,2 | | | | 39,4 |

Ter vergelijking: het doel voor 2030 is een omzet boven € 550 mln en een EBIT-marge boven 13%, dus een EBIT van € 71 mln. Het basisscenario rekent daar niet mee.

### Scenarios (3 stuks — exact deze labels)

| Scenario | FCF-groei % | WACC % | Fair value | Upside % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 0 / 1 / 1,5 (start-FCF 28,3 bij 8% marge) | 9,30 | 0,498 | −5,9 | 30 |
| Basis | 3 / 2,5 / 2 | 8,30 | 1,009 | +90,8 | 50 |
| Optimistisch | 8 / 3 / 2,5 | 8,30 | 1,395 | +163,7 | 20 |

- **Kansgewogen fair value**: € 0,933 (+76%)
- Afwijking van 25/50/25: de bouwmarkt wordt zwakker (Q3 2026), en terugkeer van Russisch triplex is een reëel neerwaarts risico. Het optimistische scenario vraagt dat de doelen voor 2030 grotendeels gehaald worden, iets wat de historie maar één keer liet zien (2022, door sancties).

### Reverse DCF
- **Impliciete groei %**: −8,5% per jaar in fase 1 (met fase 2 op 2,5% en terminal 2%). Bij nul groei volstaat een FCF van € 29,7 mln om de koers te rechtvaardigen.
- **Historische FCF CAGR %**: niet zinvol (3 jaar data)
- **Consensus groei %**: niet gevonden
- **Interpretatie**: de koers rekent met een structurele FCF van ongeveer € 30 mln zonder groei. Dat is een marge van ongeveer 8,4%, vergelijkbaar met 2024 en net boven het stakingsjaar 2025. De markt prijst dus het onderste deel van de cyclus in.

### EPV (Bruce Greenwald)
- **Genormaliseerde EBIT-marge %**: 10,0
- **Genormaliseerde NOPAT**: € 34,8 mln
- **Maintenance capex**: € 16 mln (capex 2023-2025 was gemiddeld 13,6; afschrijving 20,6)
- **Adjusted earnings power**: € 38,8 mln
- **EPV per aandeel**: € 0,709 (EPV € 467 mln min nettoschuld)
- **Groeipremie %**: de koers ligt 25% onder de EPV. De DCF-basis ligt 42% boven de EPV.

### Andere methoden
- **DDM uitgevoerd?**: false (nog geen dividendhistorie). Indicatie: 50% uitkering van genormaliseerde winst (€ 31 mln) geeft € 15,5 mln, een dividendrendement van 5,5%.
- **SOTP uitgevoerd?**: false
- **Multiples**: 7x genormaliseerde EBITDA (63,5) geeft € 0,665; 10x genormaliseerde EBIT geeft € 0,647. Gemiddeld € 0,656.

### Synthese fair value
- **Bandbreedte laag**: € 0,50 (DCF pessimistisch)
- **Bandbreedte centraal**: € 0,79
- **Bandbreedte hoog**: € 1,01 (DCF basis)
- **Methode-gewichten**:
  - DCF %: 35
  - EPV %: 25
  - Multiples %: 40
- **Margin of safety vereist %**: 30
- **Koopniveau**: € 0,555
- **Synthese-toelichting**: de DCF-basis ligt ruim 50% boven de multiples. De bèta is hier niet de oorzaak (1,09, niet laag). Het verschil zit erin dat de DCF een eeuwigdurende rendementseis van 8,3% hanteert, terwijl de markt Europese houtproducenten op een FCF-rendement rond 9% à 10% prijst. Daarom krijgen de multiples, net als bij TMUS, het zwaarste gewicht. Een MOS van 30% omdat de perimeter na Chudovo pas drie jaar data heeft en het bedrijf cyclisch is.

### Gevoeligheid (DCF)
- WACC range: 7,0 / 7,5 / 8,3 / 9,0 / 10,0 / 11,0
- Groei fase 1 range: 0 / 1,5 / 3 / 4,5 / 6 (fase 2 = 2,5, terminal = 2)

| g1 \ WACC | 7,0 | 7,5 | 8,3 | 9,0 | 10,0 | 11,0 |
|---|---|---|---|---|---|---|
| 0,0 | 1,12 | 1,01 | 0,86 | 0,76 | 0,65 | 0,56 |
| 1,5 | 1,22 | 1,09 | 0,93 | 0,83 | 0,70 | 0,61 |
| 3,0 | 1,31 | 1,18 | 1,01 | 0,89 | 0,76 | 0,66 |
| 4,5 | 1,41 | 1,27 | 1,09 | 0,96 | 0,82 | 0,71 |
| 6,0 | 1,52 | 1,37 | 1,17 | 1,04 | 0,89 | 0,77 |

Gevoeligheid voor de marge (basis, WACC 8,3%): 8% geeft € 0,78; 12% geeft € 1,26. De marge is de belangrijkste invoer.

---

## 13. Databronnen

### Bronnen-hiërarchie
Primair: UPM-jaarcijfers en kwartaalberichten (segment Plywood) en het prospectussupplement van WISA. Secundair: Quartr-samenvatting van de CMD, Inderes-forum (alleen context, geen cijfers gebruikt die niet primair bevestigd zijn), Damodaran, Trading Economics, Yahoo.

### Financiële bronnen (10 jaar historie — VERPLICHT)
Zie de bronnen-inventaris bovenaan. 2015-2022 HOOG (UPM-segment), 2023-2025 HOOG (geauditeerde carve-out), H1 en Q3 2026 HOOG (niet geauditeerd).

### Jaarverslagen geraadpleegd
UPM Financial Statements Release 2016, 2018, 2020, 2022 (elk met het voorgaande jaar als vergelijking).

### Beursmeldingen geraadpleegd
UPM Interim Report Q3 2026; prospectussupplement 17-8-2026; Nasdaq-bericht notering 2-11-2026; nieuwsoverzicht finanznachrichten.de.

### IPO-prospectus
Supplement van 17 augustus 2026 gelezen. Het hoofdprospectus van 16 juli 2026 is niet gevonden.

### Non-GAAP
Vergelijkbare EBITDA en EBIT volgens de definities van WISA/UPM. Reconciliatie in het supplement.

### Ontbrekende data (eerlijke lijst)
- Kasstroom, capex, betaalde belasting en balans per segment voor 2015-2022 (niet gepubliceerd door UPM).
- Omzet en EBITDA 2015-2022 zonder Chudovo en Jyväskylä (staat als grafiek in het hoofdprospectus).
- Omzet per regio en per eindmarkt in euro's.
- Top 5 aandeelhouders, free float, indexlidmaatschap.
- Analistenconsensus en koersdoelen.
- Datum jaarcijfers 2026.
- Kasstroom en nettoschuld per 30-9 of 31-10-2026.

### Peildatum analyse
2026-11-10, slotkoers € 0,529 (Nasdaq Helsinki via Yahoo).

---

## 14. Volledige bronnen-lijst
1. https://www.upm.com/siteassets/asset/investors/2016/upm-results-2016-en.pdf
2. https://www.upm.com/siteassets/asset/investors/2018/upm-results-2018-en.pdf
3. https://www.upm.com/siteassets/asset/investors/2020/upm-results-2020-en.pdf
4. https://www.upm.com/siteassets/asset/investors/2022/upm-financial-statements-release-2022-en.pdf
5. https://www.upm.com/contentassets/7b217c3378a04f0a9dbc998aafdabd77/wisa-group-plc-supplement-to-the-prospectus-17-august-2026.pdf
6. https://mb.cision.com/Main/23100/4416385/4342561.pdf
7. https://quartr.com/events/upm-kymmene-corporation-upm-wisa-group-cmd-2026_33NZ5W7F
8. https://www.inderes.fi/en/releases/upm-kymmene-oyj-the-finnish-financial-supervisory-authority-has-approved-wisa-group-plcs-demerger-and-listing-prospectus-wisa-group-plcs-group-leadership-team-appointed
9. https://www.finanznachrichten.de/nachrichten-2026-11/70002522-nasdaq-helsinki-welcomes-wisa-group-to-main-market-252.htm
10. https://www.finanznachrichten.de/nachrichten-aktien/wisa-group-oyj.htm
11. https://forum.inderes.com/t/wisa-group-premium-vaneria-pohjolasta-maailmanmarkkinoille/82797
12. https://pages.stern.nyu.edu/~adamodar/New_Home_Page/home.htm (ERP)
13. https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html (CRP Finland)
14. https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/BetasGlobal.html en Betas.html (bèta's)
15. https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/vebitda.html (EV/EBITDA)
16. https://tradingeconomics.com/finland/government-bond-yield (Finse en Duitse 10-jaarsrente)
17. https://query1.finance.yahoo.com/v8/finance/chart/WISA.HE (koers)

---

## 15. Update-historie
- 2026-11-10, v1.0: eerste analyse, ongeveer een week na notering.

---

## Opmerkingen voor Claude Code
- Dit bestand is op verzoek van Janco niet bedoeld voor de website. Geen JSON-conversie nodig.
- De perimeter verandert in 2023 (Chudovo verkocht) en na 2020 (Jyväskylä dicht). De ROIC-historie van vóór 2023 overschat waarschijnlijk wat de huidige perimeter verdient. Het hoofdprospectus (16-7-2026) bevat omzet en EBITDA 2015-2025 zonder die fabrieken; dat zou de normalisatie verbeteren. Haallijst: hoofdprospectus WISA Group, op upm.com onder de demerger-pagina.
- De segmentcijfers van UPM voor 2025 (vergelijkbare EBIT 35) en de carve-out (31,3) verschillen door de toerekening van kosten. Voor de normalisatie is de carve-out gebruikt.
- De Q3 2026-cijfers van UPM bevatten geen afschrijving (IFRS 5). Wie de TTM uit die cijfers haalt, overschat de EBIT met ongeveer € 10 mln.
- Bijwerken na de jaarcijfers 2026 (februari 2027): eerste zelfstandige kasstroom, nettoschuld per jaareinde, dividendvoorstel.
