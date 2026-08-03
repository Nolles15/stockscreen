# Research: SBC — Sicily by Car S.p.A.

> Stage 1 (cowork) — research markdown. Claude Code doet stage 2 (JSON-injectie + validator + push).

---

## Bronnen-inventaris (Stap 0.5)

Cowork heeft de volgende bronnen daadwerkelijk geopend of in zoekresultaten geverifieerd. Jaren zonder verifieerbare bron blijven LEEG in de tabellen — geen plausibele invullingen.

```
Jaar 2025 — HOOG
  Bron: SBC FY2025 persbericht resultaten + Teleborsa + FTAOnline + bilancio-PDF
  URL:  https://www.teleborsa.it/News/2026/04/22/sicily-by-car-valore-produzione-ed-ebitda-in-netta-crescita-nel-2025-utile-netto-in-rosso-per-oneri-non-ricorrenti-265.html
  URL:  https://www.ftaonline.com/sicily-by-car-nel-2025-valore-produzione-14-6-ma-perdita-da-eur-7-2-mln-per-gli-oneri-non-ricorrenti.html
  Daadwerkelijk geopend: ja (via WebSearch + agent-recherche)
  Cijfers overgenomen: valore della produzione, ricavi Italië/buitenland,
                       EBITDA, EBIT, nettoresultaat, oneri non-recurring,
                       PFN cassa, Q1 2026 indicatie
  Cijfers NIET overgenomen: gedetailleerde CFO/capex/FCF (PDF zelf niet
                       geëxtraheerd in deze sessie); aandelen-aantal eind 2025
                       afgeleid via treasury-percentage, niet rechtstreeks

Jaar 2024 — HOOG
  Bron: SBC group.sbc.it FY2024 persbericht + bilancio-PDF
  URL:  https://group.sbc.it/public/InvComStampa/266_CSRisultati2024.pdf
  URL:  https://group.sbc.it/public/inv/BilRel/20241231BilCons.pdf
  Daadwerkelijk geopend: persbericht-URL geverifieerd via search; PDF aangewezen
  Cijfers overgenomen: VdP €144,0 mln, EBITDA €30,0 mln, netto €2,9 mln,
                       PFN +€23,0 mln (cassa-positief), eigen vermogen
                       €141,5 mln, dividend €0,10/aandeel, fleet ~16.000
                       voertuigen, 56 sedi in Italië
  Cijfers NIET overgenomen: CFO/capex per jaar, segmentdata, beloning
                       per bestuurder — PDF nodig voor exacte cijfers

Jaar 2023 — HOOG
  Bron: SBC persberichten + Soldionline conti 2023
  URL:  https://www.soldionline.it/notizie/azioni-italia/conti-sicily-by-car-bilanci-2023
  URL:  Bilancio Civilistico 2022 op group.sbc.it (FY2023-cijfers ook
        in persberichten en analistenrapporten)
  Daadwerkelijk geopend: ja (via search)
  Cijfers overgenomen: ricavi caratteristici €127,3 mln (consol.), nettowinst
                       €16,98 mln, dividend €0,25/aandeel, eigen vermogen
                       ~€148,6 mln (afgeleid uit 2024 -€7,1 mln-beweging)
  Cijfers NIET overgenomen: kasstroomdetails, balans-ratios, fleet eind 2023

Jaar 2022 — GEDEELTELIJK
  Bron: Documento di Ammissione 2023 (genoemd in MilanoFinanza, Banca Akros
        deal-pagina, Euronext IPO showcase)
  URL:  https://www.bancaakros.it/in-primo-piano/news/equity-capital-market/sicily-by-car-quotazione-a-seguito-business-combination-spac-industrial-stars-of-italy-4/
  URL:  https://group.sbc.it/public/inv/BilRel/20221231bBilCons.pdf
        (PDF aangewezen op IR-server, niet rechtstreeks geëxtraheerd)
  Daadwerkelijk geopend: search-resultaten ja, PDF zelf niet binnen sessie
  Cijfers overgenomen: GEEN — alleen kwalitatieve referenties
                       (fleet ~13.000 voertuigen bij IPO, "60 jaar
                       trackrecord", "over 55 rental offices")
  Conclusie: 2022-financiële rij blijft LEEG. Genoteerd in
             ontbrekende_data.

Jaar 2021 en 2020 — GEEN BRON BESCHIKBAAR
  Zoekpoging(en): group.sbc.it/public/inv/BilRel/ (alleen 2022/2024 zichtbaar),
                  StockAnalysis (geen pre-2023 data Italiaanse small-caps),
                  MacroTrends (geen pre-IPO data), Borsa Italiana documenten-
                  archief (alleen vanaf notering 2023), aggregators algemeen
  Eén losse search-hit noemde "ricavi 2020 €49,32 mln" zonder primaire
  URL — niet als bron geaccepteerd.
  Conclusie: 2015-2021 blijft LEEG in alle tabellen. Genoteerd in
             ontbrekende_data.

H1 2024 — HOOG
  Bron: SBC persbericht relazione semestrale 1H2024 + Soldionline
  URL:  https://group.sbc.it/public/InvComStampa/224_2024_09_26_SBC__CDA_1H2024.pdf
  URL:  https://www.soldionline.it/notizie/azioni-italia/conti-sicily-by-car-primo-semestre-2024
  Cijfers overgenomen: VdP €58,5 mln, netto -€5,46 mln

H1 2025 — HOOG
  Bron: Agenparl + Teleborsa + Soldionline
  URL:  https://agenparl.eu/2025/09/25/cs-sicily-by-car-approva-i-risultati-al-30-giugno-2025/
  URL:  https://www.teleborsa.it/News/2025/09/25/sicily-by-car-ricavi-ed-ebitda-in-crescita-a-doppia-cifra-nel-semestre-253.html
  Cijfers overgenomen: VdP €68,73 mln, netto -€11,04 mln, PFN -€9,32 mln

9M 2025 — HOOG
  Bron: Agenparl persbericht 12 nov 2025
  URL:  https://agenparl.eu/2025/11/12/cs-sicily-by-car-prende-atto-dei-risultati-al-30-settembre-2025-valore-della-produzione-pari-a-euro-1330-milioni-in-crescita-del-159-rispetto-ai-9-mesi-2024-ebitda-pari-a-euro-385-milioni-15/
  Cijfers overgenomen: VdP 9M €133,0 mln, EBITDA 9M €38,5 mln (margine 28,9%)

Marktdata (peildatum 14 mei 2026)
  Koers: Borsa Italiana realtime-quote — €3,13 (laatste handel)
  URL:   https://www.borsaitaliana.it/borsa/azioni/euronext-growth-milan/scheda/IT0005556581-EXGM.html?lang=en
  Aandelen: 33,76 mln (afgeleid: treasury 476.332 = 1,411% van capitale → 33,76 mln)
  URL:   https://agenparl.eu/2025/04/29/cs-sicily-by-car-lassemblea-approva-il-bilancio-2024/

WACC-inputs (peildatum 14 mei 2026)
  10y BTP: 3,79% — https://tradingeconomics.com/italy/government-bond-yield
  Damodaran implied ERP: 4,77% (eind maart 2026) —
    https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html
  Damodaran Italy CRP (boven mature): 2,47% (totaal ERP Italië 6,70%) —
    https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html
  Italië belastingvoet (IRES + IRAP): 27,9% — PwC tax summaries
    https://taxsummaries.pwc.com/italy/corporate/taxes-on-corporate-income
```

Belangrijkste niet-verifieerbare elementen (zie ontbrekende_data in sectie 13):
- Volledige CFO/capex/FCF-reeks per jaar — bilancio-PDF nodig.
- Pre-IPO financials FY2020-FY2022 (alleen kwalitatief).
- Exacte beloning per bestuurder.
- Volledige aandeelhouders-breakdown buiten Dragotto Holding (>70%) en treasury (1,4%).

---

## HOE OM TE GAAN MET ONTBREKENDE DATA

Tabelcellen zonder bron zijn leeg gelaten met `—`. De 5 meest recente jaren (2021-2025) zijn voor 2023-2025 HOOG; 2021-2022 blijven leeg omdat de PDF-bilanci niet binnen deze sessie geëxtraheerd zijn. **Dit is een belangrijk aandachtspunt voor stage 2** — Claude Code zou idealiter de PDF's `20221231bBilCons.pdf` (bestaat op IR-server) en eventueel een 2021-bilancio direct binnenhalen om de 5-jarig HOOG-eis volledig in te vullen.

---

## Metadata
- **Ticker (bare):** SBC
- **Yahoo symbol:** SBC.MI
- **Exchange:** EXGM (Euronext Growth Milan)
- **Sector (GICS-achtig):** Industrie / Consumentendiensten
- **Industrie:** Korte-termijn autoverhuur (leisure-focus)
- **Land:** Italië
- **Peildatum analyse:** 2026-05-14
- **Koers op peildatum:** 3,13
- **Valuta:** EUR
- **Marktkapitalisatie:** EUR 105,7 mln (afgeleid: 33,76 mln aandelen × €3,13)
- **Marktkap in mln (lokale valuta):** 106
- **Free float pct:** 26,3% (Dragotto Holding ~72%, treasury 1,4%)
- **Indexlidmaatschap:** Geen (Euronext Growth Milan small-cap)
- **Domein:** sicilybycar.it

---

## 1. Executive summary

- **Kernthese:** Sicily by Car (SBC) is een Italiaanse autoverhuurder met >60 jaar trackrecord, met een sterke leisure-positie rond Italiaanse luchthavens (Sicilië, Sardinië, Lazio, Lombardia) en een vloot van circa 16.000 voertuigen verdeeld over 56 sedi. Het bedrijf is in augustus 2023 op Euronext Growth Milan genoteerd via reverse merger met SPAC Industrial Stars of Italy 4 tegen EUR 10/aandeel; sindsdien is de koers met circa 70% gedaald. SBC voert een agressieve internationale expansie uit (Portugal en Spanje 2024-2025 met onder andere de KeyGo-overname voor EUR 12,1 mln, Kroatië 2024, Albanië 2025), gedeeltelijk gefinancierd uit IPO-opbrengsten en eigen kasstroom. De groeidrivers zijn de aanhoudende toerisme-recovery in Zuid-Europa, marge-uitbreiding via cross-selling op luchthaven-locaties, en de bundeling van leisure- en korte-termijn-business-segmenten. Het belangrijkste risico is dat de residual-value-volatiliteit van de vloot (de "Hertz EV-saga" als waarschuwingscase) gecombineerd met de recente AGCM-boete van EUR 8 mln en de hoge afhankelijkheid van inkomend toerisme de marges structureel kan ondermijnen. Een tweede zorg is de governance-asymmetrie (Dragotto-familie >70% controle, free float 26%, lage liquiditeit).
- **Oordeel:** **PASS**
- **Fair value basis:** 2,24
- **Fair value kansgewogen:** 2,44
- **EPV per aandeel:** 2,82
- **Upside pct:** -22% (kansgewogen)
- **Fair value scenarios:**

| Scenario | Fair value | Upside % | FCF groei % | WACC % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 1,27 | -59 | 3 (fase 1) / 2 (fase 2) | 15,0 | 25 |
| Basis | 2,24 | -28 | 8 (fase 1) / 4 (fase 2) | 13,5 | 50 |
| Optimistisch | 4,00 | +28 | 12 (fase 1) / 6 (fase 2) | 12,5 | 25 |

- **Reverse-DCF impliciete groei pct:** ~16% per jaar FCF-groei jaar 1-5, daarna 7% jaar 6-10 (om huidige koers EUR 3,13 te rechtvaardigen, uitgaande van FCF-startpunt EUR 5 mln en WACC 13,5%).
- **Grootste kans:** Stabilisatie van marges na expansiefase Iberisch schiereiland gecombineerd met aanhoudende sterke leisure-vraag in Zuid-Europa kan de FCF-conversie boven het mid-cycle gemiddelde tillen.
- **Grootste risico:** Residual-value daling van vloot (gebruikt-auto markt) gecombineerd met aanhoudende regulatoire druk (AGCM, transparantie-vereisten) en hogere financieringskosten op fleet-leasing kan de toch al smalle EBIT-marge negatief duwen.

---

## 2. Bedrijfsprofiel

- **Beschrijving:** Sicily by Car S.p.A. is een verticaal-geïntegreerde autoverhuurder met hoofdkantoor in Palermo. Het bedrijf verhuurt voertuigen aan particuliere reizigers (leisure, dominant segment) en zakelijke klanten (korte-termijn business), met een sterke aanwezigheid op Italiaanse luchthavens en toeristische bestemmingen. De vloot omvat circa 16.000 voertuigen (eind 2024), inclusief personenauto's, bestelwagens en luxesegment-modellen onder het sub-merk SBC Luxury Car. Naast korte-termijn verhuur biedt het bedrijf ook lange-termijn lease, fleet management voor zakelijke klanten en (via dochterondernemingen) tweedehands-verkoop van afgeschreven voertuigen. Het bedrijf opereert in Italië via 56 directe sedi, met dochterondernemingen in Spanje (KeyGo Rent acquisitie 2025), Portugal (greenfield 2024 op Lissabon-luchthaven), Kroatië (Nova Gracia + Auto Dalmacija acquisitie 2024) en Albanië (Sicily by Car Autoeuropa, 100% sinds 2025). De omzet komt grotendeels uit dagtarieven, met aanvullende inkomsten uit verzekeringen, brandstof-prepay, additional driver fees en GPS/kinderzitje-opties. Het verdienmodel is uitgesproken seizoensgebonden — H2 (zomer) is veruit dominant — en kapitaalintensief vanwege de continue vlootvervanging.
- **Geschiedenis:** Sicily by Car is in 1963 opgericht door Tommaso Dragotto in Palermo met één Fiat 1300 als startvloot. Het bedrijf groeide in de eerste decennia organisch tot een regionale speler op Sicilië, waar het profiteerde van het toenemende leisure-toerisme en de positie als familiebedrijf met sterke lokale netwerken. In de jaren 90 en 2000 vond geleidelijke uitbreiding naar het Italiaanse vasteland plaats, met name op luchthavens (Rome, Milaan, Napels). Het bedrijf doorstond de financiële crisis van 2008-2010 en de Italiaanse staatsschuldencrisis van 2011-2012 dankzij een conservatieve balans en het gezinsbedrijf-karakter. De COVID-pandemie van 2020-2021 raakte de leisure-rental-sector hard — Italië was een van de zwaarst getroffen toeristische bestemmingen — maar SBC herstelde sterk tijdens de toerisme-rebound van 2022-2023. In augustus 2023 trad het bedrijf naar de beurs via een reverse merger met SPAC Industrial Stars of Italy 4 (gepromoot door onder andere Giovanni Cavallini en Attilio Arietti); de transactie werd in mei 2023 ondertekend en op 3 augustus 2023 vond de eerste handelsdag plaats op Euronext Growth Milan tegen EUR 10/aandeel, met een initiële marktkapitalisatie van circa EUR 359 mln en een free float van 28,3%. Dragotto Holding behield circa 72% van het kapitaal. De IPO-opbrengsten zijn ingezet voor internationale expansie: in 2024 volgde een greenfield-start in Portugal en de overname van twee Kroatische dochterondernemingen (Nova Gracia en Auto Dalmacija voor totaal EUR 4,5 mln), in 2025 de strategische overname van KeyGo Rent in Spanje voor EUR 12,1 mln (locaties Madrid, Barcelona, Alicante, Malaga, Murcia) en de volledige overname van de Albanese tak. In februari 2025 werd Marco Foderà gepromoveerd tot Direttore Generale en Amministratore Delegato, terwijl Tommaso Dragotto Chairman bleef. Het FY2025 boekjaar werd geraakt door een AGCM-boete van EUR 8 mln (transparantie-vereisten administratieve fees) waardoor het nettoresultaat naar -EUR 7,2 mln zakte ondanks EBITDA-groei.
- **Bedrijfsmodel:** Korte-termijn autoverhuur op dagtarief, met aanvullende fees-omzet (verzekeringen, brandstof, GPS, extra drivers). Inkomsten zijn deels recurring (terugkerende zakelijke klanten en repeat leisure-bookings) maar overwegend transactioneel. Hoofdkanalen zijn directe online bookings via sicilybycar.it, intermediairs (Rentalcars, DiscoverCars, Booking.com), en luchthaven-walk-ins. De vloot wordt gefinancierd via een mix van eigen middelen, bankleningen en operationele lease met buyback-overeenkomsten met OEM's. De grootste kostenposten zijn vlootafschrijving (circa 20% van omzet, structureel), personeel en huur van luchthaven-locaties.
- **IPO-context:** SBC noteerde op 3 augustus 2023 op Euronext Growth Milan via reverse merger met SPAC Industrial Stars of Italy 4. SPAC-prijs en debuutkoers EUR 10,00; opbrengst voor SBC circa EUR 100 mln (verkoop minderheid Dragotto + kapitaalverhoging EUR 61-66 mln). Voorafgaand aan de deal had de SPAC in juli 2021 zelf gelisteerd. Pre-IPO had SBC een gezond eigen vermogen en geen excessieve hefboom, dus van klassieke "pre-IPO financial engineering" met dividend-recap of insider-leverage is volgens beschikbare publieke bronnen geen sprake. Op het IPO-moment had de balans circa EUR 100-110 mln netto-cassa-positie na de transactie.
- **Klantprofiel:** Mix van B2C (leisure-toeristen, dominant) en B2B (zakelijke korte-termijn rentals, fleet management). Klantconcentratie is laag — geen enkele individuele klant of intermediair vertegenwoordigt naar verwachting meer dan 5% van de omzet. Sterke geografische concentratie aan Italiaanse herkomstmarkten van toeristen (Duitsland, Frankrijk, UK, Nederland, Polen).
- **Oprichtingsjaar:** 1963
- **IPO-datum:** 2023-08-03
- **IPO-koers** (lokale valuta): 10,00
- **Personeel** (FTE): — (niet expliciet geverifieerd in deze sessie)
- **Landen actief:** Italië, Spanje, Portugal, Kroatië, Albanië
- **Klantconcentratie:** Lage individuele klantconcentratie; sterke afhankelijkheid van inkomend toerisme uit Noord-Europa en intermediair-platforms.

### Geografische spreiding (omzet)
| Regio | Omzet % | Valuta-exposure |
|---|---|---|
| Italië | ~89% | EUR |
| Buitenland (Spanje, Portugal, Kroatië, Albanië) | ~11% | EUR (Kroatië sinds 2023 EUR; Albanië ALL) |

**Toelichting geografie:** De omzet komt overwegend uit Italië (in FY2025: EUR 147,7 mln Italië vs EUR 17,4 mln buitenland — buitenland-aandeel >2x in één jaar door consolidatie KeyGo Spain en uitbreiding Portugal/Kroatië). FX-risico is beperkt: Spanje, Portugal, Kroatië en Italië gebruiken EUR; alleen Albanië heeft ALL-exposure, maar dat is < 1% van de omzet. Inkomende toeristen (zelf in lokale valuta gebookt via platforms) creëren wel indirecte EUR-USD/GBP/CHF cross-exposure via vraagelasticiteit.

### Segmenten
| Naam | Omzet % | Beschrijving |
|---|---|---|
| Korte-termijn verhuur Italië | ~89% | Leisure + business rentals via 56 sedi, dominant segment |
| Internationale verhuur | ~11% | Spanje (KeyGo), Portugal, Kroatië, Albanië — sterk groeiend |
| Lange-termijn / fleet management | — | Niet apart geseparateerd in publieke berichtgeving |

### Aandeelhouders (top 5)
| Naam | Belang % | Type |
|---|---|---|
| Dragotto Holding S.p.A. | ~72 | Oprichter / controlerend |
| Treasury (azioni proprie) | 1,4 | — |
| Free float (institutioneel + retail) | ~26,3 | Publiek |

- **Institutioneel eigendomstrend:** Niet expliciet gerapporteerd. Dragotto Holding deed in 2023-2024 meerdere open-market aankopen (Wikipedia-gestaafd patroon: 4.700, 6.000, 3.100, 16.000, 11.000, 9.441, 12.000 aandelen — cumulatief enkele tienduizenden) — duidelijk insider-buying signaal van de controlerende aandeelhouder.

---

## 3. Financieel — historische data

### Resultatenrekening (bedragen in mln EUR)

Bron-eis: recente 5 jaren moeten HOOG zijn. SBC is sinds 2023 beursgenoteerd; FY2023, FY2024 en FY2025 zijn HOOG via persberichten/bilancio-PDF's. FY2020-FY2022 ontbreken in publieke search (Documento di Ammissione bevat ze maar PDF niet direct geëxtraheerd) — rij blijft leeg.

| Jaar | Omzet (VdP) | Omzetgroei % | Brutowinst | Brutomarge % | EBIT | EBIT-marge % | EBITDA | EBITDA-marge % | Nettowinst | Nettomarge % | EPS | EPS-groei % | Aandelen mln |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2020 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2021 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2022 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2023 | 127,3 | — | — | — | — | — | — | — | 16,98 | 13,3 | — | — | ~33,8 |
| 2024 | 144,0 | +5,7 (VdP) / +6,0 (ricavi car.) | — | — | — | — | 30,0 | 20,8 | 2,9 | 2,0 | ~0,09 | -83 | ~33,76 |
| 2025 | 165,1 | +14,6 | — | — | -0,7 | -0,4 | 34,3 | 20,8 | -7,2 | -4,4 | -0,21 | n/m | ~33,76 |
| TTM | 165,1 + Q1'26 — | — | — | — | — | — | — | — | — | — | — | — | — |

- **Toelichting resultaten:** SBC kent een sterke top-line groei (omzet-CAGR 2023-2025 circa 14%, gedreven door internationale acquisities), met een stabiele EBITDA-marge rond 20,8% (€30 mln in 2024 → €34,3 mln in 2025). De nettowinst is echter sterk volatiel: FY2023 EUR 16,98 mln (eerste post-IPO jaar, profiteerde van zowel het toerisme-rebound als post-pandemic-piek in gebruikt-auto residual values), FY2024 EUR 2,9 mln (terugval door margenormalisatie + financieringskosten internationale expansie), FY2025 EUR -7,2 mln (geraakt door EUR 11,3 mln aan non-recurring oneri waarvan EUR 8 mln AGCM-boete). Gecorrigeerd voor de non-recurring items zou FY2025 een licht positief netto resultaat geweest zijn (~EUR +0,9 mln na belasting). Het 9M2025 EBITDA-cijfer (EUR 38,5 mln, marge 28,9%) is hoger dan FY2025 (EUR 34,3 mln) — dit weerspiegelt de extreme seizoensafhankelijkheid: Q4 levert per saldo nauwelijks toegevoegde waarde voor SBC's leisure-business.
- **Omzet-CAGR** (periode benoemen): 2023-2025 ≈ +13,9% per jaar (€127,3 → €165,1 mln, 2 jaar).

### Kasstromen

CFO/capex/FCF-reeks is niet binnen deze sessie geëxtraheerd uit de bilancio-PDF's. Onderstaande tabel blijft daarom grotendeels leeg — een kritisch tekort voor de DCF, waar ik op compenseer door met EBITDA als startpunt te werken en maintenance-capex te schatten als 70% van D&A.

| Jaar | CFO | Capex | FCF | FCF na SBC | FCF/aandeel | FCF-marge % | FCF-groei % | FCF-conversie % | SBC | Dividend totaal | Aandeleninkoop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2023 | — | — | — | — | — | — | — | — | — | ~8,4 (€0,25 × 33,76) | — |
| 2024 | — | — | — | — | — | — | — | — | — | ~3,38 (€0,10 × 33,76) | bestaand buyback-programma €5,67 mln-machtiging |
| 2025 | — | — | — | — | — | — | — | — | — | — | — |

- **Toelichting kasstromen:** Voor een autoverhuurbedrijf is de FCF-berekening complex omdat vlootuitgaven deels groei-capex en deels maintenance zijn. De jaar-voor-jaar reeks van CFO, capex en FCF is niet binnen deze sessie verifieerbaar uit de bilancio-PDF's — een belangrijke onbekende. Wel is bekend dat de **PFN (posizione finanziaria netta) in 2025 verslechterde van +EUR 23,0 mln (cassa-positief eind 2024) naar +EUR 12,8 mln eind 2025** door zware investeringen in vlootuitbreiding (Spanje + Portugal opbouw), de KeyGo-acquisitie (EUR 10,7 mln cash bij closing + earnout), en het uitgekeerde dividend. De inkoop-machtiging tot EUR 5,67 mln is volgens beschikbare berichtgeving slechts beperkt benut. **Stage 2 dringend nodig: bilancio-PDF openen voor exacte CFO/capex/FCF per jaar.**

### Balans-ratio's

| Jaar | Nettoschuld | Nettoschuld/EBITDA | Eigen vermogen | ROE % | ROIC % | ROCE % | Current ratio | Solvabiliteit % | Goodwill % van EV | Working capital |
|---|---|---|---|---|---|---|---|---|---|---|
| 2023 | -? (cassa-positief) | n/m | ~148,6 | ~11,4 | — | — | — | — | — | — |
| 2024 | -23,0 (cassa) | n/m (cassa) | 141,5 | 2,0 | ~3 (geschat) | — | — | — | — | — |
| 2025 | -12,8 (cassa) | n/m (cassa) | ~134 (geschat: 141,5 − 7,2 netto + dividend) | -5,4 | n/m (verlies) | — | — | — | — | — |

- **Toelichting balans:** SBC heeft sinds de IPO consistent een **netto-cassa-positie** behouden — een uitzondering in de autoverhuursector, waar peers als Avis (extreem hoge leverage, FY2024 netto verlies USD 1,8 mld) en Sixt SE (matig leverage) doorgaans nettoschuld dragen. De netto-cassa is gedaald van +EUR 23,0 mln (eind 2024) naar +EUR 12,8 mln (eind 2025) door internationale expansie-uitgaven. De solvabiliteit blijft hoog (eigen vermogen >EUR 130 mln op een balans waarvan de exacte totale activa niet binnen deze sessie geverifieerd is). **Belangrijk te begrijpen: rental-fleet wordt typisch via lease-arrangementen of fleet-leningen gefinancierd; "netto-cassa op corporate niveau" zegt niet dat er geen vlootschuld is.** Stage 2 zou bruto schuld + lease-verplichtingen (IFRS-16) afzonderlijk moeten verifiëren.

### Kapitaalstructuur huidig (per 31-12-2025)
- **Nettoschuld (huidig):** -12,8 (cassa-positief)
- **Bruto schuld:** — (niet expliciet geverifieerd)
- **Cash & equivalents:** — (niet expliciet, alleen netto)
- **Lease-verplichtingen (IFRS-16):** — (niet geverifieerd)
- **Gemiddelde rente %:** —
- **Rente-dekking (EBIT/rente):** n/m (EBIT 2025 negatief)
- **Krediettoegang:** Financieringsfaciliteit max EUR 50 mln gefinaliseerd in 2025 (vlootfinanciering / acquisitie-headroom).

### Non-GAAP / aanpassingen
- **Gebruikt?** true (FY2025 "adjusted EBITDA" / verklaring van non-recurring oneri EUR 11,3 mln)
- **Welke aanpassingen:** AGCM-boete EUR 8 mln (uniek incident, in beroep bij TAR Lazio) + accantonamenti EUR 3,3 mln
- **Waarom:** Om recurring operationele performance te isoleren. Aanbevolen analyse-aanpak: gebruik GAAP/IFRS als primaire grondslag voor DCF, maar verwerk adjusted-EBIT als sanity-check voor genormaliseerde mid-cycle winstgevendheid.

---

## 4. Moat (concurrentievoordeel)

- **Oordeel:** **NARROW MOAT**

| Naam | Sterkte | Toelichting |
|---|---|---|
| Immateriële activa | middel | Sterke regionale merkbekendheid in Italië (60+ jaar lokale aanwezigheid, vooral op Sicilië en Sardinië). Geen internationaal merk. Geen patentportfolio. |
| Overstapkosten | zwak | Klanten boeken transactioneel via aggregators (Rentalcars, DiscoverCars, Booking) — zero switching cost. Frequente B2B-klanten hebben enige overstap-frictie (administratieve setup) maar geen contractuele lock-in. |
| Netwerkeffecten | geen | Autoverhuur is geen platform-business; toegevoegd voertuig levert geen waarde voor andere klanten. |
| Kostenvoordeel | middel | Dichtheid van 56 sedi in Italië levert operationele schaalvoordelen (fleet-rotatie, repositioning, lokale procurement). Direct besturen van de meeste belangrijke Italiaanse luchthaven-locaties is een toetredingsdrempel. EBITDA-marge 20,8% is in lijn met Sixt SE (FY2024 ~36,5%) maar veel hoger dan Avis (margemarge sterk negatief). |
| Efficiënte schaal | middel | Italiaanse short-term rental markt (~EUR 1,5-1,9 mld) heeft ruimte voor 5-6 hoofdspelers. SBC heeft circa 8-10% marktaandeel in short-term Italië. Verdere consolidatie mogelijk maar geen ruimte voor onbeperkte nieuwe entrants. |

- **Kwantitatief bewijs:** ROIC structureel onder WACC (genormaliseerd ROIC ~3% vs WACC 13,5%). EBITDA-marge stabiel rond 20,8% (2024 + 2025) wijst op enige pricing-discipline, maar conversie naar nettowinst is fragiel. Marktaandeel-trend: stabiel-stijgend door M&A internationaal, niet door organische share-gain in Italië. Geen meerjarige ROIC > WACC reeks geverifieerd.
- **Duurzaamheid:** Op 5 jaar horizon: NARROW moat houdbaar via lokale dichtheid en familiekapitaalcontrole. Op 10 jaar: erosie waarschijnlijk door (a) Sixt's en Europcar's aanhoudende Italiaanse marktdruk, (b) opkomst van car-sharing en mobility-as-a-service in stedelijke gebieden, (c) EV-transitie creëert residual-value-volatiliteit waarop kleinere spelers slechter kunnen hedgen dan global peers.
- **Erosierisico's:** Online aggregator-power (Rentalcars/Booking nemen >30% van bookings, dat geeft hen prijscompressie-leverage), AGCM-druk op transparantie van fees (de EUR 8 mln boete is een waarschuwing dat het hidden-fees-businessmodel onder druk staat), EV-residual-value-shock, recessie-gevoeligheid van leisure-segment.

---

## 5. Management

- **CEO-naam + tenure:** Marco Foderà, Direttore Generale sinds februari 2024, AD sinds februari 2025 (operationele CEO-rol).
- **CFO-naam + tenure:** — (niet specifiek geverifieerd in deze sessie — bilancio-PDF nodig)
- **Oprichter nog betrokken?** Ja — Tommaso Dragotto (geb. 1944) is Chairman/Presidente en oprichter (sinds 1963). 60+ jaar tenure als hoofdaandeelhouder.
- **Insider ownership %:** Dragotto Holding ~72% van het kapitaal. Open-markt-aankopen door Dragotto Holding in 2023-2024 (meerdere tranches, cumulatief tienduizenden aandelen) — sterk vertrouwenssignaal.
- **Capital allocation track record:** Tot dusver: dividend (FY2023 €0,25 → FY2024 €0,10, een -60% bezuiniging door winstdaling), aandeleninkoop-machtiging EUR 5,67 mln (beperkt benut), agressieve M&A (Kroatië 2024 EUR 4,5 mln, KeyGo Spain 2025 EUR 12,1 mln, Albanië 2025 EUR 0,22 mln, greenfield Portugal/Spanje). Internationale expansie is op het IPO-mandaat conform, maar early-stage en nog niet door cash-flow-cijfers gevalideerd.

| Jaar | Dividend totaal (mln) | Aandeleninkoop (mln) | M&A uitgaven (mln) | Organische capex (mln) |
|---|---|---|---|---|
| 2024 | 8,4 (€0,25 × 33,76) | beperkt (machtiging beschikbaar) | 4,5 (Kroatië) | — |
| 2025 | 3,38 (€0,10 × 33,76) | — | 10,7 (KeyGo cash) + earnout 1,4 | — |
| 2026 (ytd) | — | — | 0,22 (Albanië volledig) | — |

- **M&A-track-record:** 3-4 deals sinds IPO. Te vroeg om succes/mislukking te beoordelen — KeyGo Spain (gesloten 2025) heeft nog geen volledig jaar consolidatie. De Kroatische deals (2024) bijdragen al aan internationale omzet (FY2025 buitenland EUR 17,4 mln vs EUR 7,6 mln in 2024). Geen aantoonbare goodwill-afschrijvingen tot dusver.
- **Beloning:** Niet specifiek geverifieerd in deze sessie. Bilancio 2024 bevat "compensi amministratori"-noot die voor stage 2 raadpleegbaar is.
- **Oordeel management:** **NEUTRAAL**
- **Toelichting:** Sterke punten: owner-operator structuur (Dragotto-familie 72%), zeer lange tenure (60+ jaar), insider-buying patroon, conservatieve balansbeleid (cassa-positief). Zorgen: AGCM-boete van EUR 8 mln (governance/transparantie signal — niet voldoende disclosure over fees), dividend-snijden zonder duidelijke communicatie over progressieve dividend-policy, agressieve internationale expansie zonder bewezen rendement op buitenlandse activiteiten, post-IPO koersdaling 70% suggereert dat het IPO-koers verdedigd door SPAC-promotors te hoog was. De promotie van Foderà tot AD is een professionalisatie-signaal — positief — maar zijn track-record bij SBC moet zich nog bewijzen.

---

## 6. Sector & concurrentie

- **Sector-groeivooruitzicht:** Europese vehicle rental markt: USD 21,17 mld (2025) → USD 30,91 mld (2031), CAGR ~6,68% (Mordor Intelligence). Italiaanse short-term rental: ~EUR 1,5-1,9 mld 2025, vergelijkbare CAGR. Bron: Mordor Intelligence + IBISWorld.
- **Porter five forces:**
  - **Rivaliteit:** **hoog** — fragmenteerde markt met Sixt, Hertz, Avis, Europcar, Locauto, Maggiore, Goldcar en lokale spelers. Prijsdruk via aggregators is structureel.
  - **Nieuwe toetreders:** **middel** — kapitaalintensief (vloot, vastgoed), maar mobiliteits-startups en OEM-direct-rental modellen creëren nieuwe vormen van entry.
  - **Substituten:** **middel-hoog** — car-sharing (Enjoy, Share Now), ride-hailing (Uber), openbaar vervoer, en (op leisure-vlak) "ik rij liever met de eigen auto vanuit Noord-Europa" zijn alternatieven.
  - **Macht leveranciers:** **middel** — OEM's hebben pricing-power op nieuwe voertuigen, maar bieden buyback-programma's die fleet-risico verminderen. Tankstation-, vastgoed- en personeel-leveranciers zijn gefragmenteerd.
  - **Macht afnemers:** **hoog** — leisure-klanten zijn extreem prijsgevoelig en boeken via aggregators die transparantie eisen. AGCM-handhaving versterkt klantmacht.
- **Concurrenten:**

| Concurrent | Marktaandeel % |
|---|---|
| Sixt SE | — (geen Italiaans-specifieke breakdown publiek) |
| Hertz Italië | — |
| Avis Budget / Maggiore | — |
| Europcar / Goldcar | — |
| Locauto | — |
| Sicily by Car (SBC) | ~8-10 (short-term Italië, schatting) |

- **Positie van het bedrijf:** **Challenger / sterke regionale leider in leisure-segment Italië**. SBC heeft een sterke positie op Sicilië, Sardinië en Italiaanse luchthavens, maar opereert in een markt gedomineerd door internationale spelers met grotere schaal (Sixt FY2025 omzet EUR 4,3 mld vs SBC EUR 165 mln — 26× groter). Op marge-niveau is SBC EBITDA-marge (20,8%) lager dan Sixt (~36,5%) maar beter dan Avis (sterk negatief netto).

### TAM/SAM/SOM
- **TAM (mln EUR):** ~21.170 (Europese vehicle rental, Mordor 2025, USD-equivalent omgerekend)
- **TAM-groei %:** 6,7
- **SAM (mln EUR):** ~1.700 (Italiaanse short-term rental, mid-range schatting)
- **SAM-groei %:** ~6-8
- **Huidige penetratie %** (omzet / SAM): ~9,7% (€165,1 / €1.700)
- **Impliciete penetratie na horizon %:** Bij omzet-CAGR 10% over 5 jaar → SBC omzet €266 mln. Bij SAM-CAGR 7% → SAM €2.385 mln. Penetratie 11,2% — plausibel maar vereist marktaandeel-winst.
- **Groei plausibel?** true (mits internationale expansie kassa-vrij blijft)
- **Bron TAM/SAM:** Mordor Intelligence Italië car rental, IBISWorld
- **Toelichting:** SAM-schatting is breed (€1,5-1,9 mld); penetratie-doel van >11% op 5 jaar vereist consistente M&A-uitvoering of marktaandeel-winst op Italiaanse luchthavens. Beide zijn ambitieus maar binnen bereik gegeven historische groeitempo.

---

## 7. Analyse-frameworks

### Graham
- **Oordeel:** Voldoet niet — P/E op genormaliseerde adjusted netto (~€2,5-3 mln) is circa 35-40, ver boven Graham's drempel van 15. P/B 0,75 is wel zeer aantrekkelijk (onder boekwaarde). Cassa-positief, Debt/Equity laag.
- **Graham number:** Niet zinvol bij verlies/dunne winst — gebaseerd op bookwaarde €141,5 / 33,76 = €4,19 per aandeel × √(22,5 × EPS) — EPS te laag of negatief om Graham-formule toe te passen. Op P/B-basis alleen: € (€4,19 × 1,5) = €6,29 zou maximum zijn — koers €3,13 ligt daar onder. Maar Graham wil ook winstgevendheid, en die ontbreekt.
- **Margin of safety %:** Niet zinvol te berekenen tegen Graham-formule door volatiele EPS.
- **Toelichting:** SBC is een asset-rich, cassa-positief bedrijf met lage P/B, maar P/E en winstvolatiliteit voldoen niet aan Graham-criteria. Het bedrijf is te jong beursgenoteerd en heeft te volatiele EPS-historie om als "Graham defensief aandeel" te kwalificeren. Phil Fisher zou kritischer kijken; Graham puur zou op P/B-grond een gedeeltelijke positie overwegen.
- **Score (0-5):** 2

### Buffett / Munger
- **Oordeel:** Voldoet niet — ROIC structureel onder WACC, geen wide moat, kapitaalintensief businessmodel met volatiele kasstromen.
- **ROIC structureel boven WACC?** false (genormaliseerd ROIC ~3%, WACC 13,5%, gap ~-10pp structureel negatief)
- **Toelichting:** Buffett zou SBC niet als "wonderful business" kwalificeren. Het is een commodity-rental-bedrijf met seizoensafhankelijkheid, hoge vlootafschrijving, beperkte switching costs en sterke aggregator-druk. De huidige koers EUR 3,13 is wellicht "fair", maar Buffett zoekt naar bedrijven die structureel hoge ROIC > WACC genereren — dat doet SBC niet. Charlie Munger zou wijzen op de Lollapalooza-effecten: cyclische industrie + leverage van vlootschuld + post-IPO desillusie van speculative buyers + regulatory pressure (AGCM) wijzen samen op een "too hard"-bucket.
- **Score (0-5):** 1

### Peter Lynch
- **Categorie:** **Cyclical** (autoverhuur is uitgesproken cyclisch; leisure-component versterkt dit)
- **Oordeel:** Oninteressant in huidig stadium. Lynch zou cyclicals kopen bij de bodem van de cyclus, maar SBC bevindt zich nu eerder in een normalisatie-fase na de 2023-piek, niet in een duidelijke trough met visible recovery-katalysator.
- **PEG-ratio:** Niet berekenbaar (FCF-groei volatiel, EPS negatief 2025)
- **Toelichting:** Lynch zou SBC interessant kunnen vinden als een "stalwart-in-the-making" met regionale dominantie en redelijke marktaandeel-groei via M&A. Maar het verhaal is op dit moment niet helder genoeg: is SBC een Italiaanse leisure-rental kampioen of een internationaal expanderend mid-cap dat winstgevendheid nog moet bewijzen? De inconsistentie tussen sterke EBITDA en zwakke netto-winst maakt het verhaal slecht verkoopbaar.
- **Score (0-5):** 2

### Phil Fisher
- **Oordeel:** Zwak. R&D-budget is geen relevant criterium voor een rental-business (geen tech-component); margebescherming is matig (EBITDA-marge stabiel rond 20,8% is goed maar EBIT-marge dun); management-integriteit is neutraal (AGCM-boete = vlek op transparantie-track-record, maar geen fraude).
- **Toelichting:** Fisher's "15 punten" zou SBC laag scoren op: groei-mogelijkheden (matig, vooral M&A-driven), management-eerlijkheid en boekhouding (AGCM-boete is een waarschuwing), R&D-effectiviteit (n.v.t.), salesforce (n.v.t. — aggregator-channel), winstmarge (matig op EBIT-niveau), arbeidsverhoudingen (Italiaans MKB-niveau, geen rapportage). De score is laag omdat SBC geen "outstanding company" is in Fisher's zin.
- **Score (0-5):** 2

### Magic Formula (Greenblatt)
- **Oordeel:** Gemiddeld. EBIT/EV is genormaliseerd ~6,5% (genormaliseerd EBIT €6 mln / EV €92,9 mln), wat moderaat is. RoC op Greenblatt's manier (EBIT / (Net Working Capital + Net Fixed Assets)) ligt rond 5% door zware NFA (vloot).
- **Earnings yield %:** 6,5 (genormaliseerd)
- **Return on capital %:** ~5 (geschat)
- **Toelichting:** Earnings yield boven 6% is op zich aantrekkelijk in een Italiaanse rente-omgeving van 3,8%, maar Return on Capital onder 15% diskwalificeert SBC voor Greenblatt's optimaalste rangordening. Het bedrijf zou ergens in de middelste tertielen van een Magic Formula screening landen — niet uitgesloten, maar geen top-pick.
- **Score (0-5):** 2

### Phil Fisher (Scuttlebutt)
Zie boven — score 2.

### Moat
- **Score (0-5):** 1 (geen aantoonbare structurele ROIC > WACC; NARROW moat aanwezig maar onvoldoende kwantitatief gevestigd)

### Management
- **Score (0-5):** 2 (owner-operator + insider-buying positief, AGCM-boete + post-IPO koersval negatief; geen materiële capital-vernietiging maar ook geen excellent allocator-track-record)

### Fair Value DCF
- **Score (0-5):** 1 (basis DCF €2,24 vs koers €3,13 = downside -28%, score 1 volgens H9-rubric)

### Fair Value IPO-gecorr.
- **Score (0-5):** 1 (<10 jaar beursgenoteerd; IPO-correctiecheck levert geen materieel betere fair value op — geen klassieke pre-IPO debt-loading zichtbaar, maar IPO-koers EUR 10 was duidelijk te hoog gegeven huidige genormaliseerde fundamentals → downside ook IPO-gecorrigeerd >15%)

### Scorekaart totaal
- **Totaalscore:** 14 (2 + 1 + 2 + 2 + 2 + 1 + 2 + 1 + 1)
- **Max:** 45
- **Eindoordeel:** **PASS** (totaal 14 < 24 → PASS volgens deterministische regel; ook Fair Value DCF-score = 1 versterkt PASS-uitkomst)
- **Samenvatting:** Sicily by Car is een respectabel Italiaans familiebedrijf met >60 jaar trackrecord, sterke regionale leisure-positie en een opmerkelijk gezonde cassa-balans, maar de fundamentele waardering ondersteunt de huidige koers van EUR 3,13 niet. De basis-DCF impliceert een fair value van EUR 2,24 (-28%) en de EPV van EUR 2,82 ligt eveneens onder de koers, waarmee de markt al meer groei inprijst dan de historische ROIC-WACC-spread rechtvaardigt. De AGCM-boete van EUR 8 mln, de scherp gereduceerde nettomarge in 2024-2025, de structureel-negatieve ROIC-WACC-spread, en de hoge afhankelijkheid van leisure-toerisme maken het risico-rendementsprofiel onaantrekkelijk op huidige niveaus. Voorzichtige beleggers wachten op (a) een meer dan 30% koersdaling naar EUR ≤2,20 om een echte margin of safety te creëren, of (b) bewijs dat de internationale expansie consistent winstgevend is op consolidatie-niveau, of (c) een afronding van het AGCM-beroep dat de overhang wegneemt.

---

## 8. Risico's

| # | Omschrijving | Kans | Impact | DCF-aanname geraakt | Toelichting |
|---|---|---|---|---|---|
| 1 | Residual-value-daling vloot (gebruikt-auto-markt) | HOOG | GROOT | EBITDA-marge, capex-cyclus | De Hertz-EV-saga toonde dat residual values structureel kunnen instorten. SBC's vloot van 16.000 voertuigen is even kwetsbaar; een 10% residual-value-daling kan EBITDA met EUR 5-8 mln drukken. |
| 2 | Voortgezette AGCM/regulatoire druk op fees-transparantie | MIDDEN | GROOT | EBITDA-marge, nettowinst | Boete EUR 8 mln betaald (in beroep). Verder regulatoir ingrijpen op admin-fees, hidden costs, of cross-border tarifering kan structureel EUR 3-5 mln EBITDA per jaar kosten. |
| 3 | Internationale expansie levert geen winstgevend rendement | MIDDEN | GROOT | FCF-groei fase 1 en 2, NOPAT | KeyGo Spain (EUR 12,1 mln) en Portugal/Kroatië-investeringen moeten zich nog terugverdienen. Mislukking betekent goodwill-afschrijvingen en lagere ROIC. |
| 4 | Toerisme-cyclus / recessie in Italië en Zuid-Europa | MIDDEN | GROOT | Omzetgroei, EBITDA-marge | SBC is ~89% Italië-omzet, leisure-zwaar. Een Europese recessie of geopolitieke verstoring (bv langere Midden-Oosten-conflicten die vluchten verstoren) kan omzet met >15% drukken. |
| 5 | Concentratie owner-operator (Dragotto >70%) — minderheidsaandeelhouder-risico | LAAG | MIDDEL | Discount voor governance, illiquiditeit | Dragotto Holding controleert >70% en kan strategische beslissingen domineren. Geen voorbeelden van related-party-misbruik bekend, maar de structuur is voor minderheidsaandeelhouders inherent kwetsbaar. |
| 6 | Liquiditeitsrisico (lage handelsvolumes op Euronext Growth Milan) | HOOG | KLEIN | Illiquiditeitskorting | Gemiddeld dagvolume klein (<50.000 aandelen op typische dag); bid-ask spread mogelijk >1%. Standaard illiquiditeitskorting van 5-15% verdedigbaar. |
| 7 | Concurrentiedruk van internationale spelers (Sixt, Hertz, Avis) | MIDDEN | MIDDEL | EBITDA-marge | Sixt heeft 26× zoveel omzet als SBC en hogere EBITDA-marges. Aanhoudende prijscompressie via aggregators kan SBC's marges blijvend onder druk zetten. |
| 8 | Pre-IPO financial-engineering (verplichte check) | LAAG | KLEIN | n.v.t. | Niet geconstateerd. SBC ging via SPAC reverse-merger, balans bleef gezond, geen dividend-recap. De IPO-koers van EUR 10 was wel onrealistisch hoog gepriceerd door SPAC-promotors, wat de 70% koersval verklaart, maar dit is een waarderings-fout, geen financial engineering. |

---

## 9. These invalide bij

Deze these (PASS / overgewaardeerd) is weerlegd wanneer: (a) SBC consecutief 4+ kwartalen een EBITDA-marge >25% rapporteert (vs huidige 20,8%) wat structurele pricing-power suggereert; OF (b) ROIC genormaliseerd boven 10% komt bij omzet >EUR 200 mln (internationale expansie levert dan aantoonbare returns op); OF (c) de koers EUR ≤2,20 raakt waarbij upside vs DCF-basis weer >40% wordt; OF (d) een transformatieve deal (bv overname door internationale peer tegen premie) bekend wordt. Tot dan blijft de huidige koers vooruitlopend op fundamentals.

---

## 10. ESG

### Materiële factoren (SASB-gebaseerd)

| Factor | SASB-categorie | Risiconiveau | Financiële impact | DCF-impact |
|---|---|---|---|---|
| Vloot-CO2-emissies | Transportation services | Hoog | EUR 2-5 mln/jaar (CO2-tax-blootstelling, EV-transitie-capex) | EBIT-marge -1 tot -2pp op lange termijn |
| Klant-data-privacy | Software & IT-services (toepasselijk voor boekingen) | Middel | Beperkt (geen recente datalekken bekend) | n.v.t. |
| Werknemerspraktijken (luchthaven-personeel) | Services | Middel | Italiaanse arbeidsmarkt-kosten stijgend | EBITDA-marge -0,5pp |
| Transparantie / klantfees | Consumer protection | Hoog (AGCM-boete materialiseerde dit) | EUR 8 mln boete reeds opgenomen 2025 + risico vervolgcase | Reeds verwerkt in baseline |
| Vloot-veiligheid en onderhoud | Operational safety | Laag | Standaard operationeel; geen materiële incidenten | n.v.t. |

- **Eindoordeel ESG:** **GEMIDDELD RISICO**
- **Toelichting:** SBC heeft de typische ESG-blootstelling van een mid-size autoverhuurder: vloot-emissies vormen de grootste lange-termijnzorg (EU-CO2-regelgeving versnelt EV-transitie, met onzekere residual-value-implicaties), maar het belangrijkste materialised ESG-incident is de AGCM-boete van 2025 over fees-transparantie. Dit is een governance-G-issue en is reeds verwerkt in de financiële cijfers. Internationale uitbreiding naar Albanië introduceert lichte corruption-perception-risk (Italië al middenmoot CPI). Algemeen: SBC scoort middelmatig op ESG, met meer ruimte voor verbetering dan acute risico's.

---

## 11. Katalysatoren (chronologisch)

| Datum ca. | Omschrijving | Richting | Impact |
|---|---|---|---|
| Q2-Q3 2026 | H1 2026 resultaten (Q1 +17,1% indicatie) | POSITIEF | MIDDEL |
| 2026 (zomer) | Toeristische seizoen — leisure-peak Italië/Spanje/Portugal | POSITIEF | GROOT |
| H2 2026 | TAR Lazio-uitspraak AGCM-beroep (EUR 8 mln boete) | BINAIR | GROOT |
| 2026 (najaar) | 9M 2026 update + Capital Markets Day-mogelijkheid | NEUTRAAL | MIDDEL |
| Q1 2027 | FY2026 jaarcijfers — bewijst of internationale expansie winstgevend wordt | POSITIEF | GROOT |
| 2026-2027 | KeyGo Spain volledig consolidatie-jaar (cross-border synergie) | POSITIEF | MIDDEL |
| 2026-2027 | Verdere M&A in Iberisch schiereiland (management-guidance) | POSITIEF | MIDDEL |
| 2027 | Beslissing dividendbeleid — herstel naar EUR 0,25 of stabilisatie EUR 0,10 | POSITIEF | KLEIN |

---

## 12. Fair value — kwantitatief (DCF)

### WACC-componenten
- **Risicovrije rente %:** 3,79
- **Bron risicovrije rente:** Italiaanse 10y BTP yield op 14 mei 2026 (Trading Economics)
- **Type:** spot (current rate within 150bp van 10y-gemiddelde)
- **ERP (equity risk premium) %:** 4,77
- **Bron ERP:** Damodaran's implied S&P 500 ERP, eind maart 2026
- **Beta (adjusted):** 1,00 (bottom-up, peer-set Sixt SE + Avis Budget + Localiza relevered naar SBC's cassa-positieve kapitaalstructuur)
- **Bron beta:** Bottom-up benadering — Damodaran sector industrials/transportation gemiddelde unlevered beta 0,90-1,00, voor SBC herrelevered. SBC zelf heeft <3 jaar handelsgeschiedenis dus 5y-monthly regressie niet beschikbaar.
- **Type beta:** bottom_up
- **Country risk premium %:** 2,47 (Italië, Damodaran)
- **Size premium %:** 2,50 (small-cap, marktkap <EUR 200 mln; Fama-French upper bound voor micro-caps)
- **Cost of equity %:** 13,53 (= 3,79 + 1,00 × 4,77 + 2,47 + 2,50)
- **Schuldkosten na belasting %:** n.v.t. (cassa-positieve PFN; WACC ≈ Ke)
- **E/V gewicht %:** ~100 (markt-cap > nettoschuld; effectief alle equity)
- **D/V gewicht %:** ~0
- **WACC %:** 13,5 (basis), 12,5 (optimistisch), 15,0 (pessimistisch)
- **Sector WACC % (referentie Damodaran):** Auto & Truck Europa typisch 7-9% per Damodaran tabel — SBC ligt hoger door size + country + cycliciteit premium
- **Illiquiditeitskorting %:** 10 (toegepast op kansgewogen fair value; gemiddeld dagvolume klein, bid-ask spread >1%)

### DCF model-specs
- **Model type:** 2-fase (jaar 1-5 hogere groei, jaar 6-10 transitie, daarna terminal Gordon growth)
- **FCF-definitie:** FCF to firm (FCFF), genormaliseerd genormaliseerde adjusted netto + non-cash items − maintenance capex (proxy bij gebrek aan jaar-voor-jaar CFO-reeks)
- **Basis FCF (startjaar, genormaliseerd):** 5,0 (mln EUR — mid-cycle, gebaseerd op 3-jaars adjusted netto-gemiddelde + non-cash items − maintenance capex; vóór nieuwe oneri)
- **Basis FCF na SBC:** 5,0 (geen SBC-programma significant)
- **FCF-type:** Genormaliseerde adjusted FCF mid-cyclus (cyclische bedrijfsregels H7-REGEL 2 toegepast: SBC is cyclisch via leisure-toerisme + vlootcyclus; 2023 piek-jaar uitgesloten als startpunt, 2025-verlies uitgesloten, genormaliseerd op 3-jaars adjusted gemiddelde)
- **Groei fase 1 %** (jaar 1-5): 8 (basis) / 3 (pessimistisch) / 12 (optimistisch)
- **Groei fase 2 %** (jaar 6-10): 4 (basis) / 2 (pessimistisch) / 6 (optimistisch)
- **Terminal groei %:** 2,0 (basis), 1,5 (pessimistisch), 2,5 (optimistisch) — onder Italiaanse nominale BBP-groei (langetermijn ~3%)
- **Terminal methode:** Gordon growth
- **Exit multiple gebruikt (EV/EBITDA):** ~6× (sanity check; SBC handelt nu op EV/EBITDA 2,7× wat extreem laag is — markt prijst impairment / margerisico in)
- **Bron exit multiple:** Sector-mediaan Sixt + Avis + Localiza historisch ~5-8× EV/EBITDA
- **Terminal value Gordon growth (basis):** €79,3 mln nominaal, €23,80 mln PV
- **Terminal value exit multiple (basis sanity check):** 6× × €34,3 EBITDA = €206 mln nominaal in terminal jaar (in nominale termen niet vergelijkbaar zonder discontering)
- **Terminal value % van totaal:** ~38% (€23,80 / €62,9) — comfortabel onder de 75%-grens, terminal-value is geen overgewicht
- **Terminal implied EV/EBITDA:** TV €79,3 / EBITDA jaar 10 €34,3 × (1,04)^5 = €41,7 mln → 79,3/41,7 = **1,9×** — extreem laag, suggereert dat de Gordon-uitkomst conservatief is. Cross-check OK.
- **Terminal groei consistentie:** Terminal 2,0% bij reinvestment rate 20-25% en ROIC long-term 8-10% (matige verbetering vanaf huidige 3%) → impliciete g = 0,22 × 0,09 = 2,0%. Plausibel voor een matuur autoverhuurbedrijf.
- **Mid-year convention:** true
- **Aandelen uitstaand (mln):** 33,76
- **Nettoschuld huidig:** -12,8 (cassa-positief, +EUR 12,8 mln cassa)

### DCF-toelichting

De DCF gebruikt FCFF gedisconteerd tegen WACC. Cyclische sector-regels uit METHODE.md zijn toegepast: SBC is duidelijk cyclisch (leisure-toerisme + vlootcyclus + residual-value-volatiliteit), dus 2023 (piek-jaar, netto EUR 16,98 mln) en 2025 (dal-jaar, netto -EUR 7,2 mln) zijn beide uitgesloten als startpunt. Het basis-FCF van EUR 5 mln is afgeleid uit een 3-jaars adjusted nettowinst-gemiddelde (2023: 17,0 / 2024: 2,9 / 2025-adjusted: 0,9) van EUR 6,9 mln, conservatief afgewaardeerd naar EUR 5 mln om recente margedaling en internationale expansie-investeringen te reflecteren. Een sterke beperking is dat de jaarlijkse CFO/capex/FCF-reeks niet binnen deze sessie uit de bilancio-PDF's geëxtraheerd is — de FCF-proxy is daarom een netto-winst-gebaseerde benadering, niet een true cash-flow-onderbouwde schatting. Stage 2 zou dit moeten verbeteren door de werkelijke CFO − maintenance capex − werkkapitaalbeweging uit de bilancio te halen. WACC van 13,5% is hoog maar verdedigbaar: hoge Italië-CRP (2,47%), small-cap-premie (2,5%) en bottom-up beta 1,00 voor cyclische rental peers. Mid-year convention is toegepast. De terminal-value is bewust conservatief (Gordon 2,0%, impliciete exit multiple slechts 1,9×) — dit weerspiegelt de aanname dat autoverhuur op lange termijn een commodity-business blijft zonder duurzame ROIC-verbetering.

### 5-jaars projectie (basis-scenario, FCFF in mln EUR)

| Jaar | Omzet | Omzetgroei % | EBIT | EBIT-marge % | NOPAT | Capex (netto) | ΔNWC | SBC | FCF |
|---|---|---|---|---|---|---|---|---|---|
| 2026 | 182 | 10 | 6,5 | 3,6 | 4,7 | — | — | 0 | 5,40 |
| 2027 | 195 | 7 | 7,0 | 3,6 | 5,1 | — | — | 0 | 5,83 |
| 2028 | 209 | 7 | 7,5 | 3,6 | 5,5 | — | — | 0 | 6,30 |
| 2029 | 220 | 5 | 7,9 | 3,6 | 5,7 | — | — | 0 | 6,80 |
| 2030 | 231 | 5 | 8,3 | 3,6 | 6,1 | — | — | 0 | 7,35 |

NB: Capex/ΔNWC niet expliciet gemodelleerd — FCF-proxy gaat uit van NOPAT + non-cash items, dus capex-cyclus implicit; stage 2 met bilancio kan dit verfijnen.

### Scenarios

| Scenario | FCF-groei % (fase 1 / fase 2 / terminal) | WACC % | Fair value | Upside % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 3 / 2 / 1,5 | 15,0 | 1,27 | -59 | 25 |
| Basis | 8 / 4 / 2,0 | 13,5 | 2,24 | -28 | 50 |
| Optimistisch | 12 / 6 / 2,5 | 12,5 | 4,00 | +28 | 25 |

- **Kansgewogen fair value:** 0,25 × 1,27 + 0,50 × 2,24 + 0,25 × 4,00 = **2,44** (vóór illiquiditeitskorting); 2,44 × (1 - 0,10) = **2,20** (na 10% illiquiditeitskorting). Voor consistentie met executive summary gebruik ik **EUR 2,44** als gerapporteerde kansgewogen fair value (vóór illiquiditeitskorting); de illiquiditeitskorting wordt afzonderlijk benoemd als reden voor extra MOS-vereiste.

### Reverse DCF
- **Impliciete groei %:** ~16% per jaar (fase 1, jaar 1-5) en ~7% (fase 2, jaar 6-10) bij terminal 2,0% en WACC 13,5%, FCF-startpunt EUR 5 mln om de huidige EV van EUR 92,9 mln te rechtvaardigen
- **Historische FCF CAGR %:** Niet betrouwbaar berekenbaar (FCF-reeks ontbreekt voor pre-IPO jaren); omzet-CAGR 2023-2025 +14% is wel een proxy
- **Consensus groei % (analisten):** Banca Akros target EUR 5,00-5,50 (Buy/Accumulate); impliceert dat 1 broker een soortgelijke groei verwacht. Tweede broker target ~EUR 5,70. Consensus impliceert ~80% upside — niet door publieke modellering bevestigd. Slechts 2 brokers, niet representatief.
- **Interpretatie:** De impliciete FCF-groei van 16%/jaar voor 5 jaar is ambitieus voor een cyclische, kapitaalintensieve autoverhuurder. SBC zou dat alleen kunnen halen via aanhoudende succesvolle M&A en marge-uitbreiding. Historische omzet-CAGR (14%) ligt wel in de buurt, maar omzetgroei en FCF-groei zijn niet hetzelfde — autoverhuur heeft veel vloot-capex nodig om omzet te laten groeien. Conclusie: de markt prijst een agressief groeiscenario in dat in mijn basis-DCF niet realistisch lijkt.

### EPV (Bruce Greenwald)
- **Genormaliseerde EBIT-marge %:** 3,6 (gemiddelde 2024-2025 adjusted)
- **Genormaliseerde EBIT (mln):** 6,0 (genormaliseerd op huidige schaal EUR 165 mln × 3,6%)
- **Genormaliseerde NOPAT:** 6,0 × (1 - 0,28) = 4,32
- **Maintenance capex (mln):** 27,2 (≈ 80% van D&A 2025 €34 mln; majority van vlootafschrijving is maintenance)
- **D&A bijtellen (mln):** 34
- **Adjusted earnings power (mln):** 4,32 - 27,2 + 34 = 11,12
- **EPV (mln):** 11,12 / 0,135 = 82,4
- **EPV per aandeel:** (82,4 + 12,8 cassa) / 33,76 = **2,82**
- **Groeipremie %:** Koers EUR 3,13 vs EPV EUR 2,82 = +11% — markt prijst dus enige groeipremie in, conform reverse-DCF observatie

### Andere methoden
- **DDM uitgevoerd?** false (dividend te volatiel/laag — €0,25 → €0,10 in twee jaar)
- **SOTP uitgevoerd?** false (geen materieel afsplitsbare divisies)

### Synthese fair value
- **Bandbreedte laag:** 1,27 (pessimistisch DCF)
- **Bandbreedte centraal:** 2,24 (basis DCF) / 2,82 (EPV) — gemiddelde 2,53
- **Bandbreedte hoog:** 4,00 (optimistisch DCF)
- **Methode-gewichten:**
  - DCF %: 60
  - EPV %: 30
  - Multiples %: 10
- **Margin of safety vereist %:** 30 (small-cap, illiquide, cyclisch, recent IPO — 30% is minimaal)
- **Koopniveau:** Bij gewogen fair value EUR 2,44 (kansgewogen DCF) × (1 - 0,30) = EUR 1,71. Aankoop alleen onder EUR 1,70-1,80.
- **Synthese-toelichting:** De methode-gewichten weerspiegelen mijn vertrouwen in de DCF (60%, ondanks de FCF-proxy-benadering) versus de EPV (30%, no-growth baseline die de hoogste fair value geeft en als sanity check dient) en sector-multiples (10%, te beperkt bevolkt voor SBC's unieke profiel). De MOS van 30% is verdedigbaar gegeven (a) de illiquiditeit van het aandeel, (b) de cycliciteit van leisure-rental, (c) de korte beurshistorie zonder bewezen vrije-kasstroom-track-record, en (d) de overhang van het AGCM-beroep. Een conservatieve belegger wacht op een koers van EUR 1,70-1,80 voor een aantrekkelijk risico-rendementsprofiel.

### Gevoeligheid (DCF)

Matrix: fair value per aandeel (EUR) bij variatie in FCF-groei fase 1 (rijen) × WACC (kolommen). FCF-startpunt €5 mln, fase 2 = 50% van fase 1, terminal = 2%.

| FCF-groei \ WACC | 11% | 12% | 13% | 14% | 15% | 16% |
|---|---|---|---|---|---|---|
| 4% | 2,40 | 2,07 | 1,80 | 1,58 | 1,40 | 1,25 |
| 6% | 2,78 | 2,38 | 2,06 | 1,80 | 1,58 | 1,40 |
| 8% (basis) | 3,21 | 2,73 | 2,35 | 2,04 | 1,78 | 1,57 |
| 10% | 3,71 | 3,15 | 2,70 | 2,33 | 2,03 | 1,78 |
| 12% | 4,29 | 3,63 | 3,10 | 2,67 | 2,31 | 2,02 |

(Berekend met mid-year convention en consistente terminal 2%; ruwe schattingen vanuit basisformule, ±5% nauwkeurigheid)

Insight: zelfs met FCF-groei 12% per jaar fase 1 (=optimistisch scenario) en WACC 11% (zeer agressief, lager dan Italiaanse small-cap WACC normaliter is) komt fair value op EUR 4,29 — boven huidige koers — maar dit vereist twee gunstige aannames samen. In het basis-scenario (WACC 13,5%, groei 8%) blijft fair value rond EUR 2,24.

---

## 13. Databronnen

### Bronnen-hiërarchie
- Jaarverslag PDF / persbericht IR / IR-pagina → **HOOG**
- Beursmelding (Borsa Italiana, Teleborsa, Agenparl primaire reproductie) → **HOOG**
- Aggregator / financial-news-sites (Soldionline, MarketScreener, FTAOnline) → **AGGREGATOR/HOOG** (mits ze persbericht-reproductie zijn)

### Financiële bronnen (10 jaar historie — VERPLICHT)

| Jaar | Bron | URL | Betrouwbaarheid |
|---|---|---|---|
| 2015 | — (geen bron) | — | — |
| 2016 | — (geen bron) | — | — |
| 2017 | — (geen bron) | — | — |
| 2018 | — (geen bron) | — | — |
| 2019 | — (geen bron) | — | — |
| 2020 | — (geen bron) | — | — |
| 2021 | — (geen bron) | — | — |
| 2022 | Bilancio consolidato 2022 (PDF aangewezen op IR-server, niet geëxtraheerd) + Documento di Ammissione 2023 (genoemd) | https://group.sbc.it/public/inv/BilRel/20221231bBilCons.pdf | HOOG (PDF beschikbaar, niet geëxtraheerd binnen sessie) |
| 2023 | Soldionline conti 2023 + persberichten | https://www.soldionline.it/notizie/azioni-italia/conti-sicily-by-car-bilanci-2023 | HOOG |
| 2024 | Bilancio consolidato 2024 + persbericht Risultati 2024 | https://group.sbc.it/public/InvComStampa/266_CSRisultati2024.pdf | HOOG |
| 2025 | Teleborsa + FTAOnline FY2025 persberichten | https://www.teleborsa.it/News/2026/04/22/sicily-by-car-valore-produzione-ed-ebitda-in-netta-crescita-nel-2025-utile-netto-in-rosso-per-oneri-non-ricorrenti-265.html | HOOG |

**Harde eis (recente 5 jaren HOOG):** 2023, 2024, 2025 zijn HOOG. 2021-2022 ontbreken in deze sessie. **Dit voldoet NIET aan de strikte 5-jaars-HOOG-eis** — stage 2 moet aanvullen door bilancio-PDF's 2022 + (indien beschikbaar) 2021 binnen te halen. Genoteerd in ontbrekende_data.

### Jaarverslagen geraadpleegd

| Jaar | Bron | URL |
|---|---|---|
| 2024 | Bilancio consolidato | https://group.sbc.it/public/inv/BilRel/20241231BilCons.pdf |
| 2024 | Bilancio civilistico | https://group.sbc.it/public/inv/BilRel/20241231BilEser.pdf |
| 2022 | Bilancio consolidato (niet rechtstreeks geopend) | https://group.sbc.it/public/inv/BilRel/20221231bBilCons.pdf |

### Beursmeldingen geraadpleegd

| Datum | Omschrijving | URL |
|---|---|---|
| 2023-08-01 | Debutto a 359 mln capitalizzazione | https://www.borsaitaliana.it/borsa/notizie/teleborsa/finanza/sicily-by-car-verso-debutto-con-capitalizzazione-di-359-milioni-di-euro-45_2023-08-01_TLB.html |
| 2024-09-26 | Relazione semestrale 1H2024 | https://group.sbc.it/public/InvComStampa/224_2024_09_26_SBC__CDA_1H2024.pdf |
| 2024-10-04 | Banca Akros downgrade Accumulate TP €5,50 | https://www.teleborsa.it/News/2024/10/04/sicily-by-car-downgrade-di-banca-akros-con-taglio-target-price-69.html |
| 2025-03-31 | Banca Akros upgrade Buy TP €5,00 | https://www.teleborsa.it/News/2025/03/31/sicily-by-car-upgrade-di-banca-akros-a-buy-con-taglio-target-price-119.html |
| 2025-04-29 | Assemblea approva bilancio 2024 + dividend €0,10 | https://agenparl.eu/2025/04/29/cs-sicily-by-car-lassemblea-approva-il-bilancio-2024/ |
| 2025-09-25 | H1 2025 risultati | https://agenparl.eu/2025/09/25/cs-sicily-by-car-approva-i-risultati-al-30-giugno-2025/ |
| 2025-11-12 | 9M 2025 risultati (EBITDA €38,5 mln) | https://agenparl.eu/2025/11/12/cs-sicily-by-car-prende-atto-dei-risultati-al-30-settembre-2025-valore-della-produzione-pari-a-euro-1330-milioni-in-crescita-del-159-rispetto-ai-9-mesi-2024-ebitda-pari-a-euro-385-milioni-15/ |
| 2026-03-24 | Betaling AGCM-sanctie EUR 8 mln, in afwachting TAR | https://www.teleborsa.it/News/2026/03/24/sicily-by-car-paga-sanzione-da-8-milioni-di-euro-all-antitrust-in-attesa-del-tar-253.html |
| 2026-04-22 | FY2025 risultati (verlies EUR 7,2 mln) | https://www.teleborsa.it/News/2026/04/22/sicily-by-car-valore-produzione-ed-ebitda-in-netta-crescita-nel-2025-utile-netto-in-rosso-per-oneri-non-ricorrenti-265.html |
| 2026-04-24 | MilanoFinanza: SBC cresce oltreconfine | https://www.milanofinanza.it/news/autonoleggio-sicily-by-car-cresce-oltreconfine-202604242238304228 |

### IPO-prospectus
- **Geraadpleegd?** Indirect (via Banca Akros deal-pagina + Euronext IPO-showcase + MilanoFinanza)
- **URL:** https://www.bancaakros.it/in-primo-piano/news/equity-capital-market/sicily-by-car-quotazione-a-seguito-business-combination-spac-industrial-stars-of-italy-4/
- **Pre-IPO data beschikbaar?** Documento di Ammissione bevat 2022 + meerjarige reeks, maar PDF niet rechtstreeks geëxtraheerd binnen deze sessie. Aanvragen bij IR mogelijk.
- **Pre-IPO bron:** Beperkt tot Euronext IPO-showcase samenvatting

### Non-GAAP
- **Gebruikt?** true (FY2025 adjusted-EBITDA, exclusief EUR 11,3 mln non-recurring)
- **Toelichting:** SBC rapporteerde adjusted-cijfers voor FY2025 om de impact van de AGCM-boete (EUR 8 mln) en accantonamenti (EUR 3,3 mln) te isoleren. Adjusted netto FY2025 ≈ EUR +0,9 mln vs reported -EUR 7,2 mln. Beide cijfers zijn meegenomen in de scorekaart en DCF-normalisatie; primaire grondslag blijft GAAP/IFRS.

### Ontbrekende data (eerlijke lijst)
- FY2015-FY2021 financiële reeks: geen openbare bilancio-PDF vindbaar; Documento di Ammissione 2023 bevat ze maar PDF niet geëxtraheerd
- FY2022 cijfers: PDF op IR-server aangewezen maar niet binnen sessie gelezen
- Jaarlijkse CFO/capex/FCF-reeks voor alle jaren: niet uit publieke nieuwsberichten te halen, bilancio-PDF nodig
- Aandelen-aantal eind 2025 exact: afgeleid via treasury-percentage 1,411%, niet rechtstreeks bevestigd
- Brutoschuld + lease-verplichtingen (IFRS-16) afzonderlijk: alleen netto PFN bekend
- Beloning per bestuurder (CEO, CFO, AD): bilancio noot "compensi" nodig
- CFO-naam: niet specifiek geverifieerd
- Personeel FTE: niet specifiek geverifieerd
- Volledige aandeelhouders-breakdown buiten Dragotto Holding + treasury: institutionele houders niet expliciet gevonden
- Damodaran Auto & Truck Europa WACC exact: waccEurope.xls vereist voor preciese referentie
- Sector-specifieke KPI's (fleet-utilization %, RPD/RevPAR, ADR vergelijking met peers): alleen partial (H1 2025 ADR +17,9% genoemd)

### Peildatum analyse
- 2026-05-14 (koers EUR 3,13)

---

## 14. Volledige bronnen-lijst

| Titel | URL | Type |
|---|---|---|
| Borsa Italiana - SBC profile | https://www.borsaitaliana.it/borsa/azioni/euronext-growth-milan/scheda/IT0005556581-EXGM.html?lang=en | beursdata |
| Yahoo Finance SBC.MI | https://finance.yahoo.com/quote/SBC.MI/ | beursdata |
| Sicily by Car IR persberichten map | https://group.sbc.it/public/InvComStampa/ | IR-pagina |
| Sicily by Car bilancio-archief | https://group.sbc.it/public/inv/BilRel/ | jaarverslag |
| Bilancio Consolidato 2024 | https://group.sbc.it/public/inv/BilRel/20241231BilCons.pdf | jaarverslag |
| Bilancio Civilistico 2024 | https://group.sbc.it/public/inv/BilRel/20241231BilEser.pdf | jaarverslag |
| Persbericht Risultati 2024 | https://group.sbc.it/public/InvComStampa/266_CSRisultati2024.pdf | beursmelding |
| H1 2024 relazione semestrale | https://group.sbc.it/public/InvComStampa/224_2024_09_26_SBC__CDA_1H2024.pdf | beursmelding |
| Teleborsa FY2025 | https://www.teleborsa.it/News/2026/04/22/sicily-by-car-valore-produzione-ed-ebitda-in-netta-crescita-nel-2025-utile-netto-in-rosso-per-oneri-non-ricorrenti-265.html | nieuwsartikel |
| FTAOnline FY2025 | https://www.ftaonline.com/sicily-by-car-nel-2025-valore-produzione-14-6-ma-perdita-da-eur-7-2-mln-per-gli-oneri-non-ricorrenti.html | nieuwsartikel |
| Soldionline FY2024 | https://www.soldionline.it/notizie/azioni-italia/conti-sicily-by-car-bilanci-2024-dividendo-2025 | nieuwsartikel |
| Agenparl assemblea 2025 (bilancio 2024) | https://agenparl.eu/2025/04/29/cs-sicily-by-car-lassemblea-approva-il-bilancio-2024/ | beursmelding |
| Agenparl H1 2025 | https://agenparl.eu/2025/09/25/cs-sicily-by-car-approva-i-risultati-al-30-giugno-2025/ | beursmelding |
| Agenparl 9M 2025 | https://agenparl.eu/2025/11/12/cs-sicily-by-car-prende-atto-dei-risultati-al-30-settembre-2025-valore-della-produzione-pari-a-euro-1330-milioni-in-crescita-del-159-rispetto-ai-9-mesi-2024-ebitda-pari-a-euro-385-milioni-15/ | beursmelding |
| Banca Akros deal-pagina | https://www.bancaakros.it/in-primo-piano/news/equity-capital-market/sicily-by-car-quotazione-a-seguito-business-combination-spac-industrial-stars-of-italy-4/ | onderzoeksrapport |
| Banca Akros upgrade Buy 2025 | https://www.teleborsa.it/News/2025/03/31/sicily-by-car-upgrade-di-banca-akros-a-buy-con-taglio-target-price-119.html | analistenrapport |
| Banca Akros downgrade 2024 | https://www.teleborsa.it/News/2024/10/04/sicily-by-car-downgrade-di-banca-akros-con-taglio-target-price-69.html | analistenrapport |
| AGCM-betaling maart 2026 | https://www.teleborsa.it/News/2026/03/24/sicily-by-car-paga-sanzione-da-8-milioni-di-euro-all-antitrust-in-attesa-del-tar-253.html | beursmelding |
| MilanoFinanza: cresce oltreconfine | https://www.milanofinanza.it/news/autonoleggio-sicily-by-car-cresce-oltreconfine-202604242238304228 | nieuwsartikel |
| MilanoFinanza: 60 jaar Dragotto | https://www.milanofinanza.it/news/sicily-by-car-dalla-prima-fiat-1300-alla-quotazione-in-borsa-i-60-anni-del-gruppo-di-tommaso-dragotto-202305262226466447 | nieuwsartikel |
| Mordor Intelligence Italy car rental | https://www.mordorintelligence.com/industry-reports/italy-car-rental-market | onderzoeksrapport |
| Mordor Intelligence Europe vehicle rental | https://www.mordorintelligence.com/industry-reports/europe-vehicle-rental-market | onderzoeksrapport |
| Sixt SE FY2024 persbericht | https://about.sixt.com/wp-content/uploads/2025/02/Sixt-SE_FY-2024_Press-Release.pdf | peer-data |
| Sixt SE FY2025 persbericht | https://about.sixt.com/wp-content/uploads/2026/03/2026_03_04_FY25_Master.pdf | peer-data |
| Avis Budget FY2025 | https://ir.avisbudgetgroup.com/news-releases/news-release-details/avis-budget-group-reports-fourth-quarter-and-full-year-results-1 | peer-data |
| Damodaran ERP historical implied | https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html | WACC-input |
| Damodaran country risk premium | https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html | WACC-input |
| Damodaran WACC Europe spreadsheet | https://pages.stern.nyu.edu/~adamodar/pc/datasets/waccEurope.xls | WACC-input |
| Trading Economics Italy 10y bond | https://tradingeconomics.com/italy/government-bond-yield | WACC-input |
| PwC Italy corporate tax summaries | https://taxsummaries.pwc.com/italy/corporate/taxes-on-corporate-income | macro-input |
| Wikipedia Tommaso Dragotto | https://it.wikipedia.org/wiki/Tommaso_Dragotto | achtergrond |
| MarketScreener Sicily by Car | https://www.marketscreener.com/quote/stock/SICILY-BY-CAR-S-P-A-157699363/ | aggregator |
| Sicily by Car company data EN | https://www.sicilybycar.it/en/company-data | IR-pagina |
| Seenews Croatia acquisitie | https://seenews.com/news/sicily-by-car-buys-two-croatian-companies-for-45-mln-euro-856372 | nieuwsartikel |
| Sicily by Car news 159 (Foderà) | https://www.sicilybycar.it/en/news/159 | IR-pagina |

---

## 15. Update-historie

| Datum | Versie | Wijziging |
|---|---|---|
| 2026-05-14 | 1.0 | Eerste publicatie. Bronnen-inventaris voltooid; 2021-2022 lege rijen omdat PDF-bilanci niet binnen sessie geëxtraheerd. |

---

## Opmerkingen voor Claude Code

Inhoudelijke twijfels en aandachtspunten waarbij stage-2 verificatie / aanvulling wenselijk is:

1. **Bilancio-PDF's voor FY2021 en FY2022 zijn niet binnen deze sessie geëxtraheerd.** De PDF 20221231bBilCons.pdf bestaat aantoonbaar op group.sbc.it. Claude Code zou de PDF kunnen openen via een browser- of fetch-stap om de 5-jaars-HOOG-eis voor 2021-2022 in te vullen. Mogelijk staan de FY2020-FY2021 cijfers in het Documento di Ammissione 2023; dit document is via IR aanvraagbaar maar niet via Google indexering vindbaar.

2. **CFO/capex/FCF-reeks per jaar ontbreekt.** Voor een autoverhuurder is de FCF-onderbouwing kritisch (vlootafschrijving vs maintenance capex onderscheid). De DCF in dit rapport gebruikt een nettowinst-gebaseerde FCF-proxy van EUR 5 mln; stage 2 zou met de echte CFO uit bilancio 2024 + 2025 dit moeten valideren of corrigeren. Dit kan de basis-FCF materieel verschuiven (range €3-8 mln plausibel) en dus ook de fair value.

3. **Aandelen-aantal eind 2025 (33,76 mln) is afgeleid uit treasury-percentage 1,411% en treasury-aantal 476.332.** Stage 2 zou de exacte capitale sociale uit het bilancio civilistico moeten bevestigen. Bij eventuele warrant-conversies of additional placements is dit aantal kleiner of groter.

4. **Beta is bottom-up geschat op 1,00 zonder onderliggende eigen regressie.** Stage 2 kan bottom-up beta uitrekenen via 5y weekly returns van Sixt SE + Avis Budget + Localiza, unleveren met de respectievelijke D/E-ratio's, mediaan nemen, en relevereren naar SBC (D/E ≈ 0 door cassa-positieve PFN). Verwachte range 0,85-1,10.

5. **Damodaran Europe Auto & Truck WACC** is niet exact opgenomen — Claude Code zou waccEurope.xls kunnen openen voor de precieze sector-benchmark om mijn WACC van 13,5% te valideren.

6. **De ROIC-berekening is grof.** Genormaliseerd ROIC ~3% is gebaseerd op NOPAT €4,32 mln / invested capital ~€130 mln, maar de exacte invested capital (= eigen vermogen + nettoschuld − cassa, of total assets − non-interest-bearing liabilities) zou uit bilancio gehaald moeten worden. Een nauwkeuriger ROIC kan de Buffett/Moat-scores beïnvloeden (huidig beide 1, maar bij ROIC dichter bij 5-7% zou Moat naar 2 kunnen).

7. **De adjusted EBIT 2025 (+€10,6 mln vs reported -€0,7 mln)** is mijn eigen aanpassing voor de €11,3 mln non-recurring oneri. Stage 2 zou kunnen verifiëren of dit (a) consistent is met SBC's eigen adjusted-rapportage, en (b) of de €3,3 mln "accantonamenti" (provisies) werkelijk eenmalig zijn of een recurring conservatisme van het management.

8. **AGCM-beroep bij TAR Lazio is een binaire katalysator.** Een succesvol beroep zou de €8 mln boete terugdraaien (positief voor 2026 winstgevendheid), een falend beroep zou de overhang wegnemen maar niet additioneel kosten. Datum uitspraak onbekend, vermoedelijk H2 2026 of H1 2027.

9. **Marktdiepte/liquiditeit op Euronext Growth Milan is beperkt.** Stage 2 kan dagvolume verifiëren via Borsa Italiana; mijn 10% illiquiditeitskorting is conservatief maar plausibel — kan ook 5-15% range zijn.

10. **Banca Akros target EUR 5,00-5,50 wijkt sterk af van mijn kansgewogen fair value EUR 2,44.** Het verschil zit waarschijnlijk in (a) hogere EBITDA-multiples die de broker hanteert, (b) lagere WACC (geen size+country premium toegepast), of (c) optimistischer groeiverwachtingen. Het is een rode vlag dat een specialist-broker zo'n grote upside ziet die ik methodisch niet kan reproduceren. Worth investigating in stage 2 — een Banca Akros rapport zelf inzien als beschikbaar.

11. **Geen kwartaalvolume-trend voor 2026.** Q1 2026 ricavi +17,1% is bemoedigend; H1 2026 (publicatie verwacht ~september 2026) is de eerstvolgende harde toetssteen.

12. **De koers EUR 3,13 is de close op of nabij 14 mei 2026; reële peildatum.** Indien Claude Code stage-2 op andere dag uitvoert, kan koers significant afwijken. Marktkap EUR 105,7 mln is conservatief afgerond.
