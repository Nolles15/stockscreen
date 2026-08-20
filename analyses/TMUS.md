# Research: TMUS — T-Mobile US, Inc.

## Bronnen-inventaris (Stap 0.5)

Alle financiële cijfers in dit rapport komen uit de XBRL-gegevens van de 10-K/10-Q-filings van T-Mobile US bij de SEC en uit de R-pagina's van die filings (de gerenderde jaarrekeningtabellen). Dat is de jaarrekening zelf, geen aggregator, en telt daarom als **HOOG**. Elke reeks is verankerd aan minstens één extern gecontroleerd punt (nettowinst ÷ verwaterde aandelen ≈ gerapporteerde WPA; omzet en nettowinst FY2025 tegen het Q4-2025-persbericht).

Er is **geen haallijst** nodig geweest: alle bronnen waren rechtstreeks bereikbaar.

```
Jaar 2025 — HOOG
  Bron: T-Mobile US Form 10-K FY2025 (accession 0001283699-26-000010, ingediend 11-02-2026)
  URL:  https://www.sec.gov/Archives/edgar/data/1283699/000128369926000010/tmus-20251231.htm
        R3 (balans), R5 (W&V), R7 (kasstroom), R100/R102/R103 (belastingnoot), R122 (supplementeel)
  Daadwerkelijk geopend: ja (R3, R5, R7, R100, R102, R103, R122)
  Cijfers overgenomen: omzet en omzetsplitsing, kostprijs diensten en apparatuur, SG&A, D&A,
                       EBIT, rentelast, belastinglast (actueel/uitgesteld), nettowinst, WPA,
                       aandelen, CFO, capex, spectrumuitgaven, dividend, inkoop, balans
                       (kas, schuld, leases, torenverplichtingen, eigen vermogen, goodwill),
                       betaalde rente, betaalde winstbelasting, NOL-standen
  Cijfers NIET overgenomen: (geen)

Jaar 2025 — HOOG (tweede bron, kruiscontrole non-GAAP)
  Bron: T-Mobile Q4 2025 Earnings Release (11-02-2026)
  URL:  https://s29.q4cdn.com/310188824/files/doc_financials/2025/q4/Q4-2025-Earnings-Release.pdf
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: Adjusted EBITDA, Core Adjusted EBITDA, Adjusted Free Cash Flow,
                       reconciliatie nettowinst→Adjusted EBITDA, guidance 2026
  Bijbehorend Investor Factbook (klant-KPI's 2024/2025):
  URL:  https://s29.q4cdn.com/310188824/files/doc_financials/2025/q4/Q4-2025-Investor-Factbook.pdf

Jaar 2024 — HOOG
  Bron: SEC XBRL companyconcept, us-gaap-tags uit de 10-K FY2024 en FY2025
  URL:  https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/Revenues.json en
        .../RevenueFromContractWithCustomerExcludingAssessedTax.json, OperatingIncomeLoss.json,
        NetIncomeLoss.json, NetCashProvidedByUsedInOperatingActivities.json,
        PaymentsToAcquirePropertyPlantAndEquipment.json, ShareBasedCompensation.json,
        PaymentsToAcquireIntangibleAssets.json, DepreciationDepletionAndAmortization.json,
        Assets.json, StockholdersEquity.json, CashAndCashEquivalentsAtCarryingValue.json,
        Goodwill.json, LongTermDebt.json, OperatingLeaseLiability.json, FinanceLeaseLiability.json,
        InterestPaidNet.json, IncomeTaxesPaidNet.json, IncomeTaxExpenseBenefit.json,
        WeightedAverageNumberOfDilutedSharesOutstanding.json, EarningsPerShareDiluted.json,
        PaymentsOfDividendsCommonStock.json, PaymentsForRepurchaseOfCommonStock.json
  Daadwerkelijk geopend: ja (alle bovenstaande endpoints afzonderlijk)
  Cijfers overgenomen: volledige reeks W&V, kasstroom en balans
  Cijfers NIET overgenomen: torenverplichtingen vóór 2024 (staan niet als losse tag in de reeks)

Jaar 2023 — HOOG
  Bron: idem SEC XBRL + 10-K FY2025 R5/R7 (vergelijkende kolom 2023)
  URL:  https://www.sec.gov/Archives/edgar/data/1283699/000128369926000010/R5.htm en R7.htm
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: volledige reeks
  Cijfers NIET overgenomen: torenverplichtingen

Jaar 2022 — HOOG
  Bron: SEC XBRL companyconcept (10-K FY2022, accession 0001283699-23-000016)
  URL:  https://www.sec.gov/Archives/edgar/data/1283699/000128369923000016/0001283699-23-000016-index.htm
  Daadwerkelijk geopend: ja (via de XBRL-endpoints hierboven)
  Cijfers overgenomen: volledige reeks
  Cijfers NIET overgenomen: torenverplichtingen

Jaar 2021 — HOOG
  Bron: SEC XBRL companyconcept (10-K FY2021, accession 0001283699-22-000018)
  URL:  https://www.sec.gov/Archives/edgar/data/1283699/000128369922000018/0001283699-22-000018-index.htm
  Daadwerkelijk geopend: ja (via de XBRL-endpoints hierboven)
  Cijfers overgenomen: volledige reeks
  Cijfers NIET overgenomen: torenverplichtingen

Jaar 2020 — HOOG
  Bron: SEC XBRL companyconcept (10-K FY2020)
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: volledige reeks
  Cijfers NIET overgenomen: torenverplichtingen

Jaar 2019 — HOOG
  Bron: SEC XBRL companyconcept (10-K FY2019 en FY2020)
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: volledige reeks incl. eerste jaar operationele/financiële leaseverplichtingen
  Cijfers NIET overgenomen: torenverplichtingen

Jaar 2018 — HOOG (deels)
  Bron: SEC XBRL companyconcept + 10-K FY2018 R6 (kasstroom, herzien na ASU 2016-15)
  URL:  https://www.sec.gov/Archives/edgar/data/1283699/000128369919000015/R6.htm
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: omzet, EBIT, nettowinst, WPA, aandelen, D&A, CFO, capex, SBC,
                       spectrumuitgaven, kas, totale activa, eigen vermogen, goodwill,
                       betaalde rente, betaalde belasting, belastinglast
  Cijfers NIET overgenomen: nettoschuld per 31-12-2018 — de tag `LongTermDebt` bevat voor die
                       datum geen eenduidige waarde en de balanspagina van de FY2019-10-K gaf
                       een 404. Cel blijft LEEG; genoteerd bij ontbrekende data.

Jaar 2017 — HOOG
  Bron: SEC XBRL companyconcept (10-K FY2017, plus herziene kasstroom uit de FY2018-10-K)
  URL:  https://www.sec.gov/Archives/edgar/data/1283699/000128369918000011/R6.htm (oorspronkelijk)
        https://www.sec.gov/Archives/edgar/data/1283699/000128369919000015/R6.htm (herzien)
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: volledige reeks; omzet is de in de FY2017-10-K herziene waarde;
                       CFO is de herziene waarde na ASU 2016-15 (3.831 i.p.v. 7.962)
  Cijfers NIET overgenomen: financiële/operationele leaseverplichtingen (ASC 842 gold nog niet)

Jaar 2016 — HOOG
  Bron: SEC XBRL companyconcept (10-K FY2016 en FY2017; herziene kasstroom uit FY2018-10-K;
        `LongTermDebt` per 31-12-2016 uit de 10-Q's van 2017)
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: omzet (herzien: 37.490), EBIT, nettowinst, WPA, aandelen, D&A,
                       CFO (herzien: 2.779), capex, SBC, spectrum, kas, activa, eigen vermogen,
                       goodwill, schuld, betaalde rente, betaalde belasting
  Cijfers NIET overgenomen: leaseverplichtingen (ASC 842 gold nog niet)

TTM (Q3 2025 t/m Q2 2026) — HOOG
  Bron: T-Mobile US Form 10-Q Q2 2026 + Q2 2026 Earnings Release
  URL:  https://s29.q4cdn.com/310188824/files/doc_financials/2026/q2/Q2-2026-FORM-10-Q-vFinal.pdf
        https://s29.q4cdn.com/310188824/files/doc_financials/2026/q2/Q2-2026-Earnings-Release-vFinal.pdf
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: halfjaarcijfers 2026 en 2025 (omzet, EBIT, nettowinst, WPA, D&A, SBC,
                       CFO, capex, spectrum, dividend, inkoop) en de balans per 30-06-2026
  Berekening TTM: FY2025 − H1 2025 + H1 2026. Elke TTM-cel is dus een optelsom van drie
                  geopende bronnen, geen schatting.

Macro- en marktinvoeren
  Risicovrije rente: US Treasury 10-jaars par yield 19-08-2026 = 4,65%
    https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value_month=202608
  ERP: Damodaran implied ERP S&P 500 per 01-08-2026 = 4,28% (trailing 12m, adjusted payout)
    https://pages.stern.nyu.edu/~adamodar/New_Home_Page/home.htm
  Sector-WACC ter controle: Damodaran dataset januari 2026, Telecom (Wireless):
    beta 0,54 · cost of equity 6,35% · cost of capital 5,48%
    https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/wacc.html
  Beta: 5-jaars regressiebeta 0,33 (stockanalysis.com, data-datum 19-08-2026), Blume-aangepast
    naar 0,55 → https://stockanalysis.com/stocks/tmus/statistics/
  Koers: slotkoers 19-08-2026 $182,36 → https://stockanalysis.com/stocks/tmus/history/

Wat NIET verifieerbaar bleek (en dus is weggelaten of leeg gelaten)
  - Nettoschuld per 31-12-2018
  - Torenverplichtingen vóór 2024
  - CEO pay ratio FY2025
  - Officieel gepubliceerd free-float-percentage
  - Marktaandelen op basis van een totaal-US postpaid-phone-markt (carriers publiceren
    sinds 2026 geen vergelijkbare abonnee-aantallen meer)
  - Klant-KPI's vóór 2024 in één consistente reeks

---

## Metadata
- **Ticker (bare):** TMUS
- **Yahoo symbol:** TMUS
- **Exchange:** NASDAQ
- **Sector (GICS-achtig):** Communicatie
- **Industrie:** Draadloze telecomdiensten
- **Land:** Verenigde Staten
- **Peildatum analyse:** 2026-08-19
- **Koers op peildatum:** 182,36
- **Valuta:** USD
- **Marktkapitalisatie:** USD 195,6 mrd
- **Marktkap in mln (lokale valuta):** 195612
- **Free float pct:** 45,5 (afgeleid: 100% − 54,5% stemcontrole Deutsche Telekom; SEC-berekende public float bedroeg USD 112,4 mrd per 30-06-2025)
- **Indexlidmaatschap:** S&P 500 (sinds 15-07-2019), Nasdaq-100 (sinds 21-12-2015)
- **Domein:** t-mobile.com

---

## 1. Executive summary

- **Kernthese:** T-Mobile US is de op één na grootste mobiele operator van de Verenigde Staten, met ruim 142 miljoen verbindingen onder de merken T-Mobile, Metro en Mint. Sinds de Sprint-fusie van 2020 heeft het de diepste midden-band-spectrumpositie van het land. Dat leverde een netwerkvoorsprong én de laagste kosten per gigabyte op. De servicerevenue groeide in 2025 met 8%, tegen 2 à 3% bij Verizon en AT&T. De operationele kasstroom klom van USD 8,6 mrd in 2020 naar USD 28,0 mrd in 2025. Sinds medio 2022 ging ruim USD 54 mrd naar aandeelhouders. Tegelijk verschuift het speelveld: kabelbedrijven pakken bijna de helft van alle nieuwe mobiele klanten en AT&T kocht in 2026 voor USD 23 mrd spectrum bij EchoStar. Daarbovenop loopt de fiscale rugwind van verliesverrekening en versnelde afschrijving in 2027 af. Het grootste risico is dat de groeipremie verdampt voordat de kasstroom haar waarmaakt.
- **Oordeel:** **HOLD**
- **Fair value basis** (basisscenario DCF): 313,38
- **Fair value kansgewogen**: 270,95
- **EPV per aandeel**: 160,49
- **Upside pct**: 71,8
- **Fair value scenarios:**

| Scenario | Fair value | Upside % | FCF groei % | WACC % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 121,99 | −33,1 | 1,0 | 7,63 | 35 |
| Basis | 313,38 | +71,8 | 4,0 | 5,88 | 45 |
| Optimistisch | 436,18 | +139,2 | 6,5 | 5,63 | 20 |

- **Reverse-DCF impliciete groei pct:** 0,74
- **Grootste kans:** Als T-Mobile zijn eigen 2027-doelen haalt (USD 20 mrd vrije kasstroom) en de markt de huidige waarderingskorting ten opzichte van de eigen historie opheft, ligt een koers boven de USD 240 binnen bereik zonder dat er iets bijzonders hoeft te gebeuren.
- **Grootste risico:** De Amerikaanse mobiele markt wordt een nulsomspel waarin kabel, AT&T en mogelijk SpaceX de groei afsnoepen terwijl de kasbelasting van USD 0,45 mrd naar USD 3,5 mrd loopt — dan groeit de vrije kasstroom per saldo niet meer en is het aandeel op de huidige koers te duur.

---

## 2. Bedrijfsprofiel

- **Beschrijving:** T-Mobile US verkoopt draadloze communicatie aan consumenten en bedrijven in de Verenigde Staten, Puerto Rico en de Amerikaanse Maagdeneilanden. Het bedrijf bezit en exploiteert een landelijk mobiel netwerk en verdient geld door daar abonnementen op te verkopen: spraak, sms en vooral data. Daarnaast levert het internet aan huis via datzelfde mobiele netwerk (fixed wireless access) en sinds kort ook via glasvezel-samenwerkingsverbanden. In de waardeketen zit T-Mobile helemaal aan het einde, bij de klant: het koopt spectrumvergunningen bij de overheid, bouwt daar zendmasten en radio's op, koopt toestellen in bij Apple, Samsung en anderen, en verkoopt het geheel als dienst via eigen winkels, landelijke ketens en online. Het onderscheidende is de spectrumpositie: door de overname van Sprint kreeg T-Mobile een brede 2,5 GHz-band die het als eerste landelijk voor 5G kon inzetten, waardoor het jarenlang meer capaciteit per klant had dan Verizon en AT&T en dus goedkoper kon zijn zonder marge in te leveren. Het probleem dat het oplost voor de klant is banaal maar universeel: altijd en overal verbinding, tegen een lagere prijs dan de gevestigde concurrenten. De omzet komt voor ruim 80% uit terugkerende maandelijkse abonnementsgelden; de rest is toestelverkoop, die tegen of onder kostprijs gaat en vooral als klantenbinding dient.
- **Geschiedenis:** T-Mobile USA ontstond toen Deutsche Telekom in 2001 het Amerikaanse VoiceStream Wireless kocht en het in 2002 omdoopte tot T-Mobile. Het bedrijf bleef jarenlang een verre nummer vier achter Verizon, AT&T en Sprint, met te weinig spectrum en te weinig schaal. In 2011 probeerde AT&T T-Mobile USA te kopen voor USD 39 mrd; toezichthouders blokkeerden die deal, waarna T-Mobile een afkoopsom in cash en spectrum ontving die de basis legde voor het latere herstel. Het keerpunt kwam op 1 mei 2013: MetroPCS Communications nam alle aandelen T-Mobile USA over van Deutsche Telekom in ruil voor ongeveer 74% van de aandelen in het gecombineerde bedrijf, dat als T-Mobile US onder de ticker TMUS ging noteren en toen ongeveer 43 miljoen abonnees had. Onder CEO John Legere volgde de Un-carrier-strategie: contracten afschaffen, roaming gratis maken, toestelsubsidies loskoppelen van abonnementen. Die aanpak dwong de hele sector mee en leverde T-Mobile jarenlang de hoogste netto klantengroei op. De tweede transformatie was de fusie met Sprint, aangekondigd in 2018 en afgerond op 1 april 2020, tegelijk met de overdracht van het roer aan Mike Sievert. Sprint bracht 2,5 GHz-spectrum mee dat T-Mobile in de jaren daarna landelijk uitrolde; de integratie kostte tot 2023 zware afschrijvingen en fusielasten, maar leverde daarna een sprong in marge en kasstroom op. Sindsdien breidde T-Mobile uit langs de randen: Mint en Ultra Mobile (2024), de draadloze activiteiten van UScellular (augustus 2025, circa USD 4,3 mrd) en glasvezel-joint-ventures met EQT (Lumos), KKR (Metronet), Oak Hill (GoNetspeed/Greenlight) en Wren House (i3 Broadband). Op 1 november 2025 nam Srini Gopalan het CEO-schap over van Sievert. In 2026 verkende Deutsche Telekom een volledige samensmelting met het Amerikaanse bedrijf; die gesprekken strandden eind juli 2026 op verzet van Amerikaanse minderheidsaandeelhouders en CFIUS-zorgen.
- **Bedrijfsmodel:** T-Mobile verdient geld aan terugkerende maandelijkse vergoedingen per lijn. In 2025 was USD 71,3 mrd van de USD 88,3 mrd omzet servicerevenue, waarvan USD 57,9 mrd uit postpaid-abonnementen en USD 10,5 mrd uit prepaid. De resterende USD 16,0 mrd is toestelverkoop, die grotendeels via betalingsregelingen over 24 tot 36 maanden loopt en waarop nauwelijks marge zit. De sturingsmaatstaf is niet de losse lijn maar het *account*: een huishouden of bedrijf met gemiddeld meerdere lijnen. In 2025 lag de gemiddelde omzet per account (ARPA) op USD 148,97 en het maandelijkse verloop op postpaid-telefoons op 0,93%. Omdat het netwerk een grotendeels vaste kostenbasis heeft, valt elke extra lijn met een hoge incrementele marge door naar de EBITDA — dat is de kern van het verdienmodel en de reden dat schaal hier zoveel waard is.
- **IPO-context:** T-Mobile US kreeg zijn beursnotering niet via een klassieke beursgang maar via een omgekeerde fusie: op 1 mei 2013 nam het beursgenoteerde MetroPCS T-Mobile USA over van Deutsche Telekom, voerde een omgekeerde aandelensplitsing van 1 op 2 door en betaalde USD 1,5 mrd aan zijn eigen aandeelhouders. Deutsche Telekom hield circa 74% van het gecombineerde bedrijf. Er is dus geen IPO-opbrengst geweest en geen prospectus met pre-IPO-cijfers; de kapitaalstructuur van 2013 was die van een dochter van Deutsche Telekom. Sinds 2020 is dat belang verwaterd naar circa 54,5% stemcontrole doordat Sprint-aandeelhouders (SoftBank) aandelen ontvingen.
- **Klantprofiel:** T-Mobile is overwegend een consumentenbedrijf: eind 2025 waren er 142,4 miljoen verbindingen, waarvan 85,6 miljoen postpaid-telefoonklanten, 25,9 miljoen prepaidklanten en 9,4 miljoen breedbandklanten. De klantenbasis is daarmee extreem gefragmenteerd — geen enkele afnemer is materieel. De belangrijkste zakelijke relaties zijn wholesale-partners: sinds juli 2025 draaien de zakelijke mobiele klanten van Comcast en Charter op het T-Mobile-netwerk, wat kapitaalarme omzet oplevert. Retentie is de kritieke variabele: bij een maandelijks verloop van 0,93% verlaat ruim 11% van de postpaid-telefoonklanten het bedrijf per jaar, en elke tiende procentpunt verloop kost of levert honderdduizenden klanten.
- **Oprichtingsjaar:** 1994 (als VoiceStream/Western Wireless; T-Mobile USA sinds 2002)
- **IPO-datum:** 2013-05-01 (omgekeerde fusie met MetroPCS; eerste notering NYSE, nu NASDAQ)
- **IPO-koers:** — (geen uitgifteprijs; omgekeerde fusie)
- **Personeel (FTE):** circa 75.000 (aggregator-bron)
- **Landen actief:** 1 (Verenigde Staten, incl. Puerto Rico en de Amerikaanse Maagdeneilanden)
- **Klantconcentratie:** Er is geen klantconcentratie in de gebruikelijke zin: de omzet komt van tientallen miljoenen huishoudens en kleine bedrijven, en geen enkele klant vertegenwoordigt een noemenswaardig deel van de omzet. Het enige concentratiepunt zit aan de wholesale-kant, waar Comcast en Charter sinds 2026 hun zakelijke mobiele verkeer bij T-Mobile onderbrengen. Dat is winstgevende omzet zonder eigen acquisitiekosten, maar het maakt T-Mobile tegelijk afhankelijk van partijen die in het consumentensegment zijn directe concurrent zijn.

### Geografische spreiding (omzet)
| Regio | Omzet % | Valuta-exposure |
|---|---|---|
| Verenigde Staten (incl. Puerto Rico en US Virgin Islands) | 100 | USD |

**Toelichting geografie:** T-Mobile US opereert uitsluitend binnen de Verenigde Staten en de bijbehorende territoria. Omzet, kosten, schuld en aandelennotering luiden allemaal in dollar, zodat er geen transactie- of translatierisico van betekenis is. Voor een Europese belegger verschuift het valutarisico daarmee volledig naar de EUR/USD-koers op portefeuilleniveau: het rendement in euro's is het dollarrendement plus of min de koersbeweging van de dollar. De enige indirecte buitenlandse blootstelling zit in de inkoop van netwerkapparatuur en toestellen, die in dollars wordt afgerekend maar door wisselkoersen en handelstarieven in prijs kan bewegen.

### Segmenten
| Naam | Omzet % | Beschrijving |
|---|---|---|
| Postpaid-diensten | 65,6 | Maandelijkse abonnementen voor consumenten en bedrijven, het winstcentrum van het bedrijf; de groei komt uit nieuwe accounts, meer lijnen per account en migratie naar duurdere tariefplannen. |
| Prepaid-diensten | 11,9 | Metro by T-Mobile, Mint Mobile en Ultra Mobile; prijsgevoelig segment met hoger verloop (2,72% per maand) maar zonder kredietrisico en met lage acquisitiekosten. |
| Toestelverkoop en overig | 19,3 | Verkoop van smartphones en tablets, grotendeels via betalingsregelingen; nauwelijks marge en vooral bedoeld om klanten te binden. |
| Wholesale en overige diensten | 3,3 | MVNO-capaciteit voor derden, waaronder sinds 2026 het zakelijke verkeer van Comcast en Charter; kapitaalarme omzet op bestaande netwerkcapaciteit. |

### Aandeelhouders (top 5)
| Naam | Belang % | Type (oprichter / PE / institutioneel / retail) |
|---|---|---|
| Deutsche Telekom AG | 54,5 (stemcontrole per 31-03-2026) | Controlerend |
| SoftBank Group Capital Ltd | 4,2 (per 22-10-2025; nadien verder afgebouwd) | Institutioneel |
| BlackRock, Inc. | 3,68 (per 30-06-2026) | Institutioneel |
| Vanguard Group | 3,02 (per 31-03-2026) | Institutioneel |
| T. Rowe Price Group | 2,66 (per 31-03-2026) | Institutioneel |

- **Institutioneel eigendomstrend:** Stabiel tot licht dalend. Het grote bewegende deel is SoftBank, dat sinds juni 2025 systematisch verkoopt en zijn belang van 7,5% terugbracht naar circa 4%. Deutsche Telekom verklaarde in februari 2026 juist géén aandelen te willen verkopen en te blijven kijken naar verhoging van zijn belang. De klassieke indexbeleggers (BlackRock, Vanguard, State Street) zijn stabiel; hun gezamenlijke belang beweegt mee met de inkoopprogramma's.

---

## 3. Financieel — historische data (10 jaar + TTM)

### Resultatenrekening (bedragen in USD mln)

| Jaar | Omzet | Omzetgroei % | Brutowinst | Brutomarge % | EBIT | EBIT-marge % | EBITDA | EBITDA-marge % | Nettowinst | Nettomarge % | EPS | EPS-groei % | Aandelen mln |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2016 | 37490 | — | — | — | 3802 | 10,1 | 10045 | 26,8 | 1460 | 3,9 | 1,69 | — | 833,0 |
| 2017 | 40604 | 8,3 | — | — | 4888 | 12,0 | 10872 | 26,8 | 4536 | 11,2 | 5,20 | 207,7 | 871,8 |
| 2018 | 43310 | 6,7 | — | — | 5309 | 12,3 | 11795 | 27,2 | 2888 | 6,7 | 3,36 | -35,4 | 858,3 |
| 2019 | 44998 | 3,9 | — | — | 5722 | 12,7 | 12338 | 27,4 | 3468 | 7,7 | 4,02 | 19,6 | 863,4 |
| 2020 | 68397 | 52,0 | — | — | 6636 | 9,7 | 20787 | 30,4 | 3064 | 4,5 | 2,65 | -34,1 | 1154,8 |
| 2021 | 80118 | 17,1 | — | — | 6892 | 8,6 | 23275 | 29,1 | 3024 | 3,8 | 2,41 | -9,1 | 1254,8 |
| 2022 | 79571 | -0,7 | — | — | 6543 | 8,2 | 20194 | 25,4 | 2590 | 3,3 | 2,06 | -14,5 | 1255,4 |
| 2023 | 78558 | -1,3 | 48370 | 61,6 | 14266 | 18,2 | 27084 | 34,5 | 8317 | 10,6 | 6,93 | 236,4 | 1200,3 |
| 2024 | 81400 | 3,6 | 51747 | 63,6 | 18010 | 22,1 | 30929 | 38,0 | 11339 | 13,9 | 9,66 | 39,4 | 1173,2 |
| 2025 | 88309 | 8,5 | 55535 | 62,9 | 18279 | 20,7 | 31787 | 36,0 | 10992 | 12,4 | 9,72 | 0,6 | 1131,1 |
| TTM | 92189 | 4,4 | — | — | 18253 | 19,8 | 32668 | 35,4 | 10560 | 11,5 | 9,56 | -1,6 | 1105,0 |

### Kasstromen (USD mln)

| Jaar | CFO | Capex | Spectrum/immaterieel | FCF | FCF na SBC | FCF/aandeel | FCF-marge % | FCF-groei % | FCF-conversie % | SBC | Dividend totaal | Aandeleninkoop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2016 | 2779 | 4702 | 3968 | -1923 | -2158 | -2,59 | -5,8 | — | -147,8 | 235 | 0 | 0 |
| 2017 | 3831 | 5237 | 5828 | -1406 | -1712 | -1,96 | -4,2 | — | -37,7 | 306 | 0 | 427 |
| 2018 | 3899 | 5541 | 127 | -1642 | -2066 | -2,41 | -4,8 | — | -71,5 | 424 | 0 | 1071 |
| 2019 | 6824 | 6391 | 967 | 433 | -62 | -0,07 | -0,1 | — | -1,8 | 495 | 0 | 0 |
| 2020 | 8640 | 11034 | 1333 | -2394 | -3088 | -2,67 | -4,5 | — | -100,8 | 694 | 0 | 19536 |
| 2021 | 13917 | 12326 | 9366 | 1591 | 1051 | 0,84 | 1,3 | — | 34,8 | 540 | 0 | 0 |
| 2022 | 16781 | 13970 | 3331 | 2811 | 2216 | 1,77 | 2,8 | 110,8 | 85,6 | 595 | 0 | 3000 |
| 2023 | 18559 | 9801 | 1010 | 8758 | 8091 | 6,74 | 10,3 | 265,1 | 97,3 | 667 | 747 | 13074 |
| 2024 | 22293 | 8840 | 3471 | 13453 | 12804 | 10,91 | 15,7 | 58,2 | 112,9 | 649 | 3300 | 11228 |
| 2025 | 27950 | 9955 | 2568 | 17995 | 17166 | 15,18 | 19,4 | 34,1 | 156,2 | 829 | 4121 | 9974 |
| TTM | 28833 | 10434 | 2163 | 18399 | 17533 | 15,87 | 19,0 | — | 166,0 | 866 | 4343 | 12071 |

### Balans-ratio's (10 jaar + TTM)

| Jaar | Nettoschuld | Nettoschuld/EBITDA | Eigen vermogen | ROE % | ROIC % | Solvabiliteit % | Goodwill % van activa | Boekwaarde/aandeel | Accruals-ratio % |
|---|---|---|---|---|---|---|---|---|---|
| 2016 | 22286 | 2,22 | 18236 | 8,0 | 7,04 | 27,7 | 2,6 | 21,89 | -2,00 |
| 2017 | 27100 | 2,49 | 22559 | 20,1 | 7,38 | 32,0 | 2,4 | 25,88 | 1,00 |
| 2018 | — | — | 24718 | 11,7 | — | 34,1 | 2,6 | 28,80 | -1,40 |
| 2019 | 25744 | 2,09 | 28789 | 12,0 | 7,87 | 33,1 | 2,2 | 33,34 | -3,86 |
| 2020 | 63247 | 3,04 | 65344 | 4,7 | 3,87 | 32,6 | 5,6 | 56,59 | -2,79 |
| 2021 | 70137 | 3,01 | 69102 | 4,4 | 3,71 | 33,5 | 5,9 | 55,07 | -5,27 |
| 2022 | 69984 | 3,47 | 69656 | 3,7 | 3,51 | 33,0 | 5,8 | 55,49 | -6,71 |
| 2023 | 72379 | 2,67 | 64715 | 12,9 | 7,80 | 31,2 | 5,9 | 53,92 | -4,93 |
| 2024 | 78846 | 2,55 | 61741 | 18,4 | 11,73 | 29,7 | 6,3 | 52,63 | -5,27 |
| 2025 | 86486 | 2,72 | 59203 | 18,6 | 11,42 | 27,0 | 6,2 | 52,34 | -7,74 |
| TTM | 87556 | 2,68 | 56265 | 18,8 | 11,59 | 26,3 | 6,4 | 52,45 | -8,56 |

- **Toelichting resultaten:** De omzet groeide van USD 37,5 mrd (2016) naar USD 88,3 mrd (2025), een samengestelde groei van 10,0% per jaar, maar dat cijfer wordt gedomineerd door één gebeurtenis: de fusie met Sprint per 1 april 2020, die de omzet in één jaar met 52% deed springen. Zonder die sprong is het beeld anders. Tussen 2021 en 2023 stagneerde de omzet zelfs licht doordat wholesale-contracten wegvielen en Sprint-klanten werden gemigreerd. Veel interessanter is de margelijn: de EBIT-marge zakte van 12,7% (2019) naar 8,2% (2022) onder het gewicht van integratiekosten en fusieafschrijvingen, en veerde daarna terug naar 22,1% in 2024 en 20,7% in 2025. Die ommekeer, niet de omzetgroei, verklaart waarom de winst per aandeel van USD 2,06 (2022) naar USD 9,72 (2025) ging. De omzetcijfers van 2015 en 2016 zijn herzien in de 10-K over 2017; ik gebruik de herziene versies.
- **Omzet-CAGR:** 10,0% over 2016-2025 (sterk vertekend door de Sprint-fusie in 2020)

- **Toelichting kasstromen:** De operationele kasstroom steeg van USD 8,6 mrd (2020) naar USD 28,0 mrd (2025) — meer dan een verdrievoudiging. Drie dingen verdienen uitleg. Ten eerste: de cijfers voor 2016 en 2017 in de tabel zijn de herziene bedragen na invoering van ASU 2016-15; oorspronkelijk rapporteerde T-Mobile USD 6,1 mrd en USD 8,0 mrd, maar de inning van uitgestelde koopprijs op verkochte vorderingen verhuisde naar de investeringskasstroom. Op vergelijkbare basis steeg de kasstroom in 2018 met 1,8%, terwijl de ongecorrigeerde vergelijking een halvering suggereert. Ten tweede: de capexpiek van USD 12 à 14 mrd in 2020-2022 was de landelijke 5G-uitrol; sindsdien ligt de capex rond USD 10 mrd. Ten derde, en het belangrijkste: de vrije kasstroom vloog omhoog terwijl de nettowinst dat niet deed, omdat T-Mobile nauwelijks kasbelasting betaalt — zie de sectie earnings quality.

### Kapitaalstructuur huidig (per 30-06-2026)
- **Nettoschuld (huidig):** 87.556
- **Bruto rentedragende schuld:** 90.381 (kortlopend 6.117 + langlopend 78.504 + financiële leases 2.299 + torenverplichtingen 3.461)
- **Cash & equivalents:** 2.825
- **Operationele leaseverplichtingen (ASC 842):** 29.058 (3.620 kortlopend + 25.438 langlopend) — **niet** in de nettoschuld opgenomen, zie toelichting balans
- **Gemiddelde rente %:** 4,26 (betaalde rente 2025 van 3.882 gedeeld door de gemiddelde bruto rentedragende schuld van 91.233)
- **Rente-dekking (EBIT/rente):** 4,84 (EBIT 2025 van 18.279 gedeeld door rentelast 3.774)

- **Toelichting balans:** De balans is zwaar en dat is inherent aan het bedrijf: USD 98,0 mrd aan spectrumvergunningen en USD 38,3 mrd aan netwerkactiva tegenover USD 90,4 mrd rentedragende schuld. De bruto schuld liep in 2025 op van USD 78,3 mrd naar USD 86,3 mrd, terwijl het eigen vermogen daalde van USD 61,7 mrd naar USD 59,2 mrd — beide bewegingen komen van hetzelfde: T-Mobile financiert de aandeleninkoop deels met schuld. De nettoschuld/EBITDA-verhouding bleef daarbij netjes rond 2,7, ruim onder het niveau van AT&T (circa 2,5 op een veel grotere absolute schuld) en in lijn met de eigen doelstelling van 2,5. Eén methodische keuze: de operationele leaseverplichting van USD 29,1 mrd tel ik **niet** mee in de nettoschuld, omdat de leasekosten onder US GAAP volledig in de bedrijfslasten en dus in de EBIT en de operationele kasstroom zitten. Ze twee keer meetellen — als kosten én als schuld — zou de waarde dubbel verlagen. Financiële leases en torenverplichtingen tel ik wél mee: die aflossingen lopen buiten de operationele kasstroom om.

### Earnings quality

| Jaar | Nettowinst | CFO | Accruals-ratio % | Belastinglast | Betaalde belasting | Betaald / last % |
|---|---|---|---|---|---|---|
| 2021 | 3.024 | 13.917 | −5,27 | 327 | 167 | 51,1 |
| 2022 | 2.590 | 16.781 | −6,71 | 556 | 76 | 13,7 |
| 2023 | 8.317 | 18.559 | −4,93 | 2.682 | 149 | 5,6 |
| 2024 | 11.339 | 22.293 | −5,27 | 3.373 | 211 | 6,3 |
| 2025 | 10.992 | 27.950 | −7,74 | 3.289 | 451 | 13,7 |

- **Toelichting earnings quality:** De accruals-ratio is elk jaar negatief en wordt steeds negatiever (van −4,9% in 2023 naar −7,7% in 2025). Op zichzelf is dat het gunstige teken: de kasstroom loopt vóór op de winst, niet andersom. Maar de oorzaak is geen conservatieve boekhouding — het is een belastingvakantie. Van de belastinglast van USD 3.289 mln in 2025 was slechts USD 425 mln actueel en USD 2.864 mln uitgesteld; er ging maar USD 451 mln daadwerkelijk de deur uit, een kaseffectief tarief van 3,2% tegen een boektarief van 23,0%. De oorzaken zijn verliesverrekening uit de Sprint-erfenis (de bijbehorende belastingvordering daalde in 2025 van USD 3,8 mrd naar USD 2,5 mrd) en 100% versnelde afschrijving onder de One Big Beautiful Bill Act. T-Mobile zegt zelf volledig kasbelastingplichtig te worden in 2027 en begroot USD 1,5 mrd kasbelasting voor 2026 en USD 3,5 mrd voor 2027. Uitstel, geen kwijtschelding — en precies daarom normaliseer ik de kasbelasting in de DCF.

### Non-GAAP / aanpassingen
- **Gebruikt?** true
- **Welke aanpassingen:** T-Mobile rapporteert Adjusted EBITDA en Core Adjusted EBITDA (USD 33.937 mln respectievelijk USD 33.924 mln over 2025) en Adjusted Free Cash Flow (USD 17.995 mln over 2025). De brug van EBIT naar Adjusted EBITDA loopt via D&A (13.508), aandelencompensatie (772), fusiekosten (263), netwerkherstructurering (93), juridische posten (16), bijzondere waardevermindering (278) en overige posten (728).
- **Waarom:** Het bedrijf stuurt intern op deze maatstaven en de bonusdoelstellingen zijn eraan opgehangen. Twee bezwaren wegen door in mijn analyse. De aandelencompensatie van USD 772 mln wordt teruggeteld terwijl het een reële kostenpost voor aandeelhouders is; ik trek die in de FCF juist af. En de Adjusted Free Cash Flow houdt geen rekening met spectrumuitgaven, terwijl die voor een mobiele operator structureel zijn: over 2016-2025 gaf T-Mobile er gemiddeld USD 3,2 mrd per jaar aan uit. Ik gebruik GAAP als grondslag en corrigeer beide punten expliciet.

### Sector-KPI's (draadloze telecom)

| KPI | 2024 | 2025 | Q2 2026 |
|---|---|---|---|
| Totaal klanten (× 1.000, einde periode) | 129.528 | 142.388 | — |
| Postpaid-telefoonklanten (× 1.000) | 79.013 | 85.594 | — |
| Postpaid-telefoon netto-aanwas (× 1.000) | 3.077 | 3.294 | — |
| Netto-aanwas postpaid-accounts (× 1.000) | 1.097 | 1.180 | 277 (kwartaal) |
| Totaal postpaid-accounts (× 1.000) | — | — | 34.700 |
| Prepaidklanten (× 1.000) | 25.410 | 25.943 | — |
| Breedbandklanten (× 1.000) | 6.439 | 9.447 | — |
| ARPA (USD per maand) | 143,85 | 148,97 | 152,91 |
| ARPU postpaid-telefoon (USD per maand) | 49,35 | 50,37 | — |
| Verloop postpaid-telefoon (% per maand) | 0,86 | 0,93 | 0,85 |
| Verloop postpaid-accounts (% per maand) | — | — | 0,99 |
| Verloop prepaid (% per maand) | 2,73 | 2,72 | — |

- **Toelichting sector-KPI's:** De reeks loopt maar twee jaar terug omdat de Amerikaanse operators sinds 2026 geen onderling vergelijkbare postpaid-telefoonstanden meer publiceren; T-Mobile rapporteert nu alleen nog accounts. Wat de cijfers laten zien is een bedrijf dat nog steeds klanten wint maar tegen oplopende kosten: de netto-aanwas van accounts groeide in 2025 met 8%, maar in het tweede kwartaal van 2026 daalde die met 13% jaar-op-jaar naar 277.000 en het management stuurt aan op circa 250.000 in het derde kwartaal. Tegelijk stijgt de ARPA met 2 à 4% per jaar — T-Mobile ruilt volume in voor prijs. Het verloop kroop op van 0,86% naar 0,93% en dat was vóór de gedwongen tariefmigratie van juli 2026.

### Dividendhistorie

| Jaar | DPS (USD) | Groei YoY % | Uitkeringsratio op EPS % | Dividend totaal (mln) | FCF-dekkingsratio | Bijzonderheden |
|---|---|---|---|---|---|---|
| 2016-2022 | 0,00 | — | 0 | 0 | — | Geen dividend; alle kasstroom naar netwerk en schuldafbouw |
| 2023 | 0,65 | — | 9,4 | 747 | 10,83 | Eerste dividend in de bedrijfsgeschiedenis, betaald 15-12-2023 |
| 2024 | 2,83 | +335,4 | 29,3 | 3.300 | 3,88 | Eerste volledige jaar; verhoging naar 0,88 per kwartaal in Q4 |
| 2025 | 3,66 | +29,3 | 37,7 | 4.121 | 4,17 | Verhoging naar 1,02 per kwartaal per november 2025 (+16%) |
| 2026 (verwacht) | 4,08 | +11,5 | 42,7 | — | — | Vier kwartalen à 1,02; volgende ex-datum 28-08-2026 |

- **Dividend — toelichting:** T-Mobile keerde tot 2023 nooit dividend uit; alle kasstroom ging naar spectrum, netwerk en schuld. In september 2023 kondigde het bedrijf een aandeelhoudersreturn-programma van tot USD 19 mrd aan met daarin het eerste dividend ooit. Sindsdien is het dividend twee keer verhoogd: van 0,65 naar 0,88 per kwartaal (november 2024, +35,4%) en van 0,88 naar 1,02 (november 2025, +16%). Op de huidige koers levert dat een rendement van 2,24% op — bescheiden naast Verizon (5,7%) en AT&T (4,4%), maar met een heel andere groeivoet. De dekking is comfortabel: de gerapporteerde vrije kasstroom na aandelencompensatie dekt het dividend ruim vier keer. Op genormaliseerde basis, dus na volledige kasbelasting en na een normale spectrumuitgave, valt die dekking terug naar 2,6 keer. Dat is nog altijd ruim, maar de marge is de helft kleiner dan de gerapporteerde cijfers suggereren.
- **Oordeel houdbaarheid:** Conservatief. Zelfs op genormaliseerde basis gaat minder dan 40% van de vrije kasstroom naar dividend, en het management houdt daarnaast ruimte voor USD 10 mrd aan inkoop per jaar. Het dividend is het eerste dat overeind blijft als het tegenzit; de inkoop is de schokdemper.

---

## 4. Moat (concurrentievoordeel)

- **Oordeel:** **NARROW MOAT**
- **Moat-categorieën:**

| Naam | Sterkte | Toelichting |
|---|---|---|
| Immateriële activa | sterk | De spectrumvergunningen staan voor USD 98,0 mrd op de balans en zijn de facto onvervangbaar: ze worden door de FCC in schaarse veilingen uitgegeven en er komt pas in juli 2027 weer 160 MHz bij. T-Mobile bezit de diepste landelijke 2,5 GHz-positie, verworven via Sprint, en won in veiling 108 nog eens ruim 7.000 county-vergunningen voor USD 304 mln. Dat bezit is juridisch beschermd, verhandelbaar en niet te repliceren door een nieuwkomer. |
| Overstapkosten | zwak | Nummerbehoud is wettelijk verplicht, contracten zijn afgeschaft (mede door T-Mobile zelf) en toestelbetalingsregelingen worden door concurrenten routinematig afgekocht. Het maandelijkse verloop van 0,93% betekent dat ruim 11% van de klanten per jaar vertrekt. De enige echte rem zijn lopende toestelafbetalingen en gezinsbundels met meerdere lijnen — en zelfs die hielden de klantenboosheid na de tariefmigratie van juli 2026 niet tegen. |
| Netwerkeffecten | geen | Een extra abonnee maakt het netwerk voor de bestaande abonnees niet waardevoller; hij verbruikt juist capaciteit. Er is geen platform-, marktplaats- of communitylogica in draadloze telecom. Wat mensen "netwerkeffect" noemen in deze sector is in werkelijkheid schaalvoordeel aan de kostenkant. |
| Kostenvoordeel | sterk | Doordat T-Mobile meer midden-band-spectrum per klant heeft, kost het transporteren van een gigabyte er minder dan bij Verizon en AT&T: dezelfde mast draagt meer verkeer. Daar komt de Sprint-schaal bovenop, met circa USD 8 mrd aan gerealiseerde jaarlijkse synergieën en een verwachte USD 1,2 mrd extra uit UScellular. Dat is de reden dat T-Mobile structureel goedkoper kan aanbieden en tóch een EBIT-marge van ruim 20% haalt. |
| Efficiënte schaal | middel | De Amerikaanse markt heeft nog maar drie landelijke netwerken en de toetredingsdrempel — spectrum plus honderd miljard aan infrastructuur — is prohibitief. Maar "efficiënte schaal" veronderstelt dat er geen ruimte is voor een vierde speler, en die veronderstelling brokkelt af: kabelbedrijven pakten in 2025 bijna de helft van alle nieuwe postpaid-klanten via MVNO-contracten, en SpaceX verwierf in 2025-2026 voor circa USD 20 mrd aan spectrum. |

- **Kwantitatief bewijs:** Het rendement op geïnvesteerd kapitaal (ROIC) lag in 2016-2019 rond 7 à 8%, zakte tijdens de Sprint-integratie naar 3,5 à 3,9% (2020-2022) en herstelde naar 7,8% (2023), 11,7% (2024) en 11,4% (2025). Tegenover een WACC van 5,88% is de spread in de laatste twee jaar dus 5,5 tot 5,9 procentpunt — reëel maar geen fort. Over vijf jaar gemiddeld is de spread slechts 1,75 procentpunt, omdat 2021 en 2022 onder de kapitaalkosten lagen. De EBIT-marge bewoog van 12,7% (2019) via 8,2% (2022) naar 20,7% (2025); dat is geen stabiele-margeprofiel maar een integratiecyclus. Het marktaandeel binnen de drie landelijke operators steeg van circa 30% naar circa 34% (85,6 mln van de 253,7 mln gepubliceerde postpaid-telefoonverbindingen eind 2025).
- **Duurzaamheid:** 5 jaar, niet 10. De spectrum- en kostenvoordelen zijn hard en juridisch beschermd, maar de voorsprong in mid-band is aan het krimpen. AT&T rondde in juli 2026 een spectrumdeal van USD 23 mrd met EchoStar af en had de 3,45 GHz-band bij closing al op circa 23.000 masten live, met tot 80% hogere snelheden. Verizon was met USD 3,2 mrd de grootste winnaar van de AWS-3-veiling van juni 2026. In juli 2027 komt 160 MHz upper C-band onder de hamer, waar Verizon en AT&T als grootste bieders gelden en T-Mobile beperkte ruimte heeft naast een kapitaalenvelop die al voor aandeelhoudersuitkeringen is bestemd. Op tienjaarshorizon is er geen reden aan te nemen dat T-Mobile nog steeds het beste netwerk per dollar heeft.
- **Erosierisico's:** Vier concrete bedreigingen. Ten eerste kabel: Comcast en Charter hadden eind juni 2026 samen 22,7 miljoen mobiele lijnen en nemen bijna de helft van de marktgroei, met gratis lijnen als lokmiddel. Ten tweede AT&T's spectruminhaalslag. Ten derde satelliet: SpaceX overweegt volgens berichtgeving uit juni 2026 een eigen retail-mobielmerk, en AST SpaceMobile bouwt met AT&T en Verizon aan directe satelliet-naar-telefoonverbindingen. Ten vierde de krimpende groeipool zelf — een analistenraming wijst op een terugval van circa 7,3 miljoen naar 5,8 miljoen jaarlijkse netto-aanwas bij lagere immigratie, waardoor de markt een nulsomspel wordt.

---

## 5. Management

- **CEO-naam + tenure:** Srini Gopalan, CEO sinds 1 november 2025 (circa 10 maanden); daarvoor COO van T-Mobile US sinds maart 2025
- **CFO-naam + tenure:** Peter Osvaldik, CFO sinds 1 juli 2020 (ruim 6 jaar); bij het bedrijf sinds januari 2016
- **Oprichter nog betrokken?** Nee. Voormalig CEO Mike Sievert (2020-2025) is sinds november 2025 Vice Chairman met een basissalaris van USD 7 mln en zonder verdere variabele beloning; zijn termijn liep aanvankelijk tot 1 november 2026.
- **Insider ownership %:** 0,32 (alle insiders samen); CEO Gopalan hield na zijn aankoop van 6 november 2025 90.258 aandelen
- **Capital allocation track record:**

| Jaar | Dividend totaal | Aandeleninkoop | Spectrum/immaterieel | Organische capex |
|---|---|---|---|---|
| 2021 | 0 | 0 | 9.366 | 12.326 |
| 2022 | 0 | 3.000 | 3.331 | 13.970 |
| 2023 | 747 | 13.074 | 1.010 | 9.801 |
| 2024 | 3.300 | 11.228 | 3.471 | 8.840 |
| 2025 | 4.121 | 9.974 | 2.568 | 9.955 |
| TTM | 4.343 | 12.071 | 2.163 | 10.434 |

- **M&A-track-record:** Twee transformerende deals en een reeks kleinere. De fusie met MetroPCS (2013) leverde de beursnotering en schaal in prepaid. De fusie met Sprint (afgerond 1 april 2020) leverde het 2,5 GHz-spectrum dat de hele latere winstgevendheid draagt; de aangekondigde synergieën zijn ruimschoots gehaald en de EBIT-marge verdubbelde. Daarna: Ka'ena (Mint/Ultra Mobile, mei 2024, tot USD 1,35 mrd), de draadloze activiteiten van UScellular (1 augustus 2025, circa USD 4,3 mrd inclusief overgenomen schuld, synergiedoel verhoogd naar USD 1,2 mrd per jaar), Vistar Media en Blis in advertentietechnologie (2025, circa USD 175 mln voor Blis), en vier 50/50 glasvezel-joint-ventures: Lumos met EQT (circa USD 950 mln), Metronet met KKR (circa USD 4,9 mrd), GoNetspeed/Greenlight met Oak Hill (circa USD 2,0 mrd) en i3 Broadband met Wren House (circa USD 700 mln). Er is in deze periode geen goodwill afgeschreven. Het glasvezelavontuur is de openstaande vraag: circa USD 8,5 mrd toegezegd voor een businessmodel dat T-Mobile niet van huis uit beheerst, met eind 2025 nog maar 997.000 glasvezelklanten.
- **Beloning:** De totale beloning van Gopalan bedroeg over 2025 USD 35,4 mln (grotendeels aandelentoekenningen, inclusief eenmalige aantreedcomponenten) en die van Sievert USD 50,4 mln, waarin USD 13,9 mln aan vertrekgerelateerde posten. Osvaldik kwam op USD 13,2 mln. De jaarbonus over 2025 keerde uit op 161% van target en hangt aan vier maatstaven: servicerevenue (20%), totale netto-aanwas (20%), Core Adjusted EBITDA (30%) en Adjusted Free Cash Flow (30%). De langetermijnbeloning is voor de helft prestatiegebonden en meet relatieve total shareholder return en Adjusted Free Cash Flow; uitkeringen zijn gemaximeerd op 200% van target. Circa 92% van de doelbeloning is variabel. Aandelenbezitseisen: 5× basissalaris voor de CEO, 3× voor overige bestuurders.
- **Insider-transacties (augustus 2024 – augustus 2026):** twee open-markt aankopen tegenover 63 verkopen. Gopalan kocht op 6 november 2025 9.800 aandelen à USD 201,82 (USD 1,98 mln) en André Almeida op 1 mei 2026 5.097 aandelen à USD 196,18 (USD 1,00 mln); beide zijn geverifieerd in de Form 4's bij de SEC. Daartegenover staat voor circa USD 395 mln aan verkopen, waarvan alleen bestuurder Marcelo Claure al ruim USD 220 mln en Sievert ruim USD 75 mln. Netto verkochten insiders circa 1,78 miljoen aandelen voor ongeveer USD 392 mln.
- **Oordeel management:** **NEUTRAAL**
- **Toelichting:** De operationele en strategische staat van dienst is sterk: de Sprint-integratie is uitgevoerd zoals beloofd, de marges verdubbelden en er ging sinds medio 2022 ruim USD 54 mrd naar aandeelhouders. Drie dingen houden het oordeel op neutraal in plaats van sterk. Het eigen belang van insiders is met 0,32% verwaarloosbaar en de netto insiderverkopen van bijna USD 400 mln in twee jaar staan tegenover twee symbolische aankopen. De timing van de inkoop is ongelukkig: circa USD 34 mrd werd tussen 2023 en 2025 ingekocht, waarvan het leeuwendeel boven de huidige koers. En de governance is die van een controlled company waarin Deutsche Telekom tien van de dertien bestuurders benoemt — in 2026 bleek hoe dat kan wringen toen het moederbedrijf een fusie verkende die Amerikaanse minderheidsaandeelhouders als onderwaardering zagen.

---

## 6. Sector & concurrentie

- **Sector-groeivooruitzicht:** Er is geen geverifieerde onafhankelijke prognose voor de Amerikaanse draadloze servicerevenue tot 2030 gevonden. Wel de guidance van de spelers zelf: T-Mobile mikt op USD 77,0 mrd servicerevenue in 2026 en USD 80,5-81,5 mrd in 2027 (circa 6% samengesteld vanaf 2023), Verizon op 2,5-3,0% groei in 2026 en AT&T op mid-single-digit groei. De onderliggende volumepool krimpt: een analistenraming schat dat immigratie tussen 2018 en 2024 goed was voor circa 44% van de 51 miljoen netto-aanwas en dat een halvering daarvan de jaarlijkse aanwas van 7,3 naar 5,8 miljoen brengt.
- **Porter five forces:**
  - **Rivaliteit — hoog.** Drie landelijke netwerken plus twee kabel-MVNO's vechten om een krimpende pool nieuwe klanten. De EBITDA-margeverschillen tussen de operators zijn sinds 2020 sterk ingelopen. In 2025-2026 escaleerde dat in prijsgaranties van drie tot vijf jaar en gratis lijnen; pas in het tweede kwartaal van 2026 lijkt de promotie-intensiteit iets af te nemen.
  - **Nieuwe toetreders — laag, maar niet nul.** Een landelijk netwerk bouwen kost honderd miljard en schaars spectrum. De reële toetreding komt langs een andere route: kabelbedrijven huren capaciteit in en SpaceX kocht voor circa USD 20 mrd spectrum, waarmee een satellietgedreven aanbieder zonder masten denkbaar wordt.
  - **Substituten — middel.** Voor mobiele telefonie bestaat geen substituut; voor breedband wel. Glasvezel en kabel zijn sneller en betrouwbaarder dan fixed wireless access, en de FWA-capaciteit van de drie operators is fysiek begrensd op circa 32 miljoen aansluitingen omdat het op restcapaciteit van het mobiele netwerk draait.
  - **Macht leveranciers — laag tot middel.** Netwerkapparatuur komt van Ericsson en Nokia, toestellen van Apple en Samsung. Apple heeft prijszettingsmacht op de toestelkant, maar die verkoopt T-Mobile toch tegen of onder kostprijs. De echte "leverancier" is de FCC, die het spectrum uitgeeft en de veilingregels bepaalt.
  - **Macht afnemers — hoog.** Consumenten kunnen zonder contract, met nummerbehoud en met een afgekochte toestelrekening binnen een dag overstappen. Circa 30 tot 40% van alle brutoactiviteit in de markt is porting tussen operators. De klantenboosheid en de FCC-klacht na de tariefmigratie van juli 2026 lieten zien hoe smal het pad is.
  - **Conclusie Porter:** gemiddeld tot licht onaantrekkelijk. De toetredingsdrempels zijn indrukwekkend en zorgen voor een stabiele oligopolie, maar de combinatie van hoge rivaliteit, machtige afnemers en een krimpende groeipool zorgt ervoor dat de structurele winstgevendheid onder druk staat in plaats van omhoog kruipt.

- **Concurrenten:**

| Concurrent | Postpaid-telefoonklanten 31-12-2025 (× 1.000) | Aandeel binnen de drie landelijke operators % |
|---|---|---|
| Verizon Communications | 93.868 | 37,0 |
| T-Mobile US | 85.594 | 33,7 |
| AT&T | 74.200 | 29,3 |
| Charter Spectrum Mobile (MVNO) | 12.500 (mobiele lijnen, 30-06-2026) | — |
| Comcast Xfinity Mobile (MVNO) | 10.187 (mobiele lijnen, 30-06-2026) | — |

*De percentages zijn mijn eigen rekensom op de laatst gepubliceerde vergelijkbare standen (totaal 253,7 mln); een officieel totaalmarktcijfer inclusief alle MVNO's publiceert niemand.*

**Vergelijkingstabel FY2025 (USD mrd tenzij anders vermeld)**

| | T-Mobile | Verizon | AT&T |
|---|---|---|---|
| Omzet | 88,3 | 138,2 | 125,6 |
| Operationeel resultaat | 18,3 | 29,3 | 24,2 |
| Nettowinst | 11,0 | 17,2 | 21,9 |
| Adj. EBITDA | 33,9 | ca. 50,0 | 46,4 |
| Operationele kasstroom | 28,0 | 37,1 | 40,3 |
| Capex | 10,0 | 17,0 | 20,8 |
| Vrije kasstroom (gerapporteerd) | 18,0 | 20,1 | 16,6 |
| Nettoschuld / EBITDA | 2,4x | 2,2x | 2,5x |
| Servicerevenue-groei 2025 | +8% | +2 tot 3% | +2 tot 3% |
| K/W (TTM, 19-08-2026) | 19,2 | 12,9 | 8,3 |
| Forward K/W | 14,6 | 9,7 | 10,5 |
| EV/EBITDA | 9,2 | 7,8 | 7,1 |
| Dividendrendement | 2,2% | 5,7% | 4,4% |
| ROIC (aggregator) | 8,9% | 8,3% | 8,9% |

- **Positie van het bedrijf:** Challenger die marktleider aan het worden is. T-Mobile groeide van circa 30% naar circa 34% aandeel binnen de drie landelijke operators en heeft als enige een servicerevenue-groei van 8% bij een EBITDA-marge die met de anderen kan concurreren. De prijs daarvoor is een waarderingspremie: 9,2 keer EBITDA en 14,6 keer de verwachte winst tegenover 7,1 tot 7,8 keer EBITDA en circa 10 keer de winst bij Verizon en AT&T. Die premie is te verdedigen zolang de groei drie tot vier keer zo hoog blijft, maar de kwetsbaarheid ligt in convergentie: AT&T verkoopt 38,6 miljoen glasvezellocaties en bundelt bij 42,5% van die huishoudens ook mobiel, terwijl T-Mobile met 997.000 glasvezelklanten pas begint.

### TAM/SAM/SOM
- **TAM:** — (geen geverifieerde onafhankelijke marktomvang gevonden)
- **SAM:** — (idem)
- **Huidige penetratie %:** 33,7 (aandeel in de gepubliceerde postpaid-telefoonbasis van de drie landelijke operators)
- **Groei plausibel?** true, met beperking
- **Bron TAM/SAM:** niet verifieerbaar — carriers publiceren sinds 2026 geen vergelijkbare abonneetotalen meer en er is geen onafhankelijke bron met een geverifieerd totaal
- **Toelichting:** Een klassieke TAM/SAM-analyse is hier niet zinvol te maken en ik vul liever niets in dan een gefantaseerd getal. Wat wél kwantificeerbaar is: de FWA-capaciteit van de drie operators samen is door New Street Research op circa 32 miljoen aansluitingen geraamd, waarvan T-Mobile met 8,45 miljoen 5G-breedbandklanten al ongeveer tweederde van zijn eigen deel benut. Het doel van 12 miljoen FWA-klanten in 2028 is daarmee ambitieus maar niet onmogelijk; het doel van 15 miljoen in 2030 vereist extra spectrum.

---

## 7. Analyse-frameworks (9 frameworks, SCORES 1-5)

### Graham
- **Oordeel:** GEDEELTELIJK
- **Graham number:** 106,22
- **Margin of safety %:** −41,8 (de koers ligt 71,7% boven het Graham-getal)
- **Toelichting:** Graham's defensieve criteria zijn deels vervuld en deels niet. De K/W van 19,08 blijft onder de drempel van 20 uit regel 3, maar de koers-boekwaarde van 3,48 ligt boven 3,0 en het Graham-getal van USD 106,22 laat geen enkele veiligheidsmarge. De schuldpositie is voor Graham te zwaar: bruto rentedragende schuld gedeeld door eigen vermogen bedraagt 1,61. Er is sinds 2023 wel een structureel dividend, en de winst is de afgelopen tien jaar per saldo fors gegroeid. **Regelbotsing:** regel 3 (K/W ≤ 20) en regel 1 (K/B > 3,0) zijn beide waar. Conform de cascadeafspraak telt de eerste treffer van boven, dus regel 3.
- **Score (1-5):** 3

### Buffett / Munger
- **Oordeel:** GEDEELTELIJK
- **ROIC structureel boven WACC?** false (over vijf jaar niet; over drie jaar wel)
- **Toelichting:** Het bedrijf is begrijpelijk, de kasstromen zijn voorspelbaar en de prijs is met een koers/vrije-kasstroom van 11,2 op gerapporteerde basis (17,6 genormaliseerd) niet extreem. Waar het strandt is de eis dat het rendement op geïnvesteerd kapitaal structureel boven de kapitaalkosten ligt: in 2021 (3,71%) en 2022 (3,51%) lag de ROIC ver ónder de WACC van 5,88%, en dat is binnen het vijfjaarsvenster dat de rubriek voorschrijft. Pas vanaf 2023 is de spread positief en vanaf 2024 substantieel. Regel 3 vereist "structureel"; die eis wordt niet gehaald, dus valt de score terug op regel 2: ROIC boven WACC, maar niet structureel.
- **Score (1-5):** 2

### Peter Lynch
- **Categorie:** Stalwart
- **Oordeel:** NEUTRAAL
- **PEG-ratio:** 1,16
- **Toelichting:** Lynch zou T-Mobile een stalwart noemen: een groot, bekend bedrijf met voorspelbare eencijferige omzetgroei en een duidelijk verhaal dat je in twee zinnen uitlegt — beter netwerk, lagere prijs, meer klanten. De PEG bereken ik als de K/W van 19,08 gedeeld door de gemiddelde consensus-winstgroei voor 2026 en 2027 van 16,4%, wat uitkomt op 1,16. Op de forward-K/W van 14,59 zou de PEG 0,89 zijn; ik hanteer de conservatieve variant, omdat de verwachte winstgroei deels uit aandeleninkoop komt en niet uit operationele groei. Bij een PEG onder 1,5 met een helder verhaal komt regel 3 als eerste treffer bovendrijven.
- **Score (1-5):** 3

### Phil Fisher
- **Oordeel:** GEMIDDELD
- **Toelichting:** T-Mobile publiceert geen R&D-budget; in deze sector is de investering in toekomstige verdiencapaciteit de combinatie van netwerkcapex en spectrum, en die bedraagt circa USD 12,5 mrd per jaar en groeit (capex van 8,8 naar 10,4 mrd over twee jaar). Van Fishers drie kwantificeerbare criteria is er één duidelijk vervuld: de marge wordt beschermd door de moat, wat blijkt uit een EBIT-marge die van 8,2% in 2022 naar ruim 20% klom terwijl de prijzen in de markt onder druk stonden. De integriteit van het management scoort NEUTRAAL, niet STERK, en het R&D-equivalent is niet met het sectorgemiddelde te vergelijken. Eén van de drie criteria plus een groeiend investeringsbudget geeft regel 3.
- **Score (1-5):** 3

### Magic Formula (Greenblatt)
- **Oordeel:** GEMIDDELD
- **Earnings yield %:** 6,45 (EBIT TTM 18.253 gedeeld door enterprise value 283.168)
- **Return on capital %:** 40,89 (EBIT TTM gedeeld door netto werkkapitaal 6.305 plus netto materiële vaste activa 38.333)
- **Toelichting:** Greenblatt kijkt naar twee dingen: koop je de winst goedkoop, en verdient het bedrijf veel op zijn tastbare kapitaal. Op de tweede as scoort T-Mobile uitstekend: 40,9% rendement op werkkapitaal plus netwerkactiva, omdat de spectrumvergunningen in deze maatstaf buiten beschouwing blijven. Op de eerste as blijft het steken: een earnings yield van 6,45% ligt onder de 7% die regel 4 verlangt, doordat de enterprise value van USD 283 mrd wordt opgeblazen door USD 88 mrd nettoschuld. Regel 3 vraagt earnings yield boven 5% óf rendement op kapitaal boven 50%; de eerste voorwaarde is vervuld.
- **Score (1-5):** 3

### Moat
- **Score (1-5):** 3 — NARROW MOAT met twee sterke categorieën (immateriële activa en kostenvoordeel) en een ROIC-WACC-spread van 5,54 procentpunt in 2025. **Regelbotsing:** de spread over vijf jaar bedraagt gemiddeld slechts 1,75 procentpunt doordat 2021 en 2022 door de Sprint-integratie werden gedrukt. Regel 3 stelt geen periode-eis, regel 5 wel; ik hanteer de meest recente spread en noteer de botsing.

### Management
- **Score (1-5):** 3 — capital allocation GEMENGD: de Sprint-integratie en de synergierealisatie zijn excellent uitgevoerd, maar circa USD 34 mrd aandeleninkoop in 2023-2025 vond grotendeels plaats boven de huidige koers, het insiderbelang is 0,32% en er zijn materiële controverses (FCC-boetes, datalekschikking, klantenprotest na de tariefmigratie van juli 2026).

### Fair Value DCF
- **Score (1-5):** 5 — het basisscenario geeft een fair value van USD 313,38 tegenover een koers van USD 182,36, een opwaarts potentieel van 71,8%, ruim boven de drempel van 30%. Zie de nadrukkelijke voorbehouden in de DCF-toelichting: deze uitkomst hangt sterk aan de lage sectorbeta.

### Fair Value IPO-gecorr.
- **Score (1-5):** 5 — de beursnotering dateert van 1 mei 2013 en ligt dus meer dan tien jaar terug; conform de rubriek is deze score gelijk aan de Fair Value DCF-score.

### Scorekaart totaal
- **Totaalscore:** 30
- **Max:** 45
- **Eindoordeel:** **HOLD** (30 ligt tussen 24 en 33; de Fair Value DCF-score is 5 en dus niet 1)
- **Samenvatting:** T-Mobile scoort 30 van de 45 punten en komt daarmee midden in de HOLD-band uit. Het patroon is opvallend consistent: op zeven van de negen assen een 2 of 3, en twee vijven die volledig uit dezelfde bron komen — de DCF. Die twee punten steunen op een WACC van 5,88% die volgt uit een vijfjaars regressiebeta van 0,33; verhoog de kapitaalkosten met twee procentpunt en het basisscenario zakt van USD 313 naar circa USD 175, waarmee beide vijven omslaan in tweeën en het totaal naar 24 valt. De kwalitatieve assen laten intussen een bedrijf zien dat operationeel presteert maar waarvan de slotgracht smaller wordt: AT&T haalde zijn spectrumachterstand in, kabel pakt bijna de helft van de marktgroei en de fiscale rugwind loopt in 2027 af. De voornaamste onzekerheid die een belegger in de gaten moet houden is niet de winst maar de vereiste rendementseis. De katalysatorkalender is op korte termijn eerder een bedreiging dan een steun: het derde kwartaal krijgt een verhoogd klantverloop uit de tariefmigratie. Gegeven de goede datakwaliteit maar de grote gevoeligheid van de waardering acht ik een veiligheidsmarge van 25% het minimum.

---

## 8. Risico's

| # | Omschrijving | Kans | Impact | DCF-aanname geraakt | Toelichting |
|---|---|---|---|---|---|
| 1 | Einde van de fiscale rugwind: de kasbelasting loopt van USD 0,45 mrd (2025) via circa USD 1,5 mrd (2026) naar circa USD 3,5 mrd (2027) | HOOG | GROOT | Basis-FCF | Dit is geen risico maar een zekerheid; het bedrijf begroot het zelf. Het effect is groot: de gerapporteerde vrije kasstroom van USD 18,0 mrd bevat circa USD 2,8 mrd aan uitgesteld belastingvoordeel. Wie de headline-FCF extrapoleert, waardeert een belastingvakantie tot in de eeuwigheid. Ik normaliseer hiervoor. |
| 2 | Erosie van de spectrumvoorsprong: AT&T sloot in juli 2026 een spectrumdeal van USD 23 mrd met EchoStar, Verizon won de AWS-3-veiling met USD 3,2 mrd | HOOG | GROOT | FCF-groei fase 1 en 2 | De netwerkvoorsprong is de bron van zowel de prijsstelling als de marge. AT&T had de 3,45 GHz bij closing al op circa 23.000 masten live met tot 80% hogere snelheden. Zonder meetbaar beter netwerk wordt T-Mobile een van de drie in plaats van de beste. |
| 3 | Kabelbedrijven pakken bijna de helft van alle nieuwe postpaid-klanten | HOOG | MIDDEL | FCF-groei fase 1 | Comcast en Charter hadden eind juni 2026 samen 22,7 miljoen mobiele lijnen en groeiden in het tweede kwartaal met 854.000 lijnen tegenover 277.000 accounts bij T-Mobile. Zij verkopen mobiel als toevoeging op breedband en delen gratis lijnen uit; die economie kan T-Mobile niet matchen zonder marge in te leveren. |
| 4 | Klantverloop en reputatieschade na de gedwongen tariefmigratie van juli 2026 | HOOG | MIDDEL | FCF-groei fase 1 | Circa 8 miljoen klanten werden gemigreerd, ongeveer 4 miljoen betalen tot USD 6 per lijn per maand meer. Er lopen een FCC-klacht en een klacht bij de Californische procureur-generaal, en advocaten bereiden massa-arbitrage voor. Het management stuurt zelf aan op circa 250.000 accounts in het derde kwartaal tegen 277.000 in het tweede. |
| 5 | Kapitaalconflict rond de upper C-band-veiling van juli 2027 | MIDDEN | GROOT | FCF-groei en nettoschuld | De FCC veilt 160 MHz met een geraamde opbrengst van USD 30-75 mrd. T-Mobile heeft een kapitaalenvelop van USD 80 mrd tot en met 2027 waarvan tot USD 30 mrd voor aandeelhouders is bestemd. Fors meebieden gaat ten koste van de inkoop; niet meebieden vergroot risico 2. |
| 6 | Governance: Deutsche Telekom controleert 54,5% van de stemmen en benoemt tien van de dertien bestuurders | MIDDEN | GROOT | Kapitaalkosten | In april 2026 verkende het moederbedrijf een volledige samensmelting die minderheidsaandeelhouders als onderwaardering beschouwden; de gesprekken strandden eind juli 2026 op verzet en CFIUS-zorgen. Er is geen formele intrekking. Zolang die dreiging boven de markt hangt, eist de markt een hogere rendementsvergoeding. |
| 7 | Toetreding van satellietaanbieders | MIDDEN | MIDDEL | Terminale groei | SpaceX kocht voor circa USD 20 mrd EchoStar-spectrum en overweegt volgens berichtgeving uit juni 2026 een eigen retail-mobielmerk. AST SpaceMobile bouwt met AT&T en Verizon. T-Mobile heeft met T-Satellite nu een voorsprong van circa twee jaar, maar het gebruik is met 0,0003% van het netwerkverkeer verwaarloosbaar en de exclusiviteit met SpaceX is eindig. |
| 8 | Rentegevoeligheid van een balans met USD 90 mrd rentedragende schuld | MIDDEN | MIDDEL | WACC en nettoschuld | De betaalde rente steeg van USD 3,49 mrd (2022) naar USD 3,88 mrd (2025) en het bedrijf begroot circa USD 5,0 mrd voor 2027. Bij een tienjaarsrente van 4,65% wordt aflopende goedkope schuld duurder geherfinancierd; elke procentpunt op de gemiddelde rente kost circa USD 0,9 mrd per jaar. |
| 9 | Pre-IPO financial engineering | LAAG | KLEIN | — | Niet geconstateerd. T-Mobile US kreeg zijn notering in 2013 via een omgekeerde fusie met het al beursgenoteerde MetroPCS, niet via een beursgang met opbrengst. Er is geen dividendrecapitalisatie vóór de notering geweest, geen schuld bij gelieerde partijen die met beursopbrengsten is afgelost, en geen "schoon balansmoment" dat het historische beeld vertekent. Wel bestond tot en met 2025 een post "langlopende schuld aan gelieerde partijen" van USD 1.498 mln aan Deutsche Telekom; die is per 30 juni 2026 volledig afgelost. |

---

## 9. These invalide bij

Deze these is weerlegd wanneer de netto-aanwas van postpaid-accounts twee kwartalen op rij onder de 200.000 zakt of het verloop op postpaid-telefoons structureel boven 1,05% per maand uitkomt — dan is het marktaandeelverhaal voorbij en verdient het aandeel geen premie boven Verizon en AT&T. Zij is eveneens weerlegd als T-Mobile de eigen doelstelling van USD 19,5-20,5 mrd Adjusted Free Cash Flow over 2027 verlaagt, of als het bedrijf in de upper C-band-veiling van juli 2027 meer dan USD 15 mrd uitgeeft en daarvoor het inkoopprogramma terugschroeft. Een hervatting van de fusiegesprekken met Deutsche Telekom tegen voorwaarden die de Amerikaanse kasstroom naar het moederbedrijf verplaatsen, maakt de aandeelhouderswaarde-redenering ongeldig ongeacht de operationele cijfers.

---

## 10. ESG

### Materiële factoren (SASB-gebaseerd)

| Factor | SASB-categorie | Risiconiveau | Financiële impact | DCF-impact |
|---|---|---|---|---|
| Datalekken en privacy van klantgegevens | Data Security / Customer Privacy | Hoog | Schikking van USD 500 mln na het datalek van 2021 (76,6 mln getroffen personen), FCC-consent decree van USD 31,5 mln in september 2024 en een FCC-boete van ruim USD 80 mln voor locatiedata in april 2024 | Verhoogt de kostenbasis structureel (verplichte investering in zero trust en multifactorauthenticatie) en drukt de terminale marge met enkele tienden van een procentpunt |
| Energieverbruik van het netwerk | Environmental Footprint of Operations | Midden | Een landelijk 5G-netwerk is energie-intensief; elektriciteitsprijzen zijn een directe kostenpost | Beperkt; deels gecompenseerd doordat 5G per gigabyte efficiënter is dan 4G |
| Concurrentiegedrag en prijstransparantie | Competitive Behavior | Hoog | De tariefmigratie van juli 2026 leidde tot een FCC-klacht, een klacht bij de Californische procureur-generaal en voorbereiding van massa-arbitrage | Raakt de aanname over ARPA-groei: prijsverhogingen die juridisch of reputationeel worden teruggedraaid, verdwijnen uit de kasstroom |
| Werknemersbetrokkenheid en diversiteitsbeleid | Employee Engagement, Diversity & Inclusion | Midden | T-Mobile zegde de FCC in juli 2025 toe al zijn diversiteitsprogramma's te beëindigen, één dag vóór de goedkeuring van de UScellular-overname | Reputationeel en mogelijk relevant voor personeelsbehoud; niet kwantificeerbaar in de DCF |
| Toegang tot spectrum en regulering | Managing Systemic Risks / Regulatory Capture | Hoog | Spectrum is een overheidsvergunning; de veilingbevoegdheid loopt tot 30-09-2034 met een pijplijn van minimaal 800 MHz | Rechtstreeks: de genormaliseerde spectrumuitgave van USD 3,2 mrd per jaar is een expliciete aftrekpost in mijn FCF |

- **Eindoordeel ESG:** **GEMIDDELD RISICO**
- **Toelichting:** De governancekant is het zwaarste punt: een controlerende aandeelhouder met tien van de dertien bestuurszetels, een fusiepoging in 2026 die door minderheidsaandeelhouders werd geblokkeerd, en een insiderbelang van 0,32%. Op de sociale as staat een reeks toezichtsmaatregelen rond datalekken en locatiegegevens die samen ruim USD 600 mln hebben gekost, plus de klantenprotesten van 2026. Milieu is voor een telecomoperator relatief ondergeschikt. Geen van deze factoren is op zichzelf existentieel, maar samen rechtvaardigen ze eerder een opslag op de rendementseis dan een korting.

---

## 11. Katalysatoren

| Datum ca. | Omschrijving | Richting | Impact |
|---|---|---|---|
| 28-08-2026 | Ex-dividenddatum kwartaaldividend USD 1,02, betaalbaar 10-09-2026 | POSITIEF | KLEIN |
| ca. 22-10-2026 | Kwartaalcijfers Q3 2026 — de eerste volledige meting van het klantverloop na de tariefmigratie; management stuurt op circa 250.000 netto accounts | BINAIR | GROOT |
| november 2026 | Declaratie kwartaaldividend Q4; in 2024 en 2025 ging de jaarlijkse verhoging bij deze declaratie in | POSITIEF | KLEIN |
| H2 2026 | Closing glasvezel-JV i3 Broadband met Wren House (circa USD 700 mln, circa 500.000 homes passed) | NEUTRAAL | KLEIN |
| 31-12-2026 | Afloop van de aandeelhoudersreturn-autorisatie van USD 18,2 mrd; van de USD 18,2 mrd was in de eerste helft van 2026 circa USD 10,3 mrd besteed | BINAIR | MIDDEL |
| januari/februari 2027 | Jaarcijfers 2026 plus naar verwachting de nieuwe aandeelhoudersreturn-autorisatie en een geactualiseerde meerjarenoutlook | BINAIR | GROOT |
| H1 2027 | Closing glasvezel-JV GoNetspeed/Greenlight met Oak Hill (circa USD 2,0 mrd); Team Telecom onderzoekt de transactie, wat vertraging kan geven | NEUTRAAL | KLEIN |
| uiterlijk 04-07-2027 | Veiling upper C-band, 160 MHz, geraamde opbrengst USD 30-75 mrd; de procedurele consultatie loopt sinds eind juli 2026 | BINAIR | GROOT |
| doorlopend | Formele intrekking óf hervatting van de fusiegesprekken met Deutsche Telekom; er is nog geen officiële verklaring van beide partijen | BINAIR | GROOT |
| medio/eind 2027 | Afronding van de UScellular-integratie; synergiedoel verhoogd naar circa USD 1,2 mrd per jaar bij circa USD 2,6 mrd integratiekosten | POSITIEF | MIDDEL |

---

## 12. Fair value — kwantitatief (DCF)

### DCF-invoeren

```
Basis            fcf=14043  shares=1072.67  net_cash=-87556  gross_debt=90381  revenue=92189
                 koers=182.36  ipo_jaar=2013
WACC             rf=4.65  erp=4.28  beta=0.55  crp=0.0  size_premium=0.0
                 cost_of_debt_pretax=4.60  tax_rate=25.0
Pessimistisch    g1=1.0  g2=0.5  gt=1.0  wacc_adj=1.75  kans=35
Basis            g1=4.0  g2=2.5  gt=2.0  wacc_adj=0.0  kans=45
Optimistisch     g1=6.5  g2=3.5  gt=2.25  wacc_adj=-0.25  kans=20
EPV              norm_ebit_margin=20.33  maintenance_capex=10000  da=14415
                 norm_ebitda_margin=35.44
Multiples        pe=19.08  pb=3.48  p_fcf=11.16  peg=1.16
Rendement        roic=11.59  earnings_yield=6.45  roc_greenblatt=40.89
Kwalitatief      moat_oordeel=NARROW  moat_categorieen_sterk=2  management_oordeel=NEUTRAAL
                 capital_allocation=GEMENGD  insider_alignment_pct=0.32
                 roic_wacc_spread_5j_plus=false  structureel_dividend=true  debt_equity=1.606
Eenheid          bedragen in USD mln; percentages als getal (3.05 = 3,05%)
```

*Toelichting bij de basis-FCF van 14.043:* dit is de genormaliseerde vrije kasstroom naar de onderneming (FCFF) over 2025, opgebouwd als operationele kasstroom 27.950 − capex 9.955 − aandelencompensatie 829 − belastingnormalisatie 2.838 + rente na belasting 2.912 − genormaliseerde spectrumuitgave 3.197. De belastingnormalisatie vervangt de werkelijk betaalde USD 451 mln door de belastinglast van USD 3.289 mln; de spectrumuitgave is het tienjaarsgemiddelde 2016-2025. Zonder de spectrumcorrectie zou de FCFF 17.240 bedragen. De rente wordt teruggeteld omdat de betaalde rente van USD 3.882 mln onder US GAAP in de operationele kasstroom zit — zonder die terugtelling zou ik een levered kasstroom tegen de WACC verdisconteren.

### WACC-componenten
- **Risicovrije rente %:** 4,65
- **Bron risicovrije rente:** US Treasury 10-jaars constant maturity par yield, 19-08-2026
- **Type:** spot (nominaal). De huidige 4,65% ligt meer dan 150 basispunten boven het tienjaarsgemiddelde; een genormaliseerde, lagere rente zou de WACC verder verlagen en de fair value verhógen. De spotrente is hier dus de conservatieve keuze. De gevoeligheidsmatrix dekt beide varianten.
- **ERP (equity risk premium) %:** 4,28
- **Bron ERP:** Damodaran implied ERP S&P 500 per 01-08-2026, trailing twaalf maanden met aangepaste payout
- **Beta (adjusted, Blume):** 0,55
- **Bron beta:** 5-jaars regressiebeta 0,33 (stockanalysis.com, data-datum 19-08-2026), Blume-aangepast als ⅔ × 0,33 + ⅓ × 1,00 = 0,553
- **Type beta:** 5-jaars regressie, Blume-aangepast. Ter controle: Damodaran's dataset van januari 2026 geeft voor Telecom (Wireless) een beta van 0,54 — vrijwel identiek.
- **Country risk premium %:** 0,00 (Verenigde Staten)
- **Size premium %:** 0,00 (marktkapitalisatie USD 195,6 mrd)
- **Cost of equity %:** 7,00
- **Schuldkosten vóór belasting %:** 4,60 (de in 2025 werkelijk betaalde effectieve rente bedroeg 4,26%; ik verhoog die richting het huidige renteniveau omdat aflopende goedkope schuld tegen marktrente wordt geherfinancierd)
- **Schuldkosten na belasting %:** 3,45
- **E/V gewicht %:** 68,40
- **D/V gewicht %:** 31,60
- **WACC %:** 5,88
- **Sector WACC % (referentie Damodaran, januari 2026):** 5,48 voor Telecom (Wireless), 5,39 voor Telecom Services — mijn WACC ligt er dus 40 basispunten bóven
- **Illiquiditeitskorting %:** — (niet van toepassing; gemiddeld dagvolume in de miljoenen aandelen)

### DCF model-specs
- **Model type:** 2-fase (jaren 1-5 en 6-10) plus terminale waarde
- **FCF-definitie:** Free cash flow to firm, verdisconteerd tegen de WACC; nettoschuld daarna afgetrokken
- **Basis FCF:** 14.043 (genormaliseerd op 2025; kasbelasting genormaliseerd, spectrum genormaliseerd)
- **Basis FCF na SBC:** 14.043 (de aandelencompensatie van 829 is al afgetrokken)
- **FCF-type:** genormaliseerd (stated GAAP-kasstroom met drie expliciete correcties)
- **Groei fase 1 % (jaar 1-5):** 4,0
- **Groei fase 2 % (jaar 6-10):** 2,5
- **Terminal groei %:** 2,0
- **Terminal methode:** Gordon growth, met exit-multiple als kruiscontrole
- **Exit multiple gebruikt (EV/EBITDA):** 7,5
- **Bron exit multiple:** mediaan van de directe peers per 19-08-2026 (Verizon 7,76x, AT&T 7,13x)
- **Terminal value Gordon growth:** 508.190
- **Terminal value exit multiple:** 328.185 (7,5 × geprojecteerde EBITDA 2035 van 43.758)
- **Terminal value % van totaal:** 69,7 (onder de grens van 75%)
- **Terminal implied EV/EBITDA:** 11,6
- **Terminal groei consistentie:** Bij een ROIC van 11,6% vereist 2,0% eeuwige groei een netto-herinvesteringsvoet van 17,3% van NOPAT. T-Mobile investeert brúto bijna zijn hele NOPAT in netwerk en spectrum, maar nétto — na aftrek van de afschrijvingen — ligt de herinvestering momenteel rond nul, omdat de D&A door de Sprint-koopprijsallocatie kunstmatig hoog is. Naarmate die afschrijving uitdooft loopt de netto-herinvestering op naar het niveau dat 2% ondersteunt. De aanname is haalbaar maar niet ruim. Ze ligt bovendien ruim onder de nominale langetermijngroei van de Amerikaanse economie van circa 4%.
- **Mid-year convention:** true
- **Aandelen uitstaand (mln):** 1.072,67
- **Nettoschuld huidig:** 87.556

### DCF-toelichting

De groeivoeten liggen bewust onder de eigen doelstellingen van het bedrijf. T-Mobile stuurt aan op USD 19,5-20,5 mrd Adjusted Free Cash Flow in 2027, wat op mijn genormaliseerde grondslag neerkomt op circa USD 22 mrd — mijn basisscenario komt in 2027 op USD 15,2 mrd uit. Het verschil zit in drie dingen: ik trek jaarlijks USD 3,2 mrd spectrumuitgave af die het bedrijf buiten zijn maatstaf houdt, ik behandel de aandelencompensatie als kosten, en ik laat de groei na 2027 afvlakken naar 2,5% en daarna 2,0%, omdat de netto-aanwaspool krimpt en kabel bijna de helft van de marktgroei pakt. De kasbelasting is genormaliseerd van USD 451 mln naar de boeklast van USD 3.289 mln, een correctie van USD 2,8 mrd op de vrije kasstroom, in lijn met de eigen guidance dat het bedrijf in 2027 volledig kasbelastingplichtig wordt.

Twee waarschuwingen horen bij deze uitkomst. Ten eerste: de WACC van 5,88% volgt uit een vijfjaars regressiebeta van 0,33. Die beta is in lijn met wat Damodaran voor de hele sector meet, maar hij is moeilijk te rijmen met een aandeel dat in zeventien maanden 34% verloor terwijl de brede markt steeg, met een standaarddeviatie van ruim 40% en met een schuld van 2,7 keer de EBITDA. Verhoog de kapitaalkosten met twee procentpunt en het basisscenario zakt van USD 313 naar circa USD 175. Ten tweede: de terminale waarde via de exit-multiple van 7,5 keer EBITDA valt 35% lager uit dan via Gordon growth en geeft een fair value van USD 215,97 in plaats van USD 313,38. Dat verschil is precies de vraag of de markt T-Mobile blijft waarderen als een telecombedrijf of als een groeibedrijf. Ik weeg beide mee in de synthese.

### 5-jaars projectie (USD mln)

| Jaar | Omzet | Omzetgroei % | EBIT | EBIT-marge % | NOPAT | Capex | Spectrum | ΔNWC | SBC | D&A | FCFF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026 | 94.370 | 6,9 | 19.440 | 20,6 | 14.580 | 10.000 | 3.200 | 250 | 900 | 14.400 | 14.630 |
| 2027 | 98.600 | 4,5 | 20.706 | 21,0 | 15.530 | 10.300 | 3.200 | 250 | 940 | 14.200 | 15.040 |
| 2028 | 102.051 | 3,5 | 21.737 | 21,3 | 16.303 | 10.500 | 3.200 | 250 | 980 | 14.000 | 15.373 |
| 2029 | 105.113 | 3,0 | 22.599 | 21,5 | 16.949 | 10.700 | 3.200 | 250 | 1.020 | 13.900 | 15.679 |
| 2030 | 108.266 | 3,0 | 23.494 | 21,7 | 17.621 | 10.900 | 3.200 | 250 | 1.060 | 13.800 | 16.011 |

*Omzet 2026 en 2027 zijn de analistenconsensus (USD 94,37 mrd en USD 98,60 mrd); 2028-2030 zijn mijn eigen extrapolatie. De FCFF-reeks uit deze bottom-up-opbouw ligt 0 tot 6% ónder het pad dat het basisscenario van de DCF gebruikt (14.605 → 17.086), wat binnen de toegestane afwijking van 25% blijft maar wel laat zien dat het basisscenario aan de bovenkant van de bottom-up-band ligt.*

### Scenarios

| Scenario | FCF-groei fase 1 % | FCF-groei fase 2 % | Terminal % | WACC % | Fair value | Upside % | Kans % |
|---|---|---|---|---|---|---|---|
| Pessimistisch | 1,0 | 0,5 | 1,0 | 7,63 | 121,99 | −33,1 | 35 |
| Basis | 4,0 | 2,5 | 2,0 | 5,88 | 313,38 | +71,8 | 45 |
| Optimistisch | 6,5 | 3,5 | 2,25 | 5,63 | 436,18 | +139,2 | 20 |

- **Kansgewogen fair value:** 270,95

*De kansverdeling is bewust asymmetrisch (35/45/20 in plaats van 25/50/25). Reden: het pessimistische scenario test niet alleen lagere groei maar ook een kapitaalkostenvoet van 7,63%, die dichter ligt bij wat een belegger voor een aandeel met deze volatiliteit en schuldgraad zou eisen. Dat scenario verdient daarom een hoger gewicht dan de standaardverdeling voorschrijft.*

### Reverse DCF
- **Impliciete groei %:** 0,74 — dit is de constante FCFF-groei die bij een WACC van 5,88% precies de huidige koers van USD 182,36 rechtvaardigt
- **Historische FCF-CAGR %:** 82,9 over 2021-2025 op basis van operationele kasstroom minus capex (USD 1.591 mln naar USD 17.995 mln). Dit cijfer is niet representatief: het weerspiegelt het uitdoven van de 5G-uitrolcapex en de belastingvakantie, niet de onderliggende verdiencapaciteit.
- **Consensus groei % (analisten):** winst per aandeel +14,3% voor 2026 en +18,6% voor 2027; het bedrijf zelf guidet een samengestelde groei van circa 10% in Adjusted Free Cash Flow over 2023-2027
- **Interpretatie:** De markt prijst bij een kapitaalkostenvoet van 5,88% een eeuwige groei van 0,74% in — minder dan de helft van de Amerikaanse inflatiedoelstelling en dus reële krimp. Dat is een streng oordeel over een bedrijf dat zijn servicerevenue in 2025 met 8% zag groeien. Er zijn twee lezingen. Ofwel de markt gelooft de groei niet en verwacht dat kabel, AT&T en satelliet de marktaandeelwinst beëindigen. Ofwel de markt hanteert simpelweg een hogere rendementseis dan 5,88%: bij een discontovoet van 7,5% impliceert de huidige koers een eeuwige groei van 2,24%, en dat is wél een redelijke verwachting. Ik neig naar de tweede lezing, en dat is precies waarom ik de DCF-uitkomst in de synthese niet volledig laat meewegen.

### EPV (Bruce Greenwald)
- **Genormaliseerde EBIT-marge %:** 20,33 (gemiddelde van 2023, 2024 en 2025; de jaren daarvóór zijn vertekend door Sprint-integratiekosten en fusieafschrijvingen)
- **Genormaliseerde NOPAT:** 14.055 (20,33% × TTM-omzet 92.189 × 0,75)
- **Maintenance capex:** 10.000, plus 3.197 genormaliseerde spectrumuitgave
- **D&A:** 14.415
- **Adjusted earnings power:** 15.273
- **EPV (ondernemingswaarde):** 259.712
- **EPV per aandeel:** 160,49
- **Groeipremie %:** 95,3 (het basisscenario van de DCF ligt 95% boven de EPV)
- *Gevoeligheid:* zou ik de onderhoudscapex gelijkstellen aan de D&A — de conservatieve variant die METHODE.md als alternatief noemt — dan valt de EPV terug naar USD 90,50 per aandeel. Zou ik de spectrumuitgave niet aftrekken, dan stijgt hij naar USD 211,16. De EPV is dus vooral een bandbreedte van USD 90 tot USD 211, met USD 160,49 als middenwaarde, en niet één hard getal.

### Andere methoden
- **DDM uitgevoerd?** false — het dividend bestaat pas sinds 2023 en er is geen bestendige uitkeringshistorie om op te modelleren
- **SOTP uitgevoerd?** false — T-Mobile is één operationeel segment; de glasvezel-joint-ventures zijn daarvoor nog te klein

### Multiple-waardering (sanity check)

| Methode | Invoer | Multiple | Fair value per aandeel |
|---|---|---|---|
| EV/EBITDA | Core Adjusted EBITDA 2026E van 37.300 (midden guidance) | 8,5x | 213,95 |
| Forward K/W | Consensus genormaliseerde WPA 2026 van 12,18 | 16,0x | 194,88 |
| Koers/Adjusted FCF | Adjusted FCF 2026E van 18.600 (midden guidance), oftewel 17,34 per aandeel | 12,0x | 208,08 |
| **Gemiddelde** | | | **205,64** |

*De gekozen multiples liggen boven die van Verizon (7,8x EBITDA, 9,7x forward winst) en AT&T (7,1x en 10,5x), wat gerechtvaardigd is door een servicerevenue-groei die drie tot vier keer zo hoog ligt, maar onder T-Mobile's eigen niveau van begin 2025 (circa 12,4x EBITDA).*

### Synthese fair value
- **Bandbreedte laag:** 121,99
- **Bandbreedte centraal:** 232,06
- **Bandbreedte hoog:** 436,18
- **Methode-gewichten:**
  - DCF %: 35
  - EPV %: 25
  - Multiples %: 40
- **Margin of safety vereist %:** 25
- **Koopniveau:** 174,05
- **Synthese-toelichting:** De drie methoden geven een opvallend uiteenlopend beeld: de DCF zegt USD 313, de EPV zegt USD 160 en de peer-multiples zeggen USD 206. Dat verschil is geen rekenfout maar een echte inhoudelijke onenigheid over de vereiste rendementseis. Ik geef de multiples het hoogste gewicht (40%), omdat zij de enige methode zijn die de daadwerkelijke prijsstelling van vergelijkbare bedrijven bevat en dus impliciet de rendementseis die de markt hanteert. De DCF krijgt 35%: hij is methodisch het meest volledig, maar zijn uitkomst hangt vrijwel volledig aan één invoer, de sectorbeta van 0,55. De EPV krijgt 25% als bodemcontrole zonder groei. Dat geeft een centrale waarde van USD 232. De veiligheidsmarge zet ik op 25% — hoger dan de 20% die de datakwaliteit rechtvaardigt (alle cijfers komen rechtstreeks uit SEC-filings), maar noodzakelijk vanwege de spreiding tussen de methoden en de openstaande governancekwestie rond Deutsche Telekom. Het koopniveau komt daarmee op USD 174,05, ruim vier procent onder de huidige koers.

### Gevoeligheid (DCF)

WACC-bereik: 5,0% · 5,5% · 6,0% · 6,5% · 7,0% · 8,0%
Groeibereik fase 1: 1,0% · 2,5% · 4,0% · 5,5% · 7,0% (fase 2 telkens 1,5 punt lager, terminal gelijk aan fase 1 met een maximum van 2,0%)

| Groei fase 1 \ WACC | 5,0% | 5,5% | 6,0% | 6,5% | 7,0% | 8,0% |
|---|---|---|---|---|---|---|
| 1,0% | 250,77 | 214,72 | 185,87 | 162,27 | 142,59 | 111,65 |
| 2,5% | 366,78 | 304,06 | 257,00 | 220,39 | 191,09 | 147,12 |
| 4,0% | 428,52 | 356,04 | 301,70 | 259,44 | 225,64 | 174,96 |
| 5,5% | 498,41 | 414,83 | 352,19 | 303,50 | 264,58 | 206,28 |
| 7,0% | 577,43 | 481,23 | 409,15 | 353,16 | 308,43 | 241,48 |

*Lees deze tabel als de kern van de analyse. De huidige koers van USD 182,36 wordt gerechtvaardigd door bijvoorbeeld 1,0% groei bij 6,3% WACC, 2,5% groei bij 7,2% WACC of 4,0% groei bij 8,0% WACC. Alle drie zijn verdedigbare combinaties. Het aandeel is niet duidelijk goedkoop of duidelijk duur; het prijst een defensief scenario in dat je kunt geloven of niet.*

---

## 13. Databronnen

### Bronnen-hiërarchie
- **Jaarverslag PDF / IR-pagina / SEC-filing** → betrouwbaarheid **HOOG**
- **Beursmelding / persbericht met cijfers** → betrouwbaarheid **HOOG**
- **Aggregator** (StockAnalysis / MacroTrends / Yahoo) → betrouwbaarheid **AGGREGATOR**

### Financiële bronnen (10 jaar historie)

| Jaar | Bron | URL | Betrouwbaarheid |
|---|---|---|---|
| 2016 | T-Mobile US Form 10-K FY2016/FY2017 via SEC XBRL companyconcept | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/OperatingIncomeLoss.json | HOOG |
| 2017 | T-Mobile US Form 10-K FY2017 + herziene kasstroom uit FY2018-10-K | https://www.sec.gov/Archives/edgar/data/1283699/000128369918000011/R6.htm | HOOG |
| 2018 | T-Mobile US Form 10-K FY2018 (accession 0001283699-19-000015) | https://www.sec.gov/Archives/edgar/data/1283699/000128369919000015/R6.htm | HOOG |
| 2019 | T-Mobile US Form 10-K FY2019 via SEC XBRL companyconcept | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/NetCashProvidedByUsedInOperatingActivities.json | HOOG |
| 2020 | T-Mobile US Form 10-K FY2020 via SEC XBRL companyconcept | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/RevenueFromContractWithCustomerExcludingAssessedTax.json | HOOG |
| 2021 | T-Mobile US Form 10-K FY2021 (accession 0001283699-22-000018) | https://www.sec.gov/Archives/edgar/data/1283699/000128369922000018/0001283699-22-000018-index.htm | HOOG |
| 2022 | T-Mobile US Form 10-K FY2022 (accession 0001283699-23-000016) | https://www.sec.gov/Archives/edgar/data/1283699/000128369923000016/0001283699-23-000016-index.htm | HOOG |
| 2023 | T-Mobile US Form 10-K FY2023 (accession 0001283699-24-000008) + vergelijkende kolom in de FY2025-10-K | https://www.sec.gov/Archives/edgar/data/1283699/000128369926000010/R5.htm | HOOG |
| 2024 | T-Mobile US Form 10-K FY2024 (accession 0001283699-25-000012) | https://www.sec.gov/Archives/edgar/data/1283699/000128369925000012/tmus-20241231.htm | HOOG |
| 2025 | T-Mobile US Form 10-K FY2025 (accession 0001283699-26-000010) | https://www.sec.gov/Archives/edgar/data/1283699/000128369926000010/tmus-20251231.htm | HOOG |
| TTM | T-Mobile US Form 10-Q Q2 2026 | https://s29.q4cdn.com/310188824/files/doc_financials/2026/q2/Q2-2026-FORM-10-Q-vFinal.pdf | HOOG |

**Alle tien de jaren zijn HOOG.** Voor een SEC-filer is de XBRL-companyconcept-interface de jaarrekening zelf, niet een aggregator: elke waarde is voorzien van het formulier (10-K), het boekjaar en de periode waarin hij is gerapporteerd. Elke reeks is bovendien verankerd aan een extern gecontroleerd punt: nettowinst gedeeld door verwaterde aandelen komt op elke plek uit op de gerapporteerde winst per aandeel, en omzet en nettowinst over 2025 komen exact overeen met het Q4-2025-persbericht.

### Jaarverslagen en filings geraadpleegd

| Jaar | Bron | URL |
|---|---|---|
| 2025 | Form 10-K FY2025 — balans (R3), W&V (R5), kasstroom (R7), belastingnoot (R100, R102, R103), supplementeel (R122) | https://www.sec.gov/Archives/edgar/data/1283699/000128369926000010/R3.htm |
| 2025 | DEF 14A proxy statement (ingediend 27-04-2026) — beloning, governance, aandeelhouders | https://www.sec.gov/Archives/edgar/data/1283699/000119312526181884/d11985ddef14a.htm |
| 2026 | Form 10-Q Q2 2026 — balans en kasstroom per 30-06-2026 | https://s29.q4cdn.com/310188824/files/doc_financials/2026/q2/Q2-2026-FORM-10-Q-vFinal.pdf |
| 2018 | Form 10-K FY2018 — herziene kasstroom na ASU 2016-15 | https://www.sec.gov/Archives/edgar/data/1283699/000128369919000015/R6.htm |
| 2025 | 8-K CEO-wisseling Gopalan/Sievert (22-09-2025) | https://www.sec.gov/Archives/edgar/data/1283699/000119312525209990/d916281d8k.htm |

### Beursmeldingen en persberichten geraadpleegd

| Datum | Omschrijving | URL |
|---|---|---|
| 2026-07-23 | Q2 2026 resultaten en verhoogde vrije-kasstroomguidance | https://s29.q4cdn.com/310188824/files/doc_financials/2026/q2/Q2-2026-Earnings-Release-vFinal.pdf |
| 2026-02-11 | Q4 en jaarcijfers 2025 | https://s29.q4cdn.com/310188824/files/doc_financials/2025/q4/Q4-2025-Earnings-Release.pdf |
| 2026-02-11 | Capital Markets Day-update met meerjarenoutlook 2026-2027 | https://www.t-mobile.com/news/business/t-mobile-capital-markets-day-update-feb-2026 |
| 2026-02-11 | Investor Factbook Q4 2025 (klant-KPI's) | https://s29.q4cdn.com/310188824/files/doc_financials/2025/q4/Q4-2025-Investor-Factbook.pdf |
| 2025-08-01 | Afronding overname draadloze activiteiten UScellular | https://www.t-mobile.com/news/business/t-mobile-closes-uscellular-acquisition |
| 2026-04-28 | Twee nieuwe glasvezel-joint-ventures (GoNetspeed/Greenlight en i3 Broadband) | https://www.t-mobile.com/news/business/t-mobile-add-two-strategic-fiber-joint-ventures-gonetspeed-greenlight-i3 |
| 2013-05-01 | Afronding combinatie T-Mobile USA en MetroPCS; start notering als TMUS | https://www.t-mobile.com/news/press/t-mobile-and-metropcs-combination-complete-wireless-revolution |
| doorlopend | Dividendhistorie en kwartaaluitkeringen | https://investor.t-mobile.com/stock-info/dividend-history/default.aspx |

### IPO-prospectus
- **Geraadpleegd?** false
- **URL:** — (niet van toepassing)
- **Pre-IPO data beschikbaar?** true
- **Pre-IPO bron:** T-Mobile US kreeg zijn notering op 1 mei 2013 via een omgekeerde fusie met het al beursgenoteerde MetroPCS, niet via een beursgang. Er is dus geen IPO-prospectus met een uitgifteprijs. De financiële historie vanaf 2008 is wel volledig beschikbaar in de SEC-XBRL-reeksen, omdat de registrant (CIK 0001283699) al sinds de MetroPCS-tijd rapporteert.

### Non-GAAP
- **Gebruikt?** true
- **Toelichting:** T-Mobile rapporteert Adjusted EBITDA, Core Adjusted EBITDA en Adjusted Free Cash Flow. Ik gebruik GAAP als grondslag voor alle tabellen en voor de DCF. De non-GAAP-cijfers gebruik ik alleen waar het bedrijf zelf uitsluitend daarin guidet (de doelstellingen voor 2026 en 2027) en in de multiple-waardering, en dan expliciet als zodanig benoemd. Twee aanpassingen draai ik terug: de aandelencompensatie van USD 829 mln (2025) tel ik als kosten mee in plaats van terug, en de spectrumuitgaven die buiten de Adjusted Free Cash Flow blijven trek ik genormaliseerd af met USD 3,20 mrd per jaar.

### Ontbrekende data (eerlijke lijst)
- **Nettoschuld per 31-12-2018:** de XBRL-tag `LongTermDebt` bevat voor die datum geen eenduidige waarde en de balanspagina van de FY2019-10-K gaf een 404. Cel blijft leeg.
- **Torenverplichtingen vóór 2024:** niet als aparte reeks opgehaald; de nettoschuld voor 2016-2023 sluit die post daarom uit en is in zoverre licht onderschat (de post bedroeg in 2024 en 2025 USD 3,7 respectievelijk USD 3,5 mrd).
- **Brutowinst en brutomarge vóór 2023:** de uitsplitsing van kostprijs diensten en kostprijs apparatuur is alleen voor 2023-2025 opgehaald.
- **Klant-KPI's vóór 2024:** T-Mobile publiceert in het Investor Factbook geen doorlopende vijfjaarsreeks, en sinds 2026 rapporteren de operators geen vergelijkbare postpaid-telefoonstanden meer.
- **Totale omvang van de Amerikaanse postpaid-markt inclusief alle MVNO's:** door geen enkele bron gepubliceerd; de marktaandelen in dit rapport zijn daarom uitdrukkelijk aandelen binnen de drie landelijke operators.
- **CEO pay ratio FY2025:** staat op pagina 75 van de DEF 14A, die technisch niet volledig uitleesbaar was; geen secundaire bron gevonden.
- **Officieel free-float-percentage:** niet gepubliceerd. De SEC-berekende public float bedroeg USD 112,4 mrd per 30-06-2025; het percentage in de metadata is afgeleid van het belang van Deutsche Telekom.
- **Vervaldata van de resterende federale verliesverrekening (circa USD 1,8 mrd):** staan in de narratieve tekst van de belastingnoot, die niet uit de R-pagina's te extraheren viel.
- **Personeelsaantal:** circa 75.000 komt van een aggregator, niet uit de 10-K.
- **TAM/SAM:** geen geverifieerde onafhankelijke marktomvang gevonden; velden bewust leeg gelaten.

### Peildatum analyse
- 2026-08-19 (slotkoers USD 182,36)

---

## 14. Volledige bronnen-lijst

| Titel | URL | Type |
|---|---|---|
| T-Mobile US Form 10-K FY2025 | https://www.sec.gov/Archives/edgar/data/1283699/000128369926000010/tmus-20251231.htm | Jaarverslag |
| Form 10-K FY2025 — geconsolideerde balans (R3) | https://www.sec.gov/Archives/edgar/data/1283699/000128369926000010/R3.htm | Jaarverslag |
| Form 10-K FY2025 — resultaten- en totaalresultaatrekening (R5) | https://www.sec.gov/Archives/edgar/data/1283699/000128369926000010/R5.htm | Jaarverslag |
| Form 10-K FY2025 — kasstroomoverzicht (R7) | https://www.sec.gov/Archives/edgar/data/1283699/000128369926000010/R7.htm | Jaarverslag |
| Form 10-K FY2025 — belastingnoot componenten (R100) | https://www.sec.gov/Archives/edgar/data/1283699/000128369926000010/R100.htm | Jaarverslag |
| Form 10-K FY2025 — uitgestelde belastingposities (R102) | https://www.sec.gov/Archives/edgar/data/1283699/000128369926000010/R102.htm | Jaarverslag |
| Form 10-K FY2025 — belastingnoot narratief (R103) | https://www.sec.gov/Archives/edgar/data/1283699/000128369926000010/R103.htm | Jaarverslag |
| Form 10-K FY2025 — supplementele kasstroominformatie (R122) | https://www.sec.gov/Archives/edgar/data/1283699/000128369926000010/R122.htm | Jaarverslag |
| Form 10-K FY2025 — coverpagina (R1) | https://www.sec.gov/Archives/edgar/data/1283699/000128369926000010/R1.htm | Jaarverslag |
| Form 10-K FY2018 — herzien kasstroomoverzicht (R6) | https://www.sec.gov/Archives/edgar/data/1283699/000128369919000015/R6.htm | Jaarverslag |
| Form 10-K FY2017 — kasstroomoverzicht (R6) | https://www.sec.gov/Archives/edgar/data/1283699/000128369918000011/R6.htm | Jaarverslag |
| Form 10-K FY2024 | https://www.sec.gov/Archives/edgar/data/1283699/000128369925000012/tmus-20241231.htm | Jaarverslag |
| SEC XBRL companyconcept — Revenues | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/Revenues.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — RevenueFromContractWithCustomerExcludingAssessedTax | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/RevenueFromContractWithCustomerExcludingAssessedTax.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — OperatingIncomeLoss | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/OperatingIncomeLoss.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — NetIncomeLoss | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/NetIncomeLoss.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — NetCashProvidedByUsedInOperatingActivities | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/NetCashProvidedByUsedInOperatingActivities.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — PaymentsToAcquirePropertyPlantAndEquipment | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/PaymentsToAcquirePropertyPlantAndEquipment.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — PaymentsToAcquireIntangibleAssets | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/PaymentsToAcquireIntangibleAssets.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — ShareBasedCompensation | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/ShareBasedCompensation.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — DepreciationDepletionAndAmortization | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/DepreciationDepletionAndAmortization.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — Assets | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/Assets.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — StockholdersEquity | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/StockholdersEquity.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — CashAndCashEquivalentsAtCarryingValue | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/CashAndCashEquivalentsAtCarryingValue.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — Goodwill | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/Goodwill.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — LongTermDebt | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/LongTermDebt.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — OperatingLeaseLiability | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/OperatingLeaseLiability.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — FinanceLeaseLiability | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/FinanceLeaseLiability.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — InterestPaidNet | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/InterestPaidNet.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — IncomeTaxesPaidNet | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/IncomeTaxesPaidNet.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — IncomeTaxExpenseBenefit | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/IncomeTaxExpenseBenefit.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — WeightedAverageNumberOfDilutedSharesOutstanding | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/WeightedAverageNumberOfDilutedSharesOutstanding.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — EarningsPerShareDiluted | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/EarningsPerShareDiluted.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — PaymentsOfDividendsCommonStock | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/PaymentsOfDividendsCommonStock.json | Jaarverslag (XBRL) |
| SEC XBRL companyconcept — PaymentsForRepurchaseOfCommonStock | https://data.sec.gov/api/xbrl/companyconcept/CIK0001283699/us-gaap/PaymentsForRepurchaseOfCommonStock.json | Jaarverslag (XBRL) |
| T-Mobile Form 10-Q Q2 2026 | https://s29.q4cdn.com/310188824/files/doc_financials/2026/q2/Q2-2026-FORM-10-Q-vFinal.pdf | Beursmelding |
| T-Mobile Q2 2026 Earnings Release | https://s29.q4cdn.com/310188824/files/doc_financials/2026/q2/Q2-2026-Earnings-Release-vFinal.pdf | Beursmelding |
| T-Mobile Q4 2025 Earnings Release | https://s29.q4cdn.com/310188824/files/doc_financials/2025/q4/Q4-2025-Earnings-Release.pdf | Beursmelding |
| T-Mobile Q4 2025 Investor Factbook | https://s29.q4cdn.com/310188824/files/doc_financials/2025/q4/Q4-2025-Investor-Factbook.pdf | Beursmelding |
| T-Mobile Q2 2026 Investor Factbook | https://s29.q4cdn.com/310188824/files/doc_financials/2026/q2/Q2-2026-Investor-Factbook-vFinal.pdf | Beursmelding |
| Capital Markets Day-update februari 2026 | https://www.t-mobile.com/news/business/t-mobile-capital-markets-day-update-feb-2026 | Beursmelding |
| Persbericht Q2 2026 resultaten | https://www.t-mobile.com/news/business/t-mobile-q2-2026-earnings | Beursmelding |
| DEF 14A proxy statement 2026 (boekjaar 2025) | https://www.sec.gov/Archives/edgar/data/1283699/000119312526181884/d11985ddef14a.htm | Beursmelding |
| DEF 14A — Pay versus Performance (R2) | https://www.sec.gov/Archives/edgar/data/1283699/000119312526181884/R2.htm | Beursmelding |
| 8-K CEO-wisseling Gopalan/Sievert | https://www.sec.gov/Archives/edgar/data/1283699/000119312525209990/d916281d8k.htm | Beursmelding |
| Form 4 — open-markt aankoop Srini Gopalan, 06-11-2025 | https://www.sec.gov/Archives/edgar/data/1283699/000128369925000162/wk-form4_1762295864.xml | Beursmelding |
| Form 4 — open-markt aankoop André Almeida, 01-05-2026 | https://www.sec.gov/Archives/edgar/data/1283699/000128369926000069/wk-form4_1777930593.xml | Beursmelding |
| T-Mobile dividendhistorie (IR) | https://investor.t-mobile.com/stock-info/dividend-history/default.aspx | Beursmelding |
| T-Mobile executive leadership team | https://www.t-mobile.com/our-story/executive-leadership-team | Bedrijfswebsite |
| Persbericht CEO-benoeming Srini Gopalan | https://www.t-mobile.com/news/business/srini-gopalan-new-ceo | Beursmelding |
| Persbericht benoeming Peter Osvaldik tot CFO | https://www.t-mobile.com/news/un-carrier/un-carrier-names-osvaldik-chief-financial-officer | Beursmelding |
| Afronding combinatie T-Mobile USA en MetroPCS (2013) | https://www.t-mobile.com/news/press/t-mobile-and-metropcs-combination-complete-wireless-revolution | Beursmelding |
| Afronding overname UScellular | https://www.t-mobile.com/news/business/t-mobile-closes-uscellular-acquisition | Beursmelding |
| Glasvezel-JV met EQT (Lumos) — closing | https://www.t-mobile.com/news/business/t-mobile-eqt-close-lumos-fiber-jv | Beursmelding |
| Glasvezel-JV met KKR (Metronet) | https://www.t-mobile.com/news/network/t-mobile-kkr-joint-venture-to-acquire-metronet | Beursmelding |
| Twee nieuwe glasvezel-JV's (GoNetspeed/Greenlight, i3 Broadband) | https://www.t-mobile.com/news/business/t-mobile-add-two-strategic-fiber-joint-ventures-gonetspeed-greenlight-i3 | Beursmelding |
| MVNO-overeenkomst met Charter en Comcast voor zakelijke klanten | https://www.t-mobile.com/news/business/charter-and-comcast-announce-agreement-to-leverage-t-mobile-5g-for-wireless-business-customers | Beursmelding |
| T-Satellite dienstbeschrijving en tarieven | https://www.t-mobile.com/coverage/satellite-phone-service | Bedrijfswebsite |
| Deutsche Telekom — belang in T-Mobile US (feb 2026) | https://www.telekom.com/en/media/media-information/archive/t-mobile-us-1102078 | Beursmelding |
| US Treasury — daily par yield curve augustus 2026 | https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value_month=202608 | Onderzoeksrapport |
| Damodaran — implied equity risk premium (01-08-2026) | https://pages.stern.nyu.edu/~adamodar/New_Home_Page/home.htm | Onderzoeksrapport |
| Damodaran — cost of capital by industry (januari 2026) | https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/wacc.html | Onderzoeksrapport |
| StockAnalysis — TMUS koers en historie | https://stockanalysis.com/stocks/tmus/history/ | Aggregator |
| StockAnalysis — TMUS statistieken (beta, multiples) | https://stockanalysis.com/stocks/tmus/statistics/ | Aggregator |
| StockAnalysis — TMUS analistenprognoses | https://stockanalysis.com/stocks/tmus/forecast/ | Aggregator |
| StockAnalysis — TMUS dividendhistorie | https://stockanalysis.com/stocks/tmus/dividend/ | Aggregator |
| StockAnalysis — TMUS bedrijfsprofiel (personeel) | https://stockanalysis.com/stocks/tmus/company/ | Aggregator |
| StockAnalysis — Verizon statistieken | https://stockanalysis.com/stocks/vz/statistics/ | Aggregator |
| StockAnalysis — AT&T statistieken | https://stockanalysis.com/stocks/t/statistics/ | Aggregator |
| Verizon — financiële bijlage Q4 2025 | https://www.verizon.com/about/sites/default/files/2026-01/vz_4q25_fs_013026.pdf | Jaarverslag |
| AT&T — resultaten vierde kwartaal en jaar 2025 | https://about.att.com/story/2026/4q-earnings-2025.html | Beursmelding |
| AT&T — resultaten tweede kwartaal 2026 | https://about.att.com/story/2026/2q-earnings.html | Beursmelding |
| Comcast — resultaten tweede kwartaal 2026 | https://www.cmcsa.com/static-files/9ca0b7e1-e289-495a-8709-f6af42e173b2 | Beursmelding |
| Charter — resultaten tweede kwartaal 2026 | https://ir.charter.com/static-files/b7a4b9f5-df95-4438-90c3-8cbd173def5c | Beursmelding |
| FCC — resultaten AWS-3-veiling (Auction 113), juni 2026 | https://docs.fcc.gov/public/attachments/DA-26-633A1.pdf | Onderzoeksrapport |
| FCC — regels upper C-band-veiling, 22-07-2026 | https://docs.fcc.gov/public/attachments/DOC-423286A1.pdf | Onderzoeksrapport |
| FCC — boete van USD 80 mln voor locatiedata (april 2024) | https://www.fcc.gov/document/fcc-fines-t-mobile-80m-location-data-violations | Onderzoeksrapport |
| FCC — consent decree datalekken (september 2024) | https://www.fcc.gov/document/t-mobile-required-change-business-practices-after-data-breaches-0 | Onderzoeksrapport |
| Wiley — herstel FCC-veilingbevoegdheid onder de One Big Beautiful Bill Act | https://www.wiley.law/alert-One-Big-Beautiful-Bill-Act-Passes-Restoring-FCC-Auction-Authority-and-Establishing-Spectrum-Pipeline | Onderzoeksrapport |
| Light Reading — the great convergence: state of US wireless competition | https://www.lightreading.com/wireless/the-great-convergence-the-state-of-u-s-wireless-competition | Nieuwsartikel |
| Fierce Network — klantenprotest na de 5G-tariefmigratie (14-07-2026) | https://www.fierce-network.com/wireless/t-mobiles-5g-plan-refresh-triggers-customer-backlash | Nieuwsartikel |
| Fierce Network — FWA-capaciteit van de drie grote operators | https://www.fierce-network.com/broadband/big-3-now-have-room-32m-fwa-customers | Nieuwsartikel |
| Fierce Network — Verizon grootste winnaar AWS-3-veiling | https://www.fierce-network.com/wireless/verizon-emerges-biggest-winner-aws-3-auction | Nieuwsartikel |
| TechTimes — AT&T rondt EchoStar-spectrumdeal af (30-07-2026) | https://www.techtimes.com/articles/322311/20260730/t-closes-echostar-spectrum-deal-345-ghz-live-nationwide-600-mhz-not-ready.htm | Nieuwsartikel |
| Semafor — fusiegesprekken Deutsche Telekom stranden (31-07-2026) | https://www.semafor.com/article/07/31/2026/t-mobile-deutsche-telekom-merger-stalls | Nieuwsartikel |
| Investing.com — transcript Q2 2026 earnings call | https://www.investing.com/news/transcripts/earnings-call-transcript-tmobile-posts-q2-2026-eps-beat-shares-fall-premarket-93CH-4808922 | Nieuwsartikel |
| Investing.com — Wolfe Research verlaagt TMUS (14-08-2026) | https://www.investing.com/news/stock-market-news/wolfe-research-cuts-tmobile-as-revenue-growth-forecast-risk-tilts-negative-4860580 | Nieuwsartikel |
| TBR — immigratiebeleid en de groeischok voor de Amerikaanse draadloze markt | https://tbri.com/special-reports/immigration-policy-changes-portend-a-growth-shock-for-the-u-s-wireless-industry/ | Onderzoeksrapport |
| Recon Analytics — wat de operators sinds Q2 2026 niet meer publiceren | https://www.reconanalytics.com/what-the-carriers-stopped-telling-you-about-q2-2026/ | Nieuwsartikel |

---

## 15. Update-historie

| Datum | Versie | Wijziging |
|---|---|---|
| 2026-08-19 | 1.0 | Eerste publicatie. Peildatum 2026-08-19, koers USD 182,36. Oordeel HOLD, scorekaart 30/45, fair value basis USD 313,38, synthese USD 232,06. |

---

## Opmerkingen voor Claude Code

1. **De DCF-uitkomst hangt aan één invoer.** Met een Blume-aangepaste beta van 0,55 komt de WACC op 5,88% en het basisscenario op USD 313,38. Dat is 40 basispunten boven Damodaran's eigen sector-WACC voor Telecom (Wireless) van 5,48%, dus rubrieksgetrouw. Maar bij 7,88% WACC zakt het basisscenario naar circa USD 175 en kantelt het eindoordeel van HOLD (30) naar PASS (24). Als `dcf_calculator.py` een afwijking van meer dan 1% geeft, is dat waarschijnlijk een verschil in de behandeling van de mid-year convention of de nettoschuld — niet in de invoeren.
2. **Nettoschuld sluit operationele leases bewust uit.** USD 29,1 mrd operationele leaseverplichting zit onder US GAAP volledig in de bedrijfslasten en dus in EBIT en CFO. Financiële leases (USD 2,3 mrd) en torenverplichtingen (USD 3,5 mrd) zitten er wél in, omdat die aflossingen buiten de operationele kasstroom om lopen.
3. **Twee rubriekbotsingen genoteerd**, beide opgelost met de cascaderegel: Graham (regel 3 K/W ≤ 20 vóór regel 1 K/B > 3,0) en Moat (regel 3 spread > 5pp op basis van het meest recente jaar; over vijf jaar gemiddeld is de spread 1,75pp).
4. **Bronnen die goed werkten:** `data.sec.gov/api/xbrl/companyconcept/...` voor alle reeksen, en de R-pagina's van de filings via `FilingSummary.xml`. Let op: de samenvatting van een companyconcept-fetch kapt regelmatig af bij oudere jaren — vraag expliciet naar de láátste entries, anders mis je 2018-2025. Bij `Revenues` stopt de reeks überhaupt in 2017; vanaf 2018 heet de tag `RevenueFromContractWithCustomerExcludingAssessedTax`.
5. **Bronnen die niet werkten:** `sec.gov/cgi-bin/browse-edgar` is deels robots-geblokkeerd; de hoofddocumenten van de 10-K's (4-5 MB) worden bij ophalen afgekapt na Item 1A, dus gebruik altijd de R-pagina's; `finbox.com` levert alleen meta-tags.
6. **Haallijst:** geen. Alles was direct bereikbaar.

---

## Afronding (check voor je oplevert)

- [x] Elk cijfer in de tabellen heeft een bron in de bronnen-inventaris of de bronnentabel
- [x] Alle tien de jaren in sectie 13 zijn HOOG (SEC-filings, geen aggregator)
- [x] Geen enum-variant verzonnen — alleen waarden uit de template
- [x] Scorekaart heeft 9 frameworks, totaal 30 en max 45 kloppen
- [x] Synthese-toelichting aanwezig (sectie 12)
- [x] Non-GAAP-aanpassingen expliciet toegelicht (sectie 3 en 13)
- [x] IPO-carve-out niet van toepassing: notering sinds 1 mei 2013, meer dan tien jaar geleden
