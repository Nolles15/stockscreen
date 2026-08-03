# Research: TKA — Telekom Austria AG

> Stage-1 research-output. Stage 2 (JSON-injectie, validatie, build, commit) doet Claude Code.

---

## Bronnen-inventaris (Stap 0.5)

De analyse betreft de **geconsolideerde A1 Telekom Austria Group** (niet de standalone moeder Telekom Austria AG, die een aparte resultatenrekening met enkel deelnemingsresultaten kent). De geconsolideerde cijfers zijn de juiste basis voor een aandeelhoudersanalyse.

```
Jaar 2025 — HOOG
  Bron: A1 Group FY2025 persbericht + Results Report 2024 + Q3/Q1-Q3 2025 Earnings Update (geconsolideerd)
  URL:  https://a1.group/investor-relations/results-center/  ;
        https://a1.group/wp-content/uploads/sites/6/2025/10/A1-Group-%E2%80%93-Earnings-Update-Q3-2025.pdf
  Aanvullend gestructureerd (S&P Global Market Intelligence via StockAnalysis):
        https://stockanalysis.com/quote/vie/TKA/financials/
  Daadwerkelijk geopend: ja (PDF Q3-2025 volledig gelezen; FY2025 kerncijfers uit officieel persbericht)
  Cijfers overgenomen: omzet (gerapporteerd 5.577), EBITDA (gerapporteerd 2.062 / S&P-gestand. 1.630),
                       EBITDAaL (~1.656), nettowinst 613, EPS 0,92, FCF (econ. 596 / S&P 977),
                       CFO 1.844, capex, dividend/aandeel 0,42, balans (totale activa 10.228,
                       eigen vermogen 5.353), nettoschuld
  NB: de standalone "Telekom Austria AG"-resultatenrekening (omzet EUR 45 mln, nettowinst 204) is
      NIET gebruikt — dat is de holding-only jaarrekening.

Jaar 2024 — HOOG
  Bron: A1 Group Results Report 2024 (geconsolideerd, PDF) + FY2024 persbericht
  URL:  https://a1.group/wp-content/uploads/sites/6/2025/03/A1-Group_results-report-2024.pdf
  Daadwerkelijk geopend: ja (FY2024-persbericht + balanscijfers Dec'24 uit Q3-2025 PDF vergelijkingskolom)
  Cijfers overgenomen: omzet 5.413, EBITDA gerapp. ~1.962 (S&P 1.541), nettowinst 626/627, EPS 0,94,
                       FCF (econ. ~575 / S&P 924), CFO 1.814, capex 890, dividend 0,40,
                       totale activa 9.854, eigen vermogen 4.989, nettoschuld incl. lease 2.257

Jaar 2023 — HOOG
  Bron: A1 Group FY2023 + Q3-2025 PDF (vergelijkingskolommen) + S&P Global gestructureerd
  URL:  https://a1.group/wp-content/uploads/sites/6/2024/03/Telekom-Austria-AG-Financial-Report-2023.pdf ;
        https://stockanalysis.com/quote/vie/TKA/financials/cash-flow-statement/
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: omzet 5.251, EBITDA (S&P 1.596), nettowinst 645, EPS 0,97, FCF (S&P 622 / econ. ~384),
                       CFO 1.716, capex 1.094 (spectrum-zwaar jaar), dividend 0,36,
                       totale activa 9.557, eigen vermogen 4.601. NB: EuroTeleSites-afsplitsing sept 2023.

Jaar 2022 — HOOG
  Bron: A1 Combined Report 2022 + S&P Global gestructureerd
  URL:  https://a1.group/wp-content/uploads/sites/6/2023/09/A1_Combined_Report_2022_EN.pdf ;
        https://stockanalysis.com/quote/vie/TKA/financials/
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: omzet 5.005, EBITDA (S&P 1.561), nettowinst 634, EPS 0,95, FCF (S&P 822),
                       CFO 1.718, capex 896, dividend 0,32, totale activa 8.345, eigen vermogen 3.593

Jaar 2021 — HOOG
  Bron: A1 Annual Financial Report 2021 + S&P Global gestructureerd
  URL:  https://a1.com/investor-relations/results-center/annual-financial-reports/ ;
        https://stockanalysis.com/quote/vie/TKA/financials/
  Daadwerkelijk geopend: ja (S&P-reeks volledig; bevestigd tegen FY2022-rapport vergelijkingskolom)
  Cijfers overgenomen: omzet 4.748 (gerapp.) / 4.666 (operating), EBITDA (S&P 1.436), nettowinst 454,
                       EPS 0,68, FCF (S&P 732), CFO 1.586, capex 853, dividend 0,28,
                       totale activa 8.573, eigen vermogen 3.115

Jaren 2016–2020 — GEEN GEVERIFIEERDE BRON GEOPEND
  Zoekpoging(en): StockAnalysis/S&P-reeks begint betrouwbaar bij 2021; MacroTrends-pagina is gedelisted
                  (TKAGY) en gaf geen verifieerbare 2016-2020 rij in de zoekresultaten; de pre-2021
                  jaarverslagen zijn niet individueel geopend binnen deze run.
  Conclusie: 2016–2020 blijven LEEG in alle tabellen. Genoteerd in ontbrekende_data.
             Een 5-jaars (2021-2025) HOOG-reeks dekt de recente periode volledig; de afsplitsing van
             EuroTeleSites (2023) maakt pre-2023 vergelijkingen sowieso minder zuiver.
```

**TTM (per 31-3-2026):** omzet 5.577 (gerapp.), nettowinst 631, EPS 0,95, EBITDA (S&P) 1.647. Bron: S&P Global via StockAnalysis, Q1-2026-update. Q1-2026 los: omzet +3,9%, EBITDA +4,6%, nettowinst +14,7%, FCF +53,5% (bron: A1 Group Q1-2026-persbericht).

---

## Metadata
- **Ticker (bare):** TKA
- **Yahoo symbol:** TKA.VI
- **Exchange:** Wenen (Wiener Börse / WBAG)
- **Sector (GICS-achtig):** Communicatie
- **Industrie:** Telecommunicatiediensten (geïntegreerd vast + mobiel)
- **Land:** Oostenrijk
- **Peildatum analyse:** 2026-06-02
- **Koers op peildatum:** 9,83
- **Valuta:** EUR
- **Marktkapitalisatie:** EUR 6,5 mld
- **Marktkap in mln (lokale valuta):** 6527
- **Free float pct:** ~13
- **Indexlidmaatschap:** ATX (Wiener Börse Prime Market)
- **Domein:** a1.group

---

## 1. Executive summary

- **Kernthese:** Telekom Austria (A1 Group) is de grootste geïntegreerde telecomaanbieder van Oostenrijk en daarnaast marktleider of sterke nummer twee in zes Centraal- en Oost-Europese markten (Bulgarije, Kroatië, Wit-Rusland, Slovenië, Servië, Noord-Macedonië). Het bedrijf verdient zijn geld met terugkerende abonnementsinkomsten uit mobiele en vaste verbindingen, breedband, tv en zakelijke ICT-diensten — kasstromen die door hun contractuele en nutsachtige karakter zeer voorspelbaar zijn. De structurele groeidrivers zitten niet in het verzadigde, prijsconcurrerende Oostenrijkse thuisland (waar de omzet licht krimpt), maar in de internationale segmenten, die met dubbele cijfers groeien dankzij upselling, vraag naar snelle breedband en de uitrol van ICT-oplossingen. De afgelopen jaren heeft het bedrijf fors gedelederd: de nettoschuld exclusief leases is gedaald tot bijna nul, de vrije kasstroom groeit en het dividend is vijf jaar op rij verhoogd. Het belangrijkste risico is de combinatie van een dominante grootaandeelhouder (América Móvil, ~58%) met een minderheids-free float van slechts ~13%, plus de blootstelling aan het door sancties getroffen Wit-Rusland.
- **Oordeel:** HOLD
- **Fair value basis** (kansgewogen, lokale valuta): 12,12
- **Fair value kansgewogen:** 12,12
- **EPV per aandeel** (Earnings Power Value, zonder groeipremie): 14,03
- **Upside pct:** 23
- **Fair value scenarios:**

| Scenario | Fair value | Upside % | FCF groei % | WACC % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 8,81 | -10 | -1,5 | 8,21 | 30 |
| Basis | 12,56 | 28 | 1,0 | 7,71 | 50 |
| Optimistisch | 16,00 | 63 | 2,5 | 7,21 | 20 |

- **Reverse-DCF impliciete groei pct:** -0,3
- **Grootste kans:** Voortgaande dubbelcijferige groei in de CEE-segmenten plus dalende capex tilt de vrije kasstroom en het dividend structureel hoger.
- **Grootste risico:** De controlerende positie van América Móvil (~58%) en de minimale free float (~13%) beperken minderheidsaandeelhouders, met op termijn squeeze-out- of governance-risico.

---

## 2. Bedrijfsprofiel

- **Beschrijving:** Telekom Austria AG, dat onder de merknaam A1 opereert, is een geïntegreerde telecomonderneming die vaste en mobiele communicatiediensten levert aan particulieren, bedrijven, overheden en andere telecomoperators. Het bedrijf zit in het hart van de digitale waardeketen: het bouwt en beheert mobiele netwerken (2G tot 5G) en vaste netwerken (koper, kabel en in toenemende mate glasvezel), en verkoopt daarop diensten als mobiele abonnementen, vast internet, televisie, vaste telefonie en — steeds belangrijker — zakelijke ICT-, cloud- en connectiviteitsoplossingen. Voor de consument lost A1 het probleem op van betrouwbare connectiviteit en entertainment; voor bedrijven dat van veilige, schaalbare digitale infrastructuur. De omzet komt grotendeels tot stand via maandelijkse abonnementen (servicerevenue), aangevuld met de verkoop van toestellen (equipment revenue, ~15% van de omzet) en eenmalige ICT-projecten. A1 is marktleider in Oostenrijk en marktleider of sterke nummer twee in zes CEE-landen, met in totaal circa 29 miljoen mobiele lijnen (inclusief M2M) en ruim 6 miljoen vaste verbindingen. Wat het bedrijf onderscheidt is de combinatie van een defensief, kasstroomrijk thuismarktmonopolieachtig profiel met een sneller groeiende internationale poot die het concern blootstelt aan hogere groei maar ook aan landenrisico.
- **Geschiedenis:** Telekom Austria ontstond in 1996 toen de telecomactiviteiten werden afgesplitst van de Oostenrijkse post- en telegraafdienst (Post & Telekom Austria), als onderdeel van de Europese liberalisering van de telecommarkt. In november 2000 ging het bedrijf naar de beurs van Wenen, met de Oostenrijkse staat als grootaandeelhouder. In de jaren daarna bouwde Telekom Austria een internationale voetafdruk op door overnames in Centraal- en Oost-Europa: mobiele operators in Bulgarije (Mobiltel), Kroatië (Vipnet), Slovenië, Servië, Noord-Macedonië en Wit-Rusland werden ingelijfd, waarmee het bedrijf zich positioneerde als regionale consolidator. Een keerpunt kwam in 2014, toen het Mexicaanse América Móvil van Carlos Slim een meerderheidsbelang verwierf; de Oostenrijkse staatsholding (later ÖBAG) en América Móvil sloten een syndicaatsovereenkomst die de zeggenschap deelt. In 2018 werd de merknaam wereldwijd geüniformeerd onder "A1". Tijdens de coronacrisis (2020-2021) bewees het bedrijf zijn defensieve karakter: telecom bleef essentieel en de kasstromen hielden stand. Een belangrijke recente strategische stap was de afsplitsing in september 2023 van de mobiele-mastendivisie als zelfstandig beursgenoteerd bedrijf EuroTeleSites (één EuroTeleSites-aandeel per vier A1-aandelen); A1 huurt de masten sindsdien terug als ankerhuurder. De afgelopen vijf jaar kenmerken zich door consistente omzetgroei (gestuwd door internationaal), forse schuldafbouw, een progressief dividendbeleid en aanhoudende investeringen in glasvezel en 5G.
- **Bedrijfsmodel:** A1 verdient het grootste deel van zijn geld met terugkerende servicerevenue: maandelijkse abonnementen voor mobiele telefonie, vast internet, tv en vaste telefonie, plus contracten voor zakelijke ICT- en connectiviteitsdiensten. Dit is een kapitaalintensief abonnementsmodel met hoge klantretentie en lage churn (~1,1%). Daarnaast is er equipment revenue (verkoop van smartphones en hardware, ~15% van omzet, met dunne marges) en eenmalige ICT-projectomzet. De winstgevendheid wordt gedreven door schaalvoordelen in netwerkexploitatie: zodra het netwerk er ligt, vertaalt elke extra abonnee zich grotendeels in marge. Geografisch is de omzet gesplitst in Oostenrijk (~50%, defensief en licht krimpend) en Internationaal (~37-40%, sterk groeiend). Prijsindexatie (inflatiekoppeling van tarieven) is een belangrijke omzethefboom geweest, al neemt dat effect af nu de inflatie daalt.
- **IPO-context:** Telekom Austria ging in november 2000 naar de beurs van Wenen, in het kader van de privatisering van de voormalige staatstelecom. De Oostenrijkse staat bleef destijds grootaandeelhouder. De kapitaalstructuur is sindsdien ingrijpend veranderd: in 2014 nam América Móvil een meerderheidsbelang, en de staat bracht zijn belang onder bij staatsholding ÖBAG. Een kapitaalverhoging financierde destijds de balansversterking. Er is sinds de IPO geen sprake geweest van pre-IPO financial engineering die het huidige beeld vertekent — de relevante structuurwijziging is de América Móvil-overname van 2014, niet de beursgang zelf.
- **Klantprofiel:** A1 bedient zowel consumenten (B2C) als bedrijven en overheden (B2B), met daarnaast wholesale-omzet aan andere operators. De klantenbasis is breed en zeer gefragmenteerd: geen enkele klant is materieel voor de omzet, wat het klantconcentratierisico laag maakt. In Oostenrijk is A1 marktleider; internationaal is het in de meeste landen de sterke nummer twee. De retentie is hoog door overstapdrempels (gebundelde diensten, contractduur, nummerbehoud) en de churn is met circa 1,1% laag voor de sector. Het aantal postpaid-abonnees groeit, terwijl prepaid licht daalt — een gunstige mixverschuiving naar hoogwaardiger, stabieler inkomen. De zakelijke ICT-tak groeit het hardst en verhoogt de gemiddelde klantwaarde.
- **Oprichtingsjaar:** 1996
- **IPO-datum:** 2000-11
- **IPO-koers** (lokale valuta): — (niet geverifieerd binnen deze run)
- **Personeel** (FTE): ~16.900
- **Landen actief:** 7 (Oostenrijk, Bulgarije, Kroatië, Wit-Rusland, Slovenië, Servië, Noord-Macedonië)
- **Klantconcentratie:** Geen materiële klantconcentratie; brede consumenten- en zakelijke basis met circa 29 miljoen mobiele lijnen en ruim 6 miljoen vaste verbindingen. Klantconcentratierisico is laag.

### Geografische spreiding (omzet)
| Regio | Omzet % | Valuta-exposure |
|---|---|---|
| Oostenrijk | ~50 | EUR |
| Internationaal (CEE) | ~37-40 | EUR (Bulgarije, Kroatië, Slovenië via euro/koppeling), BYN (Wit-Rusland), RSD (Servië), MKD (N.-Macedonië) |
| Overig/wholesale/equipment | rest | EUR |

**Toelichting geografie:** De helft van de omzet komt uit het euro-thuisland Oostenrijk; de internationale poot levert het leeuwendeel van de groei. Een deel van de CEE-markten gebruikt de euro of euro-gekoppelde valuta's (Bulgarije, Kroatië, Slovenië), wat het FX-risico beperkt. De grootste valutarisico's zitten in de Wit-Russische roebel (BYN), de Servische dinar (RSD) en de Macedonische denar (MKD). Wit-Rusland is door sancties bovendien een politiek en repatriërings-risico: winsten zijn niet altijd vrij uitkeerbaar.

### Segmenten
| Naam | Omzet % | Beschrijving |
|---|---|---|
| Oostenrijk | ~50 | Geïntegreerde marktleider mobiel + vast + tv + ICT; defensief, licht krimpende servicerevenue, hoge marge. |
| Internationaal | ~37-40 | Zes CEE-landen; sterke groei (dubbelcijferig) in servicerevenue en equipment, gedreven door Bulgarije en Wit-Rusland. |

### Aandeelhouders (top 5)
| Naam | Belang % | Type (oprichter / PE / institutioneel / retail) |
|---|---|---|
| América Móvil (Carlos Slim) | ~58 | Controlerend |
| ÖBAG (Oostenrijkse staatsholding) | ~28 | Controlerend (staat) |
| Free float (institutioneel + retail) | ~13 | Publiek |

- **Institutioneel eigendomstrend:** Stabiel tot licht dalend. América Móvil heeft zijn belang in 2023 verhoogd (tot boven 56%), waardoor de vrij verhandelbare free float verder kromp tot circa 13%. De gecombineerde zeggenschap van América Móvil en ÖBAG bedraagt ruim 86% via aandelen en een syndicaatsovereenkomst, wat de invloed van overige institutionele beleggers structureel beperkt.

---

## 3. Financieel — historische data (10 jaar + TTM)

### Resultatenrekening (bedragen in mln EUR)

Bron-eis: recente 5 jaren (2021-2025) zijn HOOG (officiële A1 Group-rapporten/persberichten, aangevuld met S&P Global gestructureerd). Jaren 2016-2020 ontbreken (geen geverifieerde bron geopend) en blijven leeg.

| Jaar | Omzet | Omzetgroei % | Brutowinst | Brutomarge % | EBIT | EBIT-marge % | EBITDA | EBITDA-marge % | Nettowinst | Nettomarge % | EPS | EPS-groei % | Aandelen mln |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2016 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2017 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2018 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2019 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2020 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2021 | 4.748 | — | 2.625 | 56,3 | 759 | 16,3 | 1.436 | 30,8 | 454 | 9,7 | 0,68 | — | 664,08 |
| 2022 | 5.005 | 5,4 | 2.771 | 56,4 | 877 | 17,8 | 1.561 | 31,8 | 634 | 12,9 | 0,95 | 39,5 | 664,08 |
| 2023 | 5.251 | 4,9 | 2.896 | 56,1 | 916 | 17,8 | 1.596 | 30,9 | 645 | 12,5 | 0,97 | 1,8 | 664,08 |
| 2024 | 5.413 | 3,0 | 3.021 | 56,8 | 853 | 16,1 | 1.541 | 29,0 | 626 | 11,8 | 0,94 | -3,0 | 664,08 |
| 2025 | 5.577 | 3,0 | 3.128 | 57,2 | 935 | 17,1 | 1.630 | 29,8 | 613 | 11,2 | 0,92 | -2,2 | 664,08 |
| TTM | 5.577 | 2,7 | 3.151 | 57,0 | 950 | 17,2 | 1.647 | 29,8 | 631 | 11,4 | 0,95 | — | 664,08 |

- **Toelichting resultaten:** De omzet groeide van EUR 4.748 mln (2021) naar EUR 5.577 mln (2025), een CAGR van circa 4,1% — bescheiden maar consistent, geheel gedragen door de internationale segmenten terwijl Oostenrijk licht kromp. De EBITDA-marge ligt stabiel rond 30% (S&P-gestandaardiseerde definitie); het bedrijf zelf rapporteert een hogere EBITDA (EUR 2.062 mln in 2025) omdat het leasekosten anders behandelt. De nettowinst piekte in 2022-2023 (~EUR 640 mln) en daalde daarna licht naar EUR 613 mln in 2025, deels door de EuroTeleSites-afsplitsing (september 2023) die vergelijkingen vertroebelt en deels door hogere afschrijvingen. De EPS volgt dit patroon (~EUR 0,92-0,97). Het aandelenaantal is al jaren constant op 664,08 mln: geen verwatering, geen inkopen.
- **Omzet-CAGR:** 4,1% (2021-2025)

### Kasstromen

| Jaar | CFO | Capex | FCF | FCF na SBC | FCF/aandeel | FCF-marge % | FCF-groei % | FCF-conversie % | SBC | Dividend totaal | Aandeleninkoop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021 | 1.586 | 853 | 732 | 732 | 1,10 | 15,7 | -0,8 | 161 | 0 | 166 | 0 |
| 2022 | 1.718 | 896 | 822 | 822 | 1,24 | 16,7 | 12,2 | 130 | 0 | 186 | 0 |
| 2023 | 1.716 | 1.094 | 622 | 622 | 0,94 | 12,1 | -24,3 | 96 | 0 | 213 | 0 |
| 2024 | 1.814 | 890 | 924 | 924 | 1,39 | 17,4 | 48,6 | 148 | 0 | 239 | 0 |
| 2025 | 1.844 | 866 | 977 | 977 | 1,47 | 17,9 | 5,8 | 160 | 0 | 266 | 0 |

- **Toelichting kasstromen:** De operationele kasstroom (CFO) groeide gestaag van EUR 1.586 mln (2021) naar EUR 1.844 mln (2025). De S&P-gerapporteerde FCF (CFO − capex) schommelde sterker: de scherpe daling in 2023 (−24%) was geen verslechtering van het bedrijf maar het gevolg van uitzonderlijk hoge capex (EUR 1.094 mln, met fors spectrum- en glasvezelinvesteringen); het herstel in 2024 (+49%) weerspiegelt de normalisatie van die capex. Belangrijk: de A1 Group rapporteert zélf een lagere "economische" vrije kasstroom (EUR 596 mln in 2025) omdat die ná leasebetalingen (~EUR 330 mln/jaar voor onder meer de teruggehuurde EuroTeleSites-masten) en sociale plannen wordt berekend. Dit onderscheid is materieel voor de waardering: de S&P-FCF van ~EUR 977 mln overschat de werkelijk distribueerbare kasstroom. SBC (aandelencompensatie) is bij A1 niet materieel. De FCF-conversie (FCF/nettowinst) ligt boven 100% door de hoge afschrijvingen — typisch voor kapitaalintensieve telecom.

### Balans-ratio's (10 jaar)

| Jaar | Nettoschuld | Nettoschuld/EBITDA | Eigen vermogen | ROE % | ROIC % | ROCE % | Current ratio | Solvabiliteit % | Goodwill % van EV | Working capital |
|---|---|---|---|---|---|---|---|---|---|---|
| 2021 | 2.745 | 1,9 | 3.115 | 15,4 | 9,1 | 13,5 | 0,61 | 36,3 | 16,7 | -1.155 |
| 2022 | 2.339 | 1,5 | 3.593 | 18,9 | 11,6 | 14,8 | 0,60 | 43,1 | 20,5 | -972 |
| 2023 | 2.510 | 1,6 | 4.601 | 15,8 | 11,1 | 11,9 | 0,84 | 48,1 | 13,9 | -301 |
| 2024 | 2.220 | 1,4 | 4.989 | 13,1 | 9,8 | 10,7 | 0,97 | 50,6 | 13,4 | -61 |
| 2025 | 1.848 | 1,1 | 5.353 | 11,9 | 10,3 | 12,5 | 0,83 | 52,3 | 12,9 | -483 |

- **Toelichting balans:** De balans is fors versterkt. De nettoschuld (inclusief leaseverplichtingen, S&P-definitie) daalde van EUR 2.745 mln (2021) naar EUR 1.848 mln (2025), en de nettoschuld/EBITDA van 1,9x naar 1,1x. Cruciaal onderscheid: A1's nettoschuld **exclusief** leases is per eind 2025 nagenoeg nul (de bruto financiële schuld van ~EUR 754 mln wordt vrijwel volledig gedekt door cash en kortlopende beleggingen van ~EUR 760 mln). De bruto schuld en de leaseverplichtingen bewegen niet in dezelfde richting: de financiële schuld is afgelost terwijl de leaseverplichting (door de EuroTeleSites-mastenhuur) hoog blijft. Wie alleen naar de leasebelaste schuld kijkt, onderschat dus de balanssterkte. Het eigen vermogen groeide van EUR 3.115 mln naar EUR 5.353 mln, de solvabiliteit van 36% naar 52%. Goodwill is met ~13% van de balans beperkt en stabiel. Fitch bevestigde in juli 2025 de A−-rating met stabiele outlook.

### Kapitaalstructuur huidig
- **Nettoschuld (huidig):** ~150 (financiële nettoschuld excl. leases, nagenoeg nul); ~1.848 inclusief leases (S&P)
- **Bruto schuld:** ~754 (financieel, excl. leases); ~2.608 inclusief leases
- **Cash & equivalents:** ~362 + ~398 kortlopende beleggingen = ~760
- **Lease-verplichtingen (IFRS-16):** ~1.855 (kort + lang, grotendeels EuroTeleSites-mastenhuur)
- **Gemiddelde rente %:** ~3,5 (pre-tax op financiële schuld)
- **Rente-dekking (EBIT/rente):** ~8,7x (EBIT 935 / netto rentelasten ~107)

### Non-GAAP / aanpassingen
- **Gebruikt?** true
- **Welke aanpassingen:** A1 rapporteert EBITDA (vóór leasekosten, EUR 2.062 mln 2025) én EBITDAaL (na leasekosten, ~EUR 1.656 mln), plus FCF na leases/sociale plannen. S&P Global hanteert een eigen gestandaardiseerde EBITDA (EUR 1.630 mln) die afwijkt van beide bedrijfsdefinities.
- **Waarom:** Vergelijkbaarheid en de IFRS-16-leaseboekhouding (de teruggehuurde masten staan als lease op de balans). In de waardering is bewust de economische, na-lease kasstroom als basis gebruikt om dubbeltelling van het leasevoordeel te vermijden.

---

## 4. Moat (concurrentievoordeel)

- **Oordeel:** NARROW MOAT
- **Moat-categorieën:**

| Naam | Sterkte (sterk/middel/zwak/geen) | Toelichting |
|---|---|---|
| Immateriële activa | middel | A1 bezit schaarse, gereguleerde activa: spectrumlicenties (2G-5G) en netwerkvergunningen die door de overheid worden toegekend en niet vrij repliceerbaar zijn. Het A1-merk geniet hoge naamsbekendheid in Oostenrijk en CEE. Dit geeft bescherming, maar licenties moeten periodiek (kostbaar) worden herverworven. |
| Overstapkosten | middel | Klanten ervaren reële drempels: gebundelde diensten (mobiel + vast + tv), contractduur, nummerbehoud en de moeite van migratie. De lage churn (~1,1%) bevestigt dit. De drempels zijn echter niet absoluut; nummerportabiliteit en agressieve promoties van concurrenten verlagen ze. |
| Netwerkeffecten | zwak | Telecom kent nauwelijks echte netwerkeffecten: de waarde voor een individuele klant groeit niet wezenlijk met het aantal andere abonnees (interconnectie is gereguleerd). Geen materieel voordeel hier. |
| Kostenvoordeel | sterk | Als marktleider in Oostenrijk en schaalspeler in CEE geniet A1 schaalvoordelen in netwerkexploitatie, inkoop en overhead. Een nieuwkomer zou miljarden moeten investeren om een vergelijkbaar dekkend netwerk te bouwen — een afschrikwekkende toetredingsdrempel. |
| Efficiënte schaal | sterk | Telecommarkten zijn natuurlijke oligopolies: de hoge vaste kosten van netwerken maken dat slechts een handvol spelers winstgevend kan opereren. In elk land waar A1 actief is, zijn er doorgaans maar 3-4 netwerkoperators. Dit beperkt structureel het aantal concurrenten. |

- **Kwantitatief bewijs:** De ROIC ligt al vijf jaar stabiel rond 9-12% (2021: 9,1%; 2025: 10,3%), structureel boven de geschatte WACC van ~7,7% — een positieve spread van circa 2,6 procentpunt die op waardecreatie wijst. De EBITDA-marge is opvallend stabiel rond 30%, wat duidt op prijszettingsvermogen en kostencontrole. Het marktleiderschap in Oostenrijk en de nummer-1/2-posities in CEE zijn al jaren bestendig.
- **Duurzaamheid:** 5-10 jaar. De moat is reëel maar smal: schaal en efficiënte schaal beschermen het bedrijf, maar de spread ROIC−WACC van ~2,6pp is bescheiden (geen WIDE-moat-niveau van >10pp).
- **Erosierisico's:** Aanhoudende prijsconcurrentie in het verzadigde Oostenrijkse thuisland holt de marge daar uit (servicerevenue krimpt al). Regulering (EU-roaming, prijscontroles, indexatie-rechtszaken) kan tarieven onder druk zetten. Kapitaalintensieve glasvezel- en 5G-uitrol vereist permanente herinvestering om de positie te behouden. Substitutie door OTT-diensten (WhatsApp, streaming) erodeert traditionele spraak- en tv-omzet.

---

## 5. Management

- **CEO-naam + tenure:** Alejandro Plater (CEO sinds 2024; daarvoor COO; in het concern sinds ~2015)
- **CFO-naam + tenure:** Sonja Wallner (CFO A1 Group sinds september 2023; bij het bedrijf sinds 2000, CFO A1 Austria sinds 2015)
- **Oprichter nog betrokken?:** N.v.t. (geen oprichter; voortgekomen uit staatstelecom). Dominante eigenaar América Móvil (Carlos Slim-familie) is via de raad sterk betrokken.
- **Insider ownership %:** Management houdt geen materieel eigen belang (geen owner-operator); de zeggenschap ligt bij grootaandeelhouders América Móvil (~58%) en ÖBAG (~28%).
- **Capital allocation track record:** Schuldafbouw, progressief dividend en aanhoudende netwerkinvesteringen; geen aandeleninkoop, geen verwatering.

| Jaar | Dividend totaal | Aandeleninkoop | M&A uitgaven | Organische capex |
|---|---|---|---|---|
| 2021 | 166 | 0 | — | 853 |
| 2022 | 186 | 0 | 15 | 896 |
| 2023 | 213 | 0 | — | 1.094 |
| 2024 | 239 | 0 | 4 | 890 |
| 2025 | 266 | 0 | 10 | 866 |

- **M&A-track-record:** Beperkte, kleine bolt-on-acquisities in recente jaren (enkele miljoenen per jaar). De grootste strategische actie was juist een desinvestering: de afsplitsing van EuroTeleSites (mobiele masten) in 2023, die waarde ontsloot en de focus op kernactiviteiten versterkte. Geen grote waardevernietigende overnames in de periode.
- **Beloning:** Volgens het remuneratierapport is de variabele beloning gekoppeld aan EBITDA-groei en vrije kasstroom (elk ~25%) en operationele ROIC (~34%) — KPI's die op langetermijnwaardecreatie zijn gericht, wat positief is. Het ontbreken van publiek gedetailleerde CEO-pay-ratio's en de dominantie van de grootaandeelhouders maken een volledig oordeel lastig.
- **Oordeel management:** NEUTRAAL
- **Toelichting:** Het management voert een degelijk, voorspelbaar beleid: consistente schuldafbouw, een vijf jaar op rij verhoogd dividend, gedisciplineerde capex en een waarde-ontsluitende afsplitsing van de mastendivisie. De bonus-KPI's (FCF, EBITDA, operationele ROIC) zijn goed afgestemd op aandeelhoudersbelang. Tegelijk is er geen owner-operator-alignment op managementniveau en wordt de governance gedomineerd door América Móvil en ÖBAG, die samen ruim 86% van de stemrechten controleren. Voor minderheidsaandeelhouders betekent dit dat strategische beslissingen buiten hun invloed liggen. Dat weegt het oordeel naar neutraal in plaats van sterk — niet vanwege gebrek aan competentie, maar vanwege de governance-asymmetrie.

---

## 6. Sector & concurrentie

- **Sector-groeivooruitzicht:** West-Europese telecomdiensten groeien laag-enkelcijferig (~1-2%/jaar); CEE-markten sneller (~3-6%). A1's eigen guidance voor 2026 is 2-3% omzetgroei.
- **Porter five forces:**
  - **Rivaliteit (HOOG):** In Oostenrijk is de markt verzadigd en zeer prijsconcurrerend met meerdere netwerkoperators en MVNO's; promotionele intensiteit drukt de marges. In CEE is de concurrentie stabieler maar aanwezig.
  - **Nieuwe toetreders (LAAG):** De kapitaalvereisten voor een landelijk dekkend netwerk plus schaarse spectrumlicenties vormen een vrijwel onoverkomelijke toetredingsdrempel. Hoog gunstig voor zittende spelers.
  - **Substituten (MIDDEL):** OTT-diensten (WhatsApp, streaming) vervangen traditionele spraak en tv, maar de onderliggende dataconnectiviteit blijft onmisbaar — A1 levert juist die pijplijn.
  - **Macht leveranciers (MIDDEL):** Netwerkapparatuurleveranciers (Ericsson, Nokia) en toestelfabrikanten (Apple, Samsung) zijn geconcentreerd; A1's schaal geeft echter inkoopmacht.
  - **Macht afnemers (MIDDEL):** Consumenten zijn individueel machteloos maar collectief prijsgevoelig; nummerportabiliteit en promoties geven hen overstapmacht. Zakelijke klanten onderhandelen harder.
  - **Conclusie:** De telecomsector is structureel een gematigd aantrekkelijke oligopolie: hoge toetredingsdrempels en efficiënte schaal beschermen de winstgevendheid, maar verzadiging, prijsconcurrentie in volwassen markten en zware herinvesteringseisen beperken het rendement. Voor A1 is de sector aantrekkelijk genoeg voor stabiele kasstromen, maar geen omgeving voor uitbundige groei.
- **Concurrenten:**

| Concurrent | Marktaandeel % |
|---|---|
| Magenta Telekom (Deutsche Telekom, AT) | — |
| Hutchison Drei (AT) | — |
| Deutsche Telekom (CEE-overlap) | — |
| Lokale CEE-operators (per land wisselend) | — |

- **Positie van het bedrijf:** Marktleider in Oostenrijk; marktleider of sterke nummer twee in zes CEE-landen. Geïntegreerde speler (vast + mobiel) met schaalvoordeel.
- **Positie_toelichting:** A1 is in zijn thuismarkt de duidelijke nummer één en concurreert daar met Magenta (Deutsche Telekom) en Drei (Hutchison). In de CEE-markten is het doorgaans de sterke nummer twee. Vergeleken met grote West-Europese telco-peers (Deutsche Telekom, Orange, Telefónica) is A1 kleiner en groeit het — dankzij CEE — relatief iets sneller, terwijl het op waardering (EV/EBITDA ~5x, P/E ~10x) op of onder het sectorgemiddelde handelt. De waarderingskorting weerspiegelt de beperkte free float, de Wit-Rusland-blootstelling en de governance-dominantie van América Móvil, niet zozeer een operationele zwakte.

### TAM/SAM/SOM
- **TAM (mln lokale valuta):** — (niet betrouwbaar te kwantificeren binnen deze run)
- **TAM-groei %:** ~2-3 (gewogen Oostenrijk + CEE)
- **SAM (mln):** —
- **SAM-groei %:** —
- **Huidige penetratie %:** —
- **Impliciete penetratie na horizon %:** —
- **Groei plausibel?** true
- **Bron TAM/SAM:** —
- **Toelichting:** Telecom is een verzadigde nutsmarkt; groei komt uit ARPU-stijging (upselling, indexatie), datavolume en ICT-uitbreiding, niet uit nieuwe-abonnee-penetratie. Een TAM/SAM-kwantificering is voor een gevestigde telco minder zinvol en is hier niet betrouwbaar onderbouwd.

---

## 7. Analyse-frameworks (9 frameworks, SCORES 0-5)

### Graham
- **Oordeel:** GEDEELTELIJK
- **Graham number:** ~9,9 (√(22,5 × EPS 0,92 × boekwaarde/aandeel 8,06) ≈ 12,9... ; conservatief ~9,9 op genormaliseerde basis)
- **Margin of safety %** (t.o.v. huidige koers): ~28 (t.o.v. basis fair value 12,56)
- **Toelichting:** Telekom Austria voldoet aan diverse defensieve Graham-criteria: de P/E van ~10,4 ligt ruim onder de drempel van 15, de P/B van ~1,23 onder 1,5, en het bedrijf betaalt een stabiel, groeiend dividend. De schuldpositie is sterk (financiële nettoschuld ~nul). Wat ontbreekt voor een topscore is een margin of safety van ≥30% op de basiswaarde — die ligt op ~28%. Voor een waardebelegger is dit een degelijk, laaggewaardeerd defensief aandeel dat grotendeels aan Grahams eisen voldoet, maar net niet met de allergrootste veiligheidsmarge.
- **Score (0-5):** 4

### Buffett / Munger
- **Oordeel:** GEDEELTELIJK
- **ROIC structureel boven WACC?** true
- **Toelichting:** A1 is een begrijpelijk bedrijf met voorspelbare, nutsachtige kasstromen — precies het soort onderneming dat Buffett waardeert. De ROIC (~10,3%) ligt structureel boven de WACC (~7,7%), wat duidt op waardecreatie, en de prijs is redelijk (P/FCF ~11 op economische FCF). Maar de moat is slechts NARROW (de spread ROIC−WACC van ~2,6pp is bescheiden, geen 2×WACC), het management is neutraal door de governance-dominantie van América Móvil, en de groei is laag. Het is een "fair company at a fair price" eerder dan een "wonderful company" — vandaar een gedeeltelijk oordeel.
- **Score (0-5):** 3

### Peter Lynch
- **Categorie:** Stalwart
- **Oordeel:** NEUTRAAL
- **PEG-ratio:** ~1,7
- **Toelichting:** A1 is een klassieke "Stalwart" in Lynch-termen: een grote, gevestigde onderneming met bescheiden maar stabiele groei (omzet ~4%/jaar, EPS vlak), die vooral aantrekkelijk is voor dividend en kapitaalbehoud, niet voor explosieve koerswinst. Het verhaal is helder en in twee zinnen uit te leggen: marktleider in Oostenrijk, groei uit Oost-Europa, fors dalende schuld en stijgend dividend. De PEG-ratio van ~1,7 (P/E 10,4 gedeeld door ~6% verwachte winstgroei) is niet goedkoop genoeg voor een Lynch-koopsignaal; bij een Stalwart zoekt Lynch eerder naar een PEG onder 1.
- **Score (0-5):** 2

### Phil Fisher
- **Oordeel:** GEMIDDELD
- **Toelichting:** Vanuit Fisher-perspectief scoort A1 gemengd. De producten (connectiviteit, ICT) hebben groeipotentieel in CEE en in zakelijke diensten, en de marges worden beschermd door schaal en de oligopolistische marktstructuur. Maar telecom is geen R&D-gedreven innovatiesector: het bedrijf investeert in netwerkinfrastructuur, niet in onderscheidende productinnovatie, en het R&D-budget is geen onderscheidende factor. De managementintegriteit is redelijk maar niet uitzonderlijk (neutraal oordeel, governance-dominantie). Eén van de drie Fisher-kerncriteria (margebescherming) is duidelijk voldaan; de innovatiecultuur is dat niet.
- **Score (0-5):** 2

### Magic Formula (Greenblatt)
- **Oordeel:** AANTREKKELIJK
- **Earnings yield %:** ~11,6 (EBIT/EV)
- **Return on capital %:** ~18,6 (EBIT / (netto werkkapitaal + netto vaste activa))
- **Toelichting:** Op de Greenblatt-assen scoort A1 goed op earnings yield: met ~11,6% (EBIT gedeeld door enterprise value) is het aandeel goedkoop ten opzichte van zijn operationele winst. De return on capital van ~18,6% is degelijk maar niet uitzonderlijk hoog — kapitaalintensieve telecom bindt veel vaste activa, wat de kapitaalrendementsmaatstaf drukt. De combinatie van een hoge earnings yield met een acceptabel kapitaalrendement maakt het aandeel volgens de Magic Formula aantrekkelijk gewaardeerd, zonder in de topcategorie te vallen.
- **Score (0-5):** 3

### Moat
- **Score (0-5):** 2

### Management
- **Score (0-5):** 3

### Fair Value DCF
- **Score (0-5):** 4

### Fair Value IPO-gecorr.
- **Score (0-5):** 4

### Scorekaart totaal
- **Totaalscore:** 27
- **Max:** 45
- **Eindoordeel:** HOLD
- **Samenvatting:** Telekom Austria scoort 27 van de 45 punten — solide middenmoot. De sterke punten zijn de aantrekkelijke waardering (Graham 4, Fair Value DCF 4: ~28% upside op de basiswaarde), de hoge earnings yield (Magic Formula 3) en de bijna schuldenvrije, kasstroomrijke balans. De zwakkere punten zijn de smalle moat (2, bescheiden ROIC−WACC-spread), het lage groeiprofiel (Lynch 2, Fisher 2) en het neutrale management (3, door governance-dominantie van América Móvil). Met een totaal tussen 24 en 33 en een Fair Value DCF-score van 4 luidt het deterministische eindoordeel HOLD. De DCF wijst op onderwaardering en de markt prijst zelfs een lichte eeuwigdurende krimp in (reverse DCF ~−0,3%), wat overdreven pessimistisch oogt; tegelijk beperken de minimale free float, de Wit-Rusland-blootstelling en het ontbreken van koersaanjagers het opwaartse potentieel op korte termijn. Een belegger die het aandeel bezit kan het aanhouden voor het dividend (~4,3%) en de kasstroom; een nieuwe positie wordt pas duidelijk aantrekkelijk met meer veiligheidsmarge (koopniveau ~EUR 9 of lager).

---

## 8. Risico's

| # | Omschrijving | Kans | Impact | DCF-aanname geraakt | Toelichting |
|---|---|---|---|---|---|
| 1 | Governance: dominante grootaandeelhouder América Móvil (~58%), free float ~13% | HOOG | GROOT | WACC (risicopremie), terminal value | América Móvil en ÖBAG controleren ruim 86% van de stemrechten. Minderheidsaandeelhouders hebben geen invloed op strategie, dividend of een eventuele squeeze-out. Dit rechtvaardigt een blijvende waarderingskorting en verhoogt de cost of equity. |
| 2 | Wit-Rusland-blootstelling (sancties, repatriëring, valuta) | MIDDEN | GROOT | omzetgroei, terminal value, CRP | Wit-Rusland levert een aanzienlijk deel van de internationale EBITDA maar zit onder sancties; winsten zijn niet altijd vrij uitkeerbaar en de roebel is volatiel. Een escalatie kan de waarde van dit segment grotendeels uithollen. |
| 3 | Verzadiging en prijsconcurrentie in Oostenrijk | HOOG | MIDDEL | omzetgroei, EBIT-marge | De Oostenrijkse servicerevenue krimpt al (−2 tot −4%). Aanhoudende promotionele druk en dalende indexatie-effecten (lagere inflatie) zetten de marge in de helft van de omzet onder druk. |
| 4 | Kapitaalintensiteit: blijvend hoge capex (glasvezel, 5G, spectrum) | MIDDEN | MIDDEL | FCF-basis, capex | Telecom vereist permanente herinvestering. Spectrumveilingen (zoals de Servische 5G-tender, startprijs EUR 100 mln) en glasvezeluitrol kunnen de vrije kasstroom in piekjaren fors drukken, zoals in 2023 zichtbaar was. |
| 5 | Regulatoir risico (EU, indexatie-rechtszaken) | MIDDEN | MIDDEL | omzetgroei, EBIT-marge | Lopende rechtszaken (Arbeiterkammer, VKI over indexatieclausules) en EU-regulering kunnen tariefverhogingen beperken. A1 won de eerste aanleg, maar hoger beroep loopt nog. |
| 6 | Valutarisico CEE (BYN, RSD, MKD) | MIDDEN | KLEIN | omzetgroei, terminal value | Een deel van de internationale omzet is in niet-euro-valuta die kan depreciëren (de Wit-Russische roebel daalde gemiddeld 7% in 2024), wat de gerapporteerde groei drukt. |
| 7 | Substitutie door OTT-diensten | MIDDEN | KLEIN | omzetgroei | Streaming en messaging blijven traditionele spraak- en tv-omzet eroderen; gedeeltelijk gecompenseerd doordat A1 de onderliggende dataconnectiviteit levert. |
| 8 | Pre-IPO financial engineering | LAAG | KLEIN | FCF-basis | Niet geconstateerd. De IPO dateert van 2000; de relevante structuurwijziging was de América Móvil-overname (2014), niet schuldopbouw bij gelieerde partijen vóór beursgang. Er zijn geen aanwijzingen voor dividend-recapitalisatie of insider-schuldaflossing uit IPO-opbrengsten. Gecorrigeerde fair value = ongecorrigeerde fair value. |

---

## 9. These invalide bij

Deze these (laaggewaardeerde, kasstroomrijke defensieve telco met CEE-groei) is weerlegd wanneer: (a) de internationale segmenten stoppen met groeien of de Wit-Rusland-activiteiten door sanctie-escalatie grotendeels onbruikbaar/onuitkeerbaar worden; (b) de Oostenrijkse marge versnelt verslechtert door een prijzenoorlog waardoor de groeps-EBITDA structureel krimpt; of (c) América Móvil een squeeze-out aankondigt tegen een prijs onder de berekende fair value, of de vrije kasstroom structureel onvoldoende wordt om het dividend te dekken.

---

## 10. ESG

### Materiële factoren (SASB-gebaseerd)

| Factor | SASB-categorie | Risiconiveau (Laag/Midden/Hoog) | Financiële impact | DCF-impact |
|---|---|---|---|---|
| Datasecurity & privacy | Telecommunication Services — Data Privacy/Security | Midden | Boetes/reputatieschade bij datalekken; GDPR-naleving | Marge, eenmalige kosten |
| Energieverbruik netwerken | Environmental Footprint of Operations | Midden | Hoge en stijgende elektriciteitskosten netwerken; energieprijsvolatiliteit | EBIT-marge |
| Governance / minderheidsbescherming | Business Ethics / Governance | Hoog | Beperkte minderheidsinvloed; squeeze-out-risico | WACC, terminal value |
| Wit-Rusland (mensenrechten/sancties) | Country/operating risk | Hoog | Sanctierisico, repatriëringsbeperkingen | omzet, terminal value |
| Digitale inclusie / netwerktoegang | Access & Affordability | Laag | Regulatoire verplichtingen tot dekking | capex |

- **Eindoordeel ESG:** GEMIDDELD RISICO
- **Toelichting:** Op milieugebied is A1 een relatief lichte vervuiler (diensten, geen zware industrie), al is het energieverbruik van netwerken materieel. De grootste ESG-zorgen zitten in de governance (G): de dominantie van América Móvil en de zeer beperkte free float beperken minderheidsaandeelhouders structureel. Daarnaast vormt de aanwezigheid in het door sancties getroffen Wit-Rusland een reëel reputatie- en operationeel risico. Datasecurity is een doorlopende, beheersbare risicopost.

---

## 11. Katalysatoren (5-8 stuks, chronologisch)

| Datum ca. | Omschrijving | Richting | Impact |
|---|---|---|---|
| 2026-05 | Q1 2026-resultaten (reeds gepubliceerd: omzet +3,9%, nettowinst +14,7%, FCF +53,5%) | POSITIEF | MIDDEL |
| 2026-06 | Jaarlijkse Algemene Vergadering — goedkeuring dividend EUR 0,42 | POSITIEF | KLEIN |
| 2026-07 | Halfjaarcijfers Q2/H1 2026 | NEUTRAAL | MIDDEL |
| 2026-08 | Mogelijke Fitch-ratingbevestiging (A−) | NEUTRAAL | KLEIN |
| 2026-10 | Q3 2026-resultaten | NEUTRAAL | MIDDEL |
| 2026-Q4 | Uitkomst indexatie-rechtszaken (VKI/Arbeiterkammer hoger beroep) | BINAIR | MIDDEL |
| 2026-Q4 | Servische 5G-spectrumveiling (kapitaaluitgave) | NEGATIEF | KLEIN |
| 2027-02 | Q4/FY2026-resultaten + dividendvoorstel 2026 | POSITIEF | MIDDEL |

---

## 12. Fair value — kwantitatief (DCF)

### WACC-componenten
- **Risicovrije rente %:** 3,30
- **Bron risicovrije rente:** Oostenrijkse 10-jaars staatsobligatie, ~3,30% (mei 2026; TradingEconomics/WorldGovernmentBonds)
- **Type:** nominal (spot)
- **ERP (equity risk premium) %:** 4,33
- **Bron ERP:** Damodaran implied ERP, volwassen eurozone-basis (peildatum 2026)
- **Beta (adjusted, Blume):** 0,85
- **Bron beta:** Sectorbeta telecom (defensief, ~0,7-0,9), opwaarts bijgesteld voor EM-blootstelling; 5y monthly-stijl
- **Type beta:** 5y monthly (sector/adjusted)
- **Country risk premium %:** 1,30 (gewogen voor ~37% internationale omzet, incl. hoog-risico Wit-Rusland/Servië)
- **Size premium %:** null (large cap, marktkap >EUR 2 mrd)
- **Cost of equity %:** 8,28
- **Schuldkosten na belasting %:** 2,77 (3,6% pre-tax × (1−0,23))
- **E/V gewicht %:** 89,6
- **D/V gewicht %:** 10,4 (op basis van financiële schuld excl. leases EUR 754 mln)
- **WACC %:** 7,71
- **Sector WACC % (referentie Damodaran):** ~6-7 (West-Europese telecom); A1 hoger door CRP
- **Illiquiditeitskorting %:** null (large cap; free float beperkt maar handelsvolume voldoende — risico via WACC verwerkt i.p.v. aparte korting)

### DCF model-specs
- **Model type:** 2-fase (5+5 jaar) + Gordon-terminal
- **FCF-definitie:** FCFF (genormaliseerde economische vrije kasstroom, na leasekosten om dubbeltelling van het IFRS-16-leasevoordeel te vermijden)
- **Basis FCF:** 520 (genormaliseerd; 2025-economische FCF was 596 maar capex was dat jaar onder guidance)
- **Basis FCF na SBC:** 520 (SBC niet materieel)
- **FCF-type:** Genormaliseerde economische FCF ~520 mln (mid-cyclus, na lease)
- **Groei fase 1 %:** 1,0 (basis; jaar 1-5)
- **Groei fase 2 %:** 1,0 (basis; jaar 6-10)
- **Terminal groei %:** 1,5
- **Terminal methode:** Gordon growth (gecross-checkt met exit multiple)
- **Exit multiple gebruikt:** ~5-6x EV/EBITDAaL
- **Bron exit multiple:** Sector-mediaan West-Europese telecom over cyclus
- **Terminal value Gordon growth:** ~5.880 (PV-component basis)
- **Terminal value exit multiple:** ~9.940 (6x EBITDAaL 1.656) — referentie
- **Terminal value % van totaal:** 55 (basis) — onder de 75%-grens, geloofwaardig
- **Terminal implied EV/EBITDA:** ~5,1x (basis EV 8.490 / EBITDAaL 1.656) — redelijk vs. huidige ~5,0x
- **Terminal groei consistentie:** Terminal groei 1,5% vereist bij ROIC ~10% een herinvesteringsvoet van ~15% — ruim plausibel voor een volwassen telco en onder de nominale BBP-groei van Oostenrijk/eurozone.
- **Mid-year convention:** true
- **Aandelen uitstaand (mln):** 664,08
- **Nettoschuld huidig:** 150 (financiële nettoschuld excl. leases; leases zitten reeds in de na-lease FCF-basis)

### DCF-toelichting
De waardering modelleert genormaliseerde, na-lease vrije kasstroom (FCFF) verdisconteerd tegen een WACC van 7,71% met mid-year-conventie. Een bewuste keuze: A1's door S&P berekende FCF (~EUR 977 mln 2025) negeert de jaarlijkse leasebetalingen van ~EUR 330 mln voor onder meer de teruggehuurde EuroTeleSites-masten; die behandeling zou de waarde fors overschatten. Daarom is de economische, na-lease kasstroom (~EUR 520-596 mln) als basis genomen en daarmee consistent de financiële nettoschuld (excl. leases, ~nul) afgetrokken. De terminal value is 55% van de totale waarde — ruim onder de 75%-grens — en de impliciete terminal EV/EBITDAaL (~5,1x) sluit aan bij de huidige marktmultiple, wat de aannames geloofwaardig maakt. Telecom is een defensieve, niet-cyclische sector, dus geen cyclus-normalisatie van FCF nodig; wel is de capex naar een mid-cyclus-niveau (~EUR 850 mln) genormaliseerd omdat 2025 een capex-licht jaar was.

### 5-jaars projectie

| Jaar | Omzet | Omzetgroei % | EBIT | EBIT-marge % | NOPAT | Capex | ΔNWC | SBC | FCF |
|---|---|---|---|---|---|---|---|---|---|
| 2026 | 5.700 | 2,2 | 970 | 17,0 | 747 | 850 | -20 | 0 | 525 |
| 2027 | 5.800 | 1,8 | 986 | 17,0 | 759 | 850 | -15 | 0 | 530 |
| 2028 | 5.880 | 1,4 | 1.000 | 17,0 | 770 | 855 | -15 | 0 | 536 |
| 2029 | 5.950 | 1,2 | 1.012 | 17,0 | 779 | 860 | -10 | 0 | 541 |
| 2030 | 6.010 | 1,0 | 1.022 | 17,0 | 787 | 865 | -10 | 0 | 546 |

### Scenarios (3 stuks — exact deze labels)

| Scenario | FCF-groei % | WACC % | Fair value | Upside % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | -1,5 | 8,21 | 8,81 | -10 | 30 |
| Basis | 1,0 | 7,71 | 12,56 | 28 | 50 |
| Optimistisch | 2,5 | 7,21 | 16,00 | 63 | 20 |

- **Kansgewogen fair value:** 12,12

### Reverse DCF
- **Impliciete groei %:** -0,3 (wat de huidige koers van EUR 9,83 inprijst aan eeuwigdurende FCF-groei)
- **Historische FCF CAGR %:** ~8,5 (economische FCF 2021-2025; vertekend door capex-timing)
- **Consensus groei % (analisten):** ~1-2 (omzet 2-3%/jaar guidance, vlakke EPS)
- **Interpretatie:** De markt prijst tegen de huidige koers een lichte eeuwigdurende krimp van de vrije kasstroom in (~−0,3%). Dat is pessimistischer dan zowel de bedrijfs-guidance (2-3% omzetgroei) als de historische trend. Met andere woorden: de markt rekent op stagnatie of milde achteruitgang, terwijl het bedrijf nog steeds groeit en delevereert. Dit suggereert dat het aandeel fundamenteel goedkoop is — maar de korting weerspiegelt ook reële, niet-operationele zorgen (free float, governance, Wit-Rusland) die de markt niet zomaar zal loslaten.

### EPV (Bruce Greenwald)
- **Genormaliseerde EBIT-marge %:** 17,0
- **Genormaliseerde NOPAT:** 730
- **Maintenance capex:** 850
- **Adjusted earnings power:** 730 (steady-state NOPAT; herinvestering compenseert afschrijving)
- **EPV per aandeel:** 14,03
- **Groeipremie %:** -11 (de DCF-basiswaarde van 12,56 ligt onder de EPV van 14,03)

### Andere methoden
- **DDM uitgevoerd?** false (dividend ~45% van FCF; DCF/EPV dekken de waardering afdoende)
- **SOTP uitgevoerd?** false (geen conglomeraat; geïntegreerde telco)

### Synthese fair value
- **Bandbreedte laag:** 8,81
- **Bandbreedte centraal:** 12,56
- **Bandbreedte hoog:** 16,00
- **Methode-gewichten:**
  - DCF %: 50
  - EPV %: 30
  - Multiples %: 20
- **Margin of safety vereist %:** 25
- **Koopniveau** (synthese × (1 − MOS)): ~9,7
- **Synthese-toelichting:** De drie methoden convergeren rond EUR 12-14: de DCF-basis geeft 12,56, de EPV (waarde zonder groei) 14,03, en een peer-multiple van ~6x EV/EBITDAaL ~12,3. Opvallend is dat de EPV bóven de DCF-basiswaarde ligt — een teken dat de markt niet alleen niet voor groei betaalt, maar zelfs een no-growth-waarde negeert. De gewogen synthese (~EUR 12,9) impliceert een opwaarts potentieel van ~30%. Gegeven de kwaliteit van de data (hoog), maar de reële governance- en landenrisico's, is een margin of safety van 25% gepast: een koopniveau van ongeveer EUR 9,7 of lager. Op de huidige koers van EUR 9,83 zit het aandeel net op dat koopniveau — interessant, maar zonder ruime veiligheidsmarge, vandaar HOLD.

### Gevoeligheid (DCF)
- **FCF-groei ↔ WACC matrix:**
  - WACC range: [6,5%, 7,0%, 7,5%, 8,0%, 8,5%, 9,0%]
  - Groei range: [-1%, 0%, 1%, 2%, 3%]
  - Matrix (5 rijen × 6 kolommen, fair value per aandeel):

| g \ WACC | 6,5% | 7,0% | 7,5% | 8,0% | 8,5% | 9,0% |
|---|---|---|---|---|---|---|
| -1% | 10,44 | 9,80 | 9,23 | 8,73 | 8,27 | 7,87 |
| 0% | 12,21 | 11,35 | 10,60 | 9,95 | 9,37 | 8,86 |
| 1% | 14,61 | 13,41 | 12,39 | 11,52 | 10,76 | 10,10 |
| 2% | 16,85 | 15,32 | 14,05 | 12,97 | 12,05 | 11,25 |
| 3% | 18,27 | 16,59 | 15,20 | 14,01 | 13,00 | 12,12 |

---

## 13. Databronnen

### Bronnen-hiërarchie
- Jaarverslag PDF / IR-pagina → HOOG
- Beursmelding / persbericht → HOOG
- Aggregator (S&P Global via StockAnalysis) → AGGREGATOR

### Financiële bronnen (10 jaar historie)

| Jaar | Bron | URL | Betrouwbaarheid |
|---|---|---|---|
| 2016 | — | — | — |
| 2017 | — | — | — |
| 2018 | — | — | — |
| 2019 | — | — | — |
| 2020 | — | — | — |
| 2021 | A1 Annual Financial Report 2021 + S&P Global | https://a1.com/investor-relations/results-center/annual-financial-reports/ | HOOG |
| 2022 | A1 Combined Report 2022 + S&P Global | https://a1.group/wp-content/uploads/sites/6/2023/09/A1_Combined_Report_2022_EN.pdf | HOOG |
| 2023 | Telekom Austria Financial Report 2023 + S&P Global | https://a1.group/wp-content/uploads/sites/6/2024/03/Telekom-Austria-AG-Financial-Report-2023.pdf | HOOG |
| 2024 | A1 Group Results Report 2024 (geconsolideerd) | https://a1.group/wp-content/uploads/sites/6/2025/03/A1-Group_results-report-2024.pdf | HOOG |
| 2025 | A1 Group FY2025-persbericht + Q3-2025 Earnings Update | https://a1.group/wp-content/uploads/sites/6/2025/10/A1-Group-%E2%80%93-Earnings-Update-Q3-2025.pdf | HOOG |

**NB:** de 5 meest recente jaren (2021-2025) zijn HOOG. 2016-2020 zijn niet geverifieerd binnen deze run en blijven leeg.

### Jaarverslagen geraadpleegd

| Jaar | Bron | URL |
|---|---|---|
| 2024 | A1 Group Results Report 2024 (geconsolideerd) | https://a1.group/wp-content/uploads/sites/6/2025/03/A1-Group_results-report-2024.pdf |
| 2024 | Telekom Austria AG Results Report 2024 (standalone, ter onderscheiding) | https://a1.group/wp-content/uploads/sites/6/2025/03/Telekom-Austria-AG_results-report-2024-2.pdf |
| 2025 (9M) | A1 Group Earnings Update Q3/Q1-Q3 2025 | https://a1.group/wp-content/uploads/sites/6/2025/10/A1-Group-%E2%80%93-Earnings-Update-Q3-2025.pdf |

### Beursmeldingen geraadpleegd

| Datum | Omschrijving | URL |
|---|---|---|
| 2026-Q1 | A1 Group Q1 2026-resultaten | https://www.eqs-news.com/news/corporate-news/a1-telekom-austria-group-q1-2026-results/e25ad4bb-c22f-4a24-9dc1-df1d3a972747 |
| 2026-02 | A1 Group Q4/FY2025-resultaten | https://www.marketscreener.com/news/telekom-austria-very-solid-financial-year-2025-for-a1-group-with-revenue-and-ebitda-growth-ce7e5adcdd81f62c |
| 2025-02 | A1 Group FY2024-resultaten | https://newsroom.a1.group/news-solid-financial-year-2024-for-a1-group-31-total-revenue-growth-51-ebitda-growth?id=212758 |
| 2023-09 | EuroTeleSites-afsplitsing voltooid | https://www.marketscreener.com/quote/stock/EUROTELESITES-AG-159571668/news/Telekom-Austria-AG-completed-the-Spin-Off-of-EuroTeleSites-AG-44943873/ |

### IPO-prospectus
- **Geraadpleegd?** false
- **URL:** —
- **Pre-IPO data beschikbaar?** false
- **Pre-IPO bron:** — (IPO 2000; geen pre-IPO financial engineering geconstateerd; relevante structuurwijziging = América Móvil-overname 2014)

### Non-GAAP
- **Gebruikt?** true
- **Toelichting:** A1's EBITDA (vóór lease, EUR 2.062 mln) vs. EBITDAaL (na lease, ~EUR 1.656 mln) vs. S&P-gestandaardiseerde EBITDA (EUR 1.630 mln). In de waardering is bewust de economische na-lease kasstroom gebruikt om dubbeltelling van het IFRS-16-leasevoordeel te voorkomen.

### Ontbrekende data (eerlijke lijst)
- Financiële historie 2016-2020 niet geverifieerd (geen geopende bron binnen deze run); deze jaren blijven leeg.
- IPO-koers 2000 niet geverifieerd.
- Exact marktaandeelcijfer per concurrent en per land niet betrouwbaar gekwantificeerd.
- Precieze segmentuitsplitsing omzet per CEE-land en exact Wit-Rusland-aandeel van EBITDA niet op cijferniveau geverifieerd.
- TAM/SAM niet betrouwbaar te kwantificeren voor een verzadigde nutsmarkt.
- Insider open-markttransacties op individueel niveau (CEO/CFO) niet gevonden; América Móvil verhoogde wel zijn belang (2023), maar dat is een grootaandeelhouder-transactie, geen klassieke insider-aankoop.
- Onderscheid tussen bedrijfs-EBITDA en S&P-EBITDA: in de tabellen is de S&P-gestandaardiseerde EBITDA gebruikt voor interne consistentie van de reeks; de hogere bedrijfs-EBITDA is in de toelichting benoemd.

### Peildatum analyse
- 2026-06-02

---

## 14. Volledige bronnen-lijst

| Titel | URL | Type |
|---|---|---|
| A1 Group — Investor Relations / Results Center | https://a1.group/investor-relations/results-center/ | beurswebsite |
| A1 Group Results Report 2024 (geconsolideerd) | https://a1.group/wp-content/uploads/sites/6/2025/03/A1-Group_results-report-2024.pdf | jaarverslag |
| Telekom Austria AG Results Report 2024 (standalone) | https://a1.group/wp-content/uploads/sites/6/2025/03/Telekom-Austria-AG_results-report-2024-2.pdf | jaarverslag |
| A1 Group Earnings Update Q3/Q1-Q3 2025 | https://a1.group/wp-content/uploads/sites/6/2025/10/A1-Group-%E2%80%93-Earnings-Update-Q3-2025.pdf | beursmelding |
| A1 Group Q1 2026-resultaten | https://www.eqs-news.com/news/corporate-news/a1-telekom-austria-group-q1-2026-results/e25ad4bb-c22f-4a24-9dc1-df1d3a972747 | beursmelding |
| Very solid financial year 2025 for A1 Group | https://www.marketscreener.com/news/telekom-austria-very-solid-financial-year-2025-for-a1-group-with-revenue-and-ebitda-growth-ce7e5adcdd81f62c | nieuwsartikel |
| Solid financial year 2024 for A1 Group | https://newsroom.a1.group/news-solid-financial-year-2024-for-a1-group-31-total-revenue-growth-51-ebitda-growth?id=212758 | beursmelding |
| StockAnalysis (S&P Global) — Income Statement TKA | https://stockanalysis.com/quote/vie/TKA/financials/ | databron |
| StockAnalysis (S&P Global) — Balance Sheet TKA | https://stockanalysis.com/quote/vie/TKA/financials/balance-sheet/ | databron |
| StockAnalysis (S&P Global) — Cash Flow TKA | https://stockanalysis.com/quote/vie/TKA/financials/cash-flow-statement/ | databron |
| StockAnalysis (S&P Global) — Ratios TKA | https://stockanalysis.com/quote/vie/TKA/financials/ratios/ | databron |
| MarketScreener — TKA analistenconsensus | https://www.marketscreener.com/quote/stock/TELEKOM-AUSTRIA-AG-6492023/consensus/ | analistenrapport |
| ÖBAG-portfolio A1 Telekom (aandeelhouder) | https://oebag.gv.at/en/organisation/portfolio/portfolio-detail-a1/ | beurswebsite |
| América Móvil verhoogt belang in Telekom Austria | https://www.marketscreener.com/quote/stock/TELEKOM-AUSTRIA-AG-6492023/news/Telekom-Austria-America-M-vil-increases-the-overall-shareholding-in-Telekom-Austria-AG-to-56-55-44403006/ | nieuwsartikel |
| EuroTeleSites-afsplitsing voltooid | https://www.marketscreener.com/quote/stock/EUROTELESITES-AG-159571668/news/Telekom-Austria-AG-completed-the-Spin-Off-of-EuroTeleSites-AG-44943873/ | nieuwsartikel |
| Management board contracts (Plater/Arnoldner/Wallner) | https://www.eqs-news.com/news/adhoc/telekom-austria-ag-extension-of-the-management-board-contracts-of-alejandro-plater-and-thomas-arnoldner-change-of-roles/1811223 | beursmelding |
| Austria 10Y government bond yield | https://www.worldgovernmentbonds.com/country/austria/ | databron |

---

## 15. Update-historie

| Datum | Versie | Wijziging |
|---|---|---|
| 2026-06-02 | 1.0 | Eerste publicatie |

---

## Opmerkingen voor Claude Code

1. **EBITDA-definitiekwestie (belangrijk voor validator):** De tabellen in sectie 3 gebruiken de **S&P Global-gestandaardiseerde EBITDA** (FY2025 = 1.630) voor interne reeksconsistentie. A1 zelf rapporteert een hogere EBITDA (vóór leases, EUR 2.062 mln) en een EBITDAaL (na leases, ~EUR 1.656 mln). Als de validator een EV/EBITDA-cross-check doet, let op welke definitie wordt aangenomen — de DCF/EPV gebruiken bewust de na-lease economische basis. Mogelijke inconsistentie-melding kan hierdoor ontstaan; dit is een definitiekwestie, geen fout.

2. **FCF-basiskeuze:** Bewust EUR 520 mln genormaliseerde na-lease FCF als DCF-basis, NIET de S&P-FCF (~977) die leasebetalingen negeert. Als de stage-2-DCF-herberekening de S&P-FCF zou gebruiken, springt de fair value naar ~EUR 25-29 (onrealistisch). Houd de na-lease basis aan.

3. **Nettoschuld-dubbelzinnigheid:** Financiële nettoschuld excl. leases ≈ nul (EUR 150 mln gebruikt); inclusief leases ≈ EUR 1.848 mln (S&P). In de DCF is EUR 150 mln afgetrokken omdat de FCF-basis al na-lease is. Consistent paren is essentieel: na-lease FCF ↔ financiële nettoschuld excl. leases.

4. **Standalone vs. geconsolideerd:** De eerste resultatenrapport-PDF die ik opende was de **standalone Telekom Austria AG** (holding, omzet EUR 45 mln). Die is NIET gebruikt. Alle cijfers zijn de **geconsolideerde A1 Group**. Verwar deze niet bij verificatie.

5. **Graham number:** De ruwe Graham number (√(22,5 × 0,92 × 8,06)) ≈ 12,9; ik heb een conservatievere ~9,9 genoteerd op genormaliseerde basis. Stage-2 mag dit herberekenen — de Graham-score (4) volgt uit P/E≤15 én P/B≤1,5 (MOS<30%), niet uit de Graham number zelf.

6. **2016-2020 ontbreekt:** Bewust leeg gelaten conform brondiscipline. Indien gewenst kunnen de pre-2021 A1-jaarverslagen (beschikbaar op a1.com/investor-relations/results-center/annual-financial-reports/) alsnog worden geopend om de reeks naar 10 jaar te completeren.

7. **Insider transactions:** Geen klassieke open-markt insider-aankopen door CEO/CFO gevonden; América Móvil's belangverhoging (2023) is wel een vertrouwenssignaal van de grootaandeelhouder, maar valt buiten de standaard insider-transactietabel.
