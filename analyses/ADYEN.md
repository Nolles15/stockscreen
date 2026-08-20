# Research: ADYEN — Adyen N.V.

## Bronnen-inventaris (Stap 0.5)

*Opgesteld vóór het invullen van één numeriek veld, conform METHODE.md Stap 0.5.*

**Werkwijze en een structurele blokkade.** Adyen host al zijn jaarverslagen en
shareholder letters op `brand.adyen.com`. Dat domein is robots.txt-geblokkeerd; geen
enkele PDF is daar op te halen. De overzichtspagina `investors.adyen.com/financials` is
wél leesbaar en is geopend om per boekjaar de juiste documentpagina te vinden. De
PDF's zelf zijn vervolgens door Janco gedownload en aangeleverd (haallijst, gevraagd
direct na deze inventaris en vóór het invullen van de tabellen). Alle tien half-
jaarrapporten plus het jaarverslag 2025 zijn daarmee **daadwerkelijk geopend en
uitgelezen** met `pdftotext`.

Adyen publiceert per halfjaar een shareholder letter met de **volledige verkorte
geconsolideerde jaarrekening over dat halfjaar** (winst- en verliesrekening, balans per
periode-einde, mutatieoverzicht eigen vermogen, kasstroomoverzicht). Een boekjaar is
dus exact de som van de H1- en de H2-brief; dat is optellen van twee geverifieerde
primaire bronnen, geen schatting. Waar het jaarverslag 2025 of een officieel persbericht
het jaartotaal óók noemt, is dat als kruiscontrole gebruikt — alle controles klopten
(zie "Kruiscontroles" onderaan deze sectie).

```
Jaar 2026 (H1) — HOOG
  Bron: Adyen H1 2026 Shareholder Letter (PDF, gepubliceerd 13 augustus 2026)
  URL:  https://investors.adyen.com/financials/h1-2026-c2a1a
  Daadwerkelijk geopend: ja (via haallijst; brand.adyen.com robots-geblokkeerd)
  Cijfers overgenomen: netto-omzet, EBITDA, EBIT, D&A, nettowinst, EPS basic/diluted,
                       verwerkt volume, take rate, capex, leasebetalingen, vrije kasstroom,
                       balans 30-6-2026, kasstroomoverzicht, betaalde winstbelasting,
                       aandelencompensatie, FTE per kantoor, segment- en regio-omzet,
                       guidance 2026, prepaid acquisition consideration
  Cijfers NIET overgenomen: (geen)

Jaar 2025 — HOOG
  Bron 1: Adyen Annual Report and Consolidated Financial Statements 2025 (PDF)
  URL:    https://investors.adyen.com/financials/2025
  Bron 2: Adyen H2 2025 Shareholder Letter (PDF)
  URL:    https://investors.adyen.com/financials/h2-2025-4r9rc
  Bron 3: Adyen H1 2025 Shareholder Letter (PDF)
  URL:    https://investors.adyen.com/financials/h1-2025
  Daadwerkelijk geopend: alle drie
  Cijfers overgenomen: volledige FY-winst- en verliesrekening, balans, kasstroomoverzicht,
                       eigen vermogen, betaalde belasting, capex, leases, SBC, FTE,
                       segmentomzet, ESG-data, beloning bestuur, regulatoir vermogen
  Cijfers NIET overgenomen: (geen)

Jaar 2024 — HOOG
  Bron 1: Adyen Annual Report 2025 (vergelijkende kolom FY2024) — zie boven
  Bron 2: Adyen H1 2025 Shareholder Letter (vergelijkende kolom H1 2024)
  Bron 3: Adyen H2 2025 Shareholder Letter (vergelijkende kolom H2 2024)
  Bron 4: persbericht "Adyen publishes H2 2024 financial results"
  URL:    https://www.adyen.com/press-and-media/adyen-publishes-h2-2024-financial-results
  Daadwerkelijk geopend: alle vier
  Cijfers overgenomen: volledige W&V, balans 31-12-2024, kasstroom, volume, capex, FCF-conversie

Jaar 2023 — HOOG
  Bron 1: Adyen H1 2023 Shareholder Letter (PDF)
  URL:    https://investors.adyen.com/financials/h1-2023
  Bron 2: Adyen H2 2023 Shareholder Letter (PDF)
  URL:    https://investors.adyen.com/financials/h2-2023
  Bron 3: persbericht "Adyen publishes H2 2023 financial results"
  URL:    https://www.adyen.com/press-and-media/adyen-publishes-h2-2023-financial-results
  Daadwerkelijk geopend: alle drie
  Cijfers overgenomen: W&V H1+H2, balans 31-12-2023, kasstroom H1+H2, volume, FTE

Jaar 2022 — HOOG
  Bron 1: Adyen H1 2023 Shareholder Letter (vergelijkende kolom H1 2022)
  Bron 2: Adyen H2 2023 Shareholder Letter (vergelijkende kolom H2 2022)
  Bron 3: persbericht "Adyen publishes h2 2022 financial results"
  URL:    https://www.adyen.com/press-and-media/adyen-publishes-h2-2022-financial-results
  Daadwerkelijk geopend: alle drie
  Cijfers overgenomen: W&V H1+H2, balans 31-12-2022, kasstroom H1+H2, volume, capex, FCF-conversie

Jaar 2021 — HOOG
  Bron 1: Adyen H1 2021 Shareholder Letter (PDF)
  URL:    https://investors.adyen.com/financials/h1-2021
  Bron 2: Adyen H2 2021 Shareholder Letter (PDF)
  URL:    https://investors.adyen.com/financials/h2-2021
  Daadwerkelijk geopend: beide
  Cijfers overgenomen: W&V H1+H2, balans 31-12-2021, kasstroom H1+H2, volume, take rate

Jaar 2020 — HOOG
  Bron 1: Adyen H1 2021 Shareholder Letter (vergelijkende, ge-restatete kolom H1 2020)
  Bron 2: Adyen H2 2021 Shareholder Letter (vergelijkende kolom H2 2020)
  Bron 3: persbericht "Adyen Publishes H2 2020 Financial Results"
  URL:    https://www.adyen.com/press-and-media/adyen-publishes-h2-2020-financial-results
  Daadwerkelijk geopend: alle drie
  LET OP: de oorspronkelijk gepubliceerde H1 2020-cijfers (netto-omzet EUR 279,9 mln,
  EBITDA EUR 140,9 mln) zijn later herzien naar EUR 304,8 mln respectievelijk
  EUR 165,7 mln. De herziene reeks uit de H1 2021-brief is gebruikt; die sluit exact
  aan op het gerapporteerde jaartotaal.

Jaar 2019 — HOOG
  Bron 1: Adyen H1 2019 Shareholder Letter (PDF)
  URL:    https://investors.adyen.com/financials/h1-2019
  Bron 2: Adyen H2 2019 Shareholder Letter (PDF)
  URL:    https://investors.adyen.com/financials/h2-2019
  Daadwerkelijk geopend: beide
  Cijfers overgenomen: W&V H1+H2, balans 31-12-2019, kasstroom H1+H2, volume, FTE

Jaar 2018 — HOOG
  Bron 1: Adyen H1 2019 Shareholder Letter (vergelijkende kolom H1 2018)
  Bron 2: Adyen H2 2019 Shareholder Letter (vergelijkende kolom H2 2018)
  Bron 3: persbericht "Adyen H1 2018 Financial Results"
  URL:    https://www.adyen.com/press-and-media/adyen-h1-2018-financial-results
  Daadwerkelijk geopend: alle drie
  Cijfers overgenomen: W&V H1+H2, balans 31-12-2018, kasstroom H1+H2, volume, FTE
  Kanttekening: de som H1+H2 geeft een netto-omzet van EUR 348,9 mln; het persbericht
  bij de jaarcijfers 2019 noemt voor 2018 EUR 350,5 mln. Het verschil van EUR 1,6 mln
  komt uit een presentatiewijziging. De additieve reeks (348,9) is gebruikt omdat die
  op dezelfde grondslag staat als alle andere jaren.

Jaren 2017, 2016, 2015 — HOOG
  Bron: Adyen N.V. IPO-prospectus, 4 juni 2018
  URL:  https://live.euronext.com/sites/default/files/adyen_prospectus.pdf
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: bruto-omzet, netto-omzet, EBITDA, nettowinst, totale activa,
                       eigen vermogen, operationele kasstroom, capex, personeel 2017,
                       aandelen uitstaand per 31-5-2018, IPO-voorwaarden en
                       aandeelhoudersstructuur vóór de beursgang
  Cijfers NIET overgenomen: EBIT (niet apart gespecificeerd), EPS, balansdetail per post,
                       kasstroomdetail — deze cellen blijven LEEG in de tabellen

Aandelenkapitaal per 30-6-2026 — HOOG
  Bron: AFM-register geplaatst kapitaal
  URL:  https://www.afm.nl/en/sector/registers/meldingenregisters/geplaatst-kapitaal/details?id=196519
  Daadwerkelijk geopend: ja — 31.566.677 gewone aandelen, nominaal EUR 0,01

Aandeelhouders — HOOG (register)
  Bron: AFM-register substantiële deelnemingen, individuele meldingsrecords
  URL:  https://www.afm.nl/en/sector/registers/meldingenregisters/substantiele-deelnemingen
  Daadwerkelijk geopend: ja (Temasek 5-6-2025, BlackRock 13-9-2021 en zeven oudere records).
  Beperking: het zoekformulier is JavaScript-gebaseerd; een volledige actuele lijst
  is langs deze weg niet te genereren. Zie ontbrekende data.

Bankvergunning — HOOG
  Bron: AFM-vergunningenregister, DNB-vergunning kredietinstelling d.d. 25-4-2017
  URL:  https://www.afm.nl/en/sector/registers/vergunningenregisters/beleggingsondernemingen/details?id=E8C39F96-795E-E311-B05D-005056BE011E

Koers, beta, volume — AGGREGATOR
  Bron: StockAnalysis.com quote- en statistics-pagina
  URL:  https://stockanalysis.com/quote/ams/ADYEN/ en /statistics/
  Daadwerkelijk geopend: ja. Koers EUR 972,10 per 13-8-2026 09:01 CET (+6,82%);
  beta 5Y 1,86 per 11-8-2026; 20-daags gemiddeld volume 126.327 stuks.
  LET OP: de door deze bron getoonde enterprise value (EUR 18,78 mrd) is voor Adyen
  onbruikbaar — hij trekt de vólledige kaspositie van EUR 12,4 mrd af, terwijl daarvan
  EUR 8,1 mrd merchant-gelden zijn die Adyen niet toebehoren. De EV in dit rapport is
  zelf berekend op basis van de eigen nettokas.

Macro-invoeren WACC — HOOG
  Rf:  Nederlandse 10-jaars staatsobligatie 3,248% per 13-8-2026
       https://tradingeconomics.com/netherlands/government-bond-yield
  ERP: Damodaran implied ERP 4,28% per 1-8-2026
       https://pages.stern.nyu.edu/~adamodar/New_Home_Page/home.htm
  CRP: Damodaran country risk premiums, tabel 5-1-2026
       https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html
  Sector-WACC ter referentie: 9,34% (Software System & Application, VS, januari 2026)
       https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/wacc.html
```

**Kruiscontroles die zijn uitgevoerd en klopten.** FY2025 netto-omzet uit H1+H2
(EUR 2.364,191 mln) tegen het jaarverslag 2025 (EUR 2.364,191 mln): exact. FY2025 EBITDA
uit EBIT+D&A (EUR 1.245,749 mln) tegen de non-IFRS-noot in het jaarverslag
(EUR 1.245,749 mln): exact. FY2025 betaalde winstbelasting uit H1+H2 (EUR 264,328 mln)
tegen het jaarverslag (EUR 264,328 mln): exact. FY2024 idem (EUR 320,543 mln): exact.
FY2025 operationele kasstroom uit H1+H2 (EUR 1.030,423 mln) tegen het jaarverslag
(EUR 1.030,422 mln): afrondingsverschil van EUR 1.000. FY2024 nettowinst uit het
mutatieoverzicht eigen vermogen (EUR 925,163 mln) tegen de W&V: exact. De nettowinstreeks
2021-2025 komt bovendien exact overeen met de reeks bij StockAnalysis, wat bevestigt dat
er geen herzieningen zijn gemist.

**Wat er ondanks alles ontbreekt.** Voor 2015-2017 geeft het prospectus alleen
kerncijfers, geen volledige balans- of kasstroomdetails; die cellen blijven leeg. Voor
2015-2016 is geen verwerkt volume gepubliceerd. Een actueel free-floatpercentage en een
volledige actuele lijst van substantiële deelnemingen zijn niet uit een primaire bron te
halen. Zie sectie 13 voor de volledige lijst.

---

## Metadata
- **Ticker (bare):** ADYEN
- **Yahoo symbol:** ADYEN.AS
- **Exchange:** ENXTAM (Euronext Amsterdam)
- **Sector (GICS-achtig):** Financieel
- **Industrie:** Betaaltechnologie / merchant acquiring (licensed credit institution)
- **Land:** Nederland
- **Peildatum analyse:** 2026-08-13
- **Koers op peildatum:** 972,10
- **Valuta:** EUR
- **Marktkapitalisatie:** EUR 30,7 mld
- **Marktkap in mln (lokale valuta):** 30686
- **Free float pct:** — (niet verifieerbaar; zie sectie 13)
- **Indexlidmaatschap:** AEX (gewicht 3,13% per 31-3-2026)
- **Domein:** adyen.com

---

## 1. Executive summary

- **Kernthese**: Adyen bouwt en exploiteert één zelfgeschreven betaalplatform waarop
grote internationale bedrijven hun betalingen online, in de winkel en in apps afhandelen.
Waar concurrenten hun aanbod door overnames aan elkaar hebben geknoopt, draait alles bij
Adyen op dezelfde codebase en op eigen bankvergunningen. Dat levert een kostenvoordeel op
dat zichtbaar is in de cijfers: over 2025 bleef 53% van de netto-omzet als EBITDA over,
tegenover 2,6% bij Worldline en 13,1% bij Nexi. De omzet groeide tussen 2018 en 2025 met
gemiddeld 31,4% per jaar en in de eerste helft van 2026 met 21% op constante-valutabasis.
De groei komt vrijwel volledig uit bestaande klanten die een steeds groter deel van hun
betaalvolume verleggen: het aandeel van Adyen bij een klant loopt van onder 20% in de
eerste jaren naar boven 40% na tien jaar. Het bedrijf heeft geen rentedragende schuld en
bijna EUR 4,9 mld eigen kas. Daar staat tegenover dat Stripe met een verwerkt volume van
USD 1,9 biljoen inmiddels ruim groter is dan Adyen en dat de tarieven per transactie al
zes jaar dalen, van 21,9 naar 16,2 basispunten.

- **Oordeel**: **HOLD**
- **Fair value basis** (basisscenario, EUR): 1.062,04
- **Fair value kansgewogen**: 1.008,64
- **EPV per aandeel** (Earnings Power Value, zonder groeipremie): 444,65
- **Upside pct**: 9,3
- **Fair value scenarios**:

| Scenario | Fair value | Upside % | FCF groei % | WACC % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 590,70 | −39,2 | 7,69 | 12,28 | 30 |
| Basis | 1.062,04 | +9,3 | 17,75 | 11,28 | 45 |
| Optimistisch | 1.414,04 | +45,5 | 20,83 | 10,53 | 25 |

- **Reverse-DCF impliciete groei pct**: 15,97
- **Grootste kans**: De verschuiving van betaalverwerker naar volledige financiële
infrastructuur — banking, issuing, capital, billing en loyalty — verbreedt de omzet per
klant en verlengt de groeiperiode ver voorbij het huidige tarievenpad.
- **Grootste risico**: Stripe verwerkt inmiddels meer volume dan Adyen en drukt samen met
de tariefstaffels de take rate; blijft die daling doorgaan zonder compensatie uit
financiële producten, dan verdampt de helft van de waarde in dit model.

---

## 2. Bedrijfsprofiel

- **Beschrijving**: Adyen is een Nederlandse betaalonderneming met een volwaardige
bankvergunning die het volledige betaalproces voor grote bedrijven afhandelt. Wie online
of in een winkel afrekent bij Spotify, Uber, McDonald's, H&M, eBay of Microsoft, loopt
er een reële kans op dat de transactie via Adyen loopt. Traditioneel is die keten
versnipperd: een payment gateway neemt de betaling aan, een risicopartij beoordeelt fraude,
een acquirer int het geld bij de kaartmaatschappij en een bank maakt het over naar de
verkoper. Elk van die schakels heeft een eigen contract, eigen data en eigen storingen.
Adyen heeft die vier rollen in één zelfgebouwd systeem samengevoegd en houdt daarnaast
eigen bankvergunningen aan in Europa, de Verenigde Staten en het Verenigd Koninkrijk,
zodat het geld ook zelf kan vasthouden en uitbetalen. Voor de klant betekent dat één
integratie, één rapportage over alle landen en kanalen, en de mogelijkheid om een klant
die online iets kocht te herkennen wanneer die de winkel binnenloopt. Adyen verdient aan
elke transactie: een vaste vergoeding per betaling plus een percentage over het bedrag,
waarbij de doorbelaste kosten van kaartmaatschappijen en banken eruit worden gefilterd.
Wat overblijft — de netto-omzet — is de maatstaf waarop het bedrijf stuurt. Daarnaast
ontvangt Adyen rente over de miljardenbedragen die tussen betaling en uitbetaling
tijdelijk op de eigen balans staan.

- **Geschiedenis**: Adyen werd in 2006 in Amsterdam opgericht door Pieter van der Does en
Arnout Schuijff. Beiden kwamen van Bibit, een Nederlandse betaaldienstverlener die aan
Royal Bank of Scotland was verkocht en daar de basis vormde van de online-tak van
Worldpay. Zij wilden het opnieuw doen, maar dan zonder de technische erfenis: één platform,
helemaal zelf gebouwd, wereldwijd bruikbaar. De naam betekent volgens Adyen zelf "opnieuw
beginnen". De eerste acht jaar groeide het bedrijf vrijwel zonder extern kapitaal; het was
al winstgevend in 2011, toen Index Ventures als eerste durfkapitalist instapte. Groupon
was de eerste grote klant. In december 2014 volgde een ronde van USD 250 mln onder leiding
van General Atlantic bij een waardering van USD 1,5 mrd; in september 2015 investeerde
Iconiq Capital bij USD 2,3 mrd. Een pan-Europese acquiring-licentie kwam in 2012, een
Braziliaanse in 2016 en op 25 april 2017 verleende De Nederlandsche Bank een volwaardige
bankvergunning — de stap die Adyen structureel onderscheidde van softwarebedrijven die
voor de afwikkeling van een bank afhankelijk blijven. In januari 2018 won Adyen eBay als
primaire betaalpartner, ten koste van PayPal. Op 13 juni 2018 volgde de beursgang op
Euronext Amsterdam tegen EUR 240 per aandeel, waarbij 3.537.754 bestaande aandelen
(12,0% van het kapitaal) werden verkocht voor EUR 849 mln; het bedrijf zelf haalde geen
geld op. Het aandeel verdrievoudigde bijna op de eerste handelsdag. Daarna volgden jaren
van 40 tot 50 procent omzetgroei, tot 17 augustus 2023: de omzetgroei viel terug naar 21%,
de EBITDA daalde 10% en het aandeel verloor op één dag 39% — circa EUR 18 mrd
beurswaarde. De oorzaak was een combinatie van Amerikaanse klanten die op prijs gingen
sturen en een aanwervingsgolf van 551 mensen in één halfjaar. Op de beleggersdag van
8 november 2023 verlaagde Adyen zijn groeidoelstelling en temperde het de aanwervingen;
op 11 november 2025 verhoogde het de margedoelstelling naar boven 55% in 2028. In 2026
deed Adyen zijn eerste twee overnames ooit: Talon.One voor EUR 750 mln en Orb voor
USD 335 mln, beide afgerond op 1 juli 2026.

- **Bedrijfsmodel**: Adyen rekent per transactie een vaste processing fee plus een
percentage van het bedrag, en filtert de doorbelaste scheme- en interchangekosten eruit.
De resterende netto-omzet gedeeld door het verwerkte volume is de take rate: 16,2
basispunten in H1 2026, dalend doordat grote klanten in lagere volumestaffels vallen.
Vrijwel alle omzet is terugkerend en volumegedreven; er zijn geen abonnementen of
licentiekosten. Daarbovenop komt een groeiende laag financiële producten — Capital
(voorschotten aan platformklanten), Issuing (kaarten uitgeven), Accounts (rekeningen)
en sinds 2026 billing en loyalty — die per klant meer omzet oplevert bij hetzelfde
volume. Ten slotte houdt Adyen tussen inning en uitbetaling miljarden aan merchant-gelden
aan; de rente daarop leverde over de afgelopen twaalf maanden EUR 272,1 mln op.

- **IPO-context**: Adyen ging op 13 juni 2018 naar Euronext Amsterdam tegen EUR 240 per
aandeel, wat het bedrijf op EUR 7,1 mld waardeerde. Het was volledig een verkoop door
bestaande aandeelhouders: 3.537.754 aandelen voor EUR 849 mln, met een greenshoe van
407.608 stuks. Adyen zelf ontving geen opbrengst en gebruikte de beursgang dus niet om
schulden af te lossen of de balans op te schonen. Er was ook geen schuld: het bedrijf was
al sinds 2011 winstgevend en had zich nooit volgeladen. Sindsdien is het aantal aandelen
met slechts 7,2% toegenomen, van 29.445.458 naar 31.566.677, uitsluitend door
personeelsregelingen. Er is nooit een emissie of een inkoopprogramma geweest.

- **Klantprofiel**: Adyen bedient uitsluitend zakelijke klanten en richt zich bewust op de
bovenkant van de markt: grote internationale ondernemingen en softwareplatforms. Het
klantenbestand is opvallend kleverig — de volumeklanten die in een periode wegvallen
vertegenwoordigen al jaren minder dan 1% van het volume, ook in 2025. Tegelijk is de
groei geconcentreerd: ongeveer 300 klanten zijn goed voor circa 60% van de groei, al is
dat een verbetering ten opzichte van ruim 70% drie jaar geleden. Binnen Unified Commerce
verwerken 486 klanten op schaal zowel in de winkel als online; binnen Platforms zijn er
37 klanten met meer dan EUR 1 mrd jaarvolume, tegen 32 een jaar eerder. Er is geen
individuele klant die als afzonderlijk concentratierisico wordt gerapporteerd, maar één
grote klant vertekende in 2025 wel de volumegroei.

- **Oprichtingsjaar**: 2006
- **IPO-datum**: 2018-06-13
- **IPO-koers** (EUR): 240,00
- **Personeel** (FTE): 5.020 per 30-6-2026 (4.771 per 31-12-2025)
- **Landen actief**: 30 kantoren in 23 landen (per augustus 2026); 29 kantoren in 21
  landen per ultimo 2025
- **Klantconcentratie**: circa 300 klanten leveren ongeveer 60% van de omzetgroei; het
  volumeverloop bedraagt al jaren minder dan 1%. Adyen publiceert geen omzetaandeel van
  individuele klanten. In 2025 vertekende één grote klant de volumegroei: inclusief die
  klant groeide het volume 8%, exclusief 21%.

### Geografische spreiding (omzet)
Netto-omzet naar factuurland, H1 2026.

| Regio | Omzet % | Valuta-exposure |
|---|---|---|
| EMEA | 55 | EUR, GBP, PLN, SEK, AED |
| Noord-Amerika | 27 | USD, CAD |
| Azië-Pacific | 11 | SGD, AUD, JPY, CNY, INR |
| Latijns-Amerika | 7 | BRL, MXN |

**Toelichting geografie**: Adyen rapporteert in euro terwijl bijna de helft van de omzet
buiten de eurozone wordt gefactureerd, en dat is in 2026 zichtbaar geworden: de netto-omzet
groeide 19% zoals gerapporteerd maar 21% op constante valuta, en in Noord-Amerika was het
verschil zeven procentpunten (23% versus 30%). Het bedrijf dekt de dagelijkse
valutapositie uit de handelsactiviteit actief af, maar de omrekening van buitenlandse
omzet naar euro wordt niet afgedekt. Het jaarverslag noemt het Britse pond de grootste
netto-balanspositie, doordat lokale betaalrails liquiditeit in die valuta vereisen. Er is
een natuurlijke buffer: kosten worden grotendeels gemaakt in de landen waar ook wordt
gefactureerd.

### Segmenten
Netto-omzet per commerciële pijler, H1 2026.

| Naam | Omzet % | Beschrijving |
|---|---|---|
| Digital | 55,2 | Online-eerst bedrijven: abonnementsdiensten, content, reizen en mobiliteit. De oudste en grootste pijler, EUR 719,7 mln (+13% j-o-j), maar ook de traagst groeiende doordat de grootste klanten al diep in de volumestaffels zitten. |
| Unified Commerce | 32,1 | Retailers, horeca en food & beverage die winkel en webshop op één systeem afhandelen. EUR 417,7 mln (+25%); 486 klanten verwerken op schaal via beide kanalen. |
| Platforms | 12,7 | Softwarebedrijven die betalingen en financiële producten in hun eigen aanbod inbouwen. EUR 165,5 mln (+37%), de snelste groeier; 293.000 actieve eindklanten, +51% j-o-j. |

### Aandeelhouders (top 5)
Laatste geverifieerde meldingen uit het AFM-register substantiële deelnemingen; de
percentages dateren van de meldingsdatum en zijn geen actuele stand.

| Naam | Belang % | Type (oprichter / PE / institutioneel / retail) |
|---|---|---|
| Temasek Holdings (Ossa, Bayfront, Havelock, Aranda) | 6,73 (melding 5-6-2025) | Institutioneel (staatsfonds) |
| BlackRock, Inc. | 4,98 stemrecht (melding 13-9-2021) | Institutioneel |
| Pieter van der Does (medeoprichter, co-CEO) | 2,93 | Oprichter |
| Norges Bank Investment Management | 2,95 | Institutioneel |
| Partners in Equity III B.V. | 2,67 (melding 18-11-2020) | Institutioneel |

- **Institutioneel eigendomstrend**: Niet eenduidig vast te stellen. De oorspronkelijke
durfkapitaalpartijen zijn goeddeels vertrokken: Pentavest (Index Ventures) meldde bij de
beursgang 14,82% en komt in latere meldingen niet meer voor, en de Stichting
Administratiekantoor met 13,42% evenmin. Temasek is met 6,73% de enige grote partij van
vóór de beursgang die is gebleven. Het aandeel van indexbeleggers is gestegen door opname
in de AEX, waar Adyen per 31 maart 2026 met 3,13% de negende positie inneemt.

---

## 3. Financieel — historische data (10 jaar + TTM)

### Resultatenrekening (bedragen in EUR mln)

Boekjaren 2018-2025 zijn de som van de H1- en de H2-shareholder letter; 2015-2017 komen
uit het IPO-prospectus. TTM = H2 2025 + H1 2026. Adyen rapporteert geen brutowinst: de
kosten die kaartmaatschappijen en banken doorbelasten worden direct van de bruto-omzet
afgetrokken en het resultaat heet netto-omzet. Die netto-omzet is in dit rapport
consequent de omzetregel; de bruto-omzet (EUR 2.646,9 mln in 2025) is als maatstaf
betekenisloos omdat hij meebeweegt met kosten die volledig worden doorgegeven.

| Jaar | Omzet | Omzetgroei % | Brutowinst | Brutomarge % | EBIT | EBIT-marge % | EBITDA | EBITDA-marge % | Nettowinst | Nettomarge % | EPS | EPS-groei % | Aandelen mln |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2015 | 98,5 | — | — | — | — | — | 43,1 | 43,8 | 33,6 | 34,1 | — | — | — |
| 2016 | 158,0 | 60,4 | — | — | — | — | 123,4 | 78,1 | 97,2 | 61,5 | — | — | — |
| 2017 | 218,3 | 38,2 | — | — | — | — | 99,4 | 45,5 | 71,3 | 32,7 | — | — | — |
| 2018 | 348,9 | 59,8 | — | — | 173,2 | 49,6 | 181,9 | 52,1 | 131,1 | 37,6 | 4,45 | — | 29,60 |
| 2019 | 496,7 | 42,4 | — | — | 257,0 | 51,7 | 279,3 | 56,2 | 204,0 | 41,1 | 6,89 | 54,8 | 30,10 |
| 2020 | 684,2 | 37,8 | — | — | 373,9 | 54,6 | 402,5 | 58,8 | 261,0 | 38,1 | 8,63 | 25,3 | 30,40 |
| 2021 | 1.001,5 | 46,4 | — | — | 595,0 | 59,4 | 630,0 | 62,9 | 469,7 | 46,9 | 15,41 | 78,6 | 31,00 |
| 2022 | 1.330,2 | 32,8 | — | — | 664,7 | 50,0 | 728,3 | 54,8 | 564,1 | 42,4 | 18,21 | 18,2 | 31,00 |
| 2023 | 1.626,1 | 22,2 | — | — | 657,6 | 40,4 | 743,0 | 45,7 | 698,3 | 42,9 | 22,52 | 23,7 | 31,00 |
| 2024 | 1.996,1 | 22,8 | — | — | 887,8 | 44,5 | 992,3 | 49,7 | 925,2 | 46,3 | 29,74 | 32,1 | 31,49 |
| 2025 | 2.364,2 | 18,4 | — | — | 1.109,9 | 46,9 | 1.245,7 | 52,7 | 1.062,5 | 44,9 | 33,73 | 13,4 | 31,54 |
| TTM | 2.573,6 | — | — | — | 1.191,5 | 46,3 | 1.343,6 | 52,2 | 1.125,6 | 43,7 | 35,70 | — | 31,57 |

Halfjaarcijfers ter controle: H1 2026 netto-omzet EUR 1.302,9 mln (+19% gerapporteerd,
+21% constante valuta), EBITDA EUR 641,5 mln (+18%), marge 49%, nettowinst EUR 544,1 mln
(+13%), EPS basic EUR 17,24.

- **Toelichting resultaten**: De omzet is tussen 2015 en 2025 met gemiddeld 37,4% per jaar
gegroeid en tussen 2018 en 2025 met 31,4% — een reeks van tien jaar zonder één krimpjaar,
inclusief de coronaperiode, toen het wegvallen van reizen en horeca ruimschoots werd
gecompenseerd door e-commerce. De marge vertelt het interessantere verhaal. Tot en met
2021 liep de EBITDA-marge op naar 62,9%, waarna Adyen bewust versnelde met aanwerven: het
personeelsbestand ging van 2.180 naar 4.196 mensen in twee jaar en de marge zakte naar
45,7% in 2023. Sindsdien loopt de operationele hefboom weer op — 49,7% in 2024, 52,7% in
2025 — precies zoals het management had aangekondigd. De nettomarge ligt structureel boven
de EBIT-marge omdat Adyen honderden miljoenen aan rente-inkomsten boekt op de
merchant-gelden die tijdelijk op de balans staan; in 2025 was dat EUR 267,6 mln, ofwel
19% van de winst vóór belasting.
- **Omzet-CAGR**: 37,4% over 2015-2025; 31,4% over 2018-2025 (de volledige beursperiode).

### Kasstromen

De operationele kasstroom van Adyen is als maatstaf onbruikbaar zonder correctie. De
mutatie in "payables to merchants and financial institutions" loopt er direct doorheen:
in 2025 een min van EUR 2,36 mrd in de tweede helft, in H1 2026 een plus van EUR 1,73 mrd.
Dat is geld van klanten dat toevallig op de balansdatum wel of niet was uitbetaald, geen
kasstroom van het bedrijf. De kolom "CFO ex-float" corrigeert daarvoor. Adyen stuurt zelf
op een eigen definitie: vrije kasstroom = EBITDA minus capex minus leasebetalingen. Die is
hieronder als hoofdmaatstaf opgenomen. Capex is hier de volledige investeringsuitgave uit
het kasstroomoverzicht (materiële vaste activa plus geactiveerde immateriële activa) en
ligt daardoor iets boven de door Adyen gerapporteerde capex, die POS-huurterminals
uitsluit.

| Jaar | CFO | CFO ex-float | Capex | FCF (Adyen-def.) | FCF/aandeel | FCF-marge % | FCF-groei % | FCF-conversie % | SBC | Leasebetalingen | Betaalde winstbelasting |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2015 | 138,4 | — | 6,9 | — | — | — | — | — | — | — | — |
| 2016 | 189,6 | — | 12,5 | — | — | — | — | — | — | — | — |
| 2017 | 200,6 | — | 17,9 | — | — | — | — | — | — | — | — |
| 2018 | 384,0 | — | 13,8 | 168,1 | 5,68 | 48,2 | — | 92,4 | — | 0,0 | 21,6 |
| 2019 | 529,5 | 282,7 | 20,0 | 249,4 | 8,29 | 50,2 | 48,4 | 89,3 | — | 9,9 | 53,5 |
| 2020 | 1.016,6 | 389,8 | 21,9 | 371,1 | 12,21 | 54,2 | 48,8 | 92,2 | 0,9 | 9,4 | 72,0 |
| 2021 | 1.820,1 | 549,8 | 54,3 | 566,6 | 18,28 | 56,6 | 52,7 | 89,9 | 0,3 | 9,0 | 127,4 |
| 2022 | 2.021,2 | 569,7 | 99,1 | 607,0 | 19,58 | 45,6 | 7,1 | 83,4 | — | 22,1 | 149,6 |
| 2023 | 1.870,0 | 844,8 | 69,7 | 639,5 | 20,63 | 39,3 | 5,3 | 86,1 | 24,7 | 33,8 | 155,2 |
| 2024 | 1.704,8 | 1.130,9 | 101,2 | 849,7 | 26,99 | 42,6 | 32,9 | 85,6 | 34,8 | 41,4 | 320,5 |
| 2025 | 1.030,4 | 1.246,2 | 125,8 | 1.072,3 | 34,00 | 45,4 | 26,2 | 86,1 | 42,4 | 47,6 | 264,3 |
| TTM | 649,5 | — | 148,0 | 1.145,8 | 36,30 | 44,5 | 6,9 | 85,3 | 53,3 | 49,8 | 320,7 |

- **Toelichting kasstromen**: De vrije kasstroom is in acht jaar met gemiddeld 30,3% per
jaar gegroeid en de conversie ligt al die tijd tussen 83% en 92%, wat past bij een bedrijf
dat nauwelijks werkkapitaal nodig heeft. Twee bewegingen verdienen uitleg. De eerste is
2022-2023: de FCF groeide slechts 7,1% en 5,3% terwijl de omzet met ruim 20% steeg. Dat
was geen verslechtering van het bedrijfsmodel maar de aanwervingsgolf, samen met een
capex-piek van EUR 99,1 mln in 2022 voor nieuwe datacenters en het Amsterdamse kantoor. De
tweede is de betaalde winstbelasting, die in 2024 sprong van EUR 155,2 naar EUR 320,5 mln
— meer dan een verdubbeling bij 32% winstgroei. Dat was een inhaaleffect na de lage
afdrachten van 2022-2023 en geen structurele lastenverzwaring; over 2024, 2025 en H1 2026
samen bedraagt het kastarief 23,2% tegen een last in de winst- en verliesrekening van
24,0%. Het verschil is klein genoeg om de vrije kasstroom niet te vertekenen. De
aandelencompensatie is wel structureel opgelopen, van vrijwel nul in 2021 naar EUR 53,3 mln
over de laatste twaalf maanden; die kost drukt de EBITDA al, dus de FCF hierboven is
netto na SBC.

### Balans-ratio's

Adyen heeft geen rentedragende schuld — nul, in elk jaar sinds de beursgang. De enige
schuldachtige post is de leaseverplichting voor kantoren. De kaspositie op de balans is
grotendeels niet van Adyen: van de EUR 12,4 mrd per 30 juni 2026 is EUR 8,1 mrd verschuldigd
aan handelaren. De kolom "eigen nettokas" corrigeert daarvoor (kas minus merchant-schulden
plus merchant-vorderingen) en is de basis voor de waardering.

| Jaar | Totale activa | Eigen vermogen | Kas totaal | Merchant-gelden | Eigen nettokas | Leaseverplichting | Rentedragende schuld | ROE % | ROIC % | Solvabiliteit % | Boekwaarde/aandeel |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2015 | 821,4 | 245,8 | — | — | — | — | 0,0 | — | — | 29,9 | — |
| 2016 | 1.375,6 | 315,0 | — | — | — | — | 0,0 | — | — | 22,9 | — |
| 2017 | 1.137,2 | 389,8 | — | — | — | — | 0,0 | — | — | 34,3 | — |
| 2018 | 1.860,4 | 582,4 | 1.231,9 | 1.186,9 | 400,7 | 0,0 | 0,0 | — | — | 31,3 | 19,68 |
| 2019 | 2.609,0 | 868,3 | 1.745,4 | 1.521,4 | 667,3 | 61,7 | 0,0 | 28,1 | 88,5 | 33,3 | 28,85 |
| 2020 | 4.158,5 | 1.218,1 | 2.737,5 | 2.588,9 | 1.032,6 | 131,5 | 0,0 | 25,0 | 98,7 | 29,3 | 40,07 |
| 2021 | 5.775,6 | 1.810,4 | 4.616,1 | 3.608,5 | 1.640,8 | 143,0 | 0,0 | 31,0 | 144,6 | 31,3 | 58,40 |
| 2022 | 7.620,3 | 2.416,1 | 6.522,3 | 4.795,8 | 2.095,6 | 203,1 | 0,0 | 26,7 | 121,6 | 31,7 | 77,94 |
| 2023 | 9.568,4 | 3.159,6 | 8.307,0 | 5.942,0 | 2.855,0 | 223,1 | 0,0 | 25,0 | 95,7 | 33,0 | 101,92 |
| 2024 | 11.425,3 | 4.231,5 | 9.965,0 | 6.684,7 | 3.939,2 | 228,3 | 0,0 | 25,0 | 129,6 | 37,0 | 134,40 |
| 2025 | 12.256,8 | 5.285,4 | 10.797,4 | 6.371,8 | 4.987,3 | 252,4 | 0,0 | 22,3 | 158,5 | 43,1 | 167,60 |
| 30-6-2026 | 14.724,1 | 5.908,8 | 12.392,8 | 8.097,4 | 4.898,3 | 409,4 | 0,0 | 20,1 | 92,5 | 40,1 | 187,18 |

- **Toelichting balans**: De balans is de sterkste die je in deze sector tegenkomt: geen
enkele lening, een eigen kaspositie die in acht jaar van EUR 401 mln naar EUR 4,9 mrd is
gegroeid, en een kredietbeoordeling A- van S&P. Bruto en netto bewegen hier bewust uiteen:
de leaseverplichting steeg in H1 2026 fors, van EUR 252 naar EUR 409 mln, door nieuwe
datacenter- en kantoorcontracten, terwijl de eigen nettokas licht daalde van EUR 4.987 naar
EUR 4.898 mln. Die daling is geen verslechtering maar het gevolg van EUR 655 mln die vóór
30 juni al was betaald voor Talon.One en Orb en op de balans staat als vooruitbetaalde
overnamesom. Ook het rendement op geïnvesteerd kapitaal moet in dat licht: de val van
158,5% naar 92,5% komt bijna volledig doordat die EUR 655 mln nu als geïnvesteerd kapitaal
meetelt zonder dat er al winst tegenover staat. Als toezichthouder-gereguleerde bank hield
Adyen ultimo 2025 EUR 4.638 mln aan toetsingsvermogen aan, uitsluitend kernkapitaal, bij
een liquiditeitsdekkingsratio van 1.717%.

### Kapitaalstructuur huidig (30 juni 2026)
- **Nettoschuld (huidig)**: nettokaspositie van EUR 4.898,3 mln (eigen kas, exclusief
  merchant-gelden)
- **Bruto schuld**: EUR 0,0 mln rentedragende schuld; EUR 409,4 mln leaseverplichtingen
  (EUR 338,3 mln langlopend, EUR 71,1 mln kortlopend)
- **Cash & equivalents**: EUR 12.392,8 mln totaal, waarvan EUR 8.097,4 mln verschuldigd
  aan handelaren
- **Lease-verplichtingen (IFRS-16)**: EUR 409,4 mln
- **Gemiddelde rente %**: niet van toepassing — geen rentedragende schuld. De
  rentelast van EUR 6,1 mln in H1 2026 betreft vrijwel volledig lease-rente en
  negatieve rente op enkele valutaposities.
- **Rente-dekking (EBIT/rente)**: 93,3x (EBIT H1 2026 EUR 564,3 mln / rentelast
  EUR 6,1 mln); economisch niet relevant, Adyen is een netto-renteontvanger.

### Non-GAAP / aanpassingen
- **Gebruikt?** true
- **Welke aanpassingen**: Adyen rapporteert drie non-IFRS-maatstaven en definieert ze
  expliciet in noot 1.1: netto-omzet (bruto-omzet minus doorbelaste kosten van financiële
  instellingen en kostprijs van verkochte goederen, plus netto rentebaten uit de
  embedded-financial-productssuite), EBITDA (resultaat vóór netto financiële baten en
  belastingen, plus afschrijvingen) en vrije kasstroom (EBITDA minus capex minus
  leasebetalingen exclusief rente). In H1 2026 meldt Adyen daarnaast dat de EBITDA-marge
  zonder de EUR 6,0 mln eenmalige transactiekosten voor Talon.One en Orb 50% zou zijn
  geweest in plaats van 49%.
- **Waarom**: De bruto-omzet van Adyen bevat de interchange- en schemekosten die
  één-op-één worden doorbelast; die post beweegt met kaartregels en merchantmix en zegt
  niets over de prestatie. De netto-omzet is daarom de enige zinvolle omzetmaatstaf en
  wordt in dit rapport consequent gebruikt. De EBITDA-definitie is conservatief: hij sluit
  de rentebaten op merchant-gelden uit, terwijl die wel echt geld opleveren. De
  transactiekostencorrectie in H1 2026 is met EUR 6,0 mln klein en aantoonbaar eenmalig;
  de IFRS-cijfers zijn desondanks als basis voor de waardering gebruikt.

### Earnings quality

De accruals ratio (nettowinst minus operationele kasstroom, gedeeld door de gemiddelde
totale activa) is bij Adyen zonder correctie onbruikbaar, omdat de merchant-float er
doorheen loopt. Beide varianten staan daarom in de tabel.

| Jaar | Accruals ratio % | Accruals ratio ex-float % | SBC | SBC als % van FCF | SBC als % van marktkap |
|---|---|---|---|---|---|
| 2021 | −27,19 | −1,61 | 0,3 | 0,1 | — |
| 2022 | −21,75 | −0,08 | — | — | — |
| 2023 | −13,63 | −1,70 | 24,7 | 3,9 | — |
| 2024 | −7,43 | −1,96 | 34,8 | 4,1 | — |
| 2025 | 0,27 | −1,55 | 42,4 | 4,0 | — |
| TTM | — | — | 53,3 | 4,7 | 0,17 |

- **Toelichting earnings quality**: Op de gecorrigeerde basis is de accruals ratio elk jaar
licht negatief, tussen −0,1% en −2,0%, wat betekent dat de kasstroom de winst iets
overtreft en de winst dus conservatief is opgebouwd. Er is geen oplopende trend die op
winststuring wijst. De niet-gecorrigeerde reeks laat vooral zien hoe groot de vertekening
door de float is: van −27,2% in 2021 naar +0,3% in 2025, puur door de timing van
uitbetalingen. De non-IFRS-aanpassingen zijn beperkt en helder gedefinieerd; er is geen
"adjusted earnings" die jaar na jaar dezelfde posten uitsluit. De aandelencompensatie is
wel opgelopen naar EUR 53,3 mln over de laatste twaalf maanden, 4,7% van de vrije kasstroom
en 0,17% van de beurswaarde — laag voor een technologiebedrijf, en de kost drukt de
gerapporteerde EBITDA al.

### Rendementsindicatoren en ROIC-WACC-spread

Geïnvesteerd kapitaal = eigen vermogen minus eigen nettokas plus leaseverplichtingen,
gemiddeld over begin- en eindstand. NOPAT = EBIT × (1 − 23,5%).

| Jaar | ROE % | ROIC % | ROA % | WACC % (schatting) | Spread (pp) | Oordeel |
|---|---|---|---|---|---|---|
| 2019 | 28,1 | 88,5 | 9,1 | 11,3 | +77,2 | Extreme waardecreatie |
| 2020 | 25,0 | 98,7 | 7,7 | 11,3 | +87,4 | Extreme waardecreatie |
| 2021 | 31,0 | 144,6 | 9,5 | 11,3 | +133,3 | Extreme waardecreatie |
| 2022 | 26,7 | 121,6 | 8,4 | 11,3 | +110,3 | Extreme waardecreatie |
| 2023 | 25,0 | 95,7 | 8,1 | 11,3 | +84,4 | Extreme waardecreatie |
| 2024 | 25,0 | 129,6 | 8,8 | 11,3 | +118,3 | Extreme waardecreatie |
| 2025 | 22,3 | 158,5 | 9,0 | 11,3 | +147,2 | Extreme waardecreatie |
| TTM | 20,1 | 92,5 | 8,3 | 11,3 | +81,2 | Extreme waardecreatie |

- **Toelichting rendement**: Adyen verdient structureel acht tot veertien keer zijn
kapitaalkosten op het kapitaal dat werkelijk in de operatie zit, en dat is geen
eenmalig verschijnsel: de spread is in geen enkel jaar sinds de beursgang onder de 77
procentpunten gezakt, ook niet in het slechte jaar 2023. De verklaring is dat het bedrijf
nauwelijks kapitaal nodig heeft — de vaste activa bedragen EUR 652 mln en het werkkapitaal
EUR 34 mln, tegenover EUR 2,6 mrd omzet. Het rendement op eigen vermogen daalt wél
gestaag, van 31,0% in 2021 naar 20,1% nu, maar dat is een boekhoudkundig effect: de
winst wordt volledig ingehouden en stapelt zich op als kas die nauwelijks meer dan de
risicovrije rente opbrengt. Precies dat is het zwakke punt in de kapitaalallocatie.

### Waarderingsratio's (peildatum 13-8-2026, koers EUR 972,10)

| Maatstaf | Huidig | Toelichting |
|---|---|---|
| P/E (TTM, verwaterd) | 27,3 | EPS TTM EUR 35,58 |
| Forward P/E | 24,5 | StockAnalysis, 11-8-2026 |
| P/FCF (Adyen-definitie) | 26,8 | FCF TTM EUR 1.145,8 mln |
| P/FCF (na kasbelasting, DCF-basis) | 31,2 | Basis-FCFF EUR 984,8 mln |
| FCF-rendement | 3,21% | op basis-FCFF |
| EV/EBITDA | 19,2 | EV EUR 25.787,7 mln (marktkap minus eigen nettokas) |
| EV/omzet | 10,0 | |
| P/B | 5,19 | boekwaarde EUR 187,18 per aandeel |
| Dividendrendement | 0,00% | Adyen keert geen dividend uit |
| PEG | 1,37 | P/E gedeeld door 20% verwachte groei (Investor Day-doelstelling) |
| Graham number | 387,10 | EUR 387,10 tegenover een koers van EUR 972,10 |

- **Toelichting waardering**: Adyen noteert op 19,2 keer EBITDA en 27,3 keer de winst,
tegenover een sector waarin PayPal op 7,8 keer EBITDA staat, Fiserv op 6,9 en Worldline op
6,7. Die vergelijking is misleidend: die drie groeien niet of krimpen, terwijl Adyen 21%
groeit met een marge van 52%. De relevante peers zijn de groeiers, en daar staat Adyen
juist goedkoop: Toast noteert op 37,3 keer EBITDA en Block op 30,6. Belangrijker dan de
peervergelijking is de eigen historie: het aandeel staat 39% onder de top van twaalf
maanden geleden (52-weeksbereik EUR 772,40 tot EUR 1.600,80) terwijl de winst per aandeel
in diezelfde periode is gestegen. Let bij externe bronnen op de enterprise value —
gangbare databronnen trekken de volledige kaspositie van EUR 12,4 mrd af en komen op
EUR 18,8 mrd, maar EUR 8,1 mrd daarvan is geld van klanten.

### Sector-specifieke KPI's (technologie/betalingen)

| KPI | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | H1 2026 |
|---|---|---|---|---|---|---|---|---|
| Verwerkt volume (EUR mrd) | 239,6 | 303,6 | 516,0 | 767,5 | 970,1 | 1.285,9 | 1.394,3 | 803,8 |
| Take rate (bps) | 20,7 | 22,5 | 19,4 | 17,3 | 16,8 | 15,5 | 17,0 | 16,2 |
| Full-stack aandeel van volume (%) | 72 | — | 81 | — | 83 | 83 | 84 | — |
| Point-of-sale volume (EUR mrd) | — | — | 41,8 | 112,5 | 159,9 | — | 310,9 | 175,7 |
| POS als % van volume | — | — | 8 | 15 | 16 | 18 | 22 | 22 |
| Volumeverloop (churn, %) | — | — | <1 | <1 | <1 | <1 | <1 | — |
| FTE (ultimo) | 1.182 | 1.747 | 2.180 | 3.332 | 4.196 | 4.345 | 4.771 | 5.020 |
| Netto-omzet per FTE (EUR '000) | 420 | 392 | 459 | 399 | 388 | 459 | 496 | 513 |
| Rule of 40 (omzetgroei % + EBITDA-marge %) | 98,6 | 96,6 | 109,3 | 87,6 | 67,9 | 72,5 | 71,1 | 70,2 |

- **Toelichting sector-KPI's**: Twee reeksen vertellen samen het hele verhaal. Het volume
groeit onverstoord door — van EUR 240 mrd in 2019 naar EUR 1.394 mrd in 2025 — terwijl de
take rate in diezelfde periode van 20,7 naar 16,2 basispunten daalt. Adyen noemt dat zelf
een natuurlijk gevolg van zijn staffelmodel: hoe meer een klant verwerkt, hoe lager het
tarief. Het is ook de reden dat de omzet trager groeit dan het volume, en de kern van het
neerwaartse risico. Daar staan drie compensaties tegenover: het aandeel full-stack volume,
waarbij Adyen de hele keten bedient en dus meer per transactie verdient, is gestegen van
72% naar 84%; het aandeel fysieke betalingen — het duurdere segment — is verdrievoudigd
van 8% naar 22%; en 2025 liet voor het eerst sinds jaren een stijging van de take rate
zien, van 15,5 naar 17,0 basispunten. De Rule of 40 staat op 70 en is daarmee al vier jaar
comfortabel boven de drempel, al is de piek van 109 uit 2021 ver weg.

### Dividendanalyse

Adyen keert **geen dividend** uit en heeft dat sinds de beursgang in 2018 nooit gedaan. Er
is ook nooit een aandeleninkoopprogramma geweest. Het jaarverslag 2025 formuleert het
beleid expliciet: *"The current dividend policy is not to pay dividends, as retained
earnings are used to support and finance the growth strategy of the Company."* De
uitkeerbare reserves bedroegen ultimo 2025 EUR 4.171,8 mln, dus de beperking is een keuze,
geen noodzaak.

| Jaar | DPS | Groei YoY % | Uitkeringsratio % | FCF-dekkingsratio | Type | Bijzonderheden |
|---|---|---|---|---|---|---|
| 2018-2025 | 0,00 | — | 0,0 | n.v.t. | Geen | Geen regulier, speciaal of stockdividend |

- **Dividend — toelichting**: Voor een inkomensbelegger is Adyen niet interessant, en dat
zal voorlopig zo blijven. De vraag die er wél toe doet is of het inhouden van alle winst
te verdedigen is. Tot ongeveer 2022 was het antwoord onomstreden ja: het bedrijf verdiende
meer dan 100% op zijn geïnvesteerde kapitaal en elke ingehouden euro werd meer waard.
Inmiddels is de situatie anders. Er staat EUR 4,9 mrd eigen kas op de balans die vrijwel
niets doet behalve rente ontvangen; die rente was in 2025 EUR 267,6 mln op een gemiddelde
positie van meer dan EUR 10 mrd. De operatie zelf heeft dat geld niet nodig — er wordt
maar EUR 126 mln per jaar geïnvesteerd. In 2026 is voor het eerst een deel ingezet, ruim
EUR 1 mrd voor Talon.One en Orb, wat een verstandiger bestemming is dan oppotten, maar het
saldo groeit sneller dan het wordt uitgegeven. Een structureel inkoopprogramma zou
verdedigbaar zijn geweest toen het aandeel in 2025 op EUR 772 stond.
- **Oordeel houdbaarheid**: Niet van toepassing — er is geen dividend om houdbaar te
noemen. Het feit dát er niets wordt uitgekeerd is geen signaal van zwakte maar van een
kapitaalallocatiekeuze die met een groeiende kasberg steeds meer uitleg vraagt.

---

## 4. Moat (concurrentievoordeel)

- **Oordeel**: **WIDE MOAT**
- **Moat-categorieën**:

| Naam | Sterkte | Toelichting |
|---|---|---|
| Immateriële activa | sterk | Adyen is een door De Nederlandsche Bank vergunde kredietinstelling (25 april 2017), met daarnaast een Amerikaanse federal foreign branch en een Britse third-country-vergunning van PRA en FCA. Die vergunningen laten Adyen toe geld zelf aan te houden en uit te betalen zonder tussenpartij, en zijn voor een nieuwkomer een traject van jaren. Ook het platform zelf is een immaterieel actief: één codebase, twintig jaar doorontwikkeld, die niet is na te bouwen door een overname te doen. De S&P-rating A- op een niet-schuldenaar illustreert hoe zwaar die licentiepositie weegt. |
| Overstapkosten | sterk | Migreren betekent voor een klant als Uber of H&M dat elk land, elk kanaal, elke betaalmethode en elke afstemming opnieuw moet worden ingeregeld, terwijl er live geld doorheen loopt. Het bewijs zit in de cijfers: het volume van klanten die vertrekken is al vijf jaar op rij minder dan 1% van het totaal, en het aandeel van Adyen bij een klant groeit van onder 20% in de eerste jaren naar boven 40% na tien jaar. Google, Microsoft, Spotify en Uber breiden na meer dan tien jaar nog steeds uit. |
| Netwerkeffecten | beperkt | Er is geen klassiek tweezijdig netwerk: consumenten kiezen niet voor Adyen. Wel is er een datavoordeel dat erop lijkt. Doordat dezelfde shopper bij honderden merchants op hetzelfde platform betaalt, herkent Adyen die shopper en stijgt het percentage geslaagde transacties. Het bedrijf claimt 10% gemiddelde besparing op scheme- en interchangekosten via Adyen Uplift. Dat effect versterkt zichzelf met schaal, maar is geen slot op de deur — Stripe en Checkout.com hebben ook grote datasets. |
| Kostenvoordeel | sterk | Het beslissende voordeel. Eén platform voor alle landen en kanalen betekent dat elke extra transactie vrijwel niets kost. De EBITDA-marge van 52,7% over 2025 tegenover 2,6% bij Worldline, 13,1% bij Nexi en 22,3% bij Fiserv is geen toeval maar het verschil tussen zelf bouwen en tientallen overgenomen systemen aan elkaar knopen. Met EUR 513.000 netto-omzet per medewerker is Adyen bovendien ruim productiever dan de sector. |
| Efficiënte schaal | zwak | De markt is enorm — McKinsey raamt de wereldwijde betaalopbrengsten op USD 2,5 biljoen in 2024 — en biedt ruimte aan tientallen spelers. Er is geen natuurlijke bovengrens die concurrentie afremt. Integendeel: Stripe verwerkte in 2025 USD 1,9 biljoen en Checkout.com USD 300 mrd, beide met dubbelcijferige groei. |

- **Kwantitatief bewijs**: Het rendement op geïnvesteerd kapitaal lag in elk jaar sinds
2019 tussen 88,5% en 158,5%, bij kapitaalkosten van 11,3%. Een spread van meer dan 77
procentpunten die zeven jaar aanhoudt, ook in het slechte jaar 2023, is niet met
toeval te verklaren. De EBITDA-marge herstelde binnen twee jaar van 45,7% (2023) naar
52,7% (2025) zonder dat er klanten weggingen, wat laat zien dat de margedaling een
investeringskeuze was en geen prijsdruk. Het volumeverloop bleef vijf jaar op rij onder
1%. Het aandeel full-stack volume steeg van 72% (2019) naar 84% (2025).

- **Duurzaamheid**: 10 jaar. De vergunningen en de overstapkosten houden zeker een
decennium; die verdwijnen niet door een technologiegolf. Het kostenvoordeel uit het
enkelvoudige platform is duurzaam zolang Adyen zelf niet gaat overnemen en integreren —
en juist daar is in 2026 een eerste barst ontstaan met Talon.One en Orb, die voorlopig als
aparte dochters draaien.

- **Erosierisico's**: De slotgracht beschermt het klantenbestand, niet de prijs. De take
rate is van 21,9 basispunten in 2018 naar 16,2 in H1 2026 gedaald, en dat is precies wat
je verwacht als klanten groter worden en meer onderhandelingsmacht krijgen. Stripe is
inmiddels groter in volume en heeft met het Tempo-blockchainnetwerk (live sinds 18 maart
2026) en zijn agentic-betaalprotocollen een eigen technologische agenda. Als
AI-agenten een groot deel van de aankopen gaan doen, verschuift de macht mogelijk naar
wie het agentprotocol bezit; Adyen reageerde daarop met Adyen Agentic (16 juni 2026), maar
dat is voorlopig alleen in de Verenigde Staten beschikbaar. Een derde route is
regelgeving: de Amerikaanse interchange-schikking van 9 juni 2026 verlaagt de tarieven
die door de keten lopen, wat de bruto-omzet raakt maar de netto-omzet in principe niet.

---

## 5. Management

- **CEO-naam + tenure**: Adyen kent een duaal leiderschap. **Pieter van der Does**
(medeoprichter, 57) is bestuurder sinds juli 2007 en co-CEO; hij werd op de AGM van
28 mei 2026 herbenoemd voor vier jaar, tot 2030. **Ingo Uytdehaage** (53) trad in juni
2011 aan als CFO en is co-CEO sinds mei 2023.
- **CFO-naam + tenure**: **Ethan Tandowsky**, bij Adyen sinds 2016, CFO sinds mei 2023,
**treedt af per 31 augustus 2026** voor een functie buiten de fintechsector (aangekondigd
27 mei 2026). **Hwa Tsao**, tot dan SVP Group Finance met een achtergrond bij ServiceNow,
HP en Citigroup, wordt per 1 september 2026 interim-CFO. De raad van commissarissen zoekt
via een extern bureau een definitieve opvolger.
- **Oprichter nog betrokken?**: Ja, Pieter van der Does is co-CEO. Medeoprichter Arnout
Schuijff trad per 1 januari 2021 af als CTO en bestuurder.
- **Insider ownership %**: Pieter van der Does houdt circa 2,93% (922.815 aandelen).
Individuele insiders samen circa 4,49%. In augustus 2020 verkochten Van der Does,
Schuijff, Uytdehaage en Prins gezamenlijk 507.631 aandelen tegen EUR 1.365 — circa 15% van
hun belangen, EUR 692,9 mln bruto — expliciet om hun vermogen te spreiden, met een
lock-up op de rest.
- **Capital allocation track record**: Tot 2026 bestond de kapitaalallocatie uit precies
één ding: alles herinvesteren in het bedrijf en de rest laten staan. Geen dividend, geen
inkoop, geen overname, geen emissie. De organische investeringen — datacenters,
vergunningen, mensen — leverden een rendement op geïnvesteerd kapitaal van 88% tot 158%
per jaar. In 2026 kwam de ommekeer met twee overnames van samen ruim EUR 1 mrd.

| Jaar | Dividend totaal | Aandeleninkoop | M&A uitgaven | Organische capex |
|---|---|---|---|---|
| 2021 | 0,0 | 0,0 | 0,0 | 54,3 |
| 2022 | 0,0 | 0,0 | 0,0 | 99,1 |
| 2023 | 0,0 | 0,0 | 0,0 | 69,7 |
| 2024 | 0,0 | 0,0 | 0,0 | 101,2 |
| 2025 | 0,0 | 0,0 | 0,0 | 125,8 |
| H1 2026 | 0,0 | 0,0 | 655,2 (vooruitbetaald) | 69,4 |

- **M&A-track-record**: Tot april 2026 nul overnames in twintig jaar. Daarna twee, allebei
afgerond op 1 juli 2026: **Talon.One** (Berlijn, 2015, loyaliteits- en promotieplatform,
300+ merchants, verwachte ARR van circa EUR 60 mln eind 2026, groeiend 30-40% per jaar)
voor **EUR 750 mln**, en **Orb** (San Francisco, 2021, enterprise-billingplatform) voor
**USD 335 mln**, beide uit eigen middelen. Bij beide herinvesteren de oprichters een
substantieel deel van de opbrengst in nieuw uit te geven Adyen-aandelen — een gunstig
signaal. Er is nog geen track record om te beoordelen; wat wel opvalt is de prijs:
EUR 750 mln voor EUR 60 mln ARR is 12,5 keer de omzet. Orb draait voorlopig als aparte
dochter onder een incubatormodel, wat het enkelvoudige-platformverhaal enigszins verwatert.

- **Beloning**: Adyen betaalt zijn bestuurders opvallend sober en zonder prestatieprikkels.
Over 2025 verdienden beide co-CEO's elk een vast salaris van EUR 833.458 en de CFO
EUR 570.833. Het jaarverslag stelt letterlijk dat er over 2025 géén variabele beloning is
toegekend, en dat variabele beloning "not part of our approach today" is. De CFO, CHRO en
CRCO ontvangen sinds 2025 daarnaast aandelen ter waarde van 100% van hun basissalaris
(was 50%), met een houdperiode van vijf jaar; de co-CEO's en de CCO krijgen uitsluitend
contanten. Er zijn geen opties en geen prestatie-afhankelijke toekenningen. De totale
bestuursbeloning bedroeg EUR 9,22 mln en de verhouding tussen CEO-beloning en de mediane
werknemer is 8:1 — uitzonderlijk laag. Commissarissen krijgen EUR 125.000 (voorzitter) of
EUR 75.000, zonder variabele of aandelencomponent. Adyen mikt bewust onder de mediaan van
zijn beloningspeergroep.

- **Oordeel management**: **STERK**
- **Toelichting**: Het beeld is overwegend gunstig, met één duidelijke kanttekening. De
prikkels zijn zo zuiver als je ze in een beursfonds tegenkomt: geen bonus, geen opties,
een salaris onder de mediaan, aandelen met vijf jaar houdplicht en een medeoprichter met
bijna 3% eigen belang. In twintig jaar is het aantal aandelen met 7,2% toegenomen en is er
nooit kapitaal opgehaald. Er zijn geen boetes, toezichtmaatregelen of rechtszaken
gevonden, en de aankondiging van augustus 2023 laat zien dat het management slecht nieuws
niet verstopt, ook al kostte het 39% koers op één dag. De kanttekening is de
kapitaalallocatie: EUR 4,9 mrd die tegen ongeveer de risicovrije rente staat te wachten,
terwijl het aandeel in 2025 op EUR 772 stond en er geen inkoop kwam. Dat is
kapitaalallocatie op "goed", niet op "excellent". Daarbij komt dat de CFO per 31 augustus
vertrekt en de voorzitter van de raad van commissarissen eind 2026, midden in de
integratie van twee overnames — twee sleutelposities die tegelijk wisselen.

### Insider transactions (laatste 24 maanden)

| Datum | Persoon | Functie | Type | Aantal | Koers | Waarde |
|---|---|---|---|---|---|---|
| 2026-02-27 | Tom Adams | CTO | Verkoop | 396 | ~979 | ~387.684 |
| 2026-02-19 | Caoimhe Keogan | Commissaris | Koop | 100 | ~986 | ~98.600 |
| 2025-10-23 | Tom Adams | CTO | Verkoop | 362 | 1.427,27 | ~516.672 |

Buiten deze transacties bestaat het beeld uit maandelijkse toekenningen van kleine
aantallen aandelen aan bestuursleden als onderdeel van de vaste beloning (10 tot 31 stuks
per keer). De bedragen zijn in alle gevallen verwaarloosbaar ten opzichte van de
beurswaarde. **Netto beeld: NEUTRAAL** — geen betekenisvolle open-marktaankopen door
bestuurders, maar ook geen patroon van verkopen bij koersherstel. Let op: één publieke
bron (insiderscreener.com) toont voor 19 februari 2026 een aankoop van 37.800 stuks tegen
EUR 111,18; dat cijfer is intern inconsistent en strijdig met de koers van dat moment en
is daarom **niet overgenomen**.

---

## 6. Sector & concurrentie

- **Sector-groeivooruitzicht**: McKinsey raamt de wereldwijde betaalopbrengsten op
USD 2,5 biljoen in 2024, groeiend naar USD 3,0 biljoen in 2029 — circa **4% per jaar**,
met een scenariobandbreedte van 3% tot 6% (Global Payments Report, 26 september 2025).
BCG komt op een engere definitie uit op USD 1,9 biljoen (2024) naar USD 2,4 biljoen
(2029), eveneens circa 4%, met Europa op 3,5% en Latijns-Amerika op 7,9% (22 september
2025). De markt zelf groeit dus traag; Adyens groei van 20% moet vrijwel volledig uit
marktaandeel komen.

- **Porter five forces**:
  - **Rivaliteit: HOOG.** Stripe verwerkte in 2025 USD 1,9 biljoen aan volume tegenover
    Adyens EUR 1.394 mrd en is bij USD 159 mrd gewaardeerd. Checkout.com groeide 64% naar
    USD 300 mrd. Global Payments verwerkt na de Worldpay-overname USD 3,7 biljoen.
    Tegelijk is de concurrentie ongelijk verdeeld: de legacyspelers krimpen (Fiserv
    verlaagde op 7 augustus 2026 zijn jaarverwachting en staat 60% lager over twaalf
    maanden), terwijl de moderne platforms hard groeien.
  - **Nieuwe toetreders: LAAG.** Een nieuwkomer moet bankvergunningen in meerdere
    jurisdicties verwerven, wereldwijde betaalrails aansluiten en het vertrouwen van
    ondernemingen winnen die geen storing kunnen hebben. De vergunningsroute alleen al
    duurt jaren. De relevante nieuwe toetreders zijn dan ook geen start-ups maar bestaande
    grote partijen die naast de kaartrails willen gaan opereren.
  - **Substituten: MIDDEL.** Rekening-naar-rekeningbetalingen groeiden wereldwijd 40% in
    2024 en vormen al meer dan de helft van de digitale retailbetalingen in India en
    Brazilië. Stablecoins verplaatsten USD 26 biljoen, maar volgens BCG betreft daarvan
    slechts circa 1% echte betalingen. Adyen ondersteunt deze methoden zelf en verliest bij
    verschuiving vooral marge op de doorbelaste kosten, niet de klantrelatie.
  - **Macht leveranciers: HOOG.** Visa en Mastercard bepalen de schemekosten en de regels;
    Visa had in de eerste helft van 2025 38,5% van de wereldwijde kaarttransacties. Adyen
    belast die kosten door, dus de directe margedruk is beperkt, maar het is een
    afhankelijkheid waar geen onderhandelingsruimte tegenover staat.
  - **Macht afnemers: HOOG.** Dit is de kern van het probleem. Adyens klanten zijn
    multinationals met inkoopafdelingen die om de paar jaar een tender uitschrijven. Het
    staffelmodel geeft ze automatisch korting bij groei, wat rechtstreeks zichtbaar is in
    de take rate: van 21,9 naar 16,2 basispunten in acht jaar.
  - **Conclusie Porter**: Een gemiddeld aantrekkelijke sector. De toetredingsdrempels zijn
    reëel en de klantrelaties kleverig, maar de structurele prijsdruk van grote klanten en
    de dominantie van de kaartschema's beperken wat een verwerker op lange termijn kan
    verdienen. Dat Adyen er 52% EBITDA-marge haalt zegt meer over Adyen dan over de sector.

- **Concurrenten**:

| Concurrent | Omzetgroei % | EBIT-marge % | ROIC % | Nettoschuld/EBITDA | EV/EBITDA | Marktkap |
|---|---|---|---|---|---|---|
| **Adyen** | +18,4 (FY25); +21 cc (H1 26) | 46,3 | 92,5 | netto kas | 19,2 | EUR 30,7 mrd |
| Stripe (privaat) | +34 (volume 2025) | — | — | — | — | USD 159 mrd (feb 2026) |
| PayPal | +5,7 | 17,4 | 22,3 | 0,43x | 7,8 | USD 50,6 mrd |
| Fiserv | −1,2 | 22,3 | 6,9 | 3,43x | 6,9 | USD 27,4 mrd |
| Global Payments | +32,5 (M&A) | 12,4 | 3,9 | 4,81x | 11,0 | USD 23,4 mrd |
| Worldline | −3,2 | 2,6 | 1,6 | 5,20x | 6,7 | EUR 0,7 mrd |
| Nexi | +2,7 | 13,1 | 7,4 | 2,89x | ~6,5 | EUR 5,0 mrd |
| Block | +5,1 | 4,7 | 2,7 | 0,28x | 30,6 | USD 47,0 mrd |
| Shift4 | +32,4 | 8,1 | 5,6 | 4,57x | 8,3 | USD 3,5 mrd |
| Toast | +23,0 | 6,3 | n.b. | netto kas | 37,3 | USD 19,7 mrd |
| Checkout.com (privaat) | +64 (volume 2025) | >10 (adj. EBITDA) | — | — | — | USD 12 mrd (sep 2025) |

- **Positie van het bedrijf**: Adyen is geen marktleider in volume — Stripe en de
gecombineerde Global Payments/Worldpay zijn groter — maar wel de winstgevendste speler in
de sector en de voorkeurspartij bij grote internationale ondernemingen die één platform
voor alle landen en kanalen willen. **Challenger met een leidende positie in het
premiumsegment.**
- **Positie-toelichting**: In het peeroverzicht valt Adyen op twee assen buiten de reeks.
De EBIT-marge van 46,3% is bijna twee keer die van de beste beursgenoteerde concurrent
(Corpay, 43,7%) en achttien keer die van Worldline. Het rendement op geïnvesteerd kapitaal
van 92,5% staat tegenover 22,3% bij PayPal en minder dan 8% bij Fiserv, Nexi en Global
Payments. Op de waarderingsas betaalt de belegger daarvoor: 19,2 keer EBITDA tegenover 7
tot 11 keer bij de legacyspelers. De premie is te verdedigen zolang Adyen groeit terwijl
zij krimpen, maar hij is niet houdbaar als de groei terugvalt naar de 5 tot 10% waar
PayPal en Block nu zitten — dan zakt het aandeel naar de multiples van die groep.

### TAM/SAM/SOM
- **TAM (mln)**: USD 2.500.000 mln wereldwijde betaalopbrengsten (McKinsey, 2024)
- **TAM-groei %**: 4,0 (2025-2029, McKinsey; scenariobereik 3-6%)
- **SAM (mln)**: — (niet verifieerbaar)
- **SAM-groei %**: —
- **Huidige penetratie %**: — (zie toelichting)
- **Impliciete penetratie na horizon %**: —
- **Groei plausibel?**: true
- **Bron TAM/SAM**: McKinsey Global Payments Report 2025; BCG Global Payments Report 2025
- **Toelichting**: Een specifiek en met bron gestaafd cijfer voor de markt van uitsluitend
merchant acquiring is niet te vinden; McKinsey en BCG splitsen dat niet publiek uit en de
commerciële marktrapporten die het wel doen geven onderling onverenigbare uitkomsten (van
USD 22 tot USD 53 mrd). Die cellen blijven daarom leeg. Wat wel toetsbaar is: Adyens
netto-omzet van EUR 2,6 mrd is circa 0,1% van de door McKinsey geraamde betaalopbrengsten.
Zelfs een vervijfvoudiging in tien jaar brengt Adyen op ongeveer een half procent van de
wereldwijde opbrengstenpool. De aangenomen groei is dus niet begrensd door de marktomvang
maar door concurrentie en prijsdruk.

---

## 7. Analyse-frameworks (9 frameworks, SCORES 1-5)

### Graham
- **Oordeel**: VOLDOET NIET
- **Graham number**: 387,10
- **Margin of safety %**: −60,2 (koers EUR 972,10 tegenover Graham number EUR 387,10)
- **Toelichting**: Graham zou dit aandeel niet aanraken, en de rubriek laat geen ruimte
voor interpretatie: bij een koers-winstverhouding van 27,3 en een koers-boekwaarde van
5,19 valt de score op 1. De Graham number, de vuistregel voor de maximale prijs van een
defensieve belegging, komt uit op EUR 387 tegenover een koers van EUR 972 — de markt
betaalt twee en een half keer die grens. Dat is geen verrassing: Grahams criteria zijn
gebouwd voor kapitaalintensieve bedrijven met tastbare activa, en Adyen heeft
EUR 652 mln aan vaste activa tegenover EUR 2,6 mrd omzet. Wat Graham wél zou waarderen is
de balans: geen schuld, EUR 4,9 mrd eigen kas en tien jaar ononderbroken winst.
- **Score (1-5)**: **1**

### Buffett / Munger
- **Oordeel**: GEDEELTELIJK
- **ROIC structureel boven WACC?**: true
- **Toelichting**: Dit is qua bedrijf precies wat Buffett zoekt en qua prijs precies wat
hij vermijdt. Het rendement op geïnvesteerd kapitaal ligt sinds 2019 tussen 88% en 158%
bij kapitaalkosten van 11,3% — een spread die zeven jaar standhoudt, ook in het slechte
jaar. Het bedrijfsmodel is uitlegbaar in twee zinnen, de kasstroom is voorspelbaar, de
klanten vertrekken niet en het management betaalt zichzelf onder de mediaan zonder bonus.
De prijs is het probleem: 31,2 keer de vrije kasstroom na kasbelasting. Voor een score van
5 eist de rubriek P/FCF van 20 of lager; daar zit Adyen ver boven. Een uitzonderlijk
bedrijf tegen een volle prijs.
- **Score (1-5)**: **4**

### Peter Lynch
- **Categorie**: Fast grower
- **Oordeel**: NEUTRAAL
- **PEG-ratio**: 1,37
- **Toelichting**: Adyen is een klassieke fast grower: de omzet groeide zeven jaar op rij
met meer dan 18% en de winst per aandeel ging van EUR 4,45 in 2018 naar EUR 35,70 over de
laatste twaalf maanden. Het verhaal is helder genoeg om aan een leek uit te leggen — één
systeem dat betalingen voor grote bedrijven afhandelt, waar concurrenten er tien aan
elkaar hebben geplakt. De PEG-ratio van 1,37, gebaseerd op de eigen
groeidoelstelling van circa 20% per jaar uit de beleggersdag van november 2025, ligt boven
Lynch' drempel van 1 maar onder de 1,5 die de rubriek als redelijk aanmerkt. Je betaalt
dus iets meer dan de groei waard is, niet veel meer.
- **Score (1-5)**: **3**

### Phil Fisher
- **Oordeel**: STERK
- **Toelichting**: Op Fisher-criteria doet Adyen het goed op twee van de drie toetsbare
punten. De marge wordt aantoonbaar beschermd door de slotgracht: de EBITDA-marge veerde
binnen twee jaar terug van 45,7% naar 52,7% zonder klantverlies, en het volumeverloop
bleef onder 1%. De integriteit van het management is sterk: geen bonussen, geen opties,
beloning onder de mediaan, geen controverses of toezichtmaatregelen, en in augustus 2023
werd slecht nieuws direct gemeld ondanks 39% koersverlies op één dag. Het derde punt,
onderzoeksuitgaven als percentage van de omzet, is niet te toetsen omdat Adyen geen
aparte R&D-post rapporteert — alles zit in personeelskosten, die met EUR 756 mln 32% van
de netto-omzet uitmaken. De innovatiecultuur is wel zichtbaar in de productlancering van
Adyen Agentic (juni 2026) en Intelligent Money Movement.
- **Score (1-5)**: **4**

### Magic Formula (Greenblatt)
- **Oordeel**: GEMIDDELD
- **Earnings yield %**: 4,62
- **Return on capital %**: 173,6
- **Toelichting**: Greenblatt kijkt naar twee dingen: hoeveel bedrijfswinst je krijgt voor
je aankoopprijs, en hoe goed het bedrijf is in het omzetten van kapitaal in winst. Op de
tweede as scoort Adyen buitengewoon: EUR 1.191 mln bedrijfsresultaat op EUR 686 mln aan
netto vaste activa plus werkkapitaal geeft een rendement op kapitaal van 173,6%, ver boven
de drempel van 50% die de hoogste categorie markeert. Op de eerste as is het beeld
middelmatig: het bedrijfsresultaat is 4,62% van de ondernemingswaarde, net onder de 5% die
de rubriek als ondergrens hanteert. De formule zou Adyen dus wel opnemen in het
kwaliteitsdeel van de ranglijst, maar niet in het koopjesdeel.
- **Score (1-5)**: **3**

### Moat
- **Score (1-5)**: **4** — WIDE MOAT met drie sterke categorieën (immateriële activa,
overstapkosten, kostenvoordeel) en een ROIC-WACC-spread van meer dan 80 procentpunten. De
hoogste score is voorbehouden aan monopolies of duopolies met prijszettingsmacht; Adyen is
geen van beide, en het feit dat de take rate al acht jaar daalt bewijst dat het die macht
niet heeft.

### Management
- **Score (1-5)**: **4** — kapitaalallocatie GOED (niet EXCELLENT: EUR 4,9 mrd kas die
tegen bijna de risicovrije rente staat, geen inkoop toen het aandeel op EUR 772 stond),
prikkels volledig in lijn met aandeelhouders, insiderbelang boven 1%, geen controverses.

### Fair Value DCF
- **Score (1-5)**: **3** — upside van +9,3% ten opzichte van het basisscenario
(EUR 1.062,04 tegenover een koers van EUR 972,10), dus binnen de bandbreedte van 0 tot 15%.

### Fair Value IPO-gecorr.
- **Score (1-5)**: **3** — de beursgang van 13 juni 2018 ligt 8,2 jaar terug, dus binnen
de tienjaarsgrens, maar er is niets te corrigeren. Zie sectie 8 voor de volledige
pre-IPO-toets: de beursgang was volledig een verkoop door bestaande aandeelhouders, Adyen
ontving geen opbrengst, er was geen schuld om af te lossen en er is geen
dividendherkapitalisatie geweest. De gecorrigeerde fair value is daarom identiek aan de
ongecorrigeerde: EUR 1.062,04.

### Scorekaart totaal

| Framework | Score | Oordeel |
|---|---|---|
| Graham | 1 / 5 | P/E 27,3 > 25 én P/B 5,19 > 3,0 |
| Buffett / Munger | 4 / 5 | ROIC > 2×WACC structureel, WIDE moat, maar P/FCF 31,2 > 20 |
| Peter Lynch | 3 / 5 | PEG 1,37 ≤ 1,5 en het verhaal is helder |
| Phil Fisher | 4 / 5 | 2 van 3 criteria voldaan (margebescherming, integriteit STERK) |
| Magic Formula | 3 / 5 | ROC 173,6% ≥ 50%, maar earnings yield 4,62% < 5% |
| Moat | 4 / 5 | WIDE moat, spread > 10pp, geen monopolie/duopolie |
| Management | 4 / 5 | Capital allocation GOED, prikkels aligned, geen controverses |
| Fair Value DCF (basis) | 3 / 5 | Upside +9,3% (≥ 0% en < 15%) |
| Fair Value IPO-gecorr. | 3 / 5 | Geen IPO-correctie van toepassing; gelijk aan basis |
| **TOTAALSCORE** | **29 / 45** | **HOLD** |

- **Totaalscore**: 29
- **Max**: 45
- **Eindoordeel**: **HOLD** (29 ≥ 24 en 29 < 33)
- **Samenvatting**: Adyen is naar bedrijfskwaliteit gemeten een van de beste namen in dit
hele universum: een rendement op geïnvesteerd kapitaal dat zeven jaar op rij boven 88%
ligt, een EBITDA-marge van 52% waar de sector op 3 tot 22% zit, geen enkele lening,
EUR 4,9 mrd eigen kas en een bestuur dat zichzelf zonder bonus onder het marktmediaan
betaalt. De scorekaart komt desondanks uit op 29 punten en dus op HOLD, en dat komt door
twee dingen. Ten eerste de prijs: 27,3 keer de winst en 31,2 keer de vrije kasstroom
levert een enkel punt op bij Graham en een drie bij zowel Greenblatt als de eigen
kasstroomwaardering. Ten tweede het feit dat de tarieven per transactie al acht jaar dalen,
van 21,9 naar 16,2 basispunten, terwijl Stripe inmiddels meer volume verwerkt. De
koersval van 39% over twaalf maanden heeft de waardering wel teruggebracht naar een
niveau waar de markt nu ongeveer 16% kasstroomgroei inprijst tegenover een
managementdoelstelling van 20%. Interessant, maar nog geen koopje.

---

## 8. Risico's

| # | Omschrijving | Kans | Impact | DCF-aanname geraakt | Toelichting |
|---|---|---|---|---|---|
| 1 | Aanhoudende daling van de take rate zonder compensatie | HOOG | GROOT | omzetgroei fase 1 en 2 | De take rate zakte van 21,9 bps (2018) naar 16,2 bps (H1 2026). Dat is deels een bewuste keuze — volumestaffels binden grote klanten — maar het betekent dat het volume 25% harder moet groeien dan de omzet om hetzelfde te bereiken. Blijft de daling doorlopen op circa 0,7 bps per jaar zonder dat financiële producten het gat vullen, dan komt het pessimistische scenario van EUR 590 per aandeel in zicht. |
| 2 | Stripe wint het enterprise-segment | MIDDEN | GROOT | omzetgroei, terminale groei | Stripe verwerkte in 2025 USD 1,9 biljoen (+34%) tegenover Adyens EUR 1.394 mrd, is gewaardeerd op USD 159 mrd en bouwt met Tempo en zijn agentic-protocollen een eigen laag onder het betaalverkeer. Adyens overstapkosten beschermen het bestaande boek, niet de nieuwe klanten. Verliest Adyen de nieuwe aanwas, dan valt de groei binnen vijf jaar terug naar het niveau van de markt. |
| 3 | Rentedaling raakt de rente-inkomsten op merchant-gelden | MIDDEN | MIDDEL | basis-FCF | Van de EUR 1.394,0 mln winst vóór belasting in 2025 kwam EUR 267,6 mln uit rente op kasposities, ofwel 19%. Die post is direct gekoppeld aan het rentepeil; bij de ECB-rentes van 2021 was hij nagenoeg nul. Een terugkeer naar 1% beleidsrente kost circa EUR 200 mln aan winst vóór belasting, ruwweg 14%. In dit model is de floatrente daarom bewust nominaal constant gehouden. |
| 4 | Integratierisico Talon.One en Orb | MIDDEN | MIDDEL | EBITDA-marge | Na twintig jaar organisch bouwen kocht Adyen in 2026 voor ruim EUR 1 mrd twee bedrijven, tegen circa 12,5 keer de omzet voor Talon.One. Orb draait voorlopig als aparte dochter onder een incubatormodel — precies de constructie die Adyen bij concurrenten altijd als zwakte heeft aangewezen. De EBITDA-marge komt in 2026 al één procentpunt lager uit door deze overnames. |
| 5 | Twee sleutelposities wisselen tegelijk | MIDDEN | MIDDEL | uitvoeringsrisico, WACC | CFO Ethan Tandowsky vertrekt op 31 augustus 2026 met alleen een interim-opvolger (Hwa Tsao), en commissarissenvoorzitter Piero Overmars treedt eind 2026 af. Dat gebeurt midden in de integratie van twee overnames en vlak na een verhoging van de capex-verwachting. Een CFO-wissel is op zichzelf normaal; de samenloop verdient aandacht. |
| 6 | Concentratie van de groei bij een kleine groep klanten | MIDDEN | MIDDEL | omzetgroei | Circa 300 klanten leveren ongeveer 60% van de groei. In 2025 vertekende één grote klant het volumebeeld zo sterk dat Adyen twee groeicijfers moest rapporteren: 8% inclusief en 21% exclusief die klant. Het verlies of de tariefheronderhandeling van enkele grote namen raakt de groei onmiddellijk. |
| 7 | Verhoogde capex-behoefte voor datacenters | MIDDEN | MIDDEL | vrije kasstroom | Bij de halfjaarcijfers van 13 augustus 2026 verhoogde Adyen zijn capex-verwachting voor 2026 van "tot 5%" naar **7% van de netto-omzet**, wegens vooruitgeplaatste datacenteraankopen bij aanhoudende leveringsproblemen. Bij een omzet van EUR 2,8 mrd is dat circa EUR 57 mln extra. Blijkt dit structureel in plaats van eenmalig, dan daalt de kasstroomconversie permanent. |
| 8 | Technologische verschuiving naar agentic commerce en stablecoins | LAAG | GROOT | terminale groei | Als AI-agenten een substantieel deel van de aankopen gaan doen en die betalingen over nieuwe rails lopen — Stripe's Tempo ging op 18 maart 2026 live met een machine payments protocol — verschuift de waarde mogelijk naar wie het protocol bezit. Adyen lanceerde op 16 juni 2026 Adyen Agentic, maar alleen in beperkte beschikbaarheid in de Verenigde Staten. De kans op materiële verschuiving binnen vijf jaar is laag, de impact op de terminale waarde zou groot zijn. |

### Verplicht risico-item: pre-IPO financial engineering

- **Zijn er pre-IPO schulden geladen bij gerelateerde partijen?** Niet geconstateerd.
  Adyen was al sinds 2011 winstgevend en haalde in totaal circa USD 266 mln aan
  durfkapitaal op, uitsluitend als eigen vermogen. De balans per 31 december 2017 in het
  prospectus toont geen leningen.
- **Zijn IPO-opbrengsten gebruikt voor schuldaflossing aan insiders?** Nee. Het
  prospectus stelt expliciet: *"The Company will not receive any proceeds from the
  Offering."* De volledige EUR 849 mln ging naar veertien verkopende aandeelhouders.
- **Is er dividend recapitalisatie uitgevoerd vóór de IPO?** Niet geconstateerd. Adyen
  keerde vóór noch na de beursgang dividend uit.
- **Wat is de gecorrigeerde fair value als je dit normaliseert?** Identiek aan de
  ongecorrigeerde: **EUR 1.062,04**. Er is niets te normaliseren. De historische
  kasstroomreeks bevat geen pre-IPO rentelasten en geen schuldaflossingen, en het
  eigen vermogen is uitsluitend door ingehouden winst gegroeid.

---

## 9. These invalide bij

Deze these is weerlegd wanneer de take rate twee halfjaren op rij met meer dan één
basispunt daalt zonder dat de omzet uit financiële producten dat compenseert, wanneer het
volumeverloop boven de 3% uitkomt of een top-tien-klant vertrekt, of wanneer de
EBITDA-marge in 2027 onder de 50% blijft nadat de eenmalige overnamekosten zijn
weggevallen. Een vierde trigger is kapitaalallocatie: nog een overname boven de tien keer
de omzet terwijl er geen aandelen worden ingekocht, ondermijnt het argument dat het
management de kasberg verstandig beheert. Op koersniveau geldt dat boven ongeveer
EUR 1.400 het optimistische scenario volledig is ingeprijsd en er geen veiligheidsmarge
meer over is.

---

## 10. ESG

### Materiële factoren (SASB-gebaseerd)

| Factor | SASB-categorie | Risiconiveau | Financiële impact | DCF-impact |
|---|---|---|---|---|
| Databeveiliging en betalingsintegriteit | Data Security | Hoog | Een groot datalek of langdurige storing raakt direct de kern van het aanbod: klanten kiezen Adyen op betrouwbaarheid. In april 2025 trof een DDoS-aanval in meerdere golven de Europese datacenters; de prestaties verslechterden, maar er zijn géén gegevens verloren en geen consumentendata gecompromitteerd. Over 2025 rapporteert Adyen geen enkel security-incident met dataverlies. | Terminale groei; bij een ernstig incident direct de omzetgroei |
| Energieverbruik datacenters | GHG Emissions / Energy Management | Midden | Adyen draait op eigen datacenters, en de capaciteitsuitbreiding is juist de reden dat de capex-verwachting voor 2026 naar 7% van de omzet ging. Locatiegebonden scope 2-emissies stegen in 2025 naar 17.628 tCO2e (2024: 11.966). Marktgebaseerd bleef de uitstoot met 2.926 tCO2e beperkt doordat 83% van het scope 2-verbruik hernieuwbaar is. | Capex-aanname |
| Regelgeving en toezicht (bankvergunning) | Business Ethics / Regulatory Capture | Midden | Adyen staat als kredietinstelling onder direct toezicht van DNB; elke benoeming van bestuurders en commissarissen vereist DNB-goedkeuring. Het toetsingsvermogen bedroeg ultimo 2025 EUR 4.638 mln, uitsluitend kernkapitaal, bij een liquiditeitsdekkingsratio van 1.717% en een overlevingsperiode van 48 maanden. Er zijn geen boetes of handhavingsmaatregelen gevonden. | WACC (regulatoir kapitaalbeslag beperkt de uitkeerbaarheid van de kas) |
| Financiële criminaliteit en witwassen | Business Ethics | Midden | Als vergunde bank draagt Adyen volledige verantwoordelijkheid voor klantonderzoek en transactiemonitoring over honderdduizenden eindklanten binnen Platforms. Een handhavingsactie zou zowel een boete als reputatieschade betekenen. Adyen heeft een aparte Chief Risk & Compliance Officer in de raad van bestuur. | Terminale groei |
| Menselijk kapitaal | Human Capital | Midden | Het bedrijf is volledig afhankelijk van technisch personeel: 5.020 FTE, 249 netto nieuwe medewerkers in H1 2026, en een salarissom van EUR 756 mln in 2025. De aanwervingsgolf van 2022-2023 kostte 17 procentpunten marge. Beloning is bewust laag ten opzichte van de markt, wat bij krapte een risico is. | EBITDA-marge |
| Klimaatemissies keten | GHG Emissions | Laag | Totale broeikasgasintensiteit 35 tCO2e per EUR mln netto-omzet (2024: 36), marktgebaseerd. De grootste post is scope 3 categorie 1 (ingekochte goederen en diensten, 35.170 tCO2e herzien over 2024). Voor een bedrijf zonder fabrieken is de absolute voetafdruk klein. | Verwaarloosbaar |

- **Eindoordeel ESG**: **GEMIDDELD RISICO**
- **Toelichting**: De milieuvoetafdruk van Adyen is klein en goed gedocumenteerd; met 83%
hernieuwbare stroom in scope 2 en een intensiteit van 35 tCO2e per miljoen euro omzet is
er geen materieel klimaatrisico voor de waardering. Het zwaartepunt ligt bij governance en
databeveiliging, en dat is inherent aan een vergunde bank die geld van derden verwerkt.
Daar staat een sterk profiel tegenover: een aparte risico- en compliancebestuurder, DNB-
toetsing van elke benoeming, een liquiditeitsdekkingsratio van 1.717% en geen enkele boete
of handhavingsmaatregel in acht jaar beursnotering. Het enige operationele incident van
betekenis was de DDoS-aanval van april 2025, zonder datalek. De sociale kant is netjes:
gelijke beloning wordt jaarlijks geaudit, de verhouding CEO tot mediane werknemer is 8:1
en 1% van de netto-omzet gaat naar goede doelen.

---

## 11. Katalysatoren

| Datum ca. | Omschrijving | Richting | Impact |
|---|---|---|---|
| 2026-08 | Toelichting op de H1 2026-cijfers tijdens de earnings call van 13 augustus, met de verhoogde omzetverwachting (21-23%) én de verhoogde capex-verwachting (7% van de netto-omzet) | NEUTRAAL | MIDDEL |
| 2026-09 | Vertrek CFO Ethan Tandowsky per 31 augustus; Hwa Tsao neemt per 1 september interim waar | NEGATIEF | KLEIN |
| 2026-11 | Q3 2026 business update: eerste kwartaal met Talon.One en Orb geconsolideerd, en de eerste toets of de omzetverwachting van 21-23% haalbaar is | BINAIR | MIDDEL |
| 2026-12 | Aftreden van commissarissenvoorzitter Piero Overmars; Herna Verhagen volgt op | NEUTRAAL | KLEIN |
| 2026-12 | Benoeming van een definitieve CFO (extern zoekproces loopt) | BINAIR | MIDDEL |
| 2027-02 | H2 2026-cijfers: eerste volledige halfjaar met de overnames, definitief oordeel over de capex-piek en de EBITDA-marge over 2026 | BINAIR | GROOT |
| 2027-Q1 | Wereldwijde uitrol van Adyen Agentic buiten de Verenigde Staten; concurrentiepositie in agentic commerce wordt zichtbaar | POSITIEF | MIDDEL |
| 2027-2028 | Formele aanname en inwerkingtreding van PSD3/PSR na het politieke akkoord van 27 november 2025 en de ECON-goedkeuring van 5 mei 2026 | NEUTRAAL | KLEIN |
| 2028 | Toetsmoment van de margedoelstelling van boven 55% EBITDA uit de beleggersdag van 11 november 2025 | BINAIR | GROOT |

De belangrijkste binaire gebeurtenis op korte termijn is de rapportage over de tweede
helft van 2026 in februari 2027. Dan blijkt of de capex-verhoging naar 7% eenmalig was,
of de overnames de marge structureel drukken, en of de verhoogde omzetverwachting is
gehaald. Alle drie raken rechtstreeks de basisaannames van dit model: de
kasstroomconversie, de margecurve en de groei in fase 1. De doelstelling van 55%
EBITDA-marge in 2028 is de langere binaire toets; wordt die gemist, dan valt het
basisscenario terug richting het pessimistische.

---

## 12. Fair value — kwantitatief (DCF)

### DCF-invoeren

```
Basis            fcf=984.8  shares=31.566677  net_cash=4898.3  gross_debt=0.0
                 revenue=2573.6  koers=972.10  ipo_jaar=2018
WACC             rf=3.248  erp=4.28  beta=1.58  crp=1.27  size_premium=0.0
                 cost_of_debt_pretax=4.25  tax_rate=23.5
Pessimistisch    g1=7.69   g2=4.06   gt=2.0  wacc_adj=+1.00  kans=30
Basis            g1=17.75  g2=10.70  gt=2.5  wacc_adj=0.00   kans=45
Optimistisch     g1=20.83  g2=12.69  gt=3.0  wacc_adj=-0.75  kans=25
EPV              norm_ebit_margin=48.2  maintenance_capex=152.1  da=152.1
                 norm_ebitda_margin=54.1
Multiples        pe=27.3  pb=5.19  p_fcf=31.2  peg=1.37
Rendement        roic=92.5  earnings_yield=4.62  roc_greenblatt=173.6
Kwalitatief      moat_oordeel=WIDE  moat_categorieen_sterk=3  management_oordeel=STERK
                 capital_allocation=GOED  insider_alignment_pct=2.93
                 roic_wacc_spread_5j_plus=true  structureel_dividend=false  debt_equity=0.069
Eenheid          bedragen in EUR mln; percentages als getal (3.05 = 3,05%)
```

### WACC-componenten
- **Risicovrije rente %**: 3,248
- **Bron risicovrije rente**: Nederlandse 10-jaars staatsobligatie, tradingeconomics.com,
  stand 13 augustus 2026 (3,2480%). De kasstromen worden in euro gemodelleerd en Adyen is
  in Nederland gevestigd, dus dit is de juiste referentie.
- **Type**: spot (nominaal). De huidige rente ligt meer dan 150 basispunten boven het
  tienjaars gemiddelde, dat door de periode met negatieve rentes rond 1% ligt. Conform
  METHODE.md is dat gesignaleerd; er is toch voor spot gekozen omdat de rente-omgeving
  sinds 2023 structureel is genormaliseerd en een terugkeer naar 1% een aanname zou zijn,
  geen waarneming. Een variant met een lagere disconteringsvoet staat in de
  gevoeligheidsmatrix (kolommen 9,53% en 10,28%).
- **ERP (equity risk premium) %**: 4,28
- **Bron ERP**: Damodaran implied equity risk premium per 1 augustus 2026
  (pages.stern.nyu.edu/~adamodar). Dit is de premie voor een volwassen markt.
- **Beta (adjusted, Blume)**: 1,58
- **Bron beta**: ruwe vijfjaars maandbeta van 1,86 (StockAnalysis, 11 augustus 2026),
  gecorroboreerd door Yahoo Finance (1,87) en de quotepagina van StockAnalysis (1,83).
  Blume-correctie: 0,67 × 1,86 + 0,33 = 1,58.
- **Type beta**: 5-jaars maandelijkse regressie. Adyen is meer dan vijf jaar genoteerd en
  voldoende liquide (20-daags gemiddeld volume 126.327 stuks, ruim boven de drempel van
  100.000 uit METHODE.md), dus een regressiebeta is toegestaan en een bottom-up beta
  niet nodig.
- **Country risk premium %**: 1,27
- **Size premium %**: 0,00 (marktkapitalisatie EUR 30,7 mrd, ruim boven EUR 2 mrd)
- **Cost of equity %**: 11,28
- **Schuldkosten na belasting %**: 3,25 (niet gebruikt; zie onder)
- **E/V gewicht %**: 100,0
- **D/V gewicht %**: 0,0
- **WACC %**: 11,28
- **Sector WACC % (referentie Damodaran)**: 9,34 (Software System & Application, VS,
  januari 2026). Adyen ligt daar 1,9 procentpunt boven, volledig verklaard door de hoge
  beta; de sectorwaarde voor "Financial Svcs. (Non-bank & Insurance)" van 5,00% is voor
  een groeibedrijf als dit niet representatief.
- **Illiquiditeitskorting %**: null (niet van toepassing)

**Toelichting op de WACC-opbouw.** Adyen heeft geen rentedragende schuld, dus de WACC is
gelijk aan de kosten van eigen vermogen. De leaseverplichting van EUR 409,4 mln is
behandeld als operationele kostenpost — de leasebetalingen worden in de vrije kasstroom
afgetrokken — en dus niet als schuld in de kapitaalstructuur of in de brug naar de
aandeelhouderswaarde. Zo wordt de lease precies één keer geteld. De country risk premium
is gewogen naar het aandeel in de netto-omzet van H1 2026: 55% EMEA tegen de
Damodaran-regioweging voor West-Europa (0,39%), 27% Noord-Amerika tegen de Verenigde
Staten (0,23%), 11% Azië-Pacific tegen het Aziatische regiogemiddelde (4,24%) en 7%
Latijns-Amerika tegen het regiogemiddelde (7,51%). Dat is aan de conservatieve kant, omdat
Adyens blootstelling binnen die twee regio's geconcentreerd zit in landen met een lager
risicoprofiel (Japan, Singapore, Australië; Brazilië en Mexico). Zonder country risk
premium zou de WACC 10,01% zijn en de fair value in het basisscenario EUR 1.242 in plaats
van EUR 1.062.

### DCF model-specs
- **Model type**: 2-fase (5 + 5 jaar) plus terminale waarde, met mid-year convention
- **FCF-definitie**: FCFF (free cash flow to firm), verdisconteerd tegen de WACC
- **Basis FCF**: 984,8 — genormaliseerde vrije kasstroom over de laatste twaalf maanden
  (H2 2025 + H1 2026)
- **Basis FCF na SBC**: 984,8 — de aandelencompensatie van EUR 53,3 mln drukt de EBITDA
  al; er is dus geen extra correctie nodig
- **FCF-type**: adjusted. Opbouw vanaf de gerapporteerde cijfers: EBITDA EUR 1.343,6 mln,
  plus EUR 155,5 mln rente-inkomsten op merchant-gelden, minus EUR 316,5 mln
  genormaliseerde kasbelasting over EBIT plus die rente, minus EUR 148,0 mln capex, minus
  EUR 49,8 mln leasebetalingen. Ter vergelijking: Adyens eigen FCF-definitie (EBITDA minus
  capex minus lease, vóór belasting) komt over dezelfde twaalf maanden uit op
  EUR 1.145,8 mln.
- **Groei fase 1 %**: 17,75 (jaar 1-5)
- **Groei fase 2 %**: 10,70 (jaar 6-10)
- **Terminal groei %**: 2,5
- **Terminal methode**: Gordon growth, met exit-multiple als kruiscontrole
- **Exit multiple gebruikt (EV/EBITDA)**: 8,3x
- **Bron exit multiple**: mediaan EV/EBITDA van de beursgenoteerde betaalpeers (PayPal
  7,8x, Fiserv 6,9x, Global Payments 11,0x, Worldline 6,7x, Nexi circa 6,5x, Shift4 8,3x,
  Corpay 13,1x), StockAnalysis 13 augustus 2026
- **Terminal value Gordon growth**: 47.061 (nominaal, jaar 10)
- **Terminal value % van totaal**: 55 (basisscenario)
- **Terminal implied EV/EBITDA**: 7,9x
- **Terminal groei consistentie**: Bij een terminale groei van 2,5% en een ROIC in de
  volwassen fase van 20% — een forse verlaging ten opzichte van de huidige 92,5%, omdat de
  huidige spread niet eeuwig houdbaar is — hoort een herinvesteringsvoet van 12,5%
  (g = herinvesteringsvoet × ROIC). Dat is ruim haalbaar: Adyen investeert nu 5 tot 7%
  van de netto-omzet, wat bij een NOPAT-marge van circa 35% neerkomt op 15 tot 20%
  herinvestering. De terminale groei van 2,5% ligt bovendien onder de langetermijn
  nominale bbp-groei van de eurozone en de Verenigde Staten samen.
- **Mid-year convention**: true
- **Aandelen uitstaand (mln)**: 31,566677 (AFM-register geplaatst kapitaal, 30 juni 2026)
- **Nettoschuld huidig**: −4.898,3 (nettokaspositie)

### DCF-toelichting

De waardering rust op drie keuzes die uitleg verdienen. De eerste is de behandeling van de
merchant-gelden. Adyen houdt EUR 12,4 mrd aan kas aan, waarvan EUR 8,1 mrd toebehoort aan
handelaren en op enig moment wordt uitbetaald. Alleen de eigen nettokas van EUR 4.898,3 mln
is bij de bedrijfswaarde opgeteld. De rente op die miljarden — over de laatste twaalf
maanden EUR 272,1 mln — is gesplitst: het deel dat op de float wordt verdiend (57,1%,
oftewel EUR 155,5 mln) is als operationele kasstroom meegenomen omdat het meegroeit met het
volume en structureel bij het bedrijfsmodel hoort, en het deel op de eigen kas is
weggelaten om dubbeltelling te voorkomen, want die kas wordt apart tegen boekwaarde
opgeteld. De tweede keuze is de kasbelasting. Conform de verplichte controle is de
betaalde winstbelasting naast de last in de winst- en verliesrekening gelegd: over 2024,
2025 en de eerste helft van 2026 samen betaalde Adyen 23,2% terwijl de last 24,0% was. Dat
verschil is te klein om te corrigeren; er is met een genormaliseerd tarief van 23,5%
gerekend. De derde keuze is de groei. Fase 1 volgt uit een expliciete
omzetprojectie op basis van Adyens eigen doelstellingen — 20% omzetgroei in 2026 aflopend
naar 17% in 2030, een EBITDA-marge van 52% oplopend naar 56,5%, en capex die van de
verhoogde 7% terugzakt naar 5% — wat neerkomt op een kasstroomgroei van 17,75% per jaar.
De terminale waarde is 55% van het totaal, ruim onder de grens van 75%, en de daaruit
volgende EV/EBITDA van 7,9x sluit goed aan bij de peermediaan van 8,3x.

### 5-jaars projectie (basisscenario, EUR mln)

| Jaar | Omzet | Omzetgroei % | EBITDA | EBITDA-marge % | EBIT | NOPAT | Capex | Lease | FCFF |
|---|---|---|---|---|---|---|---|---|---|
| 2026 | 2.837,0 | 20,0 | 1.475,3 | 52,0 | 1.305,0 | 998,4 | 198,6 | 53,9 | 1.035,0 |
| 2027 | 3.404,4 | 20,0 | 1.838,4 | 54,0 | 1.634,1 | 1.250,1 | 204,3 | 64,7 | 1.304,4 |
| 2028 | 4.051,3 | 19,0 | 2.248,5 | 55,5 | 2.005,4 | 1.534,1 | 222,8 | 77,0 | 1.596,3 |
| 2029 | 4.780,5 | 18,0 | 2.677,1 | 56,0 | 2.390,3 | 1.828,5 | 239,0 | 90,8 | 1.904,5 |
| 2030 | 5.593,2 | 17,0 | 3.160,2 | 56,5 | 2.824,6 | 2.160,8 | 279,7 | 106,3 | 2.229,4 |

Afschrijvingen zijn gemodelleerd op 6,0% van de netto-omzet (recent niveau: 5,9%),
leasebetalingen op 1,9% (recent: 1,9%) en de rente op merchant-gelden nominaal constant op
EUR 155,5 mln — bewust conservatief, omdat volumegroei en een mogelijke rentedaling elkaar
in dat geval opheffen. De mutatie in het werkkapitaal is nul verondersteld; Adyen heeft
EUR 34 mln operationeel werkkapitaal op EUR 2,6 mrd omzet, dus het effect is
verwaarloosbaar.

### Scenarios

| Scenario | FCF-groei fase 1 % | FCF-groei fase 2 % | Terminal % | WACC % | Fair value | Upside % | Kans % |
|---|---|---|---|---|---|---|---|
| Pessimistisch | 7,69 | 4,06 | 2,0 | 12,28 | 590,70 | −39,2 | 30 |
| Basis | 17,75 | 10,70 | 2,5 | 11,28 | 1.062,04 | +9,3 | 45 |
| Optimistisch | 20,83 | 12,69 | 3,0 | 10,53 | 1.414,04 | +45,5 | 25 |

- **Kansgewogen fair value**: **1.008,64** (upside +3,8%)

Het pessimistische scenario gaat uit van een omzetgroei die van 18% in 2026 terugvalt naar
7% in 2030 en 3% in 2035, een EBITDA-marge die op 50-51% blijft steken en capex die op 6%
van de omzet blijft — het beeld dat ontstaat als Stripe het nieuwe enterprisewerk wint en
de tariefdaling doorzet. Het optimistische scenario veronderstelt 22% groei in 2026,
aflopend naar 9% in 2035, een marge die naar 59% loopt en capex terug naar 4,5%. De
kansverdeling is bewust asymmetrisch naar beneden (30/45/25) omdat de tariefdaling een
waargenomen trend van acht jaar is en het optimistische pad afhangt van het slagen van een
productstrategie die pas net is gestart.

### Reverse DCF
- **Impliciete groei %**: 15,97 (kasstroomgroei in jaar 1-5 die bij de huidige koers hoort,
  met fase 2 op 60% daarvan en een terminale groei van 2,5%)
- **Historische FCF CAGR %**: 30,3 (2018-2025)
- **Consensus groei %**: circa 20% netto-omzetgroei per jaar na 2026 volgens de eigen
  doelstelling van de beleggersdag van 11 november 2025; 34 analisten hebben een
  gemiddeld koersdoel van EUR 1.327,35 en een koopadvies (MarketScreener, augustus 2026)
- **Interpretatie**: Bij EUR 972,10 prijst de markt een kasstroomgroei van bijna 16% per
jaar in voor de komende vijf jaar, aflopend naar ruim 9% in de vijf jaar daarna. Dat is
ruim onder de 30% die Adyen tussen 2018 en 2025 werkelijk realiseerde, en ook onder de
circa 20% die het management zelf als doelstelling heeft neergezet. De markt is dus niet
euforisch: hij prijst in dat Adyen zijn eigen plan net niet haalt. Voor een belegger is de
vraag daarmee niet of het aandeel te duur is, maar of hij de doelstelling van 20% geloofwaardiger
vindt dan de 16% die er nu in zit. Het verschil tussen die twee is precies de 9% opwaarts
potentieel in het basisscenario — een smalle marge voor een bedrijf waarvan de tarieven al
acht jaar dalen.

### EPV (Bruce Greenwald)
- **Genormaliseerde EBIT-marge %**: 48,2 (gemiddelde 2021-2025; ter vergelijking: 46,3%
  over de laatste twaalf maanden)
- **Genormaliseerde NOPAT**: 949,9
- **Maintenance capex**: 152,1 (gelijkgesteld aan de afschrijvingen over de laatste twaalf
  maanden)
- **Adjusted earnings power**: 1.030,8 (NOPAT minus onderhoudscapex plus afschrijvingen,
  plus EUR 118,9 mln floatrente na belasting, minus EUR 38,1 mln leasebetalingen na
  belasting)
- **EPV per aandeel**: **444,65** (EUR 9.138,0 mln bedrijfswaarde plus EUR 4.898,3 mln
  eigen nettokas, gedeeld door 31,566677 mln aandelen)
- **Groeipremie %**: 138,8 — de fair value in het basisscenario ligt 138,8% boven de
  waarde zonder groei. De koers zelf ligt 118,6% boven de EPV.

De EPV is voor Adyen een streng maar verhelderend getal. Hij zegt: als Adyen vanaf morgen
niet meer groeit en alleen het huidige verdienvermogen in stand houdt, is het aandeel
EUR 445 waard. Alles daarboven is groei waarvoor je vooruit betaalt. Bij een koers van
EUR 972 betaalt de belegger dus meer dan de helft van de prijs voor toekomstige groei die
nog moet komen. Dat is een wezenlijk andere positie dan bij een bedrijf waar de EPV boven
de koers ligt en de groei gratis is. Het is verdedigbaar bij een rendement op geïnvesteerd
kapitaal van 92,5% — bij zulke rendementen ís groei buitengewoon veel waard — maar het
laat geen ruimte voor teleurstelling.

### Andere methoden
- **DDM uitgevoerd?**: false — Adyen keert geen dividend uit en heeft geen dividendbeleid;
  een dividenddisconteringsmodel is niet toepasbaar.
- **SOTP uitgevoerd?**: false — Adyen is één operationele eenheid op één platform; de drie
  commerciële pijlers delen dezelfde infrastructuur en worden niet apart met kosten en
  kapitaal gerapporteerd.

### Synthese fair value
- **Bandbreedte laag**: 590,70
- **Bandbreedte centraal**: 929,81
- **Bandbreedte hoog**: 1.414,04
- **Methode-gewichten**:
  - DCF: 70%
  - EPV: 15%
  - Multiples: 15%
- **Margin of safety vereist %**: 30
- **Koopniveau** (fair value basis × 0,70): **743,43**
- **Synthese-toelichting**: De drie methoden geven ver uiteenlopende uitkomsten en dat is
op zichzelf informatief. De kasstroomwaardering komt op EUR 1.062, de waardering zonder
groei op EUR 445, en een multiple-benadering op EUR 798 — dat laatste is het midden tussen
de peermediaan van 8,3 keer EBITDA, die past bij de krimpende legacyspelers, en de 21,9
keer die de groeiende fintechs krijgen. De kasstroomwaardering krijgt met 70% verreweg het
zwaarste gewicht omdat Adyen kasstromen genereert die goed te modelleren zijn en omdat de
peergroep te heterogeen is om als anker te dienen. De EPV en de multiples dienen als
tegenwicht en trekken het centrale punt naar EUR 930, iets onder de huidige koers. De
vereiste veiligheidsmarge is op 30% gezet, aan de hoge kant van wat de datakwaliteit
rechtvaardigt — die is uitstekend — maar passend bij het feit dat 55% van de waarde in de
terminale periode zit, dat 19% van de winst uit rente komt die met het rentepeil kan
verdampen, en dat de tarieven al acht jaar dalen. Onder de EUR 745 wordt het aandeel
aantrekkelijk; dat niveau is in de afgelopen twaalf maanden bijna geraakt (dieptepunt
EUR 772,40).

### Gevoeligheid (DCF)

Fair value per aandeel bij variatie in de kasstroomgroei van fase 1 (rijen) en de WACC
(kolommen). Fase 2 is telkens 60% van fase 1; terminale groei 2,5%.

| FCF-groei fase 1 | WACC 9,53% | WACC 10,28% | WACC 11,28% | WACC 12,03% | WACC 12,78% | WACC 13,53% |
|---|---|---|---|---|---|---|
| 8,00% | 801 | 737 | 670 | 628 | 593 | 562 |
| 12,00% | 979 | 894 | 804 | 749 | 703 | 663 |
| 17,75% | 1.320 | 1.194 | **1.061** | 980 | 911 | 852 |
| 21,00% | 1.569 | 1.412 | 1.247 | 1.146 | 1.061 | 988 |
| 24,00% | 1.843 | 1.652 | 1.451 | 1.329 | 1.225 | 1.137 |

De matrix laat zien hoe smal de marge is. De huidige koers van EUR 972 wordt bereikt bij
ongeveer 17% kasstroomgroei tegen 12% kapitaalkosten, of bij 15% groei tegen 11%. Het
verschil tussen een aandeel dat 30% te goedkoop en een aandeel dat 30% te duur is, zit
binnen een bandbreedte van vijf procentpunten groei. Dat is de prijs van een waardering
waarin 55% van de waarde na tien jaar ligt.

---

## 13. Databronnen

### Bronnen-hiërarchie
- **Jaarverslag PDF / IR-pagina** → betrouwbaarheid **HOOG**
- **Beursmelding / prospectus / officieel persbericht** → betrouwbaarheid **HOOG**
- **Aggregator** (StockAnalysis / MarketScreener / Yahoo) → betrouwbaarheid **AGGREGATOR**

### Financiële bronnen (11 jaar historie)

| Jaar | Bron | URL | Betrouwbaarheid |
|---|---|---|---|
| 2015 | Adyen N.V. IPO-prospectus, 4 juni 2018 | https://live.euronext.com/sites/default/files/adyen_prospectus.pdf | HOOG |
| 2016 | Adyen N.V. IPO-prospectus, 4 juni 2018 | https://live.euronext.com/sites/default/files/adyen_prospectus.pdf | HOOG |
| 2017 | Adyen N.V. IPO-prospectus, 4 juni 2018 | https://live.euronext.com/sites/default/files/adyen_prospectus.pdf | HOOG |
| 2018 | Shareholder Letters H1 2019 en H2 2019 (vergelijkende kolommen) + persbericht H1 2018 | https://investors.adyen.com/financials/h1-2019 · https://investors.adyen.com/financials/h2-2019 | HOOG |
| 2019 | Shareholder Letters H1 2019 en H2 2019 | https://investors.adyen.com/financials/h1-2019 · https://investors.adyen.com/financials/h2-2019 | HOOG |
| 2020 | Shareholder Letters H1 2021 en H2 2021 (vergelijkende, herziene kolommen) + persbericht H2 2020 | https://investors.adyen.com/financials/h1-2021 · https://investors.adyen.com/financials/h2-2021 | HOOG |
| 2021 | Shareholder Letters H1 2021 en H2 2021 | https://investors.adyen.com/financials/h1-2021 · https://investors.adyen.com/financials/h2-2021 | HOOG |
| 2022 | Shareholder Letters H1 2023 en H2 2023 (vergelijkende kolommen) + persbericht H2 2022 | https://investors.adyen.com/financials/h1-2023 · https://investors.adyen.com/financials/h2-2023 | HOOG |
| 2023 | Shareholder Letters H1 2023 en H2 2023 | https://investors.adyen.com/financials/h1-2023 · https://investors.adyen.com/financials/h2-2023 | HOOG |
| 2024 | Annual Report 2025 (vergelijkende kolom) + Shareholder Letters H1 2025 en H2 2025 | https://investors.adyen.com/financials/2025 · https://investors.adyen.com/financials/h1-2025 | HOOG |
| 2025 | Annual Report and Consolidated Financial Statements 2025 + Shareholder Letters H1 2025 en H2 2025 | https://investors.adyen.com/financials/2025 · https://investors.adyen.com/financials/h2-2025-4r9rc | HOOG |
| H1 2026 | Adyen H1 2026 Shareholder Letter, 13 augustus 2026 | https://investors.adyen.com/financials/h1-2026-c2a1a | HOOG |

Alle twaalf regels zijn HOOG. De onderliggende PDF's staan op `brand.adyen.com`, dat
robots.txt-geblokkeerd is; de documentpagina's op `investors.adyen.com` zijn wel
toegankelijk en zijn geopend om per jaar het juiste document te identificeren. De PDF's
zelf zijn via de haallijst aangeleverd en met `pdftotext` volledig uitgelezen.

### Jaarverslagen en halfjaarrapporten geraadpleegd

| Jaar | Bron | URL |
|---|---|---|
| 2026 H1 | Adyen H1 2026 Shareholder Letter (13-8-2026) | https://investors.adyen.com/financials/h1-2026-c2a1a |
| 2025 | Adyen Annual Report and Consolidated Financial Statements 2025 | https://investors.adyen.com/financials/2025 |
| 2025 H2 | Adyen H2 2025 Shareholder Letter (12-2-2026) | https://investors.adyen.com/financials/h2-2025-4r9rc |
| 2025 H1 | Adyen H1 2025 Shareholder Letter (14-8-2025) | https://investors.adyen.com/financials/h1-2025 |
| 2023 H2 | Adyen H2 2023 Shareholder Letter (8-2-2024) | https://investors.adyen.com/financials/h2-2023 |
| 2023 H1 | Adyen H1 2023 Shareholder Letter (16-8-2023) | https://investors.adyen.com/financials/h1-2023 |
| 2021 H2 | Adyen H2 2021 Shareholder Letter (8-2-2022) | https://investors.adyen.com/financials/h2-2021 |
| 2021 H1 | Adyen H1 2021 Shareholder Letter (19-8-2021) | https://investors.adyen.com/financials/h1-2021 |
| 2019 H2 | Adyen H2 2019 Shareholder Letter (27-2-2020) | https://investors.adyen.com/financials/h2-2019 |
| 2019 H1 | Adyen H1 2019 Shareholder Letter (21-8-2019) | https://investors.adyen.com/financials/h1-2019 |
| 2018 | Adyen N.V. IPO-prospectus (4-6-2018) | https://live.euronext.com/sites/default/files/adyen_prospectus.pdf |

### Beursmeldingen en persberichten geraadpleegd

| Datum | Omschrijving | URL |
|---|---|---|
| 2026-08-13 | H1 2026 Shareholder Letter en earnings call | https://investors.adyen.com/events/earnings-call-h1-2026 |
| 2026-07-01 | Afronding overnames Talon.One en Orb; benoeming Gayathri Rajan (CPO) en Hwa Tsao (interim-CFO) | https://www.adyen.com/press-and-media/adyen-closes-talonone-and-orb-acquisitions-announces-leadership-updates |
| 2026-06-16 | Lancering Adyen Agentic | https://www.adyen.com/press-and-media/adyen-agentic |
| 2026-06-11 | Overname Orb voor USD 335 mln | https://www.adyen.com/press-and-media/jtrg4qd7j3p4rj |
| 2026-05-27 | CFO Ethan Tandowsky treedt af per 31-8-2026 | https://www.adyen.com/press-and-media/adyen-cfo-ethan-tandowsky-to-step-down-to-pursue-opportunity-outside-fintech |
| 2026-05-06 | Q1 2026 business update | https://www.adyen.com/press-and-media/adyen-publishes-q1-2026-business-update-4gyhh5 |
| 2026-04-23 | Overname Talon.One voor EUR 750 mln | https://www.adyen.com/press-and-media/2i4hnm0pfcpvc7 |
| 2026-02-12 | H2 2025 financiële resultaten | https://www.adyen.com/press-and-media/adyen-publishes-h2-2025-financial-results-3pgu2 |
| 2026-01-20 | Nominatie Herna Verhagen als commissaris en voorzitter | https://www.adyen.com/press-and-media/adyen-nominates-herna-verhagen-as-supervisory-board-member-and-chair |
| 2025-11-11 | Investor Day 2025: doelstellingen 2028 | https://www.adyen.com/press-and-media/adyen-hosts-investor-day-2025-in-amsterdam |
| 2025-08-14 | H1 2025 financiële resultaten | https://www.adyen.com/press-and-media/adyen-publishes-h1-2025-financial-results |
| 2025-02 | H2 2024 financiële resultaten | https://www.adyen.com/press-and-media/adyen-publishes-h2-2024-financial-results |
| 2024-08-15 | H1 2024 financiële resultaten | https://www.adyen.com/press-and-media/adyen-publishes-h1-2024-financial-results |
| 2024-02-08 | H2 2023 financiële resultaten | https://www.adyen.com/press-and-media/adyen-publishes-h2-2023-financial-results |
| 2023-08-17 | H1 2023 financiële resultaten | https://www.adyen.com/press-and-media/adyen-publishes-h1-2023-financial-results |
| 2023-02-08 | H2 2022 financiële resultaten | https://www.adyen.com/press-and-media/adyen-publishes-h2-2022-financial-results |
| 2023-02-08 | Ingo Uytdehaage co-CEO, Ethan Tandowsky CFO | https://www.adyen.com/press-and-media/ingo-uytdehaage-new-co-ceo-ethan-cfo |
| 2021-08-19 | H1 2021 financiële resultaten | https://www.adyen.com/press-and-media/adyen-publishes-h1-2021-financial-results |
| 2021-02-10 | H2 2020 financiële resultaten | https://www.adyen.com/press-and-media/adyen-publishes-h2-2020-financial-results |
| 2020-09-23 | Medeoprichter en CTO Arnout Schuijff treedt af | https://www.adyen.com/press-and-media/co-founder-and-cto-arnout-schuijff-to-step-down |
| 2020-08-21 | Vier bestuursleden verkopen circa 15% van hun belangen | https://www.adyen.com/press-and-media/4-members-of-the-adyen-management-board-complete-sale-of-approximately-15-of-their-holdings |
| 2020-08 | H1 2020 financiële resultaten | https://www.adyen.com/press-and-media/adyen-publishes-h1-2020-financial-results |
| 2019-12-17 | Wereldwijde overeenkomst met McDonald's | https://www.adyen.com/press-and-media/adyen-announces-international-mobile-app-payments-agreement-with-mcdonalds |
| 2019-08-21 | H1 2019 financiële resultaten | https://www.adyen.com/press-and-media/h1-2019-financial-results |
| 2018-08 | H1 2018 financiële resultaten | https://www.adyen.com/press-and-media/adyen-h1-2018-financial-results |
| 2018-06-13 | IPO geprijsd op EUR 240 per aandeel | https://www.adyen.com/press-and-media/adyen-ipo-priced-at-240-per-share |
| 2026-08-13 | AGM 2026: alle besluiten aangenomen | https://www.adyen.com/press-and-media/all-resolutions-adopted-at-2026-agm |

### IPO-prospectus
- **Geraadpleegd?**: true
- **URL**: https://live.euronext.com/sites/default/files/adyen_prospectus.pdf
- **Pre-IPO data beschikbaar?**: true
- **Pre-IPO bron**: het prospectus van 4 juni 2018 bevat de geconsolideerde kerncijfers
  over 2015, 2016 en 2017 (bruto-omzet, netto-omzet, EBITDA, nettowinst, totale activa,
  eigen vermogen, operationele kasstroom, capex) alsmede de volledige
  aandeelhoudersstructuur van vóór de beursgang en de voorwaarden van de aanbieding.

### Non-GAAP
- **Gebruikt?**: true
- **Toelichting**: Adyen definieert netto-omzet, EBITDA en vrije kasstroom als non-IFRS
  maatstaven in noot 1.1 van de jaarrekening. Netto-omzet is de bruto-omzet minus de
  doorbelaste kosten van financiële instellingen en de kostprijs van verkochte goederen,
  plus netto rentebaten uit de embedded-financial-productssuite. EBITDA is het resultaat
  vóór netto financiële baten en belastingen, plus afschrijvingen. Vrije kasstroom is
  EBITDA minus capex minus leasebetalingen exclusief rente. Alle drie zijn in dit rapport
  overgenomen omdat ze eenduidig zijn gedefinieerd en jaar op jaar consistent worden
  toegepast; de IFRS-cijfers (bedrijfsresultaat, nettowinst, kasstroom) staan er telkens
  naast. Voor de waardering is teruggerekend naar een op IFRS gebaseerde vrije kasstroom
  na kasbelasting.

### Ontbrekende data
- Voor 2015 en 2016 is geen verwerkt volume gepubliceerd; die cellen blijven leeg.
- Voor 2015-2017 geeft het prospectus alleen kerncijfers. Balansposten anders dan totale
  activa en eigen vermogen, kasstroomdetails, EPS, aandelenaantal, EBIT en afschrijvingen
  ontbreken; die cellen blijven leeg. Voor die drie jaren is dus geen EBIT-marge,
  rendement op geïnvesteerd kapitaal of aandeelhoudersrendement berekend.
- Adyen rapporteert geen brutowinst of brutomarge; dat concept bestaat niet in zijn
  resultatenrekening, waar de doorbelaste kosten direct van de omzet worden afgetrokken.
- Aandelencompensatie is voor 2018, 2019 en 2022 niet in beide halfjaarrapporten
  gespecificeerd; die cellen blijven leeg.
- Een actueel free-floatpercentage is niet uit een primaire bron te halen. De Euronext-
  productpagina rendert die gegevens client-side. Bij de beursgang bedroeg de free float
  12,0%; op basis van de bekende meldingen is de huidige float hoger, maar een exact
  cijfer is niet verifieerbaar en daarom weggelaten.
- Een volledige actuele lijst van substantiële deelnemingen is niet te genereren: het
  AFM-zoekformulier vereist JavaScript. De opgenomen posten zijn individueel geopende
  meldingsrecords met hun meldingsdatum; ze zijn geen momentopname per 13 augustus 2026.
- Het marktaandeel van Adyen als percentage is niet verifieerbaar. De Nilson-ranglijsten
  met marktaandelen per acquirer zitten achter een betaalmuur, en een acquiring-specifieke
  marktomvang publiceren McKinsey en BCG niet. De TAM/SAM-cellen zijn daarom deels leeg.
- Het lidmaatschap van de EURO STOXX 50 kon niet uit een primaire bron worden bevestigd
  (de officiële STOXX-componentenlijst is robots-geblokkeerd) en is daarom niet vermeld.
  Het AEX-lidmaatschap is wel primair bevestigd via de Euronext-indexfactsheet.
- Het volledige remuneratierapport als los document was niet op te halen; de
  beloningsgegevens komen uit het remuneratiehoofdstuk van het jaarverslag 2025 zelf.
- De operationele kasstroom over 2015-2017 komt uit het prospectus, maar zonder de
  onderliggende mutatie in merchant-gelden; een correctie voor de float is voor die jaren
  daarom niet mogelijk.

### Peildatum analyse
- 2026-08-13 (koers EUR 972,10, 09:01 CET)

**Noot over de koersbepaling.** De cijfers over de eerste helft van 2026 zijn op de ochtend
van de peildatum vóór beurs gepubliceerd en het aandeel opende fors hoger. Op 13 augustus
om 09:01 CET noteerde het EUR 972,10, +6,82% ten opzichte van de slotkoers van
12 augustus (EUR 910,00, bevestigd door drie onafhankelijke bronnen: beursduivel.be,
beurs.nl en investing.com, alle met dagbereik EUR 898,20-921,60). Persberichten van
diezelfde ochtend meldden een stijging van "ruim 10 procent", wat op een intradagkoers
rond EUR 1.000 wijst. De gebruikte koers is dus een openingskoers, geen slotkoers.
Robuustheidstoets van het eindoordeel over die bandbreedte: bij EUR 910,00 bedraagt het
opwaarts potentieel op het basisscenario +16,7% en scoort de kasstroomwaardering 4 in
plaats van 3, waardoor de totaalscore op 30 uitkomt — nog steeds HOLD. Bij EUR 1.050 is
het potentieel +1,1%, score 3, totaalscore 29 — eveneens HOLD. Het oordeel is over de hele
gemeten spreiding stabiel; alleen de exacte upside verschuift. Werk de koers bij de
volgende actualisatie bij naar de slotkoers van 13 augustus.

---

## 14. Volledige bronnen-lijst

| Titel | URL | Type |
|---|---|---|
| Adyen H1 2026 Shareholder Letter (13-8-2026) | https://investors.adyen.com/financials/h1-2026-c2a1a | jaarverslag |
| Adyen Annual Report and Consolidated Financial Statements 2025 | https://investors.adyen.com/financials/2025 | jaarverslag |
| Adyen H2 2025 Shareholder Letter | https://investors.adyen.com/financials/h2-2025-4r9rc | jaarverslag |
| Adyen H1 2025 Shareholder Letter | https://investors.adyen.com/financials/h1-2025 | jaarverslag |
| Adyen H2 2023 Shareholder Letter | https://investors.adyen.com/financials/h2-2023 | jaarverslag |
| Adyen H1 2023 Shareholder Letter | https://investors.adyen.com/financials/h1-2023 | jaarverslag |
| Adyen H2 2021 Shareholder Letter | https://investors.adyen.com/financials/h2-2021 | jaarverslag |
| Adyen H1 2021 Shareholder Letter | https://investors.adyen.com/financials/h1-2021 | jaarverslag |
| Adyen H2 2019 Shareholder Letter | https://investors.adyen.com/financials/h2-2019 | jaarverslag |
| Adyen H1 2019 Shareholder Letter | https://investors.adyen.com/financials/h1-2019 | jaarverslag |
| Adyen N.V. IPO-prospectus, 4 juni 2018 | https://live.euronext.com/sites/default/files/adyen_prospectus.pdf | jaarverslag |
| Adyen overzichtspagina financiële publicaties | https://investors.adyen.com/financials | beurswebsite |
| Adyen IPO geprijsd op EUR 240 per aandeel | https://www.adyen.com/press-and-media/adyen-ipo-priced-at-240-per-share | beursmelding |
| Adyen publishes H2 2025 financial results | https://www.adyen.com/press-and-media/adyen-publishes-h2-2025-financial-results-3pgu2 | beursmelding |
| Adyen publishes Q1 2026 Business Update | https://www.adyen.com/press-and-media/adyen-publishes-q1-2026-business-update-4gyhh5 | beursmelding |
| Adyen publishes H2 2024 financial results | https://www.adyen.com/press-and-media/adyen-publishes-h2-2024-financial-results | beursmelding |
| Adyen publishes H1 2024 financial results | https://www.adyen.com/press-and-media/adyen-publishes-h1-2024-financial-results | beursmelding |
| Adyen publishes H2 2023 financial results | https://www.adyen.com/press-and-media/adyen-publishes-h2-2023-financial-results | beursmelding |
| Adyen publishes H1 2023 financial results | https://www.adyen.com/press-and-media/adyen-publishes-h1-2023-financial-results | beursmelding |
| Adyen publishes h2 2022 financial results | https://www.adyen.com/press-and-media/adyen-publishes-h2-2022-financial-results | beursmelding |
| Adyen publishes H1 2022 financial results | https://www.adyen.com/press-and-media/adyen-publishes-h1-2022-financial-results | beursmelding |
| Adyen publishes H1 2021 financial results | https://www.adyen.com/press-and-media/adyen-publishes-h1-2021-financial-results | beursmelding |
| Adyen Publishes H2 2020 Financial Results | https://www.adyen.com/press-and-media/adyen-publishes-h2-2020-financial-results | beursmelding |
| Adyen publishes H1 2020 Financial Results | https://www.adyen.com/press-and-media/adyen-publishes-h1-2020-financial-results | beursmelding |
| Adyen H1 2019 Financial Results | https://www.adyen.com/press-and-media/h1-2019-financial-results | beursmelding |
| Adyen H1 2018 Financial Results | https://www.adyen.com/press-and-media/adyen-h1-2018-financial-results | beursmelding |
| Adyen to acquire Talon.One (EUR 750 mln) | https://www.adyen.com/press-and-media/2i4hnm0pfcpvc7 | beursmelding |
| Adyen to acquire Orb (USD 335 mln) | https://www.adyen.com/press-and-media/jtrg4qd7j3p4rj | beursmelding |
| Adyen closes Talon.One and Orb acquisitions; leadership updates | https://www.adyen.com/press-and-media/adyen-closes-talonone-and-orb-acquisitions-announces-leadership-updates | beursmelding |
| Adyen CFO Ethan Tandowsky to step down | https://www.adyen.com/press-and-media/adyen-cfo-ethan-tandowsky-to-step-down-to-pursue-opportunity-outside-fintech | beursmelding |
| Adyen nominates Herna Verhagen as Supervisory Board member and Chair | https://www.adyen.com/press-and-media/adyen-nominates-herna-verhagen-as-supervisory-board-member-and-chair | beursmelding |
| Adyen hosts Investor Day 2025 in Amsterdam | https://www.adyen.com/press-and-media/adyen-hosts-investor-day-2025-in-amsterdam | beursmelding |
| All resolutions adopted at 2026 AGM | https://www.adyen.com/press-and-media/all-resolutions-adopted-at-2026-agm | beursmelding |
| Adyen announces Adyen Agentic | https://www.adyen.com/press-and-media/adyen-agentic | beursmelding |
| Adyen and Uber expand global partnership | https://www.adyen.com/press-and-media/adyen-and-uber-expand-global-partnership-to-power-new-markets-launch-uber-kiosks | beursmelding |
| Adyen mobile app payments agreement with McDonald's | https://www.adyen.com/press-and-media/adyen-announces-international-mobile-app-payments-agreement-with-mcdonalds | beursmelding |
| Ingo Uytdehaage to become co-CEO, Ethan Tandowsky CFO | https://www.adyen.com/press-and-media/ingo-uytdehaage-new-co-ceo-ethan-cfo | beursmelding |
| Co-founder and CTO Arnout Schuijff to step down | https://www.adyen.com/press-and-media/co-founder-and-cto-arnout-schuijff-to-step-down | beursmelding |
| Four Management Board members complete sale of ~15% of holdings | https://www.adyen.com/press-and-media/4-members-of-the-adyen-management-board-complete-sale-of-approximately-15-of-their-holdings | beursmelding |
| Adyen kantorenoverzicht | https://www.adyen.com/offices | beurswebsite |
| Adyen earnings call H1 2026 (13-8-2026, 15:00 CEST) | https://investors.adyen.com/events/earnings-call-h1-2026 | beurswebsite |
| AFM-register geplaatst kapitaal Adyen N.V. (30-6-2026) | https://www.afm.nl/en/sector/registers/meldingenregisters/geplaatst-kapitaal/details?id=196519 | beurswebsite |
| AFM-vergunningenregister: DNB-vergunning kredietinstelling 25-4-2017 | https://www.afm.nl/en/sector/registers/vergunningenregisters/beleggingsondernemingen/details?id=E8C39F96-795E-E311-B05D-005056BE011E | beurswebsite |
| AFM substantiële deelneming Temasek Holdings (5-6-2025) | https://www.afm.nl/en/sector/registers/meldingenregisters/substantiele-deelnemingen/details?id=175839 | beurswebsite |
| AFM substantiële deelneming BlackRock (13-9-2021) | https://www.afm.nl/en/sector/registers/meldingenregisters/substantiele-deelnemingen/details?id=115624 | beurswebsite |
| AFM substantiële deelneming Pentavest (Index Ventures, 13-6-2018) | https://www.afm.nl/en/sector/registers/meldingenregisters/substantiele-deelnemingen/details?id=60101 | beurswebsite |
| Euronext AEX Index Factsheet (31-3-2026) | https://live.euronext.com/sites/default/files/documentation/index-fact-sheets/AEX_Index_Factsheet.pdf | beurswebsite |
| Euronext productpagina Adyen (ISIN NL0012969182) | https://live.euronext.com/en/product/equities/NL0012969182-XAMS | beurswebsite |
| FinancialReports.eu — Adyen N.V. filings | https://financialreports.eu/companies/adyen-nv/ | databron |
| StockAnalysis.com — Adyen quote en statistics | https://stockanalysis.com/quote/ams/ADYEN/statistics/ | databron |
| StockAnalysis.com — Adyen financials | https://stockanalysis.com/quote/ams/ADYEN/financials/ | databron |
| StockAnalysis.com — PayPal statistics | https://stockanalysis.com/stocks/PYPL/statistics/ | databron |
| StockAnalysis.com — Fiserv statistics | https://stockanalysis.com/stocks/FI/statistics/ | databron |
| StockAnalysis.com — Global Payments statistics | https://stockanalysis.com/stocks/GPN/statistics/ | databron |
| StockAnalysis.com — Worldline statistics | https://stockanalysis.com/quote/epa/WLN/statistics/ | databron |
| StockAnalysis.com — Nexi statistics | https://stockanalysis.com/quote/bit/NEXI/statistics/ | databron |
| StockAnalysis.com — Block statistics | https://stockanalysis.com/stocks/XYZ/statistics/ | databron |
| StockAnalysis.com — Shift4 statistics | https://stockanalysis.com/stocks/FOUR/statistics/ | databron |
| StockAnalysis.com — Toast statistics | https://stockanalysis.com/stocks/TOST/statistics/ | databron |
| StockAnalysis.com — Corpay statistics | https://stockanalysis.com/stocks/CPAY/statistics/ | databron |
| MarketScreener — Adyen analistenconsensus | https://www.marketscreener.com/quote/stock/ADYEN-N-V-44211922/consensus/ | analistenrapport |
| MarketScreener — Adyen insidertransacties | https://www.marketscreener.com/quote/stock/ADYEN-N-V-44211922/company-insider-trading/ | databron |
| Simply Wall St — Adyen ownership | https://simplywall.st/stocks/nl/diversified-financials/ams-adyen/adyen-shares/ownership | databron |
| Damodaran — implied equity risk premium (1-8-2026) | https://pages.stern.nyu.edu/~adamodar/New_Home_Page/home.htm | onderzoeksrapport |
| Damodaran — country risk premiums (5-1-2026) | https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html | onderzoeksrapport |
| Damodaran — cost of capital by industry (januari 2026) | https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/wacc.html | onderzoeksrapport |
| Tradingeconomics — Nederlandse 10-jaars staatsobligatie | https://tradingeconomics.com/netherlands/government-bond-yield | databron |
| McKinsey Global Payments Report 2025 (26-9-2025) | https://www.mckinsey.com/industries/financial-services/our-insights/global-payments-report | onderzoeksrapport |
| BCG — Reshaping the global payments landscape (22-9-2025) | https://www.bcg.com/press/22september2025-reshape-global-payments-landscape | onderzoeksrapport |
| Nilson Report — global brand cards midyear 2025 | https://nilsonreport.com/articles/global-brand-cards-worldwide-midyear-2025/ | onderzoeksrapport |
| Stripe 2025 update (24-2-2026) | https://stripe.com/newsroom/news/stripe-2025-update | nieuwsartikel |
| Checkout.com jaarupdate 2025 (24-2-2026) | https://www.checkout.com/newsroom/checkout-com-returns-to-full-year-profitability-and-surpasses-300b-in-volume-as-it-positions-for-the-era-of-agentic-commerce | nieuwsartikel |
| Global Payments rondt overname Worldpay af (12-1-2026) | https://investors.globalpayments.com/news-events/press-releases/detail/498/global-payments-completes-acquisition-of-worldpay-and | nieuwsartikel |
| Worldline Q1 2026 en kapitaalverhoging (28-4-2026) | https://investors.worldline.com/en/home/news-events/financial-press-releases/2026/pr-2026_04_28_01 | nieuwsartikel |
| PayPal Q2 2026 earnings release (28-7-2026) | https://s205.q4cdn.com/875401827/files/doc_financials/2026/q2/PYPL-2Q-26-Earnings-Release.pdf | nieuwsartikel |
| Fiserv verlaagt jaarverwachting (7-8-2026) | https://paymentexpert.com/2026/08/07/fiserv-cuts-2026-outlook-q2-earnings/ | nieuwsartikel |
| Europees Parlement — herziening betaalregelgeving (PSD3/PSR) | https://www.europarl.europa.eu/legislative-train/theme-an-economy-that-works-for-people/file-revision-of-eu-rules-on-payment-services | onderzoeksrapport |
| Rechter keurt Visa/Mastercard-schikking van USD 38 mrd goed (9-6-2026) | https://www.paymentsdive.com/news/court-approves-visa-mastercard-settlement/822440/ | nieuwsartikel |
| Stripe-gesteunde blockchain Tempo live (18-3-2026) | https://www.coindesk.com/tech/2026/03/18/stripe-led-payments-blockchain-tempo-goes-live-with-protocol-for-ai-agents | nieuwsartikel |
| Adyen verhoogt omzetverwachting na sterke eerste helft (13-8-2026) | https://www.globalbankingandfinance.com/adyen-lifts-2026-revenue-outlook-strong-first-half/ | nieuwsartikel |
| Adyen nieuwe langetermijndoelen Investor Day (11-11-2025) | https://www.investing.com/news/stock-market-news/adyen-gains-after-unveiling-new-20-longterm-growth-and-higher-margin-targets-4348357 | nieuwsartikel |
| CNBC — Adyen verliest 39% na H1 2023-cijfers (17-8-2023) | https://www.cnbc.com/2023/08/17/adyen-earnings-h1-2023-stock-down-28percent-after-record-low-sales-growth.html | nieuwsartikel |
| CNBC — Adyen faces big challenges (21-8-2023) | https://www.cnbc.com/2023/08/21/adyen-europes-fintech-darling-faces-big-challenges.html | nieuwsartikel |
| PYMNTS — Adyen verlaagt doelstellingen op Investor Day (8-11-2023) | https://www.pymnts.com/earnings/2023/adyen-updates-financial-objectives-amid-investor-scrutiny/ | nieuwsartikel |
| eBay kiest Adyen als primaire betaalpartner (31-1-2018) | https://www.ebayinc.com/stories/news/ebay-to-intermediate-payments-on-its-marketplace-platform/ | nieuwsartikel |
| Crowdfund Insider — DDoS-aanval op Adyen (28-4-2025) | https://www.crowdfundinsider.com/2025/04/239003-payments-provider-enabler-adyen-hit-by-ddos-attack-targets-eu-data-centers-degrading-performance/ | nieuwsartikel |
| Index Ventures — Adyen, a global success born in Amsterdam | https://www.indexventures.com/perspectives/adyen-a-global-success-born-in-amsterdam/ | nieuwsartikel |
| TechCrunch — Iconiq investeert in Adyen bij USD 2,3 mrd (30-9-2015) | https://techcrunch.com/2015/09/30/adyen-iconiq/ | nieuwsartikel |
| CNBC — alles over de Adyen-beursgang (13-6-2018) | https://www.cnbc.com/2018/06/13/adyen-ipo-everything-you-need-to-know-about-the-8-billion-fintech-company.html | nieuwsartikel |
| Forbes — profiel Pieter van der Does | https://www.forbes.com/profile/pieter-van-der-does/ | nieuwsartikel |

---

## 15. Update-historie

| Datum | Versie | Wijziging |
|---|---|---|
| 2026-08-13 | 1.0 | Eerste publicatie. Volledige analyse vanaf nul opgebouwd, inclusief de halfjaarcijfers over H1 2026 die diezelfde ochtend zijn gepubliceerd. Alle twaalf boekjaren (2015 t/m H1 2026) op betrouwbaarheid HOOG. |

---

## Opmerkingen voor Claude Code

1. **`brand.adyen.com` is structureel robots.txt-geblokkeerd.** Alle jaarverslagen en
   shareholder letters van Adyen staan daar. De documentpagina's op `investors.adyen.com`
   zijn wél leesbaar en geven per boekjaar de juiste asset-URL, maar de PDF zelf is niet op
   te halen. Zet dit domein bij een volgende update meteen op de haallijst; het kost anders
   een half uur zoeken. De persberichten op `www.adyen.com/press-and-media/...` zijn wél
   fetchbaar en bevatten per halfjaar de kerncijfers (netto-omzet, EBITDA, marge, volume,
   capex als percentage, kasstroomconversie) — voldoende voor een kruiscontrole, niet voor
   de balans en de kasstroom.
2. **Adyen publiceert geen jaarrekening in zijn halfjaarrapporten, maar wél een volledige
   halfjaarrekening.** Een boekjaar reconstrueer je exact als H1 + H2. Vraag dus altijd
   beide brieven van een jaar op, niet alleen de H2. Vijf H1- plus vijf H2-brieven dekken
   acht boekjaren volledig.
3. **De H1 2020-cijfers zijn later herzien** (netto-omzet van EUR 279,9 naar EUR 304,8 mln,
   EBITDA van EUR 140,9 naar EUR 165,7 mln). Gebruik altijd de vergelijkende kolom uit de
   H1 2021-brief, niet het oorspronkelijke persbericht van augustus 2020.
4. **De enterprise value van externe databronnen is voor Adyen fout.** StockAnalysis geeft
   EUR 18,78 mrd door de volledige kaspositie van EUR 12,4 mrd af te trekken, terwijl
   EUR 8,1 mrd daarvan merchant-gelden zijn. De juiste EV is EUR 25,8 mrd. Dit geldt voor
   elke betaalverwerker met een eigen bankvergunning.
5. **Openstaande haallijst: geen.** Alle tien de shareholder letters en het jaarverslag
   2025 zijn geleverd en verwerkt. Voor een volgende update volstaan de H2 2026- en
   H1 2027-brief.
6. **Betrouwbaar gebleken**: `live.euronext.com/sites/default/files/adyen_prospectus.pdf`
   (IPO-prospectus, direct fetchbaar), `afm.nl` detail-URL's van de meldingsregisters (de
   zoekformulieren niet — die zijn JavaScript), `tradingeconomics.com` voor de
   staatsrente mét datum, en `pages.stern.nyu.edu/~adamodar/` voor ERP, country risk
   premiums en sector-WACC.

---

## Afronding (check voor je oplevert)

- [x] Elk cijfer in de tabellen heeft een bron-voetnoot of staat in de bronnen-tabel
- [x] De recente 5 jaren in sectie 13 zijn allemaal HOOG (alle 12 jaren zijn HOOG)
- [x] Geen enum-variant verzonnen — alleen waarden uit de template
- [x] Scorekaart heeft 9 frameworks, totaal 29 en max 45 kloppen
- [x] Synthese-toelichting aanwezig (sectie 12)
- [x] Non-GAAP adjustments expliciet toegelicht (sectie 3 en 13)
- [x] IPO-carve-out: Adyen is 8,2 jaar beursgenoteerd; de pre-IPO-toets staat volledig
      uitgewerkt in sectie 8 en de ontbrekende pre-IPO-cellen staan in sectie 13
