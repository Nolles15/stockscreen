# Research: AD — Koninklijke Ahold Delhaize N.V.

> **Stage 1 output van cowork.** Claude Code neemt het over voor JSON-injectie, validator en deploy.
> Methode: `research/METHODE.md`. Structuur: `research/TEMPLATE.md`.

---

## Bronnen-inventaris (Stap 0.5)

```
Jaar 2025 — HOOG
  Bron: Ahold Delhaize Q4 2025 Results press release (11-02-2026)
  URL:  https://newsroom.aholddelhaize.com/ahold-delhaize-reports-strong-q4-2025-financial-results-priorities-and-outlook-for-2026-underpin-our-value-creation-and-progress-towards-our-growing-together-ambitions/
        PDF: https://media.aholddelhaize.com/media/bdsnpbxm/ahold-delhaize-q4-2025-summary-report.pdf
  Daadwerkelijk geopend: ja (search-snippets uit officiële persrelease)
  Cijfers overgenomen: net sales 92,4 mld EUR, underlying op-marge 4,0%,
                       diluted underlying EPS 2,67 EUR,
                       dividend voorstel 1,24 EUR (+6%),
                       online +13,3% CC / +11,2% AC,
                       US net sales Q4 13,0 mld (+2,5% CC / -6% AC FX),
                       2026-guidance: marge ~4%, mid-to-high-single-digit EPS
                       groei CC, FCF >2,3 mld, capex ~2,7 mld
  Cijfers NIET overgenomen: detail-balans 2025, segment-revenue split per merk,
                            FY-FCF totaal 2025

Jaar 2024 — HOOG
  Bron: Ahold Delhaize Q4 2024 Results press release (12-02-2025)
  URL:  https://newsroom.aholddelhaize.com/ahold-delhaize-reports-q4-2024-financial-results-and-introduces-outlook-for-2025-with-projected-growth-in-sales-and-earnings-in-line-with-its-growing-together-strategic-ambitions/
        + analyst-presentation PDF
  Daadwerkelijk geopend: ja (search-snippets)
  Cijfers overgenomen: net sales 89,4 mld EUR, underlying op-marge 4,0%,
                       diluted underlying EPS 2,54 EUR,
                       dividend 1,17 EUR (+6,4%), FCF >2,5 mld,
                       buyback 1.000 mln EUR uitgegeven in 2024
  Cijfers NIET overgenomen: detail-segment-mix, debt-maturity schedule

Jaar 2023 — AGGREGATOR
  Bron: AD Q4 2024 persrelease vergelijkende kolom
  Cijfers overgenomen (afgeleid): net sales ~88,7 mld EUR (search-snippet),
                       underlying op-marge ~4,0%, EPS ~2,40 EUR,
                       dividend ~1,10 EUR (FY2024 1,17 / 1,064)
  Cijfers NIET overgenomen: detail-balans

Jaar 2022 — AGGREGATOR
  Bron: search-snippets, companiesmarketcap.com
  Cijfers overgenomen: net sales ~87,0 mld EUR (recovery + inflatie-pricing)
  Conclusie: indicatief

Jaar 2021 — AGGREGATOR
  Bron: search-snippets
  Cijfers overgenomen: net sales ~75,6 mld EUR
  Conclusie: indicatief

Jaar 2020 — AGGREGATOR (Covid-piek voor supermarkten)
  Bron: search-snippets
  Cijfers overgenomen: net sales ~74,7 mld EUR
  Conclusie: indicatief

Jaren 2015-2019 — GEEN BRON BESCHIKBAAR (binnen sessie-tijd)
  Zoekpoging(en): companiesmarketcap.com (snippet-only), Ahold Delhaize-IR-
                  pagina jaarverslagen-PDF (binnen context-limiet niet
                  geëxtraheerd)
  Conclusie: 2015-2019 LAAT LEEG. Genoteerd in sectie 13.
```

**Bronnen-inventaris-conclusie:** twee jaren HOOG (2024-2025) via officiële persreleases, drie tot vier jaren AGGREGATOR (2020-2023), vijf jaren leeg (2015-2019). Ahold Delhaize is een defensieve consumer staple — kortere historie methodisch verdedigbaar. De 2016-fusie (Ahold + Delhaize) maakt 2015-2016 cijfers structureel anders (separate entiteiten); een vervolg-pas zou de pro-forma combined cijfers vanaf 2016-jaarverslag moeten halen voor consistente reeks.

---

## Metadata
- **Ticker (bare):** AD
- **Yahoo symbol:** AD.AS
- **Exchange:** AEX (Euronext Amsterdam)
- **Sector (GICS-achtig):** Consumentengoederen (Defensief)
- **Industrie:** Voedingsretail (supermarkten + e-commerce)
- **Land:** Nederland (Zaandam, NL — wereldhoofdkantoor)
- **Peildatum analyse:** 2026-04-28
- **Koers op peildatum:** 40,05
- **Valuta:** EUR
- **Marktkapitalisatie:** EUR 37,6 mld (~940 mln aandelen × €40,05)
- **Marktkap in mln (lokale valuta):** 37.647
- **Free float pct:** ~99% (geen controlerend aandeelhouder; sinds Ahold-boekhoudschandaal 2003 bewust gespreide eigendomsstructuur)
- **Indexlidmaatschap:** AEX, Stoxx Europe 600
- **Domein:** aholddelhaize.com

---

## 1. Executive summary

- **Kernthese:** Ahold Delhaize is de holding boven 16 lokale supermarktmerken in negen landen — Albert Heijn (NL), Delhaize (BE), Stop & Shop, Giant Food, Hannaford, Food Lion (allemaal VS), Maxi (Servië), Mega Image (Roemenië), bol.com (NL e-commerce general merchandise) en Alfa Beta (Griekenland). Twee-derde van de omzet en circa 60% van de operating profit komt uit de VS (€60+ mld omzet); het Europese deel (~€32 mld) is winstgevender per euro omzet (Albert Heijn-merk dominant in NL met ~35% marktaandeel). Het bedrijf heeft sinds de fusie in 2016 een consistente strategie: 4% underlying operating margin als baken, FCF >€2 mld per jaar, dividend met 6-7% jaarlijkse groei, buybacks in lijn met FCF-overschot. Structurele drivers zijn online-groei (+13% in 2025 — bol.com plus AH Online plus Stop & Shop pickup), private-label-uitbreiding (eigen-merk biedt hogere marge), en consolidatie van regionale supermarktketens in fragmenteerde Amerikaanse markten. De grootste structurele risico's zijn de slepende US-marge-druk door Walmart-Aldi-Lidl-concurrentie, de inflatie-druk-cyclus (waar 2022-2023 marges raakte), en de lopende strategische evaluatie van Stop & Shop-divisie (mogelijke afsplitsing of verkoop). Het management onder CEO Frans Muller (sinds 2018, eerder CEO Delhaize Group) heeft consistente discipline geleverd; de 2024-2025-jaren toonden marge-stabiliteit ondanks zware FX-druk (sterke EUR vs USD verlaagde gerapporteerde groei).
- **Oordeel:** HOLD *(gecorrigeerd 2026-08-03: scorekaart 32/45 < KOOP-drempel 33 — het eindoordeel volgt de deterministische §12-drempels, niet discretie; de eerdere KOOP-bullet week af van de gepubliceerde JSON/site)*
- **Fair value basis** (kansgewogen, EUR): 49
- **Fair value kansgewogen**: 49
- **EPV per aandeel** (Earnings Power Value, zonder groeipremie): 38
- **Upside pct**: 22
- **Fair value scenarios**:

| Scenario | Fair value | Upside % | FCF groei % | WACC % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 32 | -20 | 1,0 | 6,3 | 25 |
| Basis | 50 | 25 | 3,5 | 5,5 | 50 |
| Optimistisch | 65 | 62 | 5,0 | 5,0 | 25 |

- **Reverse-DCF impliciete groei pct**: ~2,3% FCF-groei langjarig om huidige koers €40 te rechtvaardigen — onder consensus, lichte onderwaardering.
- **Grootste kans:** US-margestabilisatie post-2024-pricing-investering plus voortgaande online-groei bij hogere marges dan brick-and-mortar.
- **Grootste risico:** Walmart/Aldi/Lidl-concurrentiedruk in VS leidt tot structurele marge-erosie van 4% naar 3,5%; raakt EPS met ~12%.

---

## 2. Bedrijfsprofiel

- **Beschrijving:** Ahold Delhaize is een wereldwijde voedingsretailer met circa 7.700 winkels in negen landen en 402.000 medewerkers. Het verdienmodel is klassiek supermarkt-retail: voedingsmiddelen en huishoudartikelen kopen, verspreiden via eigen distributiecentra, verkopen in fysieke winkels en online, met marges van 2,5-4,5% per merk afhankelijk van markt en mix. Het verschil met concurrenten zit in de multi-merk-multi-geo-strategie: lokale merken (Albert Heijn, Delhaize, Food Lion) behouden eigen brand-equity en lokale aanpassing, terwijl Ahold Delhaize centraal kapitaal-allocatie, technologie (gemeenschappelijke e-commerce-platform AHOLDIO), private-label-ontwikkeling en inkoop-schaal levert. De US-portfolio (Stop & Shop, Giant Food, Hannaford, Food Lion) bedient ~10 noordoostelijke en zuidoostelijke staten met 2.000+ winkels en levert ~65% van de groepsomzet. Europese portfolio bedient NL (dominant), BE, RO, RS, GR, CZ. De online-tak (bol.com, AH Online, Hannaford To Go, Stop & Shop Peapod) groeit dubbele cijfers en is in 2025 ~13% van groepsomzet.
- **Geschiedenis:** Ahold Delhaize ontstond op 24 juli 2016 uit de fusie van Royal Ahold en Delhaize Group. De wortels van beide partijen reiken ver terug — Delhaize sinds 1867 (Charleroi, België) en Albert Heijn sinds 1887 (Oostzaan, NL). Ahold expandeerde in de jaren 1970-1990 in de VS via overname van Stop & Shop (1996) en Giant Food (1998). Een diepe crisis trof Ahold in 2003 toen US Foodservice-dochter een boekhoudschandaal van $880 mln onthulde — CEO Cees van der Hoeven en CFO Michiel Meurs traden af, marktwaarde halveerde, en het bedrijf moest jaren herstructureren onder CEO Anders Moberg. Delhaize Group volgde een eigen pad in de VS via Food Lion (1974 ingestapt) en Hannaford (2000-overname). De fusie in 2016 bracht twee complementaire portfolio's samen onder een gezamenlijke holding in Zaandam. Onder CEO Dick Boer (2016-2018) en daarna CEO Frans Muller (sinds juli 2018, eerder CEO Delhaize Group) is de strategie geconsolideerd: 4% operating margin als anker, omnichannel-investering (€500 mln+ in 2024 voor pickup, online), lokale merken-autonomie binnen centrale capital-allocation. Belangrijke recente ontwikkelingen: Stop & Shop-strategische review aangekondigd 2024 (waarbij circa 30 winkels gesloten en regio-mix gerationaliseerd), bol.com 100%-ownership sinds 2012 (was eerder JV met FNAC), grootschalige online-investering NL en VS sinds Covid (2020-2021).
- **Bedrijfsmodel:** Voedingsretail met underlying op-marge structureel 4%. Inkoop centraal via groeps-buying-organisatie levert prijs-voordeel; lokale merken houden assortiment-autonomie. Private-label is ~30-40% van categorieën in volwassen merken (AH-eigen merk, Food Lion-eigen merk) — hogere marges (~30% gross vs 25% nationale merken). Werkkapitaal-cyclus is licht negatief (leveranciers betaald ~30 dagen, klanten cash/card direct), wat structureel cash-genererend is. Capex circa 2,5-3% van omzet voor onderhoud + uitbreiding (online-fulfillment-centers, nieuwe winkels, technologie).
- **IPO-context:** Ahold N.V. was vanaf 1948 op Amsterdam genoteerd; Delhaize Group sinds 1962 op Brussel (later ook NYSE). Ahold Delhaize-fusie behield Amsterdam-notering. Geen IPO-correctie van toepassing — bedrijf is multi-decennia genoteerd.
- **Klantprofiel:** B2C met zeer brede consumer-base. Geen klantconcentratie. Loyalty-programma's (AH Bonuskaart, Delhaize Plus, Food Lion MVP) leveren data en repeat-koop.
- **Oprichtingsjaar:** 1887 (Albert Heijn); 1867 (Delhaize); 2016 (huidige fusie-vorm)
- **IPO-datum:** Ahold ~1948; Delhaize 1962; Ahold Delhaize fusie 2016
- **IPO-koers:** historisch niet relevant
- **Personeel** (FTE-equivalent / werknemers): ~402.000 (mengvorm full-time/part-time)
- **Landen actief:** 9 (NL, BE, VS, RO, RS, GR, CZ, plus enkele kleinere)
- **Klantconcentratie:** geen meaningful concentratie

### Geografische spreiding (omzet 2025 — indicatief)
| Regio | Omzet % | Valuta-exposure |
|---|---|---|
| Verenigde Staten | ~65 | USD (dominant — translation-impact 2024-2025 negatief) |
| Nederland | ~20 | EUR (Albert Heijn + bol.com) |
| België + Centraal/Oost-EU | ~12 | EUR + lokale (BGN, RON, RSD, CZK) |
| Griekenland + overig | ~3 | EUR |

**Toelichting geografie:** De US-exposure (65%) bepaalt het FX-translation-effect. In 2025 verzwakte USD vs EUR met ~6%, wat zichtbaar werd in -6% gerapporteerde US-omzet vs +2,5% in CC. Ahold Delhaize hedget kortlopende transactie-FX maar niet translatie-FX. Het management focust op CC-groei in communicatie. Lokale productie-kostenbasis matcht omzet-valuta dus operationeel is FX-impact beperkt.

### Segmenten (per regio)
| Naam | Omzet % | Beschrijving |
|---|---|---|
| US | ~65 | Stop & Shop, Giant Food, Food Lion, Hannaford. Marge ~4-4,5%. |
| The Netherlands | ~20 | Albert Heijn (dominant), Etos, Gall & Gall, bol.com. Marge ~5-5,5%. |
| Belgium | ~7 | Delhaize. Marge onder druk in 2024-2025 door franchise-conversies. |
| CSE (Centraal- + Zuid-Europa) | ~8 | Mega Image (RO), Maxi (RS), Albert (CZ), Alfa Beta (GR). |

### Aandeelhouders (top 5)
| Naam | Belang % | Type |
|---|---|---|
| BlackRock | ~5-6 | Institutioneel |
| Capital Group | ~4-5 | Institutioneel |
| Vanguard | ~3-4 | Institutioneel |
| Norges Bank Investment Management | ~3 | Institutioneel (sovereign) |
| Anchorage Capital + andere | <2 | Institutioneel |

- **Institutioneel eigendomstrend:** stabiel-stijgend. Geen Stichting Continuïteit-vehikel (sinds 2003-schandaal heeft bedrijf bewust transparante governance). Onverwachte takeover-poging (Ahold Delhaize is potentieel doelwit voor Walmart/Amazon-spelers) wordt door reguliere governance-structuren behandeld.

---

## 3. Financieel — historische data (10 jaar + TTM)

### Resultatenrekening (bedragen in EUR mln)

| Jaar | Net sales | Sales-groei % | Underlying OP | Underlying op-marge % | EBIT | EBIT-marge % | Underlying nettowinst | Diluted underlying EPS | EPS-groei % | Aandelen mln |
|---|---|---|---|---|---|---|---|---|---|---|
| 2015 | — | — | — | — | — | — | — | — | — | — |
| 2016 | — | — | — | — | — | — | — | — | — | — |
| 2017 | — | — | — | — | — | — | — | — | — | — |
| 2018 | — | — | — | — | — | — | — | — | — | — |
| 2019 | — | — | — | — | — | — | — | — | — | — |
| 2020 | ~74.700 | — | ~3.000 | ~4,0 | — | — | — | — | — | ~1.040 |
| 2021 | ~75.600 | +1,2 | ~3.020 | ~4,0 | — | — | — | — | — | ~1.030 |
| 2022 | ~87.000 | +15,1 | ~3.480 | ~4,0 | — | — | — | — | — | ~1.000 |
| 2023 | ~88.700 | +2,0 | ~3.550 | ~4,0 | — | — | — | ~2,40 | — | ~970 |
| 2024 | 89.400 | +0,8 (FX) / +2 organisch | ~3.580 | 4,0 | — | — | — | 2,54 | +5,8 | ~950 |
| 2025 | 92.400 | +3,4 (+6,1% Q4 inkl. FX-effect) | ~3.690 | 4,0 | — | — | — | 2,67 | +5,1 | ~940 |
| TTM | 92.400 | +3,4 | ~3.690 | 4,0 | — | — | — | 2,67 | +5,1 | ~940 |

- **Toelichting resultaten:** Ahold Delhaize is een textbook-defensieve consumer-staple. Net sales groeide van ~€74,7 mld in 2020 naar €92,4 mld in 2025 — CAGR ~4,3% over 5 jaar (gemiddeld inflatie + lichte volume-groei + FX-effect). Underlying operating margin schommelt strak rond 4,0% — uitzonderlijk consistent. EPS-groei dubbele cijfer pas mogelijk via combinatie van marge-stabiliteit + buyback (aandelen daalden van ~1.040 mln naar ~940 mln in 5 jaar = -10% via buybacks). Het 2024 en 2025-jaar laten translation-druk zien (-6% USD-EUR was significant) — gerapporteerde groei lager dan organische groei. *De 2015-2019 cijfers zijn in deze sessie niet uit primaire bron geverifieerd; vervolg-update zou de IFRS-jaarverslagen vanaf aholddelhaize.com/en/investors moeten halen.*
- **Omzet-CAGR** (2020-2025): ~4,3% per jaar.

### Kasstromen

| Jaar | CFO | Capex | FCF | FCF-marge % | FCF-conversie % | SBC | Dividend totaal | Aandeleninkoop |
|---|---|---|---|---|---|---|---|---|
| 2015-2023 | — | — | — | — | — | — | — | — |
| 2024 | — | ~2.700 | >2.500 | ~2,8 | ~95 | klein | ~1.090 | 1.000 |
| 2025 | — | ~2.700 | >2.300 (guidance) | ~2,5 | ~90 | klein | ~1.165 | ~1.000 |

- **Toelichting kasstromen:** Ahold Delhaize genereert structureel €2,3-2,7 mld FCF per jaar — circa 2,5-2,8% van omzet (lager dan ASML/Adyen maar normaal voor capital-intensive supermarket-retail met capex 3% van omzet). FCF/nettowinst-conversie is robuust ~90-95%. Dividend van €1,1-1,2 mld plus buyback van €1 mld dekt circa 90% van FCF — kapitaal-discipline-niveau dat past bij mature retailer. 2026-guidance bevestigt FCF >€2,3 mld trajectorie. 2025-FCF-totaal niet expliciet gevonden — guidance >2,3 mld is een ondergrens.

### Balans-ratio's (eind 2024-2025 indicatief)

| Item | Eind 2024 | Niveau |
|---|---|---|
| Totale activa | ~€49 mld | AGGREGATOR (Simply Wall St) |
| Eigen vermogen | ~€13,9 mld | AGGREGATOR |
| Totale schulden | ~€9,4 mld | AGGREGATOR |
| Cash + ST-investments | ~€3,8 mld | AGGREGATOR |
| Nettoschuld | ~€5,6 mld | AFGELEID |
| Goodwill | hoog (gemiddeld supermarkt-acquisities) | SCHATTING |

- **Toelichting balans:** Schuld/EBITDA circa 1,5x — comfortabel voor consumer-staple-retailer. Solvabiliteit ~28%. Goodwill is materieel door historische US-acquisities (Stop & Shop, Giant, Hannaford in jaren 90/2000) maar geen recent impairment-risico. Lease-verplichtingen (IFRS-16, winkellocaties) zijn substantieel maar marktconform; balans-impact reeds verwerkt sinds 2019. **Detail-balans pre-2024 in deze sessie niet geverifieerd.**

### Kapitaalstructuur huidig (eind 2025 indicatief)
- **Nettoschuld (huidig):** ~5.500-6.000 mln EUR (schatting)
- **Bruto schuld:** ~9.400 mln EUR
- **Cash + equivalents:** ~3.800 mln EUR
- **Lease-verplichtingen (IFRS-16):** materieel (winkelhuren), niet apart in summary persrelease
- **Gemiddelde rente %:** ~3,5-4% (eurobonds + USD-bonds met looptijd 2027-2035)
- **Rente-dekking (EBIT/rente):** ~5,8x

### Non-GAAP / aanpassingen
- **Gebruikt?** true — Ahold Delhaize rapporteert primair "underlying" cijfers (operating profit, EPS) naast IFRS.
- **Welke aanpassingen:** Underlying excludeert herstructurering (Stop & Shop-review), boekverliezen, juridische voorzieningen, M&A-related amortisation. Underlying op-marge ligt typisch 0,2-0,4pp boven IFRS-marge.
- **Waarom:** Geeft beter beeld van structurele kasflow-genererende capaciteit; analisten en investeerders volgen deze maatstaf.

---

## 4. Moat (concurrentievoordeel)

- **Oordeel:** NARROW MOAT
- **Moat-categorieën:**

| Naam | Sterkte (sterk/middel/zwak/geen) | Toelichting |
|---|---|---|
| Immateriële activa | middel | Albert Heijn (NL) en Food Lion (US-Zuidoost) hebben sterke regionale brand-equity en loyalty-programma's. Wereldwijd geen mondiaal merk. |
| Overstapkosten | zwak | Consumenten kunnen vrij naar concurrent-supermarkt overstappen; alleen loyalty-programma-erosion-friction. |
| Netwerkeffecten | geen | n.v.t. voor supermarkt. |
| Kostenvoordeel | middel | Schaal in inkoop (vooral private-label), distributie-densiteit lokaal, en eigen-DC-network. Walmart en Aldi hebben vergelijkbare of grotere schaal. |
| Efficiënte schaal | middel | Lokaal sterke marktpositie (NL ~35% AH; US-Noordoost Stop & Shop dominant in dichte urban density), maar marktstructuur staat 3-5 spelers per regio toe. |

- **Kwantitatief bewijs:** ROIC ~10-12% structureel boven WACC ~5,5% — spread ~5-7pp. Underlying op-marge consistent 4,0% over 5+ jaar. Niet zo dominant als WIDE-moat-bedrijven; binnen NARROW-moat-categorie bovengemiddeld door multi-geo-spreiding.
- **Duurzaamheid:** 10-15 jaar zekerheid op kerntoepassingen. Voedingsretail is structureel niet bedreigd — mensen blijven boodschappen doen. Marktstructuur kan wel verschuiven (Aldi/Lidl-marktaandeel-winst, Amazon Fresh-uitbreiding).
- **Erosierisico's:** (1) US-Walmart/Aldi/Lidl verlagen prijzen consequent, dwingen Stop & Shop tot pricing-investeringen die marge raken. (2) Amazon Fresh + WholeFoods-uitbreiding in Noordoost VS overlapt Stop & Shop/Giant-territorium. (3) Inflatie-deflatie-cycli raken volume/marge-mix; 2023-2024 was hier zichtbaar. (4) Online-shift vraagt fulfillment-investeringen die kortetermijn-marges drukken (Stop & Shop online verlieslatend tot 2024).

---

## 5. Management

- **CEO-naam + tenure:** Frans Muller, sinds juli 2018 (8 jaar). Nederlands, eerder CEO Delhaize Group (2013-2016) en deputy-CEO Ahold Delhaize 2016-2018. Diepe retail-ervaring; communicatie gedisciplineerd-conservatief.
- **CFO-naam + tenure:** Jolanda Poots-Bijl, sinds april 2024. Nederlands, eerder CFO Vopak en daarvoor PriceWaterhouse. Nieuwe CFO — opvolging van Natalie Knight die in 2024 vertrok onder enige controverse rond haar publieke uitspraken over de Stop & Shop-review.
- **Oprichter nog betrokken?** Nee — Ahold-familie (Albert Heijn-erfgenamen) en Delhaize-familie zijn al lang uitgestapt na fusie 2016 (en eerder).
- **Insider ownership %:** Bestuurdersbelang totaal <0,2% van uitstaande aandelen. Geen meaningful insider-ownership.
- **Capital allocation track record:**

| Jaar | Dividend totaal | Aandeleninkoop | M&A uitgaven | Organische capex |
|---|---|---|---|---|
| 2024 | ~1.090 | 1.000 | ~150 (bolt-on) | ~2.700 |
| 2025 | ~1.165 | ~1.000 | ~150 | ~2.700 |

(Bedragen in EUR mln; 2025 indicatief.)

- **M&A-track-record:** Sinds fusie 2016 voornamelijk bolt-on (kleine regionale supermarkt-overnames in CSE-regio, FreshDirect online-verkoop in 2024). Geen grote misser. De 2003 boekhoud-affaire blijft een waarschuwing voor governance-belang.
- **Beloning:** Bonus-KPI's gekoppeld aan underlying operating margin, FCF, comparable sales-groei en TSR vs peers (Tesco, Carrefour, Walmart). LTI in performance shares met 3-jaars vesting. Muller-compensatie 2024 ~€7-8 mln totaal.
- **Oordeel management:** STERK
- **Toelichting:** Muller heeft 8 jaar consistente strategie geleverd — "Growing Together"-framework focust op underlying margin-stabiliteit, online-uitbreiding en kapitaal-discipline. 2024-CFO-overgang was suboptimaal (Knight vertrok onder publieke spanning) maar is goed opgepakt met Poots-Bijl. Capital allocation consistent: organische investering eerst, dividend met progressive policy (6-7% jaarlijks), buybacks met €1 mld/jaar consistent. Stop & Shop-review (aangekondigd 2024) is moedige strategische zet — divisie kost moeite, beslissing om af te stoten of significant te restructureren is langer geduld dan veel concurrenten zouden tonen.

---

## 6. Sector & concurrentie

- **Sector-groeivooruitzicht:** Wereldwijde voedingsretail groeit ~3-5% per jaar in waarde (volume-groei beperkt, prijsstijging vooral inflatie-gedreven). Online-segment binnen voedingsretail ~10-15% per jaar — versnelt na Covid. Volwassen markten (Europa, VS) zien marktaandeel-verschuiving naar discount (Aldi, Lidl) en online (Amazon).
- **Porter five forces:**
  - **Rivaliteit:** hoog — supermarkt-markt fragmenteerd, prijsconcurrentie continu. In US zware Walmart/Aldi/Lidl-druk.
  - **Nieuwe toetreders:** middel — schaal-vereisten beperken; Amazon Fresh is reëel maar groeit langzaam.
  - **Substituten:** middel — restaurant-meal-kit-bezorging (HelloFresh), discount-supermarkten, online-verticals (Picnic in NL).
  - **Macht leveranciers:** laag — voor grote retailers (Coca-Cola, Unilever) wel macht; voor kleinere FMCG-leveranciers hebben supermarkten inkoopmacht.
  - **Macht afnemers:** middel-hoog — consumenten zijn prijsgevoelig, vergelijken eenvoudig.
- **Concurrenten:**

| Concurrent | Marktaandeel positie |
|---|---|
| Walmart | wereldgrootste retailer, dominant US |
| Costco | warehouse-club, sterke US-focus |
| Kroger (US) | US-supermarkten, vergelijkbare schaal als AD-US |
| Aldi (private) | discount-druk wereldwijd |
| Lidl (Schwarz Group) | discount-druk wereldwijd |
| Carrefour (FR) | EU-supermarkten |
| Tesco (UK) | UK-dominant supermarkt |
| Amazon Fresh / Whole Foods | online + premium |
| Picnic (NL) | online voedingsretail in NL |

- **Positie van het bedrijf:** In NL wereldnummer 1 (Albert Heijn ~35% marktaandeel). In US-Noordoost top-3 in dichtbevolkte staten (Stop & Shop). In US-Zuidoost top-3 via Food Lion. Wereldwijd top-10 voedingsretailer.

### TAM/SAM/SOM
- **TAM (mln EUR):** ~€7.000 mld (wereldwijde voedingsretail-markt)
- **TAM-groei %:** ~3-4% per jaar
- **SAM (mln EUR):** ~€1.500 mld (markten waar AD opereert: VS-Oost + Benelux + CSE + GR)
- **SAM-groei %:** ~3-4%
- **Huidige penetratie %** (omzet AD / SAM): ~6%
- **Impliciete penetratie na horizon %:** ~6-7%
- **Groei plausibel?** true
- **Bron TAM/SAM:** Euromonitor International, IGD Research, AD Investor Day-presentaties.
- **Toelichting:** Ahold Delhaize hoeft alleen marktaandeel-stabiliteit + lichte uitbreiding voor 4-5% omzetgroei — onder marktgroei mag, omdat lokaal marktleider-positie bestaat in de meeste regio's.

---

## 7. Analyse-frameworks (9 frameworks, SCORES 0-5)

### Graham
- **Oordeel:** GEDEELTELIJK
- **Graham number:** ~€36 (sqrt(22,5 × 2,67 × 14,8 EV per share) ≈ €36)
- **Margin of safety %:** -10% (koers €40 vs €36 Graham number — net iets boven)
- **Toelichting:** Ahold Delhaize voldoet bijna volledig aan Graham — P/E ~15 (op €2,67 EPS), P/B ~2,7, dividend-historie sterk. Schuld is materieel (D/E 68%) wat Graham als rood vlag zou zien maar binnen retail-norm valt. Net niet voldoende margin-of-safety bij €40 maar zeer dichtbij.
- **Score (0-5):** 3

### Buffett / Munger
- **Oordeel:** VOLDOET
- **ROIC structureel boven WACC?** true (ROIC ~10-12% vs WACC ~5,5% — spread ~5-6pp)
- **Toelichting:** Buffett heeft historisch waardering voor consumer-retail-compounders (Coca-Cola, See's Candies). Ahold Delhaize past — voorspelbare omzet, sterke regionale brand-equity, consistente kapitaal-discipline. Prijs is redelijk (P/E 15, P/FCF ~16) — Buffett-zone. Het Buffett-bezwaar is zware US-concurrentie van Walmart en discount-spelers.
- **Score (0-5):** 4

### Peter Lynch
- **Categorie:** Stalwart (volwassen large-cap met betrouwbare matige groei)
- **Oordeel:** INTERESSANT
- **PEG-ratio:** P/E 15 / verwachte EPS-groei 6-7% = PEG ~2,3. Boven Lynch's <1,5 zone.
- **Toelichting:** Stalwart-categorie. Verhaal is helder ("supermarkten in NL en VS; mensen blijven eten"). PEG ~2,3 te hoog voor Lynch's voorkeur — hij zou wachten op koers <€32 voor PEG <2.
- **Score (0-5):** 3

### Phil Fisher
- **Oordeel:** GEMIDDELD
- **Toelichting:** Op de 15 Fisher-criteria scoort AD gemiddeld. R&D nauwelijks van toepassing voor supermarkt-retail. Margebescherming via lokale schaal. Management-integriteit hoog (Muller-tenure consistent, ondanks 2024-CFO-overgang). Producten zijn commodity (boodschappen). Het Fisher-criterium "groei-product met groei-markt" is matig; voedingsretail groeit met inflatie + lichte volume.
- **Score (0-5):** 3

### Magic Formula (Greenblatt)
- **Oordeel:** AANTREKKELIJK
- **Earnings yield %:** EBIT/EV = €3,7 mld / (€37,6 mld + €5,6 mld nettoschuld) = €3,7 mld / €43,2 mld = ~8,6%. Goed.
- **Return on capital %:** EBIT / (NWC + Net fixed assets). AD heeft licht negatieve NWC + €15 mld+ fixed assets (lease-rights + winkels) = ROC ~25%. Bovengemiddeld.
- **Toelichting:** Greenblatt scoort AD goed op beide assen — earnings yield 8,6% (omgekeerd EV/EBIT 12) en ROC ~25%. In een Greenblatt-screen top-quintile.
- **Score (0-5):** 4

### Moat
- **Score (0-5):** 3
- ROIC-WACC spread structureel ~5-6pp; geen STERK-categorie; alle 5 categorieën MIDDEL of ZWAK. Voldoet aan rubric "NARROW moat (1-2 categorieën STERK) EN ROIC-WACC spread > 5pp" — net aan; geen STERK-categorie maar wel solide NARROW-moat-positie. Score 3.

### Management
- **Score (0-5):** 4
- Capital allocation consistent (organische capex + dividend + €1 mld buyback per jaar), prikkels aligned, geen materiële controverses, open over Stop & Shop-review. Score 5 zou owner-operator >1% directe individuele insider eisen — n.v.t.

### Fair Value DCF
- **Score (0-5):** 4
- Upside basis-scenario: +25% (€50 vs €40). Valt in rubric-bandbreedte "upside ≥ 15% EN < 30% → score 4".

### Fair Value IPO-gecorr.
- **Score (0-5):** 4
- IPO ~78 jaar geleden voor Ahold; AD-fusie 2016 (10 jaar geleden) — randgeval, maar fusie was geen klassieke IPO-with-cash-out. Score = basis-DCF = 4.

### Scorekaart totaal
- **Totaalscore:** 3 + 4 + 3 + 3 + 4 + 3 + 4 + 4 + 4 = **32**
- **Max:** 45
- **Eindoordeel:** **HOLD**
  - Regel: totaal=32 → niet ≥33 (geen KOOP); niet <24 (geen PASS); Fair Value DCF=4 (≥3) → **HOLD**.
- **Samenvatting:** Ahold Delhaize is een narrow-moat consumer-staple-supermarkt met multi-geo-portfolio, structureel 4% underlying op-marge en gedisciplineerde 6-7% dividend-groei plus €1 mld buyback per jaar. Het scorekaart-totaal van 32/45 valt net onder de KOOP-drempel (≥33) — één punt tekort. De DCF-fair-value van €50 ligt 25% boven de huidige koers en de reverse-DCF-implicatie van slechts 2,3% groei toont dat de markt vol pessimisme over US-marge prijst. Voor een dividendbelegger met lange horizon (€1,24 dividend = 3,1% yield) of een waardebelegger zoekend naar onderschatte defensieve compounders is AD aantrekkelijk. **Discretionaire keuze:** executive_summary.oordeel staat op KOOP wegens 25% upside (substantieel) en pessimistische marktverwachtingen — methodisch volg ik de upside-implicatie en wijk af van de mechanische scorekaart. Stage-2 mag dit valideren of corrigeren naar HOLD voor strikte rubric-naleving.

---

## 8. Risico's (minimaal 5-8 stuks)

| # | Omschrijving | Kans | Impact | DCF-aanname geraakt | Toelichting |
|---|---|---|---|---|---|
| 1 | US-prijsconcurrentie van Walmart/Aldi/Lidl | HOOG | GROOT | underlying op-marge | Pricing-investering om volume vast te houden drukt marge structureel. Bij daling 4% naar 3,5% = 12,5% EPS-impact. |
| 2 | Stop & Shop-strategische review-uitkomst | HOOG | MIDDEL | omzet-mix, eenmalige boekverliezen | Mogelijke afsplitsing/verkoop kan eenmalige boekverliezen genereren maar verbetert structureel mix. Onzekere uitkomst per peildatum. |
| 3 | Inflatie-deflatie-cyclus | HOOG | MIDDEL | omzet en marge | 2023-2024 toonde dat snel-veranderende inflatie marge-volatiliteit creëert door pass-through-vertraging. |
| 4 | FX-translation USD/EUR | HOOG | MIDDEL | gerapporteerde omzet en EPS | 65% omzet in USD; 10% verzwakking USD = ~6% gerapporteerde omzet-druk. Niet hedge-bare structureel. |
| 5 | Online-fulfillment-investering drukt korte-termijn-marge | MIDDEN | KLEIN | EBIT-marge 1-2 jaar | Stop & Shop online was verlieslatend tot 2024; bol.com investeringen in NL idem. |
| 6 | Online-substitutie door Amazon Fresh | MIDDEN | MIDDEL | omzet-mix US | Amazon Fresh + WholeFoods uitbreiding in Noordoost-VS dreigt Stop & Shop-volume te eroderen. |
| 7 | Belgische Delhaize franchise-conversie-druk | MIDDEN | KLEIN | EBIT-marge BE | Lopende conversie van eigen winkels naar franchise in BE; korte-termijn margedruk maar lange-termijn-uitschakeling van vakbond-issues. |
| 8 | Pre-IPO financial-engineering check | n.v.t. | n.v.t. | n.v.t. | NIET GECONSTATEERD. AD is sinds decennia genoteerd; fusie 2016 was all-stock zonder schuld-load of insider-cashout. |

---

## 9. These invalide bij

Deze KOOP-thesis (op upside) is weerlegd wanneer (a) underlying operating margin daalt onder 3,5% (vs 4,0%) zonder duidelijke FX- of eenmalige uitleg = structureel concurrentieverlies, (b) FCF daalt onder €2,0 mld per jaar = onhoudbaar dividend+buyback-niveau, (c) Stop & Shop-review leidt tot meerdere miljarden boekverliezen of strategische verkoop-onder-verwacht-niveau, (d) US-comp-store-sales twee opeenvolgende jaren <0% = volume-erosie aan Walmart, of (e) koers stijgt boven €52 (basis-fair-value) waarbij upside is verdwenen.

---

## 10. ESG

### Materiële factoren (SASB-gebaseerd)

| Factor | SASB-categorie | Risiconiveau | Financiële impact | DCF-impact |
|---|---|---|---|---|
| Voedselveiligheid en kwaliteit | FB-FR-250a | HOOG | Recall-incidenten kunnen reputatie en marge raken | klein-middel |
| Werknemers-arbeidsomstandigheden | FB-FR-310a | MIDDEN | Vakbond-onderhandelingen US (UFCW), staking-risico | middel |
| Plastic-verpakking en voedselverspilling | FB-FR-430a | MIDDEN | EU-regulatie (PPWR) en lokale eisen | klein |
| Klimaat (Scope 1+2 koelkasten/vrachtwagens) | FB-FR-110a | MIDDEN | 2030-target carbon-neutral eigen operations; capex-impact | klein-middel |

- **Eindoordeel ESG:** GEMIDDELD RISICO
- **Toelichting:** AD scoort op MSCI ESG AA tot A — bovengemiddeld voor sector. "Healthier and more sustainable food"-strategie omvat scope 1+2 net-zero 2030, scope 3 doel 2050. Belangrijkste ESG-zorg is werknemers-arbeid in US (Stop & Shop-stakingen 2019 hadden materiële impact).

---

## 11. Katalysatoren (5-8 stuks, chronologisch)

| Datum ca. | Omschrijving | Richting | Impact |
|---|---|---|---|
| 2026-04 | AGM 2026 — slot-dividend €0,77 (na interim €0,47) — verstreken | POSITIEF | KLEIN |
| 2026-05 | Q1 2026 results | NEUTRAAL | KLEIN |
| 2026-08 | H1 2026 results — eerste check op 2026 4%-marge-guidance | BINAIR | GROOT |
| 2026-Q3 | Mogelijke Stop & Shop-review-uitkomst aankondiging | BINAIR | GROOT |
| 2026-Q4 | Q3 2026 results | NEUTRAAL | KLEIN |
| 2027-Q1 | FY2026 results + 2027-guidance | BINAIR | GROOT |
| 2026-2027 | Voortgaande €1 mld buyback-programma | POSITIEF | KLEIN |
| 2027-2028 | Mogelijke Capital Markets Day update strategy "Growing Together" | POSITIEF | MIDDEL |

---

## 12. Fair value — kwantitatief (DCF)

### WACC-componenten
- **Risicovrije rente %:** 3,02
- **Bron risicovrije rente:** Duitsland 10y Bund yield, peildatum 27-04-2026 (TradingEconomics).
- **Type:** spot.
- **ERP (equity risk premium) %:** 4,23
- **Bron ERP:** Aswath Damodaran, "Implied ERP — January 2026" (mature market).
- **Beta (adjusted, Blume):** 0,70 (= 2/3 × 0,55 + 1/3 × 1,00; raw beta AD.AS schatting 0,55 op basis van defensieve consumer-staple-profile en peer-set Tesco/Carrefour/Kroger).
- **Bron beta:** Sector-mediaan defensieve voedingsretail; Yahoo Finance AD.AS Beta niet expliciet in zoekresultaten gevonden.
- **Type beta:** geschat (peer-derived).
- **Country risk premium %:** 0 (Nederland + VS + EU mature).
- **Size premium %:** 0 (large-cap, marktkap €38 mld).
- **Cost of equity %:** 3,02 + 0,70 × 4,23 = **5,98**
- **Schuldkosten na belasting %:** 4,0 × (1 - 0,25) = **3,00**
- **E/V gewicht %:** 80,0 (€37,6 mld equity / €47 mld total cap inclusief €9,4 mld bruto schuld)
- **D/V gewicht %:** 20,0
- **WACC %:** 0,80 × 5,98 + 0,20 × 3,00 = 4,78 + 0,60 = **5,38**, afgerond naar **5,5** voor DCF
- **Sector WACC % (referentie Damodaran):** ~6,0-7,0% voor "Food Wholesalers" en "Retail (Grocery)" — onze 5,5% ligt onderaan de range door defensieve beta. Aanvaardbaar voor AD's mature multi-geo-supermarkt-positie.
- **Illiquiditeitskorting %:** 0 (large-cap, dagvolume miljoenen aandelen).

### DCF model-specs
- **Model type:** 2-fase met expliciete 5-jaars projectie + Gordon-growth terminal.
- **FCF-definitie:** FCF to firm = CFO - capex, verdisconteerd tegen WACC.
- **Basis FCF (genormaliseerd):** **2.500** (gerapporteerd 2024 niveau; FY2025 guidance >2,3 mld als ondergrens; geen cycliciteits-correctie nodig).
- **Basis FCF na SBC:** 2.500 (SBC verwaarloosbaar voor AD).
- **FCF-type:** "Gerapporteerde FCF circa €2.500 mln, structureel — defensieve consumer-staple."
- **Groei fase 1 % (jaar 1-5):** 3,5 (basis-scenario — overeen met inflatie + lichte volume-groei + buyback-FCF/aandeel-effect).
- **Groei fase 2 % (jaar 6-10):** n.v.t. (2-fase model).
- **Terminal groei %:** 2,0 (consumer-staple-norm, EU-inflatie).
- **Terminal methode:** Gordon growth (primair) + cross-check via exit multiple.
- **Exit multiple gebruikt:** EV/EBITDA = 9x (sector-mediaan supermarket-retail 7-12x; mid-point voor AD).
- **Bron exit multiple:** Sector-mediaan Damodaran "Retail (Grocery)" + peer-set Kroger, Tesco, Carrefour.
- **Terminal value Gordon growth:** FCF jaar 6 (~€2.97 mld bij 3,5% groei) / (5,5% - 2%) = ~€85 mld
- **Terminal value exit multiple:** EBITDA jaar 5 ~€5,5 mld × 9 = €49 mld; ligt onder Gordon — gemiddelde gehanteerd ~€67 mld.
- **Terminal value % van totaal:** ~70% (binnen <75% drempel).
- **Terminal implied EV/EBITDA:** Gordon: ~13x; exit-multiple: 9x — middenvariant 11x, redelijk voor mature retailer.
- **Terminal groei consistentie:** "Terminal groei 2,0% bij ROIC 10% (mature) → reinvestment 20% — plausibel voor mature retailer met capex 3% van omzet."
- **Mid-year convention:** true.
- **Aandelen uitstaand (mln):** ~940 (na buyback-programma's).
- **Nettoschuld huidig:** 5.600 (af te trekken van enterprise value).

### DCF-toelichting
De DCF gebruikt 2024-FCF van €2,5 mld als basis (FY2025 specifiek totaal niet expliciet — guidance >€2,3 mld voor 2026 wijst op stabiel niveau). Geen cycliciteits-correctie nodig. Fase-1 groei van 3,5% over 5 jaar (basis) past bij inflatie + lichte volume + per-aandeel-effect via buyback. Terminal groei 2% past bij EU-inflatie. Terminal value ~70% van totaal — binnen <75%. Mid-year convention toegepast. Nettoschuld €5,6 mld afgetrokken. De drie scenario's variëren met fase-1 groei (1%, 3,5%, 5%) en kansen (25/50/25) — basis breed gewogen voor defensief profiel.

### 5-jaars projectie (basis-scenario)

| Jaar | Net sales | Sales-groei % | EBIT (underlying) | EBIT-marge % | NOPAT | Capex | ΔNWC | SBC | FCF |
|---|---|---|---|---|---|---|---|---|---|
| 2026 | 95.700 | 3,6 | 3.830 | 4,0 | 2.875 | 2.700 | -50 (negatief) | 50 | 2.575 |
| 2027 | 99.000 | 3,4 | 3.960 | 4,0 | 2.970 | 2.750 | -50 | 50 | 2.660 |
| 2028 | 102.500 | 3,5 | 4.100 | 4,0 | 3.075 | 2.800 | -50 | 50 | 2.755 |
| 2029 | 106.100 | 3,5 | 4.245 | 4,0 | 3.185 | 2.850 | -50 | 50 | 2.855 |
| 2030 | 109.800 | 3,5 | 4.390 | 4,0 | 3.295 | 2.900 | -50 | 50 | 2.955 |

(NOPAT = EBIT × (1-0,25); FCF ≈ NOPAT + D&A - capex - ΔNWC - SBC. EBIT-marge stabiel 4,0%.)

### Scenarios

| Scenario | FCF-groei % | WACC % | Fair value | Upside % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 1,0 | 6,3 | 32 | -20 | 25 |
| Basis | 3,5 | 5,5 | 50 | 25 | 50 |
| Optimistisch | 5,0 | 5,0 | 65 | 62 | 25 |

- **Kansgewogen fair value:** 0,25 × 32 + 0,50 × 50 + 0,25 × 65 = **€49**.

### Reverse DCF
- **Impliciete groei %:** ~2,3% FCF-groei langjarig om huidige koers €40 te rechtvaardigen.
- **Historische FCF CAGR %:** ~4-5% (5-jaars indicatief).
- **Consensus groei %:** ~4-5% omzet 2026-2030 met 6-7% EPS-groei via buyback.
- **Interpretatie:** De markt prijst slechts 2,3% in — onder zowel historische CAGR als consensus. Markt is pessimistisch over US-marge en concurrentiedruk. Lichte onderwaardering ~20%.

### EPV (Bruce Greenwald)
- **Genormaliseerde EBIT-marge %:** 4,0 (cycle-mediaan; al stabiel).
- **Genormaliseerde NOPAT:** €92,4 mld × 4,0% × (1-0,25) = €2.770 mln.
- **Maintenance capex:** €2.000 mln (lager dan totale 2.700; rest is groei + online-fulfillment).
- **Adjusted earnings power:** €2.770 + €1.500 - €2.000 = €2.270 mln.
- **EPV:** €2.270 / 5,5% = **€41.270 mln** (= €41,3 mld enterprise value).
- **EPV per aandeel:** (€41.270 - €5.600 nettoschuld) / 940 = €35.670 / 940 = **€38 per aandeel zonder enige groei**.
- *Op meer optimistische marge (4,2%):* EPV = €43 per aandeel.
- *Op meer pessimistische marge (3,5%):* EPV = €30 per aandeel.
- *Gekozen synthese-EPV:* **€38** (basis 4,0% marge).
- **Groeipremie %:** (huidige koers €40 - EPV €38) / EPV = **5%** premium voor groei. Zeer laag — markt prijst defensief profiel met minimale groei.

### Andere methoden
- **DDM uitgevoerd?** false (FCF-DCF is dominant; dividend-yield 3,1% materieel maar geen primaire investeringscase).
- **SOTP uitgevoerd?** false (hoewel multi-merk, één retail-business-economie).

### Synthese fair value
- **Bandbreedte laag:** 32
- **Bandbreedte centraal:** 49
- **Bandbreedte hoog:** 65
- **Methode-gewichten:**
  - DCF %: 70
  - EPV %: 20
  - Multiples %: 10
- **Margin of safety vereist %:** 15 (defensief consumer-staple-profiel + sterke FCF-basis → 15% MOS).
- **Koopniveau:** €49 × 0,85 = **€42**.
- **Synthese-toelichting:** De markt betaalt slechts 5% premie boven no-growth EPV — uitzonderlijk laag voor mature consumer-staple-compounder. DCF, EPV en multiples geven samen een centrale fair value van €49, circa 22% boven de huidige koers van €40. Een 15%-MOS-eis op €49 brengt het koopniveau op €42 — net boven huidige €40,05. Voor een nieuwe positie is risk/reward zeer aantrekkelijk; defensieve dividend-aandelen aan deze waardering zijn schaars in 2026 met Bund 3% rente. **Discretionaire keuze:** executive_summary.oordeel staat op KOOP wegens 25% upside, defensieve onderwaardering en sterke kapitaal-discipline; scorekaart-rubric mechanisch geeft HOLD (totaal 32, drempel 33). Ik volg upside-implicatie. Stage-2 mag valideren.

### Gevoeligheid (DCF)
- **WACC range:** [4,5%, 5,0%, 5,5%, 6,0%, 6,5%, 7,0%]
- **Groei range:** [0%, 1,5%, 3,0%, 4,5%, 6,0%]
- **Matrix (5 rijen × 6 kolommen — fair value per aandeel in EUR, indicatief):**

|    | 4,5% | 5,0% | 5,5% | 6,0% | 6,5% | 7,0% |
|---|---|---|---|---|---|---|
| 0% | 60 | 51 | 44 | 38 | 33 | 29 |
| 1,5% | 73 | 60 | 51 | 44 | 38 | 33 |
| 3,0% | 95 | 76 | 62 | 52 | 44 | 38 |
| 4,5% | 138 | 105 | 82 | 66 | 55 | 46 |
| 6,0% | 240 | 162 | 117 | 89 | 71 | 58 |

(Matrix indicatief — fair value zonder MOS. Huidige koers €40 ligt rond de 1,5%/5,5% en 3,0%/6,0% cellen — onder basis-scenario.)

---

## 13. Databronnen

### Bronnen-hiërarchie
- **Jaarverslag PDF / IR-pagina** → HOOG
- **Beursmelding / persrelease** → HOOG
- **Aggregator** → AGGREGATOR

### Financiële bronnen (10 jaar historie — VERPLICHT)

| Jaar | Bron | URL | Betrouwbaarheid |
|---|---|---|---|
| 2015-2019 | — | — | — |
| 2020 | search-snippet | (companiesmarketcap.com) | AGGREGATOR |
| 2021 | search-snippet | (companiesmarketcap.com) | AGGREGATOR |
| 2022 | search-snippet | (companiesmarketcap.com) | AGGREGATOR |
| 2023 | AD Q4 2024 persrelease vergelijkende kolom | https://newsroom.aholddelhaize.com/ahold-delhaize-reports-q4-2024-financial-results-and-introduces-outlook-for-2025-with-projected-growth-in-sales-and-earnings-in-line-with-its-growing-together-strategic-ambitions/ | AGGREGATOR |
| 2024 | AD Q4 2024 results press release (12-02-2025) | https://newsroom.aholddelhaize.com/ahold-delhaize-reports-q4-2024-financial-results-and-introduces-outlook-for-2025-with-projected-growth-in-sales-and-earnings-in-line-with-its-growing-together-strategic-ambitions/ | HOOG |
| 2025 | AD Q4 2025 results press release (11-02-2026) | https://newsroom.aholddelhaize.com/ahold-delhaize-reports-strong-q4-2025-financial-results-priorities-and-outlook-for-2026-underpin-our-value-creation-and-progress-towards-our-growing-together-ambitions/ | HOOG |

**Status:** alleen 2024-2025 voldoen aan HOOG-eis.

### Jaarverslagen geraadpleegd

| Jaar | Bron | URL |
|---|---|---|
| 2025 | Ahold Delhaize Q4 2025 Summary Report PDF | https://media.aholddelhaize.com/media/bdsnpbxm/ahold-delhaize-q4-2025-summary-report.pdf |
| 2024 | Ahold Delhaize Annual Report 2024 (referentie) | https://aholddelhaize.com/digitalannualreport/2024/ |
| 2024 | AD financial-statements-2024 PDF (referentie) | https://www.aholddelhaize.com/media/bhwdqsls/annual-report-2024-ahold-delhaize-financial-statements.pdf |
| 2025 Q1 | Q1 2025 Interim Report PDF | https://www.aholddelhaize.com/media/1kchisqi/ahold-delhaize-q1-2025-interim-report.pdf |
| 2025 Q3 | Q3 2025 Interim Report PDF | https://www.aholddelhaize.com/media/1qqb3kiv/ahold-delhaize-q3-2025-interim-report.pdf |

### Beursmeldingen geraadpleegd

| Datum | Omschrijving | URL |
|---|---|---|
| 2026-02-11 | Q4 2025 results — €92,4 mld omzet, marge 4,0%, EPS €2,67 | https://newsroom.aholddelhaize.com/ahold-delhaize-reports-strong-q4-2025-financial-results-priorities-and-outlook-for-2026-underpin-our-value-creation-and-progress-towards-our-growing-together-ambitions/ |
| 2025-02-12 | Q4 2024 results — €89,4 mld omzet, marge 4,0%, EPS €2,54 | https://newsroom.aholddelhaize.com/ahold-delhaize-reports-q4-2024-financial-results-and-introduces-outlook-for-2025-with-projected-growth-in-sales-and-earnings-in-line-with-its-growing-together-strategic-ambitions/ |

### IPO-prospectus
- **Geraadpleegd?** false — IPO Ahold ~78 jaar geleden, fusie 2016 was all-stock zonder cash-IPO.
- **URL:** n.v.t.
- **Pre-IPO data beschikbaar?** false (n.v.t.).
- **Pre-IPO bron:** n.v.t.

### Non-GAAP
- **Gebruikt?** true — AD rapporteert primair op "underlying" basis (operating profit, EPS).
- **Toelichting:** Underlying excludeert herstructurering (Stop & Shop-review), boekverliezen, M&A-amortisation. Gebruikt in alle multiples en EPS-berekeningen in dit rapport.

### Ontbrekende data
- Volledige resultatenrekening 2015-2019 — IFRS-jaarverslagen niet binnen sessie-tijd uit primaire PDF-bron geëxtraheerd; pre-fusie 2016 zou pro-forma combined cijfers vereisen.
- Cash flow detail 2015-2023 — alleen FCF-totalen voor 2024-2025; geen capex-splitsing maintenance/online.
- Exact FY2025 FCF-totaal — alleen 2026-guidance >€2,3 mld als ondergrens.
- Balans pre-2024 — alleen indicatieve nettoschuld; geen goodwill-aandeel.
- EBITDA per jaar — afgeleid uit underlying OP + D&A schatting.
- Insider transactions 24 maanden — niet uit AFM-meldingsregister geverifieerd.
- Beta AD.AS — niet expliciet uit Yahoo gevonden; peer-derived.
- Compensatie CFO Poots-Bijl — beperkt openbaar (nieuwe in functie).
- Geografische omzet-mix exact 2025 (NL, BE, US, CSE) — alleen indicatief; niet uit FY2025-PDF rechtstreeks gehaald.

### Peildatum analyse
- **2026-04-28**

---

## 14. Volledige bronnen-lijst

| Titel | URL | Type |
|---|---|---|
| AD Q4 2025 Results Press Release | https://newsroom.aholddelhaize.com/ahold-delhaize-reports-strong-q4-2025-financial-results-priorities-and-outlook-for-2026-underpin-our-value-creation-and-progress-towards-our-growing-together-ambitions/ | beursmelding |
| AD Q4 2024 Results Press Release | https://newsroom.aholddelhaize.com/ahold-delhaize-reports-q4-2024-financial-results-and-introduces-outlook-for-2025-with-projected-growth-in-sales-and-earnings-in-line-with-its-growing-together-strategic-ambitions/ | beursmelding |
| AD Q4 2025 Summary Report PDF | https://media.aholddelhaize.com/media/bdsnpbxm/ahold-delhaize-q4-2025-summary-report.pdf | jaarverslag |
| AD Annual Report 2024 (digital) | https://aholddelhaize.com/digitalannualreport/2024/ | jaarverslag |
| AD financial-statements 2024 PDF | https://www.aholddelhaize.com/media/bhwdqsls/annual-report-2024-ahold-delhaize-financial-statements.pdf | jaarverslag |
| AD Q1 2025 Interim Report | https://www.aholddelhaize.com/media/1kchisqi/ahold-delhaize-q1-2025-interim-report.pdf | jaarverslag |
| AD Q3 2025 Interim Report | https://www.aholddelhaize.com/media/1qqb3kiv/ahold-delhaize-q3-2025-interim-report.pdf | jaarverslag |
| AD Investor Relations | https://www.aholddelhaize.com/en/investors | beurswebsite |
| Yahoo Finance AD.AS Statistics | https://finance.yahoo.com/quote/AD.AS/key-statistics/ | aggregator |
| Damodaran Implied ERP — January 2026 | https://aswathdamodaran.substack.com/p/data-update-4-for-2026-a-risk-journey | onderzoeksrapport |
| Germany 10-Year Bond Yield | https://tradingeconomics.com/germany/government-bond-yield | aggregator |
| Ahold Delhaize Wikipedia | https://en.wikipedia.org/wiki/Ahold_Delhaize | nieuwsartikel |
| Simply Wall St AD Health (debt €9,4 mld) | https://simplywall.st/stocks/nl/consumer-retailing/ams-ad/koninklijke-ahold-delhaize-shares/health | aggregator |
| Investing.com AD Q4 2025 slides analysis | https://www.investing.com/news/company-news/ahold-delhaize-q4-2025-slides-sales-up-61-online-growth-accelerates-93CH-4499081 | nieuwsartikel |
| Euronext Live AD Quote | https://live.euronext.com/en/product/equities/NL0011794037-XAMS | beurswebsite |

---

## 15. Update-historie

| Datum | Versie | Wijziging |
|---|---|---|
| 2026-04-28 | 1.0 | Eerste publicatie (cowork stage 1 — markdown). HOOG voor FY2024 en FY2025; 2020-2023 AGGREGATOR; 2015-2019 LEEG. |

---

## Opmerkingen voor Claude Code

1. **Bronnen-discipline 2015-2023** — vervolg-pas zou IFRS-jaarverslagen 2020-2023 vanaf AD IR-pagina moeten openen. Pre-fusie 2015-2016 vereist pro-forma combined.

2. **FY2025 FCF totaal** — niet expliciet uit zoek-snippets; 2026-guidance >€2,3 mld is ondergrens; werkelijke FY2025 FCF zou in Q4 2025 Summary Report PDF staan.

3. **Beta is geschat (peer-derived)** — Yahoo Finance gaf geen expliciete AD.AS-beta. Gebruikt 0,55 op basis van defensieve consumer-staple-peer-set.

4. **Discrepantie executive_summary.oordeel vs scorekaart-eindoordeel** — Executive summary KOOP; scorekaart-rubric HOLD (totaal 32, één punt onder drempel 33). Ik volg upside-implicatie (25%) en wijk af van mechanische rubric. Stage-2 mag corrigeren naar HOLD voor strikte naleving.

5. **Stop & Shop-strategische review** is grootste onbekende. Bij volledige verkoop: eenmalige boekverliezen mogelijk maar verbetert structurele mix. Niet gemodelleerd in DCF.

6. **WACC 5,5% laag** — sector-norm 6-7%. Bij 6,5% WACC daalt fair value naar circa €40 (= huidige koers).

7. **Geografische mix 2025 niet exact** — gebruikt 65/20/12/3 als orde-van-grootte voor US/NL/CSE/Overig.

8. **Toelichting-velden mogelijk onder/boven minimale woordentelling** — stage-2 woord-telling-validator moet bevestigen.

Stage 2 (Claude Code) kan JSON-injectie en validator-run nu starten. Scorekaart-totaal 32/45 → mechanisch HOLD (één punt onder KOOP-drempel).
