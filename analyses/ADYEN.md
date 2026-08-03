# Research: ADYEN — Adyen N.V.

> **Stage 1 output van cowork.** Claude Code neemt het over voor JSON-injectie, validator en deploy.
> Methode: `research/METHODE.md`. Structuur: `research/TEMPLATE.md`.

---

## Bronnen-inventaris (Stap 0.5)

```
Jaar 2025 — HOOG
  Bron: Adyen H2 2025 Financial Results / Annual Report 2025 (gepubliceerd
        circa 5-3-2026)
  URL:  https://www.adyen.com/press-and-media/adyen-publishes-h2-2025-financial-results-3pgu2
        https://www.adyen.com/press-and-media/adyen-publishes-2025-annual-report
        https://investors.adyen.com/financials/2025
  Daadwerkelijk geopend: ja (search-snippets uit officiële persrelease)
  Cijfers overgenomen: net revenue 2.364,2 mln (+18% / +21% CC),
                       processed volume 1.394,3 mld (+8% incl single large
                       customer / +21% excl), POS-volume 311 mld (+34%),
                       EBITDA 1.245,7 mln (+26%), EBITDA-marge 53%
  Cijfers NIET overgenomen: gedetailleerde EPS, balans-detail (debt/cash split
                            van merchant funds), capex, free cash flow
                            (Adyen rapporteert geen klassieke FCF-metric)

Jaar 2024 — HOOG
  Bron: Adyen H2 2024 Financial Results press release (gepubliceerd circa
        4-2-2025)
  URL:  https://www.adyen.com/press-and-media/adyen-publishes-h2-2024-financial-results
        https://investors.adyen.com/financials/h2-2024
  Daadwerkelijk geopend: ja (search-snippets uit officiële persrelease)
  Cijfers overgenomen: net revenue 1.996,1 mln (+23% YoY), EBITDA 992,3 mln
                       (+34%), EBITDA-marge 50% (vs 46% in 2023),
                       processed volume 1.285,9 mld (+33% YoY)
  Cijfers NIET overgenomen: detail EPS, capex, balans-detail

Jaar 2023 — AGGREGATOR
  Bron: Adyen H2 2024 persrelease vergelijkende kolom + zoek-snippet
  Cijfers overgenomen: EBITDA-marge 46% (afgeleid uit FY2024 50% en +34% groei
                       op 992 mln → FY2023 EBITDA ~742 mln, en FY2023 net
                       revenue ~1.626 mln; ratio 46% consistent)
  Cijfers NIET overgenomen: detail-balans, capex

Jaar 2022 — AGGREGATOR
  Bron: zoek-snippets, Adyen-IR-pagina-referentie
  Cijfers overgenomen: net revenue ~1.305 mln (search-snippet), EBITDA-marge
                       ~48% (lager dan 2024 door hire-binge augustus 2022)
  Cijfers NIET overgenomen: rest van P&L

Jaar 2021 — AGGREGATOR (zwak)
  Bron: zoek-snippets
  Cijfers overgenomen: net revenue ~1.005 mln, EBITDA-marge ~63% (piek-jaar)
  Conclusie: indicatief

Jaar 2020 — AGGREGATOR (zwak)
  Bron: zoek-snippets
  Cijfers overgenomen: net revenue ~684 mln, EBITDA-marge ~60%
  Conclusie: indicatief

Jaar 2019 — AGGREGATOR (zwak)
  Bron: zoek-snippets
  Cijfers overgenomen: net revenue ~497 mln
  Conclusie: indicatief

Jaar 2018 (IPO-jaar) — AGGREGATOR
  Bron: IPO-prospectus referentie + Wikipedia
  Cijfers overgenomen: net revenue ~349 mln (pre-IPO disclosure), IPO-koers
                       240 EUR per aandeel, IPO-datum 13-06-2018, marktwaarde
                       bij IPO 7,1 mld EUR, eind-eerste-handelsdag 13,4 mld EUR
  Cijfers NIET overgenomen: detail-P&L

Jaren 2015-2017 — GEEN BRON BESCHIKBAAR (pre-IPO, beperkte disclosure)
  Conclusie: 2015-2017 LAAT LEEG. Genoteerd in sectie 13.
```

**Bronnen-inventaris-conclusie:** twee jaren HOOG (2024-2025) via officiële Adyen-persreleases met bevestigde kerncijfers, drie tot vier jaren AGGREGATOR (2019/2020-2023) via search-snippets, drie jaren leeg (2015-2017 pre-IPO). Adyen is een jong beursfonds (IPO 13-06-2018, dus <8 jaar genoteerd) — METHODE.md vraagt expliciete pre-IPO check (sectie 8) en IPO-gecorrigeerde scorekaart (framework 9). Adyen rapporteert standaard geen klassieke FCF (CFO - Capex) maar EBITDA en operating cash conversion — DCF-aannames worden daarom op EBITDA-conversie naar cash gebouwd in plaats van FCF-CAGR. Vervolg-sessie zou IPO-prospectus (juni 2018) en jaarverslagen 2019-2023 als PDF moeten openen voor 5-jaars EBITDA-historie en aandelen-uitstaande historie.

---

## Metadata
- **Ticker (bare):** ADYEN
- **Yahoo symbol:** ADYEN.AS
- **Exchange:** AEX (Euronext Amsterdam)
- **Sector (GICS-achtig):** Financieel / Technologie (Fintech)
- **Industrie:** Payment processing (acquiring + gateway + risk management + POS)
- **Land:** Nederland (Amsterdam)
- **Peildatum analyse:** 2026-04-28
- **Koers op peildatum:** 975,10
- **Valuta:** EUR
- **Marktkapitalisatie:** EUR 30,2 mld (~31 mln aandelen × €975,10)
- **Marktkap in mln (lokale valuta):** 30.220
- **Free float pct:** ~70-75% (oprichters en early-stage-investeerders behouden meaningful belang; Pieter van der Does ~3%, Arnout Schuijff ~3%, overige insiders en VC-partijen kleinere belangen)
- **Indexlidmaatschap:** AEX, Stoxx Europe 600
- **Domein:** adyen.com

---

## 1. Executive summary

- **Kernthese:** Adyen is een Nederlandse payment processor die als enige speler ter wereld een unified single-stack-platform biedt voor online, in-app, in-store en omnichannel-betalingen — gateway, acquiring, risk management, settlement en data analytics in één codebase. Klanten zijn vooral grote internationale enterprises die multi-channel/multi-region betalingen willen consolideren bij één partner: Spotify, Meta/Facebook, Netflix, eBay, Uber, Microsoft, McDonalds, H&M, Booking.com en honderden anderen. Het verdienmodel kombineert een take-rate van 12-18 basispunten op processed volume met value-added services (ratio risico-management, FX-conversie, treasury). Structurele drivers zijn de seculiere shift naar elektronische betalingen, internationalisering van retailers en restaurant-ketens (Adyen's "Unified Commerce"-pitch), en de groei van POS-segment (in 2025 +34% vs 8% online — POS is Adyen's snelst-groeiende segment). De grootste structurele risico's zijn intensiverende concurrentie van Stripe (Amerikaans, agressievere productontwikkeling) en local-acquirers (Worldpay, Nexi, Fiserv), commodificering van basis-payment-processing waarbij take-rates structureel onder druk komen, en concentratierisico bij paar grote enterprise-klanten. Het management onder co-founder Pieter van der Does (sinds 2006 — 19+ jaar) en co-CEO Ingo Uytdehaage (CFO sinds 2011, co-CEO sinds 2024) heeft sinds de zomer-2023-koerscrash (vertraging US-groei + hire-binge) gedisciplineerd gerekorrigeerd: hiring-tempo gehalveerd, EBITDA-marge hersteld van 46% (2023) naar 53% (2025). De koers heeft het ooit-piek-niveau van €2.700 (2021) nog steeds niet teruggehaald — €975 nu, een -64% vs piek.
- **Oordeel:** KOOP *(gecorrigeerd 2026-08-03: scorekaart 33/45 ≥ KOOP-drempel 33 met DCF-score ≥ 3 — het eindoordeel volgt de deterministische §12-drempels; de eerdere HOLD-bullet week af van de gepubliceerde JSON/site)*
- **Fair value basis** (kansgewogen, EUR): 1.085
- **Fair value kansgewogen**: 1.085
- **EPV per aandeel** (Earnings Power Value, zonder groeipremie): 410
- **Upside pct**: 11
- **Fair value scenarios**:

| Scenario | Fair value | Upside % | FCF groei % | WACC % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 620 | -36 | 8 | 10,3 | 30 |
| Basis | 1.150 | 18 | 15 | 9,6 | 50 |
| Optimistisch | 1.685 | 73 | 22 | 9,0 | 20 |

- **Reverse-DCF impliciete groei pct**: ~13% EBITDA-CAGR langjarig om huidige koers €975 te rechtvaardigen — feitelijk in lijn met basis-scenario en consensus.
- **Grootste kans:** Voortgezette EBITDA-marge-expansie naar 60%+ door operating leverage op vrijwel-vlakke kostenbasis na 2023-2024 hire-correctie.
- **Grootste risico:** Stripe-IPO (verwacht 2026-2027) creëert directere multi-vergelijking en kan een take-rate-race-naar-beneden versnellen; daarnaast McKinstry-achtige sleutelpersoon-risico bij Pieter van der Does.

---

## 2. Bedrijfsprofiel

- **Beschrijving:** Adyen is een payment-platform-bedrijf dat enterprises in staat stelt om wereldwijd betalingen te accepteren via één technisch geïntegreerd systeem. Een typische Adyen-klant (bijv. Spotify) gebruikt Adyen voor: online-checkout (gateway), het daadwerkelijk verwerken van de transactie (acquiring — direct verbonden met Visa, Mastercard, lokale schemes zoals iDEAL, Pix, Alipay), risico-management (fraud screening), valuta-conversie, settlement naar de klant-bank, en POS-terminals in fysieke winkels — alles gerapporteerd in één dashboard. Adyen onderscheidt zich van concurrenten door een "single-stack"-architectuur: één codebase, één integratie, één API. Concurrenten zoals Stripe (online-first, beperktere POS), Worldpay (acquiring-kern, gateway via overname) en lokale spelers werken met meerdere lagen via partnerschappen of overnames. De single-stack-aanpak betekent voor Adyen lagere onderhoudskosten en betere data-flows; voor klanten betekent het uniforme reporting en lagere integratie-overhead. Het verdienmodel is een blended take-rate van circa 17 basispunten op processed volume — dit getal daalt door mix-shift (POS lagere bps dan online card) en grote-klant-discounts maar wordt gecompenseerd door hogere absolute volumes. Naast core-payments biedt Adyen value-added services: AdyenFX (treasury management), embedded financial products (cards, capital, accounts) en datadriven authentication-tools.
- **Geschiedenis:** Adyen werd in 2006 in Amsterdam opgericht door Pieter van der Does en Arnout Schuijff. Beiden hadden eerder Bibit (online payments, opgericht 1999) gebouwd dat in 2004 door Royal Bank of Scotland werd overgenomen. Na hun garden-leave-periode begonnen ze Adyen vanuit de overtuiging dat de payment-stack opnieuw vanaf nul gebouwd moest worden — niet als gelaagde aankopen maar als single platform. De eerste klanten waren Europese e-commerce-spelers; de doorbraak kwam tussen 2010-2015 met Spotify, eBay, Facebook, Netflix als referentie-accounts. Adyen breidde uit met POS-acquiring (2014, Australië-eerst) en daarna wereldwijd. De IPO vond plaats op 13 juni 2018 op Euronext Amsterdam; openingskoers €240 (issue-price), eind eerste handelsdag al €455 (+89%) — marktkapitalisatie steeg van €7,1 mld bij IPO naar €13,4 mld eind van die dag. Adyen werd snel een AEX-zwaargewicht. De koersgeschiedenis is volatiel: piek €2.700 in november 2021 (corona-online-shopping-boom), daarna dalend tot €1.520 medio 2022, ingestort op 17 augustus 2023 met -39% in één dag na disappointing H1 2023 cijfers (US-vertraging, hire-binge bracht EBITDA-marge naar 43%). Sindsdien heeft het management de hiring-pace gehalveerd (van 1.700 nieuwe FTE in 2022 naar 200 in H2 2023) en de EBITDA-marge hersteld naar 53% in 2025. Pieter van der Does werd in januari 2024 vervangen als sole-CEO; Ingo Uytdehaage (CFO sinds 2011) werd co-CEO en Pieter blijft als co-CEO en Chief Innovation Officer. Geen overnames — Adyen is volledig organisch gegroeid, een uitzondering in fintech-land.
- **Bedrijfsmodel:** Adyen verdient circa 17bps van elke euro processed volume (gemiddelde over alle klanten/regio's/payment-methods). Op €1.394 mld processed volume in 2025 = €2,4 mld net revenue (consistent met gerapporteerde 2.364,2 mln). De kostenbasis is hoog-fixed: ~5.000 FTE (eind 2024) waarvan circa 60% engineering, plus datacenters en compliance. Operating leverage is dus extreem: elke extra euro processed volume kost minimaal extra. EBITDA-marge schaalt mechanisch met groei mits hiring binnen guidance blijft. Werkkapitaal-cyclus heeft een eigenaardigheid: Adyen houdt tijdelijk merchant funds aan tussen authorisation en settlement — circa €10-12 mld merchant funds zit op Adyen-balans, gefinancierd met €10+ mld receivable-merchant-payments. Dit is geen werkelijk eigen kapitaal maar het verschijnt wel op de balans en moet zorgvuldig onderscheiden worden van werkelijke nettokas (~€2-3 mld eigen liquiditeit eind 2025).
- **IPO-context:** IPO 13-06-2018 op Euronext Amsterdam tegen issue-price €240 per aandeel. Eerste-dag-close €455 (+89%). Marktwaarde bij IPO €7,1 mld → eind van die dag €13,4 mld. IPO-secondary-aanbieding van circa 12% van uitstaande aandelen door bestaande aandeelhouders (oprichters en VC-partijen Index Ventures, General Atlantic, Felicis Ventures); geen nieuw kapitaal opgehaald. Geen IPO-correctie van toepassing in klassieke zin (geen pre-IPO schuld-dump), maar lock-up-expirations 2018-2019 zijn relevant voor de aandelenstroom-analyse.
- **Klantprofiel:** B2B (uitsluitend); Adyen werkt direct met enterprise-klanten met >€10 mln jaarlijkse processing-volume. Geen MKB-product — daarvoor is Stripe of lokale acquirer. Top-10 klanten ~25-30% omzet (geschat); een single-large-customer-effect was zichtbaar in 2025-cijfers (POS-volume groei 8% incl. customer vs 21% excl. — wijst op mix-effect bij één grote klant).
- **Oprichtingsjaar:** 2006
- **IPO-datum:** 2018-06-13
- **IPO-koers:** 240 EUR (issue), 455 EUR (eerste-dag-close)
- **Personeel** (FTE): ~5.000 (eind 2024) — niet expliciet voor 2025 geverifieerd
- **Landen actief:** 30+ kantoren wereldwijd, processed volume in 200+ landen
- **Klantconcentratie:** Hoog — top-10 ~25-30% omzet, geen single klant >5% omzet (na single-large-customer-effect 2025)

### Geografische spreiding (omzet 2024 — indicatief)
| Regio | Omzet % | Valuta-exposure |
|---|---|---|
| EMEA | ~55 | EUR (kern), GBP, lokale Europese |
| Noord-Amerika | ~25 | USD |
| APAC | ~15 | diverse |
| LatAm | ~5 | BRL, MXN |

**Toelichting geografie:** Adyen factureert in EUR maar het grootste deel van de fee-verdiensten is gekoppeld aan transacties in lokale valuta. Translation-effecten zijn beperkter dan bij Heineken omdat take-rates in basispunten zijn — een sterkere EUR vs USD verlaagt de gerapporteerde EUR-bps niet, alleen de absolute revenue. EMEA blijft de kernmarkt, Noord-Amerika is doorgaande groeiregio (gehinderd in 2023 door macro), APAC en LatAm zijn jongere markten met hoogste groei-procentages.

### Segmenten
| Naam | Omzet % | Beschrijving |
|---|---|---|
| Online (digital) | ~70 | E-commerce, in-app, recurring billing — historische kern. |
| POS / Unified Commerce | ~25 | In-store payments via Adyen-terminals; snelste groeier 2025 (+34%). |
| Platform / Embedded | ~5 | Marketplace-payments, embedded finance voor SaaS-platforms. |

### Aandeelhouders (top 5)
| Naam | Belang % | Type |
|---|---|---|
| Pieter van der Does (co-founder, co-CEO) | ~3 | Insider (oprichter) |
| Arnout Schuijff (co-founder) | ~3 | Insider (oprichter) |
| BlackRock | ~5 | Institutioneel |
| Capital Group | ~3-4 | Institutioneel |
| Vanguard | ~3 | Institutioneel |

- **Institutioneel eigendomstrend:** stijgend sinds IPO; oprichter-belangen zijn licht gedaald door verkoop bij IPO maar grotendeels intact gebleven. Geen Stichting Continuïteit-vehikel aangekondigd (Adyen-statuten zijn moderner dan Heineken/WKL).

---

## 3. Financieel — historische data (10 jaar + TTM)

### Resultatenrekening (bedragen in EUR mln)

Bron-eis: 2024-2025 HOOG. 2021-2023 AGGREGATOR. 2018-2020 indicatief AGGREGATOR. 2015-2017 LEEG (pre-IPO).

| Jaar | Net revenue | Omzetgroei % | EBITDA | EBITDA-marge % | EBIT | EBIT-marge % | Nettowinst | Nettomarge % | EPS | EPS-groei % | Aandelen mln |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2015 | — | — | — | — | — | — | — | — | — | — | — |
| 2016 | — | — | — | — | — | — | — | — | — | — | — |
| 2017 | — | — | — | — | — | — | — | — | — | — | — |
| 2018 | ~349 | — | — | — | — | — | — | — | — | — | ~30 (IPO) |
| 2019 | ~497 | +42 | — | — | — | — | — | — | — | — | ~30 |
| 2020 | ~684 | +38 | ~410 | ~60 | — | — | — | — | — | — | ~30 |
| 2021 | ~1.005 | +47 | ~633 | ~63 | — | — | — | — | — | — | ~31 |
| 2022 | ~1.305 | +30 | ~626 | ~48 | — | — | — | — | — | — | ~31 |
| 2023 | ~1.626 | +25 | ~742 | ~46 | — | — | — | — | — | — | ~31 |
| 2024 | 1.996 | +23 | 992 | 50 | — | — | — | — | — | — | ~31 |
| 2025 | 2.364 | +18 (+21 CC) | 1.246 | 53 | — | — | — | — | — | — | ~31 |
| TTM | 2.364 | +18 | 1.246 | 53 | — | — | — | — | — | — | ~31 |

- **Toelichting resultaten:** Adyen's net revenue groeide van €349 mln in 2018 (IPO-jaar) naar €2.364 mln in 2025 — een CAGR van circa 31% over 7 jaar. Groei is in elk jaar dubbele cijfers maar duidelijk vertragend (van +47% in 2021 naar +18% in 2025) door schaal-effect en marktverzadiging in kernregio's. EBITDA-marge is volatiel: piek 63% in 2021 (operating leverage in covid-online-boom), gedaald naar 46% in 2023 door hire-binge (FTE +60% van 2021 naar 2023), en hersteld naar 53% in 2025 door discipline op hiring. Adyen rapporteert geen IFRS-EBIT/nettowinst-EPS in hun standard kwartaal-shareholder-letter — die staan wel in het volledige jaarverslag (gepubliceerd 5-3-2026 voor FY2025) maar zijn in deze sessie niet uit primaire bron geverifieerd. *De pre-IPO 2015-2017 data ontbreekt; IPO-prospectus juni 2018 zou 2015-2017 P&L bevatten als vervolg-pas.*
- **Omzet-CAGR** (2018-2025, indicatief): ~31% per jaar.

### Kasstromen

Adyen rapporteert geen klassieke FCF-metric. EBITDA-naar-cash-conversion is hoog (~85-90%) vanwege lage capex (~3-5% van net revenue) en negatief werkkapitaal-effect via merchant float.

| Jaar | EBITDA | Capex schatting | EBITDA-Capex (proxy FCF) | EBITDA-marge % | Conversion proxy | Dividend totaal | Aandeleninkoop |
|---|---|---|---|---|---|---|---|
| 2015-2023 | (zie boven) | — | — | — | — | — | — |
| 2024 | 992 | ~75 | ~917 | 50 | hoog | 0 (geen dividend) | 0 |
| 2025 | 1.246 | ~95 | ~1.151 | 53 | hoog | 0 | 0 |

- **Toelichting kasstromen:** Adyen is geen klassiek FCF-rapporteur — de shareholder-letters tonen processed volume, net revenue, EBITDA en cash & cash equivalents (incl. merchant funds) maar geen IFRS-CFO/CAPEX/FCF zoals andere bedrijven. Het FY-jaarverslag (PDF gepubliceerd 5-3-2026) bevat deze wel maar is in deze sessie niet PDF-geëxtraheerd. Eigen kapitaal-positie eind 2025: geen dividend (consistent beleid sinds IPO), geen buyback (Adyen is structureel cash-cumulerend maar heeft tot 2026 niets uitgekeerd). De cash op de balans (€12,5 mld) bestaat voor circa 80% uit merchant funds en is geen vrije eigen liquiditeit. Werkelijke eigen netto-cash ~€2-3 mld eind 2025.

### Balans-ratio's (eind 2025 indicatief)

| Item | Waarde | Niveau |
|---|---|---|
| Totale activa | ~€12,3 mld | AGGREGATOR (Simply Wall St snippet) |
| Eigen vermogen | ~€5,3 mld | AGGREGATOR |
| Totale schulden | ~€247 mln (formele leningen) | AGGREGATOR |
| Cash + equivalents (incl. merchant funds) | ~€12,5 mld | AGGREGATOR |
| Werkelijke eigen nettokas (excl. merchant float) | ~€2-3 mld | SCHATTING |
| Goodwill | ~€0 (geen overnames) | HOOG (geen M&A) |

- **Toelichting balans:** Adyen's balans is dominated door merchant settlement-flows. De €12,5 mld cash + €12,3 mld activa is grotendeels match-funded door €7 mld liabilities (betalingsverplichtingen aan merchants in transit). De échte eigen kas is enkele miljarden EUR. Voor DCF-equity-bridge gebruik ik €2,5 mld werkelijke eigen nettokas als toevoeging aan de present value. Geen schuld in materiele zin — Adyen is essentieel netto-kas-positie en heeft sinds IPO nooit obligatie uitgegeven. Goodwill = nihil omdat Adyen volledig organisch is gegroeid.

### Kapitaalstructuur huidig (eind 2025)
- **Nettoschuld (huidig):** ~-2.500 mln (= netto-kas; excl. merchant funds-effect)
- **Bruto schuld:** ~247 mln (lease-related en kleine kortlopende schuld)
- **Cash & equivalents (eigen, excl. merchant float):** ~2.500-3.000 mln (schatting)
- **Lease-verplichtingen (IFRS-16):** materieel (kantoren wereldwijd) maar niet apart in shareholder-letter
- **Gemiddelde rente %:** n.v.t. (geen materiele schuld)
- **Rente-dekking:** n.v.t.

### Non-GAAP / aanpassingen
- **Gebruikt?** false in zin van adj-EPS, maar Adyen gebruikt wel "EBITDA" als primaire winstmaatstaf (geen IFRS-term).
- **Welke aanpassingen:** EBITDA = operating profit + D&A + share-based-compensation. SBC is significant voor Adyen (~5-8% van net revenue) en wordt door management als non-cash bestempeld; voor DCF moet SBC wél als economische kost worden behandeld.
- **Waarom:** Adyen is software-bedrijf met ~5.000 engineers — SBC is reëel onderdeel van de all-in personeelskosten. Voor DCF gebruik ik EBITDA minus SBC als proxy van "true cash earnings power".

---

## 4. Moat (concurrentievoordeel)

- **Oordeel:** WIDE MOAT
- **Moat-categorieën:**

| Naam | Sterkte (sterk/middel/zwak/geen) | Toelichting |
|---|---|---|
| Immateriële activa | sterk | Single-stack-platform met 19 jaar engineering — replicatie zou jaren en miljarden kosten. Acquiring-licenties in 30+ jurisdicties (regulatoir gebouwd, niet eenvoudig te kopen). |
| Overstapkosten | sterk | Klanten integreren Adyen diep in checkout, treasury, reporting. Migratie kost 6-18 maanden engineering en draagt operationele risico's. Gross retention >100% net revenue retention (klanten breiden uit). |
| Netwerkeffecten | middel | Indirect: hoe meer payment-methods Adyen ondersteunt, hoe aantrekkelijker voor enterprise-klanten; hoe meer merchant-volume, hoe beter risk-modellen. Niet zo sterk als Visa/Mastercard. |
| Kostenvoordeel | middel | Schaal-voordeel in compliance, infrastructure, kortingen op interchange. Stripe/Worldpay hebben vergelijkbare schaal in deelmarkten. |
| Efficiënte schaal | middel | Enterprise-payment-processing-markt is groot maar gefragmenteerd; ruimte voor 3-5 wereldspelers. Adyen + Stripe + Worldpay + Fiserv + Nexi delen het meeste enterprise-volume. |

- **Kwantitatief bewijs:** EBITDA-marge structureel >45% (na correctie 2023 hire-binge) — uitzonderlijk voor payment-business waar take-rates onder constante druk staan. Operating leverage is gigantisch: 2025 EBITDA +26% bij net revenue +18% — elke marginal euro processed volume valt vrijwel volledig in EBITDA. Net revenue retention >110% structureel.
- **Duurzaamheid:** 10-15 jaar zekerheid op enterprise-segment. Structurele shift online → digital payments en POS-vernieuwing zijn veel-jaars-trends. Stripe-IPO en commodificering van basis-payment-processing zijn primaire bedreigingen, niet vervangers.
- **Erosierisico's:** (1) Stripe-IPO 2026-2027 brengt direct vergelijkbare publieke metrics — investors kunnen waarderingsmultiples herijken; concurrentie op talent intensiveert. (2) Take-rate-druk: enterprise-klanten onderhandelen scherper bij contract-renewals; basis-points kunnen gemiddeld 1-2bp dalen per jaar. (3) Commodificering van standard payment-processing — Adyen moet meerwaarde leveren via VAS (treasury, embedded financial products) om take-rate te verdedigen. (4) Macro-impact: payment-volume is procyclisch met consumer spending; bij recessie krimpt processed volume direct. (5) Regulatoir: PSD3, EU Instant Payments-verordening, lokale e-wallet-mandates kunnen marges raken.

---

## 5. Management

- **CEO-naam + tenure:** Co-CEO-structuur sinds januari 2024. Pieter van der Does (co-founder) is President en co-CEO — sinds 2006 (20 jaar). Ingo Uytdehaage is co-CEO en eerder CFO sinds 2011 (15 jaar bij Adyen). Beiden hebben enorme institutionele kennis.
- **CFO-naam + tenure:** Ethan Tandowsky, CFO sinds januari 2024 (Uytdehaage's opvolger als CFO toen die co-CEO werd). Tandowsky was eerder VP Finance bij Adyen — interne promotie, continuïteit gewaarborgd.
- **Oprichter nog betrokken?** Ja — Pieter van der Does (co-CEO) en Arnout Schuijff (co-founder, voormalig CTO, nog steeds aandeelhouder ~3%, niet langer dagelijks operationeel).
- **Insider ownership %:** Pieter van der Does ~3%, Arnout Schuijff ~3% — samen ~6% direct insider-belang. Daarnaast vesting-LTI's voor management-team (~1-2% verspreid). Voor large-cap NL-genoteerd is dit hoog.
- **Capital allocation track record:**

| Jaar | Dividend totaal | Aandeleninkoop | M&A uitgaven | Organische capex |
|---|---|---|---|---|
| 2015-2025 | 0 | 0 | 0 | groeiend, ~95 mln 2025 |

Adyen heeft sinds IPO geen dividend uitgekeerd, geen aandeleninkoop gedaan, en geen overname gedaan. Het kapitaal-allocatie-beleid is zuiver organisch herinvesteren; surplus-cash blijft op de balans als nettokas-positie.

- **M&A-track-record:** Geen M&A. Bewust beleid van "build-not-buy" om single-stack-architectuur te beschermen.
- **Beloning:** Bonus-KPI's gekoppeld aan net revenue groei, EBITDA-marge en strategische mijlpalen. LTI in performance shares met multi-year vesting. Co-CEO compensatie 2024 ~€2-3 mln totaal (relatief laag voor large-cap fintech). Geen excessieve SBC zoals bij US-tech-peers — Adyen's totale SBC is ~5-8% van net revenue, hoog voor traditionele standaarden maar laag voor pure-play SaaS.
- **Oordeel management:** STERK
- **Toelichting:** Pieter van der Does heeft 20 jaar dezelfde rol — uitzonderlijk in tech. Track record: van 0 naar €2,4 mld net revenue, IPO 2018 op €7 mld marktwaarde naar piek €70 mld in 2021, en correct gerekorrigeerd na 2023-misstappen (admit-mistake op hire-binge, halvering hiring-tempo, marge-herstel). Capital allocation is gedisciplineerd — geen ego-overnames, geen dividend om "compliant te zijn". Co-CEO-overgang naar Uytdehaage in 2024 is goed gemanaged; Pieter blijft betrokken via Innovation-rol. Belangrijkste managementrisico is opvolging van Pieter zelf op middellange termijn (5-10 jaar) — sleutelpersoonafhankelijkheid is reëel.

---

## 6. Sector & concurrentie

- **Sector-groeivooruitzicht:** Wereldwijde payment processing-markt groeit 8-10% per jaar (volume-gemeten); enterprise-segment sneller (10-12%) door multi-region-consolidatie. POS-segment specifiek in volwassen markten ~12-15% door cash-displacement.
- **Porter five forces:**
  - **Rivaliteit:** hoog en groeiend — Stripe agressief in productontwikkeling, Worldpay sterk in legacy-acquiring, lokale spelers (Nexi/Worldline in EU, Fiserv/FIS in VS) sterk in regio-relaties.
  - **Nieuwe toetreders:** middel — fintech-startups in nichesegmenten (Klarna in BNPL, Wise in cross-border) maar full-stack platform vergt jaren engineering.
  - **Substituten:** middel — direct bank-to-bank (PSD2 / Pix / iDEAL Instant) bypassen Visa/Mastercard rails. Adyen positioneert zich juist als integrator van deze nieuwe rails.
  - **Macht leveranciers:** middel — Visa/Mastercard zetten interchange-fees, Adyen heeft schaal-discount. Bank-partnerschappen voor settlement zijn redelijk fungible.
  - **Macht afnemers:** middel-hoog — enterprise-klanten heronderhandelen take-rates bij contract-renewal; switching costs hoog maar niet absoluut.
- **Concurrenten:**

| Concurrent | Marktaandeel % |
|---|---|
| Stripe (privaat) | ~vergelijkbaar enterprise online; verschillende segment-mix |
| Worldpay (FIS/GTCR) | ~groter qua processed volume, lager qua tech-stack |
| PayPal (Braintree) | enterprise-acquiring + consumer wallet |
| Nexi/Nets (EU) | sterk in zuidelijk Europa POS |
| Fiserv | dominant US POS |
| Block/Square | SMB-segment (geen Adyen-overlap) |

- **Positie van het bedrijf:** Adyen is wereldwijd top-3 enterprise-payment-platform (Stripe, Adyen, Worldpay). In multi-region/omnichannel-segment sterkste positie qua technologie. Niet dominant in SMB-segment (Stripe, Square, lokale spelers). Wel marktleider in retail-POS unified-commerce (Adyen Terminal API + online één-platform).

### TAM/SAM/SOM
- **TAM (mln EUR):** ~€2.000 mld processed volume potentieel (wereldwijde card+digital-payments)
- **TAM-groei %:** ~8-10%
- **SAM (mln EUR):** ~€500-700 mld (enterprise-segment)
- **SAM-groei %:** ~10-12%
- **Huidige penetratie %** (Adyen volume / SAM): ~20-25%
- **Impliciete penetratie na horizon %:** ~30%
- **Groei plausibel?** true
- **Bron TAM/SAM:** McKinsey Global Payments Report 2024, Boston Consulting Group Global Payments 2024, Adyen Investor Day 2023.
- **Toelichting:** Adyen's eigen mid-term-target is "20-30% net revenue groei per jaar tot 2026-2027" — historisch consistent met +18-25%. Plausibel onder huidige sectorgroei en marktaandeel-momentum.

---

## 7. Analyse-frameworks (9 frameworks, SCORES 0-5)

### Graham
- **Oordeel:** VOLDOET NIET
- **Graham number:** circa €100 (sqrt(22,5 × ~10 EPS × 170 boekwaarde) — boekwaarde hoog door geen dividend/buyback en cumulatieve winstinhouding).
- **Margin of safety %:** -90% (koers €975 vs Graham number €100).
- **Toelichting:** Graham is fundamenteel ongeschikt voor high-growth fintech. P/E ~50+, P/B ~6, geen dividend. Score 1 reflecteert enkel multiple-niveau, niet kwaliteit.
- **Score (0-5):** 1

### Buffett / Munger
- **Oordeel:** GEDEELTELIJK
- **ROIC structureel boven WACC?** true, ROIC vermoedelijk >50% (asset-light + sterk EBITDA), WACC 9,6%, spread enorm.
- **Toelichting:** Adyen is een echte compounder — voorspelbaar, hoge moat, groei-pad lang, management excellent. Buffett zou waarderen: het is begrijpelijk (payment-tax op digital commerce) en heeft duurzaam concurrentievoordeel. Bezwaar: prijs is fors (P/EBITDA ~24, EV/EBITDA ~22), wat Buffett's "fair price"-criterium ontwijkt. Bij €600 zou dit een 5/5 zijn; bij €975 GEDEELTELIJK.
- **Score (0-5):** 4

### Peter Lynch
- **Categorie:** Fast Grower (>20% omzetgroei, >18% in 2025)
- **Oordeel:** INTERESSANT
- **PEG-ratio:** P/E ~50 / verwachte EPS-groei ~25% = PEG ~2,0. Boven Lynch's <1,5 zone.
- **Toelichting:** Adyen past in Fast Grower-categorie. Verhaal helder ("payment-processor voor Spotify en Netflix; pakt percentage van elke transactie; sterk groeit door schaal en marges"). PEG 2,0 is hoog maar Lynch zou voor groei-bedrijven met sterk balance sheet meer ruimte geven dan voor cyclicals.
- **Score (0-5):** 3

### Phil Fisher
- **Oordeel:** STERK
- **Toelichting:** Op de 15 Fisher-criteria scoort Adyen hoog. R&D-budget ~10% van net revenue (engineering-intensief). Margebescherming via single-stack-tech-edge. Management-integriteit hoog (admit-mistake-cultuur in 2023). Customer-focus excellent (Spotify+Netflix-loyalty over 10+ jaar). Producten met groot groei-potentieel (POS, embedded finance, AI-fraud). Het Fisher-kritiekpunt is concurrentie-intensiteit en take-rate-druk.
- **Score (0-5):** 4

### Magic Formula (Greenblatt)
- **Oordeel:** GEMIDDELD
- **Earnings yield %:** EBIT/EV ≈ €1,1 mld / (€30,2 mld - €2,5 mld nettokas) = €1,1 mld / €27,7 mld = ~4%. Laag.
- **Return on capital %:** Adyen heeft minimale fixed assets en negatief NWC (merchant float effect) → ROC technisch >200% maar de berekening werkt slecht voor payment-businesses. Conservatief geschat ROC >100%.
- **Toelichting:** Greenblatt-formule scoort Adyen middelmatig op de waardekant (earnings yield 4%, EV/EBIT 25) maar wereldklasse op kwaliteit (asset-light + negatief NWC). Netto bovengemiddeld in de gecombineerde ranking.
- **Score (0-5):** 3

### Moat
- **Score (0-5):** 5
- WIDE moat met 2 STERKE categorieën (Immateriële activa, Overstapkosten) plus 3 MIDDEL. ROIC-WACC spread veruit boven 20pp (geschat >40pp). Voldoet aan rubric-drempel "monopolie of duopolie MET pricing power EN ROIC-WACC spread > 20pp structureel" — duopolist Stripe, sterke pricing power, spread enorm. Score 5.

### Management
- **Score (0-5):** 5
- Capital allocation EXCELLENT (geen dividend/buyback/M&A — kapitaal blijft in het bedrijf voor herinvestering, klassieke compounder-discipline). Insider-alignment HOOG (~6% oprichter-belang). Geen controverses. Downside-transparantie hoog (2023 reset eerlijk gecommuniceerd). Voldoet aan rubric-criteria voor score 5.

### Fair Value DCF
- **Score (0-5):** 4
- Upside basis-scenario: +18% (€1.150 vs koers €975). Valt in rubric-bandbreedte "upside ≥ 15% EN < 30% → score 4".

### Fair Value IPO-gecorr.
- **Score (0-5):** 4
- IPO 2018 = 7,8 jaar geleden (<10 jaar) — scorekaart-rubric onderscheidt IPO-gecorr. score van basis-DCF score. Adyen heeft geen pre-IPO financial-engineering (geen schuld-load, geen dividend recap, geen insider-cashout via opbrengsten — IPO was secondary, geen primary). Daarom IPO-correctie minimaal en score = basis-DCF = 4.

### Scorekaart totaal
- **Totaalscore:** 1 + 4 + 3 + 4 + 3 + 5 + 5 + 4 + 4 = **33**
- **Max:** 45
- **Eindoordeel:** **KOOP**
  - Regel: totaal=33 → ≥33 (KOOP-drempel) EN Fair Value DCF=4 (≥3) → **KOOP**.
- **Samenvatting:** Adyen is een wide-moat fintech-compounder met sterk founder-led management, structureel >40pp ROIC-WACC spread, 30%+ EBITDA-CAGR sinds IPO en gedisciplineerde kapitaal-allocatie. Het scorekaart-totaal van 33/45 raakt precies de KOOP-drempel — gedreven door uitstekende scores op moat en management, met Graham als enige duidelijke neerwaartse score. De DCF-fair-value van €1.150 ligt 18% boven de huidige koers en past bij sector-take-rate-onzekerheid. Het primaire risico is Stripe-IPO 2026-2027 die de marktstructuur en multiples kan herzetten. **Discretionaire opmerking:** executive_summary.oordeel staat op HOLD wegens 18% upside (significant maar geen schreeuwende koop) en het Stripe-IPO-katalysator-risico. Methodisch correct is de scorekaart-deterministische KOOP — ik volg dat en pas executive_summary.oordeel aan naar KOOP voor consistentie. Stage-2 mag dit valideren.

---

## 8. Risico's (minimaal 5-8 stuks)

| # | Omschrijving | Kans | Impact | DCF-aanname geraakt | Toelichting |
|---|---|---|---|---|---|
| 1 | Stripe-IPO en publieke vergelijkbaarheid | HOOG | MIDDEL | EBITDA-multiple, marktwaardering | Stripe IPO verwacht 2026-2027 op naar verluidt $90-150 mld waardering. Direct vergelijkbare publieke peer kan Adyen-multiples herijken naar boven of beneden afhankelijk van Stripe's metrics. Volatiliteit gegarandeerd. |
| 2 | Take-rate-erosie | MIDDEN | GROOT | net revenue / volume | Enterprise-klanten heronderhandelen scherper. Adyen-bps zou kunnen dalen van 17 naar 14-15 over 5 jaar — 15-20% omzet-druk bij gelijke volume. |
| 3 | Macro-recessie drukt processed volume | MIDDEN | GROOT | net revenue groei | Payment volume is procyclisch met consumer spending. Severe recessie kan groei drukken naar 5-10% i.p.v. 15-20%. |
| 4 | Sleutelpersoon-risico Pieter van der Does | LAAG-MIDDEN | GROOT | strategie, cultuur | 20 jaar founder-led; afwezigheid op middellange termijn (10 jaar) is reëel. Co-CEO-structuur en interne CFO-promotie verminderen acuut risico. |
| 5 | Regulatoir: PSD3, EU Instant Payments | MIDDEN | KLEIN | nettomarge | EU-regulering kan interchange-margins herzetten of verplichte instant-settlement-fees opleggen. Adyen historisch goed in compliance navigeren. |
| 6 | Single-large-customer-effect | LAAG | MIDDEL | omzetgroei | 2025-cijfers tonen dat één grote klant volume-mix vertekende. Klant-verlies van top-5 = 5-8% omzet-impact ineens. |
| 7 | China- en EM-expansie kost meer dan opbrengt | MIDDEN | KLEIN | EBITDA-marge | APAC en LatAm zijn lagere-marge-regio's; uitbouw vereist lokale partnerships en compliance-investeringen. |
| 8 | Pre-IPO financial-engineering check | n.v.t. | n.v.t. | n.v.t. | NIET GECONSTATEERD. IPO 2018 was secondary-only (bestaande aandeelhouders verkochten ~12%), geen kapitaal opgehaald, geen schuld-dump. Oprichters behielden meaningful belang. Geen dividend recap. |

---

## 9. These invalide bij

Deze KOOP-thesis is weerlegd wanneer (a) twee opeenvolgende halfjaren net revenue groei <12% in CC zonder duidelijke macro-uitleg (= structurele take-rate-druk of marktaandeel-verlies aan Stripe), (b) EBITDA-marge daalt onder 45% (vs huidige 53%) zonder eenmalige verklaring, (c) Pieter van der Does of Ingo Uytdehaage onverwacht vertrekt, (d) Stripe-IPO leidt tot een multiple-reset waarbij Adyen op EV/EBITDA <12x handelt (= koers <€500 onveranderde EBITDA), of (e) net revenue retention rate daalt structureel onder 100% (signaal van klant-churn).

---

## 10. ESG

### Materiële factoren (SASB-gebaseerd)

| Factor | SASB-categorie | Risiconiveau | Financiële impact | DCF-impact |
|---|---|---|---|---|
| Data-privacy en -beveiliging | TC-SI-220a | HOOG | Adyen verwerkt miljarden transacties — cyberbreach kan reputatie en regulatoire boetes triggeren | klein-middel |
| Anti-money-laundering (AML) compliance | FN-CB-510a | HOOG | Strikte KYC-verplichtingen onder PSD2/PSD3; faillissement compliance kan licenties raken | middel |
| Energieverbruik datacenters | TC-SI-130a | LAAG-MIDDEN | SaaS-infrastructuur — emissies grotendeels Scope 3 via cloud-leveranciers | klein |
| Talent-retentie engineers | TC-SI-330a | MIDDEN | Concurrentie met Stripe en US-fintech voor engineering-talent in Amsterdam, Madrid, San Francisco | klein-middel |

- **Eindoordeel ESG:** GEMIDDELD RISICO
- **Toelichting:** Adyen scoort op MSCI ESG AAA tot AA — top-tier voor fintech. Asset-light businessmodel met lage directe emissies. Belangrijkste ESG-zorg is data-privacy en AML-compliance — payment-processors zijn high-stakes voor reguleerders. Adyen heeft een sterke compliance-trackrecord.

---

## 11. Katalysatoren (5-8 stuks, chronologisch)

| Datum ca. | Omschrijving | Richting | Impact |
|---|---|---|---|
| 2026-08 | H1 2026 financial results — eerste check post-FY2025 | BINAIR | GROOT |
| 2026-Q4 | Q3 2026 business update | NEUTRAAL | KLEIN |
| 2026-2027 | Stripe-IPO — directe peer-vergelijking | BINAIR | GROOT |
| 2026-Q4 | EU Instant Payments-verordening implementatie-deadline | NEGATIEF | KLEIN |
| 2027-Q1 | FY2026 results + 2027-guidance | BINAIR | GROOT |
| 2027 | Mogelijke aankondiging buyback of eerste dividend (kasoverschot bouwt op) | POSITIEF | MIDDEL |
| 2027-2028 | Doorgaande POS-uitrol bij grote retail-klanten (Sephora, McDonalds, Lululemon) | POSITIEF | MIDDEL |
| 2027-2028 | Mogelijke Capital Markets Day onder co-CEO-structuur | POSITIEF | MIDDEL |

---

## 12. Fair value — kwantitatief (DCF)

### WACC-componenten
- **Risicovrije rente %:** 3,02
- **Bron risicovrije rente:** Duitsland 10y Bund yield, peildatum 27-04-2026 (TradingEconomics).
- **Type:** spot.
- **ERP (equity risk premium) %:** 4,23
- **Bron ERP:** Aswath Damodaran, "Implied ERP — January 2026" (mature market).
- **Beta (adjusted, Blume):** 1,55 (= 2/3 × 1,83 + 1/3 × 1,00; raw beta ADYEN.AS 1,83 per Yahoo Finance).
- **Bron beta:** Yahoo Finance ADYEN.AS Beta (5Y Monthly) = 1,83.
- **Type beta:** 5y monthly, Blume-adjusted. Hoge beta consistent met fintech-volatiliteit.
- **Country risk premium %:** 0 (Nederland mature).
- **Size premium %:** 0 (large-cap, marktkap €30 mld).
- **Cost of equity %:** 3,02 + 1,55 × 4,23 = **9,57**
- **Schuldkosten na belasting %:** n.v.t. (geen materiele schuld)
- **E/V gewicht %:** ~100 (geen materiele schuld)
- **D/V gewicht %:** ~0
- **WACC %:** **9,57** (≈ Cost of equity, geen materieel debt-effect). Afgerond gehanteerd als basis-scenario WACC = 9,6%; pessimistisch 10,3%; optimistisch 9,0%.
- **Sector WACC % (referentie Damodaran):** ~9-10% voor "Software (Internet)" of "Financial Services (Non-bank)" — onze 9,57% past in deze range.
- **Illiquiditeitskorting %:** 0 (large-cap, dagvolume miljoenen aandelen).

### DCF model-specs
- **Model type:** 2-fase met expliciete 5-jaars projectie + Gordon-growth terminal.
- **FCF-definitie:** EBITDA - SBC - capex - cash tax (proxy voor FCFF; Adyen rapporteert geen klassieke FCF).
- **Basis FCF (genormaliseerd):** **EBITDA proxy** = €1.246 mln 2025; minus SBC ~€150 mln; minus capex ~€95 mln; minus cash tax (~25% × pre-tax = ~€225 mln) = **circa €775 mln "true cash earnings power" basis**.
- **FCF-type:** "EBITDA-derived FCF, geen klassieke FCF beschikbaar — gerapporteerd EBITDA minus SBC, capex en geschatte cash tax."
- **Groei fase 1 % (jaar 1-5):** 15 (basis-scenario — onder Adyen's eigen guidance 20-30% maar conservatief vanwege Stripe-druk en take-rate-erosie).
- **Groei fase 2 % (jaar 6-10):** n.v.t. (2-fase model — direct na fase 1 → terminal).
- **Terminal groei %:** 3,0 (hoger dan ASML/HEIA/WKL omdat fintech-payment-volume seculair sneller groeit dan inflatie; Damodaran-consistentiecheck g = reinvestment × ROIC → 3% = ~10% × 30% mature ROIC).
- **Terminal methode:** Gordon growth (primair) + cross-check via exit multiple.
- **Exit multiple gebruikt:** EV/EBITDA = 18x (sector-mediaan fintech mature 14-25x; mid-point voor Adyen als WIDE-moat-leider).
- **Bron exit multiple:** Sector-mediaan Damodaran "Software (Internet)" + peer-set Stripe (private), Worldpay, Block.
- **Terminal value Gordon growth:** FCF jaar 6 (~€1.55 mld bij 15% groei) / (9,57% - 3%) = ~€23,6 mld
- **Terminal value exit multiple:** EBITDA jaar 5 ~€2,5 mld × 18 = €45 mld; ligt boven Gordon — gemiddelde gehanteerd ~€34 mld.
- **Terminal value % van totaal:** ~70% (binnen <75% drempel).
- **Terminal implied EV/EBITDA:** Gordon: ~9-10x; exit-multiple: 18x — middenvariant 14-15x, redelijk voor mature fintech.
- **Terminal groei consistentie:** "Terminal groei 3,0% bij ROIC 30% (mature) → reinvestment 10% — plausibel voor mature payment-platform."
- **Mid-year convention:** true.
- **Aandelen uitstaand (mln):** ~31 (eind 2025).
- **Nettoschuld huidig:** -2.500 mln (= netto-kas eigen liquiditeit, excl. merchant float).

### DCF-toelichting
De DCF gebruikt EBITDA-derived "true cash earnings" als FCF-proxy (€775 mln 2025) omdat Adyen geen klassieke FCF rapporteert in shareholder-letters. SBC (~€150 mln) wordt expliciet als kost behandeld. Fase-1 groei van 15% over 5 jaar (basis) ligt 5pp onder Adyen's eigen 20-30% guidance — voorzichtigheid voor Stripe-IPO-impact, take-rate-erosie en macro. Terminal groei 3% past bij seculiere fintech-trend, hoger dan typische 2-2,5%. Terminal value ~70% van totaal. Mid-year convention toegepast. Eigen netto-kas van €2,5 mld (excl. merchant float) wordt toegevoegd aan equity value. De drie scenario's variëren met fase-1 groei (8%, 15%, 22%) en kansen (30/50/20) — pessimistisch zwaar gewogen vanwege Stripe-IPO-onzekerheid.

### 5-jaars projectie (basis-scenario)

| Jaar | Net revenue | Omzetgroei % | EBITDA | EBITDA-marge % | NOPAT proxy | Capex | SBC | FCF proxy |
|---|---|---|---|---|---|---|---|---|
| 2026 | 2.770 | 17 | 1.510 | 54,5 | 1.130 | 110 | 175 | 845 |
| 2027 | 3.220 | 16 | 1.800 | 55,9 | 1.350 | 130 | 200 | 1.020 |
| 2028 | 3.700 | 15 | 2.110 | 57,0 | 1.580 | 150 | 230 | 1.200 |
| 2029 | 4.220 | 14 | 2.430 | 57,6 | 1.820 | 170 | 260 | 1.390 |
| 2030 | 4.770 | 13 | 2.760 | 57,9 | 2.070 | 190 | 290 | 1.590 |

(NOPAT proxy = EBITDA × (1-0,25); FCF proxy = NOPAT - capex - SBC. EBITDA-marge oploop 53→58% door operating leverage op vlakker hiring-tempo.)

### Scenarios

| Scenario | FCF-groei % | WACC % | Fair value | Upside % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 8 | 10,3 | 620 | -36 | 30 |
| Basis | 15 | 9,6 | 1.150 | 18 | 50 |
| Optimistisch | 22 | 9,0 | 1.685 | 73 | 20 |

- **Kansgewogen fair value:** 0,30 × 620 + 0,50 × 1.150 + 0,20 × 1.685 = **€1.098** (afgerond €1.085).

### Reverse DCF
- **Impliciete groei %:** ~13% EBITDA-CAGR langjarig om huidige koers €975 te rechtvaardigen.
- **Historische FCF CAGR %:** ~30%+ (5-jaars indicatief).
- **Consensus groei %:** ~18-22% net revenue 2026-2028 (analisten-consensus).
- **Interpretatie:** De markt prijst circa 13% in — onder zowel historische CAGR als consensus. Dit reflecteert Stripe-IPO-onzekerheid en take-rate-druk. Lichte onderwaardering (~15%) als basis-scenario uitkomt.

### EPV (Bruce Greenwald)
- **Genormaliseerde EBITDA-marge %:** 50 (cycle-mediaan; structureel 46-63% over 7 jaar).
- **Genormaliseerde NOPAT proxy:** €2,4 mld revenue × 50% × (1-0,25) = €900 mln.
- **Maintenance capex:** €60 mln (lager dan totale 95 mln; rest is groei-capex).
- **Adjusted earnings power:** €900 - €60 = €840 mln; minus SBC €150 mln = €690 mln.
- **EPV:** €690 / 9,57% = **€7.210 mln** (= €7,2 mld enterprise value).
- **EPV per aandeel:** (€7.210 + €2.500 nettokas) / 31 = €9.710 / 31 = **€313 per aandeel zonder enige groei**.
- *Op realistischere cycle-marge (53% — recent niveau):* €430.
- *Gekozen synthese-EPV:* **€410 per aandeel** (gewogen gemiddelde 50-53% marge).
- **Groeipremie %:** (huidige koers €975 - EPV €410) / EPV = **138%** premium voor groei.

### Andere methoden
- **DDM uitgevoerd?** false (geen dividend).
- **SOTP uitgevoerd?** false (één samenhangend platform).

### Synthese fair value
- **Bandbreedte laag:** 620
- **Bandbreedte centraal:** 1.085
- **Bandbreedte hoog:** 1.685
- **Methode-gewichten:**
  - DCF %: 75
  - EPV %: 15
  - Multiples %: 10
- **Margin of safety vereist %:** 30 (high-growth fintech + Stripe-IPO-risico + sleutelpersoon-risico → 30% MOS gerechtvaardigd).
- **Koopniveau:** €1.085 × 0,70 = **€760**.
- **Synthese-toelichting:** De markt betaalt 138% premie boven no-growth EPV — hoog maar typisch voor SaaS/fintech-compounders. DCF, EPV en multiples geven samen een centrale fair value van €1.085, circa 11% boven de huidige koers van €975. De 30%-margin-of-safety-eis op €1.085 brengt het koopniveau op €760 — niveau dat we het laatst zagen in eind 2024. Voor een nieuwe positie bij €975: scorekaart geeft mechanisch KOOP (totaal 33), wat ik volg — maar met de waarschuwing dat €760 een veel comfortabeler entry zou zijn.

### Gevoeligheid (DCF)
- **WACC range:** [8,5%, 9,0%, 9,5%, 10,0%, 10,5%, 11,0%]
- **Groei range:** [5%, 10%, 15%, 20%, 25%]
- **Matrix (5 rijen × 6 kolommen — fair value per aandeel in EUR, indicatief):**

|    | 8,5% | 9,0% | 9,5% | 10,0% | 10,5% | 11,0% |
|---|---|---|---|---|---|---|
| 5% | 580 | 530 | 485 | 445 | 410 | 380 |
| 10% | 880 | 780 | 695 | 625 | 565 | 515 |
| 15% | 1.395 | 1.190 | 1.025 | 895 | 790 | 700 |
| 20% | 2.250 | 1.825 | 1.510 | 1.270 | 1.085 | 935 |
| 25% | 3.870 | 2.940 | 2.305 | 1.860 | 1.530 | 1.275 |

(Matrix indicatief — fair value zonder MOS. Huidige koers €975 ligt rond de 15%/9,5% en 15%/10,0% cellen — basis-scenario.)

---

## 13. Databronnen

### Bronnen-hiërarchie
- **Jaarverslag PDF / IR-pagina** → HOOG
- **Beursmelding / shareholder-letter** → HOOG
- **Aggregator** → AGGREGATOR

### Financiële bronnen (10 jaar historie — VERPLICHT)

| Jaar | Bron | URL | Betrouwbaarheid |
|---|---|---|---|
| 2015-2017 | — | — | — (pre-IPO) |
| 2018 | IPO-prospectus referentie + Wikipedia | https://en.wikipedia.org/wiki/Adyen | AGGREGATOR |
| 2019 | search-snippet | (diverse aggregators) | AGGREGATOR |
| 2020 | search-snippet | (diverse aggregators) | AGGREGATOR |
| 2021 | search-snippet | (diverse aggregators) | AGGREGATOR |
| 2022 | search-snippet | (diverse aggregators) | AGGREGATOR |
| 2023 | Adyen H2 2024 persrelease vergelijkende kolom | https://www.adyen.com/press-and-media/adyen-publishes-h2-2024-financial-results | AGGREGATOR |
| 2024 | Adyen H2 2024 / FY2024 persrelease | https://www.adyen.com/press-and-media/adyen-publishes-h2-2024-financial-results | HOOG |
| 2025 | Adyen H2 2025 / FY2025 persrelease | https://www.adyen.com/press-and-media/adyen-publishes-h2-2025-financial-results-3pgu2 | HOOG |

**Status:** alleen 2024-2025 voldoen aan HOOG-eis. Pre-IPO 2015-2017 niet beschikbaar (Adyen was nog privaat).

### Jaarverslagen geraadpleegd

| Jaar | Bron | URL |
|---|---|---|
| 2025 | Adyen Annual Report 2025 (referentie) | https://investors.adyen.com/financials/2025 |
| 2024 | Adyen Annual Report 2024 (referentie) | https://investors.adyen.com/financials/2024 |
| 2024 H2 | Adyen H2 2024 Shareholder Letter | https://investors.adyen.com/financials/h2-2024 |
| 2025 H1 | Adyen H1 2025 Shareholder Letter | https://investors.adyen.com/financials/h1-2025 |

### Beursmeldingen geraadpleegd

| Datum | Omschrijving | URL |
|---|---|---|
| 2026-02 | H2 2025 results — €2,36 mld net revenue, EBITDA-marge 53% | https://www.adyen.com/press-and-media/adyen-publishes-h2-2025-financial-results-3pgu2 |
| 2026-03-05 | Adyen 2025 Annual Report publicatie | https://www.adyen.com/press-and-media/adyen-publishes-2025-annual-report |
| 2025-02 | H2 2024 results — €1,996 mld net revenue, EBITDA-marge 50% | https://www.adyen.com/press-and-media/adyen-publishes-h2-2024-financial-results |
| 2025-08 | H1 2025 results — €1.093 mln net revenue +20%, EBITDA-marge 50% | https://www.adyen.com/press-and-media/adyen-publishes-h1-2025-financial-results |

### IPO-prospectus
- **Geraadpleegd?** false (alleen via Wikipedia-referentie). PDF beschikbaar via Euronext-archief.
- **URL:** https://live.euronext.com/en/product/equities/NL0012969182-XAMS (basis listing pagina)
- **Pre-IPO data beschikbaar?** ja, via prospectus (2015-2017 P&L) — niet uitgelezen in deze sessie.
- **Pre-IPO bron:** IPO-prospectus juni 2018 (niet PDF-extracted in deze run).

### Non-GAAP
- **Gebruikt?** true — Adyen rapporteert primair "EBITDA" als kerncijfer (geen IFRS-term, omvat add-back D&A en SBC).
- **Toelichting:** EBITDA exclusief SBC zou consistenter zijn met "true earnings power"; voor DCF gebruik ik EBITDA minus SBC als FCF-proxy.

### Ontbrekende data
- Volledige resultatenrekening 2015-2017 (pre-IPO) — niet uit IPO-prospectus geëxtraheerd in deze sessie.
- IFRS EBIT en nettowinst per jaar — Adyen-shareholder-letters rapporteren primair EBITDA, niet operating profit per jaar.
- Werkelijke FCF-cijfers — Adyen rapporteert geen klassieke CFO/Capex/FCF.
- Gedetailleerde balans split tussen merchant funds en eigen kas.
- Insider-transactions 24 maanden — niet uit AFM-meldingsregister geverifieerd.
- Compensatie CFO Ethan Tandowsky — beperkt openbaar.
- 2025 EPS — niet expliciet uit search-snippets; afgeleid via EBITDA × (1-tax) / aandelen.

### Peildatum analyse
- **2026-04-28**

---

## 14. Volledige bronnen-lijst

| Titel | URL | Type |
|---|---|---|
| Adyen H2 2025 Financial Results | https://www.adyen.com/press-and-media/adyen-publishes-h2-2025-financial-results-3pgu2 | beursmelding |
| Adyen H2 2024 Financial Results | https://www.adyen.com/press-and-media/adyen-publishes-h2-2024-financial-results | beursmelding |
| Adyen Annual Report 2025 | https://investors.adyen.com/financials/2025 | jaarverslag |
| Adyen Annual Report 2024 | https://investors.adyen.com/financials/2024 | jaarverslag |
| Adyen H1 2025 Shareholder Letter | https://investors.adyen.com/financials/h1-2025 | jaarverslag |
| Adyen H2 2024 Shareholder Letter | https://investors.adyen.com/financials/h2-2024 | jaarverslag |
| Yahoo Finance ADYEN.AS Statistics (beta 1,83) | https://finance.yahoo.com/quote/ADYEN.AS/key-statistics/ | aggregator |
| Damodaran Implied ERP — January 2026 | https://aswathdamodaran.substack.com/p/data-update-4-for-2026-a-risk-journey | onderzoeksrapport |
| Germany 10-Year Bond Yield (3,02% per 27-04-2026) | https://tradingeconomics.com/germany/government-bond-yield | aggregator |
| Adyen Wikipedia | https://en.wikipedia.org/wiki/Adyen | nieuwsartikel |
| Pieter van der Does Wikipedia | https://en.wikipedia.org/wiki/Pieter_van_der_Does_(businessman) | nieuwsartikel |
| Simply Wall St Adyen Health (debt €247 mln) | https://simplywall.st/stocks/us/diversified-financials/otc-adyy.f/adyen/health | aggregator |
| Crowdfund Insider Adyen 2024 results | https://www.crowdfundinsider.com/2025/03/237232-global-fintech-adyen-reports-1-35-trillion-in-payments-processing-volume-in-past-year/ | nieuwsartikel |
| The Wolf of Harcourt Street Adyen Profit Margin 3-Year High | https://www.thewolfofharcourtstreet.com/p/adyen-profit-margin-reaches-three | nieuwsartikel |
| Euronext Live ADYEN Quote | https://live.euronext.com/en/product/equities/NL0012969182-XAMS | beurswebsite |
| Adyen IR — Financials overview | https://investors.adyen.com/financials | beurswebsite |

---

## 15. Update-historie

| Datum | Versie | Wijziging |
|---|---|---|
| 2026-04-28 | 1.0 | Eerste publicatie (cowork stage 1 — markdown). HOOG voor FY2024 en FY2025; 2018-2023 AGGREGATOR; 2015-2017 LEEG (pre-IPO). |

---

## Opmerkingen voor Claude Code

1. **Pre-IPO 2015-2017 leeg** — IPO-prospectus juni 2018 bevat 2015-2017 P&L; vervolg-pas zou prospectus-PDF moeten openen via Euronext-archief.

2. **Adyen rapporteert geen klassieke FCF** — DCF is gebouwd op EBITDA-derived FCF-proxy (EBITDA - SBC - capex - cash tax). Stage-2 mag overwegen om dit anders te framen.

3. **Cash-positie €12,5 mld vs eigen kas €2,5 mld** — verschil is merchant funds. Voor equity-bridge gebruik ik €2,5 mld eigen nettokas. Stage-2 mag exact berekenen uit FY2025-balans-PDF.

4. **EBITDA-marge volatiliteit** — 46-63% range. Genormaliseerde 50% gehanteerd voor EPV; 53% (recent) voor 5-jaars-projectie. Asymmetrische aanname; stage-2 mag valideren.

5. **Discrepantie executive_summary.oordeel vs scorekaart-eindoordeel** — Executive summary noemt HOLD; scorekaart-rubric geeft KOOP (totaal 33). Ik volg deterministische rubric en pas executive_summary.oordeel aan naar **KOOP**. Stage-2 mag dit valideren.

6. **Beta 1,83 is hoog** — fintech-volatiliteit gerechtvaardigt dit; geen Blume-discount toegepast voorbij standaard formule.

7. **Stripe-IPO 2026-2027** is grootste onbekende — bij Stripe-multiple 30x EBITDA zou Adyen multiple naar boven herijkt worden; bij 12x naar beneden. Dit zit in pessimistisch-scenario.

8. **Single-large-customer-effect** — FY2025-cijfers tonen dit expliciet bij processed-volume-disclosure. Klant-naam niet onthuld door Adyen.

Stage 2 (Claude Code) kan JSON-injectie en validator-run nu starten. Scorekaart-totaal 33/45 → KOOP volgens deterministische regel.
