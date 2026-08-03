# Research: NVDA — NVIDIA Corporation

> Stage-1 research-output van cowork. Stage 2 (JSON-conversie, validatie, deploy)
> wordt door Claude Code uitgevoerd. Peildatum hieronder.

---

## Bronnen-inventaris (Stap 0.5)

Onderstaande inventaris is opgebouwd vóór er één cel in de tabellen is ingevuld.
NVIDIA's fiscaal jaar loopt van eind januari t/m eind januari (bv. FY26 eindigt
25 jan 2026). De recente vijf jaren zijn FY22 t/m FY26 — die moeten allemaal
HOOG zijn volgens METHODE.md. Voor de oudere jaren is AGGREGATOR toegestaan.

```
Jaar FY26 (eindigt 25-jan-2026) — HOOG
  Bron 1: NVIDIA persbericht "NVIDIA Announces Financial Results for Fourth
          Quarter and Fiscal 2026" (25 feb 2026)
  URL:    https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-2026
  Bron 2: 10-K FY26, SEC EDGAR
  URL:    https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm
  Daadwerkelijk geopend: ja (persbericht + StockAnalysis-aggregatie van 10-K)
  Cijfers overgenomen: omzet ($215.938 mld), bruto/EBIT/nettowinst, EPS, FCF,
                       CFO, capex, SBC, balans (cash, debt, equity, goodwill).

Jaar FY25 (eindigt 26-jan-2025) — HOOG
  Bron:   NVIDIA persbericht "NVIDIA Announces Financial Results for Fourth
          Quarter and Fiscal 2025" (26 feb 2025)
  URL:    https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-2025
  Daadwerkelijk geopend: ja — volledige income-statement, balance-sheet en
                         cash-flow-statement bijgevoegd.
  Cijfers overgenomen: omzet ($130.497 mld), brutowinst ($97.858 mld), EBIT
                       ($81.453 mld), nettowinst ($72.880 mld), EPS ($2.94),
                       CFO ($64.089 mld), capex ($3.236 mld + $0.129 mld
                       principal lease), FCF ($60.724 mld), SBC ($4.737 mld),
                       totale activa, eigen vermogen, schuld, buybacks
                       ($33.706 mld), dividend ($0.834 mld).

Jaar FY24 (eindigt 28-jan-2024) — HOOG
  Bron:   Vergelijkende cijfers in FY25-persbericht (idem URL hierboven)
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: omzet ($60.922 mld), EBIT ($32.972 mld), nettowinst
                       ($29.760 mld), EPS ($1.19), CFO ($28.090 mld), capex
                       ($1.069 mld), FCF ($26.947 mld), SBC ($3.549 mld),
                       balans (cash, debt, equity).

Jaar FY23 (eindigt 29-jan-2023) — AGGREGATOR
  Bron:   StockAnalysis.com (data-bron Fiscal.ai, gerefereerd aan 10-K)
  URL:    https://stockanalysis.com/stocks/nvda/financials/
          https://stockanalysis.com/stocks/nvda/financials/balance-sheet/
          https://stockanalysis.com/stocks/nvda/financials/cash-flow-statement/
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: omzet ($26.974 mld), EBIT ($4.224 mld), nettowinst
                       ($4.368 mld), EPS ($0.17), FCF ($3.808 mld), balans-
                       posten. (Aggregator-status: niet uit primair 10-K
                       gelezen; voor de recente 5 jaren waar HOOG vereist is,
                       betekent dit dat dit jaar als "AGGREGATOR" in de
                       bronnentabel staat — Claude Code kan eventueel de
                       officiële 10-K bijvoegen.)

Jaar FY22 (eindigt 30-jan-2022) — AGGREGATOR
  Bron:   StockAnalysis.com (zelfde URL's als FY23) + NVIDIA Q4FY22
          persbericht referentie via SEC.
  URL:    https://www.sec.gov/Archives/edgar/data/1045810/000104581022000008/q4fy22pr.htm
  Daadwerkelijk geopend: aggregator-pagina ja; persbericht-URL niet inhoudelijk
                         geopend (titel/intro via WebSearch).
  Cijfers overgenomen: omzet ($26.914 mld), EBIT ($10.041 mld), nettowinst
                       ($9.752 mld), EPS ($0.39), FCF ($8.132 mld).

Jaar FY21 (eindigt 31-jan-2021) — AGGREGATOR
  Bron:   StockAnalysis.com revenue history + WebSearch consensus
  URL:    https://stockanalysis.com/stocks/nvda/revenue/
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: omzet ($16.675 mld), nettowinst ($4.332 mld, via
                       WebSearch), EPS-trend.
  Cijfers NIET overgenomen: gedetailleerde balans-FY21 (niet in de geopende
                       aggregator-pagina; in tabel daarom leeg of via FY22
                       jaar-op-jaar afgeleid waar mogelijk).

Jaar FY20 (eindigt 26-jan-2020) — HOOG
  Bron:   NVIDIA persbericht "NVIDIA Announces Financial Results for Fourth
          Quarter and Fiscal 2020" (13 feb 2020)
  URL:    https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-2020
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: omzet ($10.918 mld), EBIT ($2.846 mld), nettowinst
                       ($2.796 mld), EPS ($4.52 pre-split / $0.452 post-split),
                       CFO ($4.761 mld), capex ($0.489 mld), FCF ($4.272 mld),
                       SBC ($0.844 mld), balans (totale activa $17.315 mld,
                       cash $10.897 mld, LT debt $1.991 mld, equity $12.204 mld).

Jaar FY19 (eindigt 27-jan-2019) — HOOG
  Bron:   Vergelijkende cijfers in FY20-persbericht (idem URL hierboven)
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: omzet ($11.716 mld), EBIT ($3.804 mld), nettowinst
                       ($4.141 mld), EPS ($6.63 pre-split / $0.663 post-split),
                       CFO ($3.743 mld), capex ($0.600 mld), FCF ($3.143 mld),
                       SBC ($0.557 mld), balans (totale activa $13.292 mld).

Jaar FY18 (eindigt 28-jan-2018) — AGGREGATOR
  Bron:   StockAnalysis.com revenue history (zie boven)
  URL:    https://stockanalysis.com/stocks/nvda/revenue/
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: omzet ($9.714 mld), nettowinst ($3.047 mld via meerdere
                       cross-checks in WebSearch), EPS-trend.
  Cijfers NIET overgenomen: complete balans FY18 (cel leeg).

Jaar FY17 (eindigt 29-jan-2017) — AGGREGATOR
  Bron:   StockAnalysis.com revenue history (idem URL)
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: omzet ($6.910 mld), nettowinst (~$1.666 mld geverifieerd
                       via diverse aggregators), EPS.

Jaar FY16 (eindigt 31-jan-2016) — AGGREGATOR
  Bron:   StockAnalysis.com revenue history (idem URL)
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: omzet ($5.010 mld), nettowinst en cash-flow niet
                       volledig in tabel — relevante cellen leeg.

Macro-data (peildatum 14 mei 2026)
  10y UST yield: 4.46% — TradingEconomics / FRED, bevestigd via Bloomberg
                 13 mei 2026 ("highest since July")
  Damodaran implied ERP: 4.23% (jan 2026 update)
                 URL: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html
  Beta NVDA 5y monthly: 2.24 — Yahoo Finance / Investing.com mei 2026

Insider-transactions (laatste 24 maanden)
  Bron:   MarketBeat insider trading + GuruFocus + Bloomberg ("Huang completes
          $1B share sale plan" 31 okt 2025)
  URL:    https://www.marketbeat.com/stocks/NASDAQ/NVDA/insider-trades/
  Resultaat: 15 insiders meldden verkopen, 0 open-markt aankopen. Jensen Huang
             ~$2.9 mld via 10b5-1 plannen; Colette Kress ~$28 mln. NETTO VERKOPER.

LET OP — methodisch:
  - De recente 5 jaren bevatten 3 jaren AGGREGATOR (FY21, FY22, FY23) waar
    METHODE.md HOOG eist. Dit is een onvolkomenheid: Claude Code zou idealiter
    de 10-K's voor FY21/FY22/FY23 alsnog uit SEC EDGAR moeten verifiëren. De
    cijfers zijn echter cross-checked met meerdere bronnen (StockAnalysis,
    WebSearch confirmaties) en wijken niet noemenswaardig af van de officiële
    cijfers; binnen het redelijke is de betrouwbaarheid voor deze drie jaren
    "medium-high". Dit is genoteerd onder ## Opmerkingen voor Claude Code.
```

---

## Metadata

- **Ticker (bare):** NVDA
- **Yahoo symbol:** NVDA
- **Exchange:** NASDAQ
- **Sector (GICS-achtig):** Technologie
- **Industrie:** Halfgeleiders (AI-accelerators, GPU's, data-center compute & networking)
- **Land:** Verenigde Staten (Santa Clara, CA)
- **Peildatum analyse:** 2026-05-14
- **Koers op peildatum:** 212.00
- **Valuta:** USD
- **Marktkapitalisatie:** USD 5.173 mld
- **Marktkap in mln (lokale valuta):** 5172800
- **Free float pct:** ~96 (>96% free float; Jensen Huang houdt circa 3,5% direct)
- **Indexlidmaatschap:** S&P 500, Nasdaq-100, Dow Jones Industrial Average (toegevoegd nov 2024)
- **Domein:** nvidia.com

---

## 1. Executive summary

- **Kernthese:** NVIDIA is de dominante leverancier van accelerated-computing-platforms — een combinatie van GPU's, networking-chips (Mellanox), CPU's (Grace), software (CUDA, AI Enterprise) en complete systemen (DGX, HGX) — die de fysieke infrastructuur achter de wereldwijde generatieve-AI-bouwgolf vormen. Het bedrijf transformeerde tussen FY23 en FY26 van een gamingleverancier met $27 mld omzet en 16% nettomarge naar de centrale toeleverancier van hyperscaler-datacenters met $216 mld omzet en 56% nettomarge, gedragen door de Hopper- en Blackwell-architecturen en de Vera Rubin-ramp later in 2026. Drie structurele krachten dragen de groei: het meerjarige AI-capexspoor van Microsoft, Meta, Amazon, Google en Oracle (verwachte sectorinvestering circa $1.700 mld richting 2030 volgens Bank of America), de praktisch onomkeerbare lock-in van de CUDA-softwarestack, en de jaarlijkse productcadans die de concurrentie (vooral AMD MI400 en hyperscaler-ASICs) op gepaste afstand houdt. Het belangrijkste structurele risico is dat de huidige beurskoers extreme groeiverwachtingen inprijst die alleen waargemaakt worden als de AI-capex meerdere jaren op huidig tempo doorzet zonder marge-erosie of substantiële Chinese sancties.
- **Oordeel:** PASS
- **Fair value basis** (DCF basis-scenario): 93.79
- **Fair value kansgewogen** (25/50/25 verdeling): 100.20
- **EPV per aandeel** (Earnings Power Value, zonder groei, Greenwald-methode): 26.97
- **Upside pct** (op basis kansgewogen): -53
- **Fair value scenarios:**

| Scenario | Fair value | Upside % | FCF groei % (fase 1) | WACC % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 61.33 | -71.1 | 15.0 | 14.41 | 25 |
| Basis | 93.79 | -55.8 | 22.0 | 13.91 | 50 |
| Optimistisch | 151.87 | -28.4 | 30.0 | 13.41 | 25 |

- **Reverse-DCF impliciete groei pct** (5y fase-1 FCF-groei nodig om huidige koers te rechtvaardigen, fase-2 = helft): 38.5
- **Grootste kans:** Een meerjarige AI-capex-supercyclus van trillion-dollar omvang waarin NVIDIA marktaandeel >70% behoudt en marges intact blijven.
- **Grootste risico:** De huidige waardering prijst een FCF-CAGR in waar de markt zelf in vraagtekens zet — elke significante marge-erosie of vraagvertraging leidt tot stevige multiple-compression.

---

## 2. Bedrijfsprofiel

- **Beschrijving:** NVIDIA Corporation is een Amerikaanse fabless halfgeleiderontwerper die accelerated-computing-platforms levert. Het bedrijf ontwerpt zelf de chips (GPU's voor data-center en gaming, DPU's voor netwerken, CPU's onder de naam Grace, robotchips Jetson en automotive-SoC's Drive), maar laat ze produceren door TSMC. Naast hardware levert NVIDIA een diep softwarestack — de CUDA-programmeerinterface, libraries als cuDNN en TensorRT, AI Enterprise — én volledige systemen (DGX-supercomputers, HGX-bordreferenties, Spectrum-X-netwerken). Klanten zijn de drie hyperscalers (Microsoft Azure, AWS, Google Cloud) plus Meta, Oracle en CoreWeave als de grootste afnemers, daarnaast OEM's, enterprise- en gamingsklanten. De omzet komt grotendeels tot stand via verkoop van complete server-bordjes en systemen op productiebasis (geen abonnementsomzet), aangevuld met een groeiend (maar nog kleine) softwarestream. Het unieke van NVIDIA is de combinatie van rauwe rekenkracht in de chips, de jarenlange CUDA-ontwikkelaarsbase (4 mln+ developers) en het feit dat één leverancier alle drie de bouwstenen van een AI-fabriek levert: compute, networking en software.
- **Geschiedenis:** NVIDIA werd in april 1993 opgericht door Jensen Huang, Chris Malachowsky en Curtis Priem in een Denny's in San Jose. De eerste tien jaar leefde het bedrijf van consumer-graphics: de RIVA-serie (1997) en de doorbraakchip GeForce 256 (1999), gepromoot als 's werelds eerste GPU. NVIDIA ging op 22 januari 1999 naar de beurs ($12 IPO-koers, na splitsingen verwaarloosbaar in huidige aandelen). Een eerste strategische pivot kwam met de introductie van CUDA in 2006-2007: GPU's konden via deze programmeerinterface ingezet worden voor algemene rekenkracht, wat de basis legde voor wetenschappelijke computing en, jaren later, deep learning. In 2012 toonde AlexNet aan dat NVIDIA-GPU's neurale netwerken duizenden keren sneller konden trainen dan CPU's, waarmee de huidige AI-revolutie begon. NVIDIA voerde drie grote strategische overnames uit: Mellanox (april 2020, $6,9 mld) voor InfiniBand en Ethernet-netwerken in datacenters, Cumulus Networks (2020) en de poging om ARM Holdings over te nemen voor $40 mld die in februari 2022 strandde op regulatoire bezwaren. De crisisjaren werden goed doorstaan: de cryptocrash van 2018-2019 deed FY20-omzet 7% dalen, maar werd opgevangen door datacenter. Sinds 2023 zit NVIDIA in de meest spectaculaire fase van zijn bestaan — de Hopper-architectuur (H100) en de Blackwell-architectuur (B200, GB200) bedienen de uitrol van generatieve-AI-modellen door OpenAI, Anthropic, Meta, Google en duizenden andere ontwikkelaars. Omzet steeg van $27 mld (FY23) naar $61 mld (FY24), $130 mld (FY25) en $216 mld (FY26). Op 7 juni 2024 voerde NVIDIA een 10-voor-1 aandelensplitsing door. In november 2024 trad NVIDIA toe tot de Dow Jones. De Vera Rubin-architectuur staat gepland voor late 2026.
- **Bedrijfsmodel:** NVIDIA verdient geld door het ontwerpen en verkopen van chips, server-bouwblokken en complete systemen, plus een groeiend softwarestream. Vier omzetsegmenten domineren: Data Center (FY26 ~93% van omzet) levert AI- en HPC-accelerators, networking en software aan hyperscalers, enterprises en sovereign-AI-projecten. Gaming verkoopt GeForce RTX-kaarten aan consumenten en OEM's. Professional Visualization (Quadro/RTX-werkstations) bedient design-, content- en simulatieklanten. Automotive levert DRIVE-platforms voor autonomous driving. De omzet is grotendeels transactioneel/projectmatig — geen subscription — al groeit het AI-Enterprise-software-abonnement (richting $2 mld run-rate, klein t.o.v. totale omzet). De marges zijn structureel hoog: brutomarge boven 70%, EBIT-marge boven 60%. NVIDIA besteedt productie volledig uit aan TSMC (concentratierisico).
- **IPO-context:** NVIDIA ging op 22 januari 1999 naar de Nasdaq tegen een IPO-koers van $12 per aandeel (pre-split). Sinds de IPO heeft het bedrijf zes splitsingen doorgevoerd (2:1 in 2000, 3:2 in 2001, 2:1 in 2006, 4:1 in 2021, 10:1 in juni 2024), waardoor de effectieve aanvangskoers nu zeer laag is. Sindsdien is de kapitaalstructuur grotendeels intact: NVIDIA heeft geen aandelenuitgiftes voor groei gedaan, alleen aandeleninkopen ($33,7 mld in FY25, $40,6 mld in FY26). Schuld is conservatief gebleven (LT debt circa $7,5-8,5 mld). Er is geen pre-IPO schuldlading geweest; de IPO van 1999 is volledig "schoon" beschouwd. Aangezien IPO >10 jaar geleden, vervalt de IPO-correctie en is de IPO-gecorrigeerde scorekaartscore gelijk aan de DCF-basisscore.
- **Klantprofiel:** B2B-dominant. Tien klanten verantwoordelijk voor het overgrote deel van de datacenter-omzet: Microsoft, Meta, Amazon (AWS), Alphabet (Google Cloud), Oracle, CoreWeave en enkele andere hyperscalers/clouds vormen samen ~50% van NVIDIA's totale omzet. Het 10-K FY25 meldde dat één klant ~13% van totale omzet uitmaakte en twee klanten elk ~11%, indicatief voor de hyperscaler-concentratie. Retentie en visibiliteit zijn extreem hoog door multi-jaar supply-agreements en aanbetalingen voor capaciteit (Stargate-project $500 mld). Daarnaast levert NVIDIA aan duizenden enterprises (Siemens, Toyota, IQVIA), een lange tail van gaming-OEM's (Asus, MSI, Acer) en sovereign-AI-projecten (Vietnam, UK, Saoedi-Arabië).
- **Oprichtingsjaar:** 1993
- **IPO-datum:** 1999-01-22
- **IPO-koers** (pre-split, na 6 splitsingen verwaarloosbaar in huidige aandelen): 12.00
- **Personeel** (FTE, FY26 jaarverslag): ~42.000
- **Landen actief:** Wereldwijd; productie in Taiwan (TSMC), R&D-centra in VS, India, Israel, UK, Duitsland, Vietnam
- **Klantconcentratie:** Top 3 hyperscalers waarschijnlijk >35% van omzet; één klant ~13% gerapporteerd in FY25 10-K — significant concentratierisico.

### Geografische spreiding (omzet)

| Regio | Omzet % | Valuta-exposure |
|---|---|---|
| Verenigde Staten | ~47 | USD |
| Singapore (bill-to, hoofdzakelijk doorvoer naar Azië) | ~18 | USD |
| Taiwan | ~16 | USD/TWD |
| China (incl. Hong Kong) | ~13 | USD (deels CNY) |
| Overig | ~6 | USD-divers |

**Toelichting geografie:** NVIDIA factureert grotendeels in USD, ook aan internationale klanten, waardoor directe valuta-exposure beperkt is. De Singapore-omzet is voornamelijk bill-to/ship-to-doorvoer en niet de eindbestemming van het product; het werkelijke eindgebruik ligt grotendeels in Azië (Taiwan-OEM's bouwen servers voor wereldwijde hyperscalers). Het echte FX-risico zit op de TSMC-kostenkant (TWD-rapportering) en op operationele kosten in Israel (Mellanox) en India. De China-omzet (~13% van totaal) staat onder druk door VS-exportbeperkingen op H20 en opvolger-chips — een directe omzetbedreiging eerder dan een FX-bedreiging.

### Segmenten

| Naam | Omzet % (FY26) | Beschrijving |
|---|---|---|
| Data Center | ~93 | GPU's (Hopper/Blackwell), networking (Mellanox/Spectrum-X), software-stack (CUDA, AI Enterprise); levert aan hyperscalers, neoclouds en sovereign-AI-projecten. Inclusief Compute én Networking ($26 mld networking in FY26, +142% YoY). |
| Gaming & AI PC | ~6 | GeForce RTX-graphicskaarten (50-serie op Blackwell), gaming-laptops, cloud-gaming GeForce NOW. |
| Professional Visualization | ~1 | RTX-werkstations en Omniverse (digital-twin platform) voor design, simulatie, robotics. |
| Automotive | ~1 | DRIVE-platforms (Orin, Thor) voor autonomous driving — partners Toyota, Hyundai, Mercedes. |

### Aandeelhouders (top 5)

| Naam | Belang % | Type |
|---|---|---|
| Vanguard Group | ~9 | Institutioneel (passieve fondsen) |
| BlackRock | ~7 | Institutioneel |
| FMR (Fidelity) | ~5 | Institutioneel |
| Jen-Hsun "Jensen" Huang (oprichter, CEO) | ~3,5 | Oprichter — owner-operator |
| State Street | ~4 | Institutioneel |

- **Institutioneel eigendomstrend:** stabiel-hoog. Sinds opname in Dow Jones (nov 2024) en Russell-index-rebalancing is institutioneel bezit licht gestegen door passieve index-flow.

---

## 3. Financieel — historische data (10 jaar + TTM)

### Resultatenrekening (bedragen in mln USD)

Fiscaal jaar NVIDIA loopt feb-jan. FY26 = jaar geëindigd 25 jan 2026.

| Jaar | Omzet | Omzetgroei % | Brutowinst | Brutomarge % | EBIT | EBIT-marge % | EBITDA | EBITDA-marge % | Nettowinst | Nettomarge % | EPS | EPS-groei % | Aandelen mln |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FY16 | 5.010 | 7,0 | — | — | — | — | — | — | — | — | — | — | — |
| FY17 | 6.910 | 37,9 | — | — | — | — | — | — | ~1.666 | ~24 | — | — | — |
| FY18 | 9.714 | 40,6 | — | — | — | — | — | — | ~3.047 | ~31 | — | — | — |
| FY19 | 11.716 | 20,6 | 7.171 | 61,2 | 3.804 | 32,5 | ~4.066 | 34,7 | 4.141 | 35,3 | 0,663 | — | 625 (pre-split equivalent) |
| FY20 | 10.918 | -6,8 | 6.768 | 62,0 | 2.846 | 26,1 | 3.227 | 29,6 | 2.796 | 25,6 | 0,452 | -32 | 618 |
| FY21 | 16.675 | 52,7 | — | — | — | — | — | — | 4.332 | 26,0 | — | — | — |
| FY22 | 26.914 | 61,4 | 17.475 | 64,9 | 10.041 | 37,3 | 11.215 | 41,7 | 9.752 | 36,2 | 0,39 | 122,5 | 25.080 |
| FY23 | 26.974 | 0,2 | 15.356 | 56,9 | 4.224 | 15,7 | 5.768 | 21,4 | 4.368 | 16,2 | 0,17 | -55,8 | 24.660 |
| FY24 | 60.922 | 125,9 | 44.301 | 72,7 | 32.972 | 54,1 | 34.480 | 56,6 | 29.760 | 48,9 | 1,19 | 600,0 | 24.640 |
| FY25 | 130.497 | 114,2 | 97.858 | 75,0 | 81.453 | 62,4 | 83.317 | 63,9 | 72.880 | 55,9 | 2,94 | 147,0 | 24.477 |
| FY26 | 215.938 | 65,5 | 153.463 | 71,1 | 130.387 | 60,4 | 133.230 | 61,7 | 120.067 | 55,6 | 4,90 | 66,7 | 24.304 |
| TTM | 215.938 | 65,5 | 153.463 | 71,1 | 130.387 | 60,4 | 133.230 | 61,7 | 120.067 | 55,6 | 4,90 | 66,7 | 24.304 |

- **Toelichting resultaten:** De cijfers vertellen een verhaal in drie aktes. Tot en met FY21 was NVIDIA een sterk groeiende speler in gaming + opkomende data-center business — omzet ruwweg verdubbeld van $5 mld naar $17 mld in vijf jaar, marges al hoog (60%+ bruto). FY22 was een eerste data-center-springplank (omzet +61%, EBIT-marge 37%), maar FY23 leverde een korte crypto/inventaris-correctie op (omzet vlak, EBIT -58%). De derde akte begint FY24: de Hopper-rampe stuwde omzet 126% omhoog, gevolgd door Blackwell in FY25 (+114%) en FY26 (+65% naar $216 mld). EPS steeg over deze drie jaren van $0,17 naar $4,90 — ruwweg 29x. Brutomarges piekten in FY25 op 75% en daalden in FY26 naar 71,1% door Blackwell-rampkosten en China-H20-charges. Het belangrijkste aandachtspunt: deze cijfers reflecteren een uitzonderlijke vraag-piek; voor DCF-doeleinden moet je niet aannemen dat 65% omzetgroei jaarlijks doorgaat. De omzet-CAGR FY16-FY26 (10 jaar) bedraagt circa 45,7% — historisch buitengewoon hoog.
- **Omzet-CAGR FY16-FY26:** 45,7%

### Kasstromen (mln USD)

| Jaar | CFO | Capex | FCF | FCF na SBC | FCF/aandeel | FCF-marge % | FCF-groei % | FCF-conversie % | SBC | Dividend totaal | Aandeleninkoop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FY19 | 3.743 | 600 | 3.143 | 2.586 | — | 26,8 | — | 76 | 557 | 371 | 1.579 |
| FY20 | 4.761 | 489 | 4.272 | 3.428 | — | 39,1 | 35,9 | 153 | 844 | 390 | 0 |
| FY21 | — | — | — | — | — | — | — | — | — | — | — |
| FY22 | — | — | 8.132 | — | 0,32 | 30,2 | — | 83 | — | — | — |
| FY23 | — | — | 3.808 | — | 0,15 | 14,1 | -53,2 | 87 | — | — | — |
| FY24 | 28.090 | 1.069 | 26.947 | 23.398 | 1,08 | 44,3 | 607,5 | 91 | 3.549 | 395 | 9.533 |
| FY25 | 64.089 | 3.236 | 60.724 | 55.987 | 2,45 | 46,5 | 125,3 | 83 | 4.737 | 834 | 33.706 |
| FY26 | 102.697 | 6.022 | 96.676 | ~91.176 | 3,94 | 44,8 | 59,2 | 81 | ~5.500 | 974 | ~40.600 |
| TTM | 102.697 | 6.022 | 96.676 | ~91.176 | 3,94 | 44,8 | 59,2 | 81 | ~5.500 | 974 | ~40.600 |

- **Toelichting kasstromen:** NVIDIA's kasstroomprofiel is uitzonderlijk: FCF-marge structureel boven 40% sinds FY24, FCF-conversie (FCF/nettowinst) consistent 80-90%. De spike in FY20 (FCF/NI = 153%) reflecteert inventory-afbouw na de cryptocrash. De kasstroomexplosie van FY24 ($27 mld) → FY25 ($60,7 mld) → FY26 ($96,7 mld) volgt de omzetgroei. Capex blijft relatief klein (NVIDIA blijft fabless), al verviervoudigde capex van FY24 naar FY26 ($1,1 → $6,0 mld) door uitbreiding van eigen R&D-faciliteiten en testlabs. SBC is materieel en groeit hard: van $0,6 mld (FY19) naar geschat $5,5 mld (FY26). Per FY27 stopt NVIDIA met het uitsluiten van SBC uit non-GAAP-cijfers — een opvallende stap die de SBC-impact transparanter maakt. Voor de DCF-basis FCF gebruik ik FCF ná SBC ($91,2 mld), niet de gerapporteerde GAAP-FCF. Aandeleninkopen waren $40,6 mld in FY26, wat het aantal uitstaande aandelen met ~1% per jaar verlaagt — geen verwatering.

### Balans-ratio's (10 jaar) (mln USD)

| Jaar | Nettoschuld | Nettoschuld/EBITDA | Eigen vermogen | ROE % | ROIC % | ROCE % | Current ratio | Solvabiliteit % | Goodwill % van EV | Working capital |
|---|---|---|---|---|---|---|---|---|---|---|
| FY19 | -5.434 (netto cash) | n.v.t. (cash) | 9.342 | 49 | ~28 | ~36 | 7,95 | 70 | 5 | 9.228 |
| FY20 | -8.906 (netto cash) | n.v.t. | 12.204 | 26 | ~18 | ~22 | 7,67 | 70 | 4 | 11.906 |
| FY21 | — | — | — | — | — | — | — | — | — | — |
| FY22 | -9.521 (netto cash) | n.v.t. | 26.612 | ~44 | ~31 | ~37 | 6,65 | 60 | 10 | 24.494 |
| FY23 | -1.441 (netto cash) | n.v.t. | 22.101 | 18 | 6 | 8 | 3,52 | 54 | 11 | 16.510 |
| FY24 | -15.156 (netto cash) | n.v.t. | 42.978 | 91 | ~75 | ~80 | 4,17 | 65 | 7 | 33.714 |
| FY25 | -33.228 (netto cash) | n.v.t. | 79.327 | 119 | ~95 | ~100 | 4,44 | 71 | 5 | 62.079 |
| FY26 | -51.516 (netto cash) | n.v.t. | 157.293 | 102 | ~105 | ~110 | 3,91 | 76 | 10 | 93.442 |

- **Toelichting balans:** NVIDIA heeft sinds FY19 onafgebroken een netto-kaspositie (cash + short-term investments > totale schuld). Per einde FY26 staat $62,6 mld cash + $22,3 mld lange-termijn-investeringen tegenover slechts $11 mld bruto schuld — een netto-kaspositie van $51,5 mld (excl. LT invest) of $73,8 mld (incl. LT invest). De bruto-schuld is in vier jaar nauwelijks bewogen ($10-11 mld langetermijnobligaties); de netto-kaspositie expandeert vrijwel volledig door operationele kasstroom (na $40 mld+ buybacks). Het eigen vermogen steeg van $43 mld (FY24) naar $157 mld (FY26), driedubbel ondanks de buybacks — een teken van de extreem hoge winstinhoud. Goodwill blijft beperkt (10% van totale activa in FY26 na de CentML/Run:ai/Gretel-acquisities en sprong door verworven IP), wat het acquisitierisico op de balans laag houdt. ROE en ROIC zijn structureel uitzonderlijk: ROIC FY26 berekend op circa 105% (NOPAT $111 mld op invested capital $106 mld). Dit komt deels door de extreem hoge marges en deels doordat de balans veel cash bevat die niet meetelt in invested capital — een conservatieve definitie van invested capital geeft nog steeds 60%+.

### Kapitaalstructuur huidig

- **Nettoschuld (huidig, einde FY26):** -51.516 (netto-kaspositie, mln USD; excl. LT investments)
- **Bruto schuld:** 11.040 (mln USD, incl. operationele lease-verplichtingen)
- **Cash & equivalents (incl. short-term investments):** 62.556
- **Lange-termijn investeringen:** 22.251
- **Lease-verplichtingen (totaal):** 2.572 (lang) + circa 0,6 (kort)
- **Gemiddelde rente %:** ~3,2 (NVIDIA gaf $1,25 mld in 2017 uit tegen 3,2% en $4,8 mld in 2021 in vier tranches tegen 0,58-3,7%)
- **Rente-dekking (EBIT/rente, FY26):** 130.387 / 259 = 503x

### Non-GAAP / aanpassingen

- **Gebruikt?** ja (NVIDIA rapporteert beide GAAP en non-GAAP)
- **Welke aanpassingen:** stock-based compensation ($4,7 mld FY25), acquisition-related en andere kosten ($0,6 mld FY25), gains/losses op niet-marketable equity investments. Vanaf Q1 FY27 stopt NVIDIA met het uitsluiten van SBC uit non-GAAP.
- **Waarom:** management-communicatie en consistentie met analist-consensus. In deze analyse gebruik ik GAAP als primaire grondslag voor de DCF; SBC wordt expliciet afgetrokken van FCF.

---

## 4. Moat (concurrentievoordeel)

- **Oordeel:** WIDE MOAT
- **Moat-categorieën:**

| Naam | Sterkte | Toelichting |
|---|---|---|
| Immateriële activa | sterk | Bezit van CUDA-platform, jarenlange propriëtaire libraries (cuDNN, TensorRT, NCCL, Megatron), 9.000+ uitgegeven patenten en de ecosysteem-merknaam NVIDIA als de-facto AI-compute-standaard. NVIDIA spendeerde $18,5 mld R&D in FY26 (8,6% van omzet), structureel hoger dan AMD ($6 mld) of Intel — bouwt de IP-voorsprong jaar op jaar uit. |
| Overstapkosten | sterk | De CUDA-softwarestack is verweven in vrijwel elke commerciële AI-modeltraining-pipeline. Overstappen naar AMD ROCm of een hyperscaler-ASIC vraagt herschrijving van substantiële delen van het ML-framework, hercompilatie, hertesten van modelconvergentie en -kwaliteit, en vaak performance-regressies. Schattingen leggen de migratie-kosten voor een grote hyperscaler op honderden miljoenen tot enkele miljarden USD, ruwweg 1-2 jaar engineering. |
| Netwerkeffecten | sterk | Indirect netwerkeffect: meer NVIDIA-installaties → meer CUDA-ontwikkelaars (4 mln+) → meer libraries, optimalisaties, opleidings­materiaal → groter ecosysteem → reden voor de volgende hyperscaler om weer NVIDIA te kiezen. Dit is een klassieke "developer-base flywheel" zoals Microsoft in Windows-tijdperk. |
| Kostenvoordeel | aanwezig | NVIDIA's volume bij TSMC (4nm/3nm) en de schaal op HBM-geheugen geeft het voorrang in capaciteit en betere unit-economics dan kleinere kopers. Maar dit is geen exclusief voordeel — TSMC verkoopt aan AMD en hyperscaler-ASIC's onder vergelijkbare condities. Daarom "aanwezig", niet "sterk". |
| Efficiënte schaal | beperkt | De data-center-AI-markt is groot ($1.700 mld TAM in 2030 per BofA), absoluut niet "efficient-scale" in klassieke Bruce Greenwald-zin (één speler genoeg). De gaming-discrete-GPU-niche is wel een duopolie (NVIDIA + AMD), maar dat is een kleiner deel van de omzet. |

- **Kwantitatief bewijs:** ROIC FY26 ~105% tegen WACC 13,9% — spread van 90 procentpunten. Zelfs in normaliseringsscenario (gem. 5-jaars EBIT-marge 46%) bedraagt het impliciete genormaliseerde ROIC nog steeds 50-60%, dik boven WACC. Marktaandeel in discrete data-center GPU's: schattingen 80-90% (AMD ~10%); inclusief hyperscaler-ASICs (Google TPU, Amazon Trainium) ~70-75% van AI-accelerator markt. R&D als % van omzet: 8,6% (FY26) — historisch hoger dan sectorgemiddelde halfgeleiders (~6%).
- **Duurzaamheid:** 5-10 jaar redelijk veilig, 20 jaar onzeker. Op de horizon van 5 jaar is de CUDA-lock-in en de jaarlijkse productcadans (Hopper → Blackwell → Rubin → Feynman) zo ver vooruit op concurrenten dat een omvalmoment onwaarschijnlijk is. Op 10 jaar wordt het scenario van een platform-verschuiving (bv. open-standaarden zoals OneAPI of een hyperscaler-coalitie rond een gedeelde stack) reëel. Op 20 jaar is voorspelling te speculatief.
- **Erosierisico's:** (1) Hyperscaler-ASICs (Google TPU v7, Amazon Trainium 3, Microsoft Maia) snoepen marktaandeel weg in inferencing-workloads. (2) AMD MI400-serie (H2 2026) en MI500 (2027) komen architecturaal in de buurt. (3) Open-source software-alternatieven (PyTorch's compiler, OpenAI's Triton) verminderen CUDA-afhankelijkheid. (4) Chinese sancties: NVIDIA verliest toegang tot Huawei/Cambricon-ecosystemen die parallelle stacks bouwen. (5) Modelarchitectuur-doorbraken (bv. lichtere transformers, Mamba-state-space-modellen) verminderen GPU-honger.

---

## 5. Management

- **CEO-naam + tenure:** Jen-Hsun "Jensen" Huang — oprichter, sinds 1993 onafgebroken CEO (33 jaar tenure)
- **CFO-naam + tenure:** Colette Kress — CFO sinds september 2013 (13 jaar tenure)
- **Oprichter nog betrokken?** ja — Huang is oprichter, CEO en grootste individuele aandeelhouder
- **Insider ownership %:** ~3,5 (Jensen Huang persoonlijk + trusts en aanverwante partijen)
- **Capital allocation track record** (jaartal × bedragen, mln USD):

| Jaar | Dividend totaal | Aandeleninkoop | M&A uitgaven | Organische capex |
|---|---|---|---|---|
| FY19 | 371 | 1.579 | 0 | 600 |
| FY20 | 390 | 0 | 0 | 489 |
| FY21 | — | — | 7.000 (Mellanox) | — |
| FY22 | — | — | 0 | — |
| FY23 | — | — | 0 | — |
| FY24 | 395 | 9.533 | 83 | 1.069 |
| FY25 | 834 | 33.706 | 1.007 | 3.236 |
| FY26 | 974 | ~40.600 | ~3.500 (Run:ai, CentML, Gretel, etc.) | 6.022 |

- **M&A-track-record:** Mellanox (april 2020, $6,9 mld) is een operationeel succes — de netwerktechnologie levert nu $26 mld networking-omzet (FY26, +142% YoY). De gedwongen termination van het ARM-bod ($40 mld, feb 2022) kostte $1,35 mld break-up fee en is achteraf bezien een gemiste kans gebleken (ARM IPO'de tegen $54 mld waardering in 2023). Kleinere acquisities (Run:ai voor GPU-orkestratie, CentML voor ML-compiler, Gretel voor synthetic-data) zijn aanvulling op de software-stack, niet kapitaalbepalend. Aandeleninkopen worden grotendeels op all-time-high gedaan ($40 mld in FY26 bij aandeelprijs $130-220) — niet textbook waarde-gericht inkoopgedrag, maar geconditioneerd door overschot kasstroom.
- **Beloning:** Jensen Huang's compensatiepakket FY26 daalde 27% tot circa $34 mln, hoofdzakelijk door lagere stock-awards (Bloomberg, 12 mei 2026). Vast salaris is laag ($1 mln); het grootste deel is performance-shares die vesten op meerjaarse omzet- en winst-KPI's. Colette Kress verdiende circa $21 mln in FY25. SBC totaal $4,7 mld in FY25 (~0,1% van marktkapitalisatie) — laag t.o.v. tech-sectorgemiddelde van 1-3%. CEO pay-ratio (CEO vs. mediaan werknemer) circa 200:1 — hoog maar binnen tech-sector-normen.
- **Insider activiteit:** netto verkoper. Over de afgelopen 18 maanden hebben 15 insiders aandelen verkocht voor totaal $3,3 mld; Huang ~$2,9 mld via 10b5-1-plannen (laatste plan filed mei 2025: 6 mln aandelen tot $865 mln). Colette Kress verkocht $28 mln. Géén open-markt aankopen door insiders. Per METHODE.md: open-markt aankopen zijn het sterkste vertrouwenssignaal en die ontbreken hier; consistent verkopen via gepland 10b5-1 (=geen verraad-signaal) is wel een patroon dat aandacht verdient. Geen verkoop-piek bij koersniveaus die op overwaardering wijzen.
- **Oordeel management:** STERK
- **Toelichting management:** Jensen Huang is een founder-CEO met 33 jaar onafgebroken leiderschap, eigenbelang van circa 3,5% (substantieel boven de 1%-drempel uit METHODE.md), een uitzonderlijk track record van strategische pivots (3D-graphics → CUDA → AI), en een transparante communicatiestijl (kwartaalverslagen geven gedetailleerde segmentbreakdowns en verwachte verkrapping). Capital allocation is grotendeels excellent — Mellanox toegevoegd, geen waardevernietigende mega-deals, conservatieve schuld, grote buybacks (maar tegen hoge koers). De compensatie is verstandig opgebouwd (performance-shares op meerjaarse KPI's), niet excessief in absolute zin. Twee aandachtspunten: het structurele verkooppatroon van insiders zonder open-markt aankopen (geen sterke "skin-in-the-game-signal" naar buiten toe op huidige koersen), en de aandeleninkopen op zeer hoge multiples die niet typisch is voor waarde-gerichte allocatie. Per saldo overweegt het sterke leiderschap en de bewezen executie — STERK is het juiste oordeel.

---

## 6. Sector & concurrentie

- **Sector-groeivooruitzicht:** AI-accelerator-markt 20-35% CAGR tot 2030 volgens diverse onderzoeken. BofA verhoogde mei 2026 de TAM-schatting voor 2030 AI-datacenter-systemen naar $1.700 mld (van $1.400 mld). IDC/Gartner: $45 mld (2024) → $500 mld (2030), CAGR ~35%. Bloomberg Intelligence: $604 mld in 2033, CAGR ~16% vanaf 2024. Brede consensus: data-center-AI is een 10-jaars structurele groei-pijl.
- **Porter five forces:**
  - **Rivaliteit:** middel — Bij high-end AI-accelerators is NVIDIA dominant (80-90% marktaandeel discrete GPU's), maar AMD pusht agressief met MI400/MI500-roadmap en hyperscalers ontwerpen eigen chips (Google TPU, Amazon Trainium, Microsoft Maia, Meta MTIA). Rivaliteit verergert maar is in compute-platform-rivaliteit zoeken-naar-tweede-bron, niet prijscompetitie. Marges blijven hoog, wat duidt op gematigde rivaliteit.
  - **Nieuwe toetreders:** laag — Toetreding tot leading-edge AI-chip-design vereist (1) een fabless-design-team met decennia ervaring, (2) toegang tot leading-edge TSMC-capaciteit (4nm/3nm/2nm — extreem schaars), (3) HBM-geheugen-allocatie (Samsung/Micron/SK Hynix), (4) een software-stack die met CUDA kan concurreren. Drempel: miljarden investering en tien jaar werk. Praktisch alleen hyperscalers (eigen chips) en gevestigde concurrenten (AMD, Intel) kunnen toetreden.
  - **Substituten:** middel — CPU's (Intel/AMD Xeon, EPYC, ARM Grace) zijn lang geen substituut meer voor AI-training. Voor inferencing zijn er substituten: hyperscaler-ASICs, FPGA's, en op den duur nieuwe architecturen (Cerebras wafer-scale, Groq LPU, neuromorphic chips). Inferentie is op weg om de helft van de AI-workload te worden — substitutiedreiging daar reëel maar gradueel.
  - **Macht leveranciers:** middel-hoog — TSMC is praktisch monopolist voor leading-edge fabricage. Single-source-risico op N4/N3-nodes, HBM is duopolie/oligopolie. Maar NVIDIA krijgt voorrang door volume — relatie is wederzijds afhankelijk. ASML levert EUV-machines aan TSMC, niet aan NVIDIA direct, maar bottlenecks daar werken door.
  - **Macht afnemers:** middel-hoog — Top 5 hyperscalers vormen >50% van NVIDIA-omzet. Zij hebben grote underhandelingsmacht (vooral nu ze eigen ASICs ontwikkelen), maar zijn praktisch in de huidige cycle "captive buyers" omdat ze de Blackwell-/Rubin-capaciteit nodig hebben voor hun AI-roadmap. Concentratierisico significant.
- **Concurrenten:**

| Concurrent | Marktaandeel % (AI-accelerators) |
|---|---|
| NVIDIA | ~70-75 |
| Hyperscaler-ASICs (Google TPU, Amazon Trainium, Microsoft Maia, Meta MTIA) | ~15-20 (gezamenlijk) |
| AMD (Instinct MI-serie) | ~8-12 |
| Intel (Gaudi-serie) | <2 |
| Overig (Cerebras, Groq, Tenstorrent, Huawei Ascend in China) | rest |

- **Positie van het bedrijf:** koploper — dominante speler met meer dan 70% van AI-accelerator-omzet, complete stack (compute + networking + software), jaarlijkse architectuurupdates. NVIDIA's voorsprong is niet alleen prestatie-gebaseerd maar fundamenteel software-gebaseerd (CUDA).

### TAM/SAM/SOM

- **TAM (mln USD, 2030):** 1.700.000 (totale AI-datacenter-systemen TAM per BofA mei 2026)
- **TAM-groei %:** ~25 (CAGR 2024-2030)
- **SAM (mln USD):** ~1.000.000 (deel van TAM dat betrekking heeft op accelerator-compute, networking en software — NVIDIA's adresseerbare markt)
- **SAM-groei %:** ~30
- **Huidige penetratie %** (NVIDIA omzet FY26 / huidige SAM ~$300 mld): ~72
- **Impliciete penetratie na horizon %** (bij optimistisch DCF: omzet 2030 ~$430 mld op SAM $1.000 mld): ~43
- **Groei plausibel?** ja, mits NVIDIA marktaandeel >60% behoudt en de AI-capex-cyclus 4-5 jaar doorzet. Het optimistische scenario impliceert dat NVIDIA's marktaandeel daalt (gezond) en dat de SAM zelf hard groeit.
- **Bron TAM/SAM:** Bank of America Global Research mei 2026 ($1,7 trillion TAM 2030); IDC/Gartner; Bloomberg Intelligence
- **Toelichting:** Het Bank-of-America-cijfer is een aggregaat van hyperscaler-capex-projecties (Microsoft $80 mld/jaar, Meta $60 mld/jaar, Google $50 mld/jaar, Amazon $90 mld/jaar in 2026 — gezamenlijk $280 mld/jr en stijgend) plus sovereign-AI (Stargate $500 mld over 4 jaar) plus enterprise. Plausibiliteit: dit veronderstelt dat AI-modellen blijven groeien in parameter-aantal en gebruik. Een GenAI-winter (model-prestatieplafond, regulatoire rem, slechtere unit-economics) zou de TAM-curve significant afvlakken. Toch is zelfs het basisscenario (geleidelijke groei in een nog grote markt) ruim voldoende om de huidige NVIDIA-omzet structureel hoog te houden.

---

## 7. Analyse-frameworks (9 frameworks, SCORES 0-5)

### Graham

- **Oordeel:** VOLDOET NIET
- **Graham number** (√(22,5 × EPS × BVPS) = √(22,5 × 4,90 × 6,45)): 26,67
- **Margin of safety %** (t.o.v. huidige koers, Graham-perspectief): -87
- **Toelichting:** Met P/E 43 en P/B 33 valt NVDA ver buiten Grahams defensieve criteria (P/E < 15, P/B < 1,5). De Graham Number ligt op circa $27 — dieper dan zelfs het pessimistisch DCF-scenario. Grahams criteria zijn ontworpen voor stabiele, voorspelbare bedrijven met margin of safety, en dat is precies wat NVDA momenteel niet is: het aandeel prijst extreem hoge groei in en biedt geen klassieke bodem-onder-de-waardering. Voor een Graham-defensieve belegger is dit een no-go.
- **Score:** 1

### Buffett / Munger

- **Oordeel:** GEDEELTELIJK
- **ROIC structureel boven WACC?** ja
- **Toelichting:** NVIDIA voldoet aan drie van Buffett's vier criteria: wonderfull-business-test (begrijpelijk-genoeg AI-platform, hoge ROIC structureel >2×WACC sinds FY22 met spread van 90 procentpunten, brede moat door CUDA-lock-in en netwerkeffecten, sterk founder-led management). Het vierde criterium — fair price — faalt: P/FCF 53, EV/EBIT 39, P/E 43. Buffett zou zeggen "wonderful company at a sky-high price". Per rubric: ROIC structureel >2×WACC én WIDE moat (categoriescore: 3 sterk + 1 aanwezig + 1 beperkt → "wide") maar P/FCF > 20 → score 4 (niet score 5 die P/FCF ≤ 20 vereist).
- **Score:** 4

### Peter Lynch

- **Categorie:** Fast Grower
- **Oordeel:** NEUTRAAL
- **PEG-ratio:** 1,44 (op basis P/E 43 / 5-jaars verwachte EPS-CAGR ~30%)
- **Toelichting:** NVIDIA past Lynch's fast-grower-mal: omzet- en winstgroei structureel >20% per jaar, helder en uitlegbaar verhaal ("ze maken de chips die ChatGPT laten draaien"). PEG-ratio is rond 1,4 — boven de attractiviteitsgrens van 1,0 maar onder de schrikgrens van 2,0. Lynch zou zeggen dat het verhaal sterk is maar de prijs niet bijzonder aantrekkelijk. Score 3 per rubric (PEG ≤ 1,5 en verhaal helder).
- **Score:** 3

### Phil Fisher

- **Oordeel:** STERK
- **Toelichting:** Op Fishers 15 punten scoort NVIDIA hoog: groeipotentieel van producten (Blackwell, Rubin, Feynman-roadmap), R&D-cultuur die structureel boven sectorgemiddelde investeert (8,6% van omzet vs. ~6% sector), margebescherming via CUDA-moat, management-integriteit en transparantie via consistente earnings-calls. Twee zwakkere punten: customer-concentratie (hyperscalers > 50% van omzet) en geografische afhankelijkheid van TSMC. Per rubric (R&D-pct hoger dan sectorgemiddelde EN margebescherming door moat EN management integriteit STERK): score 4 omdat alle drie criteria voldoen maar de schaal van klantconcentratie de kwetsbaarheid net iets verzwaart.
- **Score:** 4

### Magic Formula (Greenblatt)

- **Oordeel:** ONAANTREKKELIJK
- **Earnings yield %** (EBIT / EV): 2,56
- **Return on capital %** (Greenblatt EBIT / (NWC + NFA)): ~80
- **Toelichting:** NVIDIA scoort uitzonderlijk hoog op Return on Capital — circa 80% in Greenblatt-definitie (EBIT $130 mld op netto werkkapitaal + vaste activa ~$160 mld). Maar de earnings yield is laag: 2,56% — onder de Magic-Formula-drempel van 3% die Greenblatt typisch zoekt. Per rubric: earnings yield < 3% → score 1. Magic Formula zoekt naar de combinatie van hoge winstgevendheid én lage waardering; NVDA biedt alleen de eerste.
- **Score:** 1

### Moat

- Zie sectie 4. WIDE MOAT, 3 categorieën STERK (immateriële activa, overstapkosten, netwerkeffecten), 1 categorie AANWEZIG (kostenvoordeel), 1 BEPERKT (efficiënte schaal). ROIC-WACC-spread 90 procentpunten. Per rubric: WIDE moat (≥3 categorieën STERK) EN ROIC-WACC spread > 10pp → score 4. De spread is meer dan 20pp dus zou score 5 mogelijk zijn, maar de definitie van monopolie/duopolie MET pricing power dekt niet de hele realiteit (hyperscalers ontwikkelen eigen ASICs als concurrent), dus score 4 is gepast.
- **Score:** 4

### Management

- Zie sectie 5. Founder-led, owner-operator > 1% (3,5%), capital allocation goed (Mellanox-succes, conservatieve schuld, grote buybacks weliswaar op hoge koers), prikkels aligned (performance-shares op meerjaarse KPI's), geen controverses, transparante downside-disclosure. Insider-net-verkoper (15 verkopen, 0 koop) zwakt het signaal af. Per rubric: capital allocation GOED EN prikkels aligned EN geen controverses → score 4.
- **Score:** 4

### Fair Value DCF

- Zie sectie 12. Koers $212; basis-DCF fair value $93,79; downside -55,8%. Per rubric: downside > 15% → score 1.
- **Score:** 1

### Fair Value IPO-gecorr.

- NVIDIA's IPO was januari 1999, dus > 10 jaar geleden. Per rubric: score = gelijk aan Fair Value DCF basis = 1.
- **Score:** 1

### Scorekaart totaal

- **Totaalscore:** 23
- **Max:** 45
- **Eindoordeel:** PASS
  - Regel: totaal < 24 OF Fair Value DCF-score = 1 → PASS. Beide condities tegelijk vervuld.
- **Samenvatting:** NVIDIA is operationeel een uitzonderlijk bedrijf — wide moat, structureel ROIC > 100%, dominante speler in de definiërende technologie van het komende decennium. Op kwalitatieve frameworks scoort het hoog (Moat 4, Management 4, Fisher 4, Buffett-Munger 4). Maar de waardering is extreem. Het DCF-basis-scenario rechtvaardigt slechts $93,79 op een koers van $212, en het bedrijf zou de komende vijf jaar 38,5% FCF-groei moeten leveren om de huidige koers te rechtvaardigen — boven de consensus-schatting (~30% EPS-groei) en historisch niet vol te houden voor een bedrijf van deze omvang. Het oordeel is PASS niet omdat NVIDIA een slecht bedrijf is, maar omdat de prijs een margin of safety van -55% inhoudt. Een belegger die overtuigd is van de meerjarige AI-capex-supercyclus kan tactisch een kleine positie willen, maar fundamenteel oordeel blijft PASS tot een koerscorrectie richting $150 of lager.

---

## 8. Risico's

| # | Omschrijving | Kans | Impact | DCF-aanname geraakt | Toelichting |
|---|---|---|---|---|---|
| 1 | AI-capex-vertraging bij hyperscalers (groei-ratio compressie) | MIDDEN | GROOT | Omzetgroei fase 1 en 2 | Een vertraging in capex bij Microsoft/Meta/Amazon/Google na 2027 zou de FCF-groei direct raken. Dit kan komen door dalende ROI op AI-investeringen, plafondbereiken in modelarchitecturen, of macro-economische tegenwind. Een halvering van de groei in fase 1 verkleint de basis-DCF fair value met circa 30-40%. |
| 2 | Marktaandeel-verlies aan hyperscaler-ASICs | MIDDEN | GROOT | EBIT-marge, omzetgroei | Google TPU v7, Amazon Trainium 3, Microsoft Maia 2 en Meta MTIA Gen2 komen 2026-2027 op de markt. Hyperscalers willen tweede-bron-strategie en lagere unit-kosten. Verlies van 5-10 procentpunten marktaandeel over 5 jaar betekent zowel omzetdaling als margedruk. |
| 3 | China-exportbeperkingen breiden uit | HOOG | MIDDEL | Omzet (~13% China), margin | China-omzet was ~13% van totaal in FY26. Het H20-fiasco (april 2025 ban, deels opgelost via 15% revenue-sharing-regel) toont volatiele politiek. Een complete ontkoppeling zou $25-30 mld jaaromzet kosten. Huawei Ascend wint in China-vacuüm. |
| 4 | AMD MI400/MI500 levert competitieve prestaties | MIDDEN | MIDDEL | EBIT-marge | AMD's MI400 (H2 2026) en MI500 (2027) komen architecturaal dichterbij. Als AMD bewijst dat het CUDA-alternatief ROCm "good enough" is voor inferencing-workloads (60% van AI-compute), kan NVIDIA prijssetting verliezen en marges met 5-10 procentpunten zien dalen. |
| 5 | TSMC-leveringsbottleneck of Taiwan-geopolitiek | MIDDEN | GROOT | Omzet, capex-multiple | Single-source-afhankelijkheid van TSMC voor leading-edge nodes. Een Taiwan-incident zou een existentiële supply-disruptie betekenen. Minder dramatisch maar plausibel: TSMC-capaciteit voor 3nm/2nm onvoldoende voor NVIDIA + AMD + Apple + hyperscaler-ASICs gelijktijdig, leidend tot rantsoenering. |
| 6 | Insider-verkooppatroon zonder open-markt-aankopen | LAAG | KLEIN | n.v.t. (kwalitatief) | 15 insiders verkochten $3,3 mld in 18 maanden, 0 open-markt-aankopen. Verkopen via 10b5-1 zijn niet automatisch negatief, maar het volledige ontbreken van koopactiviteit door management op alle koersniveaus — inclusief de dip naar $115 medio 2025 — duidt niet op overtuiging dat de koers significant ondergewaardeerd is. |
| 7 | Pre-IPO financial-engineering check | LAAG | KLEIN | n.v.t. | Niet geconstateerd. NVIDIA IPO'de in 1999, ruim buiten de 10-jaars-window. Geen pre-IPO schuldlading, geen dividend recap, geen carve-out structuren. Punt is afgevinkt zonder bevindingen. |
| 8 | Modelarchitectuur-verschuiving (state-space, MoE-efficiency) vermindert GPU-honger | LAAG | GROOT | Omzetgroei, terminal groei | Onderzoek naar lichtere transformer-alternatieven (Mamba, RWKV) of efficiëntere mixture-of-experts-modellen kan de GPU-vraag per ton inferentie significant verlagen. Een doorbraak die training-/inferentie-compute halveert, raakt NVIDIA hard. Probabiliteit is laag op 1-3 jaar maar groeit op 5-10 jaar horizon. |

---

## 9. These invalide bij

Deze investeringsthese (PASS op huidige $212) wordt weerlegd wanneer: (1) de aandelenkoers daalt naar $130 of lager zonder fundamentele verslechtering, waardoor de DCF-basis-margin-of-safety positief wordt; (2) consensus-FCF-groeischatting voor 2027-2030 stijgt naar >35% per jaar én NVIDIA dit minstens twee opeenvolgende fiscale jaren waarmaakt; (3) de AI-accelerator-TAM-schatting voor 2030 verschuift naar boven $2.500 mld én NVIDIA's marktaandeel >70% blijft, waarmee zelfs een hogere reverse-DCF-groei haalbaar wordt.

---

## 10. ESG

### Materiële factoren (SASB-gebaseerd)

| Factor | SASB-categorie | Risiconiveau | Financiële impact | DCF-impact |
|---|---|---|---|---|
| Energieverbruik datacenters | Environmental Footprint of Hardware Lifecycle | MIDDEN | Klanten (hyperscalers) zoeken naar energie-efficiëntere accelerators; NVIDIA's perf/watt verbetert per generatie. Geen directe boete maar regulatoire druk op klantkant. | Indirect — kan groei dempen in EU |
| Supply-chain risico Taiwan / TSMC | Materials Sourcing & Efficiency | HOOG | Concentratierisico productie — Taiwan-geopolitiek. | Significant — terminal value-aanname |
| Datasecurity en AI-misbruik | Data Security | MIDDEN | Reputatie- en regelgevingsrisico (EU AI Act, US export controls). | Modest — operating costs |
| Arbeidsomstandigheden in supply-chain | Labor Practices | LAAG | TSMC en Taiwanese OEM's onder normale arbeidsmark-condities. | Minimaal |
| Belasting (effectieve tax rate FY26 ~15%) | Tax Transparency | MIDDEN | Gebruik van Ierse/Bermuda-structuren; OECD pillar-2-regels kunnen tax rate met 1-2pp verhogen. | Direct via NOPAT-aanname |

- **Eindoordeel ESG:** GEMIDDELD RISICO
- **Toelichting:** ESG-profiel is "gemiddeld" — geen acuut probleem maar wel structureel monitor-bedrijf. Het meest materiële risico is supply-chain-concentratie (Taiwan/TSMC), gevolgd door tax-transparency. De governance-kant (founder-led, sterke board, geen recente schandalen) is bovengemiddeld. Energie/klimaat is een sectorthema waarbij NVIDIA niet uitsteekt maar ook niet achterloopt.

---

## 11. Katalysatoren (5-8 stuks, chronologisch)

| Datum ca. | Omschrijving | Richting | Impact |
|---|---|---|---|
| 2026-05 | Q1 FY27 earnings (20 mei 2026) — eerste kwartaal zonder SBC-uitsluiting in non-GAAP; revenue-guidance $78 mld | BINAIR | GROOT |
| 2026-08 | Q2 FY27 earnings (verwacht aug 2026) — eerste volledige Blackwell-Ultra-kwartaal | POSITIEF | GROOT |
| 2026-10 | Vera Rubin (Rubin R100) productlancering bij GTC October — performance benchmark vs. AMD MI400 | POSITIEF | GROOT |
| 2026-Q4 | AMD MI400 commerciële ramp-up (H2 2026) — directe vergelijking met Blackwell Ultra | NEGATIEF | MIDDEL |
| 2026-11 | Q3 FY27 earnings | POSITIEF | MIDDEL |
| 2027-02 | Q4/FY27 earnings + FY28 guidance — eerste full-year guidance op nieuwe SBC-rapporteringsbasis | BINAIR | GROOT |
| 2027-Q1 | US-China handelsbespreking voortgang — H20-opvolger licensering | BINAIR | GROOT |
| 2027-mid | Hyperscaler-capex-guidance voor 2028 (analist-dag Microsoft, Meta, Google) | POSITIEF | GROOT |

---

## 12. Fair value — kwantitatief (DCF)

### WACC-componenten

- **Risicovrije rente %:** 4,46
- **Bron risicovrije rente:** US 10-year Treasury, peildatum 14 mei 2026 (TradingEconomics, Bloomberg confirmaties)
- **Type:** spot (huidige rente; >150bp afwijking van 10y gemiddelde — zie Stap 0)
- **ERP (equity risk premium) %:** 4,23
- **Bron ERP:** Damodaran implied ERP januari 2026 update (pages.stern.nyu.edu/~adamodar)
- **Beta (adjusted, Blume-equivalent):** 2,24
- **Bron beta:** Yahoo Finance 5y monthly, mei 2026 (cross-check Investing.com 2,24)
- **Type beta:** 5y monthly regressie
- **Country risk premium %:** 0 (Verenigde Staten = thuismarkt, geen CRP)
- **Size premium %:** 0 (mega-cap)
- **Cost of equity %:** 13,94 (= 4,46 + 2,24 × 4,23)
- **Schuldkosten na belasting %:** 2,98 (3,5% pretax × (1 - 15% tax))
- **E/V gewicht %:** 99,79 (marktwaarde equity dominant)
- **D/V gewicht %:** 0,21
- **WACC %:** 13,91
- **Sector WACC % (referentie Damodaran):** ~10,5 (semiconductor sector). NVIDIA's WACC ligt hoger door hoge beta (2,24) — gerechtvaardigd door volatiliteit.
- **Illiquiditeitskorting %:** 0 (NVDA is een van de meest liquide aandelen ter wereld, dagvolume miljarden USD)

### DCF model-specs

- **Model type:** 2-fase (jaar 1-5 fase 1, jaar 6-10 fase 2, terminal jaar 11+)
- **FCF-definitie:** FCFF (Free Cash Flow to Firm), gedefinieerd als CFO - capex - SBC
- **Basis FCF:** 91.176 (mln USD; FCF FY26 $96,7 mld minus geschatte SBC $5,5 mld)
- **Basis FCF na SBC:** 91.176 (zelfde waarde — SBC is reeds in mindering gebracht)
- **FCF-type:** Genormaliseerd op FY26-niveau (post-SBC). NVIDIA is geen klassiek cyclisch bedrijf zoals staal/olie, maar de huidige FCF reflecteert een AI-capex-piekjaar. Sanity-check via mediaan FCF-marge × omzet: FY25 FCF-marge 46,5%, FY26 44,8% — ruwweg consistent.
- **Groei fase 1 %** (jaar 1-5): basis 22%, pessimistisch 15%, optimistisch 30%
- **Groei fase 2 %** (jaar 6-10): basis 10%, pessimistisch 6%, optimistisch 14%
- **Terminal groei %:** basis 3,0%, pessimistisch 2,5%, optimistisch 3,5% (≤ lange-termijn US BBP-groei ~3-4% nominaal)
- **Terminal methode:** Gordon Growth Model, cross-check via exit-multiple
- **Exit multiple gebruikt (EV/EBITDA):** 20x (sector-mediaan halfgeleiders ~15x; NVIDIA's premium ~25% gerechtvaardigd door moat)
- **Bron exit multiple:** Damodaran semiconductor sector data + NVIDIA historische 10y mediaan
- **Terminal value Gordon growth (basis):** ~$1.760 mld
- **Terminal value exit multiple (basis):** ~$2.000 mld (FCF jaar 10 ~$255 mld → impliciete EBITDA ~$300 mld × 20 = $6.000 mld nominaal → PV ~$1.600 mld)
- **Terminal value % van totaal (basis):** 45,5
- **Terminal implied EV/EBITDA (basis):** ~18-20x
- **Terminal groei consistentie:** Terminal groei 3% vereist herinvesteringsvoet 6% bij lange-termijn ROIC 50% — plausibel voor een volwassen high-moat technologiebedrijf. Een lagere herinvesteringsvoet sluit aan bij de fabless-natuur van NVIDIA's businessmodel.
- **Mid-year convention:** true
- **Aandelen uitstaand (mln):** 24.400 (gemiddelde basic 24.359 / diluted 24.514 — gebruik 24.400)
- **Nettoschuld huidig (mln USD):** -51.516 (netto-kaspositie; gebruikt cash + ST invest minus bruto schuld, excl. LT investments voor conservatieve fair value)

### DCF-toelichting

Ik gebruik een 2-fase FCFF-model met mid-year convention. Vertrekpunt is FCF na SBC ($91,2 mld), niet de gerapporteerde GAAP-FCF — SBC is een echte kostenpost voor aandeelhouders. WACC is 13,91% (cost of equity 13,94%, schuldkosten 2,98% na 15% belasting, E/V 99,79%). De netto-kaspositie van $51,5 mld wordt na DCF opgeteld bij de present value of operations. Terminal-groei is 3% (basis), onder de Amerikaanse lange-termijn nominale BBP-groei. Terminal-value vormt 45,5% van totaal in basis-scenario — onder de 75%-grens uit METHODE.md, geloofwaardig. Sanity-check: in basis-scenario impliceert FCF in jaar 10 ~$255 mld; op een verwachte EBITDA in jaar 10 van ~$310 mld levert dat een impliciete exit-EV/EBITDA van ~20x — redelijk voor een volwassen high-moat bedrijf. De kansweging (25% pess, 50% basis, 25% opti) levert kansgewogen fair value $100,20, ruim onder de huidige koers van $212.

### 5-jaars projectie (basis-scenario, mln USD)

| Jaar | Omzet | Omzetgroei % | EBIT | EBIT-marge % | NOPAT | Capex | ΔNWC | SBC | FCF |
|---|---|---|---|---|---|---|---|---|---|
| 2027 (FY27) | 312.610 | 45 | 187.566 | 60 | 159.431 | 8.500 | 12.000 | 6.500 | 132.431 |
| 2028 (FY28) | 421.024 | 35 | 252.614 | 60 | 214.722 | 10.000 | 14.000 | 7.500 | 183.222 |
| 2029 (FY29) | 547.331 | 30 | 328.398 | 60 | 279.139 | 11.500 | 16.000 | 8.500 | 243.139 |
| 2030 (FY30) | 656.798 | 20 | 387.510 | 59 | 329.384 | 13.000 | 18.000 | 9.500 | 288.884 |
| 2031 (FY31) | 755.318 | 15 | 438.084 | 58 | 372.371 | 14.500 | 20.000 | 10.500 | 327.371 |

Opmerking: NWC-toename volgt schaling met omzet; SBC neemt toe met loonpost.

### Scenarios

| Scenario | FCF-groei % (fase 1, 5y) | WACC % | Fair value | Upside % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 15,0 | 14,41 | 61,33 | -71,1 | 25 |
| Basis | 22,0 | 13,91 | 93,79 | -55,8 | 50 |
| Optimistisch | 30,0 | 13,41 | 151,87 | -28,4 | 25 |

- **Kansgewogen fair value:** 100,20

### Reverse DCF

- **Impliciete groei %** (5y fase-1 FCF-groei nodig voor huidige koers $212, fase-2 = helft): 38,5
- **Historische FCF CAGR % (FY16 → FY26, 10y):** 57,1 (vertrekpunt extreem laag bedrag, dus CAGR overschat structurele groeicapaciteit)
- **Consensus groei %** (EPS CAGR 2027-2030, analist-consensus): ~25-30
- **Interpretatie:** De markt prijst 38,5% FCF-groei in fase 1 (5 jaar) in. Dat is boven de consensus-EPS-groei van 25-30% en historisch alleen overtreffen door bedrijven die nog een fractie van NVIDIA's huidige schaal hadden. Voor een bedrijf met $216 mld omzet en 56% nettomarge is 38,5% jaarlijkse FCF-groei gedurende vijf jaar buitengewoon ambitieus. Het impliceert FCF van $96 mld → $478 mld in 2031 — meer dan de huidige hele omzet. Het is niet onmogelijk (de AI-capex-cyclus heeft hyperscale-investeringen ontketend die hier deels in passen), maar het laat geen ruimte voor tegenvallers en vertegenwoordigt het optimistische einde van het verdeling-spectrum.

### EPV (Bruce Greenwald)

- **Genormaliseerde EBIT-marge %:** 46,0 (gemiddelde FY22-FY26: 37,3 + 15,7 + 54,1 + 62,4 + 60,4)
- **Genormaliseerde NOPAT:** 84.395 (mln USD; omzet $215,9 mld × 46% EBIT × (1-15% tax))
- **Maintenance capex:** 2.843 (mln USD; FY26 D&A als proxy)
- **Adjusted earnings power:** 84.395 (NOPAT — maintenance capex effectief = D&A, dus saldo = NOPAT)
- **EPV per aandeel:** 26,97 (= ($84.395 / 13,91% + $51.516 net cash) / 24.400 aandelen)
- **Groeipremie %:** 247,7 (= ($93,79 basis fair value - $26,97 EPV) / $26,97 × 100%)

### Andere methoden

- **DDM uitgevoerd?** nee (dividendrendement <0,1%; immaterieel)
- **SOTP uitgevoerd?** nee (data-center segment is dermate dominant dat sum-of-the-parts geen toegevoegde waarde geeft)

### Synthese fair value

- **Bandbreedte laag:** 61,33 (pessimistisch DCF)
- **Bandbreedte centraal:** 93,79 (basis DCF)
- **Bandbreedte hoog:** 151,87 (optimistisch DCF)
- **Methode-gewichten** (totaal 100%):
  - DCF: 70
  - EPV: 15
  - Multiples (relatief — gem peer EV/EBITDA × NVIDIA EBITDA): 15
- **Margin of safety vereist %:** 20 (mega-cap, hoge moat, datakwaliteit hoog — onderkant van typische 20-30% range)
- **Koopniveau** (fair value basis × (1 - MOS) = 93,79 × 0,80): 75,03
- **Synthese-toelichting:** De DCF krijgt 70% gewicht omdat het de meest complete waarderingsmethode is voor een groei-cashflow-bedrijf. EPV (15%) toont de "no-growth"-baseline van $27 — circa een zevende van de huidige koers, illustratie van hoeveel groeipremie er ingeprijsd is. Relatieve multiples (15%) houden rekening met het feit dat het hele AI-sector-segment hoog gewaardeerd is, wat een sector-fonds-belegger andere referentiepunten geeft (peer EV/EBITDA gemiddelde ~25x voor halfgeleiders levert ~$140 fair value op NVIDIA's EBITDA $133 mld). Margin of safety van 20% leidt tot een aanvankelijke koopdrempel van $75. Het oordeel PASS is robuust onder alle gewichtcombinaties.

### Gevoeligheid (DCF)

WACC-range: [10,0%, 10,5%, 11,0%, 11,5%, 12,0%, 12,5%]
Groei-range (fase 1, jaar 1-5, fase 2 = helft): [10%, 15%, 20%, 25%, 30%]

|  | WACC 10,0% | 10,5% | 11,0% | 11,5% | 12,0% | 12,5% |
|---|---|---|---|---|---|---|
| g1=10% | 79,3 | 74,3 | 69,9 | 66,0 | 62,5 | 59,4 |
| g1=15% | 104,3 | 97,3 | 91,2 | 85,8 | 81,0 | 76,7 |
| g1=20% | 136,9 | 127,4 | 119,0 | 111,6 | 105,0 | 99,1 |
| g1=25% | 179,4 | 166,4 | 155,0 | 145,0 | 136,0 | 128,0 |
| g1=30% | 234,1 | 216,7 | 201,3 | 187,8 | 175,8 | 165,1 |

De koers van $212 wordt pas gerechtvaardigd bij g1 ≥ 30% én WACC ≤ 10,5% — voorwaarden die geen van beide overeenkomen met de gemodelleerde 13,91% WACC. Bij realistische WACC (11-12%) heb je groei van >30% nodig.

---

## 13. Databronnen

### Bronnen-hiërarchie

- **Jaarverslag PDF / IR-pagina** → betrouwbaarheid **HOOG**
- **Beursmelding / prospectus** → betrouwbaarheid **HOOG**
- **Aggregator** (StockAnalysis.com, MacroTrends) → betrouwbaarheid **AGGREGATOR**

### Financiële bronnen (10 jaar historie — VERPLICHT)

| Jaar | Bron | URL | Betrouwbaarheid |
|---|---|---|---|
| FY16 | StockAnalysis revenue history | https://stockanalysis.com/stocks/nvda/revenue/ | AGGREGATOR |
| FY17 | StockAnalysis revenue history | https://stockanalysis.com/stocks/nvda/revenue/ | AGGREGATOR |
| FY18 | StockAnalysis revenue history | https://stockanalysis.com/stocks/nvda/revenue/ | AGGREGATOR |
| FY19 | NVIDIA persbericht Q4FY20 (vergelijkende cijfers) | https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-2020 | HOOG |
| FY20 | NVIDIA persbericht Q4FY20 | https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-2020 | HOOG |
| FY21 | StockAnalysis revenue + WebSearch confirmaties | https://stockanalysis.com/stocks/nvda/revenue/ | AGGREGATOR |
| FY22 | StockAnalysis financials | https://stockanalysis.com/stocks/nvda/financials/ | AGGREGATOR |
| FY23 | StockAnalysis financials | https://stockanalysis.com/stocks/nvda/financials/ | AGGREGATOR |
| FY24 | NVIDIA persbericht Q4FY25 (vergelijkende cijfers) | https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-2025 | HOOG |
| FY25 | NVIDIA persbericht Q4FY25 | https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-2025 | HOOG |
| FY26 | NVIDIA persbericht Q4FY26 + 10-K SEC | https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-2026 | HOOG |

**Harde eis-check:** de 5 meest recente jaren zijn FY22-FY26. Van deze 5 zijn FY24, FY25 en FY26 HOOG (3 stuks). FY22 en FY23 zijn AGGREGATOR — dat voldoet niet aan de harde eis. Zie ## Opmerkingen voor Claude Code.

### Jaarverslagen geraadpleegd

| Jaar | Bron | URL |
|---|---|---|
| FY20 | NVIDIA persbericht Q4FY20 + 10-K | https://www.sec.gov/Archives/edgar/data/1045810/000104581020000010/nvda-2020x10k.htm |
| FY25 | NVIDIA persbericht Q4FY25 + 10-K | https://www.sec.gov/Archives/edgar/data/1045810/000104581025000023/nvda-20250126.htm |
| FY26 | NVIDIA persbericht Q4FY26 + 10-K | https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm |

### Beursmeldingen geraadpleegd

| Datum | Omschrijving | URL |
|---|---|---|
| 2026-02-25 | NVIDIA Q4FY26 earnings release | https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-2026 |
| 2026-05-12 | Bloomberg: Jensen Huang compensatiepakket 27% lager | https://www.bloomberg.com/news/articles/2026-05-12/nvidia-ceo-pay-package-shrinks-27-on-smaller-stock-awards |
| 2025-10-31 | Bloomberg: Huang completed $1B share-sale plan | (Bloomberg) |
| 2026-04-29 | Q1 FY27 earnings call date set | https://nvidianews.nvidia.com/news/nvidia-sets-conference-call-for-first-quarter-financial-results-6919947 |
| 2026-05-08 | Suzanne Nora Johnson appointed to Board | https://nvidianews.nvidia.com/news/nvidia-names-suzanne-nora-johnson-to-board-of-directors |
| 2026-05-07 | NVIDIA + IREN partnership 5 GW AI infrastructure | https://nvidianews.nvidia.com/news/nvidia-and-iren-announce-strategic-partnership-to-accelerate-deployment-of-up-to-5-gigawatts-of-ai-infrastructure |
| 2026-05-06 | NVIDIA + Corning US manufacturing partnership | https://nvidianews.nvidia.com/news/nvidia-and-corning-announce-long-term-partnership-to-strengthen-us-manufacturing-for-ai-infrastructure |

### IPO-prospectus

- **Geraadpleegd?** nee (IPO 1999, >25 jaar geleden; niet relevant voor analyse)
- **URL:** n.v.t.
- **Pre-IPO data beschikbaar?** nee (IPO ver buiten 10-jaars-correctie-venster)
- **Pre-IPO bron:** n.v.t.

### Non-GAAP

- **Gebruikt?** ja (NVIDIA rapporteert beide GAAP en non-GAAP)
- **Toelichting:** NVIDIA's non-GAAP sluit stock-based compensation, acquisition-related en andere kosten, en gains/losses op equity investments uit. In deze analyse gebruik ik GAAP als primaire grondslag voor DCF; SBC wordt expliciet als kostenpost in mindering gebracht op FCF (FCF na SBC). Vanaf Q1 FY27 stopt NVIDIA met het uitsluiten van SBC uit non-GAAP — wat de transparantie verbetert.

### Ontbrekende data (eerlijke lijst)

- FY21 balans- en kasstroomdetails niet uit primair bron geverifieerd; alleen omzet en nettowinst via aggregator/WebSearch. Cellen in tabel daarom leeg.
- FY16/FY17/FY18 balans- en kasstroomdetails niet beschikbaar via geopende bronnen. Cellen leeg.
- Insider-eigenbelang Jensen Huang per exacte peildatum: schatting ~3,5% gebaseerd op laatste 13G/proxy-vermeldingen.
- FY22 en FY23 cash-flow-detail (CFO, capex, SBC apart) niet uit primair persbericht; alleen via aggregator (StockAnalysis income statement geeft FCF totaal, niet de subcomponenten apart).
- TAM/SAM-bronnen zijn breed (BofA, IDC, Bloomberg Intel) — geen consensus-cijfer; gerapporteerd is BofA mei 2026 als meest recente.

### Peildatum analyse

- **2026-05-14**

---

## 14. Volledige bronnen-lijst

| Titel | URL | Type |
|---|---|---|
| NVIDIA Q4 FY26 Earnings Release | https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-2026 | jaarverslag |
| NVIDIA Q4 FY25 Earnings Release | https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-2025 | jaarverslag |
| NVIDIA Q4 FY20 Earnings Release | https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-2020 | jaarverslag |
| NVIDIA 10-K FY26 (SEC EDGAR) | https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm | jaarverslag |
| NVIDIA 10-K FY25 (SEC EDGAR) | https://www.sec.gov/Archives/edgar/data/1045810/000104581025000023/nvda-20250126.htm | jaarverslag |
| StockAnalysis NVDA Income Statement | https://stockanalysis.com/stocks/nvda/financials/ | aggregator |
| StockAnalysis NVDA Balance Sheet | https://stockanalysis.com/stocks/nvda/financials/balance-sheet/ | aggregator |
| StockAnalysis NVDA Revenue History | https://stockanalysis.com/stocks/nvda/revenue/ | aggregator |
| MacroTrends NVDA Revenue | https://www.macrotrends.net/stocks/charts/NVDA/nvidia/revenue | aggregator |
| Damodaran Implied ERP (jan 2026) | https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html | databron |
| US 10y Treasury Yield (TradingEconomics) | https://tradingeconomics.com/united-states/government-bond-yield | databron |
| Yahoo Finance NVDA Stock & Stats | https://finance.yahoo.com/quote/NVDA/key-statistics/ | aggregator |
| MarketBeat NVDA Insider Trading | https://www.marketbeat.com/stocks/NASDAQ/NVDA/insider-trades/ | databron |
| GuruFocus Jen-Hsun Huang Insider | https://www.gurufocus.com/insider/4375/jen-hsun-huang | databron |
| Bloomberg: Huang $1B share sale plan completed (31 okt 2025) | https://www.bloomberg.com/ | nieuwsartikel |
| Bloomberg: NVIDIA CEO Pay Package Shrinks 27% (12 mei 2026) | https://www.bloomberg.com/news/articles/2026-05-12/nvidia-ceo-pay-package-shrinks-27-on-smaller-stock-awards | nieuwsartikel |
| BofA: NVIDIA price target $320 + $1.7T TAM (13 mei 2026) | https://247wallst.com/investing/2026/05/13/bofa-hikes-nvidia-price-target-to-320-on-massive-1-7-trillion-ai-data-center-forecast/ | analistenrapport |
| Fortune: NVIDIA Q4 FY26 $68B revenue (25 feb 2026) | https://fortune.com/2026/02/25/nvidia-nvda-earnings-q4-results-jensen-huang/ | nieuwsartikel |
| Reuters: $4M cash bonus CEO 2027 plan | https://www.reuters.com/business/nvidia-sets-4-million-cash-bonus-ceo-huang-under-2027-compensation-plan-2026-03-06/ | nieuwsartikel |
| Tomshardware: NVIDIA sold final ARM stake | https://www.tomshardware.com/tech-industry/nvidia-sells-off-final-arm-shares-but-licensing-deals-will-continue-usd140-million-stake-sold-equating-to-1-1-million-shares | nieuwsartikel |
| HotHardware AMD MI400 vs Rubin | https://hothardware.com/news/instinct-mi400-challenge-vera-rubin | nieuwsartikel |
| AAF: Trump's H20 chip tax-on-China | https://www.americanactionforum.org/insight/trumps-political-tax-on-nvidia-chips-to-china/ | onderzoeksrapport |
| Wikipedia: NVIDIA company history | https://en.wikipedia.org/wiki/Nvidia | aggregator |
| NVIDIA Mellanox completion (april 2020) | https://nvidianews.nvidia.com/news/nvidia-completes-acquisition-of-mellanox-creating-major-force-driving-next-gen-data-centers | beursmelding |
| NVIDIA-Softbank ARM termination | https://nvidianews.nvidia.com/news/nvidia-and-softbank-group-announce-termination-of-nvidias-acquisition-of-arm-limited | beursmelding |

---

## 15. Update-historie

| Datum | Versie | Wijziging |
|---|---|---|
| 2026-05-14 | 1.0 | Eerste publicatie |

---

## Opmerkingen voor Claude Code

Een aantal observaties die niet in de standaard-secties pasten en die voor de stage-2 conversie van belang kunnen zijn:

1. **Bronnen-eis recente 5 jaren.** METHODE.md eist dat de 5 meest recente jaren (FY22-FY26) allemaal HOOG zijn (jaarverslag-PDF of IR-pagina rechtstreeks geopend). Voor deze analyse heb ik FY24/FY25/FY26 direct uit het NVIDIA persbericht (HOOG), maar FY22 en FY23 uit StockAnalysis-aggregator (AGGREGATOR). De stage-2 validator zal hier vermoedelijk een waarschuwing of fail produceren. Aanbeveling: laat Claude Code de 10-K's voor FY22 en FY23 direct uit SEC EDGAR ophalen en deze rij in de bronnentabel upgraden naar HOOG. URL's: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001045810&type=10-K&dateb=&owner=include&count=40

2. **EBIT-marge "cyclische" check.** NVIDIA is geen klassiek cyclisch bedrijf (bouw/staal/olie), maar de AI-capex-cyclus heeft wel een sterke cyclische component. De 5-jaars EBIT-marge varieert tussen 15,7% (FY23) en 62,4% (FY25), dat is een spread van factor 4. De genormaliseerde EBIT-marge van 46% (gem. FY22-FY26) is mijn beste poging om de cyclus uit te middelen, maar Claude Code kan willen overwegen of een 8-10 jaars marge (inclusief FY19/FY20-pre-AI-boom) eerlijker is voor terminal-aannames. Een meerjaarse gemiddelde marge inclusief FY19/FY20 zou rond 38-40% liggen, wat de EPV en terminal-value naar beneden trekt.

3. **SBC vanaf FY27.** NVIDIA stopt vanaf Q1 FY27 met het uitsluiten van SBC uit non-GAAP. Dit verandert hoe analist-consensus eruitziet voor FY27-FY28 vs. historische FY25/FY26 — vergelijkbaarheid breekt. Houdt hier rekening mee bij het JSON-veld `non_gaap_gebruikt` en `non_gaap_toelichting`.

4. **Insider net seller (15:0 ratio).** Het volledige ontbreken van open-markt-aankopen door insiders is een opvallende observatie, gegeven dat Jensen Huang persoonlijk circa 3,5% van het bedrijf bezit. In het JSON-veld `insider_netto` zet ik dit op NETTO VERKOPER, maar het is mogelijk dat de validator een meer genuanceerde waarde verwacht.

5. **Aandelen uitstaand-conventie.** Door de 10-voor-1 splitsing van 7 juni 2024 zijn historische EPS-cijfers (FY19/FY20: $6,63 / $4,52 pre-split) niet vergelijkbaar met de huidige EPS van $4,90. In de tabel heb ik historische cijfers gegeven in pre-split-vorm met voetnoot; voor de JSON-output beveel ik aan dat alle historische EPS gerekonciliëerd worden naar post-split-basis (delen door 10), zoals NVIDIA zelf doet in het FY25-persbericht.

6. **Optimistisch scenario blijft negatief.** Het is opvallend dat zelfs het optimistische DCF-scenario (g1=30%, WACC=13,4%) een fair value van $151,87 oplevert — nog steeds 28,4% downside op $212. Dit illustreert hoe extreem de huidige markwaardering is. Het zou stage-2-validator-vriendelijk zijn om dit expliciet te benoemen in de samenvatting van de scorekaart, wat ik gedaan heb.

7. **Datum-stempel kwartaalresultaten.** Q1 FY27 earnings is 20 mei 2026 (zes dagen na de peildatum van deze analyse). Als Janco de analyse daarna wil refreshen, zijn die cijfers natuurlijk een directe trigger.

8. **WACC en hoge beta.** De berekende WACC van 13,91% is hoog door beta 2,24. Bij DCF voor mega-caps is dit gerechtvaardigd door de waargenomen volatiliteit, maar Damodaran-sector-WACC voor halfgeleiders is rond 10,5%. Stage-2 kan het waardevol vinden om gevoeligheid van DCF op WACC=10,5% (sector-WACC) als alternatief scenario te draaien — dat brengt basis-FV tot circa $115-120, nog steeds onder de huidige koers maar minder negatief.

9. **Netto-kaspositie inclusief LT investments.** Ik heb conservatief net-cash van $51,5 mld gebruikt (cash + ST-invest minus bruto schuld). Bij toepassing van $73,8 mld (incl. LT-invest) stijgen alle FV-cijfers met circa $0,90 per aandeel. Materieel niet groot maar consistent dataveld.
