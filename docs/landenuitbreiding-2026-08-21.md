# Welke landen zijn het waard om aan de screener toe te voegen?

Onderzoek van 21 augustus 2026. Uitgangspunt: de screener draait op 2.812
tickers, vrijwel volledig Europees, haalt alles bij Yahoo via yfinance, en heeft
per aandeel minimaal drie jaar jaarcijfers nodig. Broker: DEGIRO en Interactive
Brokers. Bronbelasting weegt mee maar is niet doorslaggevend.

Een land moet door vier zeven vallen voordat het zinvol is:

1. **Koopbaar** via DEGIRO of IBKR voor een Nederlandse particulier.
2. **Screenbaar**: Yahoo levert bruikbare jaarcijfers, ook voor smallcaps.
3. **De moeite waard**: goedkoop ten opzichte van de eigen historie, genoeg
   bedrijven, en liefst een katalysator die waarde vrijmaakt.
4. **Fiscaal draaglijk**: wat er aan de bron wordt ingehouden is grotendeels
   verrekenbaar in box 3.

Van de 25 onderzochte landen halen er zeven alle vier. Die staan hieronder in
volgorde van wat ze opleveren.

---

## Samenvatting: wat ik zou toevoegen

| Prioriteit | Markt | Suffix | Bedrijven | Waarom | Grootste bezwaar |
|---|---|---|---:|---|---|
| **1** | Verenigd Koninkrijk (verdiepen) | `.L` | ~1.950 | Je hebt er nu 44. 0% bronbelasting, overnamegolf als realisatiekanaal | Duur t.o.v. eigen historie (80e pct) |
| **2** | Japan | `.T` | 3.713 | Diepste smallcapmarkt ter wereld, neutraal gewaardeerd, TSE-hervorming werkt | 5,3% fiscaal lek; yen-risico |
| **3** | Hongkong | `.HK` | 2.802 | Laagste koers/boekwaarde ter wereld (1,11), **0% bronbelasting** | Governance, politiek risico |
| **4** | Zuid-Korea | `.KS` `.KQ` | ~2.600 | 75% van de KOSPI onder boekwaarde, harde katalysator in november | Alleen via IBKR; 7% lek |
| **5** | Canada | `.TO` `.V` | ~3.700 | Je hebt er nu 1. DEGIRO, 15% verrekenbaar | Historisch duur (82e pct) |
| **6** | Australië | `.AX` | 1.839 | Je hebt er nu 2. Gefrankeerd dividend = **0% bronbelasting** | Duur; 25% smallcaps zonder omzet |
| **7** | Singapore | `.SI` | ~530 | **0% bronbelasting**, S$6,5 mrd overheidsprogramma voor small/midcaps | Dun universum |

**Verdiepen binnen wat je al hebt:** Zwitserland (37 van 241), Griekenland
(6 van 138), Duitsland (368, Xetra heeft er meer), Polen NewConnect (~340 extra,
zelfde `.WA`-suffix).

**Niet doen:** Indonesië, Israël, Thailand, Turkije, India, Taiwan, en alles wat
je niet kunt kopen (Maleisië, Vietnam, Chili, Mexico, Zuid-Afrika, Saoedi-Arabië).
Redenen staan onderaan.

---

## 1. Verenigd Koninkrijk — de grootste quick win

Je hebt 44 Britse aandelen. De LSE heeft er **1.993** (1.119 hoofdmarkt, 819
AIM). Dat is geen nieuw land toevoegen maar een gat dichten in een markt die je
al ondersteunt, met een broker die je al hebt.

- **Fiscaal het schoonste land in het onderzoek:** 0% bronbelasting op gewoon
  dividend. Uitzondering: Property Income Distributions van REITs, 20% (naar 22%
  vanaf 6 april 2027).
- Marktbrede P/E 16,4× tegen een driejaars gemiddelde van 17,9×.
- Overnames zijn inmiddels de grootste oorzaak van AIM-delistings. Voor een
  waardebelegger is dat geen probleem maar precies het mechanisme dat de
  korting verzilvert.
- Kosten: 0,5% stamp duty bij aankoop. Sinds Autumn Budget 2025 vrijgesteld voor
  bedrijven die vanaf 27 november 2025 nieuw noteren, gedurende drie jaar.

**Bezwaar:** de CAPE staat op het 80e percentiel van de eigen historie. Het VK is
goedkoop tegenover de VS, niet tegenover zichzelf. Veel Britse "value" is
cyclisch of in structureel verval.

**Valkuil die je moet afvangen:** van de 1.015 `.L`-symbolen in het
Yahoo-universum zijn er **516 van de vorm `0XXX.L`** — dat zijn geen Britse
bedrijven maar LSE-lijnen voor buitenlandse aandelen (`0R2M.L` = Regeneron,
`0HJI.L` = ADP). Eén dag koershistorie, nul jaarcijfers. Filter
`^0[A-Z0-9]{3}\.L` weg vóór import. Doe je dat niet, dan verklaart dat in één
klap waarom het VK er in je dashboard slecht uitziet.

Tickerlijst: [LSE Instrument list.xlsx](https://docs.londonstockexchange.com/sites/default/files/reports/Instrument%20list.xlsx)
— de beste lijst uit het hele onderzoek: TIDM, ISIN, ICB-sector, handelsvaluta
en Main/AIM in één bestand. Gebruik níet de Issuer list, die heeft geen ticker.

---

## 2. Japan — het diepste bruikbare universum ter wereld

3.713 binnenlandse noteringen: Prime 1.558, Standard 1.559, Growth 596. Dat is
groter dan je hele huidige screener.

- **Yahoo-dekking gemeten en uitstekend.** In een steekproef van 64 Japanse
  aandelen, inclusief smallcaps: 0% zonder jaarcijfers, mediaan 5 jaar historie,
  EBITDA/vrije kasstroom/schuld 100% aanwezig.
- CAPE 28,3 op het 47,5e percentiel van de eigen historie — neutraal, niet duur.
- De TSE-hervorming is de langstlopende en meest bewezen ter wereld. Inkoop eigen
  aandelen bereikte **¥16,2 biljoen in januari–mei 2026, +34% j-o-j**, gedreven
  door de ontvlechting van kruisparticipaties. Activisme op recordniveau.
  Sinds april 2025 is gelijktijdige Engelstalige resultatenpublicatie verplicht.
- Beschikbaar bij **zowel DEGIRO als IBKR**.

**Fiscaal:** 15,315% wordt ingehouden, het verdrag met Nederland staat 10% toe.
Om die 10% te krijgen moet vooraf Form 1-2 via een Japanse betaalagent worden
ingediend; bij een buitenlandse broker gebeurt dat in de praktijk niet. **Reken
op een structureel lek van 5,3%** dat je noch aan de bron krijgt, noch in box 3
mag verrekenen. Bij 2% dividendrendement is dat ruim 0,1% per jaar — vervelend,
niet fataal.

**Bezwaren.** De makkelijke winst is eruit: een simpele "koers onder boekwaarde"-
screen leverde in 2023 goud op en levert nu vooral bedrijven op die om goede
redenen goedkoop zijn. En de TSE verscherpt de Growth Market-eisen per 2030 naar
¥10 mrd beurswaarde; ongeveer **70% van de 610 Growth-bedrijven** zat daar eind
2024 onder. Dat is tegelijk een delistingsrisico en een dwingende reden tot
overname of herstructurering — maar je moet weten in welke van de twee je zit.

Tickerlijst: [JPX data_j.xls](https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls),
werkt vanaf een server. Bevat sectorcodes en TOPIX-groottecategorie, geen ISIN.
Filter de 476 ETF/ETN's en 185 PRO Market-namen eruit.

---

## 3. Hongkong — de laagste boekwaardering, en fiscaal gratis

- **MSCI Hongkong noteert op koers/boekwaarde 1,11** — de laagste in het hele
  onderzoek. Marktbrede P/E 11,5× tegen een driejaars gemiddelde van 12,7×.
  CAPE op het 41,6e percentiel.
- 2.802 aandelen (Main Board 2.465, GEM 307).
- **0% bronbelasting op dividend.** Alleen 0,1% stamp duty per kant.
- Yahoo-dekking gemeten: 0% zonder jaarcijfers, alle velden 100%.
- Beschikbaar bij **DEGIRO en IBKR**.
- Tickerlijst met ISIN: [HKEX ListOfSecurities.xlsx](https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx).
  Filter op category, anders krijg je er 7.315 warrants en 5.951 CBBC's bij.

**Bezwaren, en ze zijn reëel.** Politiek en kapitaalrisico zijn niet te
modelleren en dus ook niet in je fair value te verwerken. Het smallcap-segment in
Hongkong bevat historisch veel lege hulzen met bedenkelijke governance — precies
waar een lage-koers/boekwaarde-screen naartoe trekt. En een deel van die lage
P/B is een terechte korting voor vastgoedexposure waarvan de boekwaarde niet
realiseerbaar is. Als je Hongkong toevoegt, zet er dan een strengere
kwaliteitseis op dan je elders hanteert.

**Goed nieuws voor je code:** veel Hongkongse bedrijven handelen in HKD maar
rapporteren in CNY. Je `data_fetcher` rekent `financialCurrency` al om naar
handelsvaluta (dezelfde route als AUTO.OL die in USD rapporteert), dus die
valkuil heb je al dichtgetimmerd.

---

## 4. Zuid-Korea — de interessantste paradox van 2026

Dit is de opvallendste bevinding uit het onderzoek, en hij vraagt uitleg omdat de
cijfers elkaar lijken tegen te spreken.

**Op indexniveau is Korea spotduur:** CAPE 33,9 op het **99,2e percentiel** van
de eigen historie. Verklaring: Samsung Electronics en SK Hynix stegen beide ruim
100% in 2026 en waren samen goed voor de helft van de marktkapitalisatiewinst.

**Op aandeelniveau is Korea spotgoedkoop:** op 29 juli 2026 noteerde **75% van
de KOSPI-fondsen onder boekwaarde — 600 van de 802 bedrijven**, opgelopen van 66%
in januari. De marktbrede P/E is 11,6× tegen een driejaars gemiddelde van 19,6×.
Gemiddelde P/B over het afgelopen decennium: 0,99, tegen 2,65 voor 23
ontwikkelde economieën.

Dat verschil ís de kans voor een screener die juist niet in de index zoekt. En er
is een datum:

- **6 maart 2026:** verplichte intrekking van ingekochte eigen aandelen binnen
  één jaar, met boetes tot KRW 50 mln. In maart alleen al trokken 102
  beursfondsen KRW 15,8 biljoen in, +159% j-o-j.
- **2 november 2026:** Korea Exchange gaat laag-PBR-bedrijven publiek benoemen —
  de onderste 25% per sector op KOSPI, onderste 10% op KOSDAQ, op driejaars-PBR.
  Naar schatting 120 tot 220 bedrijven, met een label in de handelssystemen van
  brokers en gesprekken met de toezichthouder.

Met andere woorden: de beursorganisator publiceert over tien weken zelf een lijst
die sterk lijkt op wat jouw screener zou moeten vinden. Dat is een katalysator
die je in geen enkele andere markt krijgt.

- 2.671 genoteerde bedrijven, `.KS` (KOSPI ~850) en `.KQ` (KOSDAQ ~1.780).
- Yahoo-dekking: 0% zonder jaarcijfers, 96% screenbaar.
- **Alleen via Interactive Brokers** — live sinds 7 mei 2026, ruim 2.700 effecten.
  DEGIRO biedt Korea niet.

**Bezwaren.** Je koopt tegen de stroom in van een markt die net een historische
rally achter de rug heeft; de volatiliteit is extreem (circuit breakers op 28 en
29 juli, daarna op 31 juli de grootste dagwinst ooit: +17,91%). Laag-PBR-
bedrijven in Korea zijn vaak laag-PBR *omdat* de controlerende familie er belang
bij heeft — ruim 60% heeft een ROE onder 7%, en naming-and-shaming hoeft daar
niets aan te veranderen. Fiscaal 22% ingehouden tegen 15% verdrag: **7% lek**,
en per 1 januari 2026 is relief-at-source administratief zwaarder geworden.

**Praktisch:** Koreaanse tickers zijn zescijferig met voorloopnullen. Bewaar ze
als string, anders verlies je `005930.KS`.

---

## 5, 6, 7. Canada, Australië, Singapore — goedkoop toe te voegen

Deze drie ondersteun je technisch al (`.TO`, `.AX` staan in `SUFFIX_INFO`) maar
je hebt er samen 3 aandelen in. Ze zijn niet de meest kansrijke markten, maar de
marginale kosten van toevoegen zijn laag.

**Canada** — ~3.700 noteringen over vier beurzen: `.TO` (TSX 2.264), `.V` (TSXV
1.501), en ook `.CN` en `.NE`, die je nu niet kent. DEGIRO biedt TSX én TSXV.
25% ingehouden, 15% onder verdrag mits de broker NR301-documentatie aanlevert.
*Bezwaar:* CAPE op het 82e percentiel, dus historisch duur. *Valkuil:* de 133
CDR's op TSX en de Cboe `.NE`-lijnen zijn Amerikaanse bedrijven in Canadese
dollars (`NVDA.NE`) — dubbeltellingen.
Lijst: [TMX resource/en/571](https://www.tsx.com/resource/en/571), twee tabbladen.

**Australië** — 1.839 noteringen. Fiscaal opvallend gunstig: **volledig
gefrankeerd dividend is vrijgesteld van bronbelasting, 0%.** Ongefrankeerd 30%,
15% onder verdrag. Banken en binnenlandse industrie keren meestal volledig
gefrankeerd uit; REITs en bedrijven met buitenlandse winst niet. DEGIRO en IBKR.
*Bezwaar:* CAPE op het 79e percentiel, en 25% van de smallcaps zijn
mijnbouwexplorers zónder omzet. Dat is geen datafout maar het breekt wel elke
omzet- of EBITDA-screen. Zet er een harde omzetfilter op.
Lijst: [ASX directory CSV](https://asx.api.markitdigital.com/asx-research/1.0/companies/directory/file?access_token=83ff96335c2d45a094df02a206a39ff4),
met GICS en marktkap.

**Singapore** — ~530 noteringen, dus dun. Maar: **0% bronbelasting** (one-tier
stelsel), goede accountingkwaliteit, DEGIRO én IBKR, en een gefinancierde
katalysator: het Equity Market Development Programme van S$5 mrd, in Budget 2026
uitgebreid met S$1,5 mrd, **expliciet gericht op small- en midcaps**, plus een
S$30 mln "Value Unlock"-pakket. *Bezwaar:* CAPE op het 57e percentiel — niet
goedkoop, en met 530 namen blijft de vangst klein.
Lijst: [SGX API](https://api.sgx.com/securities/v1.1?excludetypes=bonds&params=nc%2Cadjusted-vwap).

---

## Verdiepen binnen wat je al hebt

| Markt | Nu | Beschikbaar | Opmerking |
|---|---:|---:|---|
| Zwitserland `.SW` | 37 | 241 issuers (205 Zwitsers) | Duur (74e pct, P/B 4,21) en **20% fiscaal lek**: 35% ingehouden, 15% verdrag, terugvragen bij de ESTV en DEGIRO helpt daar expliciet niet bij. Lage prioriteit. Wel goed nieuws: de Zwitserse stamp duty geldt niet via een buitenlandse broker. |
| Griekenland `.AT` | 6 | 138 | 5% bronbelasting, volledig verrekenbaar. Sinds de Euronext-overname dezelfde infrastructuur als Amsterdam. MSCI-opwaardering naar Developed per mei 2027. *Maar:* de index is voor 49% bank, en de marktbrede P/E (13,7×) ligt bóven het driejaars gemiddelde (10,2×). |
| Duitsland `.DE` | 368 | Meer via Xetra | [Xetra CSV met ISIN](https://www.xetra.com/resource/blob/1528/8a9cbe9b1b1c1cb0b6b0d5cd5a5c1c0e/data/t7-xetr-allTradableInstruments.csv). Pas op: Duitsland heeft zeven suffixen (`.DE .F .SG .MU .DU .HM .HA`) en duizenden buitenlandse noteringen. Yahoo-regio `de` geeft 15.589 symbolen — dat zijn geen 15.589 Duitse bedrijven. |
| Polen NewConnect | in `.WA` | ~340 extra | NewConnect en de hoofdmarkt delen `.WA`, dus je kunt ze niet op suffix scheiden — alleen via de GPW- en NewConnect-lijsten zelf. Yahoo levert er wél jaarcijfers voor. |

---

## Wat je niet moet toevoegen, en waarom

**Indonesië en Israël — Yahoo heeft de cijfers niet.** Dit is empirisch gemeten,
niet geschat. Van een steekproef van 64 aandelen per markt had **Indonesië 30%**
en **Israël 23%** wél koershistorie maar een volledig lege winst- en
verliesrekening. In alle 21 andere onderzochte markten was dat 0–5%. Het gat zit
precies in het smallcap-segment waar een screener zijn waarde haalt. Indonesië
heeft er bovendien een acuut probleem bij: MSCI onderzoekt afwaardering naar
Frontier wegens gecoördineerde koersmanipulatie tussen gelieerde partijen,
besluit in november 2026, geschatte uitstroom US$13 mrd.

**Thailand — drie grote accountingfraudes in twee jaar.** Stark Corporation
(~40 mrd baht schade, ruim 8 mrd baht fictieve omzet), Thonburi Healthcare
(7,6 mrd baht, vijf verzonnen ziekenhuisprojecten, oprichter gevlucht), JKN
Global (gedelisted december 2025). De SET staat op een zevenjarig dieptepunt en
buitenlanders kopen liever Non-Voting Depository Receipts dan echte aandelen.
Bovendien niet koopbaar via DEGIRO of IBKR.

**Turkije — de cijfers zijn niet vergelijkbaar.** Inflatie 31,75% in juli 2026 en
IAS 29-hyperinflatieboekhouding. Boekwaardes worden opgewaardeerd naar koopkracht
per balansdatum, wat de P/B optisch verlaagt zonder dat er economisch iets
gebeurt. Dat de marktbrede P/E (16,6×) bóven het driejaars gemiddelde (14,3×)
ligt terwijl de CAPE op het 16e percentiel staat, is precies die vervorming. Voor
een mechanische screener onbruikbaar.

**India — de toegang, niet de belasting.** Een buitenlandse particulier moet zich
registreren als Categorie III FPI via een Designated Depository Participant, met
demat-account en KYC in India. Niet praktisch. Bovendien de duurste grote
opkomende markt op boekwaarde (P/B 3,23, CAPE 82e percentiel).

**Taiwan — te duur.** CAPE 49,5 op het 99,2e percentiel, P/B 4,29. Data en
toegang zijn prima (IBKR biedt sinds november 2025 ook TPEx), maar er is niets te
halen. Bovendien 21% ingehouden tegen 10% verdrag: 11% lek.

**Niet koopbaar via DEGIRO of IBKR**, hoe aantrekkelijk het scherm er ook uitziet:
Maleisië (CAPE 31e pct, 1.077 noteringen, 0% bronbelasting), Vietnam (P/E 9,9×,
FTSE-opwaardering per 21 september 2026), Chili, Mexico, Zuid-Afrika (marktbrede
P/E 8,6×, maar de JSE krimpt van 500+ naar ~280 noteringen), Saoedi-Arabië
(QFI-regime afgeschaft per 1 februari 2026, maar handelen vereist een Saoedische
broker). Alleen via ETF's te spelen, en daarmee zinloos voor een screener op
individuele aandelen.

**Brazilië — het randgeval.** Goedkoopste grote markt (marktbrede P/E 8,3×, CAPE
23e pct), IBKR-toegang sinds 11 december 2025, Yahoo-dekking prima, en de nieuwe
bronbelasting van 10% per 1 januari 2026 is volledig verrekenbaar in box 3. Maar:
verkiezingen op 4 oktober 2026 met R$32 mrd buitenlandse uitstroom sinds april,
Selic op 14,75%, en met 496 noteringen smaller dan het land suggereert. Als je
één gok wilt nemen, is dit hem — maar niet vóór oktober.

---

## De rem waar je tegenaan gaat lopen

Dit is het praktische probleem waar geen enkel land iets aan verandert.

Je haalt **250 jaarcijfers per nacht** op. Met 2.812 tickers duurt een volledige
ronde 11 dagen. Voeg je Japan, Hongkong en Korea toe, dan kom je op ongeveer
**12.000 tickers en 48 nachten per ronde** — dan zijn je jaarcijfers gemiddeld
bijna twee maanden oud en werkt het model op verouderde data.

Drie manieren eruit, in volgorde van voorkeur:

1. **Importeer selectief.** Neem in Japan alleen Prime en Standard (3.117 i.p.v.
   3.713), in Hongkong alleen de Main Board (2.465), in Korea alleen KOSPI (850).
   Of leg er een ondergrens op marktkapitalisatie onder — bedrijven van onder de
   €20 mln koop je toch niet.
2. **Splits de rotatie.** Jaarcijfers veranderen per kwartaal, maar niet voor
   iedereen tegelijk. Ververs aandelen met een recente rapportagedatum vaker en
   de rest langzamer, in plaats van iedereen even vaak.
3. **Fasegewijs invoeren.** Voeg één markt per keer toe en meet wat de
   verversing doet voordat je de volgende erbij zet.

Ik zou beginnen met het VK — dat is het kleinste stuk werk met het schoonste
fiscale profiel — en pas daarna Japan.

---

## Wat je in de code moet aanpassen

`engine/markets.py`, `SUFFIX_INFO` uitbreiden:

```python
"T":   ("JPY", "JP"),   # Tokyo
"HK":  ("HKD", "HK"),   # Hongkong
"KS":  ("KRW", "KR"),   # KOSPI
"KQ":  ("KRW", "KR"),   # KOSDAQ
"SI":  ("SGD", "SG"),   # Singapore
"CN":  ("CAD", "CA"),   # Canadian Securities Exchange
"NE":  ("CAD", "CA"),   # Cboe Canada
"SA":  ("BRL", "BR"),   # B3, als je Brazilië doet
```

Verder:

- **Filter `^0[A-Z0-9]{3}\.L` weg** bij de VK-import. Ook `.XC` en `.IL` zijn
  buitenlandse lijnen op de LSE — die geven wél jaarcijfers, maar van het
  buitenlandse moederbedrijf, dus het zijn duplicaten.
- **Dedupliceer op ISIN, niet op ticker.** `SAP.DE`, `SAP.F`, `SAP.MU` en
  `SAP.SG` geven alle vier dezelfde omzet van €36,8 mrd; `BHP.AX` en `BHP` ook.
  Zonder deduplicatie verschijnt één bedrijf meerdere keren in je kansenlijst.
- **Pence-achtige valuta's.** Je vangt `GBp`/`GBX` al af met de deling door 100.
  Zou je ooit Zuid-Afrika (`ZAc`) of Israël (`ILA`) toevoegen, dan geldt daar
  hetzelfde — beide raad ik af, maar noteer het als bekende val.
- **Rapportagevaluta:** dat handelt `data_fetcher` al af via `financialCurrency`
  en `_fx_rate`. Voor Hongkong (HKD-koers, CNY-cijfers) is dat precies wat je
  nodig hebt.
- **Één gedeelde HTTP-sessie voor alle Yahoo-aanroepen.** Per-thread sessies
  geven HTTP 401 op `.info` — je krijgt dan lege sector en valuta terwijl
  `.financials` gewoon doorkomt. Dat leest als ontbrekende data terwijl het een
  authenticatiefout is, en het kostte in dit onderzoek een hele meetronde.

**Let op bij de tickerlijsten:** de officiële downloads van LSE, JPX, HKEX, SGX,
SIX, TWSE, TMX, ASX en B3 werken vanaf een server. Die van NSE India, KRX, IDX,
TASE, Bursa Malaysia, SET, Saudi Exchange, GPW en Borsa İstanbul geven 403 vanaf
een datacenter-IP. Voor Korea betekent dat een handmatige maandelijkse download
of een headless browser.

---

## Fiscale samenvatting voor de zeven

Vuistregel: reken op het **statutaire** tarief als feitelijke inhouding en op het
**verdragstarief** als maximale verrekening in box 3. Het verschil is permanent
verlies tenzij je zelf een terugvraagprocedure start. De tweede box 3-limiet
(verrekening niet hoger dan de verschuldigde belasting) knelt bij een normale
aandelenportefeuille vrijwel nooit.

| Markt | Ingehouden | Verrekenbaar | Lek | Extra kosten |
|---|---:|---:|---:|---|
| Verenigd Koninkrijk | 0% | — | **0%** | 0,5% stamp duty bij aankoop |
| Hongkong | 0% | — | **0%** | 0,1% stamp duty per kant |
| Singapore | 0% | — | **0%** | — |
| Australië (gefrankeerd) | 0% | — | **0%** | — |
| Australië (ongefrankeerd) | 15% | 15% | 0% | — |
| Canada | 15%¹ | 15% | 0%¹ | — |
| Japan | 15,315% | 10% | **5,3%** | — |
| Zuid-Korea | 22% | 15% | **7%** | — |

¹ Mits je broker NR301-documentatie aanlevert; zonder dat 25% en dan 10% lek.

---

## Bronnen

Belastingen: [PwC Worldwide Tax Summaries](https://taxsummaries.pwc.com/) (per
land), [NTA Japan — Application Form for Income Tax Convention](https://www.nta.go.jp/english/taxes/withholing/tax_convention.htm),
[Belastingdienst — teruggaaf of vrijstelling buitenlandse bronbelasting](https://www.belastingdienst.nl/wps/wcm/connect/bldcontentnl/belastingdienst/zakelijk/internationaal/vermogen/teruggaaf_of_vrijstelling_van_buitenlandse_bronbelasting/formulieren),
[TaxLive — bronbelasting maximaal tot verdragstarief verrekenbaar](https://www.taxlive.nl/nl/documenten/vn-vandaag/buitenlandse-bronbelasting-voor-maximaal-15-verrekenbaar-ook-als-meer-is-ingehouden/),
[DEGIRO — wettelijke bronbelastingtarieven](https://www.degiro.nl/helpdesk/belasting/bronbelasting/welk-wettelijk-bronbelastingtarief-van-toepassing-op-de-dividendinkomsten),
[Deloitte Taxscape — SDRT UK Listing Relief](https://taxscape.deloitte.com/measures-autumn-budget-2025/stamp-duty-reserve-tax-uk-listing-relief.aspx),
[EY — Braziliaanse bronbelasting op dividend](https://www.ey.com/en_gl/technical/tax-alerts/brazilian-tax-authority-issues-guidance-on-new-withholding-tax-on-dividends-paid-to-nonresidents).

Waarderingen: [Siblis Research — CAPE per land](https://siblisresearch.com/data/cape-ratios-by-country/),
[PortfolioLab — goedkoopste en duurste markten 2026](https://www.portfoliolab.app/blog/cheapest-expensive-stock-markets-2026)
(Research Affiliates, 31-07-2026), iShares/Global X factsheets per land,
[Simply Wall St marktpagina's](https://simplywall.st/markets/kr).

Hervormingen: [JPX — cost of capital and stock price](https://www.jpx.co.jp/english/equities/follow-up/02.html),
[Japanse inkoop eigen aandelen recordhoogte](https://www.marketscreener.com/news/japan-share-buybacks-hit-record-16-2-trillion-yen-in-early-2026-ce7f5fdbdb8ff321),
[75% van KOSPI onder boekwaarde](https://en.sedaily.com/finance/2026/07/29/75-percent-of-kospi-stocks-trade-below-book-value-as),
[Korea benoemt laag-PBR-aandelen vanaf november](https://finance.biggo.com/news/ff8e5fa4-7d62-4883-a452-73e72bbdae82),
[Koreaanse regels voor eigen aandelen — Sodali](https://sodali.com/resources/insights/koreas-new-treasury-share-rules-and-what-they-mean-for-companies),
[MAS — uitbreiding Equity Market Development Programme](https://www.mas.gov.sg/news/media-releases/2026/mas-announces-expansion-of-equity-market-development-programme),
[MSCI 2026 Market Classification Review](https://ir.msci.com/news-releases/news-release-details/msci-announces-results-msci-2026-market-classification-review),
[The Nation — accountingschandalen Thailand](https://www.nationthailand.com/business/banking-finance/40060447).

Toegang: [DEGIRO — beurzen](https://www.degiro.nl/beurzen),
[IBKR opent Koreaanse aandelen](https://www.interactivebrokers.com/en/general/about/mediaRelations/5-7-26.php),
[IBKR voegt B3 toe](https://www.interactivebrokers.com/en/general/about/mediaRelations/12-11-25.php).

Datakwaliteit: gemeten op 21-08-2026 met yfinance 1.6 tegen de live Yahoo-API,
steekproeven van 24 tot 64 aandelen per markt, bottom-up op marktkapitalisatie
zodat smallcaps meetellen. Verschillen van 92% versus 100% zijn ruis; het gat van
23–30% bij Indonesië en Israël is significant.

---

## Wat ik niet heb kunnen dichtzetten

Eerlijkheidshalve, zodat je weet waar de zachte plekken zitten:

- **Of IBKR relief-at-source toepast per markt.** Er is geen publieke
  IBKR-matrix. De Japanse 5,3% en Koreaanse 7% zijn mijn verwachting op basis van
  hoe de betaalketen normaal werkt, niet een bevestigd afschrift. Controleer het
  op je eerste echte dividend.
- **Of de bedragen die Yahoo teruggeeft kloppen**, niet alleen of de regels
  bestaan. Vooral EBITDA en vrije kasstroom zijn afgeleide velden. Voor banken en
  verzekeraars zou ik daar in geen enkele markt blind op vertrouwen.
- **De marktbrede P/E voor Japan** die Simply Wall St geeft (13,7×) strookt niet
  met hun eigen driejaars gemiddelde van 16,7×. Die ene waarde heb ik laten
  vallen; het CAPE-percentiel van 47,5 staat wel.
