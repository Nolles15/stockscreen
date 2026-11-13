# Kritische review november 2026 — stap 1: het waarderingsmodel

Datum: 2026-11-13. Read-only: code gelezen op commit van 10 november, cijfers uit
de live site (`/api/dashboard`, `/api/dashboard/selectie?tab=kansen`, `/api/trace`).
Er is niets gewijzigd.

Gelezen: `valuation.py`, `normalizer.py`, `screener.py`, `quality_score.py`,
`moat_profile.py`, `groei_profiel.py`, `config.yaml`, de TTM- en multiple-code in
`data_fetcher.py`, `_add_rank_scores` in `app.py`, `docs/REVIEW-GEVOELIGHEID.md`
en `docs/groeiprofiel-ijking.md`.
Niet regel voor regel gelezen: `exit_regels.py` (alleen ontwerp en constanten),
`scorebord.py`, `besluiten.py`, `groei_backtest.py`.

## Stand van het universum (13 november)

| | aantal |
|---|---:|
| Actieve tickers | 4.012 |
| Met een oordeel (FV + signaal) | 2.950 (74%) |
| Geen oordeel | 1.062 (26%) |
| HOLD / SELL | 2.044 / 859 |
| BUY + STRONG BUY | 47 (1,6% van beoordeeld) |
| Vertrouwen in FV: laag / middel / hoog | 1.681 / 996 / 273 |
| Kwaliteitsscore mediaan | 3,5 (328 halen ≥ 7, 65 halen ≥ 9) |

## A. Fouten in de code

### A1. De historische bedrijfswaarde is verkeerd berekend — `data_fetcher.py:777`

```python
net_debt = (total_debt or 0) - max(0, -(net_cash or 0))
```

`net_cash` is kas min schuld. Uitgeschreven komt deze regel neer op
`net_debt = min(schuld, kas)`:

| schuld | kas | code geeft | juist |
|---:|---:|---:|---:|
| 100 | 20 | 20 | 80 |
| 20 | 100 | 20 | −80 |

De historische EV is dus beurswaarde + het kleinste van schuld en kas. Daaruit
komen de eigen historische EV/EBITDA en EV/FCF, die voor 65% het multiple-anker
vormen van twee van de vier multiple-methodes. In `valuation.py` wordt daarna de
échte nettoschuld afgetrokken. Gevolg: bij bedrijven met nettoschuld is de
historische multiple te laag en valt de fair value te laag uit; bij kasrijke
bedrijven andersom. De omvang is niet gemeten; dat kan met een herberekening van
`historical_multiples` en `POST /api/scores/recompute` met `dry_run`.

### A2. Historische koersen zijn waarschijnlijk dividend-gecorrigeerd — `data_fetcher.py:749`

`t.history(period="5y", interval="1mo")` staat in yfinance standaard op
`auto_adjust=True`. De slotkoers van vier jaar terug is dan verlaagd met alle
sindsdien betaalde dividenden, en de historische K/W en K/B vallen te laag uit
bij dividendbetalers (bij 4–5% dividend loopt dat op tot 15–20% voor het oudste
jaar). Niet gemeten; te toetsen door één ticker met en zonder de vlag op te halen.

Daarnaast pakt `_historical_year_end_price` de laatste maand van het
**kalender**jaar. Bij een gebroken boekjaar (Games Workshop, boekjaar tot eind
mei) wordt de winst van FY2026 dus gedeeld op de koers van eind 2026.

### A3. De F-score vergelijkt de TTM-rij met het laatste boekjaar

`quality_score` krijgt `calc_rows`, met de TTM-rij vooraan. `piotroski_fscore`
neemt `rows[0]` als "dit jaar" en `rows[1]` als "vorig jaar". Vijf van de negen
toetsen zijn "is gestegen/gedaald"-toetsen, en die vergelijken dan twee perioden
die tot negen maanden overlappen. Bij ASML staat nu TTM tegen FY2025; bij Infotel
(geen TTM-rij) FY2025 tegen FY2024. Dezelfde toets meet per aandeel iets anders.
De review van augustus zag het symptoom al (BETS-B: 10,0 zonder en 6,5 met TTM).

### A4. Het stabiliteitscriterium bestraft groei

Criterium 3 eist een variatiecoëfficiënt onder 20% rond het **gemiddelde**. Een
bedrijf dat vijf jaar lang exact 15% per jaar groeit heeft een variatiecoëfficiënt
van 0,20 en zakt. De maat meet dus groei én volatiliteit door elkaar. Stabiliteit
hoort gemeten te worden rond de trend.

### A5. Kleinere punten

- ROIC bij nettokas: `eigen vermogen + schuld − max(0, net_cash)` trekt
  `kas − schuld` af, zodat de schuld dubbel in de noemer staat. Effect klein,
  want bij nettokas is de schuld meestal klein.
- Rentelasten onbekend → rentedekking telt als gehaald (een gratis halve punt).
- Cijfers in een andere rapportagevaluta worden voor alle jaren met de koers van
  vandaag omgerekend; de historische multiples delen dan een koers van toen door
  een winst tegen de wisselkoers van nu.
- Silvano staat nog met twee noteringen in de top vijf van Kansen (bewust
  "markeren, niet wegfilteren", maar het kost een van de twintig plekken).

## B. Structurele zwaktes van het model

### B1. Het zijn geen drie methodes maar één

- Graham met de huidige instellingen is `WPA × (8,5 + 2g) × 0,88`: een vaste K/W
  van 12,8 (g=3) tot 21,6 (g=8).
- Perpetuity is `FCF / (r − g)`: een vaste veelvoud van 14 tot 17 keer de vrije
  kasstroom.
- De multiples-methode is winst of kasstroom maal een veelvoud.

Alle drie zijn "genormaliseerde winst × een getal", op dezelfde twee invoerreeksen.
Dat ze het eens zijn zegt daarom weinig; "hoog vertrouwen" meet overeenstemming
tussen drie varianten van dezelfde som. Het model zegt dat ook over zichzelf
(Worldline-les), maar het label en de 20% weging in `rank_score` blijven staan.

### B2. Het model beloont krimp en straft groei — gemeten

De genormaliseerde winst is de mediaan van vijf jaar. Bij een groeier loopt die
twee jaar achter, bij een krimper twee jaar voor.

| 3-jaars omzetgroei | n | mediane "korting" | op SELL | korting ≥ 30% |
|---|---:|---:|---:|---:|
| krimp (< −2%) | 682 | **+19,7%** | 19% | 38% |
| vlak (−2 tot 5%) | 768 | +4,4% | 25% | 29% |
| groei 5–15% | 844 | −10,2% | 33% | 20% |
| groei > 15% | 446 | **−20,3%** | 43% | 16% |

Twee voorbeelden uit de trace: ASML wordt gewaardeerd op een WPA van 20,59 terwijl
FY2025 24,71 gaf en de laatste twaalf maanden 35,00 (SELL, −154%). Infotel wordt
gewaardeerd op 2,63 terwijl de winst vier jaar op rij daalde naar 2,37 (+30%
korting). Dat de markt meer betaalt voor groei is geen fout van de markt; het
model heeft binnen een sector geen enkele knop voor groei.

In de huidige top twintig van Kansen hebben acht aandelen een krimpende omzet.

### B3. De trechter levert weinig op, gemeten tegen je eigen oordelen

Van de twintig Kansen hebben er veertien een tussencheck of analyse:

| oordeel | aantal |
|---|---:|
| KOOP | 2 |
| HOLD | 5 |
| TWIJFEL | 2 |
| OVERSLAAN | 5 |

Twee op veertien. Zonder vergelijkingsgroep (hoeveel KOOP levert een willekeurige
greep of het tabblad Kwaliteit op?) is dat geen bewijs dat de zeef slecht is, maar
het is het getal waar het systeem op beoordeeld hoort te worden.

Correctie na stap 4: het dashboard toont deze telling wél, boven Kansen en boven
Kwaliteit ("15 van deze 20 zijn al onderzocht: 5× HOLD, 2× KOOP, 6× OVERSLAAN,
2× TWIJFEL"). Wat ontbreekt is het verloop in de tijd en de vergelijking tussen
tabbladen; de telling geldt alleen voor de lijst van dat moment.

### B4. De kwaliteitspoort staat sinds augustus open

Aanbeveling A2 uit de vorige review ("kwaliteitspoort uit BUY") wacht op een
besluit. Sindsdien is er niets veranderd: mediaan 3,5, correlatie 0,28 met de
eigen analyses, en de twee echte KOOP-oordelen van toen hadden kwaliteit 5,0.
Het koopsignaal is daarmee vooral een kwaliteitsfilter met de eigenaardigheden
van A3 en A4 erin.

### B5. Financials krijgen een signaal dat niets betekent

417 banken en verzekeraars hebben een oordeel. Hun mediane kwaliteitsscore is 2,0
en drie hebben een koopsignaal. De pagina toont "n.v.t." bij kwaliteit, maar de
score stuurt het signaal nog wel, en de waardering rekent voor een bank met
EV/EBITDA en vrije kasstroom. Het groeiprofiel sluit de sector terecht uit; de
waardering doet dat niet.

### B6. De ijkingen zijn in-sample

- **Moat:** vijf drempels afgesteld op 15–16 gevallen, en de tests pinnen
  diezelfde gevallen vast. Er is geen controle buiten de steekproef. De vorige
  review vond al dat de twee spannende gevallen (PAY, WTN) bij ±20% kantelen.
- **Groeiprofiel:** één bruikbaar cohort (F2024, 488 bedrijven, twaalf maanden
  koers, één marktregime). Drempels zijn gekozen en beoordeeld op dezelfde data.
  Het koersverschil tussen groen en rood is 51% tegen 46% boven de mediaan
  (+2 tegen −3 procentpunt); bij deze aantallen is dat ruis.
  Het sterke verschil zit in "fundament ok" (31% tegen 8%), maar die uitkomst is
  de kwaliteitsscore van nu, en die bevat de F-score (toets T3) en
  winststabiliteit (verwant aan T6). Deels voorspelt het profiel dus zichzelf.
  In beide cohorten is nul verdwenen: het universum is in augustus 2026
  samengesteld en bevat alleen overlevers.
  Het document noemt deze beperkingen zelf, en het profiel heet "voorlopig". Dat
  is eerlijk; "geijkt" in de code en op de uitlegpagina is te sterk.
- **Sectorankers:** nieuwe profielen worden afgeleid met een factor 1,45 die zelf
  uit de bestaande, niet onderbouwde profielen komt.

### B7. Een kwart heeft geen oordeel, en van de rest heeft 57% laag vertrouwen

Van de 4.012 tickers krijgt 26% geen oordeel. Van de 2.950 met oordeel heeft 57%
het label laag vertrouwen. Hoog vertrouwen: 273, en zie B1 voor wat dat waard is.

## C. Wat goed is

- De poorten: geen oordeel bij afgekeurde data, plausibiliteitsgrens, minimaal
  twee methodes, omzetbreuk. Het systeem zwijgt liever dan dat het gokt.
- Narekenbaarheid: `/api/trace` toont de werkelijke tussenwaarden van de motor.
- De framing: "zeef, geen oordeel", "onderzoeksopdracht, geen verkoopopdracht".
  De screener is de eerste van drie trappen en hoeft alleen een redelijke
  voorselectie te zijn.
- De documentatie benoemt eigen beperkingen ongewoon eerlijk, en de vorige review
  is grotendeels uitgevoerd (11 van 13 punten).
- De ingeprijsde groei (`implied_growth`) is de zinnigste maat in het systeem:
  een feit over de prijs, zonder modelaanname over het bedrijf.

## D. Oordeel

De techniek eromheen is zorgvuldig; de kern is zwakker dan de omlijsting doet
vermoeden. De fair value is een blend van veelvouden op een achterlopende
winstmaat, met een aantoonbare voorkeur voor krimpende bedrijven, en twee van de
vier multiples rusten op een verkeerd berekende bedrijfswaarde. De signalen
(BUY/SELL) suggereren meer dan het model kan dragen. Als voorselectie voor eigen
onderzoek is het bruikbaar, mits je weet welke kant het op helt.

## E. Aanbevelingen, op volgorde

| # | Wat | Soort | Omvang |
|---|---|---|---|
| 1 | EV-formule repareren, `historical_multiples` herberekenen, effect meten met `dry_run` | bugfix | klein |
| 2 | `auto_adjust=False` toetsen op één ticker; zo ja, doorvoeren en herberekenen | bugfix | klein |
| 3 | Trefkans per tabblad bijhouden op het scorebord (Kansen, Kwaliteit, Herstellers): hoeveel procent wordt VERDIEPEN of KOOP | meten | klein |
| 4 | Besluit nemen over de kwaliteitspoort (A2 van augustus) | beslissing | — |
| 5 | Krimpbias zichtbaar maken: winstrichting als kolom of als factor in `rank_score`; FV op laatste jaar naast FV op mediaan | model | middel |
| 6 | F-score altijd boekjaar tegen boekjaar; stabiliteit rond de trend | model | klein |
| 7 | Financials: geen signaal, of een eigen waardering (K/B tegen ROE) | model | middel |
| 8 | "Geijkt" vervangen door "voorlopig afgesteld" tot een tweede cohort (F2025, juli 2027) het bevestigt | tekst | klein |
