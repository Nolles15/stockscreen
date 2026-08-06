# Stockscreen — Architectuur

Dit document beschrijft de interne werking: data-pipeline, fair-value-formules, cron-ontwerp, DB-schema en de redenen achter de belangrijkste ontwerpkeuzes.

Voor een overzicht: [README.md](../README.md). Voor operationele zaken (wat gaat vaak mis, wat staat open): [CLAUDE.md](../CLAUDE.md).

---

## Overzicht

```
                    scheduler-thread ín het Flask-proces
                    (tick per kwartier, state in de database)
                                │
                                ▼
┌─────────────┐          ┌─────────────────┐         ┌──────────────┐
│  Yahoo      │◄─────────│  Flask op Fly   │────────►│  Neon        │
│  Finance    │ yfinance │ (stockscreen-   │ psycopg2│ PostgreSQL   │
│             │          │  janco, ams)    │         │  (Frankfurt) │
└─────────────┘          └─────────────────┘         └──────────────┘
                                ▲
                                │ handmatige noodknop
                       GitHub Actions (workflow_dispatch)
                                │
                                │ dashboard.html + /api/dashboard
                                ▼
                            Browser
```

**Stack.** Flask 3 (gunicorn 1 worker / 8 threads) · psycopg2 · yfinance · pandas · PyYAML. Alles in `engine/`.

**Data-flow.** (1) tickers komen uit `import_tickers.py` (complete noteringslijsten van de beurzen), (2) nachtelijke refresh haalt per ticker financials + koers op, (3) valuation-engine rekent fair value + signal, (4) dashboard leest uit DB en recomputeert het signal live met de actuele prijs.

### Universe-import (`import_tickers.py`)

De tickerlijst komt van de beurzen zelf, niet van een index. Een index is een selectie en zou per definitie kansen buiten die selectie missen; een noteringslijst is compleet.

| `--source` | Bron | Dekking |
|---|---|---|
| `euronext` | CSV-endpoint van live.euronext.com | `.AS` `.BR` `.PA` `.LS` `.IR` `.OL` |
| `xetra` | "Listed companies"-xlsx van Deutsche Börse | `.DE` |
| `nasdaq-nordic` | `api.nasdaq.com/api/nordic` screener, Main Market per beurs | `.ST` `.CO` `.HE` `.IC` |
| `baltic` | share-list-xlsx van nasdaqbaltic.com | `.TL` `.RG` `.VS` |
| `gpw` | biznesradar.pl (gpw.pl is niet altijd bereikbaar) | `.WA` |

**Waarom er een probe-stap tussen zit.** Een beurs publiceert haar eigen symbool, en dat is niet altijd het symbool waaronder Yahoo hetzelfde aandeel kent — Balder staat bij Nasdaq als `BALD B` en bij Yahoo als `FAST-B.ST`. Een niet-bestaand symbool importeren is niet onschuldig: de ticker levert nooit data, blijft als "GEEN DATA" in het dashboard staan en vult de refresh-rotatie met kansloze pogingen. Dat is precies het probleem dat fase 0 heeft opgeruimd (114 lege tickers). `--probe` stuurt de kandidaten daarom eerst naar `POST /api/stocks/probe` op de Fly-machine, die één bulk-`yf.download` doet en per symbool teruggeeft of er een koers uitkomt. Alleen die symbolen worden geïmporteerd.

Een mislukte probe-chunk telt niet als "onvindbaar": die tickers blijven onbeoordeeld liggen voor een volgende run. Een storing bij Yahoo mag nooit leiden tot het permanent afschrijven van geldige symbolen — dezelfde redenering als de storm-guard in `engine/refresh.py`.

**Dubbelingen** worden op ISIN gevangen. Dezelfde ISIN op twee beurzen betekent hetzelfde bedrijf; de notering in het thuisland van de ISIN wint, want daar heeft yfinance vrijwel altijd de beste dekking. Bekende uitzonderingen staan handmatig in [engine/remap_rules.py](../engine/remap_rules.py) en worden overgeslagen bij import.

---

## Modules

| Module | Rol |
|---|---|
| [app.py](../app.py) | Flask-routes, dashboard-cache, startup-init, cron-endpoints |
| [engine/db.py](../engine/db.py) | Connection pool, schema-init, upserts per tabel, override-helpers |
| [engine/data_fetcher.py](../engine/data_fetcher.py) | yfinance wrapper (`_yf_retry`), `fetch_and_store`, `fetch_all_tickers` (3 parallelle workers) |
| [engine/data_quality.py](../engine/data_quality.py) | Classificeert elke ticker als `ok` / `warning` / `bad` / `missing` obv compleetheid & plausibiliteit |
| [engine/normalizer.py](../engine/normalizer.py) | Mediaan-EPS, mediaan-EBITDA, mediaan-FCF over de laatste jaren + TTM |
| [engine/valuation.py](../engine/valuation.py) | Multiples + Graham + Perpetuity → `combined_fv`, plus scenario's |
| [engine/quality_score.py](../engine/quality_score.py) | Quality score (0-10) op basis van ROE, FCF-margin, debt/equity, consistency |
| [engine/screener.py](../engine/screener.py) | `run_ticker(ticker, cfg)` — bindt alles samen en schrijft naar `calculated_scores` |

---

## Fair value — de volledige formule

Drie methodes, daarna een gewogen gemiddelde.

### A. Multiples (gewicht 60%)

Vier onafhankelijke multiples, elk met een eigen fair value per aandeel:

| Methode | Formule |
|---|---|
| P/E       | Normalized EPS × P/E-multiple |
| EV/EBITDA | (Normalized EBITDA × EV/EBITDA − Net Debt) / Shares |
| P/B       | Book value per share × P/B-multiple |
| EV/FCF    | (Normalized FCF × EV/FCF − Net Debt) / Shares |

**Multiple-keuze.** Per metric is de gebruikte multiple een blend:

```
multiple = 0.65 × historische mediaan (5-jaars, uit historical_multiples)
         + 0.35 × sector default (config.yaml → sectors.<sector>.<metric>)
```

Dat voorkomt dat een bedrijf met een structureel hoge/lage re-rating volledig meegaat of volledig genegeerd wordt. Bij ontbrekende historie valt hij terug op pure sector-default.

**Outlier-filter (per methode).** Kandidaat FV's die < 0.20× of > 5.0× de mediaan van de 4 methodes zijn, worden gedropt (`MULTIPLE_OUTLIER_LOW` / `HIGH` in [engine/valuation.py](../engine/valuation.py#L43)).

**Normalization.** Multiples gebruiken **genormaliseerde** cijfers (mediaan van de laatste N jaar + TTM), niet een enkel jaar. Zo voorkom je dat één cyclisch piekjaar de FV dubbelteld.

### B. Graham + Perpetuity (gewicht 40%, 50/50 gesplitst)

**Graham IV.** Benjamin Graham's klassieke formule, gemoderniseerd met een yield-scaler:

```
Graham = EPS × (8.5 + 2 × g) × (4.4 / required_return)
```

Graham ijkte zijn formule in 1962 toen AAA-obligaties 4.4% gaven. In een 5%+ renteomgeving is de klassieke versie te optimistisch — de schaler zorgt dat Graham proportioneel zakt wanneer required_return stijgt. Zie [engine/valuation.py:36](../engine/valuation.py#L36).

**Perpetuity (Gordon Growth).**

```
Perpetuity = Owner Earnings per share / (r − g)   met min(r − g) = 2%
```

Owner Earnings ≈ Operating Cash Flow − Maintenance Capex. De min-spread van 2% voorkomt dat `(r − g)` naar nul gaat en de FV explodeert. Zie [PERPETUITY_MIN_SPREAD](../engine/valuation.py#L40).

### Combined FV

```
combined_fv = 0.60 × multiples_fv + 0.40 × (graham + perpetuity) / 2
```

**Cross-method outlier-filter.** Bij ≥3 beschikbare methodes (multiples, Graham, perpetuity) wordt elke methode die > 3× of < 0.33× de mediaan afwijkt gedropt (`METHOD_OUTLIER_LOW` / `HIGH`). Gedropte methodes worden gelogd in `calculated_scores.fv_methods_dropped`.

### Scenario's

Conservative / Base / Optimistic worden berekend door de sector-configuratie te variëren:

| Scenario | Groei | Required return |
|---|---|---|
| Conservative | low | high |
| Base | base | base |
| Optimistic | high | low |

Ranges per sector staan in [config.yaml](../config.yaml) onder `sectors.<name>`.

### Signal

```python
mos = (fair_value - price) / fair_value
quality = calculated_scores.quality_score  # 0-10

STRONG BUY  als mos ≥ 40%
BUY         als mos ≥ 25%
HOLD        als -15% ≤ mos < 25%
SELL        als mos < -30% of (quality ≥ 8 en mos < -75%)
```

Drempels in `config.yaml` → `signals`. De high-quality SELL-drempel (-75%) is expres strenger: voor kwalitatief superieure bedrijven tolereren we een grotere premium.

### Data-quality gate

`run_ticker` berekent **geen** FV/signal als `data_status` = `bad` of `missing`. In plaats daarvan krijgt de ticker `signal = INSUFFICIENT DATA`. Garbage in, garbage out — zie [engine/screener.py:247](../engine/screener.py#L247).

### Staleness-fix in dashboard

`api_dashboard` berekent `margin_of_safety` en `signal` **live** uit de laatste `price` + opgeslagen `combined_fv`, niet uit `calculated_scores.margin_of_safety`. Zie [app.py:190](../app.py#L190).

**Waarom.** `market_data` (prijs) refresht op andere cadans dan `calculated_scores` (FV). Als je het signal niet live recomputeert, krijg je stale signalen van het type "+88% korting" nadat de prijs al lang is gestegen.

---

## Refresh-ontwerp

### De huidige aanpak (fase 1, juli 2026) — scheduler op de machine zelf

De verversing draait in het Flask-proces op Fly, dat altijd aan staat
(`auto_stop_machines = false`). Een daemon-thread tikt elk kwartier en besluit
op basis van de tabel `refresh_state` wat er moet gebeuren. Zie
[engine/refresh.py](../engine/refresh.py) en `_scheduler_loop` in [app.py](../app.py).

```
elk kwartier een tick
   │
   ├─ na 18:30 Amsterdam en vandaag nog niet gedraaid?
   │     └─ refresh_prices_bulk()  — alle tickers, chunks van 200
   │
   ├─ na 03:00 Amsterdam en vandaag nog niet gedraaid?
   │     └─ refresh_fundamentals_batch(250)  — de 250 langst niet-geprobeerde
   │        (`refresh.fundamentals_per_night` in config.yaml)
   │
   └─ langer dan 7 dagen geleden?
         └─ weekly_reprobe(40) + activity_log ouder dan 90 dagen opruimen
```

**Waarom de state in de database staat.** Elke beslissing komt uit
`refresh_state`, niet uit geheugen. Een herstart van de machine — bij een deploy
of een crash — mag geen ronde overslaan en ook geen dubbele ronde veroorzaken.
De vorige scheduler keek naar de leeftijd van willekeurige marktdata; zodra één
ticker toevallig vers was leek alles vers en bleef de verversing uit.

**Waarom koersen en cijfers gescheiden zijn.** Koersen veranderen dagelijks en
zijn in bulk op te halen (honderden per call). Jaarcijfers veranderen hooguit
per kwartaal en kosten ~4 seconden per ticker. Ze in één pijplijn stoppen
betekende dat het trage deel het snelle deel ophield: bij 900 tickers duurde een
volledige ronde ruim een uur, en bij de beoogde 3.000 zou het niet meer binnen
een nacht passen.

**Storm-guard.** `refresh_fundamentals_batch` telt mislukkingen pas ná afloop.
Levert meer dan een kwart van de batch niets op, dan gaan we uit van een storing
bij Yahoo: er wordt `storm_detected` gelogd en geen enkele failure-teller gaat
omhoog. Dit is de directe tegenmaatregel tegen wat er in april gebeurde, toen
één slechte periode 115 tickers automatisch liet suspenderen. Een fetch die
zonder foutmelding voltooit maar niets oplevert telt hierin mee als mislukking —
delisted tickers gooien namelijk geen fout.

**Suspenderen vraagt drie voorwaarden tegelijk.** Pas na 3 mislukkingen,
gespreid over minstens 21 dagen, en alleen als er geen enkel jaarcijfer in de
database staat. Daarna volgt wekelijks een nieuwe poging; pas na 45 dagen zonder
resultaat komt er `presumed_delisted_at` op te staan, en ook dan blijft de ticker
zichtbaar in het beheerscherm. Niets verdwijnt stilzwijgend.

Die getallen stonden ooit op 10/30/90 en zijn verlaagd omdat ze in de praktijk
anders uitpakten dan bedoeld: een ticker krijgt maar eens per elf dagen een
beurt, dus tien pogingen duurden maanden. **De bescherming tegen massa-suspensies
zit niet in deze drempels maar in de storm-guard hierboven** — die laat de
tellers bij een storing helemaal met rust, en dat is waarom ze omlaag konden.

**Zichtbaarheid.** `GET /api/health` (zonder token) geeft de leeftijd van beide
rondes, versheidspercentages, dekkingscijfers en `scheduler_alive`. Elke pagina
toont daarvan een banner; het dashboard toont bovendien hoeveel van de universe
echt beoordeeld is.

**GitHub Actions is teruggebracht tot noodknop.** Het `schedule:`-blok is
verwijderd. Reden: GitHub schakelt een scheduled workflow automatisch uit na 60
dagen zonder push naar de repo. Dat gebeurde op 19 juni 2026, waarna de
verversing zes weken stillag zonder enige melding. Een motor die afhangt van hoe
recent er code gepusht is, is geen betrouwbare motor.

### De aanpak daarvóór (april–juli 2026) — externe cron per ticker

GitHub Actions hield zelf de loop aan en riep per ticker een synchrone endpoint aan. Betrouwbaarder dan wat eraan voorafging, maar afhankelijk van een externe planner die stilviel zonder melding — zie hierboven. De endpoints (`/api/cron/next-batch`, `/api/cron/refresh-one/<T>`) bestaan nog en werken; ze zijn nu de noodknop.

### De aanpak daarvóór (verwijderd april 2026)

`POST /api/cron/refresh-batch` startte een async job in een thread, polde `/api/cron/refresh-batch/status/<id>` elke 30s. Problemen:

1. **In-memory state op Fly.** `_jobs` dict verdween bij een machine-restart → status endpoint 404 → workflow zag "unknown" status en hing.
2. **Geen timeout op yfinance.** Één hangende ticker blokkeerde een worker voor minuten.
3. **Gunicorn 120s timeout.** Batch van 90 tickers × 15s = 22 minuten. Gunicorn killde de worker ruim vóór het einde.

### De nieuwe aanpak (commit c97758a)

GitHub Actions houdt **zelf** de loop aan. Fly krijgt per ticker één korte synchrone call.

```
┌─────────────────────────────┐
│  GitHub Actions             │
│                             │
│  1. GET  /api/cron/next-    │──► Fly geeft N oudste tickers terug
│        batch?limit=N        │
│  2. for T in tickers:       │
│       POST /api/cron/       │──► Fly fetch + recalc + log, synchroon (~5-15s)
│         refresh-one/<T>     │◄── { ok: true, signal, fv, elapsed_s }
│                             │
└─────────────────────────────┘
```

**Eigenschappen.**

- **Stateless op Fly.** Geen `_jobs` dict, geen polling. Een Fly restart midden in een batch kost hooguit één ticker.
- **curl `--max-time 100`** op de workflow-kant ligt onder de `120s` gunicorn-timeout — als yfinance hangt, sneuvelt deze ene call en gaat de workflow door.
- **Workflow faalt alleen als >20% van de tickers faalt.** Losse yfinance-timeouts zijn normaal en mogen de nachtelijke run niet als rood markeren.
- **Batch-selectie is `ORDER BY last_fetched ASC`.** De `limit=90` in de workflow stamt uit de tijd van ~620 tickers, toen alles daarmee binnen zeven dagen vers was. Het universum is inmiddels ~2.760 groot en de interne scheduler draait 250 per nacht (elf dagen per ronde); deze noodknop is dus alleen bedoeld om een achterstand in te lopen, niet om de ronde over te nemen. Geef er dan handmatig een hogere `limit` aan mee.

**Auth.** Elke `/api/cron/*` endpoint checkt `X-Cron-Token: <CRON_TOKEN>`. Zowel Fly (`fly secrets set CRON_TOKEN=...`) als GitHub (`Settings → Secrets → CRON_TOKEN`) moeten dezelfde waarde hebben.

**Scheduler-gate (vervallen).** Tot fase 1 sloeg `_on_startup` de in-process scheduler over zodra `CRON_TOKEN` gezet was. Bedoeld om dubbelwerk te voorkomen, maar het maakte de externe cron tot enige motor — en toen die stilviel, ving niets het op. De gate is vervangen door `SCHEDULER_ENABLED` (default `1`).

### Relevante endpoints

| Method + Path | Wie roept | Doet |
|---|---|---|
| `GET /api/cron/next-batch?limit=N` | GitHub Actions | Geeft N oudste tickers + totalen terug (geen fetch) |
| `POST /api/cron/refresh-one/<T>` | GitHub Actions | Synchroon één ticker: fetch + recalc + log, retourneert JSON met signal/fv/elapsed |
| `POST /api/refresh` | UI | Start in-process async job (handmatige "Alles verversen" vanuit browser) |
| `POST /api/recalculate` | UI of debug | Herrekent scores uit DB zonder netwerk-call |

### Workflow-bestand

[`.github/workflows/daily-refresh.yml`](../.github/workflows/daily-refresh.yml) — in de repo-root. De bestandslocatie is kritisch: GitHub Actions zoekt alleen onder de repo-root.

---

## DB-schema

Alle 9 tabellen worden aangemaakt bij startup via `init_db()` in [engine/db.py](../engine/db.py). Migraties gebeuren idempotent met `CREATE TABLE IF NOT EXISTS` en `ALTER TABLE … ADD COLUMN IF NOT EXISTS`.

| Tabel | Primary key | Doel |
|---|---|---|
| `stocks` | ticker | Metadata: naam, sector, markt, currency, financial_currency, active, added_date |
| `financials` | (ticker, year) | Jaar + TTM cijfers: revenue, ebitda, net_income, fcf, operating_cf, capex, etc. |
| `market_data` | ticker | Huidige prijs, market cap, TTM-multiples (pe_ttm, ev_ebitda_ttm, pb), last_fetched |
| `historical_multiples` | (ticker, year) | 5-jaars historie van pe, ev_ebitda, pb, ev_fcf → input voor 65/35 blend |
| `calculated_scores` | ticker | combined_fv, conservative/base/optimistic FV, signal, margin_of_safety, quality_score, warnings, fv_methods_used/dropped |
| `data_quality` | ticker | status, completeness_pct, issues (JSON), consecutive_failures, last_checked |
| `overrides` | (ticker, field, year) | Handmatige cijfers. `year=NULL` betekent "alle jaren" (bv. voor shares_outstanding) |
| `exchange_rates` | (base, quote, date) | Alleen voor legacy; native currency pipeline gebruikt dit nauwelijks meer |
| `activity_log` | id (serial) | Alle fetch/recalc/override/delete events met JSON details |

**Native currency pipeline.** Er is geen `price_eur` kolom meer. Prijs én FV staan in `stocks.currency`. Als een endpoint 500 gooit met `UndefinedColumn: price_eur`, loopt de code achter op de DB-migratie → `fly deploy`.

---

## Overrides-systeem

Yahoo Finance mist regelmatig velden — zeker voor dual-listed tickers, kleine exchanges (.WA, .LS, .MC) en recent gesplitste bedrijven. Oplossing: handmatige overrides uit het jaarverslag.

**API.**

- `GET  /api/overrides/<ticker>` → huidige overrides
- `POST /api/overrides/<ticker>` body: `{ field: "eps_diluted", year: 2024, value: 3.25 }`
- `DELETE /api/overrides/<ticker>` body: `{ field: "eps_diluted", year: 2024 }`

**Ondersteunde velden.** `eps_diluted`, `fcf`, `ebitda`, `net_income`, `revenue`, `operating_cf`, `total_equity`, `total_debt`, `shares_outstanding`, `book_value_ps`, `roe`, `interest_expense`, `capex`, `ebit`, `total_assets`, `current_assets`, `current_liabilities`, `gross_profit`.

**`year=NULL`-override.** Voor velden die per definitie niet jaarsgebonden zijn (bv. `shares_outstanding` als latest count). Override-resolver pakt altijd eerst de specifieke jaar-override, anders de `year=NULL` fallback, anders de Yahoo-waarde.

**Beoogde workflow.** Een AI-skill (cowork) leest een jaarverslag (PDF of HTML), extraheert de relevante cijfers en POST ze naar `/api/overrides/<T>` per veld + jaar. Voor tickers met Yahoo-gaten is dit de manier om ze toch gevalueerd te krijgen. Zie [CLAUDE.md → Huidige plan](../CLAUDE.md#huidige-plan--status) voor het `/api/gaps-report` endpoint dat deze workflow aanstuurt.

---

## Ontwerpkeuzes — waarom zo?

**Waarom Flask en geen FastAPI?** Het project begon als 1-proces desktop-tool. Threads + requests was genoeg; async voegt niets toe want yfinance is blocking I/O en de pool is sowieso maar 3 workers.

**Waarom Neon en geen Fly Postgres?** Free tier, geografisch dichtbij (Frankfurt), branchable voor experimenten, geen Fly-lock-in. Voor dit werkvolume ruim voldoende.

**Waarom native currency en geen EUR-conversie?** Het FV-model werkt op de lokale financials. Als je prijs én FV in dezelfde currency houdt, is MoS betekenisvol ongeacht FX. Eerdere poging met `price_eur` gaf alleen maar refactoring-pijn.

**Waarom `auto_stop_machines=false`?** De per-ticker cron verwacht een levende machine per call; bij cold-start zou de eerste call van elke workflow een ~10s spin-up pakken × 90 tickers = 15 minuten overhead. Voor 256mb is altijd-aan goedkoper.

**Waarom 3 workers in `fetch_all_tickers` en niet meer?** Yahoo rate-limit'ed (HTTP 429) boven ~5 parallelle requests. `_yf_retry` doet 3 pogingen met exp backoff en 3× langere backoff bij 429 — dat werkt alleen bij lage parallelisatie.

**Waarom blend 65/35 en geen 50/50?** Pure sector-defaults zijn te generiek (Ahold ≠ Casino ≠ Jerónimo Martins); pure historie overfit op individuele re-ratings. 65/35 leunt op het aandeel zelf maar voorkomt dat een jarenlang te duur aandeel tot in de eeuwigheid te duur blijft bij rerating.
