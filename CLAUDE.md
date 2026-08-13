# Stockscreen — operational notes for Claude

Dutch-language stock screener voor Europese small/mid-caps. Flask + PostgreSQL (Neon) + yfinance, gehost op Fly.io.

## Kerngegevens

| | |
|---|---|
| **Live** | https://stockscreen-janco.fly.dev |
| **Fly app** | `stockscreen-janco` (region `ams`, 512mb, 1 worker / 8 threads, gunicorn 120s timeout) |
| **DB** | Neon PostgreSQL, Frankfurt |
| **GitHub** | github.com/Nolles15/stockscreen (branch `main`) |
| **Local dev** | `DATABASE_URL="..." python app.py` → http://localhost:5001 |
| **Deploy** | `fly deploy --remote-only --depot=false`, en **daarna `fly apps destroy fly-builder-*`**. Zie de twee regels hieronder; die volgorde is geen detail. |

## Kritieke operationele feiten

**Native currency pipeline** — prijs én fair value zijn altijd in de native currency van het aandeel. Er is GEEN valutaconversie; `price_eur`-kolommen zijn in een eerdere migratie verwijderd. Als een endpoint 500 geeft met `UndefinedColumn`, is de code achter op de DB-migratie → `fly deploy`.

**De verversing draait op de machine zelf, niet op GitHub Actions.** De scheduler zit in `_scheduler_loop` in app.py en tikt elk kwartier; `fly.toml` houdt de machine daarvoor altijd aan. GitHub Actions is nog uitsluitend een handmatige noodknop.
- Werkverdeling: koersen in bulk na 18:30 Amsterdam, jaarcijfers na 03:00 (`fundamentals_per_night: 250` uit config.yaml), wekelijks re-probe en logopschoning.
- Noodknop: [.github/workflows/daily-refresh.yml](.github/workflows/daily-refresh.yml), alleen `workflow_dispatch` — **het `schedule`-blok is bewust verwijderd**, want een tweede motor naast de interne scheduler verdubbelt de druk op de Yahoo-rate-limits. Endpoints `GET /api/cron/next-batch?limit=N` + `POST /api/cron/refresh-one/<T>`, secrets `CRON_TOKEN` (Fly + GH identiek) en `APP_URL` (GH).
- **`CRON_TOKEN` zet de scheduler níét meer uit.** Die gate heeft ooit bestaan en heeft zes weken schade aangericht: GitHub schakelde de scheduled workflow automatisch uit na 60 dagen inactiviteit, de interne scheduler lag stil omdat het token bestond, en niets ving het op. Nu stuurt alleen `SCHEDULER_ENABLED=0` hem uit. Zie [app.py:2818](app.py#L2818).

**Staleness-fix in dashboard** — `api_dashboard` berekent `margin_of_safety` en `signal` LIVE vanuit laatste `price` + opgeslagen `combined_fv` (niet uit `calculated_scores.margin_of_safety`). Zie [app.py:190](app.py#L190). Reden: market_data refresht op andere cadans dan FV; zonder live-recompute krijg je misleidende +88% korting-signalen.

**Data-kwaliteit gate** — `run_ticker` berekent GEEN FV/signal als `data_status` = `bad` of `missing` (garbage in, garbage out). Zie [engine/screener.py:247](engine/screener.py#L247).

## Fair value berekening

60% gewicht multiples + 40% gewicht (Graham + Perpetuity)/2:
- **Multiples** (4 methoden, mediaan): P/E × norm_EPS, EV/EBITDA → per share, P/B × BVPS, EV/FCF → per share. Multiples zijn blend 65% eigen historisch + 35% sector-default.
- **Graham**: `EPS × (8.5 + 2g) × (4.4 / required_return)` — de yield-scaler voorkomt opblazen in hoge-rente-omgevingen.
- **Perpetuity**: `OE/share / (r − g)`, min spread 2%.
- **Cross-method filter**: bij ≥3 methodes dropt outlier-filter elke methode die >3× of <0.33× afwijkt van de mediaan.

Sector-multiples en groei-aannames: [config.yaml](config.yaml) sectie `sectors`. Signal-drempels (STRONG BUY / BUY / HOLD / SELL): sectie `signals`.

## DB schema (9 tabellen, init op startup)

| tabel | doel |
|---|---|
| `stocks` | ticker + metadata (naam, sector, markt, currency, active) |
| `financials` | jaar/TTM-cijfers per ticker (revenue, ebitda, net_income, fcf, etc.) |
| `market_data` | huidige prijs, market cap, multiples (pe_ttm, ev_ebitda_ttm, pb) |
| `historical_multiples` | 5-jaars historie van pe, ev_ebitda, pb, ev_fcf per ticker |
| `calculated_scores` | combined_fv, conservative/base/optimistic, quality, signal, warnings |
| `data_quality` | status (ok/warning/bad/missing), completeness, issues, consecutive_failures |
| `overrides` | handmatige cijfers per field+year (inclusief `year=NULL` voor alle jaren) |
| `exchange_rates` | FX rates (gebruik minimaal — alles native) |
| `activity_log` | alle fetch/refresh/override events, JSON details |

## Bekende valkuilen

- **Dual-listings** — bv. EXOR.AS (primary = EXO.MI), ACOMO.BR (primary = ACOMO.AS). Yahoo levert voor de secundaire ticker vaak GEEN financials. Oplossing: mapping tabel óf primaire ticker gebruiken óf handmatige overrides.
- **Negatief eigen vermogen** — P/B-methode geeft absurde waarden. Outlier-filter vangt het op maar houd `fv_methods_dropped` in de gaten.
- **yfinance is flaky** — retry-logic in `_yf_retry` (3 pogingen, exponential backoff 2s/4s/8s, 3× langer bij 429). Rate-limits in bulk: 3 workers max in `fetch_all_tickers`.
- **INSUFFICIENT DATA** na refresh — betekent echt: Yahoo gaf niks terug. Categoriseer via `/api/data-quality` endpoint.
- **"Compleet" zegt niets over wélk jaar** — `completeness_pct` telt de vulgraad van de opgeslagen jaren, `freshness_days` telt wanneer we Yahoo belden. Geen van beide merkt dat het nieuwste boekjaar van twee jaar terug is. Daarvoor is `data_quality.verwacht_boekjaar()` / `boekjaar_achterstand()` — de enige plek waar die kalenderregel staat. Screener, dashboard, Kansen en de detailpagina lezen daaruit; bouw hem nergens na.
- **Yahoo levert voor sommige tickers structureel één boekjaar te weinig** — yfinance krijgt maximaal 4 jaarkolommen en bij ~3% van de tickers eindigen die een jaar te vroeg (o.a. LASTIK.HE, COLO-B.CO, AMBU-B.CO, KWS.DE, geverifieerd met verse fetches op 2026-08-03). Opnieuw ophalen helpt niet; alleen handmatige overrides. Zeg dus nooit "klik Refresh" als `freshness_days` klein is.
- **Handmatige cijfers staan in een andere tabel dan de rest** — `overrides`, niet `financials`. Wie alleen `financials` leest mist ze. Gebruik altijd `db.jaarrijen_met_overrides()`; die voegt jaren toe die alleen handmatig bestaan én past de correcties toe. Er stonden drie kopieën van die lus en de vierde plek (de datakwaliteit) miste hem, waardoor het dashboard "cijfers t/m 2024" bleef tonen naast een FY2025 dat er gewoon was.
- **Een handmatige jaarrij moet álle velden bevatten** — `db.lege_jaarrij()`. In Jinja is een ontbrekende sleutel `Undefined`, en `Undefined is not none` is waar; de opmaakmacro loopt dan door naar `val | abs` en de hele pagina geeft een 500. Zie `tests/test_handmatig_boekjaar.py`, die de echte template rendert.
- **`net_cash` is geen sierveld** — `valuation.py` bouwt er de brug mee van bedrijfswaarde naar waarde per aandeel (`net_debt_ps = -net_cash / shares`), en pakt daarvoor de nieuwste jaarrij met een aandelenaantal. Ontbreekt hij daar, dan leest die code nettoschuld = 0 en valt de FV te hoog uit. Bij LASTIK.HE scheelde dat 150 miljoen euro schuld op een marktwaarde van 258 miljoen.
- **Live herrekenen mag de data-poort niet omzeilen** — `_effective_signal` in app.py rekent het signaal opnieuw uit tegen de verse koers. Het keek alleen niet naar `data_status`, terwijl de screener een 'bad'-ticker wél op INSUFFICIENT DATA zet maar de oude `combined_fv` laat staan. Resultaat: 19 afgekeurde aandelen met een vers HOLD/SELL erop. Elke afleiding die de motor overdoet moet dezelfde poorten passeren.
- **De sectorsleutels in `config.yaml` zijn Yahoo-namen, geen GICS-namen.** Yahoo zegt "Consumer Cyclical", "Consumer Defensive" en "Basic Materials" waar GICS "Consumer Discretionary", "Consumer Staples" en "Materials" zegt. Tot 6 augustus 2026 stonden de GICS-namen in de config: 617 aandelen (22% van het universum) vielen stil terug op `Default`, want `_sector_cfg` valt netjes terug en meldt niets. Drie profielen die er keurig uitzagen werden dus nooit gebruikt. De schade zat in de richting: Basic Materials hoort op K/W 15 en EV/EBITDA 9 te staan, Default geeft 18 en 11 — een royaler anker blaast de fair value op en levert schijnkortingen, precies de vals-positieven die onderzoekstijd kosten. Gemeten effect van de reparatie: fair value ×0,88 voor Basic Materials, ×1,06 voor Consumer Cyclical, ×1,01 voor Consumer Defensive. Bewaakt door `tests/test_sector_config.py`, die ook omvalt bij een profiel dat géén enkele ticker kan raken. Alleen `Unknown` (155) staat nu nog bewust op Default; dat is daar de juiste uitkomst.
- **Een nieuw sectorprofiel leid je af uit het eigen universum, niet uit een externe bron.** De ijkmethode staat in de docstring van `tests/test_sector_config.py`: deel per bestaande sector de configwaarde door de mediaan van de eigen historische multiples van de aandelen erin. Die factor ligt voor de K/W rond 1,45 (Technology 1,42 · Utilities 1,43 · Consumer Defensive 1,45 · Industrials 1,78 · Energy 1,87); pas hem toe op de mediaan van de nieuwe sector. Zo is Communication Services op 6 augustus 2026 op K/W 17 en EV/EBITDA 10 gezet (mediaan 12,8 en 5,1). Meet de uitkomst vóór de deploy — daar was dat fair value ×0,911 over 136 tickers en zes keer HOLD → SELL, allemaal telecom, wat bevestigt dat de correctie de juiste subgroep raakt. Let op dat één sectorgetal nooit klopt voor een bak waar telecom (K/W 11) en games (K/W 16) in zitten; dat het toch beperkt schaadt, komt doordat de fair value voor 65% op de eigen historie van het aandeel leunt en maar voor 35% op het sectoranker.
- **De waardering rekent met een TTM-rij die de API niet teruggeeft.** `run_ticker` zet de TTM-rij (`fiscal_year=0`) vóór de jaarrijen zodra het nieuwste boekjaar ouder is dan het huidige kalenderjaar; alles daarna — normalisatie, kwaliteit, fair value — draait dus over vijf rijen waarvan de eerste geen boekjaar is. `/api/stock` levert alleen `period_type='annual'`. Wie de waardering lokaal nabouwt uit die API krijgt daarom stelselmatig een andere uitkomst dan productie, zonder foutmelding: bij BMW.DE was dat 140 tegen 175 op 6 augustus 2026. Wil je het effect van een configwijziging meten, vergelijk dan de **verhouding** tussen oud en nieuw over identieke invoer, of gebruik `POST /api/scores/recompute` met `dry_run` — die draait de echte code.
- **ETFs** — screener werkt niet voor ETFs (bv. BFIT). Deactiveer met `active=0`.
- **Fly auto-stop UIT** — `fly.toml` heeft `auto_stop_machines = false` + `min_machines_running = 1`. Dat is expres en sinds fase 1 essentieel: de scheduler draait ín het proces, dus een slapende machine betekent geen verversing.
- **Ruim de bouwmachine op na elke deploy: `fly apps destroy fly-builder-*`.** `--remote-only` laat Fly een bouwmachine aanmaken met een schijf van **50 GB**, en die schijf wordt maandelijks doorberekend of hij gebruikt wordt of niet — op 7 augustus 2026 was dat meer dan de helft van de hele Fly-rekening, voor niets anders dan een Docker-cache. Fly maakt hem bij de volgende deploy vanzelf opnieuw aan, dus opruimen kost je niets. De deploys zelf zijn goedkoop (40–90 seconden bouwtijd, 5,5 MB context); het was nooit het deployen dat geld kostte, maar het achterlaten.
- **`fly deploy --local-only` werkt hier NIET, ook al draait er Docker.** Geprobeerd op 7 augustus 2026: de TLS-onderschepping op dit netwerk laat `pip install` in de container niet bij PyPI (`CERTIFICATE_VERIFY_FAILED ... unable to get local issuer certificate`), dus de build valt om op `No matching distribution found for flask`. Dezelfde onderschepping die de depot.dev-builder blokkeert. De remote builder heeft er geen last van omdat die binnen het netwerk van Fly draait. Het is te repareren door de onderscheppende CA in het image te zetten, maar dat betekent een bedrijfscertificaat in een publieke repo — eerst overleggen. **Andere apps kunnen wél lokaal bouwen** (`triathlon-dashboard` deed dat op dezelfde dag zonder problemen); het hangt af van wat de Dockerfile tijdens de build ophaalt.
- **Secrets via de Fly-website blijven op "Staged"** — de draaiende app pakt ze dan niet op. Altijd afsluiten met `fly secrets deploy -a stockscreen-janco`. Via `fly secrets set` gebeurt dat automatisch.
- **Draai `pyflakes` vóór elke deploy** — `py_compile` ziet ongedefinieerde namen niet. Zo ging `/api/dashboard` op 2026-08-01 onderuit met een NameError die netjes compileerde. `pip install -r requirements-dev.txt`, dan `python -m pyflakes app.py engine/*.py`.
- **Sjablonen delen één JavaScript-scope.** `base.html` en de pagina-sjabloon staan als losse `<script>`-blokken in dezelfde pagina, en klassieke scripts delen de globale scope. Een `const fmt` in `stock.html` botst dus met de `function fmt()` in `base.html`: *"Identifier 'fmt' has already been declared"* — een SyntaxError die het **hele** blok onuitgevoerd laat, dus ook alles wat er verder in staat. Zo verdwenen op 2026-08-02 de beslisboom, de moat-cijfers én de koersgrafiek tegelijk, zonder zichtbare melding.
  **Controleer JS daarom altijd samengevoegd met `base.html`, nooit per bestand.** `node --check` op één sjabloon vindt deze fout niet; los van elkaar zijn beide bestanden geldig. Bestaande globals in `base.html`: `fmt`, `fmtMoney`, `fmtBig`, `fmtBigMoney`, `currencySymbol`, `CURRENCY_SYMBOLS`.
- **Geen Jinja-commentaar (`{# … #}`) binnen een JavaScript-template-literal.** Jinja haalt het er bij het renderen wel uit, maar dan is de JS alléén geldig ná het renderen — en de syntaxcontrole draait op het onbewerkte sjabloon. Zet uitleg over een stuk HTML-in-JS als gewone `//`-regel bóven de expressie, en bouw de cel desnoods in een losse variabele. Bij het samenvoegen voor `node --check` moet je naast `{{ }}` en `{% %}` dus ook `{# #}` wegstrippen, anders keurt de controle iets af dat in productie prima werkt (of erger: keurt hij iets goed dat je niet gecontroleerd hebt).
- **`truncate` verbergt alles wat na de tekst komt.** De naamkolom van de grote tabel had `max-w-[160px] truncate`; een badge die daarachter in dezelfde cel stond viel bij elke wat langere bedrijfsnaam buiten beeld. Hij stond er wel, je zag hem nooit. Zet de tekst in een eigen `<span class="truncate">` binnen een `flex`-rij en de badge ernaast, met `flex-shrink: 0` op de badge.
- **Vang fouten in de UI zichtbaar af.** Een `.catch(() => {})` rond het opbouwen van een pagina laat bij een fout een lege plek achter zonder enig spoor. De beslisboom stond daardoor sinds fase 3 leeg zonder dat iemand het merkte — de tweede oorzaak daarvan (`fv_methods_dropped` als onontlede JSON-tekst) was pas te vinden nadat de melding zichtbaar werd gemaakt.
- **Kleuren in JS-strings ontsnappen aan de thema-overrides.** `base.html` mapt Tailwind-slate-classes naar het cream-thema, maar een hex-waarde in een JavaScript-template of een lokaal `<style>`-blok wordt daar niet door geraakt. Zo bleef de koersgrafiek maandenlang in dark-theme-kleuren staan: rasterlijnen in `#1e293b` (bijna zwart) op een witte kaart, en een waardelijn in `#7ee2a8` die bedoeld was voor een donkere ondergrond. Gebruik in SVG en inline styles altijd `var(--border)`, `var(--text-3)` enzovoort. De twee grafiekkleuren staan als `--reeks` en `--waarde` op `#koersgrafiek` en zijn getoetst met de validator uit de dataviz-skill (lightness, chroma, contrast, kleurenblindheid).
- **Machine draait op UTC, beurzen op Amsterdamse tijd.** De scheduler rekent expliciet om via `ZoneInfo("Europe/Amsterdam")`; vergelijk nooit direct met `datetime.utcnow()`.
- **De analyse-rapporten moeten in `analyses/` op de repo-root staan, niet in `data/`.** `.dockerignore` sluit `data/`, `scripts/` en `docs/` uit van de build-context. Een rapport in `data/analyses/` werkt lokaal perfect en ontbreekt in productie zonder enige foutmelding — de pagina toont dan simpelweg nul analyses.
- **Een oordeel gaat over het bedrijf, niet over de notering.** `engine/oordelen.py` hangt de uitslag van een tussencheck of analyse aan de screener-rijen. Koppelen op ticker alleen is niet genoeg: Silvano staat in Warschau als `SFG.WA` en in Tallinn als `SFG1T.TL` en die delen geen enkele letter, terwijl het één bedrijf met één tussencheck is. Op 4 augustus 2026 bezetten ze samen twee van de twintig plekken in Kansen met een oordeel OVERSLAAN dat de pagina niet kende. Vandaar drie sleutels in volgorde: Yahoo-symbool uit de metadata, de kale ticker (alleen als er precies één kandidaat is, anders zou een tussencheck op `AD` zich aan elke `AD.*` hechten), en de bedrijfsnaam. Een koppeling via de naam krijgt `via` mee en dan wordt het koersverschil bewust níet berekend — de andere notering staat in een andere valuta.
- **De module verwijdert niets uit Kansen.** Dat is een keuze, geen omissie: een lijst waar dingen ongemerkt uit verdwijnen ga je wantrouwen. Het oordeel wordt getoond, de rangorde blijft van de screener.
- **Rapporten mengen Europese en Amerikaanse getalnotatie.** `1.239,40` (ASML) staat naast `93.79` (NVDA). Wie de punt blind als decimaalteken leest, maakt van ASML's koers 1,239 en krijgt een upside van +114476%. `engine/analyses.py:_parse_getal` handelt dit af: komma wint altijd als decimaalteken, een punt met precies drie cijfers erachter is een duizendtalscheiding.

## Verkoopregels en bezit (2026-08-13)

Het sluitstuk van de pijplijn: tussencheck → analyse → gekocht → **en dan?** `_routekaart()`
eindigde bij "bijwerken wanneer de cijfers verouderen"; er was geen enkele regel die zei wanneer
een aandeel weer weg mag. Tabblad **💼 Ik bezit** houdt twaalf regels tegen wat je bezit, plus één
informatieve. Motor: [engine/exit_regels.py](engine/exit_regels.py), pagina in
[templates/index.html](templates/index.html), tabel `bezit` in de database.

- **Geen aankoopprijs, en dat is de kern van het ontwerp.** Er wordt geen aantal en geen
  instapkoers opgeslagen; geen enkele regel kijkt ernaar. Wat je betaald hebt is geen eigenschap
  van het bedrijf, en eraan vasthouden is de bekendste manier om winnaars te vroeg te verkopen.
  Bijkomend voordeel: de repo is publiek en de site heeft geen afscherming, dus er staat niets
  gevoeligs opgeslagen. **Wel zichtbaar voor iedereen met de URL: wélke aandelen je bezit.**
- **Een geraakte regel is een onderzoeksopdracht, geen verkoopopdracht.** Die zin staat op de
  pagina en hoort er te blijven staan, net als bij `/tussenchecks`.
- **De momentopname is het ijkpunt.** Bij vastleggen bevriest `exit_regels.momentopname()` de
  cijfers (kwaliteit, ROIC-mediaan, margetrend, omzet-CAGR, kasstroom, ingeprijsde groei).
  `bezit_vastleggen` overschrijft die bij een herbevestiging **niet** — anders wist elke klik de
  vergelijking. Gevolg dat je moet kennen: op dag één zijn de vergelijkende regels (B2, B3, B5)
  per definitie stil; de absolute regels doen dan het werk. Bewaakt in `tests/test_exit_regels.py`.
- **Geen absolute kwaliteitsvloer.** De verleiding is "kwaliteitsscore onder de 5 = verkopen".
  Dat ís `sell_quality_floor`, in fase 2 geschrapt omdat 72,5% van het universum onder de 6
  scoort. Kwaliteit telt alleen als *daling sinds vastleggen*.
- **"Boven het optimistische scenario van de screener" is geen bruikbare regel** — stond zo in
  het plan en is na narekenen vervangen. De multiples wegen 60% en bewegen niet mee met de
  scenario's, dus `optimistic_fv` ligt maar 8% (Technology) tot 27% (Consumer Defensive) boven
  `base_fv`. Die regel vuurt rond 108-127% van de fair value — lósser dan de SELL-drempel van
  130% die de screener zelf hanteert. In plaats daarvan is A2 (de enige harde waarderingsregel)
  `sell_pct_high_quality` uit config.yaml: 175%, de marge die compounders krijgen. Hergebruikte
  drempel, geen nieuw getal.
- **Eén regel per waarneming, anders telt het oordeel dubbel.** Rood vraagt twee geraakte regels
  uit twee verschillende families (of één harde). Zouden ROIC-hoogte, standvastigheid en
  margeërosie elk een eigen regel zijn, dan zou één instortend bedrijf drie "regels" scoren.
  Daarom neemt B1 het gekalibreerde oordeel van `moat_profile` in z'n geheel over.
- **De C-regels lezen de scenario's uit je eigen rapport** — `fair_value_kansgewogen` en de
  scenariotabel, sinds nu geparst in `engine/analyses.py`. **Elk rapport bevat die tabel twee
  keer**: in de samenvatting én in de DCF-sectie, met een andere kolomvolgorde (fair value staat
  daar op kolom drie, vier of zes). Er wordt daarom alleen binnen de executive summary gezocht en
  de kolom wordt uit de kop bepaald. Bewaakt in `tests/test_analyses.py`.
- **C-regels zwijgen bij een andere valuta of een koppeling via de bedrijfsnaam.** Dezelfde
  Silvano-les als in `oordelen._koers_verschil`: 4,49 PLN en 1,11 EUR zijn dezelfde waarde.
- **De datapoort gaat vóór alles.** `data_status` bad/missing of twee boekjaren achterstand →
  grijs, geen enkele regel getoetst. Elke afleiding die de motor overdoet moet dezelfde poorten
  passeren.
- **"Beter alternatief" (D1) telt nooit mee in het eindoordeel.** De rangorde schuift per
  verversing; er een verkoopoordeel op bouwen geeft elk kwartaal vals alarm. Tonen, niet wegen.
- **Twee eindpunten met opzet.** `GET /api/bezit/tickers` is één query en wordt bij elke
  paginalading opgehaald voor de knopstand; `GET /api/bezit` doet de volledige toetsing (per
  ticker een moat-profiel) en draait alleen als je het tabblad opent. `/api/dashboard` is
  bewust níet uitgebreid — daar gaan 2.759 rijen doorheen en die respons heeft de machine al
  eens in een OOM-crashloop gebracht.
- **`_dashboard_rows(cfg)` is uit `api_dashboard` gelicht** zodat `/api/bezit` exact dezelfde
  rijen gebruikt. Daar zit meer in dan een SELECT: live herrekend signaal, reden-labels,
  `rank_score` en de oordeelkoppeling. Bouw die keten nergens na.
- **Nog niet gedaan:** verkooplogboek, kalibratie-scorebord (hielp verkopen echt?), wekelijkse
  toetsing met historie in `_run_weekly_tasks`, een blok op `stock.html`, en meldingen.

## `/start` — "Wat nu?"

Ticker invullen, terugkrijgen waar dat aandeel staat in de pijplijn en wat de volgende stap is, met de tekst om te plakken erbij. Bedoeld om de route (tussencheck in Claude Code → research in cowork → stage 2 in Claude Code → publiceren) niet te hoeven onthouden.

- **De logica zit in `_routekaart()` in app.py.** Die leest de screener-DB voor de feiten en de mappen `analyses/` en `tussenchecks/` voor de voortgang. Wijzigt de pijplijn, dan is dat de enige plek die bij moet.
- **Korte ticker mag ook**: `RBT` vindt `RBT.PA`. Levert dat meerdere treffers op, dan vraagt de pagina om het volledige symbool.
- **Dual-listing-waarschuwing**: staan er meerdere noteringen onder dezelfde bedrijfsnaam met een koers/winst die meer dan 1,5× uiteenloopt, dan komt er een rood blok bovenaan met welke notering je moet hebben. Zie de valkuil hieronder.

## Valse koopsignalen bij tweede noteringen

`scripts/check_noteringen.py` scant alle tickers hierop; `--zelftest` controleert de detectie zonder de app.

Robertet (2026-08-04) stond met twee noteringen in de screener: RBT.PA (€802, HOLD) en CBR.PA, een investeringscertificaat (€91). Yahoo hing aan beide dezelfde groepscijfers, dus bij CBR werd de winst van het hele bedrijf gedeeld door alleen die kleine klasse: koers/winst 1,8 en een schijnbare korting van 80%, met een BUY op het dashboard. CBR.PA is gedeactiveerd.

Dit is gevaarlijker dan de dual-listings die hierboven al staan (EXOR.AS, ACOMO.BR): daar ontbreken de financials, hier zijn ze van de verkeerde entiteit — het resultaat oogt volkomen geloofwaardig.

**Koersen vergelijken werkt niet om dit te vinden.** Silvano noteert 4,49 PLN in Warschau en 1,11 EUR in Tallinn: dezelfde waarde, andere valuta. De koers/winst is wél valuta-onafhankelijk en loopt alleen uiteen als de winst niet bij die koers hoort.

## Analyses-pagina (`/analyses`)

Toont de fundamentele analyses uit de aandelenanalyse-pipeline; publiek, geen afscherming.

- **Bron zijn de markdown-rapporten**, niet de analyse-JSON's. De MD is altijd de nieuwste versie; een JSON loopt achter tot stage 2 draait. `engine/analyses.py` parseert kop, `## Metadata` en `## 1. Executive summary`, rendert per sectie naar HTML en cachet op mtime.
- **Alleen genummerde secties (1 t/m 15) worden getoond.** Alles zonder nummer is pipeline-administratie: bronnen-inventaris, metadata-blok, afrondingschecklist, opmerkingen voor Claude Code. Filteren op "heeft een nummer" houdt ook toekomstige interne secties buiten de pagina; een blacklist van titels zou dat niet doen.
- **Nieuwe of bijgewerkte analyse publiceren:** `python scripts/sync_analyses.py` (mirror vanuit `C:\Users\janco\aandelenanalyse\research`), dan committen en `fly deploy --remote-only --depot=false`. Hetzelfde script synct ook de tussenchecks.
- **`/tussenchecks` is de tweede set.** Beslisdocumenten van vóór een analyse (VERDIEPEN / TWIJFEL / OVERSLAAN), uit `research/_tussencheck/` naar `tussenchecks/`. Ze hebben géén genummerde secties, dus het sectiefilter van de analyses zou er nul overhouden — `_parse_tussencheck` toont juist álle H2's, inclusief de voorspelling. Ook geen domein of Yahoo-symbool in het document, dus geen logo en geen live koers. Beide pagina's zeggen expliciet dat een tussencheck geen analyse is en niet als bron dient; laat die tekst staan.
- **De koers in de header komt uit twee bronnen.** De peildatum-koers hoort bij het rapport en blijft staan zoals hij is — fair value, upside en oordeel zijn daarop gebaseerd en worden niet herrekend. Daarnaast zet `_verrijk_met_actuele_koers` in app.py de live koers uit `market_data` (gekoppeld via het Yahoo-symbool uit de rapport-metadata) plus de upside op die koers. Zit een ticker niet in de screener of hapert de database, dan vallen die velden weg en toont de pagina alleen de peildatum-cijfers; de analyse zelf heeft de database niet nodig.
- **Parser-wijzigingen: draai de rendertest.** `engine/analyses.py` leest met de hand geschreven rapporten waarin de bullets in vorm variëren, dus elk veld faalt afzonderlijk naar een streepje. De geparseerde scorekaart-totalen horen exact overeen te komen met die in de JSON's van de aandelenanalyse-repo — dat is de goedkoopste kruisvalidatie dat de parser nog klopt.

## Huidige plan / status

**Lopend traject: "Stockscreen 2.0"** — plan in `~/.claude/plans/lovely-enchanting-mitten.md` (6 fases, geschreven om zonder extra context uitvoerbaar te zijn). **Lees dat plan voordat je aan dit project werkt.** Fase 0 en 1 zijn afgerond (2026-07-30/31); fase 2 (herkalibratie naar een bruikbare BUY-lijst) is de volgende stap.

**Fase 1 afgerond (2026-07-31) — refresh-motor v2:**
- **De verversing draait nu op de Fly-machine zelf** ([engine/refresh.py](engine/refresh.py) + scheduler in [app.py](app.py)). GitHub Actions is nog slechts een handmatige noodknop; het `schedule:`-blok is bewust verwijderd.
- **Koersen in bulk** — `refresh_prices_bulk()` haalt ze op in chunks van 200 via `yf.download` in plaats van één voor één. Gemeten: 10 tickers in 4s (was ~45s). Split-guard weigert een koers die >40% afwijkt en haalt die ticker volledig opnieuw op, zodat koers/shares/EPS bij elkaar horen.
- **Storm-guard** — `refresh_fundamentals_batch()` telt failures pas ná afloop. Boven 25% mislukking in één batch gaan we uit van een storing bij Yahoo: `storm_detected` in het log, geen enkele teller omhoog, geen suspensies. Dit is de directe tegenmaatregel tegen de 115 tickers die in april verdwenen.
- **Suspenderen**: pas na **3** fouten, gespreid over **≥21 dagen** (`data_quality.first_failure_at`), én alleen als er nul jaarcijfers in de DB staan. Wekelijkse `weekly_reprobe(40)` geeft gesuspendeerde tickers een nieuwe kans; na **45 dagen** zonder resultaat `presumed_delisted_at` (blijft zichtbaar in Beheer). De drempels staan als `SUSPEND_MIN_FAILURES` / `SUSPEND_MIN_DAYS` / `DELISTED_AFTER_DAYS` in [engine/refresh.py:403](engine/refresh.py#L403). Ze stonden ooit op 10/30/90 en zijn verlaagd omdat een ticker maar eens per elf dagen een beurt krijgt: tien pogingen duurden dan drie maanden en het archief werd pas na een half jaar bereikt. **De bescherming tegen massa-suspensies zit niet in deze getallen maar in de storm-guard** — bij een storing bij Yahoo worden de tellers helemaal niet opgehoogd.
- **NULL-bescherming**: `upsert_financials/market_data/historical_multiples` gebruiken `COALESCE(excluded.col, tabel.col)` via `db._update_clause()`. Een half antwoord van Yahoo wist geen goede cijfers meer. Tijdstempels overschrijven wél altijd. **`upsert_scores` bewust NIET** — dat is een afgeleide tabel die volledig herberekend wordt; daar is een lege uitkomst een echt resultaat en zou COALESCE stale fair values laten blijven staan.
- **Scheduler v2** stuurt uitsluitend op de tabel `refresh_state` (restart-safe), niet meer op de leeftijd van willekeurige marktdata. Koersen na 18:30 Amsterdam, fundamentals na 03:00, wekelijks re-probe + logopschoning (90 dagen). Tick elk kwartier. Uitzetten kan met env `SCHEDULER_ENABLED=0`.
- **`GET /api/health`** (zonder token) toont in één blik of de motor draait: `scheduler_alive`, leeftijd van beide rondes, versheidspercentages, en de dekking (actief/beoordeeld/geen oordeel/gesuspendeerd/delisted). Versheidsbanner in [templates/base.html](templates/base.html) op elke pagina, dekkingsverklaring in de dashboardkop.
- **`POST /api/refresh/prices`** start handmatig een koersronde (body `{"tickers":[...]}` optioneel).

**`POST /api/refresh/fundamentals`** (body `{"limit": N}`, default uit config) draait direct een ronde jaarcijfers, met een knop in `/triage`. Roept dezelfde `_do_fundamentals_refresh` aan als de scheduler, dus storm-guard en suspend-regels gelden identiek. **Er kan er maar één tegelijk lopen**: scheduler en knop delen `_fundamentals_running`; een tweede verzoek krijgt 409 en de nachtelijke ronde slaat zichzelf over met status `skipped`. Reden: twee batches tegelijk verdubbelen de Yahoo-aanroepen per seconde en lopen in de rate-limits. Status opvragen kan via `GET /api/refresh/fundamentals/status`.

**Fase 0 afgerond (2026-07-30):**
- Repo-structuur omgezet: de app staat nu in de **repo-root** (was een submap van een repo die de hele home-directory omvatte). De home-repo is verwijderd. `CLAUDE.md`, `README.md`, `docs/` en enkele scripts stonden niet onder versiebeheer en zijn toegevoegd. `.dockerignore` toegevoegd (venv niet meer in build-context; image 361MB).
- **Waarom de refresh 6 weken stil lag:** GitHub schakelde de scheduled workflow uit met `disabled_inactivity` — dat gebeurt automatisch na 60 dagen zonder push naar de repo. Weer aangezet met `gh workflow enable`. **Elke fase eindigt daarom met een push**; dat is functioneel, niet cosmetisch.
- 115 auto-suspended tickers geheractiveerd via `POST /api/stocks/unsuspend/<T>` (zet active=1, wist de suspend-markering én reset `consecutive_failures` — gebruik dit, niet `bulk-activate`, die de teller laat staan). Dashboard: 798 → 912 actief.
- **Rotatie-bug gefixt.** `/api/cron/next-batch` sorteerde op `financials.fetched_date`. Een ticker waarvoor Yahoo niets teruggeeft krijgt nooit een financials-rij, dus die bleef eeuwig vooraan staan. Met 114 zulke tickers zou elke nachtbatch van 90 volledig uit kansloze tickers bestaan — groene workflow, nul echte verversing. Sorteert nu op `data_quality.last_checked` (wordt bij élke poging geschreven) via `db.get_last_attempt_dates()`.
- **Bevinding over de 115:** na heractivering leverden 114 van hen nul data op, óók opgehaald vanaf Fly. Ze zijn dus niet slachtoffer van de rate-limit-storm maar leeg aan de bron: delisted, overgenomen, hernoemd of verkeerd symbool. Vier zijn aantoonbaar dubbel (WDP.AS↔WDP.BR, BPOST.AS↔BPOST.BR, REN.LS↔REN.AS, AI.AS↔AI.PA — de tweede werkt telkens wel). Ze staan nu zichtbaar als "GEEN DATA" op het dashboard i.p.v. onzichtbaar suspended; fase 1.5 (presumed-delisted archief) en fase 5 (triage) ruimen ze definitief op.

**Fase 2 (herkalibratie) — AFGEROND op de BUY-quota na (2026-07-31):**
- **Eindstand:** SELL 82% → **35,7%** (norm <40% gehaald). HOLD 62,5%. BUY **1,8%** — de norm van 5–15% is *niet* gehaald, bewust. Zie de afweging onderaan.
- **`rank_score` (0–100)** op elke dashboardrij, berekend in `_add_rank_scores` ([app.py](app.py)): percentielpositie op korting (50%), kwaliteit (30%), vertrouwen in de waardering (20%). Dit is de werkelijke shortlist-motor: absolute signalen hangen af van hoe goed de fair values gekalibreerd zijn, een rangorde niet. Top-10 bevat herkenbare namen met echte korting (Evolution, Accenture, Capgemini, Ipsos).
- **Waarom BUY laag blijft:** een koopsignaal eist kwaliteitsscore ≥7 én een koers onder 70% van de fair value. Slechts 13 aandelen halen beide. Om aan 5–15% te komen moet je óf de kwaliteitseis verlagen (koopadvies op zwakke bedrijven) óf de fair values kunstmatig verhogen. Beide maken het signaal betekenisloos. **Niet aan draaien om het quotum te halen** — gebruik `rank_score`.
- **Openstaand vermoeden:** de kwaliteitsscore zelf is mogelijk te streng — mediaan 3 van de 10 over 742 Europese aandelen, 72,5% scoort onder de 6. Als die verdeling naar een realistischer midden schuift komen er vanzelf meer koopkandidaten. Dit is niet onderzocht; `engine/quality_score.py` is het startpunt.

**Fase 3 (UI v2) — grotendeels afgerond (2026-07-31):**
- **Tabbladen vervangen de zes filters** in [templates/index.html](templates/index.html). Elk tabblad beantwoordt één vraag: ⭐ Kansen (top-20 op `rank_score`, standaard open), 🌱 Groeiers, 📌 Mijn lijst, 📋 Alles (met de oude filters), ❓ Geen oordeel, 🔧 Beheer. Tabkeuze in localStorage onder `stockscreen.tab`.
- **Pins** in localStorage onder `stockscreen.pins`, ster-knop per rij. **Nog niet gedaan:** dezelfde ster op de detailpagina.
- **Eén regel uitleg** per rij in de gerichte tabbladen ("Kwaliteit 9/10 · 44% onder geschatte waarde · cijfers compleet").
- **Beslisboom op [templates/stock.html](templates/stock.html)**: vier blokken met stoplichtkleur (Data / Kwaliteit / Waardering / Oordeel). `KWALITEIT_UITLEG` en `PIOTROSKI_UITLEG` vertalen de sleutels uit `quality_breakdown` en `piotroski_breakdown` naar gewone taal. `/api/stock/<T>` geeft nu ook `data_quality` terug (nodig voor blok 1).
- **Afwijking van het plan:** er is géén aparte `/beheer`-pagina gebouwd; het tabblad linkt naar de bestaande `/triage`. Die dekt de triage al; overrides, activity-log en `/api/diagnostics/mos` zijn daar nog niet samengebracht.
- **Niet geverifieerd in een browser** — het Fly-domein is geblokkeerd in de browsertool van deze sessie. Wel getest: JS-syntax via `node --check`, en alle tabselecties plus de beslisboom-logica tegen de echte productiedata (TXT.WA, ASML.AS, RE.PA, DDRIL.OL). **Kijk zelf even of het er goed uitziet.**

**Fase 2 — verloop:**
- **Gedaan:** Graham-formule gebruikte de rendementseis op aandelen (10–12%) op de plek waar Graham de AAA-obligatierente bedoelt. Dat kortte de Graham-waarde met 55–60% en telde het risico dubbel. Nu `valuation.bond_yield: 5.0`. Fair values stegen ~8%. Koopdrempels aangescherpt naar 45/60% van FV. Alles herberekend en gedeployed.
- **Gedaan:** [scripts/calibrate_report.py](scripts/calibrate_report.py) meet de verdeling, mediane marge per sector/markt en een referentieset van 25 bekende namen. Draaien met `--file` als het netwerk TLS onderschept.
- **De hoofdoorzaak was de signaallogica, niet de waardering.** In `determine_signal` ([engine/screener.py](engine/screener.py)) stond `if quality < sell_quality_floor: signal = "SELL"` — ongeacht de prijs, en dat gold voor 72,5% van de portefeuille. UMI.BR noteerde onder zijn fair value en stond tóch op SELL. **Verwijderd:** SELL betekent nu uitsluitend "te duur". Kwaliteit bewaakt nog wel de koopsignalen. `sell_quality_floor` in config.yaml wordt niet meer gebruikt.

**Fase 4 (universe uitbreiden) — importer klaar, onboarding nog niet gestart (2026-08-01):**
- **`import_tickers.py` is herbouwd** tot `--source {euronext,xetra,nasdaq-nordic,baltic,gpw}` en leest de complete noteringslijsten van de beurzen zelf. Elke bron doet auto-download; `--file` is de terugval als een site dichtzit. Handige vlaggen: `--dry-run`, `--limit N` (default 250), `--include-growth`, `--out <csv>`.
- **Gemeten gat: 1.715 noteringen die we nog niet volgen** — 913 → 2.628, zónder Growth/Access/First North. Per markt: PL 341, SE 296, DE 267, FR 254, NO 135, FI 117, DK 89, BE 66, NL 62, IS 23, LT 19, IE 15, PT 12, EE 11, LV 8.
- **Het beurssymbool is niet het Yahoo-symbool.** Balder noteert op Nasdaq als `BALD B`, bij Yahoo als `FAST-B.ST`. Dat geldt voor een deel van élke lijst. Zulke symbolen klakkeloos importeren levert permanente "GEEN DATA"-regels op — precies de rommel die fase 0 heeft opgeruimd. **Importeer daarom altijd met `--probe`.** Dat test elke kandidaat via `POST /api/stocks/probe` (draait op Fly, want daar wérkt Yahoo) en laat alleen symbolen met een koers door; de rest gaat naar `<out>_onvindbaar.csv` als remap-werk.
- **ISIN** staat nu in `stocks` en ontdubbelt dual-listings binnen een batch: bij dezelfde ISIN wint de notering in het thuisland van die ISIN. Ving o.a. NDA-SE.ST/NDA-DK.CO en SAMPO-SEK.ST/SAMPO-DKK.CO af.
- **Bronvalkuilen:** een onbekende MIC in de Euronext-URL breekt het endpoint (301 → lege pagina), dus `MERK` (Growth Oslo) staat er bewust niet in. De GPW-lijst komt van biznesradar.pl omdat gpw.pl vanaf dit netwerk niet bereikbaar is; de grootste Poolse namen staan daar zónder haakjes-notatie ("PZU" i.p.v. "PZU (PZU)") — dat kostte eerst 29 tickers, waaronder PZU en PGE. `tests/test_import_parsers.py` bewaakt dit.
- **Uitgevoerd op 2026-08-01: universe 913 → 2.759.** Janco wilde niet op de twee-weken-gate wachten. Alle vijf bronnen geïmporteerd met `--probe --apply-via-api`. Probe-opbrengst per bron: Baltic 100%, GPW 97%, Nasdaq Nordic 97%, Euronext 92–96%, Xetra 92% (na de `.F`-terugval). ~1.850 toegevoegd, ~70 symbolen afgewezen omdat Yahoo ze niet kent — dat zijn vrijwel allemaal preferente aandelen (`-PREF.ST`) en warrants.
- **`fundamentals_per_night` van 100 → 250** in config.yaml. Met 100 zou de inhaalslag voor ~2.000 nieuwe tickers 20 nachten duren; nu 11, en een volledige ronde blijft op 11 dagen. Niet verder verhogen: elke ticker is een losse Yahoo-aanroep.
- **256MB was te weinig — machine geschaald naar 512MB.** Kort na de import raakte de app in een OOM-crashloop: de gunicorn-worker werd elke ~25s door de kernel gedood (`Out of memory: Killed process (gunicorn) anon-rss:133508kB`), homepage gaf 502 en `/api/dashboard` deed 28s. Oorzaak is niet een refresh-ronde maar de omvang zelf: de basis (pandas + yfinance) kost al ~100MB van de 212MB bruikbaar, en een dashboardrespons over 2.759 rijen paste daar niet meer bij. Na `fly scale memory 512` (~€2/mnd extra): MemAvailable 288MB van 470MB, dashboard 0,77s, homepage 0,34s, geen OOM meer. **Terugschalen naar 256MB kan niet meer zolang het universum boven ~1.500 tickers ligt.**
- **Dekkingsoverzicht per beurs** (wat we wel/niet hebben, incl. VS): https://claude.ai/code/artifact/cf6bc61d-a01f-4f8c-86fd-d0a7e13b216e
- **Grootste resterende gaten:** Verenigde Staten (51 van 7.485 — de Nasdaq Trader symbolenlijst is gratis en getest, zie `scratchpad`-notities), VK (14 van ~1.545), Zwitserland (20 van ~237), Spanje (57 van ~143). Plus ~1.250 groeisegment-noteringen die met `--include-growth` binnen handbereik liggen maar bewust uit staan.

**Wat blijft er bewaard als Yahoo ermee stopt? (uitgezocht 2026-08-01)**
- **Jaarcijfers stapelen op.** `upsert_financials` gebruikt `ON CONFLICT(ticker, period_type, fiscal_year) DO UPDATE` en er staat nergens een `DELETE FROM financials`. Een oud boekjaar wordt dus nooit verwijderd. Yahoo levert een venster van ~5 jaar; de database bouwt daar elk jaar één jaar bovenop. Nu heeft iedereen 2021–2025; in 2031 staat er 2021–2030 terwijl Yahoo er nog steeds vijf toont. Zelfde principe voor `historical_multiples` (sleutel `ticker, fiscal_year`).
- **Koershistorie wordt sinds 2026-08-01 wél bewaard** in de tabel `price_history` (`ticker, date, close`, PK op de eerste twee). Daarvóór hield `market_data` alleen de laatste koers vast — één rij per aandeel — en bestond er dus nergens een koersreeks. Twee wegen erheen: `refresh_prices_bulk` legt alle vijf dagen uit de bulk-download vast (gaten van gemiste dagen vullen zich vanzelf), en `POST /api/price-history/backfill` haalt op wat Yahoo nú nog heeft (`period` 1y/2y/5y/10y/max, `limit` per ronde). **Bestaande dagen worden nooit overschreven** (`ON CONFLICT DO NOTHING`): een vastgelegde slotkoers is een feit en mag niet stilletjes herschreven worden door een herziening of een fout in een bulk-download. De backfill deelt `_fundamentals_running` met de rondes jaarcijfers. Omvang volgen via `GET /api/price-history/stats`; knop in `/triage`.
- **Het archief begint pas bij de eerste geslaagde fetch.** Een boekjaar dat nooit is opgehaald vóórdat Yahoo het liet vallen, is definitief weg. Voor koersen geldt dat niet meer zolang de backfill nog kan draaien — dáárom is die er.
- **Let op de groei:** ~2.800 tickers × ~252 handelsdagen ≈ 700k rijen per jaar. Vijf jaar backfill is eenmalig ~3,5M rijen. Nog niet tegen een grens gelopen, maar wel het eerste om naar te kijken als de Neon-opslag knelt; thinnen naar weekkoersen voor oude jaren is dan de goedkoopste ingreep.

**Moat-blok op de aandeelpagina (2026-08-01):** vijfde blok in `bouwBeslisboom()` ([templates/stock.html](templates/stock.html)), gevoed door [engine/moat_profile.py](engine/moat_profile.py) via `/api/stock/<T>`.
- **Waarom:** uit de 19 afgeronde diepe analyses in `aandelenanalyse` blijkt dat het oordeel bijna volledig door de kwaliteitskant wordt bepaald — de 3 KOOP-oordelen scoorden gemiddeld 12,7/15 op moat+Buffett+management, al het andere 8,8. Vijf aandelen waren goedkoop (DCF-score 4–5) maar zwak, en werden **alle vijf** HOLD. De fair value van de screener was dus niet fout; de ontbrekende dimensie was de concurrentiepositie. `quality_score` meet die niet (Adyen: 5,0 hier tegenover 14/15 diep; Ambra 7,5 tegenover 8/15).
- **Rekent:** ROIC per boekjaar (mediaan + slechtste jaar, NOPAT/(eigen vermogen+schuld), vast belastingtarief 25%), bruto- en operationele marge met trend, cyclustest op `price_history`.
- **Drempels gekalibreerd op 15 bekende gevallen.** Groen eist mediaan ROIC ≥15% **én** slechtste jaar ≥70% van de mediaan. Rood bij standvastigheid <65%, mediaan <6%, of brutomarge-erosie ≥3pp. Uitkomst: groen ging naar precies de drie hoogst scorende (13/14/14), rood naar vijf met score 8–9. **Twee eisen die niet mogen sneuvelen:** geen valse positief wordt groen, geen KOOP-aandeel wordt rood — bewaakt in `tests/test_moat_profile.py`.
- **Geel is bewust ruim** en betekent "de cijfers spreken zich niet uit". Ambra en Telekom Austria ogen cijfermatig prima en scoorden toch 8/15; het middenveld is niet uit getallen te beslissen. Dat is precies waar de skill `tussencheck` in `aandelenanalyse` voor is.

**Koersgrafiek + opslagbeheer (2026-08-02):**
- **Grafiek op de aandeelpagina** ([templates/stock.html](templates/stock.html)): handgetekende SVG met de koersreeks, de geschatte waarde als stippellijn en de band conservatief–optimistisch als vlak. **Geen grafiekbibliotheek** — het project draait zonder build-stap. Valt de waardelijn buiten beeld, dan wordt dat gemeld in plaats van stilzwijgend weggelaten. Data via `GET /api/price-history/<T>?period=1y|5y|max`, uitgedund tot ~400 punten.
- **`price_history` gaat niet meer mee in `/api/stock`.** Daar werden tot 3.650 regels meegestuurd die geen enkele pagina opvroeg (~40 KB per paginalading). Het moat-profiel gebruikt de reeks nog wel, server-side.
- **Overbodige index verwijderd.** `price_history` had `PRIMARY KEY (ticker, date)` én `idx_price_history_ticker` op `(ticker, date DESC)` — dezelfde index, want Postgres leest een btree ook achterstevoren. Kostte ~⅓ van de tabel. **Let op bij nieuwe indexen: controleer eerst of de PK het al dekt.**
- **Wekelijks verdunnen** via `db.thin_price_history()`, aangehaakt in `_run_weekly_tasks()`. Twee jaar dagkoersen (`PRICE_HISTORY_DAGELIJKS_DAGEN = 730`), daarbuiten de laatste handelsdag per ISO-week. Eerste en laatste punt van een reeks blijven altijd staan; de operatie is **idempotent**. Gemeten: vijf jaar gaat van 1.304 naar 679 regels. Handmatig met `POST /api/price-history/thin` (`{"dry_run":true}` toont eerst de opbrengst).
- **De datumkolom blijft bewust `TEXT`.** Omzetten naar `DATE` bespaart ~10 bytes/regel maar herschrijft miljoenen rijen onder een lock en breekt code die de datum als string leest (`cyclustest()` doet `cyc["vanaf"][:4]`).

**Methodepagina `/methode` (2026-08-02):** elke berekening narekenbaar, met de doorrekening van een zelf gekozen aandeel. Gebouwd omdat Janco de screener deelt met een registeraccountant die de cijfers wil kunnen controleren.
- **`GET /api/trace/<T>`** geeft de hele keten terug en rekent zelf **niets** opnieuw uit — het toont wat de motoren produceren. Zou het hier herberekend worden, dan bestaat er een tweede waarheid naast de echte berekening.
- **De motoren geven hun tussenwaarden nu weg**: `quality_score()` levert `detail` per criterium (ROE/ROIC per jaar, schuldratio, rentedekking, variatiecoëfficiënt), `piotroski_fscore()` levert `waarden` met bééde vergeleken getallen per test, en `normalizer.normalize_metric_trace()` toont ruwe jaarwaarden, IQR-grenzen en wat afviel. **Niets daarvan wordt opgeslagen** — `upsert_scores` neemt alleen benoemde kolommen.
- **Valkuil die dit blootlegde:** `run_ticker` rekent over `calc_rows` = de TTM-rij als **boekjaar 0** plus de jaarrijen, zodra het jongste jaarverslag van vorig jaar is. De trace normaliseerde eerst over alleen de jaarrijen en toonde daardoor 19,92 waar de motor 20,59 gebruikte. `run_ticker` geeft nu `calc_rows` terug en de trace gebruikt díé. **Wie hier iets aan verandert: controleer dat de trace dezelfde rijen gebruikt als de motor.** De pagina labelt boekjaar 0 als "laatste 12 mnd".
- **`tests/test_trace.py`** bewaakt dat de uitkomst volgt uit de getoonde waarden: mediaan uit de gebruikte waarden, gebruikt+afgevallen = complete invoer, grenzen verklaren wat afviel, elke Piotroski-test volgt uit zijn eigen twee getallen.
- **De methodologie in `settings.html` is verwijderd** (252 regels) en vervangen door een verwijzing. Die kopie liep al achter: hij beschreef criterium 4 zonder de cashconversie-eis die de code wél stelt.

**Landdetectie hersteld (2026-08-01):** `_detect_market` kende acht beurzen en zette al het overige op `"US"`. Zolang de watchlist uit precies die acht landen bestond viel dat niet op; na de uitbreiding naar 27 landen stond **38% van de aandelen verkeerd** (Parijs 326, Milaan 201, Helsinki 134, Kopenhagen 114, Madrid 57, …). Er zat bovendien een tikfout in die nooit was opgevallen: Finland werd gecontroleerd op `.FI` terwijl Yahoo `.HE` gebruikt.
- **De oorzaak was een dubbele tabel** — `SUFFIX_INFO` in `import_tickers.py` plus een tweede, kleinere kopie als if-statements in de fetcher. Die staan nu op één plek: [engine/markets.py](engine/markets.py). **Komt er een beurs bij, dan is dat het enige bestand dat aangepast hoeft te worden.**
- De importer zette het land wél goed, maar `fetch_and_store` schreef `_detect_market` er bij elke ronde overheen.
- **Een onbekend suffix geeft nu het suffix zelf terug**, niet `"US"`. Een ontbrekende beurs hoort op te vallen; dat stille terugvallen wás de fout. Bewaakt in `tests/test_markets.py`.
- **`POST /api/stocks/recompute-markets`** (body `{"dry_run":true}` om eerst te kijken) herstelt bestaande rijen in één keer. Draai dit ook na elke uitbreiding van `markets.py` — anders duurt het tot de rotatie elke ticker weer heeft aangeraakt.
- Geen fair value of signaal is hierdoor beïnvloed; `market` is een etiket. Het landenfilter op het dashboard leidt het land al uit het suffix af en was dus altijd correct.

**Landenfilter (2026-08-01):** staat boven de tabbladen in [templates/index.html](templates/index.html) en werkt op álle tabbladen. **Leidt het land af uit het beurssuffix, niet uit `stocks.market`** — dat veld is nooit genormaliseerd (dezelfde beurs staat er als `NL`, `Nederland` én `US` in), dus filteren daarop geeft onvolledige lijsten. `LAND_PER_SUFFIX` dekt alle 27 landen in de huidige dataset; `.DE` en `.F` tellen samen als Duitsland. Keuze in localStorage onder `stockscreen.land`.

**Open na fase 0:** ~~DATABASE_URL-rotatie~~ (gedaan 2026-07-31).
**Losse eindjes:** er hangt nog een oude **Render**-service aan deze repo die bij elke push probeert te bouwen en faalt (Janco krijgt faalmails). Controleren of daar nog een oude versie draait die naar dezelfde Neon-database schrijft — zo ja, dat is een tweede schrijver op dezelfde data. De `Procfile` in de repo is een restant daarvan.

**Refactor afgerond (2026-04-18)**: async batch-endpoint is vervangen door per-ticker endpoints + workflow die zelf de loop doet. Betrouwbaar omdat elke HTTP-call <100s is en geen in-memory state vereist.

**INSUFFICIENT-DATA aanpak afgerond (2026-06)** — zie [docs/DIAGNOSE_INSUFFICIENT_DATA.md](docs/DIAGNOSE_INSUFFICIENT_DATA.md):
- **Diagnose**: `/api/gaps-report` (per-ticker primaire blocker-bucket, hergebruikt `data_quality.classify_blockers`) + [scripts/gaps_analyze.py](scripts/gaps_analyze.py). Bevinding: het was géén data-tekort maar een te strenge gate.
- **Route A (gate-herkalibratie)**: split-detectie verscherpt (abs EPS-floor + clean-factor), structureel verlies/FCF + buyback-equity niet langer blokkerend. INSUFFICIENT DATA 261→64, 228 gered, 0 regressies. Toegepast via `POST /api/data-quality/recompute` (no-network re-eval uit cache; `{"dry_run":true}` geeft before/after-transities) gevolgd door `POST /api/recalculate`.
- **Reden-weergave**: dashboard toont 3 onderscheidende labels i.p.v. één rood "INSUFFICIENT DATA" — **GEEN DATA** (grijs), **DATABUG** (paars), **GEEN FV (VERLIES)** (oranje), via `data_quality.classify_signal_reason`. Verlieslatende groeiers krijgen 🌱 (`is_growth_lossmaker`, drempel `screening.growth_lossmaker_cagr`). Vereist `fv_methods_dropped` + `revenue_cagr` in `calculated_scores` (gevuld door recalculate).

**Open werk** (grotendeels belegd in het 2.0-plan hierboven):
1. **Route B** (gate-geblokkeerde tickers): dual-listings remappen (EXOR.AS → EXO.MI), multi-share-class (Roche/Lindt), holdings met negatieve omzet (Exor/Kinnevik), foute ticker-notaties (`NASDAQ:ICLR` → `ICLR`), fondsen deactiveren, overrides via `/api/overrides/<T>`. → wordt de kwartaal-APK + skill in fase 5.
2. **Nog niet gedaan** — markt-naam normalisatie (NL/SE/PL abbreviaties).

## Handige commands

```bash
# Deploy (vanuit de repo-root)
fly deploy --remote-only

# Log-stream
fly logs -a stockscreen-janco

# Secret checken (toont hash, niet waarde)
fly secrets list -a stockscreen-janco

# Universe-gat bekijken zonder iets te wijzigen
python import_tickers.py --source euronext --dry-run --limit 99999

# Echte onboarding-ronde (altijd met --probe, anders komen er dode symbolen in)
DATABASE_URL="..." python import_tickers.py --source gpw --probe --limit 250

# Handmatige workflow-run: GitHub → Actions → "Daily refresh" → Run workflow

# Sanity-test endpoint (zonder token → verwacht 401)
curl -k -sS -o /dev/null -w "%{http_code}\n" https://stockscreen-janco.fly.dev/api/cron/next-batch

# Dashboard-data inspecteren
curl -ks https://stockscreen-janco.fly.dev/api/dashboard | jq '[.[] | select(.signal=="INSUFFICIENT DATA") | .ticker]'
```

## Architectuur-details

Zie [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
