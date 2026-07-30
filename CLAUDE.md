# Stockscreen — operational notes for Claude

Dutch-language stock screener voor Europese small/mid-caps. Flask + PostgreSQL (Neon) + yfinance, gehost op Fly.io.

## Kerngegevens

| | |
|---|---|
| **Live** | https://stockscreen-janco.fly.dev |
| **Fly app** | `stockscreen-janco` (region `ams`, 256mb, 1 worker / 8 threads, gunicorn 120s timeout) |
| **DB** | Neon PostgreSQL, Frankfurt |
| **GitHub** | github.com/Nolles15/stockscreen (branch `main`) |
| **Local dev** | `DATABASE_URL="..." python app.py` → http://localhost:5001 |
| **Deploy** | `fly deploy --remote-only` (vanuit de repo-root) — **bij TLS-fout op de depot.dev-builder** (`x509: certificate signed by unknown authority`): voeg `--depot=false` toe zodat Fly's eigen remote-builder wordt gebruikt. |

## Kritieke operationele feiten

**Native currency pipeline** — prijs én fair value zijn altijd in de native currency van het aandeel. Er is GEEN valutaconversie; `price_eur`-kolommen zijn in een eerdere migratie verwijderd. Als een endpoint 500 geeft met `UndefinedColumn`, is de code achter op de DB-migratie → `fly deploy`.

**Externe cron via GitHub Actions** — sinds de per-ticker refactor (commit c97758a) gebruikt de nachtelijke refresh een stateless loop:
- Workflow: [.github/workflows/daily-refresh.yml](.github/workflows/daily-refresh.yml)
- Cron: `0 3 * * *` (03:00 UTC), default `limit=90`
- Endpoints: `GET /api/cron/next-batch?limit=N` + `POST /api/cron/refresh-one/<T>`
- Secrets: `CRON_TOKEN` (Fly + GH identiek), `APP_URL` (GH)
- Als `CRON_TOKEN` gezet is op Fly, slaat `_on_startup` de in-process scheduler over — dubbele refreshes voorkomen.

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
- **ETFs** — screener werkt niet voor ETFs (bv. BFIT). Deactiveer met `active=0`.
- **Fly auto-stop UIT** — `fly.toml` heeft `auto_stop_machines = false` + `min_machines_running = 1`. Dat is expres: per-ticker cron verwacht een levende machine.

## Huidige plan / status

**Refactor afgerond (2026-04-18)**: async batch-endpoint is vervangen door per-ticker endpoints + workflow die zelf de loop doet. Betrouwbaar omdat elke HTTP-call <100s is en geen in-memory state vereist.

**INSUFFICIENT-DATA aanpak afgerond (2026-06)** — zie [docs/DIAGNOSE_INSUFFICIENT_DATA.md](docs/DIAGNOSE_INSUFFICIENT_DATA.md):
- **Diagnose**: `/api/gaps-report` (per-ticker primaire blocker-bucket, hergebruikt `data_quality.classify_blockers`) + [scripts/gaps_analyze.py](scripts/gaps_analyze.py). Bevinding: het was géén data-tekort maar een te strenge gate.
- **Route A (gate-herkalibratie)**: split-detectie verscherpt (abs EPS-floor + clean-factor), structureel verlies/FCF + buyback-equity niet langer blokkerend. INSUFFICIENT DATA 261→64, 228 gered, 0 regressies. Toegepast via `POST /api/data-quality/recompute` (no-network re-eval uit cache; `{"dry_run":true}` geeft before/after-transities) gevolgd door `POST /api/recalculate`.
- **Reden-weergave**: dashboard toont 3 onderscheidende labels i.p.v. één rood "INSUFFICIENT DATA" — **GEEN DATA** (grijs), **DATABUG** (paars), **GEEN FV (VERLIES)** (oranje), via `data_quality.classify_signal_reason`. Verlieslatende groeiers krijgen 🌱 (`is_growth_lossmaker`, drempel `screening.growth_lossmaker_cagr`). Vereist `fv_methods_dropped` + `revenue_cagr` in `calculated_scores` (gevuld door recalculate).

**Open werk**:
1. **Route B** (de resterende 33 gate-geblokkeerde): dual-listings remappen (EXOR.AS → EXO.MI), multi-share-class (Roche/Lindt), holdings met negatieve omzet (Exor/Kinnevik), foute ticker-notaties (`NASDAQ:ICLR` → `ICLR`), 6 fondsen + 2 lege deactiveren, en overrides via `/api/overrides/<T>`.
2. **Nog niet gedaan** — markt-naam normalisatie (NL/SE/PL abbreviaties).
3. **Nog niet gedaan** — DATABASE_URL rotatie (credential exposure in oudere chat-historie).

## Handige commands

```bash
# Deploy (vanuit de repo-root)
fly deploy --remote-only

# Log-stream
fly logs -a stockscreen-janco

# Secret checken (toont hash, niet waarde)
fly secrets list -a stockscreen-janco

# Handmatige workflow-run: GitHub → Actions → "Daily refresh" → Run workflow

# Sanity-test endpoint (zonder token → verwacht 401)
curl -k -sS -o /dev/null -w "%{http_code}\n" https://stockscreen-janco.fly.dev/api/cron/next-batch

# Dashboard-data inspecteren
curl -ks https://stockscreen-janco.fly.dev/api/dashboard | jq '[.[] | select(.signal=="INSUFFICIENT DATA") | .ticker]'
```

## Architectuur-details

Zie [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
