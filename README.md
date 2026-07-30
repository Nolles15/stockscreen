# Stockscreen

Dutch-language stock screener voor Europese small/mid-caps. Flask + PostgreSQL + yfinance, gehost op Fly.io.

**Live:** https://stockscreen-janco.fly.dev

## Wat doet dit?

- Haalt dagelijks financials en koersen op voor ~620 tickers (ex-US, voornamelijk EU + UK + NO/SE/DK/PL/CH/IL).
- Berekent een fair value per aandeel met een mix van multiples (P/E, EV/EBITDA, P/B, EV/FCF) en DCF-achtige methodes (Graham + Perpetuity).
- Toont een signal (STRONG BUY / BUY / HOLD / SELL / INSUFFICIENT DATA) op basis van margin of safety + quality score.
- Laat je handmatige overrides invoeren wanneer Yahoo-data gaten of fouten heeft (bv. uit een jaarverslag).

Alle prijzen en fair values staan in de **native currency** van het aandeel — er is geen FX-conversie.

## Snel starten (lokaal)

```bash
python -m venv venv && venv\Scripts\activate     # Windows
pip install -r requirements.txt

# Neon-connectie zetten (pooled connection string)
set DATABASE_URL=postgresql://...neon.tech/...?sslmode=require

python app.py    # http://localhost:5001
```

Eerste keer? Importeer eerst tickers uit het Excel-bestand met indices:

```bash
python import_tickers.py "pad/naar/indices.xlsx" --dry-run   # check
python import_tickers.py "pad/naar/indices.xlsx"             # echt importeren
```

Daarna vanuit de UI op **"Alles verversen"** klikken of per ticker via `/api/refresh/<ticker>`.

## Deploy

```bash
fly deploy --remote-only
```

Secrets staan op Fly (`fly secrets list -a stockscreen-janco`):
- `DATABASE_URL` — Neon pooled URL
- `CRON_TOKEN` — bearer token voor `/api/cron/*` endpoints (ook als GitHub secret)

## Automatische refresh

Een GitHub Actions workflow ([`.github/workflows/daily-refresh.yml`](../.github/workflows/daily-refresh.yml), in de **repo-root**, niet in `stockscreen/`) draait elke nacht om 03:00 UTC. Workflow doet per-ticker synchrone HTTP-calls naar Fly (zie [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#cron-ontwerp) voor waarom).

Handmatig draaien: GitHub → Actions → **Daily refresh** → Run workflow → optioneel `limit=500` voor een volledige pass.

## Config

Sector-defaults (groei, P/E, EV/EBITDA, etc.) en signal-drempels staan in [`config.yaml`](config.yaml). Wijzigingen zijn direct actief na een refresh — er is geen code-deploy nodig om multiples aan te passen.

## Architectuur & operationele notes

- **Architectuur + FV-formules:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Operationele valkuilen en lopend werk:** [CLAUDE.md](CLAUDE.md)

## Stack

Flask 3 · PostgreSQL (Neon Frankfurt) · psycopg2 · yfinance · pandas · PyYAML · gunicorn · Fly.io (ams, 256mb)
