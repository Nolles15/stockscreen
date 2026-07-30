"""
Migratiescript: lokale SQLite → Railway PostgreSQL

Kopieert:
  - stocks (alle 222 tickers met metadata)
  - overrides (handmatig ingevoerde cijfers)

Financiële data en scores worden daarna opnieuw opgehaald via de UI (Refresh all).

Gebruik:
  DATABASE_URL="postgresql://..." python migrate_to_postgres.py
"""

import os
import sqlite3
import sys

import psycopg2
from psycopg2.extras import RealDictCursor

SQLITE_PATH = os.path.join(os.path.dirname(__file__), "data", "stocks.db")

DATABASE_URL = os.environ.get("DATABASE_URL", "")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not DATABASE_URL:
    print("ERROR: Stel DATABASE_URL in als environment variable.")
    print("  Voorbeeld: DATABASE_URL='postgresql://...' python migrate_to_postgres.py")
    sys.exit(1)

if not os.path.exists(SQLITE_PATH):
    print(f"ERROR: SQLite database niet gevonden op {SQLITE_PATH}")
    sys.exit(1)


def migrate():
    print(f"Verbinden met SQLite: {SQLITE_PATH}")
    sqlite = sqlite3.connect(SQLITE_PATH)
    sqlite.row_factory = sqlite3.Row

    print(f"Verbinden met PostgreSQL...")
    pg = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    pg_cur = pg.cursor()

    # -----------------------------------------------------------------------
    # Stocks — lees uit calculated_scores (stocks tabel bestaat niet in SQLite)
    # -----------------------------------------------------------------------
    cols = [c[1] for c in sqlite.execute("PRAGMA table_info(calculated_scores)").fetchall()]
    stocks = sqlite.execute("SELECT * FROM calculated_scores ORDER BY ticker").fetchall()
    print(f"\nMigreren: {len(stocks)} tickers uit calculated_scores...")

    for row in stocks:
        r = dict(row)
        ticker   = r.get("ticker")
        name     = r.get("name")     if "name"     in cols else None
        sector   = r.get("sector")   if "sector"   in cols else None
        market   = r.get("market")   if "market"   in cols else None
        currency = r.get("currency") if "currency" in cols else None
        active   = r.get("active", 1)

        pg_cur.execute("""
            INSERT INTO stocks (ticker, name, sector, market, currency, active, added_date)
            VALUES (%s, %s, %s, %s, %s, %s, NOW()::TEXT)
            ON CONFLICT(ticker) DO UPDATE SET
                name=COALESCE(excluded.name, stocks.name),
                sector=COALESCE(excluded.sector, stocks.sector),
                market=COALESCE(excluded.market, stocks.market),
                currency=COALESCE(excluded.currency, stocks.currency),
                active=excluded.active
        """, (ticker, name, sector, market, currency, active))

    print(f"  OK {len(stocks)} tickers gemigreerd")

    # -----------------------------------------------------------------------
    # Overrides
    # -----------------------------------------------------------------------
    overrides = sqlite.execute("SELECT * FROM overrides").fetchall()
    print(f"\nMigreren: {len(overrides)} overrides...")

    for row in overrides:
        r = dict(row)
        pg_cur.execute("""
            INSERT INTO overrides (ticker, field_name, fiscal_year, value, note, created_date)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT(ticker, field_name, fiscal_year) DO UPDATE SET
                value=excluded.value,
                note=excluded.note,
                created_date=excluded.created_date
        """, (
            r.get("ticker"), r.get("field_name"), r.get("fiscal_year"),
            r.get("value"), r.get("note"), r.get("created_date"),
        ))

    print(f"  OK {len(overrides)} overrides gemigreerd")

    pg.commit()
    pg_cur.close()
    pg.close()
    sqlite.close()

    print("\nOK Migratie compleet!")
    print("Ga nu naar de app en klik 'Refresh all' om financiële data opnieuw op te halen.")


if __name__ == "__main__":
    migrate()
