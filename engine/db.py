"""
Database layer — PostgreSQL via psycopg2.
All tables are created on first run; no migration scripts needed.
"""

import logging
import os
import json
from contextlib import contextmanager
from datetime import datetime, timedelta

import psycopg2
from psycopg2.extras import RealDictCursor

log = logging.getLogger(__name__)

# Railway geeft soms postgres:// in plaats van postgresql://
_raw_url = os.environ.get("DATABASE_URL", "")
DATABASE_URL = (
    _raw_url.replace("postgres://", "postgresql://", 1)
    if _raw_url.startswith("postgres://")
    else _raw_url
)


@contextmanager
def _cursor():
    """Levert een RealDictCursor en beheert de transactie + verbinding."""
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is niet ingesteld. "
            "Voeg DATABASE_URL toe als environment variable."
        )
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    try:
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
    finally:
        conn.close()


def init_db() -> None:
    with _cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS activity_log (
                id          SERIAL PRIMARY KEY,
                timestamp   TEXT NOT NULL,
                action      TEXT NOT NULL,
                ticker      TEXT,
                status      TEXT NOT NULL,
                details     TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stocks (
                ticker             TEXT PRIMARY KEY,
                name               TEXT,
                sector             TEXT,
                market             TEXT,
                currency           TEXT,
                financial_currency TEXT,
                active             INTEGER DEFAULT 1,
                added_date         TEXT,
                notes              TEXT,
                description        TEXT,
                auto_suspended_at  TEXT,
                auto_suspend_reason TEXT
            )
        """)
        # Migratie: kolommen toevoegen aan bestaande DB (PostgreSQL)
        cur.execute("""
            ALTER TABLE stocks ADD COLUMN IF NOT EXISTS auto_suspended_at TEXT
        """)
        cur.execute("""
            ALTER TABLE stocks ADD COLUMN IF NOT EXISTS auto_suspend_reason TEXT
        """)
        cur.execute("""
            ALTER TABLE stocks ADD COLUMN IF NOT EXISTS financial_currency TEXT
        """)
        cur.execute("""
            ALTER TABLE stocks ADD COLUMN IF NOT EXISTS quote_type TEXT
        """)
        # ISIN maakt dual-listings herkenbaar: twee tickers met dezelfde ISIN
        # zijn hetzelfde bedrijf op twee beurzen. Gevuld door import_tickers.py.
        cur.execute("""
            ALTER TABLE stocks ADD COLUMN IF NOT EXISTS isin TEXT
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS financials (
                id                  SERIAL PRIMARY KEY,
                ticker              TEXT NOT NULL,
                period_type         TEXT NOT NULL,
                fiscal_year         INTEGER,
                revenue             REAL,
                ebit                REAL,
                ebitda              REAL,
                net_income          REAL,
                eps_diluted         REAL,
                operating_cf        REAL,
                capex               REAL,
                fcf                 REAL,
                total_assets        REAL,
                total_equity        REAL,
                total_debt          REAL,
                current_assets      REAL,
                current_liabilities REAL,
                net_ppe             REAL,
                book_value_ps       REAL,
                roe                 REAL,
                gross_profit        REAL,
                interest_expense    REAL,
                shares_outstanding  REAL,
                net_cash            REAL,
                inventory           REAL,
                fetched_date        TEXT,
                UNIQUE(ticker, period_type, fiscal_year),
                FOREIGN KEY (ticker) REFERENCES stocks(ticker) ON DELETE CASCADE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                ticker               TEXT PRIMARY KEY,
                price                REAL,
                market_cap           REAL,
                enterprise_value     REAL,
                pe_ttm               REAL,
                ev_ebitda_ttm        REAL,
                pb_ratio             REAL,
                last_updated         TEXT,
                analyst_target_raw   REAL,
                analyst_consensus    TEXT,
                analyst_n            INTEGER,
                FOREIGN KEY (ticker) REFERENCES stocks(ticker) ON DELETE CASCADE
            )
        """)
        # Migratie: native-only — drop oude EUR-geconverteerde kolommen
        for col in ("price_eur", "market_cap_eur", "enterprise_value_eur", "analyst_target_eur"):
            cur.execute(f"ALTER TABLE market_data DROP COLUMN IF EXISTS {col}")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS historical_multiples (
                id          SERIAL PRIMARY KEY,
                ticker      TEXT NOT NULL,
                fiscal_year INTEGER NOT NULL,
                pe_ratio    REAL,
                ev_ebitda   REAL,
                pb_ratio    REAL,
                ev_fcf      REAL,
                UNIQUE(ticker, fiscal_year),
                FOREIGN KEY (ticker) REFERENCES stocks(ticker) ON DELETE CASCADE
            )
        """)
        # Koershistorie. Dit is de enige reeks die we niet kunnen terughalen als
        # Yahoo ermee stopt: `market_data` houdt alleen de laatste koers, en
        # jaarcijfers stapelen zichzelf wel op (een oud boekjaar wordt nooit
        # verwijderd) maar een koersreeks bestaat nergens anders in de database.
        # Bewust smal gehouden — alleen de slotkoers, geen volume of intraday —
        # zodat de tabel bij ~2.800 tickers beheersbaar blijft.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                ticker TEXT NOT NULL,
                date   TEXT NOT NULL,
                close  REAL NOT NULL,
                PRIMARY KEY (ticker, date),
                FOREIGN KEY (ticker) REFERENCES stocks(ticker) ON DELETE CASCADE
            )
        """)
        # Deze index was overbodig en is weer weggehaald. `PRIMARY KEY
        # (ticker, date)` is al een btree op precies die twee kolommen, en
        # Postgres leest een index net zo goed achterstevoren — de DESC voegde
        # niets toe. Hij kostte wel ongeveer een derde van de voetafdruk van de
        # tabel, wat bij miljoenen koersregels flink aantikt.
        cur.execute("DROP INDEX IF EXISTS idx_price_history_ticker")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS calculated_scores (
                ticker                  TEXT PRIMARY KEY,
                quality_score           REAL,
                quality_breakdown       TEXT,
                piotroski_score         REAL,
                piotroski_breakdown     TEXT,
                normalized_eps          REAL,
                normalized_ebitda       REAL,
                normalized_fcf          REAL,
                normalized_owner_earn   REAL,
                multiples_fv             REAL,
                graham_fv                REAL,
                perpetuity_fv            REAL,
                combined_fv              REAL,
                conservative_fv          REAL,
                base_fv                  REAL,
                optimistic_fv            REAL,
                fv_confidence           TEXT,
                fv_spread_pct           REAL,
                fv_methods_used         INTEGER,
                signal                  TEXT,
                margin_of_safety        REAL,
                warnings                TEXT,
                last_calculated         TEXT,
                accruals_ratio          REAL,
                hist_relative           TEXT,
                FOREIGN KEY (ticker) REFERENCES stocks(ticker) ON DELETE CASCADE
            )
        """)
        # Fase 4: FV-robustness kolommen toevoegen aan bestaande DB
        for col, typ in (
            ("fv_confidence",   "TEXT"),
            ("fv_spread_pct",   "REAL"),
            ("fv_methods_used", "INTEGER"),
            ("fv_methods_dropped", "TEXT"),   # JSON: redenen per weggevallen methode
            ("revenue_cagr",    "REAL"),      # 3-jaars omzet-CAGR (voor groei-markering)
            ("implied_growth",  "REAL"),      # ingeprijsde groei %/jr (omgekeerde som, 2026-08-08)
        ):
            cur.execute(f"ALTER TABLE calculated_scores ADD COLUMN IF NOT EXISTS {col} {typ}")
        # Migratie: native-only — rename oude *_eur kolommen zodat historische
        # data behouden blijft. PostgreSQL kent geen IF EXISTS bij RENAME COLUMN,
        # dus check de information_schema en rename alleen wat nog bestaat.
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'calculated_scores'
        """)
        existing_cols = {r["column_name"] for r in cur.fetchall()}
        for old, new in (
            ("multiples_fv_eur",    "multiples_fv"),
            ("graham_fv_eur",       "graham_fv"),
            ("perpetuity_fv_eur",   "perpetuity_fv"),
            ("combined_fv_eur",     "combined_fv"),
            ("conservative_fv_eur", "conservative_fv"),
            ("base_fv_eur",         "base_fv"),
            ("optimistic_fv_eur",   "optimistic_fv"),
        ):
            if old in existing_cols and new not in existing_cols:
                cur.execute(f"ALTER TABLE calculated_scores RENAME COLUMN {old} TO {new}")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS overrides (
                id           SERIAL PRIMARY KEY,
                ticker       TEXT NOT NULL,
                field_name   TEXT NOT NULL,
                fiscal_year  INTEGER,
                value        REAL,
                note         TEXT,
                created_date TEXT,
                UNIQUE(ticker, field_name, fiscal_year),
                FOREIGN KEY (ticker) REFERENCES stocks(ticker) ON DELETE CASCADE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS exchange_rates (
                currency     TEXT PRIMARY KEY,
                rate_to_eur  REAL,
                last_updated TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS data_quality (
                ticker           TEXT PRIMARY KEY,
                completeness_pct REAL,
                years_available  INTEGER,
                latest_fy        INTEGER,
                freshness_days   INTEGER,
                fetch_success    INTEGER,
                consecutive_failures INTEGER DEFAULT 0,
                data_status      TEXT,
                issues           TEXT,
                last_checked     TEXT,
                FOREIGN KEY (ticker) REFERENCES stocks(ticker) ON DELETE CASCADE
            )
        """)
        # Migratie: kolom toevoegen aan bestaande DB (idempotent)
        cur.execute("""
            ALTER TABLE data_quality ADD COLUMN IF NOT EXISTS consecutive_failures INTEGER DEFAULT 0
        """)
        # Wanneer de huidige reeks failures begon. Suspenderen mag pas als een
        # ticker langdurig faalt, niet na een handvol fouten op één avond.
        cur.execute("""
            ALTER TABLE data_quality ADD COLUMN IF NOT EXISTS first_failure_at TEXT
        """)
        # Tickers die zo lang niets opleveren dat ze vrijwel zeker delisted zijn.
        # Ze blijven zichtbaar in het beheerscherm, maar vallen uit de rotatie.
        cur.execute("""
            ALTER TABLE stocks ADD COLUMN IF NOT EXISTS presumed_delisted_at TEXT
        """)
        # Scheduler-state. De scheduler bewaart hier wanneer elke taak voor het
        # laatst draaide, zodat een herstart van de machine geen dubbele run
        # veroorzaakt en geen enkele run overslaat.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS refresh_state (
                key   TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        # Wat Janco bezit. Bewust zonder aantal en zonder aankoopkoers: de
        # verkoopregels kijken er niet naar en horen dat ook niet te doen (zie
        # engine/exit_regels.py). Wat hier wél staat is `these_snapshot` — de
        # cijfers zoals ze eruitzagen op het moment van vastleggen, want zonder
        # dat ijkpunt is "verslechterd" niet te onderscheiden van "was altijd al zo".
        #
        # Geen buitenlandse sleutel naar `stocks` met ON DELETE CASCADE: een
        # ticker die uit de screener verdwijnt (delisting, hernoeming) mag jouw
        # vastlegging niet stilletjes wissen. Dan hoort er een lege regel te
        # staan die opvalt.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bezit (
                ticker         TEXT PRIMARY KEY,
                sinds          TEXT,
                notitie        TEXT,
                these_snapshot TEXT
            )
        """)

        # Analyses en tussenchecks als inhoud, niet als bestand in het image.
        #
        # Ze stonden in de Docker-build, dus een nieuwe analyse vroeg om syncen,
        # committen én deployen — drie handelingen op precies het moment dat het
        # echte werk klaar is, en dus werden ze vergeten. Op 20 augustus bleek
        # dat: vier analyses stonden alleen op de laptop, niet eens gecommit.
        #
        # De database is hier het transport. Bij het opstarten worden de
        # bestanden teruggeschreven naar schijf, zodat de leeslaag in
        # analyses.py onveranderd blijft werken.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS analyse_bestand (
                soort     TEXT NOT NULL,
                naam      TEXT NOT NULL,
                inhoud    TEXT NOT NULL,
                gewijzigd TEXT,
                PRIMARY KEY (soort, naam)
            )
        """)

        # Wat je met een oordeel gedaan hebt — en vooral: wat je niet gedaan hebt.
        #
        # Aanleiding: vijf VERDIEPEN-oordelen, geen daarvan gekocht. Janco's eigen
        # verklaring was "dat was niet echt een bewuste beslissing, ik heb het
        # gewoon niet gedaan". Daar zit het probleem: niet-handelen laat geen spoor
        # na, dus valt er ook niets van te leren. Een lege `keuze` betekent hier
        # precies dat — stilzwijgend niets gedaan — en dat is een toestand waar het
        # systeem naar hoort te vragen in plaats van hem onzichtbaar te laten.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS besluit (
                id             SERIAL PRIMARY KEY,
                ticker         TEXT NOT NULL,
                aanleiding     TEXT,
                oordeel        TEXT,
                datum_oordeel  TEXT,
                koers_toen     REAL,
                valuta         TEXT,
                these_snapshot TEXT,
                keuze          TEXT,
                reden          TEXT,
                datum_keuze    TEXT,
                UNIQUE(ticker, datum_oordeel)
            )
        """)


# ---------------------------------------------------------------------------
# Refresh-state (scheduler)
# ---------------------------------------------------------------------------

def get_refresh_state(key: str) -> str | None:
    with _cursor() as cur:
        cur.execute("SELECT value FROM refresh_state WHERE key=%s", (key,))
        row = cur.fetchone()
    return row["value"] if row else None


def set_refresh_state(key: str, value: str) -> None:
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO refresh_state (key, value) VALUES (%s, %s)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )


# ---------------------------------------------------------------------------
# Analyse-bestanden
# ---------------------------------------------------------------------------

def analyse_bestand_opslaan(soort: str, naam: str, inhoud: str) -> bool:
    """Sla een analyse of tussencheck op. True als de inhoud veranderd is."""
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO analyse_bestand (soort, naam, inhoud, gewijzigd)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT(soort, naam) DO UPDATE
               SET inhoud = excluded.inhoud, gewijzigd = excluded.gewijzigd
             WHERE analyse_bestand.inhoud IS DISTINCT FROM excluded.inhoud
            """,
            (soort, naam, inhoud, datetime.utcnow().isoformat()),
        )
        return cur.rowcount > 0


def analyse_bestanden() -> list[dict]:
    """Alle opgeslagen analyses en tussenchecks, met inhoud."""
    with _cursor() as cur:
        cur.execute("SELECT soort, naam, inhoud, gewijzigd FROM analyse_bestand")
        return [dict(r) for r in cur.fetchall()]


def analyse_bestand_verwijderen(soort: str, naam: str) -> bool:
    with _cursor() as cur:
        cur.execute("DELETE FROM analyse_bestand WHERE soort=%s AND naam=%s", (soort, naam))
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Besluiten — wat je met een oordeel deed, en wat je niet deed
# ---------------------------------------------------------------------------

def besluit_vastleggen(ticker: str, aanleiding: str, oordeel: str,
                       datum_oordeel: str, koers_toen: float | None = None,
                       valuta: str | None = None) -> None:
    """Leg vast dat er een oordeel ligt waar nog niets mee gedaan is.

    Draait bij elke keer dat een oordeel gezien wordt, dus moet stil blijven als
    de regel er al staat: een bestaande keuze en reden mogen nooit overschreven
    worden door het enkele feit dat het oordeel er nog steeds is.
    """
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO besluit (ticker, aanleiding, oordeel, datum_oordeel,
                                 koers_toen, valuta)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT(ticker, datum_oordeel) DO NOTHING
            """,
            (ticker, aanleiding, oordeel, datum_oordeel, koers_toen, valuta),
        )


def besluit_afsluiten(ticker: str, keuze: str, reden: str | None = None,
                      snapshot: dict | None = None) -> bool:
    """Leg de keuze vast bij het besluit voor deze ticker. True als er iets bijgewerkt is.

    Een nog open besluit gaat voor; is er geen open besluit meer, dan wordt de
    jongste keuze overschreven. Zonder die terugval is een verkeerd ingedrukte
    knop niet meer te herstellen, en een systeem dat je op je eigen gedrag wijst
    moet je niet vastzetten op een vergissing.

    De momentopname wordt alleen gezet als hij nog ontbrak. Bij terugvullen van
    oude oordelen is er geen momentopname van toen; die dan later alsnog met de
    cijfers van vandaag vullen zou doen alsof we wisten wat we niet wisten.
    """
    with _cursor() as cur:
        cur.execute(
            """
            UPDATE besluit
               SET keuze = %s,
                   reden = COALESCE(%s, reden),
                   datum_keuze = %s,
                   these_snapshot = COALESCE(these_snapshot, %s)
             WHERE id = (SELECT id FROM besluit
                          WHERE ticker = %s
                       ORDER BY (keuze IS NULL) DESC, datum_oordeel DESC
                          LIMIT 1)
            """,
            (keuze, reden, datetime.utcnow().date().isoformat(),
             json.dumps(snapshot) if snapshot else None, ticker),
        )
        return cur.rowcount > 0


def besluiten_lijst(alleen_open: bool = False) -> list[dict]:
    """Alle besluiten, nieuwste oordeel eerst."""
    vraag = "SELECT * FROM besluit"
    if alleen_open:
        vraag += " WHERE keuze IS NULL"
    vraag += " ORDER BY datum_oordeel DESC, ticker"
    with _cursor() as cur:
        cur.execute(vraag)
        rows = cur.fetchall()
    uit = []
    for row in rows:
        rij = dict(row)
        rauw = rij.get("these_snapshot")
        try:
            rij["these_snapshot"] = json.loads(rauw) if rauw else None
        except (json.JSONDecodeError, TypeError):
            rij["these_snapshot"] = None
        uit.append(rij)
    return uit


def bezit_lijst() -> list[dict]:
    """Alles wat vastgelegd is als bezit, oudste vastlegging eerst.

    De momentopname komt ontleed terug; een kapotte JSON levert een lege dict op
    in plaats van een uitzondering. De vergelijkende verkoopregels melden dan
    netjes dat ze niets te vergelijken hebben — beter dan een pagina die omvalt.
    """
    with _cursor() as cur:
        cur.execute("SELECT * FROM bezit ORDER BY sinds, ticker")
        rows = cur.fetchall()
    uit = []
    for row in rows:
        rij = dict(row)
        try:
            rij["these_snapshot"] = json.loads(rij["these_snapshot"]) if rij.get("these_snapshot") else {}
        except (json.JSONDecodeError, TypeError):
            rij["these_snapshot"] = {}
        uit.append(rij)
    return uit


def bezit_vastleggen(ticker: str, snapshot: dict | None = None,
                     notitie: str | None = None) -> None:
    """Leg vast dat je dit aandeel bezit.

    Bij een bestaande vastlegging blijven `sinds` en de momentopname staan —
    díé zijn het ijkpunt en overschrijven zou de vergelijking wissen. Alleen de
    notitie wordt bijgewerkt, en alleen als er een nieuwe meegegeven wordt.
    """
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO bezit (ticker, sinds, notitie, these_snapshot)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT(ticker) DO UPDATE
               SET notitie = COALESCE(excluded.notitie, bezit.notitie)
            """,
            (ticker, datetime.utcnow().date().isoformat(), notitie,
             json.dumps(snapshot or {})),
        )


def bezit_verwijderen(ticker: str) -> bool:
    """Haal een aandeel uit het bezit. True als er iets weg is."""
    with _cursor() as cur:
        cur.execute("DELETE FROM bezit WHERE ticker=%s", (ticker,))
        return cur.rowcount > 0


def get_all_refresh_state() -> dict[str, str]:
    with _cursor() as cur:
        cur.execute("SELECT key, value FROM refresh_state")
        rows = cur.fetchall()
    return {r["key"]: r["value"] for r in rows}


# ---------------------------------------------------------------------------
# Upsert-helper
# ---------------------------------------------------------------------------

def _update_clause(table: str, fields, keys: tuple[str, ...], always: tuple[str, ...] = ()) -> str:
    """
    Bouw de SET-clausule van een upsert waarbij een NULL nooit een bestaande
    waarde wist.

    Yahoo levert regelmatig een half antwoord: prijs wel, jaarcijfers niet, of
    andersom. Zonder COALESCE overschrijft zo'n antwoord goede cijfers met NULL
    en is de historie weg tot de volgende geslaagde fetch — precies het
    "data raakt zoek"-effect. Met COALESCE blijft de oude waarde staan.

    `always` bevat de kolommen waar een NULL wél betekenisvol is (tijdstempels
    en statusvelden: die horen de nieuwe werkelijkheid te tonen).
    """
    parts = []
    for k in fields:
        if k in keys:
            continue
        if k in always:
            parts.append(f"{k}=excluded.{k}")
        else:
            parts.append(f"{k}=COALESCE(excluded.{k}, {table}.{k})")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Stocks
# ---------------------------------------------------------------------------

def upsert_stock(ticker: str, **fields) -> None:
    fields["ticker"] = ticker
    cols = ", ".join(fields.keys())
    placeholders = ", ".join(["%s"] * len(fields))
    updates = ", ".join(f"{k}=excluded.{k}" for k in fields if k != "ticker")
    sql = f"""
        INSERT INTO stocks ({cols}) VALUES ({placeholders})
        ON CONFLICT(ticker) DO UPDATE SET {updates}
    """
    with _cursor() as cur:
        cur.execute(sql, list(fields.values()))


def get_all_stocks() -> list[dict]:
    with _cursor() as cur:
        cur.execute("SELECT * FROM stocks WHERE active=1 ORDER BY ticker")
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_stock(ticker: str) -> dict | None:
    with _cursor() as cur:
        cur.execute("SELECT * FROM stocks WHERE ticker=%s", (ticker,))
        row = cur.fetchone()
    return dict(row) if row else None


def delete_stock(ticker: str) -> None:
    with _cursor() as cur:
        cur.execute("DELETE FROM stocks WHERE ticker=%s", (ticker,))


# ---------------------------------------------------------------------------
# Financials
# ---------------------------------------------------------------------------

def upsert_financials(ticker: str, period_type: str, fiscal_year: int | None, **fields) -> None:
    fields.update({"ticker": ticker, "period_type": period_type, "fiscal_year": fiscal_year})
    cols = ", ".join(fields.keys())
    placeholders = ", ".join(["%s"] * len(fields))
    updates = _update_clause(
        "financials", fields,
        keys=("ticker", "period_type", "fiscal_year"),
        always=("fetched_date",),
    )
    sql = f"""
        INSERT INTO financials ({cols}) VALUES ({placeholders})
        ON CONFLICT(ticker, period_type, fiscal_year) DO UPDATE SET {updates}
    """
    with _cursor() as cur:
        cur.execute(sql, list(fields.values()))


def get_financials(ticker: str, period_type: str = "annual") -> list[dict]:
    with _cursor() as cur:
        cur.execute(
            "SELECT * FROM financials WHERE ticker=%s AND period_type=%s ORDER BY fiscal_year DESC",
            (ticker, period_type),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

def upsert_market_data(ticker: str, **fields) -> None:
    fields["ticker"] = ticker
    cols = ", ".join(fields.keys())
    placeholders = ", ".join(["%s"] * len(fields))
    updates = _update_clause(
        "market_data", fields,
        keys=("ticker",),
        always=("last_updated",),
    )
    sql = f"""
        INSERT INTO market_data ({cols}) VALUES ({placeholders})
        ON CONFLICT(ticker) DO UPDATE SET {updates}
    """
    with _cursor() as cur:
        cur.execute(sql, list(fields.values()))


def get_market_data(ticker: str) -> dict | None:
    with _cursor() as cur:
        cur.execute("SELECT * FROM market_data WHERE ticker=%s", (ticker,))
        row = cur.fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Historical multiples
# ---------------------------------------------------------------------------

def upsert_historical_multiples(ticker: str, fiscal_year: int, **fields) -> None:
    fields.update({"ticker": ticker, "fiscal_year": fiscal_year})
    cols = ", ".join(fields.keys())
    placeholders = ", ".join(["%s"] * len(fields))
    updates = _update_clause("historical_multiples", fields, keys=("ticker", "fiscal_year"))
    sql = f"""
        INSERT INTO historical_multiples ({cols}) VALUES ({placeholders})
        ON CONFLICT(ticker, fiscal_year) DO UPDATE SET {updates}
    """
    with _cursor() as cur:
        cur.execute(sql, list(fields.values()))


def get_historical_multiples(ticker: str) -> list[dict]:
    with _cursor() as cur:
        cur.execute(
            "SELECT * FROM historical_multiples WHERE ticker=%s ORDER BY fiscal_year DESC",
            (ticker,),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Koershistorie
# ---------------------------------------------------------------------------

def insert_price_history(rows: list[tuple[str, str, float]]) -> int:
    """
    Sla slotkoersen op. `rows` is [(ticker, 'YYYY-MM-DD', close), ...].

    Bestaande dagen worden met rust gelaten (DO NOTHING, niet DO UPDATE): een
    eenmaal vastgelegde slotkoers is een feit, en een latere herziening door
    Yahoo — of een fout in een bulk-download — mag de vastgelegde reeks niet
    stilletjes herschrijven. Dat is precies het archiefdoel van deze tabel.
    """
    if not rows:
        return 0
    with _cursor() as cur:
        cur.executemany(
            """
            INSERT INTO price_history (ticker, date, close) VALUES (%s, %s, %s)
            ON CONFLICT (ticker, date) DO NOTHING
            """,
            rows,
        )
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0


def get_price_history(ticker: str, limit: int = 3650) -> list[dict]:
    with _cursor() as cur:
        cur.execute(
            "SELECT date, close FROM price_history WHERE ticker=%s ORDER BY date DESC LIMIT %s",
            (ticker, limit),
        )
        rows = cur.fetchall()
    return [dict(r) for r in reversed(rows)]


def price_history_stats() -> dict:
    """Omvang van het archief — voor /api/health en om de groei te bewaken."""
    with _cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) AS rijen,
                   COUNT(DISTINCT ticker) AS tickers,
                   MIN(date) AS vroegste,
                   MAX(date) AS laatste
            FROM price_history
        """)
        row = cur.fetchone()
    return dict(row) if row else {}


def thin_price_history(dagelijks_tot_dagen: int = 730, dry_run: bool = False) -> dict:
    """
    Dun oude koersen uit: recent dagelijks, ouder wekelijks.

    Waarom dit mag. De koersreeks dient één doel dat dagkoersen vereist — het
    actuele beeld en een leesbare grafiek — en één doel dat ze niet vereist: de
    cyclusvraag uit het moat-profiel, hoe diep ging dit aandeel onderuit en kwam
    het terug. Een weekkoers laat een daling van 40% in 2022 net zo goed zien.
    Bij ~2.800 tickers scheelt dat de helft van de tabel.

    Bewaard blijft, per ticker:
      - alles binnen `dagelijks_tot_dagen`;
      - daarbuiten de laatste handelsdag van elke ISO-week;
      - altijd de allereerste en allerlaatste waarneming, zodat het begin van de
        reeks en de meest recente koers nooit sneuvelen.

    Idempotent: een tweede run verwijdert niets meer.

    Returns {"te_verwijderen"|"verwijderd": n, "grens": "YYYY-MM-DD"}
    """
    grens = (datetime.utcnow().date() - timedelta(days=dagelijks_tot_dagen)).isoformat()

    # De te verwijderen rijen: ouder dan de grens, niet de weekafsluiter, en niet
    # het eerste of laatste punt van de reeks. Alles in één query zodat er geen
    # miljoen rijen door Python hoeven.
    selectie = """
        WITH gerangschikt AS (
            SELECT ticker, date,
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker, to_char(date::date, 'IYYY-IW')
                       ORDER BY date DESC
                   ) AS plek_in_week,
                   MIN(date) OVER (PARTITION BY ticker) AS eerste,
                   MAX(date) OVER (PARTITION BY ticker) AS laatste
            FROM price_history
            WHERE date < %s
        )
        SELECT ticker, date FROM gerangschikt
        WHERE plek_in_week > 1 AND date <> eerste AND date <> laatste
    """

    with _cursor() as cur:
        if dry_run:
            cur.execute(f"SELECT COUNT(*) AS n FROM ({selectie}) AS t", (grens,))
            return {"dry_run": True, "te_verwijderen": cur.fetchone()["n"], "grens": grens}

        cur.execute(f"""
            DELETE FROM price_history p
            USING ({selectie}) AS weg
            WHERE p.ticker = weg.ticker AND p.date = weg.date
        """, (grens,))
        return {"dry_run": False, "verwijderd": cur.rowcount, "grens": grens}


def tickers_needing_backfill(limit: int, min_start: str,
                             voorrang: list[str] | None = None) -> list[str]:
    """
    Actieve tickers zonder bruikbare koershistorie, interessantste eerst.

    Let op de HAVING-clausule: selecteren op "helemaal geen rijen" is niet
    genoeg. De dagelijkse koersronde schrijft vijf dagen weg, dus zodra die een
    keer langs is geweest heeft een ticker rijen en zou hij nooit meer worden
    aangevuld — met vijf dagen historie in plaats van tien jaar. Daarom kijken we
    naar de oudste opgeslagen dag: ligt die na `min_start`, dan ontbreekt de
    diepe historie nog.

    ## Waarom er groepen zijn, en waarom er bínnen een groep geloot wordt

    Er stond hier alleen `ORDER BY random()`. Dat was een reparatie van een echt
    probleem: alfabetisch sorteren loopt vast, want een ticker waarvoor Yahoo
    geen historie heeft blijft in de selectie zitten en wordt elke ronde
    opnieuw geprobeerd. Gemeten over 26 rondes zakte de opbrengst daardoor van
    56 naar 9 geslaagde tickers per ronde van honderd — tegen het eind ging
    vrijwel elke plek naar een bekende mislukking.

    Maar loten over het hele universum betekent ook: de twintig aandelen waar
    daadwerkelijk naar gekeken wordt zijn 0,7% van 2.759 tickers en krijgen dus
    0,7% van de aandacht. Gemeten op 6 augustus 2026 hadden elf van de top
    twintig van Kansen zeven koerspunten, terwijl willekeurige namen jaren
    hadden. Dat is niet alleen zonde van de ophaaltijd: het moat-profiel drukt
    "zwaarste koersval sinds jaar X" af als één van zijn drie redenen, en dat
    getal is met zeven dagen historie betekenisloos — precies bij de aandelen
    waar het ertoe doet.

    Vandaar groepen mét loting bínnen de groep: de interessante tickers gaan
    voor, en een kansloos geval kan nog steeds niet de kop van de rij bezetten.

    `voorrang` is voor tickers waar een rapport of tussencheck van ligt. Die
    kennis staat in markdownbestanden en niet in de database, dus die geeft de
    aanroeper mee.
    """
    voorrang = [t.upper() for t in (voorrang or [])]
    with _cursor() as cur:
        cur.execute("""
            SELECT s.ticker FROM stocks s
            LEFT JOIN price_history p  ON p.ticker = s.ticker
            LEFT JOIN calculated_scores c ON c.ticker = s.ticker
            LEFT JOIN data_quality dq  ON dq.ticker = s.ticker
            WHERE s.active = 1 AND s.presumed_delisted_at IS NULL
            GROUP BY s.ticker, c.quality_score, c.margin_of_safety,
                     c.combined_fv, dq.data_status
            HAVING MIN(p.date) IS NULL OR MIN(p.date) > %s
            ORDER BY
              CASE
                WHEN s.ticker = ANY(%s::text[]) THEN 0
                WHEN dq.data_status IN ('ok', 'warning')
                     AND c.combined_fv IS NOT NULL
                     AND c.margin_of_safety >= 20
                     AND c.quality_score >= 7          THEN 1
                WHEN c.margin_of_safety IS NOT NULL     THEN 2
                ELSE 3
              END,
              random()
            LIMIT %s
        """, (min_start, voorrang, limit))
        return [r["ticker"] for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Calculated scores
# ---------------------------------------------------------------------------

def upsert_scores(ticker: str, **fields) -> None:
    fields["ticker"] = ticker
    for k, v in fields.items():
        if isinstance(v, (dict, list)):
            fields[k] = json.dumps(v)
    cols = ", ".join(fields.keys())
    placeholders = ", ".join(["%s"] * len(fields))
    updates = ", ".join(f"{k}=excluded.{k}" for k in fields if k != "ticker")
    sql = f"""
        INSERT INTO calculated_scores ({cols}) VALUES ({placeholders})
        ON CONFLICT(ticker) DO UPDATE SET {updates}
    """
    with _cursor() as cur:
        cur.execute(sql, list(fields.values()))


# Kolommen in `calculated_scores` die als JSON-tekst zijn opgeslagen en dus
# ontleed moeten worden voordat ze de frontend in gaan.
#
# `fv_methods_dropped` ontbrak hier, met een venijnig gevolg: de beslisboom op de
# aandeelpagina kreeg de tekst "[]" in plaats van een lege lijst. Een tekst van
# twee tekens is niet leeg, dus de code kwam in de tak die `.join()` aanroept —
# en dat bestaat niet op een string. Die fout brak de héle beslisboom, voor élk
# aandeel, en werd opgeslokt door een lege `.catch()` in de pagina. Daarom stond
# er sinds de bouw ervan nooit één blok op het scherm.
#
# Voeg hier elke nieuwe JSON-kolom aan toe, anders herhaalt dit zich.
JSON_SCORE_KOLOMMEN = (
    "quality_breakdown", "piotroski_breakdown", "warnings",
    "hist_relative", "fv_methods_dropped",
)


# De cijferkolommen van `financials`, in de volgorde van het schema hierboven.
#
# Nodig voor rijen die niet uit de database komen maar ter plekke worden
# samengesteld: een boekjaar dat uitsluitend uit handmatige overrides bestaat.
# Zo'n rij had alleen `fiscal_year` en dus ontbraken alle andere sleutels. In
# Jinja is een ontbrekende sleutel `Undefined`, en `Undefined is not none` is
# waar — de opmaakmacro liep dan door naar `val | abs` en de hele aandeelpagina
# gaf een 500. Bouw zo'n rij daarom altijd met álle velden op None.
FINANCIAL_VELDEN = (
    "revenue", "ebit", "ebitda", "net_income", "eps_diluted",
    "operating_cf", "capex", "fcf", "total_assets", "total_equity",
    "total_debt", "current_assets", "current_liabilities", "net_ppe",
    "book_value_ps", "roe", "gross_profit", "interest_expense",
    "shares_outstanding", "net_cash", "inventory", "fetched_date",
)


def lege_jaarrij(fiscal_year: int) -> dict:
    """Een jaarrij met alle velden aanwezig en leeg — zie FINANCIAL_VELDEN."""
    rij: dict = {veld: None for veld in FINANCIAL_VELDEN}
    rij["fiscal_year"] = fiscal_year
    rij["period_type"] = "annual"
    return rij


def jaarrijen_met_overrides(ticker: str) -> tuple[list[dict], list[int]]:
    """De jaarcijfers zoals ze na handmatige correcties gelden.

    Handmatig ingevoerde cijfers staan in een eigen tabel en niet in
    `financials`. Wie alleen die laatste leest, mist ze — en dat liep uit
    elkaar: de aandeelpagina en de screener telden een handmatig FY2025 wél
    mee, het dashboard en de datakwaliteit niet. Op het dashboard bleef daardoor
    "cijfers t/m 2024" staan naast een boekjaar dat er gewoon was.

    Geeft terug: (rijen nieuwste-eerst, jaren die alleen handmatig bestaan).
    """
    rijen = get_financials(ticker, "annual")
    overrides = get_overrides(ticker)

    bestaande = {r.get("fiscal_year") for r in rijen}
    alleen_handmatig = sorted(
        {jr for (_, jr) in overrides if jr is not None and jr not in bestaande},
        reverse=True,
    )
    for jr in alleen_handmatig:
        rijen.append(lege_jaarrij(jr))
    if alleen_handmatig:
        rijen.sort(key=lambda r: r.get("fiscal_year") or 0, reverse=True)

    # Een override zonder jaartal geldt voor élk jaar; dat is de bestaande afspraak.
    for rij in rijen:
        jaar = rij.get("fiscal_year")
        for (veld, ov_jaar), entry in overrides.items():
            if ov_jaar == jaar or ov_jaar is None:
                rij[veld] = entry["value"]

    return rijen, alleen_handmatig


def get_scores(ticker: str) -> dict | None:
    with _cursor() as cur:
        cur.execute("SELECT * FROM calculated_scores WHERE ticker=%s", (ticker,))
        row = cur.fetchone()
    if not row:
        return None
    result = dict(row)
    for key in JSON_SCORE_KOLOMMEN:
        if result.get(key):
            try:
                result[key] = json.loads(result[key])
            except (json.JSONDecodeError, TypeError):
                pass
    return result


def get_all_scores() -> list[dict]:
    with _cursor() as cur:
        cur.execute("SELECT * FROM calculated_scores")
        rows = cur.fetchall()
    results = []
    for row in rows:
        r = dict(row)
        for key in JSON_SCORE_KOLOMMEN:
            if r.get(key):
                try:
                    r[key] = json.loads(r[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        results.append(r)
    return results


def get_latest_fiscal_years() -> dict[str, int]:
    """Return {ticker: max_fiscal_year} for all tickers with annual financial data."""
    with _cursor() as cur:
        cur.execute(
            "SELECT ticker, MAX(fiscal_year) as fy FROM financials WHERE period_type='annual' GROUP BY ticker"
        )
        rows = cur.fetchall()
    return {r["ticker"]: r["fy"] for r in rows}


def get_latest_fetched_dates() -> dict[str, str]:
    """Return {ticker: latest_fetched_date} for tickers with annual financial data."""
    with _cursor() as cur:
        cur.execute(
            "SELECT ticker, MAX(fetched_date) as fd FROM financials WHERE period_type='annual' GROUP BY ticker"
        )
        rows = cur.fetchall()
    return {r["ticker"]: r["fd"] for r in rows if r["fd"]}


def get_all_market_data() -> dict[str, dict]:
    """Alle marktdata in één query — voorkomt N+1 bij de bulk-prijsrefresh."""
    with _cursor() as cur:
        cur.execute("SELECT * FROM market_data")
        rows = cur.fetchall()
    return {r["ticker"]: dict(r) for r in rows}


def get_latest_shares_outstanding() -> dict[str, float]:
    """{ticker: shares_outstanding} uit het nieuwste boekjaar met een waarde."""
    with _cursor() as cur:
        cur.execute("""
            SELECT DISTINCT ON (ticker) ticker, shares_outstanding
            FROM financials
            WHERE period_type='annual' AND shares_outstanding IS NOT NULL
            ORDER BY ticker, fiscal_year DESC
        """)
        rows = cur.fetchall()
    return {r["ticker"]: r["shares_outstanding"] for r in rows if r["shares_outstanding"]}


def get_latest_eps_ttm() -> dict[str, float]:
    """{ticker: eps_diluted} uit het nieuwste boekjaar — basis voor de K/W-schatting."""
    with _cursor() as cur:
        cur.execute("""
            SELECT DISTINCT ON (ticker) ticker, eps_diluted
            FROM financials
            WHERE period_type='annual' AND eps_diluted IS NOT NULL
            ORDER BY ticker, fiscal_year DESC
        """)
        rows = cur.fetchall()
    return {r["ticker"]: r["eps_diluted"] for r in rows if r["eps_diluted"]}


def get_refresh_queue(limit: int) -> list[str]:
    """
    De `limit` actieve tickers die het langst niet zijn geprobeerd.

    Sorteert op `data_quality.last_checked` (geschreven bij élke poging), niet op
    de laatste geslaagde fetch. Tickers die niets opleveren zakken zo netjes naar
    achteren in plaats van de wachtrij permanent te blokkeren. Nooit-geprobeerde
    tickers komen vooraan, zodat nieuw toegevoegde aandelen meteen aan de beurt
    zijn. Vermoedelijk delisted tickers doen niet mee.
    """
    with _cursor() as cur:
        cur.execute("""
            SELECT s.ticker
            FROM stocks s
            LEFT JOIN data_quality dq ON dq.ticker = s.ticker
            WHERE s.active = 1 AND s.presumed_delisted_at IS NULL
            ORDER BY dq.last_checked ASC NULLS FIRST, s.ticker
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
    return [r["ticker"] for r in rows]


def get_reprobe_candidates(limit: int) -> list[str]:
    """Gesuspendeerde tickers die opnieuw geprobeerd mogen worden, oudste eerst."""
    with _cursor() as cur:
        cur.execute("""
            SELECT ticker FROM stocks
            WHERE active = 0
              AND auto_suspended_at IS NOT NULL
              AND presumed_delisted_at IS NULL
            ORDER BY auto_suspended_at ASC
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
    return [r["ticker"] for r in rows]


def count_annual_rows(ticker: str) -> int:
    """Aantal jaarcijfer-rijen — de maatstaf voor 'hebben we überhaupt iets'."""
    with _cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n FROM financials WHERE ticker=%s AND period_type='annual'",
            (ticker,),
        )
        row = cur.fetchone()
    return int(row["n"]) if row else 0


def bump_failure_counter(ticker: str) -> None:
    """
    Verhoog de failure-teller en leg vast wanneer de reeks begon.

    `first_failure_at` maakt het verschil tussen "tien keer mislukt op één avond"
    en "al een maand onbereikbaar" — alleen het tweede rechtvaardigt suspenderen.
    """
    now = datetime.utcnow().isoformat()
    with _cursor() as cur:
        cur.execute("""
            INSERT INTO data_quality (ticker, consecutive_failures, first_failure_at, last_checked)
            VALUES (%s, 1, %s, %s)
            ON CONFLICT(ticker) DO UPDATE SET
                consecutive_failures = COALESCE(data_quality.consecutive_failures, 0) + 1,
                first_failure_at = COALESCE(data_quality.first_failure_at, excluded.first_failure_at),
                last_checked = excluded.last_checked
        """, (ticker, now, now))


def reset_failure_counter(ticker: str) -> None:
    with _cursor() as cur:
        cur.execute(
            "UPDATE data_quality SET consecutive_failures = 0, first_failure_at = NULL WHERE ticker = %s",
            (ticker,),
        )


def get_last_attempt_dates() -> dict[str, str]:
    """
    Return {ticker: last_checked} — het moment van de laatste fetch-POGING.

    Anders dan get_latest_fetched_dates() (dat alleen tickers kent die ook
    daadwerkelijk jaarcijfers opleverden) wordt last_checked bij elke poging
    geschreven, ook als Yahoo niets teruggaf. Dat maakt het de juiste sleutel
    om een refresh-rotatie op te ordenen: een ticker die structureel niets
    oplevert zakt na zijn beurt naar achteren in plaats van elke nacht opnieuw
    vooraan te staan.
    """
    with _cursor() as cur:
        cur.execute("SELECT ticker, last_checked FROM data_quality")
        rows = cur.fetchall()
    return {r["ticker"]: r["last_checked"] for r in rows if r["last_checked"]}


def get_dashboard_data() -> list[dict]:
    """
    Fetch all active stocks with their market data, calculated scores, 
    latest fiscal year and latest fetched date in one optimized query.
    Used to prevent the N+1 query problem on the dashboard.
    """
    sql = """
        SELECT
            s.ticker, s.name, s.sector, s.market, s.currency, s.added_date,
            m.price, m.market_cap, m.enterprise_value, m.last_updated,
            c.quality_score, c.quality_breakdown, c.piotroski_score, c.piotroski_breakdown,
            c.normalized_eps, c.normalized_ebitda, c.normalized_fcf, c.normalized_owner_earn,
            c.multiples_fv, c.graham_fv, c.perpetuity_fv, c.combined_fv,
            c.conservative_fv, c.base_fv, c.optimistic_fv,
            c.fv_confidence, c.fv_spread_pct, c.fv_methods_used,
            c.fv_methods_dropped, c.revenue_cagr, c.implied_growth,
            c.signal, c.margin_of_safety, c.warnings, c.last_calculated, c.accruals_ratio, c.hist_relative,
            fy.latest_fy, fy.laatste_yahoo,
            dq.completeness_pct, dq.years_available, dq.freshness_days,
            dq.fetch_success, dq.consecutive_failures, dq.data_status,
            dq.issues as data_issues, dq.last_checked as dq_last_checked
        FROM stocks s
        LEFT JOIN market_data m ON s.ticker = m.ticker
        LEFT JOIN calculated_scores c ON s.ticker = c.ticker
        LEFT JOIN (
            -- Het nieuwste boekjaar is het nieuwste dat we kénnen, of dat nu van
            -- Yahoo komt of met de hand is ingevoerd. `laatste_yahoo` blijft er
            -- apart bij staan zodat het dashboard kan zeggen dát een jaar
            -- handmatig is in plaats van het stilzwijgend mee te tellen.
            SELECT
                ticker,
                MAX(fiscal_year)                                  AS latest_fy,
                MAX(fiscal_year) FILTER (WHERE bron = 'yahoo')    AS laatste_yahoo
            FROM (
                SELECT ticker, fiscal_year, 'yahoo' AS bron
                FROM financials
                WHERE period_type='annual' AND fiscal_year IS NOT NULL
                UNION ALL
                SELECT ticker, fiscal_year, 'handmatig' AS bron
                FROM overrides
                WHERE fiscal_year IS NOT NULL
            ) alle_jaren
            GROUP BY ticker
        ) fy ON s.ticker = fy.ticker
        LEFT JOIN data_quality dq ON s.ticker = dq.ticker
        WHERE s.active = 1
    """
    with _cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    results = []
    for row in rows:
        r = dict(row)
        for key in ("quality_breakdown", "piotroski_breakdown", "warnings", "hist_relative", "data_issues", "fv_methods_dropped"):
            if r.get(key):
                try:
                    r[key] = json.loads(r[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------

def set_override(ticker: str, field_name: str, fiscal_year: int | None, value: float, note: str = "") -> None:
    from datetime import datetime
    with _cursor() as cur:
        cur.execute("""
            INSERT INTO overrides (ticker, field_name, fiscal_year, value, note, created_date)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT(ticker, field_name, fiscal_year) DO UPDATE
            SET value=excluded.value, note=excluded.note, created_date=excluded.created_date
        """, (ticker, field_name, fiscal_year, value, note, datetime.utcnow().isoformat()))


def get_overrides(ticker: str) -> dict:
    """Returns {(field_name, fiscal_year): {"value": v, "note": n}} for easy lookup."""
    with _cursor() as cur:
        cur.execute(
            "SELECT field_name, fiscal_year, value, note FROM overrides WHERE ticker=%s", (ticker,)
        )
        rows = cur.fetchall()
    return {(r["field_name"], r["fiscal_year"]): {"value": r["value"], "note": r["note"]} for r in rows}


def delete_override(ticker: str, field_name: str, fiscal_year: int | None) -> None:
    with _cursor() as cur:
        cur.execute(
            "DELETE FROM overrides WHERE ticker=%s AND field_name=%s AND fiscal_year=%s",
            (ticker, field_name, fiscal_year),
        )


# ---------------------------------------------------------------------------
# Exchange rates
# ---------------------------------------------------------------------------

def upsert_exchange_rate(currency: str, rate_to_eur: float) -> None:
    from datetime import datetime
    with _cursor() as cur:
        cur.execute("""
            INSERT INTO exchange_rates (currency, rate_to_eur, last_updated)
            VALUES (%s, %s, %s)
            ON CONFLICT(currency) DO UPDATE
            SET rate_to_eur=excluded.rate_to_eur, last_updated=excluded.last_updated
        """, (currency, rate_to_eur, datetime.utcnow().isoformat()))


def get_exchange_rates() -> dict[str, float]:
    with _cursor() as cur:
        cur.execute("SELECT currency, rate_to_eur FROM exchange_rates")
        rows = cur.fetchall()
    return {r["currency"]: r["rate_to_eur"] for r in rows}


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------

def upsert_data_quality(ticker: str, **fields) -> None:
    """Persist a data-quality record. `issues` list/dict is JSON-encoded."""
    fields["ticker"] = ticker
    if isinstance(fields.get("issues"), (list, dict)):
        fields["issues"] = json.dumps(fields["issues"])
    cols = ", ".join(fields.keys())
    placeholders = ", ".join(["%s"] * len(fields))
    updates = ", ".join(f"{k}=excluded.{k}" for k in fields if k != "ticker")
    sql = f"""
        INSERT INTO data_quality ({cols}) VALUES ({placeholders})
        ON CONFLICT(ticker) DO UPDATE SET {updates}
    """
    with _cursor() as cur:
        cur.execute(sql, list(fields.values()))


def get_data_quality(ticker: str) -> dict | None:
    with _cursor() as cur:
        cur.execute("SELECT * FROM data_quality WHERE ticker=%s", (ticker,))
        row = cur.fetchone()
    if not row:
        return None
    r = dict(row)
    if r.get("issues"):
        try:
            r["issues"] = json.loads(r["issues"])
        except (json.JSONDecodeError, TypeError):
            pass
    return r


def get_all_data_quality() -> dict[str, dict]:
    """Return {ticker: data_quality_dict} for all tickers with a record."""
    with _cursor() as cur:
        cur.execute("SELECT * FROM data_quality")
        rows = cur.fetchall()
    out: dict[str, dict] = {}
    for row in rows:
        r = dict(row)
        if r.get("issues"):
            try:
                r["issues"] = json.loads(r["issues"])
            except (json.JSONDecodeError, TypeError):
                pass
        out[r["ticker"]] = r
    return out


# ---------------------------------------------------------------------------
# Activity log
# ---------------------------------------------------------------------------

def log_activity(action: str, ticker: str | None, status: str, details: dict | str | None = None) -> None:
    """
    Log an action to the activity_log table.
    action:  'add' | 'remove' | 'fetch' | 'recalculate' | 'manual_price' | 'override'
    status:  'ok' | 'warning' | 'error'
    details: dict or string with extra context
    """
    from datetime import datetime
    detail_str = json.dumps(details) if isinstance(details, (dict, list)) else (details or "")
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO activity_log (timestamp, action, ticker, status, details) VALUES (%s, %s, %s, %s, %s)",
            (datetime.utcnow().isoformat(), action, ticker, status, detail_str),
        )


def get_activity_log(ticker: str | None = None, limit: int = 200) -> list[dict]:
    """Return recent activity log entries, newest first. Optionally filter by ticker."""
    with _cursor() as cur:
        if ticker:
            cur.execute(
                "SELECT * FROM activity_log WHERE ticker=%s ORDER BY id DESC LIMIT %s",
                (ticker.upper(), limit),
            )
        else:
            cur.execute(
                "SELECT * FROM activity_log ORDER BY id DESC LIMIT %s", (limit,)
            )
        rows = cur.fetchall()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("details"):
            try:
                d["details"] = json.loads(d["details"])
            except (json.JSONDecodeError, TypeError):
                pass
        result.append(d)
    return result
