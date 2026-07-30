"""
Bulk-import tickers uit een Excel-bestand met meerdere tabbladen (per land/index).

Elk tabblad heeft:
  rij 1: indexnaam (skip)
  rij 2: headers (Ticker, Naam, Bloomberg ticker, Yahoo Finance ticker, Land)
  rij 3+: data

Het script:
  - leest alle tabbladen
  - filtert geldige Yahoo-tickers (moeten een suffix hebben, bv. ADYEN.AS)
  - leidt de valuta af uit de Yahoo-suffix (.AS -> EUR, .ST -> SEK, etc.)
  - upsert naar de stocks-tabel (ticker, name, market, currency, added_date)
  - haalt GEEN financiele data op (dat is Stap B: financials-fetch)

Gebruik:
  DATABASE_URL="postgresql://..." python import_tickers.py <excel_path>
  python import_tickers.py <excel_path> --dry-run     # laat zien zonder te schrijven
  python import_tickers.py <excel_path> --tab Nederland  # alleen een tab
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

import openpyxl

# Yahoo-suffix -> valuta. Gebaseerd op beurs-conventies.
SUFFIX_TO_CURRENCY = {
    "AS": "EUR",  # Amsterdam
    "BR": "EUR",  # Brussel
    "PA": "EUR",  # Parijs
    "DE": "EUR",  # Xetra (Duitsland)
    "F":  "EUR",  # Frankfurt
    "MU": "EUR",
    "HM": "EUR",
    "SG": "EUR",
    "BE": "EUR",
    "MI": "EUR",  # Milaan
    "MC": "EUR",  # Madrid
    "LS": "EUR",  # Lissabon
    "VI": "EUR",  # Wenen
    "HE": "EUR",  # Helsinki
    "IR": "EUR",  # Dublin
    "ST": "SEK",  # Stockholm
    "OL": "NOK",  # Oslo
    "CO": "DKK",  # Kopenhagen
    "WA": "PLN",  # Warschau
    "L":  "GBP",  # Londen
    "SW": "CHF",  # Zwitserland
    "PR": "CZK",  # Praag
    "BD": "HUF",  # Boedapest
    "IS": "TRY",  # Istanboel
    "AT": "EUR",  # Athene
    "TA": "ILS",  # Tel Aviv
}

# Basisregex: tickerstring met precies een punt en alleen toegestane tekens.
TICKER_RE = re.compile(r"^[A-Z0-9\-]{1,10}\.[A-Z]{1,3}$")


def currency_from_ticker(yh_ticker: str) -> str | None:
    """Leidt valuta af uit Yahoo-suffix. Retourneert None bij onbekende suffix."""
    if "." not in yh_ticker:
        return None
    suffix = yh_ticker.rsplit(".", 1)[1].upper()
    return SUFFIX_TO_CURRENCY.get(suffix)


def clean_country(raw: str) -> str:
    """Haalt vlag-emoji en spaties weg. '🇳🇱 Nederland' -> 'Nederland'."""
    return re.sub(r"[^\w\s]", "", raw, flags=re.UNICODE).strip()


def parse_excel(path: Path, only_tab: str | None = None) -> list[dict]:
    """Leest alle tabbladen en retourneert een lijst met ticker-dicts."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    records: list[dict] = []

    for sheet_name in wb.sheetnames:
        country = clean_country(sheet_name)
        if only_tab and only_tab.lower() not in country.lower():
            continue

        ws = wb[sheet_name]
        for row in ws.iter_rows(values_only=True):
            if len(row) < 5:
                continue
            ticker_raw, name, _bloomberg, yh, land = row[:5]
            if not yh or not isinstance(yh, str):
                continue
            yh_clean = yh.strip()
            if not TICKER_RE.match(yh_clean):
                continue

            currency = currency_from_ticker(yh_clean)
            if not currency:
                print(f"  [WAARSCHUWING] onbekende suffix voor {yh_clean} (tab: {country})", file=sys.stderr)
                continue

            market = clean_country(land) if isinstance(land, str) else country
            records.append({
                "ticker": yh_clean,
                "name": (name or "").strip() if isinstance(name, str) else None,
                "market": market,
                "currency": currency,
                "financial_currency": currency,  # aanname, wordt overschreven door Yahoo-fetch
            })

    return records


def dedupe(records: list[dict]) -> tuple[list[dict], list[str]]:
    """Verwijdert dubbele tickers, retourneert (unieke records, lijst dubbele tickers)."""
    seen: dict[str, dict] = {}
    dups: list[str] = []
    for r in records:
        if r["ticker"] in seen:
            dups.append(r["ticker"])
            continue
        seen[r["ticker"]] = r
    return list(seen.values()), dups


def main() -> int:
    ap = argparse.ArgumentParser(description="Bulk-import tickers uit Excel.")
    ap.add_argument("excel_path", help="Pad naar Excel-bestand")
    ap.add_argument("--dry-run", action="store_true", help="Alleen tonen, niets schrijven")
    ap.add_argument("--tab", default=None, help="Filter op land/tab (substring match)")
    args = ap.parse_args()

    path = Path(args.excel_path)
    if not path.exists():
        print(f"FOUT: bestand niet gevonden: {path}", file=sys.stderr)
        return 1

    print(f"Lezen: {path}")
    records = parse_excel(path, only_tab=args.tab)
    records, dups = dedupe(records)

    print(f"\n{len(records)} unieke tickers gevonden.")
    if dups:
        print(f"  {len(dups)} dubbele tickers overgeslagen: {', '.join(dups)}")

    # Overzicht per markt
    per_market: dict[str, int] = {}
    per_currency: dict[str, int] = {}
    for r in records:
        per_market[r["market"]] = per_market.get(r["market"], 0) + 1
        per_currency[r["currency"]] = per_currency.get(r["currency"], 0) + 1

    print("\nVerdeling per markt:")
    for m, n in sorted(per_market.items(), key=lambda x: -x[1]):
        print(f"  {m:<20} {n:>4}")
    print("\nVerdeling per valuta:")
    for c, n in sorted(per_currency.items(), key=lambda x: -x[1]):
        print(f"  {c:<5} {n:>4}")

    if args.dry_run:
        print("\n[DRY RUN] Geen database-wijzigingen.")
        print("Voorbeeld rijen:")
        for r in records[:5]:
            print(f"  {r}")
        return 0

    # Echt importeren
    from engine.db import init_db, upsert_stock
    init_db()

    today_iso = date.today().isoformat()
    inserted_or_updated = 0
    errors: list[tuple[str, str]] = []

    for r in records:
        try:
            upsert_stock(
                r["ticker"],
                name=r["name"],
                market=r["market"],
                currency=r["currency"],
                financial_currency=r["financial_currency"],
                active=1,
                added_date=today_iso,
            )
            inserted_or_updated += 1
        except Exception as e:
            errors.append((r["ticker"], str(e)))

    print(f"\n{inserted_or_updated} tickers ge-upsert.")
    if errors:
        print(f"{len(errors)} fouten:")
        for tk, msg in errors[:20]:
            print(f"  {tk}: {msg}")
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
