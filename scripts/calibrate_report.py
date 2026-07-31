"""
Kalibratierapport: laat zien of de fair-value-instellingen realistisch staan.

Achtergrond: het dashboard gaf 615 van de 913 aandelen een SELL. Een screener
die twee derde van een brede Europese universe te duur noemt is niet aan het
selecteren maar aan het afwijzen - dan is de meetlat verkeerd afgesteld, niet
de markt. Dit script maakt zichtbaar wáár de scheefheid zit, zodat je gericht
per sector kunt bijstellen in plaats van aan één algemene knop te draaien.

Gebruik:
    python scripts/calibrate_report.py                     # tegen productie
    python scripts/calibrate_report.py --url http://...    # tegen een andere omgeving
    python scripts/calibrate_report.py --json rapport.json # ook wegschrijven

Op een netwerk dat TLS onderschept (zoals dat van de Hanze) kan Python geen
https doen. Haal het dashboard dan eerst met curl op en voer het bestand in:

    curl -ks https://stockscreen-janco.fly.dev/api/dashboard -o dash.json
    python scripts/calibrate_report.py --file dash.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.request
from collections import defaultdict

STANDAARD_URL = "https://stockscreen-janco.fly.dev"

# Brede, herkenbare namen uit meerdere landen en sectoren. Bedoeld als
# realiteitstoets: als een screener ASML, Heineken én Equinor allemaal fors
# overgewaardeerd noemt, ligt het aan de instellingen.
REFERENTIE = [
    "ASML.AS", "AD.AS", "WKL.AS", "HEIA.AS", "INGA.AS", "PHIA.AS",
    "KBC.BR", "UMI.BR", "ABI.BR",
    "EVO.ST", "INVE-B.ST", "ATCO-A.ST", "VOLV-B.ST",
    "EQNR.OL", "ORK.OL", "DNB.OL",
    "ALE.WA", "PKO.WA", "XTB.WA",
    "SAP.DE", "ALV.DE", "SIE.DE",
    "OR.PA", "MC.PA", "AIR.PA",
]


def haal_dashboard(url: str, bestand: str | None = None) -> list[dict]:
    if bestand:
        with open(bestand, encoding="utf-8") as f:
            return json.load(f)
    with urllib.request.urlopen(f"{url.rstrip('/')}/api/dashboard", timeout=120) as r:
        return json.load(r)


def mediaan(waarden: list[float]) -> float | None:
    return statistics.median(waarden) if waarden else None


def kwartielen(waarden: list[float]) -> tuple[float, float] | None:
    if len(waarden) < 4:
        return None
    q = statistics.quantiles(waarden, n=4)
    return q[0], q[2]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=STANDAARD_URL)
    p.add_argument("--json", dest="json_out")
    p.add_argument("--file", dest="bestand", help="Lees een eerder opgehaald dashboard-JSON")
    args = p.parse_args()

    rijen = haal_dashboard(args.url, args.bestand)
    beoordeeld = [
        r for r in rijen
        if r.get("signal") not in (None, "INSUFFICIENT DATA", "N/A")
        and r.get("margin_of_safety") is not None
    ]

    print(f"Universe: {len(rijen)} tickers - {len(beoordeeld)} met een oordeel\n")

    # --- Signaalverdeling ---------------------------------------------------
    verdeling: dict[str, int] = defaultdict(int)
    for r in rijen:
        verdeling[r.get("signal") or "GEEN"] += 1

    print("SIGNAALVERDELING")
    totaal_beoordeeld = len(beoordeeld) or 1
    koop = 0
    for sig in ("STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL", "INSUFFICIENT DATA", "GEEN"):
        n = verdeling.get(sig, 0)
        if not n:
            continue
        aandeel = 100.0 * n / totaal_beoordeeld if sig not in ("INSUFFICIENT DATA", "GEEN") else 0
        if sig in ("BUY", "STRONG BUY"):
            koop += n
        extra = f"  ({aandeel:4.1f}% van beoordeeld)" if aandeel else ""
        print(f"  {sig:20s} {n:5d}{extra}")

    koop_pct = 100.0 * koop / totaal_beoordeeld
    verkoop_pct = 100.0 * (verdeling.get("SELL", 0) + verdeling.get("STRONG SELL", 0)) / totaal_beoordeeld
    print(f"\n  koopsignalen samen : {koop_pct:5.1f}%   (streefwaarde 5-15%)")
    print(f"  verkoopsignalen    : {verkoop_pct:5.1f}%   (streefwaarde onder 40%)")

    # --- Per sector ---------------------------------------------------------
    per_sector: dict[str, list[float]] = defaultdict(list)
    for r in beoordeeld:
        per_sector[r.get("sector") or "Onbekend"].append(r["margin_of_safety"])

    print("\nVEILIGHEIDSMARGE PER SECTOR  (negatief = screener vindt het te duur)")
    print(f"  {'sector':26s} {'n':>4s} {'mediaan':>9s} {'kw1':>8s} {'kw3':>8s}")
    sector_medianen = {}
    for sector, waarden in sorted(per_sector.items(), key=lambda kv: mediaan(kv[1]) or 0):
        if len(waarden) < 3:
            continue
        med = mediaan(waarden)
        sector_medianen[sector] = med
        kw = kwartielen(waarden)
        kw1 = f"{kw[0]:7.1f}%" if kw else "      -"
        kw3 = f"{kw[1]:7.1f}%" if kw else "      -"
        vlag = "  <-- fors scheef" if med is not None and med < -30 else ""
        print(f"  {sector:26s} {len(waarden):4d} {med:8.1f}% {kw1} {kw3}{vlag}")

    # --- Per markt ----------------------------------------------------------
    per_markt: dict[str, list[float]] = defaultdict(list)
    for r in beoordeeld:
        t = r.get("ticker", "")
        markt = t.rsplit(".", 1)[1] if "." in t else "US"
        per_markt[markt].append(r["margin_of_safety"])

    print("\nVEILIGHEIDSMARGE PER MARKT")
    for markt, waarden in sorted(per_markt.items(), key=lambda kv: mediaan(kv[1]) or 0):
        if len(waarden) < 3:
            continue
        print(f"  {markt:6s} {len(waarden):4d} {mediaan(waarden):8.1f}%")

    # --- Referentieset ------------------------------------------------------
    op_ticker = {r["ticker"]: r for r in rijen}
    print("\nREFERENTIESET  (bekende namen als realiteitstoets)")
    print(f"  {'ticker':12s} {'koers':>9s} {'fair value':>11s} {'ratio':>7s} {'marge':>8s}  signaal")
    extreem = []
    for t in REFERENTIE:
        r = op_ticker.get(t)
        if not r:
            continue
        prijs, fv, mos = r.get("price"), r.get("combined_fv"), r.get("margin_of_safety")
        if not prijs or not fv:
            print(f"  {t:12s} {'-':>9s} {'-':>11s} {'-':>7s} {'-':>8s}  {r.get('signal')}")
            continue
        ratio = prijs / fv
        if mos is not None and abs(mos) > 60:
            extreem.append((t, mos))
        print(f"  {t:12s} {prijs:9.2f} {fv:11.2f} {ratio:7.2f} {mos:7.1f}%  {r.get('signal')}")

    if extreem:
        print(f"\n  {len(extreem)} referentie-aandelen wijken meer dan 60% af:")
        for t, mos in sorted(extreem, key=lambda x: x[1]):
            print(f"    {t:12s} {mos:7.1f}%")

    # --- Oordeel ------------------------------------------------------------
    print("\nOORDEEL")
    goed = True
    if not (5 <= koop_pct <= 15):
        print(f"  X koopsignalen {koop_pct:.1f}% ligt buiten 5-15%")
        goed = False
    else:
        print(f"  v koopsignalen {koop_pct:.1f}%")
    if verkoop_pct >= 40:
        print(f"  X verkoopsignalen {verkoop_pct:.1f}% ligt boven 40%")
        goed = False
    else:
        print(f"  v verkoopsignalen {verkoop_pct:.1f}%")
    if extreem:
        print(f"  X {len(extreem)} referentie-aandelen met een afwijking boven 60%")
        goed = False
    else:
        print("  v referentieset binnen redelijke grenzen")
    print("\n  " + ("Kalibratie is op orde." if goed else "Kalibratie moet nog bijgesteld worden."))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({
                "koop_pct": koop_pct,
                "verkoop_pct": verkoop_pct,
                "sector_medianen": sector_medianen,
                "verdeling": dict(verdeling),
            }, f, indent=2)
        print(f"\n  rapport weggeschreven naar {args.json_out}")

    return 0 if goed else 1


if __name__ == "__main__":
    sys.exit(main())
