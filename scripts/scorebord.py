"""
scorebord.py — opdrachtregel-schil rond `engine/scorebord.py`.

De rekenlogica staat in de engine omdat `.dockerignore` de map `scripts/`
buiten de build houdt: de webpagina `/scorebord` kan hier dus niet uit
importeren. Dit bestand doet nog twee dingen die alleen op de opdrachtregel
nodig zijn: het dashboard over het net ophalen, en de uitkomst als tekst
opmaken.

Gebruik:
  python scripts/scorebord.py
  python scripts/scorebord.py --koersen
"""

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from engine import scorebord as kern  # noqa: E402

API = "https://stockscreen-janco.fly.dev/api/dashboard"

TUSSEN_VOLGORDE = kern.TUSSEN_VOLGORDE


def _haal_dashboard() -> list[dict]:
    """Het dashboard ophalen, met dezelfde gerichte TLS-terugval als
    `check_noteringen.py`.

    Op dit netwerk wordt TLS onderschept, waardoor de certificaatketen niet
    klopt. Alleen terugvallen bij precies die fout, en het melden — anders
    glipt een echt certificaatprobleem er ongemerkt doorheen.
    """
    try:
        with urllib.request.urlopen(API, timeout=180) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        if not isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
            raise
        print("  [!] certificaat niet te verifieren (TLS-onderschepping op dit "
              "netwerk) — opnieuw zonder verificatie")
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(API, timeout=180, context=context) as r:
            return json.loads(r.read().decode("utf-8"))


def _live_koersen(regels: list[dict]) -> None:
    try:
        rijen = _haal_dashboard()
    except Exception as fout:            # netwerk weg, of de app slaapt
        print(f"  [!] koersen ophalen mislukt ({fout}); rendement blijft leeg\n")
        return
    kern.met_koersen(regels, rijen)


def _tabel(regels: list[dict], koersen: bool) -> None:
    kop = f"{'Ticker':<9} {'Tussencheck':<10} {'Datum':<11} {'Voorsp.':<8} {'Analyse':<8} {'Score':>6}"
    if koersen:
        kop += f" {'Sinds':>7}"
    print(kop)
    print("-" * len(kop))
    for r in regels:
        regel = (f"{r['ticker']:<9} {(r['tussencheck'] or '—'):<10} "
                 f"{(r['tc_datum'] or r['an_datum'] or '—'):<11} "
                 f"{(r['voorspelling'] or '—'):<8} {(r['analyse'] or '—'):<8} "
                 f"{(str(r['score']) + '/45' if r['score'] is not None else '—'):>6}")
        if koersen:
            rend = r.get("rendement")
            # '-0%' leest als een fout; onder een half procent is het gewoon nul.
            tekst = "—" if rend is None else ("0%" if abs(rend) < 0.5 else f"{rend:+.0f}%")
            regel += f" {tekst:>7}"
        print(regel)


def _kalibratie(regels: list[dict]) -> None:
    k = kern.kalibratie(regels)
    print(f"\n## Kalibratie — {k['paren']} van de {k['totaal']} hebben voorspelling én uitslag")
    if not k["paren"]:
        print("  Nog niets te meten.")
        return

    print(f"  Raak: {k['raak']} van {k['paren']}")
    for ticker, voorspeld, werd, richting in k["details"]:
        print(f"    {ticker:<9} voorspeld {voorspeld:<5} -> werd {werd:<5}  {richting}")

    afwijking = k["afwijking"]
    if afwijking > 0.3:
        print(f"\n  Gemiddelde afwijking +{afwijking:.1f}: stelselmatig te optimistisch.")
        print("  Betekenis: de VERDIEPEN-drempel staat te ruim en je verspilt uren")
        print("  aan kandidaten die de analyse niet halen.")
    elif afwijking < -0.3:
        print(f"\n  Gemiddelde afwijking {afwijking:.1f}: stelselmatig te somber.")
        print("  Betekenis: je slaat waarschijnlijk kandidaten over die het wél waard waren.")
    else:
        print(f"\n  Gemiddelde afwijking {afwijking:+.1f}: geen duidelijke scheefheid.")
    if k["paren"] < 10:
        print(f"  Let op: {k['paren']} waarnemingen is te weinig om conclusies aan te verbinden.")


def _blinde_vlek(regels: list[dict]) -> None:
    v = kern.blinde_vlek(regels)
    print("\n## Verdeling van de tussenchecks")
    for oordeel in TUSSEN_VOLGORDE:
        print(f"  {oordeel:<10} {v['verdeling'].get(oordeel, 0)}")

    gedaan, totaal = v["verdiepen_getoetst"]
    print("\n## Wat je meet, en wat je niet meet")
    print(f"  VERDIEPEN met analyse : {gedaan} van {totaal}  -> vals positieven zijn zichtbaar")
    if v["vals_pos"]:
        print(f"    waarvan PASS: {', '.join(v['vals_pos'])} — uren die niets opleverden")

    ongetoetst = v["overslaan_ongetoetst"]
    print(f"  OVERSLAAN zonder analyse: {len(ongetoetst)} van {v['overslaan_totaal']}"
          f"  -> vals negatieven zijn ONZICHTBAAR")
    if ongetoetst:
        print(f"    {', '.join(ongetoetst)}")
        print("    Van deze groep weet je niets. Wil je dat wel, dan is er maar één weg:")
        print("    draai er af en toe alsnog een volledige analyse op — bij voorkeur")
        print("    degene die het dichtst bij de streep lag.")


def _rendement_per_groep(regels: list[dict]) -> None:
    groepen = kern.rendement_per_groep(regels)
    print("\n## Rendement sinds het oordeel, per groep")
    if not groepen:
        print("  Geen vergelijkbare koersen beschikbaar.")
        return
    for oordeel in TUSSEN_VOLGORDE:
        if oordeel in groepen:
            n, gem = groepen[oordeel]
            print(f"  {oordeel:<10} n={n:<3} gemiddeld {gem:+.1f}%")
    print("\n  Dit is een rookmelder, geen thermometer. Bij dit aantal namen en deze")
    print("  looptijd is het verschil tussen de groepen vrijwel zeker ruis. Het wordt")
    print("  pas interessant als OVERSLAAN over een jaar of langer stelselmatig")
    print("  bóven de andere twee uitkomt — dan gooi je winnaars weg.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--koersen", action="store_true",
                   help="haal de actuele koersen op en toon het rendement per groep")
    args = p.parse_args()

    regels = kern.verzamel()
    if not regels:
        print("Geen tussenchecks of analyses gevonden.")
        return 1

    if args.koersen:
        _live_koersen(regels)

    print(f"# Scorebord onderzoekspijplijn — {len(regels)} bedrijven\n")
    _tabel(regels, args.koersen)
    _kalibratie(regels)
    _blinde_vlek(regels)
    if args.koersen:
        _rendement_per_groep(regels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
