"""
scorebord.py — hoe goed voorspelt de tussencheck wat de analyse vindt?

De pijplijn maakt twee soorten fouten, en maar één daarvan is meetbaar:

  vals positief   VERDIEPEN -> de analyse komt op PASS
                  Kost uren. Meetbaar, want je hebt de analyse gedaan.

  vals negatief   OVERSLAAN -> het was een KOOP geweest
                  Kost een gemiste kans. NIET meetbaar, want je doet de
                  analyse nooit. Zelfde probleem als een bank die niet weet
                  hoe afgewezen aanvragers het zouden hebben gedaan.

Dit script meet wat wél te meten valt en maakt zichtbaar wat ontbreekt. Drie
onderdelen:

1. **Per ticker** — oordeel, datum, voorspelling, en de werkelijke uitslag als
   er een analyse ligt.
2. **Kalibratie** — hoe vaak klopte de voorspelling, en wijkt hij stelselmatig
   één kant op af? Dat is het goedkoopste signaal dat er is: het vraagt geen
   marktdata en geen wachttijd, alleen tien tussenchecks.
3. **Wat je niet ziet** — hoeveel OVERSLAAN-oordelen er staan zonder dat er
   ooit een analyse tegenaan is gehouden. Dat getal is de omvang van je blinde
   vlek, niet van je succes.

Met `--koersen` haalt hij ook de actuele koers op en zet het rendement sinds
het oordeel per groep naast elkaar. Waarschuwing die erbij hoort: met tien tot
dertig namen over enkele maanden is dat ruis. Het is een rookmelder, geen
thermometer — en een goed bedrijf kan een slecht aandeel zijn.

Gebruik:
  python scripts/scorebord.py
  python scripts/scorebord.py --koersen
"""

import argparse
import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from engine import analyses as analyses_mod  # noqa: E402
from engine import oordelen as oordelen_mod  # noqa: E402

API = "https://stockscreen-janco.fly.dev/api/dashboard"

# Volgorde van zwaar naar licht, voor het uitlijnen van de tellingen.
TUSSEN_VOLGORDE = ["VERDIEPEN", "TWIJFEL", "OVERSLAAN"]
ANALYSE_VOLGORDE = ["KOOP", "HOLD", "PASS"]


def _voorspelling(pad: str) -> str | None:
    """De KOOP/HOLD/PASS uit de sectie '## Voorspelling'.

    Alleen binnen die sectie zoeken, en alleen de eerste vetgedrukte waarde:
    de alinea eronder noemt vaak ook de alternatieven ('KOOP, met een reële
    kans op HOLD') en die tellen niet als voorspelling.
    """
    try:
        tekst = open(pad, encoding="utf-8").read()
    except OSError:
        return None
    m = re.search(r"^##\s*Voorspelling\s*$(.*?)(?=^##\s|\Z)", tekst, re.M | re.S)
    if not m:
        return None
    hit = re.search(r"\*\*(KOOP|HOLD|PASS)\*\*", m.group(1))
    return hit.group(1) if hit else None


def _verzamel() -> list[dict]:
    """Eén regel per onderzocht bedrijf, tussencheck en analyse samengevoegd."""
    per_ticker: dict[str, dict] = {}

    for check in analyses_mod.get_alle_tussenchecks():
        sleutel = check["ticker"].upper()
        per_ticker[sleutel] = {
            "ticker": sleutel,
            "naam": check["naam"],
            "tussencheck": check["oordeel"],
            "tc_datum": check["datum"],
            "tc_koers": check.get("koers_getal"),
            "tc_valuta": check.get("valuta"),
            "voorspelling": _voorspelling(
                os.path.join(analyses_mod.TUSSENCHECKS_DIR, check["bestand_ticker"] + ".md")
            ),
            "analyse": None, "an_datum": None, "score": None, "an_upside": None,
        }

    for analyse in analyses_mod.get_all_summaries():
        sleutel = analyse["ticker"].upper()
        regel = per_ticker.setdefault(sleutel, {
            "ticker": sleutel, "naam": analyse["naam"],
            "tussencheck": None, "tc_datum": None, "tc_koers": None,
            "tc_valuta": None, "voorspelling": None,
        })
        regel.update({
            "analyse": analyse["oordeel"],
            "an_datum": analyse["peildatum"],
            "score": analyse["score"],
            "an_upside": analyse["upside"],
        })

    return sorted(per_ticker.values(), key=lambda r: (r["tc_datum"] or r["an_datum"] or "", r["ticker"]))


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
    """Actuele koers erbij, via dezelfde koppeling als het dashboard.

    Bewust `oordelen.verrijk` hergebruiken in plaats van hier een tweede
    ticker-matcher te schrijven: twee matchers lopen uiteen, en die van
    oordelen.py kent al de gevallen die pijn doen (Yahoo-symbolen, kale
    tickers, tweede noteringen op een andere beurs).
    """
    try:
        rijen = _haal_dashboard()
    except Exception as fout:            # netwerk weg, of de app slaapt
        print(f"  [!] koersen ophalen mislukt ({fout}); rendement blijft leeg\n")
        return

    oordelen_mod.verrijk(rijen)
    per_link = {}
    for rij in rijen:
        oordeel = rij.get("oordeel")
        # Alleen directe koppelingen: een tweede notering staat in een andere
        # valuta en zou het rendement onvergelijkbaar maken.
        if oordeel and not oordeel.get("via"):
            per_link.setdefault(oordeel["link"], rij)

    for regel in regels:
        rij = (per_link.get(f"/tussenchecks/{regel['ticker']}")
               or per_link.get(f"/analyses/{regel['ticker']}"))
        if not rij:
            continue
        regel["nu_ticker"] = rij["ticker"]
        regel["nu_koers"] = rij.get("price")
        regel["nu_valuta"] = rij.get("currency")
        toen = regel.get("tc_koers")
        if (toen and rij.get("price")
                and regel.get("tc_valuta") == (rij.get("currency") or "").upper()):
            regel["rendement"] = round((rij["price"] - toen) / toen * 100, 1)


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
    paren = [r for r in regels if r["voorspelling"] and r["analyse"]]
    print(f"\n## Kalibratie — {len(paren)} van de {len(regels)} hebben voorspelling én uitslag")
    if not paren:
        print("  Nog niets te meten.")
        return

    goed = [r for r in paren if r["voorspelling"] == r["analyse"]]
    print(f"  Raak: {len(goed)} van {len(paren)}")
    rang = {"PASS": 0, "HOLD": 1, "KOOP": 2}
    for r in paren:
        verschil = rang[r["voorspelling"]] - rang[r["analyse"]]
        richting = "te optimistisch" if verschil > 0 else ("te somber" if verschil < 0 else "raak")
        print(f"    {r['ticker']:<9} voorspeld {r['voorspelling']:<5} -> werd {r['analyse']:<5}  {richting}")

    afwijking = sum(rang[r["voorspelling"]] - rang[r["analyse"]] for r in paren) / len(paren)
    if afwijking > 0.3:
        print(f"\n  Gemiddelde afwijking +{afwijking:.1f}: stelselmatig te optimistisch.")
        print("  Betekenis: de VERDIEPEN-drempel staat te ruim en je verspilt uren")
        print("  aan kandidaten die de analyse niet halen.")
    elif afwijking < -0.3:
        print(f"\n  Gemiddelde afwijking {afwijking:.1f}: stelselmatig te somber.")
        print("  Betekenis: je slaat waarschijnlijk kandidaten over die het wél waard waren.")
    else:
        print(f"\n  Gemiddelde afwijking {afwijking:+.1f}: geen duidelijke scheefheid.")
    if len(paren) < 10:
        print(f"  Let op: {len(paren)} waarnemingen is te weinig om conclusies aan te verbinden.")


def _blinde_vlek(regels: list[dict]) -> None:
    tussen = Counter(r["tussencheck"] for r in regels if r["tussencheck"])
    print("\n## Verdeling van de tussenchecks")
    for oordeel in TUSSEN_VOLGORDE:
        print(f"  {oordeel:<10} {tussen.get(oordeel, 0)}")

    verdiepen = [r for r in regels if r["tussencheck"] == "VERDIEPEN"]
    gedaan = [r for r in verdiepen if r["analyse"]]
    print("\n## Wat je meet, en wat je niet meet")
    print(f"  VERDIEPEN met analyse : {len(gedaan)} van {len(verdiepen)}  -> vals positieven zijn zichtbaar")
    vals_pos = [r for r in gedaan if r["analyse"] == "PASS"]
    if vals_pos:
        print(f"    waarvan PASS: {', '.join(r['ticker'] for r in vals_pos)} — uren die niets opleverden")

    overslaan = [r for r in regels if r["tussencheck"] == "OVERSLAAN"]
    ongetoetst = [r for r in overslaan if not r["analyse"]]
    print(f"  OVERSLAAN zonder analyse: {len(ongetoetst)} van {len(overslaan)}  -> vals negatieven zijn ONZICHTBAAR")
    if ongetoetst:
        print(f"    {', '.join(r['ticker'] for r in ongetoetst)}")
        print("    Van deze groep weet je niets. Wil je dat wel, dan is er maar één weg:")
        print("    draai er af en toe alsnog een volledige analyse op — bij voorkeur")
        print("    degene die het dichtst bij de streep lag.")


def _rendement_per_groep(regels: list[dict]) -> None:
    met = [r for r in regels if r.get("rendement") is not None and r["tussencheck"]]
    print("\n## Rendement sinds het oordeel, per groep")
    if not met:
        print("  Geen vergelijkbare koersen beschikbaar.")
        return
    for oordeel in TUSSEN_VOLGORDE:
        groep = [r for r in met if r["tussencheck"] == oordeel]
        if not groep:
            continue
        gem = sum(r["rendement"] for r in groep) / len(groep)
        print(f"  {oordeel:<10} n={len(groep):<3} gemiddeld {gem:+.1f}%")
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

    regels = _verzamel()
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
