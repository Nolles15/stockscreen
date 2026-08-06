"""
scorebord.py — hoe goed voorspelt de tussencheck wat de analyse vindt?

De pijplijn maakt twee soorten fouten, en maar één daarvan is meetbaar:

  vals positief   VERDIEPEN -> de analyse komt op PASS
                  Kost uren. Meetbaar, want je hebt de analyse gedaan.

  vals negatief   OVERSLAAN -> het was een KOOP geweest
                  Kost een gemiste kans. NIET meetbaar, want je doet de
                  analyse nooit. Zelfde probleem als een bank die niet weet
                  hoe afgewezen aanvragers het zouden hebben gedaan.

Deze module meet wat wél te meten valt en maakt zichtbaar wat ontbreekt:

1. **Per bedrijf** (`verzamel`) — oordeel, datum, voorspelling, en de
   werkelijke uitslag als er een analyse ligt.
2. **Kalibratie** (`kalibratie`) — hoe vaak klopte de voorspelling, en wijkt
   hij stelselmatig één kant op af? Dat is het goedkoopste signaal dat er is:
   het vraagt geen marktdata en geen wachttijd, alleen tien tussenchecks.
3. **Wat je niet ziet** (`blinde_vlek`) — hoeveel OVERSLAAN-oordelen er staan
   zonder dat er ooit een analyse tegenaan is gehouden. Dat getal is de omvang
   van je blinde vlek, niet van je succes.

`met_koersen` zet er het rendement sinds het oordeel bij. Waarschuwing die
erbij hoort: met tien tot dertig namen over enkele maanden is dat ruis. Het is
een rookmelder, geen thermometer — en een goed bedrijf kan een slecht aandeel
zijn.

Waarom deze code in `engine/` staat en niet in `scripts/`: `.dockerignore`
sluit `scripts/` uit van de build, dus een Flask-route kan er niet uit
importeren. `scripts/scorebord.py` is nog slechts de opdrachtregel-schil.
"""

import os
import re

from . import analyses as analyses_mod
from . import oordelen as oordelen_mod

# Volgorde van zwaar naar licht, voor het uitlijnen van de tellingen.
TUSSEN_VOLGORDE = ["VERDIEPEN", "TWIJFEL", "OVERSLAAN"]
ANALYSE_VOLGORDE = ["KOOP", "HOLD", "PASS"]

# Rangschaal om te bepalen of een voorspelling te hoog of te laag zat.
_RANG = {"PASS": 0, "HOLD": 1, "KOOP": 2}


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


def verzamel() -> list[dict]:
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

    return sorted(per_ticker.values(),
                  key=lambda r: (r["tc_datum"] or r["an_datum"] or "", r["ticker"]))


def kalibratie(regels: list[dict]) -> dict:
    """Hoe goed voorspelde de tussencheck de uiteindelijke analyse?

    Alleen de regels met beide velden tellen mee. `afwijking` is het gemiddelde
    verschil op de rangschaal PASS=0 / HOLD=1 / KOOP=2: positief betekent
    stelselmatig te optimistisch.
    """
    paren = [r for r in regels if r["voorspelling"] and r["analyse"]]
    if not paren:
        return {"paren": 0, "raak": 0, "details": [], "afwijking": None,
                "totaal": len(regels)}

    details = []
    for r in paren:
        verschil = _RANG[r["voorspelling"]] - _RANG[r["analyse"]]
        richting = ("te optimistisch" if verschil > 0
                    else ("te somber" if verschil < 0 else "raak"))
        details.append((r["ticker"], r["voorspelling"], r["analyse"], richting))

    afwijking = sum(_RANG[r["voorspelling"]] - _RANG[r["analyse"]]
                    for r in paren) / len(paren)
    return {
        "paren": len(paren),
        "raak": sum(1 for r in paren if r["voorspelling"] == r["analyse"]),
        "details": details,
        "afwijking": afwijking,
        "totaal": len(regels),
    }


def blinde_vlek(regels: list[dict]) -> dict:
    """Wat er wél en niet gemeten wordt.

    `overslaan_ongetoetst` is het getal dat ertoe doet: van die groep is per
    constructie niets bekend, want er komt nooit een analyse tegenaan.
    """
    verdiepen = [r for r in regels if r["tussencheck"] == "VERDIEPEN"]
    gedaan = [r for r in verdiepen if r["analyse"]]
    overslaan = [r for r in regels if r["tussencheck"] == "OVERSLAAN"]
    return {
        "verdeling": {o: sum(1 for r in regels if r["tussencheck"] == o)
                      for o in TUSSEN_VOLGORDE},
        "verdiepen_getoetst": (len(gedaan), len(verdiepen)),
        "vals_pos": [r["ticker"] for r in gedaan if r["analyse"] == "PASS"],
        "overslaan_ongetoetst": [r["ticker"] for r in overslaan if not r["analyse"]],
        "overslaan_totaal": len(overslaan),
    }


def met_koersen(regels: list[dict], dashboard_rijen: list[dict]) -> None:
    """Zet de actuele koers en het rendement sinds het oordeel bij elke regel.

    Bewust `oordelen.verrijk` hergebruiken in plaats van hier een tweede
    ticker-matcher te schrijven: twee matchers lopen uiteen, en die van
    oordelen.py kent al de gevallen die pijn doen (Yahoo-symbolen, kale
    tickers, tweede noteringen op een andere beurs).

    De rijen komen van de aanroeper — vanuit de webapp rechtstreeks uit de
    database, vanaf de opdrachtregel uit /api/dashboard.
    """
    rijen = list(dashboard_rijen or [])
    if not rijen:
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


def rendement_per_groep(regels: list[dict]) -> dict:
    """Gemiddeld rendement sinds het oordeel, per tussencheck-uitslag."""
    met = [r for r in regels if r.get("rendement") is not None and r["tussencheck"]]
    uit = {}
    for oordeel in TUSSEN_VOLGORDE:
        groep = [r for r in met if r["tussencheck"] == oordeel]
        if groep:
            uit[oordeel] = (len(groep), sum(r["rendement"] for r in groep) / len(groep))
    return uit


def pijplijn_overzicht() -> dict:
    """Samenvatting voor de kop boven de onderzoekspagina's."""
    regels = verzamel()
    tussenchecks = [r for r in regels if r["tussencheck"]]
    analyses = [r for r in regels if r["analyse"]]
    return {
        "n_analyses": len(analyses),
        "n_tussenchecks": len(tussenchecks),
        "verdeling_tc": {o: sum(1 for r in tussenchecks if r["tussencheck"] == o)
                         for o in TUSSEN_VOLGORDE},
        "verdeling_an": {o: sum(1 for r in analyses if r["analyse"] == o)
                         for o in ANALYSE_VOLGORDE},
        "kalibratie": kalibratie(regels),
        "vlek": blinde_vlek(regels),
    }
