"""
Moat-profiel: heeft dit bedrijf een duurzaam concurrentievoordeel?

Waarom dit bestaat. De screener meet de prijs goed, maar niet of het een goed
bedrijf is. Uit 19 voltooide diepe analyses bleek dat het oordeel bijna volledig
door die tweede vraag wordt bepaald: de drie KOOP-oordelen scoorden gemiddeld
12,7 van 15 op moat + Buffett + management, al het andere 8,8. Vijf aandelen
waren goedkoop (DCF-score 4-5) maar zwak op kwaliteit — alle vijf werden HOLD.
De bestaande `quality_score` vangt dit niet: die meet balans- en
winstgevendheidsmechaniek, geen concurrentiepositie.

Wat dit wél en niet kan. Cijfers laten zien of hoge rendementen *standhouden* —
dat is de vingerafdruk van een moat, niet het bewijs ervan. Getest op 16 bekende
gevallen scheidt dit de uitersten betrouwbaar, maar het middenveld niet: Ambra
en Telekom Austria ogen cijfermatig prima en scoorden toch 8 van 15. Daarom kent
dit profiel maar zuinig groen toe en is geel expliciet "de cijfers zeggen het
niet — dit vraagt onderzoek".

Gebruikt uitsluitend gegevens die al in de database staan: `financials` (5 jaar)
en `price_history` (tot 2016). Geen netwerkverkeer.
"""

from __future__ import annotations

import logging
import statistics
from datetime import date
from typing import Optional

log = logging.getLogger(__name__)

# Vennootschapsbelasting voor de NOPAT-benadering. Een vast tarief is hier
# eerlijker dan een afgeleid tarief: dat laatste schiet alle kanten op bij
# eenmalige posten en suggereert een precisie die er niet is.
TAX_RATE = 0.25

# Drempels, gekalibreerd op de 16 gevallen waarvan de diepe analyse bekend is.
# Groen eist én een hoog rendement én dat het rendement standhoudt: dat is wat
# WKL (19,3% mediaan, dieptepunt op 88% daarvan) en Adyen onderscheidt van
# Sdiptech (8,4% mediaan, dieptepunt op 48%).
ROIC_GROEN = 15.0          # mediaan ROIC waarboven een moat aannemelijk is
ROIC_ZWAK = 6.0            # mediaan ROIC waaronder het rendement te mager is
STABIEL_GROEN = 0.70       # slechtste jaar / mediaan — hoger is standvastiger
STABIEL_ROOD = 0.65        # hieronder stort het rendement in slechte jaren in
MARGE_EROSIE_PP = -3.0     # brutomarge-daling over de reeks die telt als erosie

MIN_JAREN = 3              # minder dan drie jaar zegt niets over standvastigheid

# De cyclusvraag is "is dit aandeel ooit hard geraakt en kwam het terug". Twee
# jaar is de ondergrens waarop dat een betekenis heeft; korter en je meet de
# ruis van een handvol handelsdagen. De puntengrens vangt het andere geval af:
# een lange reeks met maar drie koersen erin.
MIN_CYCLUS_JAREN = 2.0
MIN_CYCLUS_PUNTEN = 24


def _f(row: dict, key: str) -> Optional[float]:
    v = row.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _trend(reeks: list[float]) -> Optional[float]:
    """Verschil tussen laatste en eerste waarneming, in procentpunten."""
    return (reeks[-1] - reeks[0]) if len(reeks) >= 2 else None


# ---------------------------------------------------------------------------
# Rendement op geïnvesteerd kapitaal
# ---------------------------------------------------------------------------

def roic_reeks(annual: list[dict]) -> list[tuple[int, float]]:
    """
    ROIC per boekjaar, oplopend in tijd. [(jaar, percentage), ...]

    NOPAT / geïnvesteerd kapitaal, met geïnvesteerd kapitaal = eigen vermogen +
    rentedragende schuld. Bewust de eenvoudige definitie: goodwill eruit filteren
    vraagt gegevens die Yahoo niet betrouwbaar levert, en dan lijkt het
    nauwkeuriger dan het is.
    """
    uit = []
    for rij in sorted(annual, key=lambda r: r.get("fiscal_year") or 0):
        jaar = rij.get("fiscal_year")
        ebit = _f(rij, "ebit")
        eigen = _f(rij, "total_equity")
        schuld = _f(rij, "total_debt") or 0.0
        if jaar is None or ebit is None or eigen is None:
            continue
        kapitaal = eigen + schuld
        if kapitaal <= 0:
            # Negatief eigen vermogen maakt ROIC betekenisloos (of absurd hoog).
            continue
        uit.append((int(jaar), 100.0 * ebit * (1 - TAX_RATE) / kapitaal))
    return uit


def marge_reeks(annual: list[dict], teller: str) -> list[tuple[int, float]]:
    """Marge per boekjaar als percentage van de omzet, oplopend in tijd."""
    uit = []
    for rij in sorted(annual, key=lambda r: r.get("fiscal_year") or 0):
        jaar = rij.get("fiscal_year")
        boven = _f(rij, teller)
        omzet = _f(rij, "revenue")
        if jaar is None or boven is None or not omzet or omzet <= 0:
            continue
        uit.append((int(jaar), 100.0 * boven / omzet))
    return uit


# ---------------------------------------------------------------------------
# Cyclustest op de koershistorie
# ---------------------------------------------------------------------------

def cyclustest(price_history: list[dict]) -> dict:
    """
    Hoe kwam dit aandeel door slecht weer?

    Vijf jaar jaarcijfers laat geen enkele recessie zien. De koershistorie gaat
    verder terug en beantwoordt daarmee de vraag waar tien jaar cijfers voor
    bedoeld waren: is dit bedrijf ooit hard geraakt, en kwam het terug?

    `betrouwbaar` is het veld dat ertoe doet. Een reeks van twee weken levert
    keurige getallen op die niets betekenen: de diepste terugval is dan de
    grootste beweging van die twee weken en de "hoogste slotkoers ooit" is de
    hoogste van die twee weken. Zo belandde "staat weer op recordhoogte" in de
    tussenchecks van Wittchen, Infotel en Synergie terwijl die aandelen tientallen
    procenten onder hun top stonden, en op 6 augustus 2026 opnieuw bij Zealand
    Pharma (57% onder de top) en Moberg Pharma. Drie keer handmatig gecorrigeerd
    is twee keer te vaak.

    De cijfers blijven staan — ze worden alleen niet meer als conclusie
    gepresenteerd zolang de reeks te kort is.
    """
    koersen = [(r.get("date"), _f(r, "close")) for r in price_history]
    koersen = [(d, c) for d, c in koersen if d and c and c > 0]
    if len(koersen) < 2:
        return {"beschikbaar": False, "betrouwbaar": False}

    koersen.sort(key=lambda x: x[0])
    top = koersen[0][1]
    diepste = 0.0
    diepste_datum = None
    for datum, koers in koersen:
        if koers > top:
            top = koers
        terugval = 100.0 * (koers - top) / top
        if terugval < diepste:
            diepste, diepste_datum = terugval, datum

    alletijden_top = max(c for _, c in koersen)
    laatste = koersen[-1][1]

    # Op dagen rekenen, niet op kalenderjaren: 31 december tot 1 januari is
    # anders "1 jaar", en 27 juli tot 5 augustus "0 jaar" terwijl beide
    # net zo kort zijn.
    try:
        van = date.fromisoformat(koersen[0][0][:10])
        tot = date.fromisoformat(koersen[-1][0][:10])
        jaren = round((tot - van).days / 365.25, 1)
    except (ValueError, TypeError):
        jaren = 0.0

    return {
        "beschikbaar": True,
        "betrouwbaar": jaren >= MIN_CYCLUS_JAREN and len(koersen) >= MIN_CYCLUS_PUNTEN,
        "jaren": jaren,
        "punten": len(koersen),
        "vanaf": koersen[0][0],
        "diepste_terugval_pct": round(diepste, 1),
        "diepste_terugval_datum": diepste_datum,
        "onder_top_pct": round(100.0 * (laatste - alletijden_top) / alletijden_top, 1),
    }


# ---------------------------------------------------------------------------
# Oordeel
# ---------------------------------------------------------------------------

def _oordeel(roics: list[float], bruto_trend: Optional[float],
             jaren: int) -> tuple[str, str]:
    """Geeft (niveau, kop). Niveau is groen/geel/rood/grijs zoals de beslisboom."""
    if jaren < MIN_JAREN or not roics:
        return "grijs", "Te weinig jaren om iets over standvastigheid te zeggen"

    mediaan = statistics.median(roics)
    laagste = min(roics)
    standvastig = laagste / mediaan if mediaan > 0 else 0.0

    erosie = bruto_trend is not None and bruto_trend <= MARGE_EROSIE_PP

    # Ligt het dieptepunt in het éérste jaar en staat het laatste jaar op of
    # boven de mediaan, dan is het rendement niet ingestort maar opgeklommen.
    # Zonder dit onderscheid krijgt een bedrijf dat zijn zaken op orde bracht
    # hetzelfde rode stempel als een bedrijf dat wegzakt: bij Huuuge was het
    # slechtste jaar 2022 (11,6%) waarna het opliep naar 48,3% in 2025, en dat
    # werd gelezen als "stort in bij tegenwind".
    herstellend = len(roics) >= 3 and roics.index(laagste) == 0 and roics[-1] >= mediaan

    # Rood: het rendement houdt geen stand, is structureel mager, of de marge
    # brokkelt af. Dit is het profiel dat in de diepe analyse steevast op HOLD
    # of PASS uitkwam, ook wanneer het aandeel goedkoop leek.
    if mediaan <= 0:
        return "rood", "Verdient geen rendement op het geïnvesteerde kapitaal"
    if standvastig < STABIEL_ROOD and not herstellend:
        return "rood", "Het rendement stort in bij tegenwind"
    if mediaan < ROIC_ZWAK:
        return "rood", "Structureel mager rendement op kapitaal"
    if erosie:
        return "rood", "De brutomarge brokkelt af"

    # Een herstellende reeks is geen instorting, maar ook geen bewijs van
    # duurzaamheid: er is maar een paar jaar goed rendement en nog geen enkel
    # slecht jaar doorstaan. Dat hoort in het midden, niet in het groen.
    if herstellend and standvastig < STABIEL_GROEN:
        return "geel", "Rendement is opgeklommen, maar nog te kort op niveau"

    # Groen alleen bij hoog én standvastig rendement — dat is het profiel van
    # de aandelen die de diepe analyse tot KOOP bestempelde.
    if mediaan >= ROIC_GROEN and standvastig >= STABIEL_GROEN:
        return "groen", "Hoog rendement dat standhoudt"

    # Alles daartussen: de cijfers spreken zich niet uit. Bewust geen groen —
    # Ambra en Telekom Austria zagen er cijfermatig prima uit en scoorden toch
    # 8 van 15 in de diepe analyse.
    return "geel", "Cijfers wijzen niet op een duidelijke moat"


def bouw_profiel(annual: list[dict], price_history: list[dict] | None = None) -> dict:
    """
    Bouwt het volledige moat-profiel. Puur op meegegeven data, geen DB-toegang,
    zodat het los te testen is.
    """
    roic = roic_reeks(annual)
    bruto = marge_reeks(annual, "gross_profit")
    operationeel = marge_reeks(annual, "ebit")

    roics = [w for _, w in roic]
    brutos = [w for _, w in bruto]
    bruto_trend = _trend(brutos)

    niveau, kop = _oordeel(roics, bruto_trend, len(roics))

    profiel: dict = {
        "niveau": niveau,
        "kop": kop,
        # De gehanteerde drempels reizen mee, zodat de methodepagina ze niet
        # apart hoeft te kennen — anders ontstaat er een tweede plek waar ze
        # staan, en die loopt gegarandeerd uit de pas.
        "drempels": {
            "belastingtarief": TAX_RATE,
            "roic_groen": ROIC_GROEN,
            "roic_zwak": ROIC_ZWAK,
            "standvastig_groen": STABIEL_GROEN,
            "standvastig_rood": STABIEL_ROOD,
            "marge_erosie_pp": MARGE_EROSIE_PP,
            "minimaal_jaren": MIN_JAREN,
        },
        "formule_roic": "EBIT × (1 − belastingtarief) / (eigen vermogen + schuld)",
        "roic": [{"jaar": j, "pct": round(w, 1)} for j, w in roic],
        "brutomarge": [{"jaar": j, "pct": round(w, 1)} for j, w in bruto],
        "operationele_marge": [{"jaar": j, "pct": round(w, 1)} for j, w in operationeel],
        "cyclus": cyclustest(price_history or []),
        "redenen": [],
    }

    if roics:
        mediaan = statistics.median(roics)
        laagste = min(roics)
        laagste_jaar = min(roic, key=lambda x: x[1])[0]
        profiel["roic_mediaan"] = round(mediaan, 1)
        profiel["roic_laagste"] = round(laagste, 1)
        profiel["roic_standvastig"] = round(laagste / mediaan, 2) if mediaan > 0 else None
        profiel["redenen"].append(
            f"Rendement op kapitaal: mediaan {mediaan:.1f}%, "
            f"laagste jaar {laagste:.1f}% ({laagste_jaar})"
        )
    if brutos and bruto_trend is not None:
        richting = "stijgt" if bruto_trend > 0.5 else "daalt" if bruto_trend < -0.5 else "blijft vlak"
        profiel["brutomarge_trend_pp"] = round(bruto_trend, 1)
        profiel["redenen"].append(
            f"Brutomarge {richting}: {brutos[0]:.0f}% → {brutos[-1]:.0f}% "
            f"({bruto_trend:+.1f} procentpunt)"
        )
    cyc = profiel["cyclus"]
    if cyc.get("betrouwbaar"):
        profiel["redenen"].append(
            f"Zwaarste koersval sinds {cyc['vanaf'][:4]}: {cyc['diepste_terugval_pct']:.0f}%"
            + (f", nu {abs(cyc['onder_top_pct']):.0f}% onder de top"
               if cyc["onder_top_pct"] < -5 else ", staat weer op recordhoogte")
        )
    elif cyc.get("beschikbaar"):
        # Wel koersen, te weinig om iets te beweren. Dit expliciet zeggen is het
        # halve punt van de reparatie: een ontbrekende regel wordt gelezen als
        # "niets bijzonders", een gemarkeerde regel als "hier weten we het niet".
        maanden = max(1, round(cyc.get("jaren", 0) * 12))
        profiel["redenen"].append(
            f"Koershistorie beslaat pas {maanden} maand{'en' if maanden != 1 else ''} "
            f"({cyc.get('punten', 0)} koersen) — te kort voor een uitspraak over koersval"
        )

    return profiel
