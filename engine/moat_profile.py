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
    """
    koersen = [(r.get("date"), _f(r, "close")) for r in price_history]
    koersen = [(d, c) for d, c in koersen if d and c and c > 0]
    if len(koersen) < 2:
        return {"beschikbaar": False}

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
    return {
        "beschikbaar": True,
        "jaren": round((int(koersen[-1][0][:4]) - int(koersen[0][0][:4])), 1),
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

    # Rood: het rendement houdt geen stand, is structureel mager, of de marge
    # brokkelt af. Dit is het profiel dat in de diepe analyse steevast op HOLD
    # of PASS uitkwam, ook wanneer het aandeel goedkoop leek.
    if mediaan <= 0:
        return "rood", "Verdient geen rendement op het geïnvesteerde kapitaal"
    if standvastig < STABIEL_ROOD:
        return "rood", "Het rendement stort in bij tegenwind"
    if mediaan < ROIC_ZWAK:
        return "rood", "Structureel mager rendement op kapitaal"
    if erosie:
        return "rood", "De brutomarge brokkelt af"

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
    if cyc.get("beschikbaar"):
        profiel["redenen"].append(
            f"Zwaarste koersval sinds {cyc['vanaf'][:4]}: {cyc['diepste_terugval_pct']:.0f}%"
            + (f", nu {abs(cyc['onder_top_pct']):.0f}% onder de top"
               if cyc["onder_top_pct"] < -5 else ", staat weer op recordhoogte")
        )

    return profiel
