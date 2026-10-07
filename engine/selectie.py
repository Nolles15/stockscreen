"""
selectie.py — welke rijen horen bij welk tabblad, en bij welke filters.

Deze regels stonden in de browser, als filters over álle 2.812 rijen. Dat werkte
zolang "alle rijen" een redelijke lading was; bij 19.000 tickers is het 28 MB en
tienduizenden tabelregels die de browser moet tekenen voordat je iets ziet.

Nu kiest de server. Dat scheelt niet alleen lading — het haalt ook de definitie
van een tabblad weg uit de browser, zodat hij nog maar op één plek staat. Precies
de reden dat `reason_code` destijds niet naar de browser is gekopieerd: twee
plekken die hetzelfde beweren gaan vroeg of laat uit de pas lopen.

**Geen databaseverkeer.** Er wordt gefilterd over de rijen die `engine/cache.py`
al in het geheugen van het proces houdt. Neon merkt hier niets van; wat we hier
besparen is de lading naar de browser en het tekenwerk daar.
"""

from __future__ import annotations

from typing import Any, Callable

from engine import markets

# Boven dit aantal gaat er niet meer alles in één keer heen. Gemeten op
# 2026-08-22 zat élk tabblad daaronder — het grootste is "Alles" met 570 rijen,
# omdat lage kwaliteit daar standaard verborgen is. De grens is er dus voor
# later: hij treedt pas op als het universum flink groeit of als je "toon alles"
# aanzet. Dat wordt zichtbaar gemeld, nooit stilzwijgend afgekapt.
LIJST_MAX = 1500

Rij = dict[str, Any]


# ---------------------------------------------------------------------------
# Hulpjes
# ---------------------------------------------------------------------------

def _getal(r: Rij, sleutel: str, leeg: float = -1e9) -> float:
    """Een sorteerbare waarde; ontbrekend telt als kleinst, dus onderaan."""
    v = r.get(sleutel)
    return leeg if v is None else v


def _sector(r: Rij) -> str:
    return r.get("sector") or "Onbekend"


# ---------------------------------------------------------------------------
# De tabbladen
# ---------------------------------------------------------------------------
#
# Elk tabblad is een filter plus een volgorde. De teksten die erbij horen (de
# uitleg boven de tabel, de kolomkoppen) blijven in de browser: dat is
# presentatie en verandert niets aan wélke rijen je ziet.

def _kansen(rijen: list[Rij]) -> list[Rij]:
    uit = [r for r in rijen if r.get("rank_score") is not None]
    uit.sort(key=lambda r: -_getal(r, "rank_score"))
    return uit[:20]


def _kwaliteit(rijen: list[Rij]) -> list[Rij]:
    uit = [r for r in rijen
           if (r.get("quality_score") or 0) >= 8.5
           and r.get("data_status") in ("ok", "warning")
           and r.get("sector") != "Financial Services"]
    uit.sort(key=lambda r: (-_getal(r, "quality_score"),
                            -_getal(r, "margin_of_safety", -999)))
    return uit


def _herstellers(rijen: list[Rij]) -> list[Rij]:
    uit = []
    for r in rijen:
        q = r.get("quality_score")
        if q is None or not (5 <= q < 7):
            continue
        pvf = r.get("price_vs_fv_pct")
        if pvf is None or pvf > 70:
            continue
        if r.get("fv_confidence") != "high":
            continue
        if r.get("data_status") not in ("ok", "warning"):
            continue
        if r.get("sector") == "Financial Services":
            continue
        uit.append(r)
    uit.sort(key=lambda r: -_getal(r, "margin_of_safety", -999))
    return uit


# Rood staat onderaan maar blijft zichtbaar: markeren, niet wegfilteren.
NIVEAU_RANG = {"groen": 3, "geel": 2, "grijs": 1, "rood": 0}


def _groeiers(rijen: list[Rij]) -> list[Rij]:
    """Alle snelle groeiers, gerangschikt op het groeiprofiel (zeef, geen oordeel)."""
    uit = [r for r in rijen if r.get("is_groeier")]
    uit.sort(key=lambda r: (-NIVEAU_RANG.get(r.get("groei_niveau"), 1),
                            -_getal(r, "groei_score", -1),
                            -_getal(r, "revenue_cagr", 0)))
    return uit


def _geen_oordeel(rijen: list[Rij]) -> list[Rij]:
    uit = [r for r in rijen if r.get("signal") == "INSUFFICIENT DATA"]
    uit.sort(key=lambda r: ((r.get("reason_label") or ""), r.get("ticker") or ""))
    return uit


def _alles(rijen: list[Rij]) -> list[Rij]:
    return list(rijen)


TABBLADEN: dict[str, Callable[[list[Rij]], list[Rij]]] = {
    "kansen":      _kansen,
    "kwaliteit":   _kwaliteit,
    "herstellers": _herstellers,
    "groeiers":    _groeiers,
    "geenoordeel": _geen_oordeel,
    "alles":       _alles,
    # "pins" krijgt zijn tickers van de browser (die staan in localStorage) en
    # "bezit" heeft een eigen bron; allebei hieronder afgehandeld.
    "pins":        _alles,
    "bezit":       _alles,
}


# ---------------------------------------------------------------------------
# Filters die over álle tabbladen heen gelden
# ---------------------------------------------------------------------------

def _past_bij_signaal(r: Rij, signaal: str) -> bool:
    if not signaal:
        return True
    if signaal == "koop":
        return r.get("signal") in ("STRONG BUY", "BUY")
    return r.get("signal") == signaal


def _extra_filters(rijen: list[Rij], extra: dict) -> list[Rij]:
    """De filters van het tabblad "Alles" — markt, reden en de drie vinkjes.

    `toon_alles` staat standaard uit, en dat verbergt lage kwaliteit. Dat is
    geen detail: het brengt 2.812 rijen terug tot 570, en het is de reden dat
    zelfs "Alles" nu ruim binnen `LIJST_MAX` past.
    """
    markt = extra.get("markt") or ""
    reden = extra.get("reden") or ""
    uit = []
    for r in rijen:
        if markt and r.get("market") != markt:
            continue
        if reden:
            if reden == "growth":
                if not r.get("is_growth_lossmaker"):
                    continue
            elif r.get("reason_code") != reden:
                continue
        if extra.get("verberg_geen_data") and r.get("signal") == "INSUFFICIENT DATA":
            continue
        if extra.get("alleen_slechte_data") and r.get("data_status") not in ("bad", "missing"):
            continue
        if not extra.get("toon_alles") and r.get("low_quality"):
            continue
        uit.append(r)
    return uit


# ---------------------------------------------------------------------------
# De tellingen voor de keuzelijsten
# ---------------------------------------------------------------------------

def samenvatting(alle: list[Rij], land: str = "") -> dict:
    """Tellingen voor de land- en sectorkeuzelijst.

    Land wordt over álles geteld; sector binnen het gekozen land. Je wilt zien
    welke sectoren er in Finland te vinden zijn, niet wereldwijd — dat was al zo
    in de browser en blijft zo.
    """
    per_land: dict[str, int] = {}
    for r in alle:
        naam = markets.land_naam(r.get("ticker") or "")
        per_land[naam] = per_land.get(naam, 0) + 1

    basis = [r for r in alle
             if not land or markets.land_naam(r.get("ticker") or "") == land]
    per_sector: dict[str, int] = {}
    for r in basis:
        s = _sector(r)
        per_sector[s] = per_sector.get(s, 0) + 1

    # De marktkeuze op het tabblad "Alles" gebruikt de landcode uit de database,
    # niet de naam. Dat is een ander veld dan `land` hierboven en blijft het.
    markten = sorted({r.get("market") for r in alle if r.get("market")})

    return {
        "totaal": len(alle),
        "in_land": len(basis),
        "landen": [{"naam": k, "aantal": v} for k, v in sorted(per_land.items())],
        "sectoren": [{"naam": k, "aantal": v} for k, v in sorted(per_sector.items())],
        "markten": markten,
    }


# ---------------------------------------------------------------------------
# Sorteren
# ---------------------------------------------------------------------------
#
# Dit moet hier gebeuren en niet pas in de browser, omdat de lijst afgekapt kan
# worden: sorteren ná het afkappen zou de bovenste rijen uit een willekeurige
# greep halen in plaats van uit het geheel. Op een tabblad "Alles" van 19.000
# rijen is dat geen schoonheidsfoutje maar een verkeerd antwoord.
#
# Gelijk aan `vergelijk()` in index.html: signaal sorteert op sterkte, en lege
# waarden zakken altijd naar beneden ongeacht de richting — anders vult de
# bovenkant van de lijst zich met streepjes.

SIGNAAL_RANG = {"STRONG BUY": 5, "BUY": 4, "HOLD": 3, "SELL": 2,
                "INSUFFICIENT DATA": 1, "N/A": 0}


def _sorteerwaarde(r: Rij, kolom: str):
    if kolom == "signal":
        return SIGNAAL_RANG.get(r.get("signal"), -1)
    return r.get(kolom)


def sorteer(rijen: list[Rij], kolom: str, richting: int) -> list[Rij]:
    """Sorteer op één kolom; lege waarden onderaan, in beide richtingen."""
    if not kolom:
        return rijen

    gevuld, leeg = [], []
    for r in rijen:
        v = _sorteerwaarde(r, kolom)
        (leeg if v is None or v == "" else gevuld).append(r)

    omgekeerd = richting < 0
    try:
        gevuld.sort(key=lambda r: _sorteerwaarde(r, kolom), reverse=omgekeerd)
    except TypeError:
        # Gemengde typen in één kolom: dan maar als tekst, liever een rare
        # volgorde dan een lege tabel.
        gevuld.sort(key=lambda r: str(_sorteerwaarde(r, kolom)), reverse=omgekeerd)
    return gevuld + leeg


# ---------------------------------------------------------------------------
# Het geheel
# ---------------------------------------------------------------------------

def selecteer(alle: list[Rij], tab: str = "kansen", land: str = "",
              sector: str = "", signaal: str = "", tickers: list[str] | None = None,
              extra: dict | None = None, sorteer_op: str = "",
              richting: int = -1) -> dict:
    """De rijen voor één weergave, plus wat de keuzelijsten nodig hebben.

    De volgorde is niet vrijblijvend: land, signaal en sector gaan vóór het
    tabblad. "Koopwaardig" op Kansen levert daardoor de twintig beste
    koopwaardige aandelen, en niet het handjevol dat van de bestaande top twintig
    overblijft.
    """
    rijen = alle
    if land:
        rijen = [r for r in rijen if markets.land_naam(r.get("ticker") or "") == land]
    if signaal:
        rijen = [r for r in rijen if _past_bij_signaal(r, signaal)]
    if sector:
        rijen = [r for r in rijen if _sector(r) == sector]

    # Hoeveel er in de gekozen filters zitten, vóór het tabblad zijn greep doet.
    # Dat is het middelste getal onder de tabel: "20 aandelen · van 406 in Polen
    # · 2812 in totaal". Meten ná het tabblad zou daar "van 20" van maken, en
    # dan zegt het niets meer.
    in_selectie = len(rijen)

    if tab == "pins":
        gekozen = set(tickers or [])
        rijen = [r for r in rijen if r.get("ticker") in gekozen]
        rijen.sort(key=lambda r: -_getal(r, "rank_score", 0))
    else:
        if tab == "alles":
            rijen = _extra_filters(rijen, extra or {})
        rijen = TABBLADEN.get(tab, _alles)(rijen)

    if sorteer_op:
        rijen = sorteer(rijen, sorteer_op, richting)

    totaal = len(rijen)
    afgekapt = totaal > LIJST_MAX
    return {
        "rijen": rijen[:LIJST_MAX] if afgekapt else rijen,
        "totaal": totaal,
        "in_selectie": in_selectie,
        "afgekapt": afgekapt,
        "samenvatting": samenvatting(alle, land),
    }
