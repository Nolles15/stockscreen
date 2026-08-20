"""
dubbelingen.py — herkent hetzelfde bedrijf onder meerdere tickers.

Aanleiding: AS Silvano Fashion Group stond tweemaal in de top-10 van de
kansenlijst, als SFG.WA en SFG1T.TL. Twee van de tien plekken voor één bedrijf,
zonder dat je dat kon zien.

Er zijn twee soorten, en ze vragen om een ander oordeel:

  **dubbele notering** — hetzelfde aandeel op twee beurzen (ASML / ASML.AS).
  Eén ervan is overbodig; meestal is er één waar de databron beter dekt.

  **aandelenklasse** — verschillende stukken van hetzelfde bedrijf, met eigen
  koers en stemrecht (VOLV-A.ST / VOLV-B.ST). Dat zijn geen duplicaten in
  boekhoudkundige zin, maar voor een kansenlijst wil je ze meestal niet allebei.

Het onderscheid zit in de beurs: dezelfde beurs betekent aandelenklassen,
verschillende beurzen betekent een dubbele notering.

Deze module markeert alleen. Wegfilteren gebeurt niet: de screener laat zien wat
hij vindt en jij beslist wat ermee gebeurt.
"""

from __future__ import annotations

import re
from collections import defaultdict

# Rechtsvormen en toevoegingen die niets zeggen over de identiteit van het
# bedrijf. Langste eerst, zodat ' ab (publ)' eerder valt dan ' ab'.
_SUFFIXEN = (
    " ab (publ)", " (publ)", " a/s", " s.p.a.", " s.a.", " n.v.", " oyj", " asa",
    " plc", " inc.", " corp.", " ltd.", " ag", " se", " nv", " sa", " ab", " spa",
    " inc", " corp", " ltd", " oy", " group", " holding", " holdings",
)


def normaliseer(naam: str | None) -> str | None:
    """Bedrijfsnaam terugbrengen tot iets vergelijkbaars."""
    if not naam:
        return None
    n = naam.lower().strip()
    for s in _SUFFIXEN:
        n = n.replace(s, " ")
    n = re.sub(r"[^a-z0-9]", "", n)
    return n or None


def _beurs(ticker: str) -> str:
    """Het beurssuffix, of een lege string voor Amerikaanse tickers."""
    return ticker.rsplit(".", 1)[1] if "." in ticker else ""


def vind(stocks: list[dict]) -> dict[str, dict]:
    """
    {ticker: {"dubbel_van": [andere tickers], "dubbel_soort": str}}

    `stocks` zijn rijen met minimaal `ticker` en `name`; `isin` wordt gebruikt
    als die er is. ISIN is het sterkste bewijs maar is bij ongeveer de helft van
    de aandelen niet gevuld, dus de naam blijft nodig.
    """
    per_sleutel: dict[tuple, list[str]] = defaultdict(list)

    for s in stocks:
        ticker = s.get("ticker")
        if not ticker:
            continue
        isin = (s.get("isin") or "").strip()
        if isin:
            per_sleutel[("isin", isin)].append(ticker)
            continue
        naam = normaliseer(s.get("name"))
        if naam:
            per_sleutel[("naam", naam)].append(ticker)

    resultaat: dict[str, dict] = {}
    for (soort_sleutel, _), tickers in per_sleutel.items():
        if len(tickers) < 2:
            continue
        beurzen = {_beurs(t) for t in tickers}
        soort = "aandelenklasse" if len(beurzen) == 1 else "dubbele notering"
        for t in tickers:
            resultaat[t] = {
                "dubbel_van": sorted(x for x in tickers if x != t),
                "dubbel_soort": soort,
                "dubbel_bewijs": "isin" if soort_sleutel == "isin" else "naam",
            }
    return resultaat
