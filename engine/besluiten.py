"""
besluiten.py — wat je met een oordeel deed, en vooral wat je niet deed.

Aanleiding. Er lagen vijf VERDIEPEN-oordelen — het sterkste dat de methode kent
— en geen daarvan zat in de portefeuille. Op de vraag waarom niet was het
antwoord: *"dat was niet echt een bewuste beslissing, ik heb het gewoon niet
gedaan."*

Dat is het hele probleem in één zin. Kopen laat een spoor na, niet-kopen niet.
Er is geen moment waarop je iets doet, dus valt er ook geen knop aan te hangen
en valt er achteraf niets van te leren. De uitkomst is dat het onderzoek wél
gebeurt en de conclusie wél getrokken wordt, maar dat de stap ertussenuit valt
zonder dat iemand het merkt.

Deze module maakt dat moment. Een oordeel zonder bijbehorende actie is per
definitie een openstaande beslissing, en daar hoort het systeem naar te vragen.

`scorebord.py` doet dit één schakel eerder: dat meet of de tussencheck de
analyse voorspelt, en benoemt zuiver dat de gemiste kans daar onmeetbaar is —
je doet die analyse immers nooit. Eén schakel verderop is diezelfde fout wél
meetbaar: de analyse zegt KOOP, je koopt niet, en de these houdt stand of niet.
Dat is te zien, en dus te leren.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from . import db

log = logging.getLogger(__name__)

# Oordelen waarbij een daad verwacht wordt. Die staan bovenaan in de lijst,
# omdat daar het gat zit tussen wat je concludeert en wat je doet.
VRAAGT_BESLISSING = {"VERDIEPEN", "KOOP", "STERKE KOOP"}

# Een afgeronde volledige analyse hoort altijd in het overzicht, ook bij HOLD of
# PASS. Niet omdat er dan iets moet gebeuren, maar omdat je later wilt kunnen
# nakijken of je HOLD-oordelen klopten. Zonder die regels meet je alleen je
# koopbeslissingen en blijft negentien twintigste van je werk onzichtbaar.
#
# Een tussencheck telt alleen mee bij VERDIEPEN: TWIJFEL en OVERSLAAN zijn een
# voorselectie waarbij niets doen de conclusie zélf is.
def _hoort_erin(soort: str | None, oordeel: str | None) -> bool:
    return (soort or "") == "analyse" or (oordeel or "").upper() in VRAAGT_BESLISSING


# Vanaf wanneer niets doen "stil gebleven" heet in plaats van "nog vers". Dit is
# uitsluitend een oordeelsdrempel: een besluit is meteen zichtbaar zodra de
# analyse af is (dat was Janco's expliciete wens), maar je pas op verzuim wijzen
# als er echt tijd overheen is gegaan.
STILTE_DAGEN = 14

KEUZES = ("gekocht", "bewust_niet", "uitgesteld")


def _dagen_sinds(datum: str | None) -> int | None:
    if not datum:
        return None
    try:
        return (date.today() - datetime.fromisoformat(datum[:10]).date()).days
    except ValueError:
        return None


def synchroniseer(rijen: list[dict]) -> int:
    """
    Zorg dat er voor elk oordeel dat om een beslissing vraagt een regel bestaat.

    Draait op de verrijkte dashboardrijen, waar `oordelen.verrijk()` het rapport
    al aan heeft gekoppeld. Die koppeling hier niet nabouwen: een tweede zoekweg
    naar hetzelfde rapport gaat gegarandeerd uit de pas lopen.

    Idempotent — bestaande regels blijven ongemoeid, inclusief hun keuze.
    """
    nieuw = 0
    for rij in rijen:
        oordeel = rij.get("oordeel") or {}
        soort_oordeel = (oordeel.get("oordeel") or "").upper()
        if not _hoort_erin(oordeel.get("soort"), soort_oordeel):
            continue
        try:
            db.besluit_vastleggen(
                ticker=rij["ticker"],
                aanleiding=oordeel.get("soort") or "onbekend",
                oordeel=soort_oordeel,
                datum_oordeel=oordeel.get("datum") or "",
                koers_toen=oordeel.get("koers_toen"),
                valuta=oordeel.get("valuta_toen") or rij.get("currency"),
            )
            nieuw += 1
        except Exception:
            log.exception("Besluit vastleggen mislukt voor %s", rij.get("ticker"))
    return nieuw


def openstaand(rijen: list[dict] | None = None) -> list[dict]:
    """
    De oordelen waar nog niets mee gedaan is en die oud genoeg zijn.

    Een ticker die je bezit telt niet mee: dan is er wél gehandeld, ook als er
    nooit een keuze is vastgelegd.
    """
    bezit = {b["ticker"] for b in db.bezit_lijst()}
    per_ticker = {r["ticker"]: r for r in (rijen or [])}

    uit = []
    for b in db.besluiten_lijst(alleen_open=True):
        if b["ticker"] in bezit:
            continue
        dagen = _dagen_sinds(b.get("datum_oordeel"))
        if dagen is None:
            continue
        rij = per_ticker.get(b["ticker"], {})
        uit.append({
            **b,
            "dagen_open": dagen,
            "vraagt_daad": (b.get("oordeel") or "").upper() in VRAAGT_BESLISSING,
            "naam": rij.get("name"),
            "koers_nu": rij.get("price"),
            "signaal_nu": rij.get("signal"),
            "korting_nu": rij.get("margin_of_safety"),
        })
    # Waar een daad verwacht wordt eerst, daarbinnen het langst liggende bovenaan.
    return sorted(uit, key=lambda b: (b["vraagt_daad"], b["dagen_open"]), reverse=True)


def actiekloof(rijen: list[dict] | None = None) -> dict:
    """
    Hoe vaak leidde een oordeel tot een daad?

    Dit is de maat die het Accenture-geval zichtbaar maakt. Niet "had ik gelijk"
    — dat weet je pas veel later — maar "doe ik iets met mijn eigen conclusies".
    Die vraag is vandaag al te beantwoorden en is daarmee het goedkoopste
    leersignaal dat er is.
    """
    bezit = {b["ticker"] for b in db.bezit_lijst()}
    alle = db.besluiten_lijst()

    per_oordeel: dict[str, dict] = {}
    for b in alle:
        vak = per_oordeel.setdefault(b["oordeel"] or "onbekend", {
            "totaal": 0, "gehandeld": 0, "bewust_niet": 0, "stil": 0,
            "uitgesteld": 0, "vers": 0,
        })
        vak["totaal"] += 1
        if b["ticker"] in bezit or b.get("keuze") == "gekocht":
            vak["gehandeld"] += 1
        elif b.get("keuze") == "bewust_niet":
            vak["bewust_niet"] += 1
        elif b.get("keuze") == "uitgesteld":
            vak["uitgesteld"] += 1
        elif (_dagen_sinds(b.get("datum_oordeel")) or 0) < STILTE_DAGEN:
            # Een oordeel van vorige week is nog geen verzuim. Zonder dit
            # onderscheid telt alles wat net af is meteen als stilte, en dan
            # meet de kloof deels je eigen doorlooptijd in plaats van je gedrag.
            vak["vers"] += 1
        else:
            vak["stil"] += 1

    return {
        "per_oordeel": per_oordeel,
        "open": len(openstaand(rijen)),
        "totaal": len(alle),
    }


def sinds_het_oordeel(besluit: dict, rij: dict | None) -> dict:
    """
    Wat er sinds het oordeel is gebeurd — these eerst, koers erbij.

    De volgorde is met opzet. Op een paar weken zegt een koers vrijwel niets over
    de kwaliteit van een oordeel, en hem bovenaan zetten leert je koersen
    achternalopen in plaats van bedrijven beoordelen. De vraag die telt is of de
    reden van je twijfel terug te zien is in de cijfers.
    """
    snapshot = besluit.get("these_snapshot") or {}
    rij = rij or {}

    def _verschil(sleutel):
        toen, nu = snapshot.get(sleutel), rij.get(sleutel)
        if toen is None or nu is None:
            return None
        try:
            return round(float(nu) - float(toen), 2)
        except (TypeError, ValueError):
            return None

    dagen = _dagen_sinds(besluit.get("datum_oordeel")) or 0
    koers_toen = besluit.get("koers_toen")
    koers_nu = rij.get("price")
    rendement = None
    if koers_toen and koers_nu:
        try:
            rendement = round(100 * (float(koers_nu) / float(koers_toen) - 1), 1)
        except (TypeError, ZeroDivisionError, ValueError):
            rendement = None

    # De momentopname wordt gemaakt op het moment dat je de beslissing erkent,
    # niet toen het oordeel ontstond. Voor een net afgesloten besluit vergelijk je
    # dus vandaag met vandaag, en dan is "+0" geen bevinding maar een artefact.
    snapshot_dagen = _dagen_sinds(snapshot.get("op")) if snapshot.get("op") else None

    return {
        "dagen": dagen,
        "snapshot_dagen": snapshot_dagen,
        "these": {
            "kwaliteit": _verschil("quality_score"),
            "kasstroom": _verschil("normalized_fcf_m"),
            "roic": _verschil("roic_mediaan"),
            "brutomarge": _verschil("brutomarge_trend_pp"),
            "meetbaar": bool(snapshot),
        },
        "koers": {
            "toen": koers_toen,
            "nu": koers_nu,
            "rendement_pct": rendement,
            # Onder het jaar is dit context, geen uitslag. `scorebord.py` noemt
            # het een rookmelder en geen thermometer, en dat geldt hier ook.
            "betekenisvol": dagen >= 365,
        },
    }
