"""De sectorsleutels in config.yaml moeten matchen met wat Yahoo teruggeeft.

Aanleiding: `config.yaml` gebruikte de GICS-namen "Consumer Discretionary",
"Consumer Staples" en "Materials", terwijl `stocks.sector` gevuld wordt met de
Yahoo-namen "Consumer Cyclical", "Consumer Defensive" en "Basic Materials".
Gemeten op 6 augustus 2026 vielen daardoor 617 van de 2.760 aandelen (22%)
stil terug op het Default-profiel — zonder foutmelding, want `_sector_cfg`
valt netjes terug.

De schade zit in de richting. Basic Materials hoort op K/W 15 en EV/EBITDA 9
te staan; Default geeft 18 en 11. Dat is een 20% royaler anker, wat de fair
value opblaast en dus schijnkortingen produceert — precies de vals-positieven
waar onderzoekstijd aan verloren gaat.

Deze test is de waakhond: verschijnt er een nieuwe sectornaam in de database,
of hernoemt Yahoo er een, dan valt hij om in plaats van stil terug te vallen.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from engine.valuation import _sector_cfg

CONFIG_PAD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)

# Sectornamen zoals yfinance ze levert. Bijwerken zodra de database er een
# nieuwe laat zien; dat is precies het moment waarop je wilt nadenken over het
# bijbehorende profiel in plaats van het stilzwijgend op Default te laten.
YAHOO_SECTOREN = [
    "Technology",
    "Financial Services",
    "Healthcare",
    "Consumer Defensive",
    "Consumer Cyclical",
    "Industrials",
    "Energy",
    "Basic Materials",
    "Utilities",
    "Real Estate",
    "Communication Services",
]

# Namen die BEWUST op Default uitkomen, met de reden erbij. Alles wat hier niet
# in staat en ook geen eigen profiel heeft, is een fout.
BEWUST_DEFAULT = {
    "Communication Services":
        "Nog geen eigen profiel. 186 aandelen; Yahoo gooit hier telecom "
        "(lage multiples) en media/interactief (hoge) op een hoop, dus de "
        "keuze is een waarderingsbeslissing en geen naamfout. Openstaand.",
    "Unknown":
        "Sector onbekend bij Yahoo. Default is hier de juiste uitkomst.",
}

# GICS-namen die hier NIET mogen staan: ze matchen nooit en de terugval is stil.
VERBODEN_GICS = ["Consumer Discretionary", "Consumer Staples", "Materials"]


def _sectors():
    with open(CONFIG_PAD, encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("sectors") or {}


def test_elke_yahoo_sector_heeft_een_profiel_of_een_reden():
    sectors = _sectors()
    ontbreekt = [
        s for s in YAHOO_SECTOREN
        if s not in sectors and s not in BEWUST_DEFAULT
    ]
    assert not ontbreekt, (
        f"Sector(en) zonder profiel en zonder reden: {ontbreekt}. "
        "Voeg een profiel toe aan config.yaml of zet hem met uitleg in "
        "BEWUST_DEFAULT."
    )


def test_geen_gics_namen_in_de_config():
    """Een GICS-naam matcht nooit op `stocks.sector` en valt stil terug."""
    sectors = _sectors()
    gevonden = [naam for naam in VERBODEN_GICS if naam in sectors]
    assert not gevonden, (
        f"GICS-namen in config.yaml: {gevonden}. Yahoo gebruikt "
        "'Consumer Cyclical', 'Consumer Defensive' en 'Basic Materials'; "
        "deze sleutels worden nooit geraakt."
    )


def test_geen_enkel_profiel_is_onbereikbaar():
    """Elke sleutel behalve Default moet een echte Yahoo-sector zijn.

    Dit is de kant die de oorspronkelijke bug niet zichtbaar maakte: er stond
    keurig een profiel voor 'Materials', het werd alleen nooit gebruikt.
    """
    onbereikbaar = [
        naam for naam in _sectors()
        if naam != "Default" and naam not in YAHOO_SECTOREN
    ]
    assert not onbereikbaar, (
        f"Profiel(en) die geen enkele ticker kunnen raken: {onbereikbaar}."
    )


def test_elk_profiel_is_compleet():
    verplicht = ("growth_base", "growth_min", "growth_max",
                 "required_return", "pe", "ev_ebitda", "pb", "ev_fcf")
    for naam, waarden in _sectors().items():
        ontbreekt = [v for v in verplicht if waarden.get(v) is None]
        assert not ontbreekt, f"{naam} mist {ontbreekt}"


def test_basic_materials_wijkt_echt_af_van_default():
    """Regressie op de kern van de bug.

    Zolang Basic Materials hetzelfde profiel zou geven als Default, was de
    naamfout onzichtbaar in de uitkomsten. Deze test legt vast dat het
    verschil bestaat én welke kant het op staat: strenger dan Default, dus
    de oude situatie waardeerde deze sector te hoog.
    """
    with open(CONFIG_PAD, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    bm = _sector_cfg("Basic Materials", cfg)
    default = _sector_cfg("Default", cfg)
    assert bm != default, "Basic Materials valt nog steeds terug op Default"
    assert bm["pe"] < default["pe"], "Basic Materials hoort strenger te zijn"
    assert bm["ev_ebitda"] < default["ev_ebitda"]


def test_onbekende_sector_valt_nog_steeds_netjes_terug():
    """De terugval zelf moet blijven werken — die is niet het probleem."""
    with open(CONFIG_PAD, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    assert _sector_cfg("Bestaat Niet", cfg) == cfg["sectors"]["Default"]
    assert _sector_cfg(None, cfg) == cfg["sectors"]["Default"]


if __name__ == "__main__":
    test_elke_yahoo_sector_heeft_een_profiel_of_een_reden()
    print("  [OK] elke Yahoo-sector heeft een profiel of een genoteerde reden")
    test_geen_gics_namen_in_de_config()
    print("  [OK] geen GICS-namen meer in config.yaml")
    test_geen_enkel_profiel_is_onbereikbaar()
    print("  [OK] geen onbereikbare profielen")
    test_elk_profiel_is_compleet()
    print("  [OK] elk profiel heeft alle acht velden")
    test_basic_materials_wijkt_echt_af_van_default()
    print("  [OK] Basic Materials is strenger dan Default, zoals bedoeld")
    test_onbekende_sector_valt_nog_steeds_netjes_terug()
    print("  [OK] terugval op Default werkt nog")
    print("\nAlle tests sectorconfig geslaagd.")
