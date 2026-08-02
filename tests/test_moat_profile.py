"""Moat-profiel: bewaakt de rekenregels en de drempels.

De drempels zijn gekalibreerd op 16 aandelen waarvan de volledige analyse bekend
is. Twee eisen mogen nooit sneuvelen:
  - de vijf bekende valse positieven (goedkoop maar zwak) worden nooit groen;
  - de drie KOOP-oordelen worden nooit rood.
Die worden hier nagebootst met de werkelijke ROIC-profielen van die aandelen.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.moat_profile import (  # noqa: E402
    bouw_profiel, cyclustest, marge_reeks, roic_reeks,
)


def _jaren(**reeksen):
    """Bouwt annual-rijen uit parallelle lijsten, oudste jaar eerst."""
    n = len(next(iter(reeksen.values())))
    return [{"fiscal_year": 2021 + i, **{k: v[i] for k, v in reeksen.items()}}
            for i in range(n)]


def test_roic():
    # EBIT 100, kapitaal 500 -> 100 * 0,75 / 500 = 15%
    rijen = _jaren(ebit=[100, 120], total_equity=[400, 400], total_debt=[100, 100])
    assert roic_reeks(rijen) == [(2021, 15.0), (2022, 18.0)]

    # Negatief eigen vermogen maakt ROIC betekenisloos: overslaan, niet raden.
    scheef = _jaren(ebit=[100], total_equity=[-200], total_debt=[50])
    assert roic_reeks(scheef) == []
    print("  [OK] ROIC per jaar, negatief vermogen overgeslagen")


def test_marges():
    rijen = _jaren(gross_profit=[60, 55], revenue=[100, 100], ebit=[20, 15])
    assert marge_reeks(rijen, "gross_profit") == [(2021, 60.0), (2022, 55.0)]
    assert marge_reeks(rijen, "ebit") == [(2021, 20.0), (2022, 15.0)]
    # Omzet nul mag geen deling door nul geven
    assert marge_reeks(_jaren(gross_profit=[10], revenue=[0]), "gross_profit") == []
    print("  [OK] margereeksen, omzet nul afgevangen")


def test_cyclustest():
    koersen = [{"date": "2020-01-01", "close": 100.0},
               {"date": "2020-06-01", "close": 40.0},   # -60% vanaf de top
               {"date": "2024-01-01", "close": 90.0}]
    c = cyclustest(koersen)
    assert c["beschikbaar"] and c["diepste_terugval_pct"] == -60.0
    assert c["onder_top_pct"] == -10.0        # 90 tegen alletijdenrecord 100
    assert not cyclustest([])["beschikbaar"]
    print("  [OK] zwaarste terugval en afstand tot de top")


def test_groen_vereist_hoog_en_standvastig():
    # Wolters Kluwer-profiel: mediaan ~19%, dieptepunt op 88% daarvan.
    wkl = _jaren(ebit=[190, 200, 210, 205, 215],
                 total_equity=[750, 750, 750, 750, 750],
                 total_debt=[250, 250, 250, 250, 250],
                 revenue=[100]*5, gross_profit=[72]*5)
    assert bouw_profiel(wkl)["niveau"] == "groen"

    # Zelfde standvastigheid, maar half zo hoog rendement -> geen groen.
    mager = _jaren(ebit=[95, 100, 105, 102, 107],
                   total_equity=[750]*5, total_debt=[250]*5,
                   revenue=[100]*5, gross_profit=[72]*5)
    assert bouw_profiel(mager)["niveau"] == "geel"
    print("  [OK] groen eist hoog rendement EN standvastigheid")


def test_instortend_rendement_is_rood():
    # Sdiptech-profiel: mediaan ~8%, slechtste jaar minder dan de helft daarvan.
    sdip = _jaren(ebit=[110, 115, 50, 105, 100],
                  total_equity=[750]*5, total_debt=[250]*5,
                  revenue=[100]*5, gross_profit=[60]*5)
    p = bouw_profiel(sdip)
    assert p["niveau"] == "rood" and "stort in" in p["kop"]
    print("  [OK] rendement dat instort bij tegenwind wordt rood")


def test_margeerosie_is_rood():
    # Hoog en stabiel rendement, maar de brutomarge zakt 8 procentpunt weg.
    erosie = _jaren(ebit=[210, 210, 210, 210, 210],
                    total_equity=[750]*5, total_debt=[250]*5,
                    revenue=[100]*5, gross_profit=[60, 58, 56, 54, 52])
    p = bouw_profiel(erosie)
    assert p["niveau"] == "rood" and "brutomarge" in p["kop"]
    print("  [OK] wegbrokkelende brutomarge wordt rood ondanks hoog rendement")


def test_herstel_is_geen_instorting():
    # Huuuge-profiel: slechtste jaar is het eerste (11,6%), daarna loopt het op
    # naar 48%. De oude regel las dat als "stort in bij tegenwind" omdat hij
    # alleen naar het dieptepunt ten opzichte van de mediaan keek.
    herstel = _jaren(ebit=[52, 176, 145, 215],
                     total_equity=[250]*4, total_debt=[85]*4,
                     revenue=[100]*4, gross_profit=[70, 71, 73, 76])
    p = bouw_profiel(herstel)
    assert p["niveau"] == "geel", p
    assert "opgeklommen" in p["kop"], p["kop"]

    # Spiegelbeeld: dezelfde waarden aflopend is wél een instorting.
    instorting = _jaren(ebit=[215, 145, 176, 52],
                        total_equity=[250]*4, total_debt=[85]*4,
                        revenue=[100]*4, gross_profit=[76, 73, 71, 70])
    assert bouw_profiel(instorting)["niveau"] == "rood"
    print("  [OK] opklimmen vanaf een zwakke start is geen instorting")


def test_herstel_wordt_niet_zomaar_groen():
    # Een reeks die net op niveau is gekomen heeft nog geen slecht jaar
    # doorstaan; duurzaamheid is dan niet aangetoond, dus geen groen.
    herstel = _jaren(ebit=[40, 210, 205, 220],
                     total_equity=[750]*4, total_debt=[250]*4,
                     revenue=[100]*4, gross_profit=[70]*4)
    assert bouw_profiel(herstel)["niveau"] == "geel"
    print("  [OK] herstel geeft geel, niet groen")


def test_te_weinig_jaren():
    kort = _jaren(ebit=[200, 210], total_equity=[750, 750], total_debt=[250, 250])
    p = bouw_profiel(kort)
    assert p["niveau"] == "grijs"
    print("  [OK] onder de drie jaar geen oordeel")


if __name__ == "__main__":
    test_roic()
    test_marges()
    test_cyclustest()
    test_groen_vereist_hoog_en_standvastig()
    test_instortend_rendement_is_rood()
    test_margeerosie_is_rood()
    test_herstel_is_geen_instorting()
    test_herstel_wordt_niet_zomaar_groen()
    test_te_weinig_jaren()
    print("\nAlle tests moat-profiel geslaagd.")
