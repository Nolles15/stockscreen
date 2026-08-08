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


def test_roic_trekt_overtollige_kas_af():
    # A8 (besluit Janco 2026-08-08): zelfde definitie als quality_score.py.
    # EBIT 100, eigen vermogen 400, schuld 100, nettokas 300:
    # kapitaal = 400 + 100 - 300 = 200 -> 100 * 0,75 / 200 = 37,5%
    spaarpot = _jaren(ebit=[100], total_equity=[400], total_debt=[100],
                      net_cash=[300])
    assert roic_reeks(spaarpot) == [(2021, 37.5)]

    # Nettoschuld (negatieve nettokas) verandert niets: alleen óvertollige
    # kas gaat eraf, schuld zit al in het kapitaal.
    schuldig = _jaren(ebit=[100], total_equity=[400], total_debt=[100],
                      net_cash=[-50])
    assert roic_reeks(schuldig) == [(2021, 15.0)]
    print("  [OK] overtollige kas telt niet mee als geinvesteerd kapitaal")


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


def _reeks(vanaf: str, dagen: int, koers=100.0):
    """Dagelijkse koersen vanaf een datum, vlak verloop."""
    from datetime import date, timedelta
    d0 = date.fromisoformat(vanaf)
    return [{"date": (d0 + timedelta(days=i)).isoformat(), "close": koers}
            for i in range(dagen)]


def test_korte_reeks_is_beschikbaar_maar_niet_betrouwbaar():
    """De fout die drie tussenchecks haalde.

    Tien dagen koershistorie leverde keurige getallen op — diepste terugval en
    afstand tot de top — die allebei alleen over die tien dagen gingen. Het
    profiel meldde dan "staat weer op recordhoogte" terwijl het aandeel
    tientallen procenten onder zijn werkelijke top stond.
    """
    c = cyclustest(_reeks("2026-07-27", 10))
    assert c["beschikbaar"], "de cijfers blijven beschikbaar"
    assert not c["betrouwbaar"], "maar mogen geen conclusie dragen"
    assert c["jaren"] < 0.1


def test_lange_reeks_met_genoeg_punten_is_betrouwbaar():
    c = cyclustest(_reeks("2023-01-01", 800))
    assert c["beschikbaar"] and c["betrouwbaar"]
    assert c["jaren"] >= 2.0


def test_lange_periode_met_te_weinig_punten_is_niet_betrouwbaar():
    """Drie koersen over vijf jaar is geen cyclusbeeld."""
    koersen = [{"date": "2020-01-01", "close": 100.0},
               {"date": "2022-06-01", "close": 40.0},
               {"date": "2025-01-01", "close": 90.0}]
    c = cyclustest(koersen)
    assert c["beschikbaar"] and not c["betrouwbaar"]


def test_jaren_rekent_op_dagen_niet_op_kalenderjaren():
    """31 december tot 1 januari was eerder 'één jaar'."""
    c = cyclustest([{"date": "2025-12-31", "close": 100.0},
                    {"date": "2026-01-01", "close": 101.0}])
    assert c["jaren"] == 0.0, f"kreeg {c['jaren']}"
    assert not c["betrouwbaar"]


def test_dunne_marge_krijgt_een_waarschuwing_bij_groen():
    """Ework Group, 7 augustus 2026.

    24% mediaan rendement op kapitaal en dus groen, terwijl de operationele
    marge 0,8% is: een bemiddelaar die 13,7 mrd omzet langs 250 mln eigen
    vermogen schuift. Het rendement is hoog omdat de noemer klein is.
    """
    # ebit/revenue ≈ 0,9%; eigen vermogen klein genoeg voor een hoge ROIC
    p = bouw_profiel(_jaren(
        revenue=[16000.0, 17200.0, 15800.0, 13700.0],
        gross_profit=[440.0, 470.0, 430.0, 390.0],
        ebit=[145.0, 155.0, 140.0, 120.0],
        total_equity=[260.0, 280.0, 300.0, 250.0],
        total_debt=[0.0, 0.0, 0.0, 0.0],
    ))
    assert p["niveau"] == "groen", "de ROIC-test hoort hier nog steeds groen te geven"
    assert p["dunne_marge"] is True
    redenen = " ".join(p["redenen"])
    assert "LET OP" in redenen and "weinig kapitaal" in redenen, redenen


def test_gezonde_marge_krijgt_geen_waarschuwing():
    """De tegenproef: een groen bedrijf met een normale marge blijft schoon."""
    p = bouw_profiel(_jaren(
        revenue=[1000.0] * 4, gross_profit=[500.0] * 4, ebit=[200.0] * 4,
        total_equity=[700.0] * 4, total_debt=[300.0] * 4,
    ))
    assert p["niveau"] == "groen"
    assert p["dunne_marge"] is False
    assert "LET OP" not in " ".join(p["redenen"])


def test_waarschuwing_alleen_bij_groen():
    """Bij rood of geel zegt het oordeel zelf al genoeg; dan is de extra regel ruis."""
    p = bouw_profiel(_jaren(
        revenue=[16000.0, 17200.0, 15800.0, 13700.0],
        gross_profit=[440.0, 470.0, 430.0, 390.0],
        ebit=[145.0, 155.0, 140.0, 120.0],
        total_equity=[6000.0] * 4, total_debt=[2000.0] * 4,   # grote noemer -> lage ROIC
    ))
    assert p["niveau"] != "groen"
    assert p["dunne_marge"] is False


def test_de_reden_belooft_niets_bij_een_korte_reeks():
    """Geen 'recordhoogte' meer, maar een expliciete melding dat we het niet weten."""
    profiel = bouw_profiel(
        _jaren(revenue=[100.0] * 4, gross_profit=[50.0] * 4, ebit=[20.0] * 4,
               total_equity=[100.0] * 4, total_debt=[0.0] * 4),
        price_history=_reeks("2026-07-27", 10),
    )
    redenen = " ".join(profiel["redenen"])
    assert "recordhoogte" not in redenen
    assert "te kort" in redenen.lower(), redenen


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
    test_roic_trekt_overtollige_kas_af()
    test_marges()
    test_cyclustest()
    test_groen_vereist_hoog_en_standvastig()
    test_instortend_rendement_is_rood()
    test_margeerosie_is_rood()
    test_herstel_is_geen_instorting()
    test_herstel_wordt_niet_zomaar_groen()
    test_te_weinig_jaren()
    test_korte_reeks_is_beschikbaar_maar_niet_betrouwbaar()
    print("  [OK] korte koersreeks is beschikbaar maar niet betrouwbaar")
    test_lange_reeks_met_genoeg_punten_is_betrouwbaar()
    print("  [OK] lange reeks met genoeg punten is wel betrouwbaar")
    test_lange_periode_met_te_weinig_punten_is_niet_betrouwbaar()
    print("  [OK] drie koersen over vijf jaar telt niet als cyclusbeeld")
    test_jaren_rekent_op_dagen_niet_op_kalenderjaren()
    print("  [OK] jaren rekent op dagen, niet op kalenderjaren")
    test_de_reden_belooft_niets_bij_een_korte_reeks()
    print("  [OK] geen 'recordhoogte' bij een reeks van tien dagen")
    test_dunne_marge_krijgt_een_waarschuwing_bij_groen()
    print("  [OK] groen bij een operationele marge onder 5% krijgt LET OP")
    test_gezonde_marge_krijgt_geen_waarschuwing()
    print("  [OK] groen met een normale marge blijft schoon")
    test_waarschuwing_alleen_bij_groen()
    print("  [OK] geen waarschuwing bij rood of geel")
    print("\nAlle tests moat-profiel geslaagd.")
