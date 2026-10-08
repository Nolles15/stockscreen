"""Groeiprofiel: bewaakt de toetsen, de datapoorten en de volgorde van het oordeel."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import groei_profiel as gp  # noqa: E402
from engine import screener  # noqa: E402


def _jaren(**reeksen):
    """Bouwt annual-rijen uit parallelle lijsten, oudste jaar eerst (2022…)."""
    n = len(next(iter(reeksen.values())))
    return [{"fiscal_year": 2022 + i, "period_type": "annual",
             **{k: v[i] for k, v in reeksen.items()}} for i in range(n)]


def _gezond(**extra):
    """Snelle groeier (~25%/jr) zonder rode vlaggen."""
    basis = dict(
        revenue=[100, 125, 156, 195], total_assets=[200, 240, 290, 340],
        shares_outstanding=[10, 10, 10.1, 10.1], gross_profit=[60, 76, 96, 120],
        net_income=[5, 10, 16, 25], operating_cf=[8, 14, 22, 32],
    )
    basis.update(extra)
    return _jaren(**basis)


GOED_F = {"score": 8, "known_criteria": 9, "criteria": {}}


def _t(p, code):
    return next(t for t in p["toetsen"] if t["code"] == code)


def test_groen_profiel():
    p = gp.bouw_profiel(_gezond(), piotroski=GOED_F)
    assert p["groeier"] and p["niveau"] == "groen", (p["niveau"], p["kop"], p["score"])
    assert p["score"] >= gp.GROEN_SCORE
    print("  [OK] gezonde groeier is groen")


def test_t1_verwatering():
    rood = gp.bouw_profiel(_gezond(shares_outstanding=[10, 11, 12.1, 13.3]), piotroski=GOED_F)
    assert _t(rood, "T1")["uitkomst"] == "rood" and rood["niveau"] == "rood"
    assert "aandelen" in rood["kop"]
    # Terugkoop is geen verwatering.
    assert _t(gp.bouw_profiel(_gezond(shares_outstanding=[10, 9.5, 9, 8.5]),
                              piotroski=GOED_F), "T1")["uitkomst"] == "groen"
    # Sprong ≥ factor 2: klassewissel, onbekend en nooit rood.
    sprong = gp.bouw_profiel(_gezond(shares_outstanding=[10, 10, 25, 25]), piotroski=GOED_F)
    assert _t(sprong, "T1")["uitkomst"] == "onbekend"
    # Aandelenklasse: onbekend.
    klasse = gp.bouw_profiel(_gezond(shares_outstanding=[10, 11, 12.1, 13.3]),
                             piotroski=GOED_F, dubbel_soort="aandelenklasse")
    assert _t(klasse, "T1")["uitkomst"] == "onbekend" and klasse["niveau"] != "rood"
    # Te weinig aandelen-jaren
    leeg = gp.bouw_profiel(_gezond(shares_outstanding=[None, None, None, 10]), piotroski=GOED_F)
    assert _t(leeg, "T1")["uitkomst"] == "onbekend"
    print("  [OK] T1 rood/groen/onbekend")


def test_t2_balansgroei():
    p = gp.bouw_profiel(_gezond(total_assets=[200, 300, 450, 700]), piotroski=GOED_F)
    assert _t(p, "T2")["uitkomst"] == "rood" and p["niveau"] == "rood"
    # Gat groot maar activa zelf < 20%/jr: geen rood.
    traag = gp.bouw_profiel(_gezond(revenue=[100, 101, 102, 103],
                                    total_assets=[200, 230, 265, 305]), piotroski=GOED_F)
    assert _t(traag, "T2")["uitkomst"] == "geel"
    zonder = gp.bouw_profiel(_gezond(total_assets=[None, 240, 290, 340]), piotroski=GOED_F)
    assert _t(zonder, "T2")["uitkomst"] == "onbekend"
    print("  [OK] T2 vereist gat én hoge activagroei voor rood")


def test_t3_fscore():
    assert _t(gp.bouw_profiel(_gezond(), piotroski={"score": 3, "known_criteria": 9}), "T3")["uitkomst"] == "rood"
    assert gp.bouw_profiel(_gezond(), piotroski={"score": 3, "known_criteria": 9})["niveau"] == "rood"
    assert _t(gp.bouw_profiel(_gezond(), piotroski={"score": 5, "known_criteria": 9}), "T3")["uitkomst"] == "geel"
    assert _t(gp.bouw_profiel(_gezond(), piotroski={"score": 2, "known_criteria": 6}), "T3")["uitkomst"] == "onbekend"
    print("  [OK] T3 onbekend bij < 7 criteria")


def test_t4_brutowinst():
    mager = gp.bouw_profiel(_gezond(gross_profit=[10, 12, 15, 19]), piotroski=GOED_F)
    assert _t(mager, "T4")["uitkomst"] == "rood"
    # Erosie: brutomarge 60% → 40% terwijl het niveau hoog blijft.
    erosie = gp.bouw_profiel(_gezond(gross_profit=[60, 70, 75, 78]), piotroski=GOED_F)
    assert _t(erosie, "T4")["uitkomst"] == "rood", _t(erosie, "T4")
    geen = gp.bouw_profiel(_gezond(gross_profit=[None] * 4), piotroski=GOED_F)
    assert _t(geen, "T4")["uitkomst"] == "onbekend"
    print("  [OK] T4 niveau en erosie")


def test_t5_t6_zijn_zacht():
    # Rood op accruals alleen: nooit rood eindoordeel, wel geel.
    acc = gp.bouw_profiel(_gezond(net_income=[50, 90, 140, 200], operating_cf=[5, 8, 10, 12]),
                          piotroski=GOED_F)
    assert _t(acc, "T5")["uitkomst"] == "rood" and acc["niveau"] == "geel", acc["niveau"]
    assert "zwak punt" in acc["kop"]
    wild = gp.bouw_profiel(_gezond(revenue=[100, 180, 190, 340], gross_profit=[60, 108, 114, 204]), piotroski=GOED_F)
    assert _t(wild, "T6")["uitkomst"] == "rood"
    assert wild["niveau"] != "rood"
    print("  [OK] T5/T6 geven hooguit geel")


def test_t7_rd_nooit_rood_en_ontbreken_is_onbekend():
    geen = gp.bouw_profiel(_gezond(), piotroski=GOED_F)
    assert _t(geen, "T7")["uitkomst"] == "onbekend"
    nul = gp.bouw_profiel(_gezond(rd_expense=[0, 0, 0, 0]), piotroski=GOED_F)
    assert _t(nul, "T7")["uitkomst"] == "onbekend"
    laag = gp.bouw_profiel(_gezond(rd_expense=[1, 1, 1, 2]), piotroski=GOED_F)
    assert _t(laag, "T7")["uitkomst"] == "geel"
    hoog = gp.bouw_profiel(_gezond(rd_expense=[10, 13, 16, 20]), piotroski=GOED_F)
    assert _t(hoog, "T7")["uitkomst"] == "groen"
    # Sectorrelatief: mediaan 15% → 10% is geel.
    sec = gp.bouw_profiel(_gezond(rd_expense=[10, 13, 16, 20]), sector_mediaan_rd=15.0,
                          piotroski=GOED_F)
    assert _t(sec, "T7")["uitkomst"] == "geel"
    # Ontbrekende R&D verlaagt de score nooit.
    assert geen["score"] >= laag["score"]
    print("  [OK] T7 ontbreekt = onbekend, nooit rood")


def test_poorten():
    assert gp.bouw_profiel(_gezond(), piotroski=GOED_F, data_status="bad")["niveau"] == "grijs"
    assert gp.bouw_profiel(_gezond(), piotroski=GOED_F, data_status="missing")["niveau"] == "grijs"
    kort = gp.bouw_profiel(_gezond()[:2], piotroski=GOED_F)
    assert kort["niveau"] == "grijs" and "Te weinig" in kort["kop"]
    # Omzetbreuk (Adyen): grijs, geen rood.
    adyen = _jaren(revenue=[8936, 1863, 2226, 2647], total_assets=[1, 1, 1, 1])
    p = gp.bouw_profiel(adyen, piotroski=GOED_F)
    assert p["niveau"] == "grijs" and "Definitiewissel" in p["kop"], p["kop"]
    assert p["omzet_cagr"] is None and p["omzetbreuk"]
    # Omzetgebonden toetsen en omzet per aandeel blijven leeg bij een breuk.
    assert _t(p, "T2")["uitkomst"] == "onbekend" and _t(p, "T6")["uitkomst"] == "onbekend"
    assert p["omzet_per_aandeel_cagr"] is None
    print("  [OK] poorten: data_status, te weinig rijen, omzetbreuk")


def test_financials_zijn_grijs():
    p = gp.bouw_profiel(_gezond(), sector="Financial Services", piotroski=GOED_F)
    assert p["niveau"] == "grijs" and "banken" in p["kop"]
    assert len(p["toetsen"]) == 7
    print("  [OK] Financial Services: grijs, toetsen wel berekend")


def test_niet_groeier_is_grijs_met_toetsen():
    p = gp.bouw_profiel(_jaren(revenue=[100, 102, 104, 106], total_assets=[200, 205, 210, 215],
                               gross_profit=[50, 51, 52, 53]), piotroski=GOED_F)
    assert p["niveau"] == "grijs" and not p["groeier"] and "Geen snelle groeier" in p["kop"]
    assert len(p["toetsen"]) == 7 and p["score"] is not None
    print("  [OK] niet-groeier: grijs, toetsen wel berekend")


def test_volgorde_hard_rood_wint():
    p = gp.bouw_profiel(_gezond(shares_outstanding=[10, 11, 12.1, 13.3],
                                total_assets=[200, 300, 450, 700]),
                        piotroski={"score": 2, "known_criteria": 9})
    assert p["niveau"] == "rood" and "aandelen" in p["kop"]  # T1 vóór T2 vóór T3
    print("  [OK] eerste harde rode toets bepaalt de kop")


def test_groen_eist_bekende_toetsen():
    # Alleen T1, T2 en T6 bekend: minder dan vier, dus geen groen.
    rijen = _jaren(revenue=[100, 125, 156, 195], total_assets=[200, 240, 290, 340],
                   shares_outstanding=[10, 10, 10, 10])
    p = gp.bouw_profiel(rijen, piotroski={"score": 5, "known_criteria": 3})
    bekend = sum(1 for t in p["toetsen"] if t["uitkomst"] != "onbekend")
    assert bekend < gp.GROEN_MIN_BEKEND and p["niveau"] == "geel"
    print("  [OK] groen eist minstens vier bekende toetsen")


def test_ttm_rij_telt_niet_mee():
    rijen = _gezond() + [{"fiscal_year": 0, "period_type": "ttm", "revenue": 1}]
    assert gp.bouw_profiel(rijen, piotroski=GOED_F)["venster"] == "2022–2025"
    ook = _gezond() + [{"fiscal_year": 0, "revenue": 1}]
    assert gp.bouw_profiel(ook, piotroski=GOED_F)["venster"] == "2022–2025"
    print("  [OK] TTM-rij uitgesloten")


def test_drempels_override():
    basis = gp.bouw_profiel(_gezond(shares_outstanding=[10, 10.4, 10.8, 11.2]), piotroski=GOED_F)
    strak = gp.bouw_profiel(_gezond(shares_outstanding=[10, 10.4, 10.8, 11.2]), piotroski=GOED_F,
                            drempels={"t1_rood": 3.0})
    assert _t(basis, "T1")["uitkomst"] == "geel" and _t(strak, "T1")["uitkomst"] == "rood"
    assert strak["drempels"]["t1_rood"] == 3.0 and basis["drempels"]["t1_rood"] == gp.T1_ROOD
    print("  [OK] drempels-override werkt en reist mee")


def test_compact_is_json_zonder_series():
    c = gp.compact(gp.bouw_profiel(_gezond(), piotroski=GOED_F))
    assert "series" not in c and "redenen" not in c
    assert len(json.dumps(c)) < 600
    json.dumps(gp.bouw_profiel(_gezond(), piotroski=GOED_F))
    print("  [OK] compact() klein en json-baar")


def test_zelfde_cagr_als_screener():
    for rijen in (_gezond(), _jaren(revenue=[100, 80, 90, 70]),
                  _jaren(revenue=[100, 120, 140, 150, 170]), _gezond()[:2],
                  _jaren(revenue=[8936, 1863, 2226, 2647])):
        assert gp.omzet_cagr(rijen) == screener._calc_revenue_cagr(rijen), rijen
    print("  [OK] omzet_cagr == screener._calc_revenue_cagr")


def test_piotroski_zonder_invoer_wordt_berekend():
    p = gp.bouw_profiel(_gezond())
    assert _t(p, "T3")["uitkomst"] in ("rood", "geel", "groen", "onbekend")
    print("  [OK] F-score valt terug op berekening uit de jaarrijen")


if __name__ == "__main__":
    print("Groeiprofiel")
    for naam, f in sorted(globals().items()):
        if naam.startswith("test_") and callable(f):
            f()
    print("Alles groen.")
