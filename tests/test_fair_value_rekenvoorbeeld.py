"""De fair value, nagerekend op één voorbeeld met ronde getallen.

De overige tests bewaken randgevallen en incidenten; geen ervan zette de
formules zelf tegen een met de hand uitgerekende uitkomst. Dit bestand doet dat.
Elke verwachte waarde staat als som in het commentaar, zodat een wijziging in
`valuation.py` die de uitkomst verschuift hier zichtbaar en beredeneerd moet
worden aangepast.

Het voorbeeldbedrijf: 100 aandelen, genormaliseerde winst 2,00 per aandeel,
EBITDA 300, vrije kasstroom 160 (1,60 per aandeel), eigen vermogen 1.000
(10 per aandeel), nettoschuld 200. Sectorprofiel: K/W 18, EV/EBITDA 11, K/B 2,5,
EV/FCF 16, groei 2/4/6%, rendementseis 10%.

Draaien: python -m pytest tests/test_fair_value_rekenvoorbeeld.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.valuation import (  # noqa: E402
    combined_fair_value,
    graham_fair_value,
    multiples_fair_value,
    perpetuity_fair_value,
)

CFG = {
    "valuation": {
        "multiples_weight": 0.6, "dcf_weight": 0.4, "bond_yield": 5.0,
        "max_growth_rate": 8, "max_perpetuity_growth": 5, "default_required_return": 10,
    },
    "sectors": {"Default": {
        "growth_base": 4, "growth_min": 2, "growth_max": 6, "required_return": 10,
        "pe": 18, "ev_ebitda": 11, "pb": 2.5, "ev_fcf": 16,
    }},
}
NORM = {
    "normalized_eps": 2.0, "normalized_ebitda": 300.0, "normalized_fcf": 160.0,
    "normalized_oe_per_share": 1.6,
}


def _rijen(net_cash=-200.0):
    return [{"fiscal_year": 2025, "shares_outstanding": 100, "total_equity": 1000.0,
             "net_cash": net_cash, "net_income": 200.0}]


def _historie(pe):
    return [{"fiscal_year": j, "pe_ratio": pe, "ev_ebitda": None, "pb_ratio": None, "ev_fcf": None}
            for j in (2022, 2023, 2024, 2025)]


def test_de_vier_multiples_zonder_eigen_historie():
    m = multiples_fair_value(NORM, [], _rijen(), "Default", CFG)
    assert m["pe_fv"] == pytest.approx(36.0)          # 2,00 x 18
    assert m["ev_ebitda_fv"] == pytest.approx(31.0)   # (300 x 11 - 200) / 100
    assert m["pb_fv"] == pytest.approx(25.0)          # 10 x 2,5
    assert m["ev_fcf_fv"] == pytest.approx(23.6)      # (160 x 16 - 200) / 100
    assert m["avg_fv"] == pytest.approx(28.0)         # mediaan van 23,6 / 25 / 31 / 36
    assert m["n_methods"] == 4 and m["n_dropped"] == 0


def test_nettokas_telt_op_bij_de_bedrijfswaarde():
    m = multiples_fair_value(NORM, [], _rijen(net_cash=200.0), "Default", CFG)
    assert m["ev_ebitda_fv"] == pytest.approx(35.0)   # (3.300 + 200) / 100
    assert m["ev_fcf_fv"] == pytest.approx(27.6)      # (2.560 + 200) / 100


def test_eigen_historie_weegt_voor_65_procent():
    # Vier jaar een K/W van 10: 0,65 x 10 + 0,35 x 18 = 12,8.
    m = multiples_fair_value(NORM, _historie(10.0), _rijen(), "Default", CFG)
    assert m["multiples_used"]["pe"] == pytest.approx(12.8)
    assert m["pe_fv"] == pytest.approx(25.6)          # 2,00 x 12,8
    # De andere drie hebben geen historie en blijven op het sectorprofiel.
    assert m["multiples_used"]["ev_ebitda"] == pytest.approx(11.0)


def test_graham_per_scenario():
    # winst x (8,5 + 2g) x (4,4 / 5,0)
    assert graham_fair_value(NORM, "Default", CFG, "conservative") == pytest.approx(22.0)   # 2 x 12,5 x 0,88
    assert graham_fair_value(NORM, "Default", CFG, "base") == pytest.approx(29.04)          # 2 x 16,5 x 0,88
    assert graham_fair_value(NORM, "Default", CFG, "optimistic") == pytest.approx(36.08)    # 2 x 20,5 x 0,88


def test_graham_schaalt_nooit_omhoog_bij_lage_rente():
    laag = {**CFG, "valuation": {**CFG["valuation"], "bond_yield": 3.0}}
    assert graham_fair_value(NORM, "Default", laag, "base") == pytest.approx(33.0)          # 2 x 16,5 x 1


def test_perpetuity_per_scenario():
    # kasstroom per aandeel / (r - g); r is 10% plus of min één punt.
    assert perpetuity_fair_value(NORM, "Default", CFG, "conservative") == pytest.approx(1.6 / 0.09)  # 11% - 2%
    assert perpetuity_fair_value(NORM, "Default", CFG, "base") == pytest.approx(1.6 / 0.06)          # 10% - 4%
    # De groei van 6% wordt afgekapt op 5%: 9% - 5%.
    assert perpetuity_fair_value(NORM, "Default", CFG, "optimistic") == pytest.approx(40.0)


def test_gecombineerde_fair_value():
    r = combined_fair_value(NORM, [], _rijen(), "Default", CFG)
    # 0,6 x 28 + 0,4 x (29,04 + 26,667) / 2 = 16,8 + 11,141
    assert r["combined_fv"] == pytest.approx(0.6 * 28 + 0.4 * (29.04 + 1.6 / 0.06) / 2)
    assert r["combined_fv"] == pytest.approx(27.941, abs=0.001)
    # 16,8 + 0,4 x (22 + 17,778) / 2
    assert r["conservative_fv"] == pytest.approx(24.756, abs=0.001)
    # 16,8 + 0,4 x (36,08 + 40) / 2
    assert r["optimistic_fv"] == pytest.approx(32.016, abs=0.001)
    assert r["fv_methods_used"] == 3
    # Spreiding (29,04 - 26,667) / 28 = 8,5%: onder de 30%, dus "high".
    assert r["fv_spread_pct"] == pytest.approx(8.5, abs=0.1)
    assert r["fv_confidence"] == "high"


def test_scenarios_liggen_dicht_bij_elkaar_omdat_de_multiples_niet_meebewegen():
    # Vastgelegd als eigenschap van het model, niet als wens: 60% van de waarde
    # is in elk scenario gelijk, dus de band is smal (hier -11% tot +15%).
    r = combined_fair_value(NORM, [], _rijen(), "Default", CFG)
    assert 0.85 < r["conservative_fv"] / r["combined_fv"] < 0.92
    assert 1.10 < r["optimistic_fv"] / r["combined_fv"] < 1.18


def test_verlies_en_geen_kasstroom_geeft_geen_fair_value():
    leeg = {"normalized_eps": -1.0, "normalized_ebitda": -50.0, "normalized_fcf": -20.0,
            "normalized_oe_per_share": -0.2}
    r = combined_fair_value(leeg, [], _rijen(), "Default", CFG)
    # Alleen de boekwaarde-methode blijft over: één methode is te weinig.
    assert r["combined_fv"] is None
    assert r["fv_methods_used"] == 1


def test_een_wilde_methode_valt_buiten_de_mediaan():
    # Een boekwaarde van 100 per aandeel geeft K/B-waarde 250, bijna acht keer
    # de mediaan van de andere drie (31): die methode wordt verworpen.
    rijen = [{**_rijen()[0], "total_equity": 10000.0}]
    m = multiples_fair_value(NORM, [], rijen, "Default", CFG)
    assert m["pb_fv"] == pytest.approx(250.0)
    assert m["n_dropped"] == 1
    assert m["avg_fv"] == pytest.approx(31.0)         # mediaan van 23,6 / 31 / 36
