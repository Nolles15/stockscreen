"""De onderzoekswachtrij: wat erin komt, wat eruit blijft en in welke volgorde.

Draaien: python -m pytest tests/test_wachtrij.py
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import wachtrij  # noqa: E402

VANDAAG = date(2026, 11, 13)


def _rij(ticker, **extra):
    """Een beoordeelbare rij die in spoor 1 past: kwaliteit 8, 40% korting, lichte groei."""
    rij = {
        "ticker": ticker, "name": ticker, "sector": "Industrials",
        "data_status": "ok", "signal": "HOLD", "fy_lag": 0,
        "quality_score": 8.0, "margin_of_safety": 40.0, "rank_score": 70.0,
        "revenue_cagr": 0.02, "implied_growth": 4.0,
        "oordeel": None, "dubbel_van": None,
    }
    rij.update(extra)
    return rij


def _compounder(ticker, **extra):
    """Past alleen in spoor 2: geen korting, groeit 10%, koers prijst 3% in."""
    velden = {"margin_of_safety": -30.0, "rank_score": 20.0,
              "revenue_cagr": 0.10, "implied_growth": 3.0}
    velden.update(extra)
    return _rij(ticker, **velden)


def _tickers(uitslag, spoor):
    return [r["ticker"] for r in uitslag["sporen"][spoor]["rijen"]]


# --- Al onderzocht ------------------------------------------------------------

def test_recent_oordeel_haalt_een_bedrijf_uit_de_wachtrij():
    rijen = [_rij("NIEUW"),
             _rij("BEKEKEN", rank_score=90.0, oordeel={"datum": "2026-08-23", "oordeel": "HOLD"})]
    uitslag = wachtrij.bouw(rijen, VANDAAG)
    assert _tickers(uitslag, 0) == ["NIEUW"]
    assert uitslag["tellingen"]["al_onderzocht"] == 1


def test_oordeel_ouder_dan_een_jaar_mag_terugkomen():
    oud = _rij("OUD", oordeel={"datum": "2025-11-12", "oordeel": "OVERSLAAN"})   # 366 dagen
    net = _rij("NET", oordeel={"datum": "2025-11-14", "oordeel": "OVERSLAAN"})   # 364 dagen
    assert not wachtrij.al_onderzocht(oud, VANDAAG)
    assert wachtrij.al_onderzocht(net, VANDAAG)


def test_oordeel_zonder_leesbare_datum_telt_als_recent():
    assert wachtrij.al_onderzocht(_rij("X", oordeel={"datum": None, "oordeel": "HOLD"}), VANDAAG)
    assert wachtrij.al_onderzocht(_rij("X", oordeel={"datum": "onzin", "oordeel": "HOLD"}), VANDAAG)
    assert not wachtrij.al_onderzocht(_rij("X"), VANDAAG)


# --- Eén regel per bedrijf ----------------------------------------------------

def test_tweede_notering_verdwijnt_en_de_hoogst_geplaatste_blijft():
    rijen = [_rij("SFG.WA", rank_score=85.2, dubbel_van=["SFG1T.TL"]),
             _rij("SFG1T.TL", rank_score=85.4, dubbel_van=["SFG.WA"]),
             _rij("ANDER", rank_score=60.0)]
    uitslag = wachtrij.bouw(rijen, VANDAAG)
    assert _tickers(uitslag, 0) == ["SFG1T.TL", "ANDER"]
    assert uitslag["tellingen"]["tweede_noteringen"] == 1


# --- Krimpers -----------------------------------------------------------------

def test_krimpers_staan_onderaan_en_zijn_gemarkeerd():
    rijen = [_rij("KRIMP", rank_score=95.0, revenue_cagr=-0.08),
             _rij("GEZOND", rank_score=50.0)]
    uitslag = wachtrij.bouw(rijen, VANDAAG)
    assert _tickers(uitslag, 0) == ["GEZOND", "KRIMP"]
    per = {r["ticker"]: r for r in uitslag["sporen"][0]["rijen"]}
    assert per["KRIMP"]["wachtrij_krimper"] is True
    assert per["GEZOND"]["wachtrij_krimper"] is False
    assert "krimpt" in per["KRIMP"]["wachtrij_waarom"]


def test_krimpers_verdringen_geen_gezonde_bedrijven():
    gezond = [_rij(f"G{i}", rank_score=50.0 - i) for i in range(12)]
    krimp = [_rij(f"K{i}", rank_score=99.0 - i, revenue_cagr=-0.10) for i in range(5)]
    uitslag = wachtrij.bouw(gezond + krimp, VANDAAG)
    namen = _tickers(uitslag, 0)
    assert namen[:wachtrij.PER_SPOOR] == [f"G{i}" for i in range(wachtrij.PER_SPOOR)]
    assert namen[wachtrij.PER_SPOOR:] == ["K0", "K1", "K2"]
    assert uitslag["sporen"][0]["krimpers"] == 5


def test_lichte_krimp_binnen_de_grens_is_geen_krimper():
    assert not wachtrij.is_krimper(_rij("X", revenue_cagr=-0.01))
    assert wachtrij.is_krimper(_rij("X", revenue_cagr=-0.03))
    assert not wachtrij.is_krimper(_rij("X", revenue_cagr=None))


# --- Spoor 1: goedkope kwaliteit ---------------------------------------------

def test_spoor_1_eist_kwaliteit_en_korting():
    rijen = [_rij("GOED"),
             _rij("ZWAK", quality_score=5.5),
             _rij("DUUR", margin_of_safety=10.0)]
    assert _tickers(wachtrij.bouw(rijen, VANDAAG), 0) == ["GOED"]


# --- Spoor 2: compounders -----------------------------------------------------

def test_compounder_zonder_korting_komt_in_spoor_2():
    uitslag = wachtrij.bouw([_compounder("COMP")], VANDAAG)
    assert _tickers(uitslag, 0) == []
    assert _tickers(uitslag, 1) == ["COMP"]
    assert uitslag["sporen"][1]["rijen"][0]["groeimarge"] == 7.0    # 10% - 3%


def test_spoor_2_eist_kwaliteit_groei_en_een_koers_die_minder_inprijst():
    rijen = [_compounder("GOED"),
             _compounder("MATIG", quality_score=7.5),
             _compounder("VLAK", revenue_cagr=0.02),
             _compounder("DUUR", implied_growth=9.5)]       # 10% - 9,5% = 0,5 pp: te weinig
    assert _tickers(wachtrij.bouw(rijen, VANDAAG), 1) == ["GOED"]


def test_groei_boven_de_kap_telt_niet_verder_mee():
    snel = _compounder("SNEL", revenue_cagr=0.80, implied_growth=5.0)
    gestaag = _compounder("GESTAAG", revenue_cagr=0.14, implied_growth=2.0)
    assert wachtrij.groeimarge(snel) == 10.0       # 15% (kap) - 5%
    assert wachtrij.groeimarge(gestaag) == 12.0
    assert _tickers(wachtrij.bouw([snel, gestaag], VANDAAG), 1) == ["GESTAAG", "SNEL"]


def test_een_bedrijf_staat_nooit_in_beide_sporen():
    # Past in spoor 1 (korting) én zou in spoor 2 passen (groei, lage ingeprijsde groei).
    beide = _rij("BEIDE", revenue_cagr=0.10, implied_growth=1.0)
    uitslag = wachtrij.bouw([beide], VANDAAG)
    assert _tickers(uitslag, 0) == ["BEIDE"]
    assert _tickers(uitslag, 1) == []


# --- Datapoort ----------------------------------------------------------------

def test_banken_afgekeurde_data_en_oude_cijfers_staan_er_niet_in():
    rijen = [_rij("BANK", sector="Financial Services"),
             _rij("SLECHT", data_status="bad"),
             _rij("GEEN", signal="INSUFFICIENT DATA"),
             _rij("OUD", fy_lag=1),
             _rij("GOED")]
    uitslag = wachtrij.bouw(rijen, VANDAAG)
    assert _tickers(uitslag, 0) == ["GOED"]
    assert uitslag["tellingen"]["beoordeelbaar"] == 1


def test_de_rijen_uit_de_cache_worden_niet_aangepast():
    rij = _rij("X")
    wachtrij.bouw([rij], VANDAAG)
    assert "wachtrij_spoor" not in rij and "groeimarge" not in rij


def test_lege_invoer_geeft_twee_lege_sporen():
    uitslag = wachtrij.bouw([], VANDAAG)
    assert [s["rijen"] for s in uitslag["sporen"]] == [[], []]
