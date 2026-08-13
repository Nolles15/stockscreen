"""Verkoopregels: bewaakt de regels, de drempels en het eindoordeel.

Drie eisen die niet mogen sneuvelen:

1. **De datapoort gaat vóór alles.** Een aandeel met afgekeurde data of een
   achterlopend boekjaar krijgt geen verkoopoordeel, ook niet als de koers
   schreeuwt. Dat is dezelfde poort als in de screener; hem hier overslaan
   herhaalt de fout van de negentien afgekeurde aandelen met een vers signaal.
2. **Geen aandeel met een recente KOOP-analyse wordt rood.** Getoetst met de
   echte cijfers uit de rapporten van Evolution, Accenture en Textbook.
3. **Op dag één is de momentopname gelijk aan vandaag**, dus de vergelijkende
   regels kunnen niet afgaan. Als dat wél gebeurt is er een teken omgedraaid.

Elke regel wordt bovendien getoetst op zijn eigen twee getallen, in de trant van
tests/test_trace.py: de uitkomst moet volgen uit wat er getoond wordt.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import exit_regels  # noqa: E402

CONFIG = {
    "signals": {"sell_pct": 130, "sell_pct_high_quality": 175},
    "valuation": {"max_perpetuity_growth": 5},
}


def _rij(**extra):
    """Een gezonde rij: alle poorten open, geen enkele regel geraakt."""
    basis = {
        "ticker": "TEST.AS", "name": "Testbedrijf", "currency": "EUR",
        "price": 100.0, "combined_fv": 120.0, "price_vs_fv_pct": 83.3,
        "signal": "HOLD", "quality_score": 7.0, "data_status": "ok", "fy_lag": 0,
        "implied_growth": 2.0, "revenue_cagr": 0.06, "normalized_fcf_m": 50.0,
        "rank_score": 60.0,
    }
    basis.update(extra)
    return basis


def _regel(uitslag, rid):
    return next(r for r in uitslag["regels"] if r["id"] == rid)


def _geraakt(uitslag, rid):
    return _regel(uitslag, rid)["geraakt"]


# ---------------------------------------------------------------------------
# De datapoort
# ---------------------------------------------------------------------------


def test_datapoort_houdt_alles_tegen():
    # Peperduur én verlieslatend, maar de data is afgekeurd: geen oordeel.
    slecht = _rij(data_status="bad", price_vs_fv_pct=400.0, signal="SELL")
    uitslag = exit_regels.toets(slecht, config=CONFIG)
    assert uitslag["niveau"] == "grijs"
    assert uitslag["regels"] == []
    assert "bad" in uitslag["toelichting"]

    # Twee boekjaren achterstand telt net zo zwaar.
    oud = _rij(fy_lag=2, price_vs_fv_pct=400.0)
    assert exit_regels.toets(oud, config=CONFIG)["niveau"] == "grijs"

    # Eén jaar achterstand is normaal (jaarverslagen verschijnen met vertraging).
    assert exit_regels.toets(_rij(fy_lag=1), config=CONFIG)["niveau"] == "groen"

    # Zonder koers valt er niets te vergelijken.
    assert exit_regels.toets(_rij(price=None), config=CONFIG)["niveau"] == "grijs"
    print("  [OK] datapoort: bad/missing, boekjaarachterstand en ontbrekende koers -> grijs")


# ---------------------------------------------------------------------------
# A — waardering
# ---------------------------------------------------------------------------


def test_a1_volgt_het_screenersignaal():
    assert _geraakt(exit_regels.toets(_rij(signal="SELL"), config=CONFIG), "A1") is True
    assert _geraakt(exit_regels.toets(_rij(signal="HOLD"), config=CONFIG), "A1") is False
    # Zonder signaal is de regel niet te toetsen, niet "gehaald".
    assert _geraakt(exit_regels.toets(_rij(signal=None), config=CONFIG), "A1") is None
    print("  [OK] A1 volgt SELL van de screener, en zwijgt zonder signaal")


def test_a2_is_de_harde_waarderingsregel():
    # 175% = de drempel die compounders krijgen; daarboven is iedereen te duur.
    net_onder = exit_regels.toets(_rij(price_vs_fv_pct=174.9, signal="SELL"), config=CONFIG)
    assert _geraakt(net_onder, "A2") is False
    assert net_onder["niveau"] == "oranje"      # alleen A1 geraakt

    erboven = exit_regels.toets(_rij(price_vs_fv_pct=175.0, signal="SELL"), config=CONFIG)
    assert _geraakt(erboven, "A2") is True
    assert erboven["niveau"] == "rood"          # één harde regel is genoeg
    # De uitkomst moet volgen uit de twee getoonde getallen.
    waarden = _regel(erboven, "A2")["waarden"]
    assert waarden["koers_vs_fv_pct"] >= waarden["grens_pct"]
    print("  [OK] A2 vuurt op 175% van de fair value en maakt het oordeel meteen rood")


def test_a3_eist_beide_voorwaarden():
    # Ingeprijsd boven het modelmaximum (5%) én boven de eigen omzetgroei.
    duur = _rij(implied_growth=9.0, revenue_cagr=0.03)
    assert _geraakt(exit_regels.toets(duur, config=CONFIG), "A3") is True

    # Even hoog ingeprijsd, maar het bedrijf groeit harder: geen verkoopreden.
    verdiend = _rij(implied_growth=9.0, revenue_cagr=0.12)
    assert _geraakt(exit_regels.toets(verdiend, config=CONFIG), "A3") is False

    # Onder het modelmaximum telt niet, hoe laag de omzetgroei ook is.
    rustig = _rij(implied_growth=4.0, revenue_cagr=0.01)
    assert _geraakt(exit_regels.toets(rustig, config=CONFIG), "A3") is False

    # Geen ingeprijsde groei (verlies) -> niet te toetsen.
    assert _geraakt(exit_regels.toets(_rij(implied_growth=None), config=CONFIG), "A3") is None
    print("  [OK] A3 vuurt alleen boven het modelmaximum én boven de eigen groei")


# ---------------------------------------------------------------------------
# B — these
# ---------------------------------------------------------------------------


def test_b1_neemt_het_moat_oordeel_over():
    rood = {"niveau": "rood", "kop": "Rendement stort in slechte jaren in",
            "roic_mediaan": 4.0, "brutomarge_trend_pp": -5.0}
    assert _geraakt(exit_regels.toets(_rij(), moat=rood, config=CONFIG), "B1") is True

    groen = {"niveau": "groen", "kop": "Hoog en standvastig", "roic_mediaan": 22.0}
    assert _geraakt(exit_regels.toets(_rij(), moat=groen, config=CONFIG), "B1") is False

    # Grijs betekent "te weinig jaren": dat is geen goedkeuring en geen afkeuring.
    grijs = {"niveau": "grijs", "kop": "Te weinig jaren"}
    assert _geraakt(exit_regels.toets(_rij(), moat=grijs, config=CONFIG), "B1") is None
    print("  [OK] B1 volgt het gekalibreerde moat-oordeel, grijs blijft ongetoetst")


def test_b2_en_b3_meten_verandering_niet_niveau():
    moat_nu = {"niveau": "geel", "roic_mediaan": 9.0}
    snapshot = {"roic_mediaan": 15.0, "quality_score": 8.0}
    uitslag = exit_regels.toets(_rij(quality_score=5.5), snapshot=snapshot,
                                moat=moat_nu, config=CONFIG)
    assert _geraakt(uitslag, "B2") is True      # 15,0 -> 9,0 = 6 punten
    assert _geraakt(uitslag, "B3") is True      # 8,0 -> 5,5 = 2,5 punt

    # Een laag maar onveranderd niveau is géén these-breuk. Dit is de regel die
    # in fase 2 als `sell_quality_floor` is geschrapt; hij mag niet terugkomen.
    laag_maar_stabiel = exit_regels.toets(
        _rij(quality_score=3.0), snapshot={"roic_mediaan": 8.0, "quality_score": 3.0},
        moat={"niveau": "geel", "roic_mediaan": 8.0}, config=CONFIG)
    assert _geraakt(laag_maar_stabiel, "B2") is False
    assert _geraakt(laag_maar_stabiel, "B3") is False
    assert laag_maar_stabiel["niveau"] == "groen"
    print("  [OK] B2/B3 meten de daling, niet het niveau — geen kwaliteitsvloer via de achterdeur")


def test_b4_omzetkrimp_pas_vanaf_twee_procent():
    assert _geraakt(exit_regels.toets(_rij(revenue_cagr=-0.05), config=CONFIG), "B4") is True
    assert _geraakt(exit_regels.toets(_rij(revenue_cagr=-0.01), config=CONFIG), "B4") is False
    print("  [OK] B4 gebruikt dezelfde -2%-grens als de omzetkrimp-waarschuwing")


def test_definitiewissel_in_de_omzet_is_geen_krimp():
    """Adyen, 13 augustus 2026 — de eerste echte valse rode vlag.

    Yahoo gaf voor 2022 de brúto omzet (8.936 mln, inclusief doorbetaalde
    kaartkosten) en vanaf 2023 de netto-omzet. De driejaars-CAGR las dat als 33%
    krimp per jaar en zette twee regels aan het werk, terwijl Adyen ~19% per jaar
    groeide. De brutowinst liep wél netjes door — daaraan is de wissel te zien.
    """
    adyen = [
        {"fiscal_year": 2022, "period_type": "annual", "revenue": 8936e6},
        {"fiscal_year": 2023, "period_type": "annual", "revenue": 1863e6},
        {"fiscal_year": 2024, "period_type": "annual", "revenue": 2226e6},
        {"fiscal_year": 2025, "period_type": "annual", "revenue": 2647e6},
    ]
    breuk = exit_regels.omzetbreuk(adyen)
    assert breuk and "definitie" in breuk

    rij = _rij(revenue_cagr=-0.3334, implied_growth=5.5)
    uitslag = exit_regels.toets(rij, annual=adyen, config=CONFIG)
    assert _geraakt(uitslag, "B4") is None, "krimp op een definitiewissel"
    assert _geraakt(uitslag, "A3") is None, "ingeprijsde groei tegen een kapotte reeks"
    assert uitslag["niveau"] == "groen"

    # Zonder de reeks erbij vuurt hij nog steeds — de guard mag alleen ingrijpen
    # als hij de breuk daadwerkelijk gezien heeft.
    zonder = exit_regels.toets(rij, config=CONFIG)
    assert _geraakt(zonder, "B4") is True

    # Een normale reeks, ook een fors dalende, blijft gewoon meetellen.
    dalend = [
        {"fiscal_year": 2022, "period_type": "annual", "revenue": 1000e6},
        {"fiscal_year": 2023, "period_type": "annual", "revenue": 850e6},
        {"fiscal_year": 2024, "period_type": "annual", "revenue": 700e6},
        {"fiscal_year": 2025, "period_type": "annual", "revenue": 600e6},
    ]
    assert exit_regels.omzetbreuk(dalend) is None
    assert _geraakt(exit_regels.toets(_rij(revenue_cagr=-0.16), annual=dalend,
                                      config=CONFIG), "B4") is True
    print("  [OK] een definitiewissel in de omzetreeks telt niet als krimp (Adyen)")


def test_b5_alleen_bij_een_omslag():
    omgeslagen = exit_regels.toets(_rij(normalized_fcf_m=-10.0),
                                   snapshot={"normalized_fcf_m": 40.0}, config=CONFIG)
    assert _geraakt(omgeslagen, "B5") is True

    # Al negatief bij vastleggen: dat wist je, geen nieuw feit.
    altijd_al = exit_regels.toets(_rij(normalized_fcf_m=-10.0),
                                  snapshot={"normalized_fcf_m": -8.0}, config=CONFIG)
    assert _geraakt(altijd_al, "B5") is False
    print("  [OK] B5 vuurt op de omslag, niet op een kasstroom die altijd al negatief was")


# ---------------------------------------------------------------------------
# C — analyse
# ---------------------------------------------------------------------------


def _analyse(**extra):
    basis = {
        "valuta": "EUR", "fair_value_kansgewogen": 150.0,
        "scenarios": {"pessimistisch": 80.0, "basis": 150.0, "optimistisch": 200.0},
    }
    basis.update(extra)
    return basis


def test_c1_en_c2_leggen_de_koers_langs_het_rapport():
    onder = exit_regels.toets(_rij(price=100.0), analyse=_analyse(), config=CONFIG)
    assert _geraakt(onder, "C1") is False
    assert _geraakt(onder, "C2") is False

    tussenin = exit_regels.toets(_rij(price=160.0, price_vs_fv_pct=100.0),
                                 analyse=_analyse(), config=CONFIG)
    assert _geraakt(tussenin, "C1") is True
    assert _geraakt(tussenin, "C2") is False
    assert tussenin["niveau"] == "oranje"

    erboven = exit_regels.toets(_rij(price=210.0, price_vs_fv_pct=100.0),
                                analyse=_analyse(), config=CONFIG)
    assert _geraakt(erboven, "C2") is True
    assert erboven["niveau"] == "rood"
    print("  [OK] C1/C2 gebruiken de kansgewogen waarde en het optimistische scenario")


def test_c_zwijgt_bij_een_andere_valuta():
    """Silvano-les: dezelfde waarde in twee valuta is geen koersverschil."""
    ander = exit_regels.toets(_rij(price=100.0, currency="PLN"),
                              analyse=_analyse(valuta="EUR"), config=CONFIG)
    assert _geraakt(ander, "C1") is None
    assert _geraakt(ander, "C2") is None
    assert "niet vergelijkbaar" in _regel(ander, "C1")["uitleg"]

    # Ook een koppeling via de bedrijfsnaam betekent: andere notering, andere valuta.
    via_naam = exit_regels.toets(_rij(price=250.0), analyse=_analyse(),
                                 oordeel={"via_naam": True, "oordeel": "KOOP"}, config=CONFIG)
    assert _geraakt(via_naam, "C2") is None
    print("  [OK] C-regels zwijgen bij een andere valuta of een koppeling via de naam")


def test_c4_ziet_de_tegenspraak():
    pass_oordeel = {"oordeel": "PASS", "soort": "analyse", "datum": "2026-05-01"}
    assert _geraakt(exit_regels.toets(_rij(), oordeel=pass_oordeel, config=CONFIG), "C4") is True

    overslaan = {"oordeel": "OVERSLAAN", "soort": "tussencheck", "datum": "2026-05-01"}
    assert _geraakt(exit_regels.toets(_rij(), oordeel=overslaan, config=CONFIG), "C4") is True

    koop = {"oordeel": "KOOP", "soort": "analyse", "datum": "2026-05-01"}
    assert _geraakt(exit_regels.toets(_rij(), oordeel=koop, config=CONFIG), "C4") is False
    print("  [OK] C4 markeert bezit dat je eigen onderzoek afraadde")


# ---------------------------------------------------------------------------
# D — informatief
# ---------------------------------------------------------------------------


def test_d1_telt_nooit_mee():
    ver_onder = exit_regels.toets(_rij(rank_score=10.0), rank_grens=40.0, config=CONFIG)
    assert ver_onder["informatief"][0]["geraakt"] is True
    assert ver_onder["niveau"] == "groen"           # informatief kleurt niets
    assert all(r["id"] != "D1" for r in ver_onder["geraakt"])
    print("  [OK] D1 wordt getoond maar telt niet mee in het oordeel")


# ---------------------------------------------------------------------------
# Het eindoordeel
# ---------------------------------------------------------------------------


def test_rood_eist_twee_families():
    # Twee geraakte regels binnen dezelfde familie is nog geen rood: dat zijn
    # twee metingen van dezelfde waarneming (te duur).
    zelfde = exit_regels.toets(_rij(signal="SELL", implied_growth=9.0, revenue_cagr=0.01),
                               config=CONFIG)
    assert [r["id"] for r in zelfde["geraakt"]] == ["A1", "A3"]
    assert zelfde["niveau"] == "oranje"

    # Eén uit de waardering en één uit de these: dan wél.
    verschillend = exit_regels.toets(
        _rij(signal="SELL", revenue_cagr=-0.06), config=CONFIG)
    families = {r["familie"] for r in verschillend["geraakt"]}
    assert families == {"waardering", "these"}
    assert verschillend["niveau"] == "rood"
    print("  [OK] rood vraagt twee families, of één harde regel")


def test_groen_meldt_hoeveel_er_getoetst_is():
    uitslag = exit_regels.toets(_rij(), config=CONFIG)
    assert uitslag["niveau"] == "groen"
    # Stilte moet verklaard worden: het aantal getoetste regels hoort in de kop.
    assert str(uitslag["aantal_getoetst"]) in uitslag["kop"]
    assert uitslag["aantal_getoetst"] < uitslag["aantal_regels"]   # zonder analyse
    print("  [OK] groen noemt hoeveel regels er daadwerkelijk getoetst zijn")


def test_dag_een_kan_niet_afgaan():
    """De momentopname is bij vastleggen gelijk aan vandaag."""
    rij = _rij(quality_score=6.0, normalized_fcf_m=25.0)
    moat = {"niveau": "geel", "roic_mediaan": 11.0, "brutomarge_trend_pp": -1.0}
    snapshot = exit_regels.momentopname(rij, moat)
    uitslag = exit_regels.toets(rij, snapshot=snapshot, moat=moat, config=CONFIG)
    for rid in ("B2", "B3", "B5"):
        assert _geraakt(uitslag, rid) is False, f"{rid} ging af op dag één"
    assert uitslag["niveau"] == "groen"
    print("  [OK] op dag één gaat geen enkele vergelijkende regel af")


# ---------------------------------------------------------------------------
# Kalibratie op echte rapporten
# ---------------------------------------------------------------------------


def test_koopoordelen_worden_niet_rood():
    """Echte cijfers uit de rapporten: een KOOP met ruime upside blijft groen.

    Peildatumkoers en waardeniveaus komen uit analyses/EVO.md, ACN.md en TXT.md.
    Zou een van deze rood worden, dan staan de C-drempels verkeerd om.
    """
    gevallen = [
        # ticker, koers, valuta, kansgewogen, optimistisch
        ("EVO.ST", 735.80, "SEK", 1106.96, 1728.68),
        ("ACN", 232.66, "USD", 267.15, 345.72),
        ("TXT.WA", 39.60, "PLN", 46.11, 66.89),
    ]
    for ticker, koers, valuta, kansgewogen, optimistisch in gevallen:
        rij = _rij(ticker=ticker, price=koers, currency=valuta, price_vs_fv_pct=95.0)
        analyse = {"valuta": valuta, "fair_value_kansgewogen": kansgewogen,
                   "scenarios": {"optimistisch": optimistisch}}
        oordeel = {"oordeel": "KOOP", "soort": "analyse", "datum": "2026-08-07",
                   "verouderd": False}
        uitslag = exit_regels.toets(rij, analyse=analyse, oordeel=oordeel, config=CONFIG)
        assert uitslag["niveau"] == "groen", f"{ticker} werd {uitslag['niveau']}"
        assert _geraakt(uitslag, "C1") is False
        assert _geraakt(uitslag, "C2") is False
    print("  [OK] EVO, ACN en TXT blijven groen op hun eigen peildatumkoers")


if __name__ == "__main__":
    for naam, functie in sorted(globals().items()):
        if naam.startswith("test_") and callable(functie):
            functie()
    print("\nAlle verkoopregel-tests geslaagd.")
