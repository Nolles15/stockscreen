"""De bezit-eindpunten, met een nagebootste database.

Waarom dit er is: `engine/exit_regels.py` is los getest, maar dat zegt niets
over de bedrading. De regels lezen velden uit een dashboardrij (`price_vs_fv_pct`,
`normalized_fcf_m`, `revenue_cagr`, `fy_lag`) die pás in `_dashboard_rows`
ontstaan — deels onder een andere naam dan in de database. Een typefout daarin
levert geen foutmelding op maar een regel die stilletjes nooit afgaat, en dat is
precies het soort stille fout waar dit project al twee keer op is gestruikeld.

De database wordt vervangen door een nabootsing in `sys.modules`, dus dit draait
zonder DATABASE_URL en raakt niets echts aan. Alles daarboven is de echte code:
de echte routes, de echte rijopbouw, de echte koppeling met de rapporten in
`analyses/` en de echte regels.
"""
import os
import sys
import types

WORTEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORTEL)
os.environ["SCHEDULER_ENABLED"] = "0"

# ---------------------------------------------------------------------------
# Nagebootste database. Moet vóór de import van app in sys.modules staan:
# app.py roept db.init_db() aan bij het inladen van de module.
# ---------------------------------------------------------------------------

_BEZIT: dict[str, dict] = {}


def _rij(ticker, **extra):
    """Een rij zoals db.get_dashboard_data() hem oplevert (databasenamen)."""
    basis = {
        "ticker": ticker, "name": "Testbedrijf", "sector": "Industrials",
        "market": "NL", "currency": "EUR", "added_date": "2025-01-01",
        "price": 100.0, "market_cap": 5e9, "enterprise_value": 5e9,
        "last_updated": "2026-08-13", "quality_score": 7.0,
        "normalized_fcf": 50e6, "combined_fv": 120.0, "base_fv": 120.0,
        "conservative_fv": 100.0, "optimistic_fv": 140.0, "fv_confidence": "high",
        "fv_methods_used": 4, "revenue_cagr": 0.06, "implied_growth": 2.0,
        "signal": "HOLD", "margin_of_safety": 16.7, "last_calculated": "2026-08-13",
        "latest_fy": 2025, "laatste_yahoo": 2025, "data_status": "ok",
        "completeness_pct": 95.0, "fetch_success": 1, "consecutive_failures": 0,
    }
    basis.update(extra)
    return basis


_STOCKS = [
    _rij("GOED.AS", name="Gezond Bedrijf"),
    _rij("DUUR.AS", name="Peperduur NV", price=300.0, signal="SELL",
         margin_of_safety=-150.0),
    _rij("KAPOT.AS", name="Geen Data BV", data_status="bad", price=10.0,
         combined_fv=None, signal="INSUFFICIENT DATA"),
    _rij("EVO.ST", name="Evolution AB (publ)", currency="SEK", price=2000.0,
         combined_fv=1500.0, quality_score=9.0),
    _rij("RUST.AS", name="Niet In Bezit NV"),
]


# Jaarrijen voor het moat-profiel: rendement op kapitaal 15% en een vlakke
# brutomarge, dus een groen moat-oordeel. Zonder deze rijen zouden B1 en B2
# "niet te toetsen" zijn en zou de bedradingstest juist die twee missen.
_JAARRIJEN = {
    "GOED.AS": [
        {"fiscal_year": jaar, "period_type": "annual", "revenue": 1000e6,
         "gross_profit": 400e6, "ebit": 100e6, "ebitda": 130e6,
         "total_equity": 400e6, "total_debt": 100e6, "net_cash": 0.0}
        for jaar in (2021, 2022, 2023, 2024, 2025)
    ],
}


def _maak_db_nabootsing():
    mod = types.ModuleType("engine.db")
    mod.init_db = lambda: None
    mod.get_all_stocks = lambda: [{"ticker": r["ticker"]} for r in _STOCKS]
    mod.get_dashboard_data = lambda: [dict(r) for r in _STOCKS]
    # Sinds de cache haalt de app één rij op waar hij er één nodig heeft; de
    # nepmodule moet dat kunnen, anders test hij een pad dat niet meer bestaat.
    mod.get_dashboard_row = lambda t: next(
        (dict(r) for r in _STOCKS if r["ticker"] == t), None)
    mod.get_stock = lambda t: next(({"ticker": t} for r in _STOCKS if r["ticker"] == t), None)
    mod.bezit_lijst = lambda: [dict(v, ticker=k) for k, v in _BEZIT.items()]
    mod.get_price_history = lambda t, limit=3650: []
    mod.jaarrijen_met_overrides = lambda t: ([dict(r) for r in _JAARRIJEN.get(t, [])], [])
    mod.log_activity = lambda *a, **k: None

    def vastleggen(ticker, snapshot=None, notitie=None):
        bestond = _BEZIT.get(ticker)
        if bestond:                       # bestaande momentopname blijft staan
            if notitie:
                bestond["notitie"] = notitie
            return
        _BEZIT[ticker] = {"sinds": "2026-08-13", "notitie": notitie,
                          "these_snapshot": snapshot or {}}

    def verwijderen(ticker):
        return _BEZIT.pop(ticker, None) is not None

    mod.bezit_vastleggen = vastleggen
    mod.bezit_verwijderen = verwijderen
    return mod


sys.modules["engine.db"] = _maak_db_nabootsing()

import app as app_mod  # noqa: E402


def _client():
    app_mod.app.config["TESTING"] = True
    return app_mod.app.test_client()


def _uitslag(payload, ticker):
    return next(r for r in payload if r["ticker"] == ticker)


def _regel(rij, rid):
    return next(r for r in rij["verkoop"]["regels"] if r["id"] == rid)


# ---------------------------------------------------------------------------


def test_vastleggen_bewaart_een_momentopname():
    c = _client()
    resp = c.post("/api/bezit/GOED.AS")
    assert resp.status_code == 200
    snapshot = resp.get_json()["snapshot"]

    # De momentopname moet de velden bevatten waar de vergelijkende regels op
    # leunen. Ontbreekt er één, dan gaat die regel nooit af zonder foutmelding.
    for veld in ("quality_score", "revenue_cagr", "normalized_fcf_m", "koers"):
        assert veld in snapshot, f"momentopname mist {veld}"
    assert snapshot["quality_score"] == 7.0
    assert snapshot["normalized_fcf_m"] == 50.0        # uit normalized_fcf, in miljoenen
    print("  [OK] vastleggen bewaart een bruikbare momentopname")


def test_onbekende_ticker_geeft_404():
    resp = _client().post("/api/bezit/BESTAATNIET.XX")
    assert resp.status_code == 404
    print("  [OK] een ticker buiten de screener kan niet vastgelegd worden")


def test_de_regels_lezen_echte_veldnamen():
    """De bedradingstest: velden uit _dashboard_rows moeten aankomen."""
    c = _client()
    c.post("/api/bezit/GOED.AS")
    rij = _uitslag(c.get("/api/bezit").get_json(), "GOED.AS")

    # Geen enkele regel mag "niet te toetsen" zijn door een naamfout. Zonder
    # analyse zijn C1/C2/C3/C4 terecht ongetoetst; de rest moet een uitspraak doen.
    ongetoetst = [r["id"] for r in rij["verkoop"]["regels"] if r["geraakt"] is None]
    assert ongetoetst == ["C1", "C2", "C3", "C4"], f"onverwacht ongetoetst: {ongetoetst}"

    # En de getallen moeten kloppen met de rij, niet met een standaardwaarde.
    assert _regel(rij, "A2")["waarden"]["koers_vs_fv_pct"] == 83.3
    assert _regel(rij, "B4")["waarden"]["cagr_pct"] == 6.0
    assert rij["verkoop"]["niveau"] == "groen"
    print("  [OK] alle niet-analyseregels komen aan echte waarden, uitslag groen")


def test_te_duur_wordt_rood():
    c = _client()
    c.post("/api/bezit/DUUR.AS")
    rij = _uitslag(c.get("/api/bezit").get_json(), "DUUR.AS")
    assert rij["verkoop"]["niveau"] == "rood"
    assert _regel(rij, "A2")["geraakt"] is True          # 300 van 120 = 250%
    print("  [OK] een aandeel op 250% van de fair value wordt rood")


def test_afgekeurde_data_krijgt_geen_oordeel():
    c = _client()
    c.post("/api/bezit/KAPOT.AS")
    rij = _uitslag(c.get("/api/bezit").get_json(), "KAPOT.AS")
    assert rij["verkoop"]["niveau"] == "grijs"
    assert rij["verkoop"]["regels"] == []
    print("  [OK] de datapoort houdt afgekeurde data buiten het verkoopoordeel")


def test_analyse_koppelt_en_vuurt_op_het_eigen_rapport():
    """EVO.ST hangt aan analyses/EVO.md: optimistisch 1728,68 SEK."""
    c = _client()
    c.post("/api/bezit/EVO.ST")
    rij = _uitslag(c.get("/api/bezit").get_json(), "EVO.ST")

    assert rij["oordeel"]["soort"] == "analyse"
    assert rij["oordeel"]["rapport_ticker"] == "EVO"
    c2 = _regel(rij, "C2")
    assert c2["geraakt"] is True, "koers 2000 SEK ligt boven het optimistische scenario"
    assert c2["waarden"]["optimistisch"] == 1728.68
    assert rij["verkoop"]["niveau"] == "rood"            # C2 is een harde regel
    print("  [OK] de koppeling met analyses/EVO.md werkt en C2 vuurt op 1728,68 SEK")


def test_niet_vastgelegd_komt_er_niet_in():
    payload = _client().get("/api/bezit").get_json()
    assert all(r["ticker"] != "RUST.AS" for r in payload)
    print("  [OK] alleen vastgelegde aandelen staan in /api/bezit")


def test_rood_staat_bovenaan():
    payload = _client().get("/api/bezit").get_json()
    niveaus = [r["verkoop"]["niveau"] for r in payload]
    assert niveaus.index("rood") < niveaus.index("groen")
    print("  [OK] waar iets speelt staat bovenaan")


def test_tickerlijst_is_licht_en_klopt():
    lijst = _client().get("/api/bezit/tickers").get_json()
    assert set(lijst) == set(_BEZIT)
    print("  [OK] /api/bezit/tickers geeft precies de vastgelegde tickers")


def test_verwijderen_haalt_het_eruit():
    c = _client()
    assert c.delete("/api/bezit/GOED.AS").get_json()["verwijderd"] is True
    assert "GOED.AS" not in c.get("/api/bezit/tickers").get_json()
    # Tweede keer verwijderen is geen fout, maar meldt wel dat er niets weg is.
    assert c.delete("/api/bezit/GOED.AS").get_json()["verwijderd"] is False
    print("  [OK] verwijderen werkt en is idempotent")


def test_ontbrekende_ticker_blijft_zichtbaar():
    """Vastgelegd bezit dat de screener niet (meer) kent mag niet verdwijnen."""
    _BEZIT["WEG.AS"] = {"sinds": "2026-01-01", "notitie": None, "these_snapshot": {}}
    try:
        rij = _uitslag(_client().get("/api/bezit").get_json(), "WEG.AS")
        assert rij["ontbreekt"] is True
        assert rij["verkoop"]["niveau"] == "grijs"
        assert "screener" in rij["verkoop"]["kop"]
    finally:
        _BEZIT.pop("WEG.AS", None)
    print("  [OK] bezit buiten de screener wordt getoond, niet stilletjes weggelaten")


if __name__ == "__main__":
    volgorde = [
        test_vastleggen_bewaart_een_momentopname,
        test_onbekende_ticker_geeft_404,
        test_de_regels_lezen_echte_veldnamen,
        test_te_duur_wordt_rood,
        test_afgekeurde_data_krijgt_geen_oordeel,
        test_analyse_koppelt_en_vuurt_op_het_eigen_rapport,
        test_niet_vastgelegd_komt_er_niet_in,
        test_rood_staat_bovenaan,
        test_tickerlijst_is_licht_en_klopt,
        test_ontbrekende_ticker_blijft_zichtbaar,
        test_verwijderen_haalt_het_eruit,
    ]
    for functie in volgorde:
        functie()
    print("\nAlle bezit-eindpunttests geslaagd.")
