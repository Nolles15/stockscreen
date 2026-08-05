"""Het onderzoek dat er ligt moet bij de juiste ticker terechtkomen.

Aanleiding: op 4 augustus 2026 stonden er vier onderzochte aandelen in de top
twintig van Kansen zonder dat de pagina dat wist. Twee ervan waren dezelfde
onderneming — `SFG.WA` in Warschau en `SFG1T.TL` in Tallinn — met een
tussencheck die op OVERSLAAN uitkwam. Ze deelden geen enkele letter in hun
ticker, alleen hun naam.

Wat hier bewaakt wordt is dus vooral het koppelen, want dat is waar het stil
misgaat: een gemiste koppeling ziet er precies zo uit als "nooit onderzocht",
en een verkeerde koppeling hangt andermans oordeel aan een aandeel.

Deze tests raken de database niet. De rapporten in `analyses/` en
`tussenchecks/` zijn echte bestanden; de screener-rijen worden nagebootst,
want die komen normaal uit `/api/dashboard`.
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jinja2 import Environment, FileSystemLoader

from engine import analyses as analyses_mod
from engine import oordelen

WORTEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rij(ticker, naam, price=None, currency=None) -> dict:
    return {"ticker": ticker, "name": naam, "price": price, "currency": currency}


# ---------------------------------------------------------------------------
# koppelen tegen de echte rapporten
# ---------------------------------------------------------------------------


def test_beide_noteringen_krijgen_hetzelfde_oordeel():
    """Het geval waar het om begonnen is: Silvano in Warschau en Tallinn."""
    warschau = _rij("SFG.WA", "AS Silvano Fashion Group", 4.10, "PLN")
    tallinn = _rij("SFG1T.TL", "AS Silvano Fashion Group", 0.95, "EUR")
    oordelen.verrijk([warschau, tallinn])

    assert warschau["oordeel"]["oordeel"] == "OVERSLAAN"
    assert warschau["oordeel"]["via"] is None, "dit is de notering uit het rapport zelf"

    assert tallinn["oordeel"]["oordeel"] == "OVERSLAAN"
    assert tallinn["oordeel"]["via_naam"] is True
    assert tallinn["oordeel"]["via"] == "SFG.WA", (
        "een oordeel via de bedrijfsnaam moet vertellen wáár het gemaakt is"
    )
    print("  [OK] SFG.WA en SFG1T.TL krijgen allebei OVERSLAAN")


def test_via_noemt_alleen_een_ticker_die_bestaat():
    """Op een detailpagina staat de andere notering niet in dezelfde set rijen.

    Dan mag `via` niet de kale ticker uit het rapport worden: '/stock/SFG1T.TL'
    zou anders melden dat het oordeel op 'SFG' is gemaakt, en die ticker kent
    de screener niet.
    """
    oordeel = oordelen.voor_ticker("SFG1T.TL", "AS Silvano Fashion Group", 0.95, "EUR")
    assert oordeel["oordeel"] == "OVERSLAAN"
    assert oordeel["via"] is None
    assert oordeel["via_naam"] is True
    print("  [OK] via blijft leeg als de andere notering niet in beeld is")


def test_yahoo_symbool_koppelt_exact():
    """Een analyse noemt haar Yahoo-symbool; SHEL noemt er zelfs drie."""
    rijen = [
        _rij("WKL.AS", "Wolters Kluwer", 70.32, "EUR"),
        _rij("SHEL.L", "Shell plc", 2800, "GBp"),
    ]
    oordelen.verrijk(rijen)

    assert rijen[0]["oordeel"]["soort"] == "analyse"
    assert rijen[0]["oordeel"]["link"] == "/analyses/WKL"
    assert rijen[1]["oordeel"]["link"] == "/analyses/SHEL", (
        "SHEL.L staat als tweede in 'SHEL.AS (Amsterdam) / SHEL.L (Londen) / SHEL (NYSE)'"
    )
    print("  [OK] Yahoo-symbolen koppelen exact, ook bij meerdere per rapport")


def test_ticker_zonder_rapport_blijft_ongemoeid():
    rij = _rij("ZZZ.XX", "Bedrijf zonder rapport", 10.0, "EUR")
    oordelen.verrijk([rij])
    assert "oordeel" not in rij
    print("  [OK] een ticker zonder rapport krijgt geen oordeel-blok")


def test_koersverschil_alleen_bij_dezelfde_valuta():
    """De tussencheck van Silvano noteert PLN. De Tallinnse notering staat in
    EUR; dat verschil is een wisselkoers, geen koersdaling."""
    warschau = _rij("SFG.WA", "AS Silvano Fashion Group", 4.10, "PLN")
    tallinn = _rij("SFG1T.TL", "AS Silvano Fashion Group", 0.95, "EUR")
    oordelen.verrijk([warschau, tallinn])

    assert warschau["oordeel"]["koers_verschil"] == -8.7, "4,49 -> 4,10 PLN"
    assert tallinn["oordeel"]["koers_verschil"] is None
    print("  [OK] koersverschil alleen bij een directe koppeling in dezelfde valuta")


def test_koersverschil_leeg_bij_andere_valuta_op_dezelfde_notering():
    """SHEL.L staat in pence, het rapport in euro's — geen vergelijking."""
    rij = _rij("SHEL.L", "Shell plc", 2800, "GBp")
    oordelen.verrijk([rij])
    assert rij["oordeel"]["koers_verschil"] is None
    print("  [OK] pence tegen euro's levert geen koersverschil op")


# ---------------------------------------------------------------------------
# de koppelregels zelf, met verzonnen rapporten
# ---------------------------------------------------------------------------


class _Rapporten:
    """Vervangt de echte mappen tijdens een test."""

    def __init__(self, checks=None, analyses=None):
        self.checks, self.analyses = checks or [], analyses or []

    def __enter__(self):
        self._oud = (analyses_mod.get_alle_tussenchecks, analyses_mod.get_all_summaries)
        analyses_mod.get_alle_tussenchecks = lambda: self.checks
        analyses_mod.get_all_summaries = lambda: self.analyses
        return self

    def __exit__(self, *_):
        analyses_mod.get_alle_tussenchecks, analyses_mod.get_all_summaries = self._oud
        return False


def _check(ticker, naam, oordeel="OVERSLAAN", datum="2026-08-01", koers=None, valuta=None):
    return {"ticker": ticker, "naam": naam, "oordeel": oordeel, "datum": datum,
            "koers_getal": koers, "valuta": valuta}


def _analyse(ticker, naam, oordeel="HOLD", peildatum="2026-07-02", yahoo=None):
    return {"ticker": ticker, "naam": naam, "oordeel": oordeel, "peildatum": peildatum,
            "koers_getal": None, "valuta": None, "yahoo_symbol": yahoo}


def test_kale_ticker_koppelt_niet_bij_twee_kandidaten():
    """`AD` van Ahold mag zich niet aan een willekeurige andere AD.* hechten."""
    with _Rapporten(checks=[_check("AD", "Koninklijke Ahold Delhaize N.V.")]):
        ahold = _rij("AD.AS", "Koninklijke Ahold Delhaize N.V.")
        vreemde = _rij("AD.PA", "Iets heel anders SA")
        oordelen.verrijk([ahold, vreemde])

    assert ahold["oordeel"]["oordeel"] == "OVERSLAAN"
    assert "oordeel" not in vreemde, "alleen de naam die klopt mag gekoppeld worden"
    print("  [OK] een dubbelzinnige kale ticker koppelt alleen op naam")


def test_analyse_wint_van_tussencheck():
    """Ligt de volledige analyse er, dan is de vraag of hij onderzoek waard
    was allang beantwoord."""
    with _Rapporten(checks=[_check("XYZ", "Voorbeeld NV", "VERDIEPEN")],
                    analyses=[_analyse("XYZ", "Voorbeeld NV", "KOOP", yahoo="XYZ.AS")]):
        rij = _rij("XYZ.AS", "Voorbeeld NV")
        oordelen.verrijk([rij])

    assert rij["oordeel"]["oordeel"] == "KOOP"
    assert rij["oordeel"]["soort"] == "analyse"
    print("  [OK] een volledige analyse verdringt de tussencheck")


def test_naam_van_de_screener_telt_ook_mee():
    """Het rapport spelt de naam anders dan de screener. De koppeling naar de
    tweede notering moet dan lopen via de naam zoals de screener hem kent."""
    with _Rapporten(checks=[_check("ABC", "Voorbeeld")]):
        eerste = _rij("ABC.WA", "Voorbeeld Group AS")
        tweede = _rij("ABC9Z.TL", "Voorbeeld Group AS")
        oordelen.verrijk([eerste, tweede])

    assert eerste["oordeel"]["via"] is None and eerste["oordeel"]["via_naam"] is False
    assert tweede["oordeel"]["via"] == "ABC.WA"
    print("  [OK] de naam uit de screener koppelt de tweede notering mee")


def test_oordeel_zonder_uitslag_telt_niet():
    """Een rapport waarvan het oordeel niet te lezen was mag geen lege pil geven."""
    with _Rapporten(checks=[_check("QQQ", "Leeg NV", oordeel=None)]):
        rij = _rij("QQQ.AS", "Leeg NV")
        oordelen.verrijk([rij])
    assert "oordeel" not in rij
    print("  [OK] een rapport zonder leesbaar oordeel koppelt niet")


def test_verouderd_na_een_jaar():
    assert oordelen._verouderd("2026-08-01", date(2026, 8, 5)) is False
    assert oordelen._verouderd("2025-08-01", date(2026, 8, 5)) is True
    assert oordelen._verouderd(None, date(2026, 8, 5)) is False
    assert oordelen._verouderd("onbekend", date(2026, 8, 5)) is False
    print(f"  [OK] verouderd na {oordelen.VEROUDERD_NA_DAGEN} dagen, en robuust bij rommel")


# ---------------------------------------------------------------------------
# parsen van de koers in een tussencheck
# ---------------------------------------------------------------------------


def test_valuta_uit_koersnotatie():
    """De tussenchecks noteren de koers in twee vormen, zonder apart veld."""
    gevallen = [("4,49 PLN", "PLN"), ("€6,72", "EUR"), ("€802", "EUR"),
                ("$12.50", "USD"), ("12,30", None), (None, None)]
    for tekst, verwacht in gevallen:
        gekregen = analyses_mod._valuta_uit(tekst)
        assert gekregen == verwacht, f"{tekst!r} -> {gekregen}, verwacht {verwacht}"
    print(f"  [OK] valuta uit {len(gevallen)} koersnotaties")


def test_echte_tussenchecks_hebben_een_leesbare_koers():
    for check in analyses_mod.get_alle_tussenchecks():
        assert check["koers_getal"] is not None, f"{check['ticker']}: koers onleesbaar"
        assert check["valuta"] is not None, f"{check['ticker']}: valuta onbekend"
    print("  [OK] alle tussenchecks hebben een leesbare koers en valuta")


# ---------------------------------------------------------------------------
# de detailpagina
# ---------------------------------------------------------------------------


def _omgeving() -> Environment:
    env = Environment(loader=FileSystemLoader(os.path.join(WORTEL, "templates")))
    env.globals["url_for"] = lambda *a, **k: "#"
    env.globals["get_flashed_messages"] = lambda *a, **k: []
    env.globals["request"] = type("Verzoek", (), {"path": "/stock/SFG.WA"})()

    # Dezelfde filters als app.py; zonder deze valt de render om op de banner.
    env.filters["pct"] = lambda w: "—" if w is None else f"{w:+.1f}".replace(".", ",") + "%"
    env.filters["bedrag"] = lambda w: "—" if w is None else f"{w:,.2f}"
    env.filters["datum_nl"] = lambda w: "3 augustus 2026" if w else "onbekend"
    return env


def _context(oordeel) -> dict:
    return {
        "ticker": "SFG.WA",
        "stock": {"name": "AS Silvano Fashion Group", "sector": "Consumer Cyclical",
                  "market": "Warsaw", "currency": "PLN", "description": None},
        "annual": [], "ttm": None,
        "market": {"price": 4.10, "last_updated": "2026-08-05", "market_cap": 1.6e8},
        "scores": None, "overrides": [], "override_set": set(), "hist_mult": [],
        "config": {}, "latest_fy": 2025, "fy_lag": 0, "verwacht_fy": 2025,
        "oordeel": oordeel,
    }


def test_detailpagina_toont_de_uitslag():
    oordeel = {"oordeel": "OVERSLAAN", "soort": "tussencheck", "datum": "2026-08-03",
               "link": "/tussenchecks/SFG", "via": None, "verouderd": False,
               "koers_verschil": -8.7}
    html = _omgeving().get_template("stock.html").render(**_context(oordeel))

    assert "Tussencheck: OVERSLAAN" in html
    assert "/tussenchecks/SFG" in html
    assert "-8,7%" in html
    print("  [OK] de detailpagina toont de tussencheck met koersverschil")


def test_detailpagina_noemt_de_andere_notering():
    oordeel = {"oordeel": "OVERSLAAN", "soort": "tussencheck", "datum": "2026-08-03",
               "link": "/tussenchecks/SFG", "via": "SFG.WA", "via_naam": True,
               "verouderd": True, "koers_verschil": None}
    html = _omgeving().get_template("stock.html").render(**_context(oordeel))

    assert "gemaakt op <strong>SFG.WA</strong>" in html
    assert "ouder dan een jaar" in html
    print("  [OK] de detailpagina noemt de andere notering en de veroudering")


def test_detailpagina_verzint_geen_ticker_voor_de_andere_notering():
    """Zonder bekende zusternotering blijft de zin algemeen."""
    oordeel = {"oordeel": "OVERSLAAN", "soort": "tussencheck", "datum": "2026-08-03",
               "link": "/tussenchecks/SFG", "via": None, "via_naam": True,
               "verouderd": False, "koers_verschil": None}
    html = _omgeving().get_template("stock.html").render(**_context(oordeel))

    assert "op een andere notering van hetzelfde bedrijf" in html
    assert "gemaakt op <strong>" not in html
    print("  [OK] zonder bekende zusternotering blijft de zin algemeen")


def test_detailpagina_zwijgt_over_een_koers_die_niet_bewoog():
    """'Sindsdien bewoog de koers +0%' is ruis, geen mededeling."""
    oordeel = {"oordeel": "TWIJFEL", "soort": "tussencheck", "datum": "2026-08-01",
               "link": "/tussenchecks/HUG", "via": None, "verouderd": False,
               "koers_verschil": 0.0}
    html = _omgeving().get_template("stock.html").render(**_context(oordeel))
    assert "Sindsdien bewoog de koers" not in html
    print("  [OK] een koersverschil onder een procent blijft weg")


def test_detailpagina_zonder_oordeel_blijft_heel():
    html = _omgeving().get_template("stock.html").render(**_context(None))
    assert "Tussencheck:" not in html
    assert len(html) > 3000, "pagina is verdacht kort"
    print("  [OK] zonder oordeel staat er geen banner")


if __name__ == "__main__":
    for naam, functie in sorted(globals().items()):
        if naam.startswith("test_") and callable(functie):
            functie()
    print("\nAlle tests oordelen geslaagd.")
