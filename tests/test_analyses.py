"""De analyses-pagina moet elk rapport aankunnen zonder de database.

De rapporten in `analyses/` zijn met de hand geschreven markdown uit de
aandelenanalyse-pipeline. De bullets variëren in vorm — het oordeel staat
soms als `- **Oordeel:** KOOP` en soms als
`- **Oordeel** (enum UITSLUITEND: **KOOP** | ...): **HOLD**` — dus de parser
moet per veld kunnen falen zonder de pagina mee te nemen.

Twee dingen worden hier bewaakt die in productie al misgingen:

1. Getalnotatie. De rapporten mengen `1.239,40` (ASML) met `93.79` (NVDA).
   Toen de punt blind als decimaalteken werd gelezen, werd ASML's koers
   1,239 en toonde de site een upside van +114476%.

2. Sectiefiltering. Interne pipeline-administratie (bronnen-inventaris,
   metadata, afrondingschecklist, opmerkingen voor Claude Code) hoort niet
   op een pagina die met vrienden gedeeld wordt.

app.py kan hier niet geïmporteerd worden: die draait `db.init_db()` bij
import en heeft dus een DATABASE_URL nodig. De analyses-feature raakt de
database niet, dus deze test zet een Flask-app op met dezelfde templates
en dezelfde route-logica.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from flask import Flask, render_template

from engine import analyses as analyses_mod
from engine import scorebord as scorebord_mod

WORTEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Sectiekoppen die nooit op de pagina mogen staan. Losse verwijzingen in de
# lopende tekst ("zie de bronnen-inventaris") zijn wél gewoon rapportinhoud.
INTERNE_KOP = re.compile(
    r"<h2>\s*(Bronnen-inventaris|Metadata|Afronding"
    r"|Opmerkingen voor Claude Code|HOE OM TE GAAN)",
    re.I,
)


def _app(market_data=None, db_kapot=False) -> Flask:
    """Flask-app met de route-logica uit app.py, tegen een nagebootste screener."""
    app = Flask(
        __name__,
        template_folder=os.path.join(WORTEL, "templates"),
        static_folder=os.path.join(WORTEL, "static"),
    )

    @app.template_filter("pct")
    def _pct(waarde):
        if waarde is None:
            return "—"
        tekst = f"{waarde:+.1f}"
        if tekst.endswith(".0"):
            tekst = tekst[:-2]
        return tekst.replace(".", ",") + "%"

    @app.template_filter("bedrag")
    def _bedrag(waarde):
        if waarde is None:
            return "—"
        heel, _, decimalen = f"{waarde:,.2f}".partition(".")
        return heel.replace(",", ".") + "," + decimalen

    def verrijk(analyse):
        analyse = dict(analyse)
        analyse["nu_koers"] = analyse["nu_verschil_pct"] = None
        analyse["nu_upside_pct"] = analyse["nu_datum"] = None
        if db_kapot:
            return analyse
        market = (market_data or {}).get(analyse.get("yahoo_symbol") or "")
        prijs = (market or {}).get("price")
        if not prijs:
            return analyse
        analyse["nu_koers"] = prijs
        if analyse.get("koers_getal"):
            analyse["nu_verschil_pct"] = round((prijs / analyse["koers_getal"] - 1) * 100, 1)
        if analyse.get("fair_value_getal"):
            analyse["nu_upside_pct"] = round((analyse["fair_value_getal"] / prijs - 1) * 100, 1)
        return analyse

    def config():
        with open(os.path.join(WORTEL, "config.yaml"), encoding="utf-8") as f:
            return yaml.safe_load(f)

    @app.route("/analyses")
    def overzicht():
        return render_template(
            "analyses.html",
            analyses=[verrijk(a) for a in analyses_mod.get_all_summaries()],
            logo_token=analyses_mod.LOGO_DEV_TOKEN,
            subtab="analyses",
            pijplijn=scorebord_mod.pijplijn_overzicht(),
            config=config(),
        )

    @app.route("/analyses/<ticker>")
    def detail(ticker):
        a = analyses_mod.get_analyse(ticker)
        if not a:
            return "Analyse niet gevonden", 404
        return render_template(
            "analyse_detail.html",
            a=verrijk(a),
            logo_token=analyses_mod.LOGO_DEV_TOKEN,
            config=config(),
        )

    @app.route("/tussenchecks")
    def tussenchecks():
        return render_template(
            "tussenchecks.html",
            checks=analyses_mod.get_alle_tussenchecks(),
            subtab="tussenchecks",
            pijplijn=scorebord_mod.pijplijn_overzicht(),
            config=config(),
        )

    @app.route("/tussenchecks/<ticker>")
    def tussencheck(ticker):
        c = analyses_mod.get_tussencheck(ticker)
        if not c:
            return "Tussencheck niet gevonden", 404
        return render_template("tussencheck_detail.html", c=c, config=config())

    return app


def test_getalnotatie_europees_en_amerikaans():
    """Regressie: '1.239,40' werd 1,239 en gaf ASML +114476% upside."""
    gevallen = [
        ("−17,9", -17.9),   # echt minteken (U+2212), niet het koppelteken
        ("–17,9", -17.9),   # halve kastlijn als min
        ("2020–2024", 2020),  # maar niet in een jaartalreeks
        ("1.239,40", 1239.40),   # EUR: punt is duizendtal
        ("93.79", 93.79),        # USD: punt is decimaal
        ("57,56", 57.56),
        ("1.215", 1215),         # geen komma, drie cijfers -> duizendtal
        ("1.085", 1085),
        ("195,3", 195.3),
        ("-53", -53),
        ("226 NOK", 226),        # valuta achter het getal
        ("12,9 mld", 12.9),
    ]
    for tekst, verwacht in gevallen:
        got = analyses_mod._parse_getal(tekst)
        assert got is not None and abs(got - verwacht) < 0.001, f"{tekst!r} gaf {got}"


def test_elk_rapport_geeft_zijn_waardeniveaus_af():
    """De verkoopregels leunen op de kansgewogen waarde en de scenario's.

    Ontbreekt er één, dan zwijgt regel C1 of C2 zonder foutmelding en denk je
    dat er niets aan de hand is. Alle 29 rapporten hebben ze; dat hoort zo te
    blijven.
    """
    for a in analyses_mod.get_all_summaries():
        t = a["ticker"]
        assert a.get("fair_value_kansgewogen") is not None, f"{t} mist de kansgewogen waarde"
        scen = a.get("scenarios") or {}
        for naam in ("pessimistisch", "basis", "optimistisch"):
            assert scen.get(naam) is not None, f"{t} mist scenario {naam}"
        assert scen["pessimistisch"] < scen["optimistisch"], f"{t}: scenario's staan omgekeerd"


def test_scenariotabel_komt_uit_de_samenvatting_niet_uit_de_dcf_sectie():
    """Elk rapport heeft deze tabel twee keer, met een andere kolomvolgorde.

    In de DCF-sectie staat de fair value verderop in de rij (bij EVO op kolom
    vier, bij ACN op drie). Zoeken op de scenarionaam door het hele bestand
    levert dan een groeipercentage of een WACC op in plaats van een waarde.
    Deze tabel bootst dat na: de tweede tabel zou 7,72 opleveren.
    """
    execsum = (
        "- **Fair value scenarios:**\n\n"
        "| Scenario | Fair value (SEK) | Upside % | Kans % |\n"
        "|---|---|---|---|\n"
        "| Pessimistisch | 609,44 | −17,2 | 35 |\n"
        "| Basis | 1217,60 | +65,5 | 45 |\n"
        "| Optimistisch | 1728,68 | +134,9 | 20 |\n"
    )
    uit = analyses_mod._parse_scenarios(execsum, "SEK")
    assert uit == {"pessimistisch": 609.44, "basis": 1217.60, "optimistisch": 1728.68}

    # De kolom wordt uit de kop bepaald, niet vastgezet op de tweede.
    anders = (
        "| Scenario | FCF-groei % | WACC % | Fair value | Upside % | Kans % |\n"
        "|---|---|---|---|---|---|\n"
        "| Optimistisch | +8,0 | 7,72 | 1728,68 | +134,9 | 20 |\n"
    )
    assert analyses_mod._parse_scenarios(anders, "SEK")["optimistisch"] == 1728.68

    # Zonder herkenbare fair value-kolom liever niets dan een gok.
    zonder = ("| Scenario | WACC % | Kans % |\n|---|---|---|\n"
              "| Optimistisch | 7,72 | 20 |\n")
    assert analyses_mod._parse_scenarios(zonder, "SEK") == {}

    # Twee fair value-kolommen (EUR én SEK): die van het aandeel wint.
    twee = (
        "| Scenario | WACC % | Fair value EUR | Fair value SEK | Kans % |\n"
        "|---|---|---|---|---|\n"
        "| Optimistisch | 7,72 | 157,74 | 1728,68 | 20 |\n"
    )
    assert analyses_mod._parse_scenarios(twee, "SEK")["optimistisch"] == 1728.68


def test_koers_en_fair_value_zelfde_ordegrootte():
    """Een factor 1000 ertussen betekent een verkeerd gelezen scheidingsteken."""
    for a in analyses_mod.get_all_summaries():
        koers, fv = a.get("koers_getal"), a.get("fair_value_getal")
        if koers and fv:
            verhouding = max(koers, fv) / min(koers, fv)
            assert verhouding < 100, f"{a['ticker']}: koers={koers} fair value={fv}"


def test_elk_rapport_levert_de_vijftien_secties():
    analyses = analyses_mod.get_all_summaries()
    assert analyses, "geen rapporten gevonden in analyses/"
    for samenvatting in analyses:
        a = analyses_mod.get_analyse(samenvatting["ticker"])
        nummers = [s["nr"] for s in a["secties"]]
        assert nummers == list(range(1, 16)), f"{a['ticker']}: secties {nummers}"


def test_overzicht_toont_alle_rapporten():
    c = _app().test_client()
    r = c.get("/analyses")
    html = r.get_data(as_text=True)
    aantal = len(analyses_mod.get_all_summaries())
    assert r.status_code == 200
    assert html.count('class="analyse-kaart"') == aantal
    assert html.count("img.logo.dev") == aantal, "elk bedrijf hoort een logo-url te hebben"
    assert 'class="oordeel oordeel-onbekend"' not in html, "elk rapport hoort een oordeel te hebben"


def test_detailpagina_verbergt_interne_secties():
    c = _app().test_client()
    for samenvatting in analyses_mod.get_all_summaries():
        r = c.get(f"/analyses/{samenvatting['ticker']}")
        html = r.get_data(as_text=True)
        assert r.status_code == 200, samenvatting["ticker"]
        assert not INTERNE_KOP.search(html), f"{samenvatting['ticker']}: interne sectiekop zichtbaar"
        assert "Yahoo symbol" not in html, f"{samenvatting['ticker']}: metadata-blok zichtbaar"
        assert html.count('class="analyse-sectie"') == 15


def test_ticker_lookup_is_hoofdletterongevoelig_en_veilig():
    c = _app().test_client()
    assert c.get("/analyses/WKL").status_code == 200
    assert c.get("/analyses/wkl").status_code == 200
    assert c.get("/analyses/PREV-B").status_code == 200, "ticker met streepje"
    assert c.get("/analyses/FOO").status_code == 404
    # De lookup zoekt op geparste ticker, dus een pad kan nooit uit de URL komen.
    assert c.get("/analyses/..%2f..%2fconfig").status_code in (400, 404, 308)


def test_actuele_koers_naast_peildatum_koers():
    c = _app({"WKL.AS": {"price": 61.20}}).test_client()
    html = c.get("/analyses/WKL").get_data(as_text=True)
    assert "EUR 57,56" in html, "koers van de peildatum blijft staan"
    assert "EUR 61,20" in html, "actuele koers erbij"
    assert "+6,3%" in html, "verschil sinds de peildatum"
    assert "+195,3%" in html, "upside van het rapport wordt niet herrekend"
    assert "+177,8%" in html, "upside op de actuele koers"


def test_zonder_screener_koers_blijft_de_pagina_werken():
    """Niet elke ticker staat in de screener — NVDA en NBIS bijvoorbeeld niet."""
    c = _app({}).test_client()
    r = c.get("/analyses/HAFNI")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert 'class="nu-blok"' not in html
    assert "NOK 70,35" in html, "peildatum-koers hoort er wel te staan"


def test_database_storing_sloopt_de_pagina_niet():
    c = _app(db_kapot=True).test_client()
    for pad in ("/analyses", "/analyses/WKL"):
        assert c.get(pad).status_code == 200, pad


def test_tussenchecks_hebben_kop_en_oordeel():
    """Kop, oordeel en de meta-regel (datum, koers, beurswaarde, beurs)."""
    checks = analyses_mod.get_alle_tussenchecks()
    assert checks, "geen tussenchecks gevonden in tussenchecks/"
    for c in checks:
        assert c["oordeel"] in ("VERDIEPEN", "TWIJFEL", "OVERSLAAN"), f"{c['ticker']}: {c['oordeel']}"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", c["datum"] or ""), f"{c['ticker']}: datum {c['datum']}"
        assert c["koers"], f"{c['ticker']}: koers ontbreekt"
        assert c["beurswaarde"], f"{c['ticker']}: beurswaarde ontbreekt"
        assert c["naam"] != c["ticker"], f"{c['ticker']}: bedrijfsnaam niet geparst"


def test_tussencheck_toont_alle_secties_inclusief_voorspelling():
    """Anders dan bij een analyse wordt hier niets weggefilterd; de
    voorspelling blijft staan omdat die de methode toetsbaar maakt."""
    for samenvatting in analyses_mod.get_alle_tussenchecks():
        c = analyses_mod.get_tussencheck(samenvatting["ticker"])
        labels = [s["label"] for s in c["secties"]]
        assert len(labels) >= 4, f"{c['ticker']}: {labels}"
        assert labels[0] == "Waarom", f"{c['ticker']}: begint met {labels[0]}"
        assert "Voorspelling" in labels, f"{c['ticker']}: geen voorspelling"


def test_tussencheck_paginas_renderen():
    c = _app().test_client()
    r = c.get("/tussenchecks")
    html = r.get_data(as_text=True)
    aantal = len(analyses_mod.get_alle_tussenchecks())
    assert r.status_code == 200
    assert html.count('class="check-kaart"') == aantal
    assert "geen analyses" in html.lower(), "het onderscheid met een analyse hoort op de pagina te staan"

    for samenvatting in analyses_mod.get_alle_tussenchecks():
        r = c.get(f"/tussenchecks/{samenvatting['ticker']}")
        assert r.status_code == 200, samenvatting["ticker"]
        assert "Beslisdocument, geen analyse" in r.get_data(as_text=True)

    assert c.get("/tussenchecks/sfg").status_code == 200, "hoofdletterongevoelig"
    assert c.get("/tussenchecks/ZZZ").status_code == 404


def test_analyses_en_tussenchecks_delen_de_cache_zonder_botsing():
    """Beide sets worden op volledig pad gecachet; een ticker die in
    allebei voorkomt mag niet het verkeerde document opleveren."""
    analyse = analyses_mod.get_analyse("WKL")
    check = analyses_mod.get_tussencheck("SFG")
    assert analyse["ticker"] == "WKL" and len(analyse["secties"]) == 15
    assert check["ticker"] == "SFG" and check["oordeel"] == "OVERSLAAN"
    # Nogmaals, nu uit de cache.
    assert analyses_mod.get_analyse("WKL")["naam"] == analyse["naam"]
    assert analyses_mod.get_tussencheck("SFG")["naam"] == check["naam"]


def test_geen_nieuwe_globals_in_de_gedeelde_js_scope():
    """base.html deelt zijn JavaScript-scope met de pagina-templates."""
    html = _app().test_client().get("/analyses/WKL").get_data(as_text=True)
    eigen = [s for s in re.findall(r"<script>(.*?)</script>", html, re.S)
             if "IntersectionObserver" in s]
    assert eigen, "scrollspy-script niet gevonden"
    for script in eigen:
        assert "(function ()" in script, "hoort in een IIFE te staan"
        assert not re.search(r"^\s*function\s+\w+", script, re.M), "globale functiedeclaratie"


if __name__ == "__main__":
    mislukt = 0
    for naam, functie in sorted(globals().items()):
        if not naam.startswith("test_") or not callable(functie):
            continue
        try:
            functie()
            print(f"  v {naam}")
        except AssertionError as e:
            mislukt += 1
            print(f"  x {naam}\n      {e}")
    print("\nAlles goed." if not mislukt else f"\n{mislukt} test(s) mislukt.")
    sys.exit(1 if mislukt else 0)
