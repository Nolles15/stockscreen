"""De scorebord-pagina en de onderzoekskop moeten renderen zonder database.

Twee dingen worden hier bewaakt:

1. **De verhuizing naar `engine/`.** `.dockerignore` sluit `scripts/` uit van
   de build, dus een route kan daar niet uit importeren. Zou de rekenlogica
   ooit terugglijden naar `scripts/`, dan werkt de pagina lokaal prima en
   ontbreekt hij in productie — zonder foutmelding, net als destijds met de
   analyses in `data/`.
2. **De blinde vlek blijft zichtbaar.** Het getal dat ertoe doet is hoeveel
   OVERSLAAN-oordelen nooit tegen een analyse zijn gehouden. Als die zin uit
   de pagina verdwijnt, meet het scorebord alleen nog zijn eigen successen.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jinja2 import Environment, FileSystemLoader

from engine import scorebord as kern

WORTEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _omgeving(pad: str = "/scorebord") -> Environment:
    env = Environment(loader=FileSystemLoader(os.path.join(WORTEL, "templates")))
    env.globals["url_for"] = lambda *a, **k: "#"
    env.globals["get_flashed_messages"] = lambda *a, **k: []
    env.globals["request"] = type("Verzoek", (), {"path": pad})()
    env.filters["pct"] = lambda w: "—" if w is None else f"{w:+.1f}".replace(".", ",") + "%"
    env.filters["bedrag"] = lambda w: "—" if w is None else f"{w:,.2f}"
    env.filters["datum_nl"] = lambda w: w or "onbekend"
    return env


def test_kern_zit_in_engine_niet_in_scripts():
    """De pagina importeert uit engine/; scripts/ is niet in het image."""
    pad = os.path.join(WORTEL, "engine", "scorebord.py")
    assert os.path.exists(pad), "engine/scorebord.py ontbreekt — de route breekt in productie"
    for functie in ("verzamel", "kalibratie", "blinde_vlek", "met_koersen",
                    "pijplijn_overzicht", "rendement_per_groep"):
        assert hasattr(kern, functie), f"engine.scorebord mist {functie}"
    print("  [OK] alle zes functies staan in engine/scorebord.py")


def test_scorebord_rendert():
    regels = kern.verzamel()
    html = _omgeving().get_template("scorebord.html").render(
        regels=regels,
        kalibratie=kern.kalibratie(regels),
        vlek=kern.blinde_vlek(regels),
        pijplijn=kern.pijplijn_overzicht(),
        subtab="scorebord",
        config={},
    )
    assert len(html) > 3000, "pagina is verdacht kort"
    assert "Kalibratie" in html
    assert "Wat je meet, en wat je niet meet" in html
    assert "vals negatieven zijn onzichtbaar" in html, (
        "de blinde vlek moet met zoveel woorden op de pagina staan"
    )
    print(f"  [OK] scorebord rendert ({len(html)} tekens) met de blinde vlek erin")


def test_scorebord_bevat_geen_javascript():
    """Bewust scriptloos: base.html en de pagina delen één globale JS-scope, en
    die valkuil heeft eerder een hele pagina onuitgevoerd gelaten."""
    tekst = open(os.path.join(WORTEL, "templates", "scorebord.html"),
                 encoding="utf-8").read()
    assert "<script" not in tekst.lower()
    print("  [OK] scorebord.html bevat geen eigen script")


def test_onderzoekskop_op_alle_drie_de_paginas():
    """Analyses, tussenchecks en scorebord delen dezelfde kop met subtabs."""
    pijplijn = kern.pijplijn_overzicht()
    from engine import analyses as am

    gevallen = [
        ("scorebord.html", "/scorebord", dict(
            regels=kern.verzamel(),
            kalibratie=kern.kalibratie(kern.verzamel()),
            vlek=kern.blinde_vlek(kern.verzamel()), subtab="scorebord")),
        ("analyses.html", "/analyses", dict(
            analyses=[dict(a, nu_koers=None, nu_verschil_pct=None,
                           nu_upside_pct=None, nu_datum=None)
                      for a in am.get_all_summaries()],
            logo_token="x", subtab="analyses")),
        ("tussenchecks.html", "/tussenchecks", dict(
            checks=am.get_alle_tussenchecks(), subtab="tussenchecks")),
    ]
    for naam, pad, ctx in gevallen:
        html = _omgeving(pad).get_template(naam).render(
            pijplijn=pijplijn, config={}, **ctx)
        assert "onderzoek-tabs" in html, f"{naam} mist de subtabs"
        assert "/scorebord" in html, f"{naam} linkt niet naar het scorebord"
    print("  [OK] alle drie de onderzoekspagina's dragen de kop met subtabs")


def test_pijplijn_overzicht_telt():
    p = kern.pijplijn_overzicht()
    assert p["n_analyses"] > 0 and p["n_tussenchecks"] > 0
    assert set(p["verdeling_tc"]) == {"VERDIEPEN", "TWIJFEL", "OVERSLAAN"}
    assert set(p["verdeling_an"]) == {"KOOP", "HOLD", "PASS"}
    assert sum(p["verdeling_tc"].values()) == p["n_tussenchecks"]
    print(f"  [OK] overzicht telt {p['n_analyses']} analyses en "
          f"{p['n_tussenchecks']} tussenchecks, verdelingen sluiten")


if __name__ == "__main__":
    test_kern_zit_in_engine_niet_in_scripts()
    test_scorebord_rendert()
    test_scorebord_bevat_geen_javascript()
    test_onderzoekskop_op_alle_drie_de_paginas()
    test_pijplijn_overzicht_telt()
    print("\nAlle tests scorebord-pagina geslaagd.")
