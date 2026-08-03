"""Een boekjaar dat alleen uit handmatige cijfers bestaat mag de pagina niet slopen.

Aanleiding: na het uploaden van FY2025 voor LASTIK.HE gaf /stock/LASTIK.HE een
500. De oorzaak zat niet in de CSV maar in de rij die de app er zelf bij maakt
voor een jaar dat nog niet uit Yahoo komt: die bevatte alléén `fiscal_year`.

In Jinja is een ontbrekende sleutel geen None maar `Undefined`, en
`Undefined is not none` is waar. De opmaakmacro liet zo'n waarde dus door naar
`val | abs` en dat is een TypeError — die de hele pagina omgooit.

Daarom rendert deze test de échte template. Een test op de losse velden had dit
niet gezien; het gaat juist om wat Jinja met een ontbrekende sleutel doet.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jinja2 import Environment, FileSystemLoader

from engine import db

WORTEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _omgeving() -> Environment:
    env = Environment(loader=FileSystemLoader(os.path.join(WORTEL, "templates")))
    # De template draait normaal onder Flask; die twee globals horen daarbij.
    env.globals["url_for"] = lambda *a, **k: "#"
    env.globals["get_flashed_messages"] = lambda *a, **k: []
    env.globals["request"] = type("Verzoek", (), {"path": "/stock/LASTIK.HE"})()
    return env


# Zoals db.get_scores() hem teruggeeft: alle kolommen aanwezig, ook de lege.
# Een uitgeklede versie zou hier een Undefined-fout geven die niets met deze
# test te maken heeft.
_SCORE_KOLOMMEN = (
    "quality_score", "quality_breakdown", "piotroski_score", "piotroski_breakdown",
    "normalized_eps", "normalized_ebitda", "normalized_fcf", "normalized_owner_earn",
    "multiples_fv", "graham_fv", "perpetuity_fv", "combined_fv", "conservative_fv",
    "base_fv", "optimistic_fv", "fv_confidence", "fv_spread_pct", "fv_methods_used",
    "signal", "margin_of_safety", "warnings", "last_calculated", "accruals_ratio",
    "hist_relative", "fv_methods_dropped", "revenue_cagr",
)


def _scores() -> dict:
    s = {k: None for k in _SCORE_KOLOMMEN}
    s.update({
        "signal": "STRONG BUY", "quality_score": 9.0, "piotroski_score": 7,
        "combined_fv": 15.4, "conservative_fv": 13.67, "base_fv": 15.4,
        "optimistic_fv": 17.70, "margin_of_safety": 56.4, "warnings": [],
        "quality_breakdown": {}, "piotroski_breakdown": {}, "hist_relative": {},
        "fv_methods_dropped": [], "fv_methods_used": 4, "fv_confidence": "high",
    })
    return s


def _context(annual: list[dict]) -> dict:
    return {
        "ticker": "LASTIK.HE",
        "stock": {"name": "Lassila & Tikanoja Plc", "sector": "Industrials",
                  "market": "Helsinki", "currency": "EUR", "description": None},
        "annual": annual,
        "ttm": None,
        "market": {"price": 6.75, "last_updated": "2026-08-03", "market_cap": 2.6e8},
        "scores": _scores(),
        "overrides": [{"field": "revenue", "year": 2025, "value": 4.1e8, "note": "jaarverslag"}],
        "override_set": {"revenue:2025"},
        "hist_mult": [],
        "config": {},
        "latest_fy": 2025,
        "fy_lag": 0,
        "verwacht_fy": 2025,
    }


def _echte_jaarrij(fy: int) -> dict:
    rij = db.lege_jaarrij(fy)
    rij.update({"revenue": 4.239e8, "ebit": 4.0e7, "ebitda": 6.0e7,
                "net_income": 2.69e7, "eps_diluted": 0.70, "total_equity": 1.8e8,
                "fetched_date": "2026-08-03"})
    return rij


def test_alle_velden_aanwezig_in_lege_jaarrij():
    rij = db.lege_jaarrij(2025)
    for veld in db.FINANCIAL_VELDEN:
        assert veld in rij, f"'{veld}' ontbreekt — Jinja maakt daar Undefined van"
    assert rij["fiscal_year"] == 2025
    print(f"  [OK] lege jaarrij heeft alle {len(db.FINANCIAL_VELDEN)} velden + fiscal_year")


def test_pagina_rendert_met_handmatig_boekjaar():
    """Het geval dat de 500 gaf: FY2025 bestaat alleen als override."""
    annual = [db.lege_jaarrij(2025), _echte_jaarrij(2024), _echte_jaarrij(2023)]
    annual[0]["revenue"] = 4.1e8   # één veld uit de CSV, de rest leeg

    html = _omgeving().get_template("stock.html").render(**_context(annual))
    assert "2025" in html
    assert len(html) > 5000, "pagina is verdacht kort"
    print(f"  [OK] pagina rendert met een handmatig FY2025 ({len(html)} tekens)")


def test_pagina_rendert_met_volledig_leeg_handmatig_jaar():
    """Zelfs een override-jaar zonder één ingevuld cijfer mag niet omvallen."""
    annual = [db.lege_jaarrij(2025), _echte_jaarrij(2024)]
    html = _omgeving().get_template("stock.html").render(**_context(annual))
    assert len(html) > 5000
    print("  [OK] pagina rendert ook met een volledig leeg handmatig jaar")


def test_oude_manier_zou_zijn_omgevallen():
    """Bewijs dat de test scherp staat: de oude rij gooit de render wél om."""
    annual = [{"fiscal_year": 2025}, _echte_jaarrij(2024)]
    try:
        _omgeving().get_template("stock.html").render(**_context(annual))
    except TypeError as e:
        print(f"  [OK] oude rij faalt nog steeds zoals verwacht: {e}")
        return
    raise AssertionError(
        "de oude, kale rij rendert nu ook — dan bewaakt deze test niets meer "
        "en moet hij herzien worden"
    )


if __name__ == "__main__":
    test_alle_velden_aanwezig_in_lege_jaarrij()
    test_pagina_rendert_met_handmatig_boekjaar()
    test_pagina_rendert_met_volledig_leeg_handmatig_jaar()
    test_oude_manier_zou_zijn_omgevallen()
    print("\nAlle tests handmatig boekjaar geslaagd.")
