"""De bedrijfswaarde moet met de koers meebewegen (A10).

Aanleiding: de koersronde verving `price`, `market_cap` en `pe_ttm`, maar liet
`enterprise_value` staan tot de volgende jaarcijferronde — en die komt pas na
gemiddeld elf dagen langs (250 per nacht over ~2.760 tickers). De
EV-consistentiecheck in data_quality legde daardoor een verse beurswaarde naast
een oude EV en meldde "EV inconsistent" bij wat gewoon een koersbeweging was.

Dat is dezelfde soort valse melding als de `total_cash`-bug uit
tests/test_ev_check.py, en hij is even duur: twijfel zaaien over een cijfer dat
klopt, en die twijfel belandt in een onderzoeksdocument.

De reparatie telt het koersverschil bij de opgeslagen EV op in plaats van EV
opnieuw af te leiden uit mcap + nettoschuld. Dat onderscheid is de kern: bij
opnieuw afleiden toetst de consistentiecheck zijn eigen uitkomst en is hij
betekenisloos geworden. Deze test bewaakt allebei.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import data_quality


def _jaarrij(**velden):
    rij = {
        "fiscal_year": 2025, "revenue": 500e6, "ebit": 60e6, "ebitda": 80e6,
        "net_income": 40e6, "eps_diluted": 2.0, "operating_cf": 70e6,
        "capex": -10e6, "fcf": 60e6, "total_assets": 400e6,
        "total_equity": 200e6, "total_debt": 120e6, "shares_outstanding": 20e6,
        "net_cash": -100e6, "gross_profit": 200e6, "interest_expense": 4e6,
    }
    rij.update(velden)
    return rij


def _ev_melding(market_cap, ev):
    """Geeft de EV-melding terug, of None."""
    jaren = [_jaarrij(fiscal_year=j) for j in (2025, 2024, 2023)]
    res = data_quality.evaluate(
        "TEST.AS", jaren,
        {"price": market_cap / 20e6, "market_cap": market_cap, "enterprise_value": ev},
        {"ticker": "TEST.AS", "name": "Test", "sector": "Industrials",
         "currency": "EUR", "market": "NL"},
        fetch_success=True,
    )
    for issue in res.get("issues") or []:
        if "EV inconsistent" in issue:
            return issue
    return None


# Uitgangspunt: 20 mln aandelen à €25 = 500 mln beurswaarde, nettoschuld 100 mln
# (net_cash = -100 mln), dus een consistente EV van 600 mln.
AANDELEN = 20e6
OUDE_PRIJS = 25.0
OUDE_MCAP = AANDELEN * OUDE_PRIJS      # 500 mln
OUDE_EV = 600e6


def test_uitgangspunt_geeft_geen_melding():
    assert _ev_melding(OUDE_MCAP, OUDE_EV) is None


def test_koersval_zonder_meebewegende_ev_gaf_een_valse_melding():
    """De situatie vóór de reparatie: koers halveert, EV blijft staan."""
    nieuwe_mcap = AANDELEN * 12.0        # koers 25 -> 12, ruim gehalveerd
    melding = _ev_melding(nieuwe_mcap, OUDE_EV)
    assert melding is not None, (
        "zonder meebewegende EV hoort hier juist wél een melding te komen — "
        "dat was het probleem dat A10 beschrijft"
    )


def test_meebewegende_ev_haalt_die_melding_weg():
    """Met de reparatie: EV schuift mee met het koersverschil."""
    nieuwe_prijs = 12.0
    nieuwe_mcap = AANDELEN * nieuwe_prijs
    nieuwe_ev = OUDE_EV + AANDELEN * (nieuwe_prijs - OUDE_PRIJS)
    assert _ev_melding(nieuwe_mcap, nieuwe_ev) is None, (
        "een koersbeweging mag geen datafout worden"
    )


def test_een_echte_schaalfout_wordt_nog_steeds_gemeld():
    """De tegenproef — anders hebben we de controle uitgezet in plaats van
    gerepareerd. Een EV die een factor tien afwijkt is geen koersbeweging."""
    assert _ev_melding(OUDE_MCAP, OUDE_EV * 10) is not None


def test_de_check_toetst_niet_zijn_eigen_uitkomst():
    """Zou de koersronde EV opnieuw afleiden als mcap + nettoschuld, dan komt
    de check per definitie op factor 1,00 uit en meldt hij nooit meer iets.

    Deze test legt vast dat de check nog steeds een onafhankelijk getal ziet:
    een EV die precies mcap + nettoschuld is geeft geen melding, maar eentje
    die daarvan afwijkt wél — ook bij dezelfde beurswaarde.
    """
    netto_schuld = 100e6
    assert _ev_melding(OUDE_MCAP, OUDE_MCAP + netto_schuld) is None
    assert _ev_melding(OUDE_MCAP, (OUDE_MCAP + netto_schuld) * 2) is not None


if __name__ == "__main__":
    test_uitgangspunt_geeft_geen_melding()
    print("  [OK] consistente EV geeft geen melding")
    test_koersval_zonder_meebewegende_ev_gaf_een_valse_melding()
    print("  [OK] de oude situatie produceerde inderdaad een valse melding")
    test_meebewegende_ev_haalt_die_melding_weg()
    print("  [OK] meebewegende EV haalt de valse melding weg")
    test_een_echte_schaalfout_wordt_nog_steeds_gemeld()
    print("  [OK] een echte schaalfout wordt nog steeds gemeld")
    test_de_check_toetst_niet_zijn_eigen_uitkomst()
    print("  [OK] de check ziet nog een onafhankelijk getal")
    print("\nAlle tests meebewegende EV geslaagd.")
