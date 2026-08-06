"""De EV-consistentiecheck moet met de échte kaspositie rekenen.

Aanleiding: de check las `total_cash`, een veld dat in geen enkele jaarrij
wordt gevuld — er staat `net_cash`. De kaspositie telde daardoor altijd als
nul en de impliciete bedrijfswaarde viel structureel te hoog uit. Gemeten op
6 augustus 2026 droeg ruim een derde van de meldingen daardoor geen informatie:
SDG.PA stond op factor 2,26 (werkelijk 1,02) en INF.PA op 1,59 (werkelijk
1,03). Beide zijn in tussenchecks als aandachtspunt geciteerd.

De valse melding is duurder dan hij lijkt: hij zaait twijfel over een cijfer
dat gewoon klopt, en die twijfel belandt in een onderzoeksdocument.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import data_quality


def _jaarrij(**velden):
    """Complete jaarrij zodat de andere controles geen ruis geven."""
    rij = {
        "fiscal_year": 2025, "revenue": 500e6, "ebit": 60e6, "ebitda": 80e6,
        "net_income": 40e6, "eps_diluted": 2.0, "operating_cf": 70e6,
        "capex": -10e6, "fcf": 60e6, "total_assets": 400e6,
        "total_equity": 200e6, "total_debt": 120e6, "shares_outstanding": 20e6,
        "net_cash": -10e6, "gross_profit": 200e6, "interest_expense": 4e6,
    }
    rij.update(velden)
    return rij


def _issues(net_cash, ev, total_debt=120e6, market_cap=100e6):
    jaren = [_jaarrij(net_cash=net_cash, total_debt=total_debt),
             _jaarrij(fiscal_year=2024, net_cash=net_cash, total_debt=total_debt),
             _jaarrij(fiscal_year=2023, net_cash=net_cash, total_debt=total_debt)]
    resultaat = data_quality.evaluate(
        "TEST.AS", jaren,
        {"price": 5.0, "market_cap": market_cap, "enterprise_value": ev},
        {"currency": "EUR", "financial_currency": "EUR", "quote_type": "EQUITY"},
        fetch_success=True, fetched_date="2026-08-06",
    )
    return [i for i in (resultaat.get("issues") or []) if "EV inconsistent" in i]


def test_geen_valse_melding_bij_kloppende_ev():
    """Het SDG-geval: schuld 120, kas 110, dus net_cash −10.

    Beurswaarde 100 + nettoschuld 10 = 110, en dat is precies wat Yahoo geeft.
    De oude formule negeerde de kas en kwam op 220 — factor 2,0, melding.
    """
    assert _issues(net_cash=-10e6, ev=110e6) == []


def test_wel_een_melding_als_de_ev_echt_niet_klopt():
    """Bewijs dat de test scherp staat: bij een werkelijk afwijkende EV blijft
    de melding staan."""
    gevonden = _issues(net_cash=-10e6, ev=300e6)
    assert gevonden, "een EV die factor 2,7 afwijkt hoort wél gemeld te worden"


def test_terugval_op_schuld_als_net_cash_ontbreekt():
    """Zonder `net_cash` blijft de oude benadering over: beurswaarde plus
    bruto schuld. Beter dan niets meten, maar wel te hoog — vandaar dat het
    alleen de terugval is."""
    assert _issues(net_cash=None, ev=220e6) == []
    assert _issues(net_cash=None, ev=110e6), "zonder kaspositie is 110 wél afwijkend"


def test_label_is_geen_databug_meer():
    """Een verschil van meer dan een factor tien is meestal een schaalfout,
    maar bij een koersval van 90%+ is het de werkelijkheid. 'DATABUG' zou die
    tweede lezing uitsluiten."""
    reden = data_quality.classify_signal_reason(
        "INSUFFICIENT DATA", "ok", [], True, 3,
    )
    assert reden["reason_code"] == "databug"
    assert reden["reason_label"] == "FACTOR >10"


if __name__ == "__main__":
    test_geen_valse_melding_bij_kloppende_ev()
    print("  [OK] kloppende EV geeft geen melding meer")
    test_wel_een_melding_als_de_ev_echt_niet_klopt()
    print("  [OK] een echt afwijkende EV wordt nog steeds gemeld")
    test_terugval_op_schuld_als_net_cash_ontbreekt()
    print("  [OK] terugval op bruto schuld werkt zonder net_cash")
    test_label_is_geen_databug_meer()
    print("  [OK] label is FACTOR >10, code blijft databug")
    print("\nAlle tests EV-check geslaagd.")
