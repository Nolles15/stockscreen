"""fv_confidence: eensgezindheid van methodes is geen betrouwbaarheid.

De Worldline-les (2026-08-08): WLN.PA toonde 89,9% korting bij "hoog
vertrouwen" terwijl de invoer kapot was en de winstreeks verliesjaren
bevatte. Het label mat alleen of de methodes het met elkaar eens waren —
op dezelfde kapotte invoer zegt dat niets. Deze tests bewaken de twee
extra eisen: alle drie de methodes aanwezig, en geen vervuilde winstreeks.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.valuation import combined_fair_value  # noqa: E402


def _invoer(net_incomes=(5e6, 5e6, 5e6), met_perpetuity=True):
    normalized = {"normalized_eps": 10.0}
    if met_perpetuity:
        normalized["normalized_oe_per_share"] = 10.0
    annual = [
        {"fiscal_year": 2023 + i, "shares_outstanding": 1e6,
         "total_equity": 50e6, "net_cash": 0, "net_income": ni}
        for i, ni in enumerate(net_incomes)
    ]
    return normalized, annual


def test_drie_methodes_zonder_verlies_is_high():
    normalized, annual = _invoer()
    r = combined_fair_value(normalized, [], annual, "Default", {})
    assert r["fv_methods_used"] == 3, r["fv_methods_used"]
    assert r["fv_confidence"] == "high", r["fv_confidence"]
    print("  [OK] drie eensgezinde methodes zonder verliesjaren blijven high")


def test_twee_verliesjaren_drukken_high_naar_medium():
    normalized, annual = _invoer(net_incomes=(-1e6, -2e6, 5e6))
    r = combined_fair_value(normalized, [], annual, "Default", {})
    assert r["fv_methods_used"] == 3
    assert r["fv_confidence"] == "medium", r["fv_confidence"]
    print("  [OK] twee verliesjaren: nooit hoog vertrouwen")


def test_een_incidenteel_verliesjaar_mag():
    # Eén coronajaar maakt de reeks niet onbruikbaar.
    normalized, annual = _invoer(net_incomes=(5e6, -1e6, 5e6))
    r = combined_fair_value(normalized, [], annual, "Default", {})
    assert r["fv_confidence"] == "high", r["fv_confidence"]
    print("  [OK] een enkel verliesjaar kost het label niet")


def test_twee_methodes_zijn_nooit_high():
    # Zonder perpetuity blijven multiples + Graham over; die kunnen het
    # roerend eens zijn, maar met z'n tweeen is dat geen kruisvalidatie.
    normalized, annual = _invoer(met_perpetuity=False)
    r = combined_fair_value(normalized, [], annual, "Default", {})
    assert r["fv_methods_used"] == 2, r["fv_methods_used"]
    assert r["fv_confidence"] == "medium", r["fv_confidence"]
    print("  [OK] twee methodes: hooguit medium, hoe eensgezind ook")


if __name__ == "__main__":
    test_drie_methodes_zonder_verlies_is_high()
    test_twee_verliesjaren_drukken_high_naar_medium()
    test_een_incidenteel_verliesjaar_mag()
    test_twee_methodes_zijn_nooit_high()
    print("\nAlle tests fv_confidence geslaagd.")
