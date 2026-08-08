"""Ingeprijsde groei: de omgekeerde som (besluit Janco 2026-08-08).

Uit FV = winst / (r − g) volgt met FV = koers: g = r − winst / koers.
Geen voorspelling maar een feit over de prijs — en de reden dat groeiers
niet langer alleen "te duur" heten maar beoordeelbaar worden.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.valuation import implied_growth_pct  # noqa: E402


def test_omgekeerde_som_klopt_exact():
    # r = 10% (Default-sector), winst 10 per aandeel.
    # Koers 100 -> winstrendement 10% = r -> ingeprijsde groei 0%.
    assert implied_growth_pct({"normalized_oe_per_share": 10.0}, "Default", {}, 100.0) == 0.0
    # Koers 200 -> winstrendement 5% -> de resterende 5% moet uit groei komen.
    assert implied_growth_pct({"normalized_oe_per_share": 10.0}, "Default", {}, 200.0) == 5.0
    # Koers 50 -> winstrendement 20% -> de markt prijst 10% krimp per jaar in.
    assert implied_growth_pct({"normalized_oe_per_share": 10.0}, "Default", {}, 50.0) == -10.0
    print("  [OK] g = r - winst/koers, in beide richtingen")


def test_sector_rendementseis_telt_mee():
    config = {"sectors": {"Technology": {"required_return": 11}}}
    # Zelfde koers en winst, hogere eis -> meer groei nodig om de koers te dragen.
    assert implied_growth_pct({"normalized_oe_per_share": 10.0}, "Technology", config, 200.0) == 6.0
    print("  [OK] rendementseis van de sector bepaalt de lat")


def test_zonder_winst_geen_som():
    assert implied_growth_pct({"normalized_oe_per_share": None}, "Default", {}, 100.0) is None
    assert implied_growth_pct({"normalized_oe_per_share": -3.0}, "Default", {}, 100.0) is None
    assert implied_growth_pct({"normalized_oe_per_share": 10.0}, "Default", {}, None) is None
    print("  [OK] verlies of ontbrekende koers geeft None, geen onzin")


if __name__ == "__main__":
    test_omgekeerde_som_klopt_exact()
    test_sector_rendementseis_telt_mee()
    test_zonder_winst_geen_som()
    print("\nAlle tests ingeprijsde groei geslaagd.")
