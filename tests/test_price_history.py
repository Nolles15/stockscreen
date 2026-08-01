"""Koershistorie: de reeks die niet terug te halen is als Yahoo ermee stopt.

Bewaakt twee dingen die stil fout kunnen gaan: de pence-correctie voor Londen
(zonder deling staat er een reeks in het archief die 100x afwijkt van de
cijfers) en het overslaan van beursvakanties.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine.refresh as R  # noqa: E402


def _frame(kolommen: dict, index) -> pd.DataFrame:
    return pd.DataFrame(kolommen, index=pd.to_datetime(index))


def test_enkele_ticker():
    frame = _frame({"Close": [10.0, 10.5, 11.0]},
                   ["2026-07-29", "2026-07-30", "2026-07-31"])
    rijen = R._closes_uit_frame(frame, "ASML.AS", multi=False)
    assert rijen == [
        ("ASML.AS", "2026-07-29", 10.0),
        ("ASML.AS", "2026-07-30", 10.5),
        ("ASML.AS", "2026-07-31", 11.0),
    ], rijen
    print("  [OK] alle handelsdagen uit een enkelvoudig frame")


def test_lege_dagen_vallen_weg():
    frame = _frame({"Close": [10.0, float("nan"), 11.0]},
                   ["2026-07-29", "2026-07-30", "2026-07-31"])
    rijen = R._closes_uit_frame(frame, "ASML.AS", multi=False)
    assert [r[1] for r in rijen] == ["2026-07-29", "2026-07-31"], rijen
    print("  [OK] beursvakanties leveren geen lege rijen op")


def test_londen_in_ponden():
    # Yahoo geeft Londense koersen in pence; de jaarcijfers staan in ponden.
    frame = _frame({"Close": [1500.0, 1600.0]}, ["2026-07-30", "2026-07-31"])
    rijen = R._closes_uit_frame(frame, "SHEL.L", multi=False)
    assert [r[2] for r in rijen] == [15.0, 16.0], rijen
    print("  [OK] Londense koersen worden naar ponden omgerekend")


def test_multi_ticker_frame():
    kolommen = pd.MultiIndex.from_tuples(
        [("AAA.AS", "Close"), ("BBB.AS", "Close")])
    frame = pd.DataFrame([[1.0, 2.0], [1.5, 2.5]],
                         index=pd.to_datetime(["2026-07-30", "2026-07-31"]),
                         columns=kolommen)
    assert R._closes_uit_frame(frame, "AAA.AS", multi=True) == [
        ("AAA.AS", "2026-07-30", 1.0), ("AAA.AS", "2026-07-31", 1.5)]
    # Een ticker die niet in het antwoord zit mag geen fout geven, wel niets
    # opleveren — anders sneuvelt een hele chunk op één ontbrekend symbool.
    assert R._closes_uit_frame(frame, "WEG.AS", multi=True) == []
    print("  [OK] bulk-frame per ticker, ontbrekend symbool is geen fout")


def test_onzin_wordt_geweigerd():
    frame = _frame({"Close": [0.0, -3.0]}, ["2026-07-30", "2026-07-31"])
    assert R._closes_uit_frame(frame, "AAA.AS", multi=False) == []
    print("  [OK] nul- en negatieve koersen komen het archief niet in")


if __name__ == "__main__":
    test_enkele_ticker()
    test_lege_dagen_vallen_weg()
    test_londen_in_ponden()
    test_multi_ticker_frame()
    test_onzin_wordt_geweigerd()
    print("\nAlle tests koershistorie geslaagd.")
