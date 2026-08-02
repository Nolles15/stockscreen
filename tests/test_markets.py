"""Landdetectie uit het beurssuffix.

Achtergrond: de detectie kende acht beurzen en zette al het overige op "US".
Na de uitbreiding naar 27 landen stonden daardoor 1.062 van de 2.759 aandelen
als Amerikaans in de database — van Parijs tot Helsinki. In diezelfde functie
zat bovendien een tikfout die nooit was opgevallen: Finland werd gecontroleerd
op `.FI` terwijl Yahoo `.HE` gebruikt, dus die tak is nooit afgegaan.

Deze test bewaakt vooral dat een onbekend suffix niet stilletjes "US" wordt —
dat stille terugvallen was de kern van de fout.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import markets  # noqa: E402
from engine.data_fetcher import _detect_market  # noqa: E402


def test_alle_beurzen_uit_de_dataset():
    # Elk suffix dat in de watchlist voorkomt moet een land opleveren.
    verwacht = {
        "ASML.AS": "NL", "UMI.BR": "BE", "AI.PA": "FR", "SBC.MI": "IT",
        "EDP.LS": "PT", "C5H.IR": "IE", "EQNR.OL": "NO", "SAP.DE": "DE",
        "02P.F": "DE", "HOLN.SW": "CH", "TKA.VI": "AT", "SAN.MC": "ES",
        "EVO.ST": "SE", "MAERSK-A.CO": "DK", "LASTIK.HE": "FI",
        "MAREL.IC": "IS", "TXT.WA": "PL", "TEL1T.TL": "EE",
        "SAF1R.RG": "LV", "AKO1L.VS": "LT", "CEZ.PR": "CZ",
        "OTP.BD": "HU", "TLV.RO": "RO", "OPAP.AT": "GR", "SHEL.L": "UK",
    }
    for ticker, land in verwacht.items():
        assert markets.land_van(ticker) == land, f"{ticker} -> {markets.land_van(ticker)}"
    print(f"  [OK] {len(verwacht)} beurzen leveren het juiste land")


def test_finland_was_de_tikfout():
    # `.HE` is het Yahoo-suffix voor Helsinki. De oude code keek naar `.FI`.
    assert markets.land_van("LASTIK.HE") == "FI"
    assert markets.land_van("NOKIA.HE") == "FI"
    print("  [OK] Helsinki (.HE) wordt herkend, niet .FI")


def test_geen_suffix_is_amerikaans():
    # Yahoo-conventie: Amerikaanse noteringen hebben geen suffix.
    assert markets.land_van("NVDA") == "US"
    assert markets.land_van("AAPL") == "US"
    print("  [OK] tickers zonder suffix blijven US")


def test_onbekend_suffix_wordt_niet_stilletjes_us():
    # Dit is de kern van de oorspronkelijke fout: een beurs die we niet kennen
    # hoort op te vallen in de lijst, niet als Amerikaans weg te zakken.
    assert markets.land_van("IETS.XYZ") == "XYZ"
    assert markets.land_van("IETS.XYZ") != "US"
    print("  [OK] onbekend suffix valt op in plaats van weg")


def test_fetcher_gebruikt_dezelfde_tabel():
    # data_fetcher had een eigen, kleinere kopie. Die mag niet terugkomen.
    for ticker in ("LASTIK.HE", "SBC.MI", "SAN.MC", "NVDA", "ASML.AS"):
        assert _detect_market(ticker) == markets.land_van(ticker)
    print("  [OK] de fetcher leest uit dezelfde tabel")


def test_valuta():
    assert markets.valuta_van("EVO.ST") == "SEK"
    assert markets.valuta_van("TXT.WA") == "PLN"
    assert markets.valuta_van("SHEL.L") == "GBP"
    assert markets.valuta_van("NVDA") == "USD"
    assert markets.valuta_van("IETS.XYZ") is None
    print("  [OK] valuta per beurs, None bij onbekend suffix")


if __name__ == "__main__":
    test_alle_beurzen_uit_de_dataset()
    test_finland_was_de_tikfout()
    test_geen_suffix_is_amerikaans()
    test_onbekend_suffix_wordt_niet_stilletjes_us()
    test_fetcher_gebruikt_dezelfde_tabel()
    test_valuta()
    print("\nAlle tests landdetectie geslaagd.")
