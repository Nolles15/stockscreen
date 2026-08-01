"""Parsers voor de beurslijsten: bewaakt dat een gewijzigd bronformaat opvalt.

De bronnen leveren HTML/CSV/XLSX die zonder aankondiging kan veranderen. Als
een parser stilletjes nul rijen teruggeeft, lijkt "geen nieuwe tickers" een
geruststellende uitkomst terwijl we in werkelijkheid blind zijn.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import import_tickers as I  # noqa: E402

# --- Euronext: CSV met ;-scheiding, metadataregels bovenaan -----------------

EURONEXT_CSV = """Name;ISIN;Symbol;Market;Currency;Open Price
"European Equities"
"01 Aug 2026"
"2020 BULKERS";BMG9156K1018;2020;"Oslo Børs";NOK;3.64
"AB SCIENCE";FR0010557264;AB;"Euronext Paris";EUR;1.20
"AB SCIENCE BSA";FR001400ZRT0;ABBS;"Euronext Paris";EUR;0.01
"ABO GROUP";BE0974278104;ABO;"Euronext Growth Brussels";EUR;2.50
"ADYEN";NL0012969182;ADYEN;"Euronext Amsterdam";EUR;1400.0
"""


def test_euronext():
    recs = I.parse_euronext(EURONEXT_CSV)
    by_ticker = {r["ticker"]: r for r in recs}

    # Oslo Børs -> .OL (het label begint niet met "Euronext")
    assert "2020.OL" in by_ticker, "Oslo-notering niet herkend"
    assert by_ticker["2020.OL"]["currency"] == "NOK"
    assert by_ticker["ADYEN.AS"]["market"] == "NL"

    # Warrants zijn geen aandelen en horen niet in de screener
    assert "ABBS.PA" not in by_ticker, "warrant (BSA) had gefilterd moeten worden"

    # Growth-segment wordt gelabeld, niet weggegooid: main() filtert erop
    assert by_ticker["ABO.BR"]["segment"] == "growth"
    assert by_ticker["AB.PA"]["segment"] == "regulated"
    print(f"  [OK] Euronext: {len(recs)} records, warrant gefilterd, Oslo herkend")


# --- GPW: HTML waarin de grootste namen géén haakjes-notatie hebben ---------

GPW_HTML = """
<a href="/notowania/11-BIT-STUDIOS">11B (11BIT)</a>
<a href="/notowania/PZU">PZU</a>
<a href="/notowania/CD-PROJEKT">CDR (CDPROJEKT)</a>
<a href="/notowania/nieuws">Lees meer over de beurs</a>
"""


def test_gpw():
    recs = I.fetch_gpw_from_html(GPW_HTML)
    tickers = {r["ticker"] for r in recs}

    # Dit was de bug: labels zonder "(NAAM)" werden overgeslagen, en dat zijn
    # juist de grootste Poolse namen (PZU, PGE, LPP).
    assert "PZU.WA" in tickers, "ticker zonder haakjes-notatie gemist"
    assert {"11B.WA", "CDR.WA"} <= tickers
    assert len(recs) == 3, f"vrije tekst als ticker geparsed: {tickers}"
    print(f"  [OK] GPW: {len(recs)} records, PZU-vorm zonder haakjes herkend")


# --- ISIN-dedupe: dezelfde onderneming op twee beurzen ----------------------

def test_dedupe_isin():
    recs = [
        {"ticker": "PUBLI.OL", "isin": "SE0028799411", "name": "Publicis", "market": "NO",
         "currency": "NOK", "sector": None, "segment": "regulated"},
        {"ticker": "PUBLI.ST", "isin": "SE0028799411", "name": "Publicis", "market": "SE",
         "currency": "SEK", "sector": None, "segment": "regulated"},
    ]
    kept, dropped = I.dedupe_isin(recs)

    # Een Zweedse ISIN hoort bij de Stockholmse notering; daar heeft yfinance
    # de beste dekking.
    assert len(kept) == 1 and kept[0]["ticker"] == "PUBLI.ST", kept
    assert dropped == ["PUBLI.OL"]

    # Zonder ISIN kunnen we niets afleiden: alles behouden.
    geen_isin = [
        {"ticker": "AAA.WA", "isin": None, "name": "A", "market": "PL",
         "currency": "PLN", "sector": None, "segment": "regulated"},
        {"ticker": "BBB.WA", "isin": None, "name": "B", "market": "PL",
         "currency": "PLN", "sector": None, "segment": "regulated"},
    ]
    kept2, dropped2 = I.dedupe_isin(geen_isin)
    assert len(kept2) == 2 and not dropped2
    print("  [OK] ISIN-dedupe: thuisland wint, records zonder ISIN blijven staan")


# --- Ongeldige symbolen worden geweigerd ------------------------------------

def test_make_record():
    assert I.make_record("ASML.AS", "ASML", "NL0010273215") is not None
    assert I.make_record("ASML", "geen suffix", None) is None
    assert I.make_record("FOO.XX", "onbekende beurs", None) is None
    assert I.make_record("NASDAQ:ICLR", "malformed", None) is None
    print("  [OK] make_record weigert symbolen zonder bruikbaar beurssuffix")


if __name__ == "__main__":
    test_euronext()
    test_gpw()
    test_dedupe_isin()
    test_make_record()
    print("\nAlle parser-tests geslaagd.")
