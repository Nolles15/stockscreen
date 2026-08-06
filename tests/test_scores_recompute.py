"""De herberekening mag niets ophalen en bij een proefdraai niets wegschrijven.

Aanleiding: een wijziging in config.yaml (sectorprofielen, signaaldrempels,
gewichten in de fair value) landt pas in de database als de nachtelijke ronde
langs een ticker komt. Bij 90 per dag over ~2.760 aandelen is dat ruim een
maand, waarin het dashboard oude en nieuwe aannames door elkaar toont. Op
6 augustus 2026 gold dat tegelijk voor vier gewijzigde sectorprofielen.

`run_ticker` leest alles uit de database en belt Yahoo niet, dus hij ís al de
zuivere herberekening. Deze test legt de twee eigenschappen vast waar het
endpoint op leunt en die je stilzwijgend kwijt kunt raken:

1. `persist=False` schrijft niets weg — anders is de proefdraai een echte run
   en heb je het universum herrekend voordat je de uitkomst zag.
2. Er gaat geen netwerkverkeer overheen — anders is het geen herberekening
   maar een verkapte refresh, met de rate-limits van Yahoo eraan vast.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from engine import screener

CONFIG_PAD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)


def _config():
    with open(CONFIG_PAD, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _jaarrij(jaar):
    return {
        "fiscal_year": jaar, "period_type": "annual",
        "revenue": 500e6, "gross_profit": 200e6, "ebit": 60e6, "ebitda": 80e6,
        "net_income": 40e6, "eps_diluted": 2.0, "operating_cf": 70e6,
        "capex": -10e6, "fcf": 60e6, "total_assets": 400e6,
        "current_assets": 150e6, "current_liabilities": 80e6,
        "total_equity": 200e6, "total_debt": 120e6, "net_cash": -10e6,
        "shares_outstanding": 20e6, "book_value_ps": 10.0, "roe": 0.20,
        "interest_expense": 4e6, "inventory": 30e6, "net_ppe": 100e6,
        "fetched_date": "2026-08-01",
    }


class _NepDb:
    """Vervangt engine.db. Telt schrijfacties en levert vaste cijfers."""

    def __init__(self):
        self.schrijfacties = 0

    def upsert_scores(self, *a, **kw):
        self.schrijfacties += 1

    def get_financials(self, ticker, period_type):
        return [] if period_type == "ttm" else [_jaarrij(y) for y in (2025, 2024, 2023)]

    def get_market_data(self, ticker):
        return {"price": 30.0, "market_cap": 600e6, "pe_ttm": 15.0,
                "ev_ebitda_ttm": 9.0, "pb_ratio": 3.0, "enterprise_value": 720e6}

    def get_stock(self, ticker):
        return {"ticker": ticker, "name": "Testbedrijf", "sector": "Industrials",
                "currency": "EUR", "market": "NL"}

    def get_historical_multiples(self, ticker):
        return [{"fiscal_year": y, "pe_ratio": 15.0, "ev_ebitda": 9.0,
                 "pb_ratio": 3.0, "ev_fcf": 12.0} for y in (2025, 2024, 2023)]

    def get_overrides(self, ticker):
        return {}

    def get_data_quality(self, ticker):
        return {"data_status": "ok", "completeness_pct": 100.0}

    def jaarrijen_met_overrides(self, ticker):
        return [_jaarrij(y) for y in (2025, 2024, 2023)], []


def _met_nepdb(fn):
    """Draai fn met engine.db vervangen; geef de neppe db terug."""
    nep = _NepDb()
    echt = screener.db
    screener.db = nep
    try:
        return fn(), nep
    finally:
        screener.db = echt


def test_proefdraai_schrijft_niets_weg():
    resultaat, nep = _met_nepdb(
        lambda: screener.run_ticker("TEST.AS", _config(), persist=False)
    )
    assert nep.schrijfacties == 0, (
        f"persist=False schreef toch {nep.schrijfacties}x weg — dan is de "
        "proefdraai een echte run"
    )
    assert resultaat.get("combined_fv"), "de proefdraai moet wél doorrekenen"
    assert resultaat.get("signal"), "de proefdraai moet wél een signaal geven"


def test_echte_ronde_schrijft_wel_weg():
    """De tegenproef: zonder persist=False moet er precies één keer geschreven."""
    _, nep = _met_nepdb(
        lambda: screener.run_ticker("TEST.AS", _config(), persist=True)
    )
    assert nep.schrijfacties == 1, f"verwacht 1 schrijfactie, kreeg {nep.schrijfacties}"


def test_default_blijft_wegschrijven():
    """Bestaande aanroepers geven geen persist mee en moeten blijven werken."""
    _, nep = _met_nepdb(lambda: screener.run_ticker("TEST.AS", _config()))
    assert nep.schrijfacties == 1, "de standaardwaarde van persist is gewijzigd"


def test_herberekening_belt_niet_naar_buiten():
    """Geen netwerk in de rekenweg — anders is het een verkapte refresh."""
    import socket

    origineel = socket.socket.connect

    def _verboden(self, *a, **kw):
        raise AssertionError(
            "run_ticker legde een netwerkverbinding op; de herberekening hoort "
            "uitsluitend uit de database te lezen"
        )

    socket.socket.connect = _verboden
    try:
        _, nep = _met_nepdb(
            lambda: screener.run_ticker("TEST.AS", _config(), persist=False)
        )
    finally:
        socket.socket.connect = origineel
    assert nep.schrijfacties == 0


def test_proefdraai_en_echte_ronde_rekenen_hetzelfde():
    """Anders meet je met de proefdraai iets anders dan je straks wegschrijft."""
    proef, _ = _met_nepdb(
        lambda: screener.run_ticker("TEST.AS", _config(), persist=False)
    )
    echt, _ = _met_nepdb(
        lambda: screener.run_ticker("TEST.AS", _config(), persist=True)
    )
    for veld in ("combined_fv", "signal", "quality_score", "margin_of_safety"):
        assert proef.get(veld) == echt.get(veld), f"{veld} loopt uiteen"


if __name__ == "__main__":
    test_proefdraai_schrijft_niets_weg()
    print("  [OK] proefdraai rekent door en schrijft niets weg")
    test_echte_ronde_schrijft_wel_weg()
    print("  [OK] een echte ronde schrijft precies één keer weg")
    test_default_blijft_wegschrijven()
    print("  [OK] bestaande aanroepers zonder persist blijven wegschrijven")
    test_herberekening_belt_niet_naar_buiten()
    print("  [OK] geen netwerkverkeer in de rekenweg")
    test_proefdraai_en_echte_ronde_rekenen_hetzelfde()
    print("  [OK] proefdraai en echte ronde geven dezelfde uitkomst")
    print("\nAlle tests herberekening geslaagd.")
