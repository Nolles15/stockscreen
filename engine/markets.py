"""
Beurssuffix → land en valuta. Eén bron van waarheid.

Deze tabel stond eerder twee keer: als `SUFFIX_INFO` in `import_tickers.py` en
impliciet als een reeks if-statements in `data_fetcher._detect_market`. Die
tweede kende acht beurzen en zette al het overige op "US" — waardoor na de
uitbreiding naar 27 landen 1.062 van de 2.759 aandelen als Amerikaans in de
database stonden, van Parijs tot Helsinki. In diezelfde functie zat ook een
tikfout die nooit was opgevallen: Finland werd op `.FI` gecontroleerd terwijl
Yahoo `.HE` gebruikt, dus die tak is nooit één keer afgegaan.

Vandaar één tabel, op één plek. Komt er een beurs bij, dan is dit het enige
bestand dat verandering nodig heeft.
"""

from __future__ import annotations

# Yahoo-suffix → (valuta, landcode zoals gebruikt in `stocks.market`)
SUFFIX_INFO: dict[str, tuple[str, str]] = {
    # Euronext
    "AS": ("EUR", "NL"), "BR": ("EUR", "BE"), "PA": ("EUR", "FR"),
    "MI": ("EUR", "IT"), "LS": ("EUR", "PT"), "IR": ("EUR", "IE"),
    "OL": ("NOK", "NO"),
    # Duitstalig
    "DE": ("EUR", "DE"), "F": ("EUR", "DE"), "SW": ("CHF", "CH"),
    "VI": ("EUR", "AT"),
    # Zuid-Europa
    "MC": ("EUR", "ES"), "AT": ("EUR", "GR"),
    # Nordics en Baltische staten
    "ST": ("SEK", "SE"), "CO": ("DKK", "DK"), "HE": ("EUR", "FI"),
    "IC": ("ISK", "IS"), "TL": ("EUR", "EE"), "RG": ("EUR", "LV"),
    "VS": ("EUR", "LT"),
    # Centraal-Europa
    "WA": ("PLN", "PL"), "PR": ("CZK", "CZ"), "BD": ("HUF", "HU"),
    "RO": ("RON", "RO"),
    # Verenigd Koninkrijk — let op: koersen komen in pence binnen, zie de
    # deling door 100 in data_fetcher en refresh.
    "L": ("GBP", "UK"),
    # Buiten Europa, losse posities in de watchlist
    "AX": ("AUD", "AU"), "TO": ("CAD", "CA"),
}


def land_van(ticker: str) -> str:
    """
    Landcode bij een ticker.

    Zonder suffix is het een Amerikaanse notering — dat is de conventie bij
    Yahoo. Bij een suffix dat we niet kennen geven we het suffix zelf terug in
    plaats van terug te vallen op "US": een onbekende beurs hoort op te vallen,
    niet stilletjes als Amerikaans in de lijst te belanden. Precies dat stille
    terugvallen was de oorspronkelijke fout.
    """
    t = (ticker or "").strip().upper()
    if "." not in t:
        return "US"
    suffix = t.rsplit(".", 1)[1]
    info = SUFFIX_INFO.get(suffix)
    return info[1] if info else suffix


def valuta_van(ticker: str) -> str | None:
    """Handelsvaluta bij een ticker, of None bij een onbekend suffix."""
    t = (ticker or "").strip().upper()
    if "." not in t:
        return "USD"
    info = SUFFIX_INFO.get(t.rsplit(".", 1)[1])
    return info[0] if info else None
