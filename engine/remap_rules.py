"""
Bekende dual-listings en secondary→primary mappings.

Gebruikt door `POST /api/stocks` om de gebruiker te waarschuwen als hij een
secondary listing toevoegt terwijl er een primary beschikbaar is met betere
yfinance-coverage.

Onderhouden handmatig — voeg mapping toe als je een dual-listing tegenkomt
waarbij yfinance alleen de primary goed ondersteunt.
"""

# Secondary → (primary, reden)
REMAP_RULES: dict[str, tuple[str, str]] = {
    "EXOR.AS":     ("EXO.MI",    "Primary listing op Borsa Italiana"),
    "AFKL.AS":     ("AF.PA",     "Air France-KLM primary op Euronext Parijs"),
    "ACOMO.BR":    ("ACOMO.AS",  "Primary op Euronext Amsterdam"),
    "INPOST.AS":   ("INPST.AS",  "yfinance gebruikt INPST zonder punt"),
    "NASDAQ:ICLR": ("ICLR",      "Malformed prefix; correcte symbool is ICLR"),
    "BAM.AS":      ("BMT.AS",    "Royal BAM Group hernoemd naar BMT ticker"),
    "URW.AS":      ("URW.PA",    "Unibail-Rodamco-Westfield primary op Parijs"),
    "MHG.OL":      ("MOWI.OL",   "Marine Harvest hernoemd naar Mowi"),
    "HOLN.DE":     ("HOLN.SW",   "Holcim primary op SIX Swiss"),
    "ADS.DE":      ("ADDYY",     "Adidas; yfinance .DE data inconsistent"),
    # Gevonden in juli 2026: stonden als dubbeling in de watchlist onder een beurs
    # waar yfinance niets voor teruggeeft, terwijl hetzelfde bedrijf onder de
    # primaire notering wel gewoon werkte.
    "WDP.AS":      ("WDP.BR",    "Warehouses De Pauw primary op Euronext Brussel"),
    "BPOST.AS":    ("BPOST.BR",  "bpost primary op Euronext Brussel"),
    "REN.LS":      ("REN.AS",    "yfinance levert alleen data op de .AS-notering"),
    "AI.AS":       ("AI.PA",     "Air Liquide primary op Euronext Parijs"),
}


def lookup(ticker: str) -> tuple[str, str] | None:
    """Returns (primary_ticker, reason) als ticker een bekende secondary is, anders None."""
    return REMAP_RULES.get(ticker.upper())


def is_secondary(ticker: str) -> bool:
    return ticker.upper() in REMAP_RULES
