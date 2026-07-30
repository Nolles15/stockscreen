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
}


def lookup(ticker: str) -> tuple[str, str] | None:
    """Returns (primary_ticker, reason) als ticker een bekende secondary is, anders None."""
    return REMAP_RULES.get(ticker.upper())


def is_secondary(ticker: str) -> bool:
    return ticker.upper() in REMAP_RULES
