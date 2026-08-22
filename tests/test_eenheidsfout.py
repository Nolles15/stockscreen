"""
Blokkeert de market-cap-toets nog het juiste, en het juiste alleen?

Aanleiding: op 22 augustus 2026 droegen 85 tickers de blokkade "EENHEIDSFOUT".
Gemeten bleek geen enkele daarvan een schaalfout: geen enkele verhouding lag in
de buurt van factor 100. Wat er wel stond waren Roche-certificaten, Schindler,
Alphabet (GOOGL, factor 2,08), Duitse preferente lijnen, Merck KGaA, negen
regionale Credit-Agricole-banken en 39 Zweedse A/B-lijnen.

Bij zo'n notering telt Yahoo het hele bedrijf terwijl `shares_outstanding` een
enkele aandelenklasse telt. Die verhouding kan elk getal zijn. Een schaalfout
kan dat niet: die is per definitie een macht van tien. Daarop toetsen we nu.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import data_quality

fout = 0


def beoordeel(shares, price, market_cap):
    """Eén evaluatie met alleen de market-cap-toets in beeld."""
    jaar = [{"fiscal_year": 2025, "revenue": 1000.0, "ebit": 100.0, "ebitda": 120.0,
             "net_income": 80.0, "eps_diluted": 2.0, "operating_cf": 90.0,
             "fcf": 70.0, "total_equity": 500.0, "total_debt": 100.0,
             "shares_outstanding": shares}
            for _ in range(1)]
    jaar += [dict(jaar[0], fiscal_year=j) for j in (2024, 2023, 2022)]
    return data_quality.evaluate(
        "TEST.ST", jaar,
        {"price": price, "market_cap": market_cap},
        {"quote_type": "EQUITY", "currency": "SEK", "financial_currency": "SEK"},
        fetch_success=True,
    )


def melding(res, fragment):
    return any(fragment in i for i in (res.get("issues") or []))


# --- Een echte schaalfout blijft blokkeren -----------------------------------
#
# Aandelen in duizenden, of pence naast ponden: dan is de verhouding een macht
# van tien. Dat is het enige patroon dat een eenheidsfout kán aannemen.
for factor, naam in [(100.0, "pence tegen ponden"), (1000.0, "aandelen in duizenden"),
                     (10.0, "factor tien")]:
    r = beoordeel(shares=1_000_000, price=100.0, market_cap=100_000_000 * factor)
    ok = melding(r, "SEVERE mismatch") and r["data_status"] == "bad"
    fout += not ok
    print(f"  [{'OK ' if ok else 'FOUT'}] {naam} (factor {factor:g}) blokkeert nog steeds")

# --- Een aandelenklasse blokkeert niet meer ----------------------------------
#
# GOOGL stond op 2,08 en SSAB-A op 3,37 — geen van beide is een macht van tien.
for factor, naam in [(2.08, "Alphabet"), (3.37, "SSAB-A"), (37.76, "CAT-A"),
                     (169.0, "CORE-D")]:
    r = beoordeel(shares=1_000_000, price=100.0, market_cap=int(100_000_000 * factor))
    geblokkeerd = melding(r, "SEVERE mismatch")
    ok = (not geblokkeerd) and melding(r, "deels genoteerd")
    fout += not ok
    print(f"  [{'OK ' if ok else 'FOUT'}] {naam} (factor {factor}) is een aantekening, "
          f"geen blokkade")

# De aantekening mag de ticker niet alsnog via een andere weg blokkeren.
r = beoordeel(shares=1_000_000, price=100.0, market_cap=337_000_000)
ok = r["data_status"] in ("ok", "warning")
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] status blijft bruikbaar: {r['data_status']}")

# En hij moet als niet-blokkerend herkend worden, anders wijst het dashboard
# hem alsnog als primaire oorzaak aan.
blok = data_quality.classify_blockers(r.get("issues"), r["data_status"])
ok = blok["primary_blocker"] != "unit_mismatch_severe"
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] niet de primaire blocker: {blok['primary_blocker']}")

# --- De kleine afwijking blijft wat hij was ----------------------------------
r = beoordeel(shares=1_000_000, price=100.0, market_cap=130_000_000)
ok = melding(r, "Market cap inconsistent")
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] verschil van 30% blijft een lichte melding")

# --- Speling, want koersen bewegen tussen twee ophaalmomenten ----------------
r = beoordeel(shares=1_000_000, price=100.0, market_cap=10_400_000_000)  # 104x
ok = melding(r, "SEVERE mismatch")
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] factor 104 telt nog als pence-fout (10% speling)")

print("\nFALEND:", fout)
sys.exit(1 if fout else 0)
