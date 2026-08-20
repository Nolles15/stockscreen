"""
Herkent de module hetzelfde bedrijf onder meerdere tickers, en houdt hij
aandelenklassen apart van dubbele noteringen?

Dat onderscheid is het punt: bij een dubbele notering is er één ticker te veel,
bij aandelenklassen zijn het echt verschillende stukken en is het aan Janco.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import dubbelingen


def s(ticker, naam, isin=None):
    return {"ticker": ticker, "name": naam, "isin": isin}


universe = [
    s("SFG.WA",    "AS Silvano Fashion Group"),
    s("SFG1T.TL",  "AS Silvano Fashion Group"),
    s("VOLV-A.ST", "AB Volvo"),
    s("VOLV-B.ST", "AB Volvo"),
    s("ASML",      "ASML Holding N.V."),
    s("ASML.AS",   "ASML Holding NV"),
    s("8TRA.DE",   "Traton SE", "DE000TRAT0N7"),
    s("8TRA.ST",   "TRATON SE", "DE000TRAT0N7"),
    s("RWAY.MI",   "Rai Way S.p.A."),
    s("KPN.AS",    "Koninklijke KPN N.V."),
]

gevonden = dubbelingen.vind(universe)

verwacht = [
    ("SFG.WA",    ["SFG1T.TL"],  "dubbele notering"),
    ("SFG1T.TL",  ["SFG.WA"],    "dubbele notering"),
    ("VOLV-A.ST", ["VOLV-B.ST"], "aandelenklasse"),
    ("ASML",      ["ASML.AS"],   "dubbele notering"),
    ("8TRA.DE",   ["8TRA.ST"],   "dubbele notering"),
]

fout = 0
for ticker, andere, soort in verwacht:
    g = gevonden.get(ticker)
    ok = g is not None and g["dubbel_van"] == andere and g["dubbel_soort"] == soort
    fout += not ok
    print(f"  [{'OK ' if ok else 'FOUT'}] {ticker:11s} -> {g}")

# Losse aandelen mogen niet als dubbeling worden aangemerkt.
for ticker in ("RWAY.MI", "KPN.AS"):
    ok = ticker not in gevonden
    fout += not ok
    print(f"  [{'OK ' if ok else 'FOUT'}] {ticker:11s} terecht niet gemarkeerd")

# ASML matcht op naam ondanks 'N.V.' tegen 'NV'; Traton matcht op ISIN.
ok = gevonden["ASML"]["dubbel_bewijs"] == "naam" and gevonden["8TRA.DE"]["dubbel_bewijs"] == "isin"
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] bewijssoort klopt (naam voor ASML, isin voor Traton)")

print("\nFALEND:", fout)
sys.exit(1 if fout else 0)
