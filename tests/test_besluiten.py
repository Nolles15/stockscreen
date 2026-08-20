"""
Herkent het systeem een stille beslissing, en houdt het de juiste dingen buiten beeld?

Het gaat hier om één ding: een oordeel waar nooit iets mee gedaan is moet
zichtbaar worden, maar zonder vals alarm. Een aandeel dat je bezit is afgehandeld
ook al legde je nooit een keuze vast, een oordeel van gisteren is nog geen
verzuim, en OVERSLAAN vraagt niet om een daad.
"""

import os
import sys
import types
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import besluiten


def dagen_terug(n):
    return (date.today() - timedelta(days=n)).isoformat()


class NepDB:
    def __init__(self, regels, bezit):
        self.regels = regels
        self._bezit = bezit
        self.vastgelegd = []

    def bezit_lijst(self):
        return [{"ticker": t} for t in self._bezit]

    def besluiten_lijst(self, alleen_open=False):
        if alleen_open:
            return [r for r in self.regels if not r.get("keuze")]
        return list(self.regels)

    def besluit_vastleggen(self, **kw):
        self.vastgelegd.append(kw)


regels = [
    # oud oordeel, nooit iets mee gedaan -> moet openstaan
    {"ticker": "ACN", "oordeel": "VERDIEPEN", "datum_oordeel": dagen_terug(40), "keuze": None},
    # net zo oud, maar je bezit het -> afgehandeld
    {"ticker": "EVO.ST", "oordeel": "VERDIEPEN", "datum_oordeel": dagen_terug(40), "keuze": None},
    # van gisteren -> nog geen verzuim
    {"ticker": "WTN.WA", "oordeel": "VERDIEPEN", "datum_oordeel": dagen_terug(1), "keuze": None},
    # bewust afgesloten -> niet meer open
    {"ticker": "PAY.PA", "oordeel": "KOOP", "datum_oordeel": dagen_terug(60),
     "keuze": "bewust_niet", "reden": "te weinig van de markt begrepen"},
]

nep = NepDB(regels, bezit={"EVO.ST"})
besluiten.db = nep

fout = 0
open_lijst = besluiten.openstaand([])
tickers = [b["ticker"] for b in open_lijst]
ok = tickers == ["ACN"]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] openstaand = {tickers} (verwacht ['ACN'])")

kloof = besluiten.actiekloof([])
v = kloof["per_oordeel"]["VERDIEPEN"]
ok = v["totaal"] == 3 and v["gehandeld"] == 1 and v["stil"] == 2
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] VERDIEPEN: {v['gehandeld']}/{v['totaal']} gehandeld, {v['stil']} stil")

k = kloof["per_oordeel"]["KOOP"]
ok = k["bewust_niet"] == 1 and k["stil"] == 0
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] bewust afgeslagen telt niet als stil")

# synchroniseer legt alleen oordelen vast die om een daad vragen
nep.vastgelegd.clear()
besluiten.synchroniseer([
    {"ticker": "A.AS", "currency": "EUR",
     "oordeel": {"oordeel": "VERDIEPEN", "soort": "tussencheck",
                 "datum": dagen_terug(30), "koers_toen": 10.0, "valuta_toen": "EUR"}},
    {"ticker": "B.AS", "currency": "EUR",
     "oordeel": {"oordeel": "OVERSLAAN", "soort": "tussencheck", "datum": dagen_terug(30)}},
    {"ticker": "C.AS", "currency": "EUR"},
])
vast = [v["ticker"] for v in nep.vastgelegd]
ok = vast == ["A.AS"]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] vastgelegd = {vast} (OVERSLAAN en zonder oordeel overgeslagen)")

ok = nep.vastgelegd and nep.vastgelegd[0]["koers_toen"] == 10.0
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] koers van toen meegenomen als ijkpunt")

# these voor de koers, en korte periodes gelden niet als uitslag
verloop = besluiten.sinds_het_oordeel(
    {"datum_oordeel": dagen_terug(20), "koers_toen": 100.0,
     "these_snapshot": {"quality_score": 8.0, "roic_mediaan": 20.0}},
    {"price": 110.0, "quality_score": 9.0, "roic_mediaan": 18.0})
ok = (verloop["these"]["kwaliteit"] == 1.0 and verloop["these"]["roic"] == -2.0
      and verloop["koers"]["rendement_pct"] == 10.0
      and verloop["koers"]["betekenisvol"] is False)
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] these {verloop['these']}, koers {verloop['koers']['rendement_pct']}% "
      f"(betekenisvol: {verloop['koers']['betekenisvol']})")

print("\nFALEND:", fout)
sys.exit(1 if fout else 0)
