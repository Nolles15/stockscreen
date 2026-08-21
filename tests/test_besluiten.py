"""
Herkent het systeem een stille beslissing, en houdt het de juiste dingen buiten beeld?

Het gaat hier om één ding: een oordeel waar nooit iets mee gedaan is moet
zichtbaar worden, maar zonder vals alarm. Een aandeel dat je bezit is afgehandeld
ook al legde je nooit een keuze vast, een tussencheck met OVERSLAAN vraagt niet om
een daad, en een oordeel van gisteren mag wel zichtbaar zijn maar heet nog geen
verzuim.
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
# Zichtbaar zodra het oordeel er ligt: ACN (40 dagen) en WTN (1 dag), met de
# oudste bovenaan. EVO valt af omdat je het bezit, PAY omdat het is afgesloten.
ok = tickers == ["ACN", "WTN.WA"]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] openstaand = {tickers} (verwacht ['ACN', 'WTN.WA'])")

kloof = besluiten.actiekloof([])
v = kloof["per_oordeel"]["VERDIEPEN"]
ok = v["totaal"] == 3 and v["gehandeld"] == 1 and v["stil"] == 1 and v["vers"] == 1
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] VERDIEPEN: {v['gehandeld']}/{v['totaal']} gehandeld, "
      f"{v['stil']} stil, {v['vers']} nog vers (die van gisteren telt niet als verzuim)")

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
    {"ticker": "D.AS", "currency": "EUR",
     "oordeel": {"oordeel": "HOLD", "soort": "analyse", "datum": dagen_terug(30)}},
    {"ticker": "C.AS", "currency": "EUR"},
])
vast = [v["ticker"] for v in nep.vastgelegd]
ok = vast == ["A.AS", "D.AS"]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] vastgelegd = {vast} "
      f"(volledige analyse telt ook bij HOLD; tussencheck-OVERSLAAN niet)")

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


# --- Bezitsbesluiten horen apart geteld te worden -----------------------------
#
# Bij een bezit staat de regelcode (A2, C2) in het oordeelveld. Als groepsnaam
# zegt dat niets, en "gehouden" is daar een daad en geen verzuim: bij een
# geraakte harde regel is houden een besluit, geen gewoonte.

nep2 = NepDB([
    {"ticker": "CRWD", "oordeel": "C2", "aanleiding": "bezit",
     "datum_oordeel": dagen_terug(5), "keuze": "gehouden"},
    {"ticker": "DIS", "oordeel": "A2", "aanleiding": "bezit",
     "datum_oordeel": dagen_terug(40), "keuze": None},
], bezit=set())
besluiten.db = nep2
k = besluiten.actiekloof([])
vak = k["per_oordeel"].get("Bezit — harde regel geraakt")

ok = vak is not None
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] bezitsbesluiten staan onder een leesbare naam")

if vak:
    ok = vak["totaal"] == 2 and vak["gehandeld"] == 1 and vak["stil"] == 1
    fout += not ok
    print(f"  [{'OK ' if ok else 'FOUT'}] gehouden telt als daad: "
          f"{vak['gehandeld']} gehandeld, {vak['stil']} blijven liggen")

ok = "C2" not in k["per_oordeel"] and "A2" not in k["per_oordeel"]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] geen regelcodes als groepsnaam")

print("\nFALEND:", fout)
sys.exit(1 if fout else 0)
