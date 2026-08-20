"""
Rekent de verkoopregels correct om naar bedragen?

Dit is de vraag waar Janco om vroeg — "wat is dan de verkoopprijs?" — en het
antwoord is een bedrag waar hij naar handelt. Een fout hier is duurder dan een
fout in de weergave, dus de omrekening staat apart getest.

De grenzen zelf komen uit de bestaande regels; deze functie verzint er geen.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import exit_regels


def regel(rid, waarden, geraakt=False, hard=False):
    return {"id": rid, "waarden": waarden, "geraakt": geraakt, "hard": hard}


# Adyen zoals het er nu voor staat: analyse aanwezig, koers boven de kansgewogen
# waarde maar ruim onder het optimistische scenario.
rij = {"price": 1054.60, "currency": "EUR", "combined_fv": 1129.0}
verkoop = {"regels": [
    regel("C1", {"kansgewogen": 1008.64, "koers": 1054.60}, geraakt=True),
    regel("C2", {"optimistisch": 1414.04, "koers": 1054.60}, geraakt=False, hard=True),
    regel("A2", {"grens_pct": 175, "koers_vs_fv_pct": 93.4}, geraakt=False, hard=True),
    regel("B3", {"nu": 5.0, "bij_vastleggen": 5.0}, geraakt=False),
]}

d = exit_regels.verkoopdrempels(rij, verkoop)
fout = 0

ok = [x["id"] for x in d] == ["C1", "C2", "A2"]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] volgorde laagste grens eerst: {[x['id'] for x in d]}")

c1 = next(x for x in d if x["id"] == "C1")
ok = c1["grens"] == 1008.64 and c1["geraakt"] and c1["afstand_pct"] < 0
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] kansgewogen {c1['grens']} al voorbij ({c1['afstand_pct']}%)")

c2 = next(x for x in d if x["id"] == "C2")
ok = c2["grens"] == 1414.04 and c2["bron"] == "analyse" and c2["afstand_pct"] == 34.1
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] verkopen boven {c2['grens']} — nog {c2['afstand_pct']}% te gaan")

a2 = next(x for x in d if x["id"] == "A2")
ok = a2["grens"] == round(1129.0 * 1.75, 2) and a2["bron"] == "model"
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] modelgrens {a2['grens']} = 175% van de fair value")

ok = not any(x["id"].startswith("B") for x in d)
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] these-regels krijgen geen prijs (die vraag gaat niet over prijs)")

# Zonder analyse blijft alleen het model over — de stand voor tien van de elf.
zonder = exit_regels.verkoopdrempels(
    {"price": 201.04, "currency": "USD", "combined_fv": 51.79},
    {"regels": [regel("A2", {"grens_pct": 175}, geraakt=True, hard=True)]})
ok = len(zonder) == 1 and zonder[0]["bron"] == "model" and zonder[0]["geraakt"]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] zonder analyse alleen de modelgrens: {zonder[0]['grens']} USD, al geraakt")

# Geen koers betekent geen uitspraak; liever niets dan een verzonnen bedrag.
ok = exit_regels.verkoopdrempels({"price": None, "currency": "EUR"}, verkoop) == []
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] zonder koers geen drempels")

print("\nFALEND:", fout)
sys.exit(1 if fout else 0)
