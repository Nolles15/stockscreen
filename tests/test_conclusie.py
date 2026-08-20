"""
Trekt de bezitskaart een conclusie, en lost die de tegenspraak op?

Aanleiding: bij T-Mobile stond bovenaan "Verkoopsignaal van de screener" en
eronder dat je pas 141% hoger moet verkopen. Twee tegengestelde beweringen, en
de kaart koos niet — de vraag "wat is nu het advies?" bleef onbeantwoord.

De rangorde die getoetst wordt: these boven prijs, en je eigen analyse boven het
generieke model.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import exit_regels


def regel(rid, geraakt, hard=False):
    familie = {"A": "waardering", "B": "these", "C": "analyse"}[rid[0]]
    return {"id": rid, "familie": familie, "geraakt": geraakt, "hard": hard,
            "naam": rid, "waarden": {}}


gevallen = [
    # (omschrijving, regels, heeft analyse, verwachte kop)
    ("model roept duur, eigen analyse niet",
     [regel("A1", True), regel("C1", False), regel("C2", False)], True, "Houden"),
    ("voorbij het optimistische scenario",
     [regel("C2", True, hard=True)], True, "Overweeg verkopen"),
    ("voorbij de kansgewogen waarde, nog niet het optimistische",
     [regel("C1", True), regel("C2", False)], True, "Let op, maar nog niet verkopen"),
    ("these gebroken gaat voor op de prijs",
     [regel("B2", True), regel("C2", True, hard=True)], True, "Kijk of je these nog klopt"),
    ("duur volgens het model, geen analyse om tegen af te zetten",
     [regel("A1", True)], False, "Uitzoeken"),
    ("niets geraakt",
     [regel("A1", False), regel("B1", False)], False, "Niets te doen"),
]

fout = 0
for naam, regelset, met_analyse, verwacht in gevallen:
    c = exit_regels.conclusie({}, {"regels": regelset}, met_analyse)
    ok = c["kop"] == verwacht
    fout += not ok
    print(f"  [{'OK ' if ok else 'FOUT'}] {naam}")
    print(f"        -> {c['kop']!r} ({c['kleur']})")

# Het geval van T-Mobile zelf: de conclusie moet de tegenspraak benoemen in
# plaats van hem te laten staan.
c = exit_regels.conclusie({}, {"regels": [regel("A1", True), regel("C2", False)]}, True)
ok = "generieke model" in c["uitleg"] and "eigen analyse" in c["uitleg"]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] de tegenspraak wordt benoemd, niet verstopt")

print("\nFALEND:", fout)
sys.exit(1 if fout else 0)
