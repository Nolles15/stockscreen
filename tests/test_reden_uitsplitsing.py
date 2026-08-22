"""
Zegt het label wat er werkelijk aan de hand is?

Aanleiding: op 21 augustus 2026 droegen 176 tickers het label "FACTOR >10",
terwijl er maar 19 zo'n afwijking hadden. De rest waren 79 eenheidsfouten en 70
holdings met negatieve omzet. Drie problemen onder één noemer, waarvan er één
helemaal geen probleem is — een holding met negatieve omzet is net zo min kapot
als een verlieslatend bedrijf, het model kan er alleen niets mee.

Deze test bewaakt dat die drie uit elkaar blijven, want zolang ze hetzelfde
heten kun je van geen enkele markt vaststellen of hij werkt.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import data_quality

fout = 0


def reden(issues, oob=False, status="bad"):
    return data_quality.classify_signal_reason(
        "INSUFFICIENT DATA", status, issues, oob, 1)


# Een eenheidsfout is een echte fout: de cijfers staan in pence, de koers in
# ponden. Die hoort opgezocht en gerepareerd te worden.
r = reden(["Market cap SEVERE mismatch: 0.01x van berekende waarde"])
ok = r["reason_code"] == "eenheidsfout" and r["reason_label"] == "EENHEIDSFOUT"
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] eenheidsfout heet eenheidsfout: {r['reason_label']}")

# Negatieve omzet is geen fout maar een bedrijfstype. Oranje, net als de
# verlieslatende bedrijven, want het is dezelfde soort grens.
r = reden(["Omzet FY2024 is negatief (-12.4M)"])
ok = (r["reason_code"] == "negatieve_omzet"
      and r["reason_label"] == "NEGATIEVE OMZET"
      and r["reason_color"] == "amber")
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] negatieve omzet apart, en oranje: "
      f"{r['reason_label']} ({r['reason_color']})")

# En de echte factor-10 blijft bestaan — die gaat over de verhouding tussen
# modelwaarde en koers, niet over de onderliggende cijfers.
r = reden([], oob=True, status="ok")
ok = r["reason_code"] == "databug" and r["reason_label"] == "FACTOR >10"
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] echte factor-10 blijft apart: {r['reason_label']}")

# De drie mogen nooit meer op één hoop: als twee verschillende oorzaken hetzelfde
# label krijgen, is het label geen informatie meer.
labels = {reden(["Market cap SEVERE mismatch: x"])["reason_label"],
          reden(["Omzet FY2024 is negatief"])["reason_label"],
          reden([], oob=True, status="ok")["reason_label"]}
ok = len(labels) == 3
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] drie oorzaken, drie labels: {len(labels)}")

# Elke code moet een label én een kleur hebben, anders valt de badge terug op
# niets en zie je een lege chip in het dashboard.
ontbreekt = [k for k in data_quality._REASON_LABELS
             if k not in data_quality._REASON_COLORS]
ok = not ontbreekt
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] elke reden heeft een kleur"
      f"{' — mist: ' + str(ontbreekt) if ontbreekt else ''}")

# Elke blocker die naar een reden verwijst moet die reden ook kennen.
onbekend = [v for v in data_quality._BLOCKER_TO_REASON.values()
            if v not in data_quality._REASON_LABELS]
ok = not onbekend
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] geen blocker wijst naar een onbekende reden"
      f"{' — ' + str(onbekend) if onbekend else ''}")

print("\nFALEND:", fout)
sys.exit(1 if fout else 0)
