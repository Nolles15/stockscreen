"""
Blijft de dashboardlijst slank, en houdt de losse rij dezelfde vorm?

Aanleiding: op 22 augustus 2026 was de dashboardlading 4,2 MB voor 2.812 rijen,
en 46% daarvan bestond uit vier velden die je per rij hooguit een keer bekijkt
in een tooltip: warnings, hist_relative, data_issues en fv_methods_dropped.

Die zijn eruit gehaald. Deze test bewaakt drie dingen:
  1. ze komen niet stilletjes terug (elk nieuw veld in de SELECT is gratis
     meegenomen gewicht, en dat merk je pas bij 19.000 tickers);
  2. het aantal blijft wel staan, want daaraan hangen de badges;
  3. data_issues blijft in de cache, want classify_signal_reason leidt daar
     serverzijdig de reden uit af -- pas het antwoord aan de browser mist ze.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import db

fout = 0

rij = {
    "ticker": "TEST.AS",
    "warnings": ["eerste waarschuwing", "tweede", "derde"],
    "data_issues": ["melding een", "melding twee"],
    "fv_methods_dropped": ["Graham: verlies", "EV/FCF: negatief"],
    "hist_relative": {
        "current_ev_ebitda": 69.276, "current_pb": 0.91157293, "current_pe": None,
        "ev_ebitda_pct": 4493.6, "median_ev_ebitda": 1.5080907,
        "median_pb": 0.0287837625, "median_pe": None, "pb_pct": 3067.0,
        "pe_pct": None, "years_available": 4,
    },
}

k = db._verklein_voor_lijst(dict(rij))

# 1. De zware velden zijn weg.
weg = [v for v in ("warnings", "fv_methods_dropped") if v in k]
ok = not weg
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] warnings en fv_methods_dropped zitten er niet in"
      f"{' -- nog wel: ' + str(weg) if weg else ''}")

# 2. De aantallen blijven, anders verdwijnen de badges zonder dat iemand het merkt.
ok = k["warning_count"] == 3 and k["data_issue_count"] == 2
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] aantallen blijven: {k['warning_count']} waarschuwingen, "
      f"{k['data_issue_count']} meldingen")

# 3. data_issues blijft in de rij: de reden wordt er serverzijdig uit afgeleid.
ok = k.get("data_issues") == rij["data_issues"]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] data_issues blijft beschikbaar voor de redenclassificatie")

# 4. hist_relative houdt alleen wat de badge toont, en afgerond.
hr = k["hist_relative"]
ok = set(hr) <= {"ev_ebitda_pct", "current_ev_ebitda", "median_ev_ebitda", "years_available"}
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] hist_relative houdt alleen de getoonde sleutels: {sorted(hr)}")

ok = hr["median_ev_ebitda"] == 1.508 and hr["years_available"] == 4
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] getallen afgerond: mediaan {hr['median_ev_ebitda']} "
      f"(was {rij['hist_relative']['median_ev_ebitda']})")

# 4b. Het groeiprofiel: de lijst krijgt alleen de kop en de rode codes.
groei_rij = db._verklein_voor_lijst({
    "ticker": "G.AS",
    "groei_profiel": {"niveau": "rood", "kop": "Groei wordt betaald met nieuwe aandelen",
                      "toetsen": {"T1": "rood", "T2": "groen", "T3": "geel"},
                      "verlieslatend": True, "omzet_cagr": 0.3},
})
ok = ("groei_profiel" not in groei_rij and groei_rij["groei_rood"] == ["T1"]
      and groei_rij["groei_kop"].startswith("Groei wordt") and groei_rij["groei_verlieslatend"])
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] groei_profiel blijft uit de lijst; kop en rode codes blijven")

# 5. Een lege rij mag niet omvallen.
leeg = db._verklein_voor_lijst({"ticker": "LEEG.AS"})
ok = leeg["warning_count"] == 0 and leeg["data_issue_count"] == 0
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] een rij zonder meldingen levert nullen op")

# 6. De besparing zelf. Over de hele lijst is gemeten 39% eraf; op een enkele
#    rij hangt het af van hoeveel meldingen erin staan. Onder de 35% is er iets
#    teruggeslopen.
import json  # noqa: E402
voor = len(json.dumps(rij, separators=(",", ":")))
na = len(json.dumps(k, separators=(",", ":")))
ok = na < voor * 0.65
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] rij van {voor} naar {na} tekens "
      f"({100 - round(100*na/voor)}% eraf)")

print("\nFALEND:", fout)
sys.exit(1 if fout else 0)
