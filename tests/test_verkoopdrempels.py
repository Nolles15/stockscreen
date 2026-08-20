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

# Met een analyse wijkt de modelgrens: die zou als derde regel verschijnen met
# het label "geen analyse" terwijl die er juist wel is.
ok = [x["id"] for x in d] == ["C1", "C2"]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] met analyse alleen de analyse-grenzen: {[x['id'] for x in d]}")

c1 = next(x for x in d if x["id"] == "C1")
ok = c1["grens"] == 1008.64 and c1["geraakt"] and c1["afstand_pct"] < 0
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] kansgewogen {c1['grens']} al voorbij ({c1['afstand_pct']}%)")

c2 = next(x for x in d if x["id"] == "C2")
ok = c2["grens"] == 1414.04 and c2["bron"] == "analyse" and c2["afstand_pct"] == 34.1
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] verkopen boven {c2['grens']} — nog {c2['afstand_pct']}% te gaan")

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


# --- De datapoort mag een eigen analyse niet wegdrukken -----------------------
#
# Puig had een volledige analyse met scenario's van 9,58 tot 27,31 en kreeg toch
# "geen oordeel over verkopen", omdat de screener geen fair value had. Precies
# andersom: waar het model het niet aankan is de eigen analyse het enige
# bruikbare oordeel.

analyse = {"valuta": "EUR", "fair_value_kansgewogen": 18.13,
           "scenarios": {"pessimistisch": 9.58, "basis": 19.58, "optimistisch": 27.31}}
geblokkeerd = {"ticker": "PUIG.MC", "price": 17.12, "currency": "EUR",
               "data_status": "missing", "combined_fv": None}

uitslag = exit_regels.toets(geblokkeerd, analyse=analyse,
                            oordeel={"via_naam": False}, config={})
d2 = exit_regels.verkoopdrempels(geblokkeerd, uitslag)

ok = uitslag["niveau"] != "grijs" and uitslag["aantal_getoetst"] > 0
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] analyse overleeft de datapoort: {uitslag['kop']}")

ok = [x["grens"] for x in d2] == [18.13, 27.31]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] grenzen uit het rapport: {[x['grens'] for x in d2]}")

ok = "Alleen je eigen analyse" in (uitslag["toelichting"] or "")
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] toelichting zegt dat alleen de analyse is getoetst")

print("\nFALEND (totaal):", fout)
sys.exit(1 if fout else 0)
