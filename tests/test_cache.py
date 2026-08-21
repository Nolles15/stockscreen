"""
Doet de cache wat hij belooft, en loopt hij leeg als het moet?

Aanleiding: Neon rekent af per gigabyte verkeer, en de dashboardquery van ruim
twee megabyte draaide bij élke paginalading. Een cache die niet werkt kost geld;
een cache die niet leegloopt kost vertrouwen, want dan kijk je naar oude cijfers
zonder dat je het merkt. Beide moeten dus getest.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import cache

fout = 0
tellingen = {"n": 0}


def duur_ophalen():
    tellingen["n"] += 1
    return [{"ticker": "TEST.AS", "ronde": tellingen["n"]}]


c = cache.Houder("test", ttl_seconden=60)

# Tien keer vragen hoort één keer op te halen — dat is de hele bedoeling.
for _ in range(10):
    c.haal(duur_ophalen)
ok = tellingen["n"] == 1
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] tien aanvragen, {tellingen['n']} keer opgehaald (verwacht 1)")

ok = c.stand()["treffers"] == 9 and c.stand()["missers"] == 1
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] tellers kloppen: {c.stand()['treffers']} treffers, {c.stand()['missers']} misser")

# Leegmaken moet een verse ophaling afdwingen; anders zie je na een
# herberekening nog de oude cijfers.
c.leeg("test")
c.haal(duur_ophalen)
ok = tellingen["n"] == 2
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] na leegmaken wordt opnieuw opgehaald")

# De TTL is de terugval voor een vergeten invalidatie.
kort = cache.Houder("kort", ttl_seconden=0.2)
tellingen["n"] = 0
kort.haal(duur_ophalen)
time.sleep(0.3)
kort.haal(duur_ophalen)
ok = tellingen["n"] == 2
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] verlopen cache haalt opnieuw op ({tellingen['n']}x)")

# Gelijktijdige aanvragen mogen niet allemaal tegelijk gaan ophalen — dat is
# precies het verkeer dat we wilden vermijden.
import threading  # noqa: E402

traag = cache.Houder("traag", ttl_seconden=60)
tellingen["n"] = 0


def traag_ophalen():
    tellingen["n"] += 1
    time.sleep(0.15)
    return ["waarde"]


draden = [threading.Thread(target=lambda: traag.haal(traag_ophalen)) for _ in range(8)]
for d in draden:
    d.start()
for d in draden:
    d.join()
ok = tellingen["n"] == 1
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] acht gelijktijdige aanvragen, {tellingen['n']} keer opgehaald")

# De cache mag nooit een lege lijst als "gevuld" beschouwen bij een storing —
# dan zou een mislukte query een leeg dashboard vastzetten voor een kwartier.
leeg = cache.Houder("leeg", ttl_seconden=60)
tellingen["n"] = 0
leeg.haal(lambda: (tellingen.__setitem__("n", tellingen["n"] + 1), [])[1])
leeg.haal(lambda: (tellingen.__setitem__("n", tellingen["n"] + 1), [])[1])
ok = tellingen["n"] == 2
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] een leeg resultaat wordt niet vastgehouden ({tellingen['n']}x opgehaald)")

print("\nFALEND:", fout)
sys.exit(1 if fout else 0)
