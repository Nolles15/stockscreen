"""
Controleert wanneer een ticker wel en niet op non-actief mag.

Dit is de gevoeligste logica in het project: te streng en er verdwijnen goede
aandelen uit beeld (dat gebeurde in april 2026 met 115 stuks), te soepel en
dode symbolen blijven eeuwig rotatiecapaciteit opsouperen.
"""

import os
import sys
import types
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine.refresh as R


def _dagen_geleden(n):
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat()


class NepDB:
    def __init__(self, fouten, eerste_fout, jaarrijen):
        self.dq = {"consecutive_failures": fouten, "first_failure_at": eerste_fout}
        self.jaarrijen = jaarrijen
        self.gesuspendeerd = False

    def get_data_quality(self, t):
        return self.dq

    def count_annual_rows(self, t):
        return self.jaarrijen

    def upsert_stock(self, t, **f):
        if f.get("active") == 0:
            self.gesuspendeerd = True

    def log_activity(self, *a, **k):
        pass


gevallen = [
    # (omschrijving, fouten, eerste fout, jaarrijen, mag suspenderen)
    ("te weinig mislukkingen",          2, _dagen_geleden(60), 0, False),
    ("reeks nog te kort",               5, _dagen_geleden(10), 0, False),
    ("heeft wel jaarcijfers",          20, _dagen_geleden(90), 4, False),
    ("startdatum onbekend",             9, None,               0, False),
    ("structureel leeg, lang genoeg",   3, _dagen_geleden(25), 0, True),
    ("ruim over alle drempels",        12, _dagen_geleden(120), 0, True),
]

fout = 0
for naam, fouten, eerste, rijen, verwacht in gevallen:
    db = NepDB(fouten, eerste, rijen)
    R.db = db
    R.screener = types.SimpleNamespace()
    got = R.maybe_auto_suspend("TEST.AS")
    ok = got == verwacht and db.gesuspendeerd == verwacht
    fout += not ok
    print(f"  [{'OK ' if ok else 'FOUT'}] {naam}: suspendeert={got} (verwacht {verwacht})")

print(f"\ndrempels: {R.SUSPEND_MIN_FAILURES} mislukkingen, {R.SUSPEND_MIN_DAYS} dagen, "
      f"archief na {R.DELISTED_AFTER_DAYS} dagen")
print("FALEND:", fout)
sys.exit(1 if fout else 0)
