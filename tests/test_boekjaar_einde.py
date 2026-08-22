"""
Wordt de einddatum van het boekjaar goed uit de jaarrekening gehaald?

De wachtrij kiest sinds 2026-08-22 op publicatievenster: een boekjaar dat twee
tot zeven maanden geleden eindigde kan een nieuw verslag hebben opgeleverd.
Daarvoor is de maand nodig, en die stond nergens - `_col_year` hield alleen het
jaartal over terwijl de kolomkop van Yahoo een volledige datum is.

Deze test bewaakt het lezen van die datum. Dat het de eerste bruikbare kolom
moet zijn is niet vanzelfsprekend: Yahoo levert de jaren nieuwste-eerst, dus de
eerste kolom is het nieuwste boekjaar. Pakken we per ongeluk de laatste, dan
staat elk bedrijf vijf jaar in het verleden en vuurt het venster nooit.
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
from engine import data_fetcher  # noqa: E402

fout = 0

# Een pandas-Timestamp, zoals Yahoo hem levert.
d = data_fetcher._col_datum(pd.Timestamp("2025-12-31"))
ok = d == "2025-12-31"
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] Timestamp wordt een ISO-datum: {d}")

# Een gebroken boekjaar mag niet stilletjes op december uitkomen.
d = data_fetcher._col_datum(pd.Timestamp("2026-06-30"))
ok = d == "2026-06-30"
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] gebroken boekjaar houdt zijn maand: {d}")

# Tekst met een datum ervoor komt ook voor.
d = data_fetcher._col_datum("2025-03-31 00:00:00")
ok = d == "2025-03-31"
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] tekstkolom wordt gelezen: {d}")

# Onzin levert niets op, geen halve datum.
for onzin in ("TTM", "", None, 12345):
    d = data_fetcher._col_datum(onzin)
    ok = d is None
    fout += not ok
    print(f"  [{'OK ' if ok else 'FOUT'}] {onzin!r} levert niets op: {d}")

# Het jaartal moet gelijk blijven aan wat _col_year eruit haalt, anders lopen de
# twee uit elkaar en klopt de koppeling tussen boekjaar en einddatum niet meer.
for kol in (pd.Timestamp("2025-12-31"), pd.Timestamp("2026-06-30")):
    datum = data_fetcher._col_datum(kol)
    ok = int(datum[:4]) == data_fetcher._col_year(kol)
    fout += not ok
    print(f"  [{'OK ' if ok else 'FOUT'}] jaartal komt overeen met _col_year: "
          f"{datum} tegen {data_fetcher._col_year(kol)}")

# Het venster zelf: twee tot zeven maanden na afloop.
from engine import db  # noqa: E402

ok = db.VENSTER_VANAF_MAANDEN == 2 and db.VENSTER_TOT_MAANDEN == 7
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] venster staat op {db.VENSTER_VANAF_MAANDEN} "
      f"tot {db.VENSTER_TOT_MAANDEN} maanden na het boekjaareinde")

# December-boekjaren vormen de bulk; die horen tussen maart en juli in beeld te
# komen. Even narekenen dat het venster dat werkelijk dekt.
def in_venster(einde: date, vandaag: date) -> bool:
    maanden = (vandaag.year - einde.year) * 12 + (vandaag.month - einde.month)
    return db.VENSTER_VANAF_MAANDEN <= maanden <= db.VENSTER_TOT_MAANDEN

dec = date(2025, 12, 31)
raak = [m for m in range(1, 13) if in_venster(dec, date(2026, m, 15))]
ok = raak == [2, 3, 4, 5, 6, 7]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] december-boekjaar staat in beeld in de maanden {raak}")

print("\nFALEND:", fout)
sys.exit(1 if fout else 0)
