"""
Kiest de server dezelfde rijen als de browser deed?

De tabbladregels zijn op 2026-08-22 uit `templates/index.html` naar
`engine/selectie.py` verhuisd, zodat het dashboard niet langer alle 2.812 rijen
hoeft te laden om er twintig te tonen. Bij zo'n verhuizing is er precies een
ding dat mis kan gaan: een regel die onderweg verschuift. Deze test legt de
regels vast zoals ze in de browser stonden.

Twee dingen krijgen extra aandacht omdat ze subtiel zijn:
  - de volgorde van filteren (land en signaal gaan voor het tabblad);
  - sorteren voor het afkappen, want andersom komt de bovenkant van de lijst
    uit een willekeurige greep in plaats van uit het geheel.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import selectie

fout = 0


def aandeel(ticker, **kv):
    basis = {
        "ticker": ticker, "name": ticker, "sector": "Industrials",
        "signal": "HOLD", "data_status": "ok", "quality_score": 6.0,
        "margin_of_safety": 10.0, "rank_score": 1.0, "market": "NL",
        "fv_confidence": "high", "price_vs_fv_pct": 90.0,
        "is_growth_lossmaker": False, "revenue_cagr": None,
        "low_quality": False, "reason_label": None, "reason_code": None,
    }
    basis.update(kv)
    return basis


UNIVERSUM = [
    aandeel("TOP.AS", rank_score=9.0, signal="STRONG BUY", margin_of_safety=40.0),
    aandeel("KOOP.AS", rank_score=7.0, signal="BUY", margin_of_safety=30.0),
    aandeel("BEST.ST", rank_score=8.0, signal="BUY", margin_of_safety=35.0),
    # Kwaliteit: score >= 8.5, en banken tellen niet mee.
    aandeel("KWAL.AS", quality_score=9.0),
    aandeel("BANK.AS", quality_score=9.5, sector="Financial Services"),
    aandeel("ZWAK.AS", quality_score=8.6, data_status="bad"),
    # Hersteller: kwaliteit 5 tot 7, minstens 30% korting, betrouwbare waardering.
    aandeel("HERSTEL.AS", quality_score=6.0, price_vs_fv_pct=65.0, margin_of_safety=35.0),
    aandeel("DUUR.AS", quality_score=6.0, price_vs_fv_pct=95.0),
    aandeel("VAAG.AS", quality_score=6.0, price_vs_fv_pct=60.0, fv_confidence="low"),
    # Groeier en twee zonder oordeel.
    aandeel("GROEI.AS", is_growth_lossmaker=True, revenue_cagr=0.4, signal="INSUFFICIENT DATA"),
    aandeel("LEEG.WA", signal="INSUFFICIENT DATA", reason_label="GEEN DATA", rank_score=None),
    aandeel("OUD.WA", signal="INSUFFICIENT DATA", reason_label="CIJFERS VEROUDERD", rank_score=None),
    aandeel("SLECHT.AS", low_quality=True),
]


def tickers(uitslag):
    return [r["ticker"] for r in uitslag["rijen"]]


# --- De tabbladen ------------------------------------------------------------

k = tickers(selectie.selecteer(UNIVERSUM, tab="kansen"))
ok = k[:3] == ["TOP.AS", "BEST.ST", "KOOP.AS"]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] Kansen op ranglijst, hoogste eerst: {k[:3]}")

k = tickers(selectie.selecteer(UNIVERSUM, tab="kwaliteit"))
ok = k == ["KWAL.AS"]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] Kwaliteit laat banken en slechte data staan: {k}")

k = tickers(selectie.selecteer(UNIVERSUM, tab="herstellers"))
ok = k == ["HERSTEL.AS"]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] Herstellers: alleen wie aan alle vijf eisen voldoet: {k}")

k = tickers(selectie.selecteer(UNIVERSUM, tab="groeiers"))
ok = k == ["GROEI.AS"]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] Groeiers: {k}")

# Gesorteerd op reden, en een rij zonder reden komt vooraan -- precies wat
# `(a.reason_label || '').localeCompare(...)` in de browser deed. In de echte
# data heeft elke rij zonder oordeel een reden; deze volgorde is dus vooral het
# bewijs dat er onderweg niets is verschoven.
k = tickers(selectie.selecteer(UNIVERSUM, tab="geenoordeel"))
ok = k == ["GROEI.AS", "OUD.WA", "LEEG.WA"]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] Geen oordeel, gesorteerd op reden: {k}")

# "Alles" verbergt lage kwaliteit tenzij je het vinkje aanzet.
zonder = tickers(selectie.selecteer(UNIVERSUM, tab="alles", extra={"toon_alles": False}))
met = tickers(selectie.selecteer(UNIVERSUM, tab="alles", extra={"toon_alles": True}))
ok = "SLECHT.AS" not in zonder and "SLECHT.AS" in met
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] lage kwaliteit alleen zichtbaar met 'toon alles' "
      f"({len(zonder)} tegen {len(met)})")

# --- Volgorde van filteren ---------------------------------------------------
#
# Dit was expliciet zo gebouwd: "Koopwaardig" op Kansen hoort de beste
# koopwaardige aandelen te geven, niet het handjevol dat van de bestaande top
# overblijft. Filteren moet dus voor het tabblad komen.

k = tickers(selectie.selecteer(UNIVERSUM, tab="kansen", signaal="koop"))
ok = k == ["TOP.AS", "BEST.ST", "KOOP.AS"]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] signaal filtert voor het tabblad: {k}")

k = tickers(selectie.selecteer(UNIVERSUM, tab="kansen", land="Zweden"))
ok = k == ["BEST.ST"]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] land filtert ook op Kansen: {k}")

k = tickers(selectie.selecteer(UNIVERSUM, tab="alles", sector="Financial Services",
                               extra={"toon_alles": True}))
ok = k == ["BANK.AS"]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] sector filtert: {k}")

# --- Sorteren voor afkappen --------------------------------------------------

groot = [aandeel(f"T{i}.AS", margin_of_safety=float(i)) for i in range(selectie.LIJST_MAX + 200)]
u = selectie.selecteer(groot, tab="alles", extra={"toon_alles": True},
                       sorteer_op="margin_of_safety", richting=-1)
ok = u["afgekapt"] and u["totaal"] == len(groot) and len(u["rijen"]) == selectie.LIJST_MAX
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] afkappen wordt gemeld: {len(u['rijen'])} van {u['totaal']}")

ok = u["rijen"][0]["margin_of_safety"] == float(len(groot) - 1)
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] de grootste korting staat bovenaan, niet een "
      f"willekeurige uit de eerste greep ({u['rijen'][0]['margin_of_safety']})")

# Lege waarden zakken in beide richtingen naar beneden.
gemengd = [aandeel("A.AS", margin_of_safety=5.0), aandeel("B.AS", margin_of_safety=None),
           aandeel("C.AS", margin_of_safety=50.0)]
for richting, verwacht in ((-1, ["C.AS", "A.AS", "B.AS"]), (1, ["A.AS", "C.AS", "B.AS"])):
    k = [r["ticker"] for r in selectie.sorteer(gemengd, "margin_of_safety", richting)]
    ok = k == verwacht
    fout += not ok
    print(f"  [{'OK ' if ok else 'FOUT'}] lege waarde onderaan bij richting {richting}: {k}")

# --- De samenvatting ---------------------------------------------------------

s = selectie.selecteer(UNIVERSUM, tab="kansen")["samenvatting"]
ok = s["totaal"] == len(UNIVERSUM)
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] de telling gaat over het hele universum ({s['totaal']}), "
      f"niet over het tabblad")

landen = {x["naam"]: x["aantal"] for x in s["landen"]}
ok = landen.get("Zweden") == 1 and landen.get("Nederland") == len(UNIVERSUM) - 3
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] landen geteld: {landen}")

s2 = selectie.selecteer(UNIVERSUM, tab="alles", land="Zweden")["samenvatting"]
ok = s2["in_land"] == 1 and s2["totaal"] == len(UNIVERSUM)
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] sector wordt binnen het land geteld "
      f"({s2['in_land']}), land over alles ({s2['totaal']})")

# --- Het middelste getal onder de tabel --------------------------------------
#
# "20 aandelen . van 406 in Polen . 2812 in totaal". Dat middelste getal telt de
# selectie voor het tabblad zijn greep doet; erna gemeten zou er "van 20" staan
# en dan zegt het niets.

u = selectie.selecteer(UNIVERSUM, tab="kansen", land="Nederland")
ok = u["in_selectie"] == 10 and len(u["rijen"]) <= 10
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] in_selectie telt voor het tabblad: {u['in_selectie']}")

u = selectie.selecteer(UNIVERSUM, tab="kansen")
ok = u["in_selectie"] == len(UNIVERSUM)
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] zonder filters is dat het hele universum: {u['in_selectie']}")


# --- Mijn lijst krijgt zijn tickers van de browser ---------------------------

k = tickers(selectie.selecteer(UNIVERSUM, tab="pins", tickers=["KWAL.AS", "TOP.AS"]))
ok = sorted(k) == ["KWAL.AS", "TOP.AS"]
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] Mijn lijst volgt de meegestuurde tickers: {k}")

k = tickers(selectie.selecteer(UNIVERSUM, tab="pins", tickers=[]))
ok = k == []
fout += not ok
print(f"  [{'OK ' if ok else 'FOUT'}] geen pins, geen rijen")

print("\nFALEND:", fout)
sys.exit(1 if fout else 0)
