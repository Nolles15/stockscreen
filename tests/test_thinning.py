"""Verdunnen van de koershistorie: welke koersen mogen weg en welke nooit.

De SQL draait in Postgres en is hier niet uit te voeren zonder database. Wat
hier wél getest wordt is de selectieregel zelf, nagebouwd in Python volgens
dezelfde definitie als de query: alles binnen de grens blijft, daarbuiten
overleeft de laatste handelsdag van elke ISO-week, en het eerste en laatste punt
van een reeks blijven altijd staan.

Die regel is het risico. Te ruim en de tabel blijft groeien; te streng en de
cyclustest in het moat-profiel gaat andere uitkomsten geven omdat er dalen uit
de reeks verdwijnen.
"""
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def selecteer_te_verwijderen(datums: list[str], grens: str) -> set[str]:
    """Python-tweeling van de DELETE-selectie in db.thin_price_history()."""
    if not datums:
        return set()
    eerste, laatste = min(datums), max(datums)
    oud = [d for d in datums if d < grens]

    # Per ISO-week de nieuwste dag behouden (plek_in_week = 1 bij ORDER BY date DESC)
    per_week: dict[tuple[int, int], list[str]] = {}
    for d in oud:
        jaar, week, _ = date.fromisoformat(d).isocalendar()
        per_week.setdefault((jaar, week), []).append(d)

    weg = set()
    for dagen in per_week.values():
        for d in sorted(dagen, reverse=True)[1:]:
            weg.add(d)
    return {d for d in weg if d != eerste and d != laatste}


def _reeks(vanaf: date, dagen: int) -> list[str]:
    """Handelsdagen: maandag t/m vrijdag, zoals een echte koersreeks."""
    uit = []
    d = vanaf
    for _ in range(dagen):
        if d.weekday() < 5:
            uit.append(d.isoformat())
        d += timedelta(days=1)
    return uit


def test_recente_koersen_blijven_allemaal():
    vandaag = date(2026, 8, 2)
    grens = (vandaag - timedelta(days=730)).isoformat()
    datums = _reeks(vandaag - timedelta(days=1825), 1825)

    weg = selecteer_te_verwijderen(datums, grens)
    binnen_grens = [d for d in datums if d >= grens]
    assert not (weg & set(binnen_grens)), "er verdwijnt een koers binnen de tweejaarsgrens"
    print(f"  [OK] alle {len(binnen_grens)} koersen binnen twee jaar blijven staan")


def test_buiten_de_grens_een_koers_per_week():
    vandaag = date(2026, 8, 2)
    grens = (vandaag - timedelta(days=730)).isoformat()
    datums = _reeks(vandaag - timedelta(days=1825), 1825)

    over = [d for d in datums if d not in selecteer_te_verwijderen(datums, grens)]
    oud_over = [d for d in over if d < grens]

    weken = {}
    for d in oud_over:
        jaar, week, _ = date.fromisoformat(d).isocalendar()
        weken.setdefault((jaar, week), []).append(d)
    teveel = {w: v for w, v in weken.items() if len(v) > 1}
    # Alleen de week van het eerste punt mag twee koersen houden: dat punt is
    # expliciet uitgezonderd zodat het begin van de reeks nooit sneuvelt.
    assert len(teveel) <= 1, f"meer dan één week met meerdere koersen: {list(teveel)[:3]}"
    print(f"  [OK] buiten de grens blijft {len(oud_over)} van {len([d for d in datums if d < grens])} "
          f"koersen over, ruwweg één per week")


def test_eerste_en_laatste_blijven():
    vandaag = date(2026, 8, 2)
    grens = (vandaag - timedelta(days=730)).isoformat()
    datums = _reeks(vandaag - timedelta(days=1825), 1825)

    weg = selecteer_te_verwijderen(datums, grens)
    assert min(datums) not in weg, "het begin van de reeks mag nooit verdwijnen"
    assert max(datums) not in weg, "de meest recente koers mag nooit verdwijnen"
    print("  [OK] begin en einde van de reeks blijven altijd staan")


def test_idempotent():
    """Tweede keer draaien mag niets meer verwijderen."""
    vandaag = date(2026, 8, 2)
    grens = (vandaag - timedelta(days=730)).isoformat()
    datums = _reeks(vandaag - timedelta(days=1825), 1825)

    na_eerste = [d for d in datums if d not in selecteer_te_verwijderen(datums, grens)]
    tweede_ronde = selecteer_te_verwijderen(na_eerste, grens)
    assert not tweede_ronde, f"tweede ronde wil er nog {len(tweede_ronde)} weggooien"
    print("  [OK] tweede ronde verwijdert niets meer")


def test_besparing_is_de_moeite():
    """Levert het genoeg op om het te doen? Anders is het onnodig risico."""
    vandaag = date(2026, 8, 2)
    grens = (vandaag - timedelta(days=730)).isoformat()
    datums = _reeks(vandaag - timedelta(days=1825), 1825)

    weg = selecteer_te_verwijderen(datums, grens)
    over = len(datums) - len(weg)
    besparing = len(weg) / len(datums)
    assert besparing > 0.35, f"maar {besparing:.0%} besparing — de moeite niet waard"
    print(f"  [OK] vijf jaar gaat van {len(datums)} naar {over} regels "
          f"({besparing:.0%} minder)")


def test_korte_reeks_blijft_ongemoeid():
    """Een aandeel dat pas een jaar noteert mag niets kwijtraken."""
    vandaag = date(2026, 8, 2)
    grens = (vandaag - timedelta(days=730)).isoformat()
    datums = _reeks(vandaag - timedelta(days=300), 300)
    assert not selecteer_te_verwijderen(datums, grens)
    print("  [OK] reeks korter dan twee jaar blijft volledig intact")


if __name__ == "__main__":
    test_recente_koersen_blijven_allemaal()
    test_buiten_de_grens_een_koers_per_week()
    test_eerste_en_laatste_blijven()
    test_idempotent()
    test_besparing_is_de_moeite()
    test_korte_reeks_blijft_ongemoeid()
    print("\nAlle tests verdunnen geslaagd.")
