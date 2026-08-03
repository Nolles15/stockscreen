"""Jaarcijfers die achterlopen op de kalender moeten zichtbaar worden.

Aanleiding: LASTIK.HE stond in augustus 2026 als STRONG BUY in Kansen met het
predicaat "cijfers compleet", terwijl het nieuwste jaarverslag FY2024 was. De
oorzaak was dat `completeness_pct` telt hoe goed de opgeslagen jaren gevuld
zijn, en `freshness_days` telt wanneer we Yahoo voor het laatst gebeld hebben —
maar niemand keek of het nieuwste boekjaar nog van deze tijd was.

Het risico van de reparatie zit in de kalendergrens. Te streng en de halve
database kleurt in januari geel omdat FY(vorig jaar) nog niet gepubliceerd is;
te ruim en precies het geval waar het om begonnen was glipt er weer doorheen.
Die grens is wat hier getest wordt.
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import data_quality as dq


def _rij(fy: int, **extra) -> dict:
    """Een volledig gevulde jaarrij — zodat completeness nooit de oorzaak is."""
    rij = {
        "fiscal_year": fy,
        "revenue": 400_000_000.0, "ebit": 40_000_000.0, "ebitda": 60_000_000.0,
        "net_income": 27_000_000.0, "eps_diluted": 0.70, "operating_cf": 50_000_000.0,
        "fcf": 30_000_000.0, "total_equity": 180_000_000.0, "total_debt": 90_000_000.0,
        "shares_outstanding": 38_000_000.0,
    }
    rij.update(extra)
    return rij


def _markt() -> dict:
    # shares × price = 38M × 6.75 ≈ 257M, gelijk aan market_cap: geen unit-mismatch.
    return {"price": 6.75, "market_cap": 256_500_000.0}


def test_grens_ligt_op_juli():
    """Vanaf juli mag je vorig boekjaar verwachten, daarvoor niet."""
    assert dq.verwacht_boekjaar(date(2026, 7, 1)) == 2025
    assert dq.verwacht_boekjaar(date(2026, 12, 31)) == 2025
    assert dq.verwacht_boekjaar(date(2026, 6, 30)) == 2024
    assert dq.verwacht_boekjaar(date(2026, 1, 1)) == 2024
    print("  [OK] juli is de kantelmaand: ervoor jaar-2, erna jaar-1")


def test_januari_maakt_niet_de_halve_database_geel():
    """In januari is FY(vorig jaar) nog niet gepubliceerd — dan geen achterstand."""
    assert dq.boekjaar_achterstand(2025, date(2026, 1, 15)) == 0
    assert dq.boekjaar_achterstand(2025, date(2026, 6, 30)) == 0
    # ... maar in augustus wél.
    assert dq.boekjaar_achterstand(2025, date(2026, 8, 3)) == 0
    assert dq.boekjaar_achterstand(2024, date(2026, 8, 3)) == 1
    print("  [OK] een december-boekjaar krijgt tot en met juni de tijd")


def test_achterstand_telt_in_hele_jaren():
    vandaag = date(2026, 8, 3)
    assert dq.boekjaar_achterstand(None, vandaag) == 0
    assert dq.boekjaar_achterstand(2026, vandaag) == 0   # vooruit lopen is geen achterstand
    assert dq.boekjaar_achterstand(2024, vandaag) == 1
    assert dq.boekjaar_achterstand(2021, vandaag) == 4
    print("  [OK] achterstand in hele boekjaren, nooit negatief")


def test_het_geval_lastik(monkeypatch):
    """Compleet gevuld, vandaag opgehaald, en toch niet 'ok'.

    Dit is precies de combinatie die het probleem onzichtbaar maakte: er valt
    niets aan te merken op de opgeslagen jaren, dus alle bestaande signalen
    stonden op groen.
    """
    monkeypatch.setattr(dq, "verwacht_boekjaar", lambda vandaag=None: 2025)

    r = dq.evaluate(
        "LASTIK.HE",
        annual_rows=[_rij(2024), _rij(2023), _rij(2022)],
        market_data=_markt(),
        stock_info={"quote_type": "EQUITY"},
        fetch_success=True,
        fetched_date="2026-08-03",
    )
    assert r["completeness_pct"] == 100.0, "de velden zijn wél compleet — dat blijft zo"
    assert r["latest_fy"] == 2024
    assert r["data_status"] == "warning", f"kreeg '{r['data_status']}' terwijl FY2024 een jaar achterloopt"
    assert any("lopen 1 jaar achter" in i for i in r["issues"]), r["issues"]
    print(f"  [OK] LASTIK-geval: 100% compleet maar status '{r['data_status']}'")


def test_verse_ophaalronde_verwijst_niet_naar_refresh():
    """Als we net opgehaald hebben, is 'klik Refresh' een leugen."""
    r = dq.evaluate(
        "LASTIK.HE",
        annual_rows=[_rij(2024), _rij(2023), _rij(2022)],
        market_data=_markt(),
        stock_info={"quote_type": "EQUITY"},
        fetch_success=True,
        fetched_date=date.today().isoformat(),
    )
    tekst = " ".join(r["issues"])
    if dq.boekjaar_achterstand(2024):
        assert "Yahoo heeft dit boekjaar niet" in tekst, tekst
        assert "refresh" not in tekst.lower(), f"belooft nog steeds een refresh: {tekst}"
        print("  [OK] verse fetch geeft 'Yahoo heeft dit boekjaar niet', geen refresh-belofte")


def test_twee_jaar_achter_is_geen_waarschuwing_meer(monkeypatch):
    """Twee boekjaren achterstand blokkeert de waardering (status 'bad')."""
    monkeypatch.setattr(dq, "verwacht_boekjaar", lambda vandaag=None: 2025)

    r = dq.evaluate(
        "CCAP.DE",
        annual_rows=[_rij(2023), _rij(2022), _rij(2021)],
        market_data=_markt(),
        stock_info={"quote_type": "EQUITY"},
        fetch_success=True,
        fetched_date="2026-08-03",
    )
    assert r["data_status"] == "bad", f"kreeg '{r['data_status']}'"
    print("  [OK] twee jaar achterstand geeft 'bad', dus INSUFFICIENT DATA i.p.v. een signaal")


def test_actuele_cijfers_blijven_gewoon_ok(monkeypatch):
    """Het vangnet mag niet de rest van de database meesleuren."""
    monkeypatch.setattr(dq, "verwacht_boekjaar", lambda vandaag=None: 2025)

    r = dq.evaluate(
        "ASML.AS",
        annual_rows=[_rij(2025), _rij(2024), _rij(2023)],
        market_data=_markt(),
        stock_info={"quote_type": "EQUITY"},
        fetch_success=True,
        fetched_date="2026-08-03",
    )
    assert r["data_status"] == "ok", f"kreeg '{r['data_status']}' met issues {r['issues']}"
    assert not any("achter" in i for i in r["issues"])
    print("  [OK] bij-de-tijd blijft 'ok' zonder extra meldingen")


def test_reden_bucket_is_eigen_categorie():
    """Verouderd is iets anders dan 'GEEN DATA' — anders zoek je in de verkeerde hoek."""
    issues = ["Jaarcijfers lopen 2 jaar achter: nieuwste is FY2023, verwacht FY2025. "
              "Een refresh kan nieuwere cijfers opleveren."]
    c = dq.classify_blockers(issues, "bad")
    assert c["primary_blocker"] == "verouderde_cijfers", c

    reden = dq.classify_signal_reason("INSUFFICIENT DATA", "bad", issues)
    assert reden["reason_code"] == "verouderd"
    assert reden["reason_label"] == "CIJFERS VEROUDERD"
    print(f"  [OK] eigen reden-bucket: {reden['reason_label']}")


if __name__ == "__main__":
    class _Patch:
        """Minimale monkeypatch-vervanger zodat dit bestand ook zonder pytest draait."""
        def __init__(self): self._terug = []
        def setattr(self, obj, naam, waarde):
            self._terug.append((obj, naam, getattr(obj, naam)))
            setattr(obj, naam, waarde)
        def herstel(self):
            for obj, naam, oud in reversed(self._terug):
                setattr(obj, naam, oud)

    test_grens_ligt_op_juli()
    test_januari_maakt_niet_de_halve_database_geel()
    test_achterstand_telt_in_hele_jaren()
    for fn in (test_het_geval_lastik, test_twee_jaar_achter_is_geen_waarschuwing_meer,
               test_actuele_cijfers_blijven_gewoon_ok):
        p = _Patch()
        try:
            fn(p)
        finally:
            p.herstel()
    test_verse_ophaalronde_verwijst_niet_naar_refresh()
    test_reden_bucket_is_eigen_categorie()
    print("\nAlle tests verouderde jaarcijfers geslaagd.")
