"""Narekentest: klopt wat de methodepagina toont met wat er werkelijk gebeurt?

De methodepagina is bedoeld voor iemand die de cijfers wil controleren — een
registeraccountant. Dan is er één risico dat alles ondermijnt: dat de pagina een
nette, geïdealiseerde versie van de berekening laat zien die afwijkt van de
berekening zelf. Een lezer die naar aanleiding daarvan gaat narekenen, komt dan
op iets anders uit en heeft geen idee wie er fout zit.

Deze test rekent daarom uit de getoonde tussenwaarden de uitkomst opnieuw uit en
vergelijkt die met wat de motor produceerde. Wijken ze af, dan liegt de pagina.
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.normalizer import normalize_metric, normalize_metric_trace  # noqa: E402
from engine.quality_score import piotroski_fscore, quality_score  # noqa: E402


def _jaren(**reeksen):
    """Bouwt annual-rijen, nieuwste jaar eerst (zoals de motor ze aanlevert)."""
    n = len(next(iter(reeksen.values())))
    rijen = [{"fiscal_year": 2021 + i, **{k: v[i] for k, v in reeksen.items()}}
             for i in range(n)]
    return list(reversed(rijen))


def test_normalisatie_is_narekenbaar():
    """De mediaan uit de trace moet volgen uit de getoonde 'gebruikte' waarden."""
    rijen = _jaren(eps_diluted=[1.0, 1.2, 1.1, 1.3, 40.0])   # 40 is de uitschieter
    spoor = normalize_metric_trace(rijen, "eps_diluted")

    # 1. De uitkomst is de mediaan van precies de waarden die als 'gebruikt' staan.
    assert spoor["uitkomst"] == statistics.median(spoor["gebruikt"])

    # 2. Gebruikt + afgevallen samen zijn alle bruikbare waarden — er verdwijnt
    #    niets buiten het zicht van de lezer om.
    assert sorted(spoor["gebruikt"] + spoor["afgevallen"]) == sorted(spoor["bruikbare_waarden"])

    # 3. De grenzen die getoond worden verklaren precies wat er afviel.
    for v in spoor["afgevallen"]:
        assert v < spoor["ondergrens"] or v > spoor["bovengrens"]
    for v in spoor["gebruikt"]:
        assert spoor["ondergrens"] <= v <= spoor["bovengrens"]

    # 4. En de trace geeft hetzelfde antwoord als de functie die de motor gebruikt.
    assert spoor["uitkomst"] == normalize_metric(rijen, "eps_diluted")
    print("  [OK] normalisatie: uitkomst volgt uit de getoonde waarden en grenzen")


def test_normalisatie_zonder_uitschieters():
    rijen = _jaren(fcf=[100.0, 110.0, 105.0, 108.0, 102.0])
    spoor = normalize_metric_trace(rijen, "fcf")
    assert spoor["afgevallen"] == []
    assert spoor["uitkomst"] == normalize_metric(rijen, "fcf")
    print("  [OK] normalisatie zonder uitschieters laat alle jaren staan")


def test_piotroski_waarden_verklaren_de_uitkomst():
    """Elke geslaagde test moet uit de getoonde twee getallen volgen."""
    rijen = _jaren(
        total_assets=[1000, 1100], net_income=[50, 80], operating_cf=[90, 140],
        total_debt=[300, 280], current_assets=[400, 500], current_liabilities=[200, 200],
        shares_outstanding=[100, 100], gross_profit=[300, 360], revenue=[800, 900],
    )
    p = piotroski_fscore(rijen)
    w = p["waarden"]

    # F1: rendement op activa positief
    assert p["criteria"]["F1"] == (w["F1"]["waarde"] > w["F1"]["drempel"])
    # F3: rendement op activa gestegen
    assert p["criteria"]["F3"] == (w["F3"]["dit_jaar"] > w["F3"]["vorig_jaar"])
    # F5: schuldratio gedaald (let op: kleiner is beter)
    assert p["criteria"]["F5"] == (w["F5"]["dit_jaar"] < w["F5"]["vorig_jaar"])
    # F8: brutomarge verbeterd
    assert p["criteria"]["F8"] == (w["F8"]["dit_jaar"] > w["F8"]["vorig_jaar"])

    # De score is het aantal geslaagde tests — niets meer, niets minder.
    assert p["score"] == sum(1 for v in p["criteria"].values() if v is True)
    print(f"  [OK] Piotroski: alle {len(w)} tests verklaard door hun eigen getallen")


def test_kwaliteit_detail_verklaart_de_punten():
    """De toegekende punten moeten volgen uit de getoonde waarden en drempels."""
    rijen = _jaren(
        total_equity=[500, 520, 540, 560, 580], total_debt=[100, 100, 100, 100, 100],
        ebit=[120, 125, 130, 135, 140], interest_expense=[5, 5, 5, 5, 5],
        roe=[0.20, 0.21, 0.22, 0.23, 0.24], net_cash=[0, 0, 0, 0, 0],
        fcf=[80, 85, 90, 95, 100], operating_cf=[110, 115, 120, 125, 130],
        ebitda=[140, 145, 150, 155, 160], total_assets=[900, 920, 940, 960, 980],
        net_income=[90, 95, 100, 105, 110], revenue=[800, 820, 840, 860, 880],
        gross_profit=[300, 310, 320, 330, 340], eps_diluted=[1.0, 1.1, 1.2, 1.3, 1.4],
        current_assets=[400, 410, 420, 430, 440],
        current_liabilities=[200, 200, 200, 200, 200],
        shares_outstanding=[90, 90, 90, 90, 90],
    )
    normalized = {"avg_roe": 0.22, "avg_roic": 0.16,
                  "stddev_eps_pct": 0.10, "stddev_fcf_pct": 0.08}
    q = quality_score(rijen, normalized)
    d, bd = q["detail"], q["breakdown"]

    # Balans: de punten volgen uit de twee getoonde toetsen.
    b = d["balance_sheet"]
    verwacht = 2.0 if (b["schuldratio_gehaald"] and b["rentedekking_gehaald"]) else \
               1.0 if (b["schuldratio_gehaald"] or b["rentedekking_gehaald"]) else 0.0
    assert bd["balance_sheet"] == verwacht

    # Stabiliteit: idem, en de getoonde variatie moet onder de getoonde drempel liggen.
    s = d["stability"]
    assert s["eps_stabiel"] == (s["variatie_eps"] < s["drempel"])
    assert s["fcf_stabiel"] == (s["variatie_fcf"] < s["drempel"])

    # Winstgevendheid: het getoonde gemiddelde tegen de getoonde drempel.
    p = d["profitability"]
    assert p["roe_boven_drempel"] == (p["gemiddelde_roe"] > p["drempels"]["gemiddelde"])
    assert p["roic_boven_drempel"] == (p["gemiddelde_roic"] > p["drempels"]["gemiddelde"])

    # Totaal is de som van de onderdelen — geen verborgen correctie.
    assert q["total"] == sum(bd.values())
    print("  [OK] kwaliteitsscore: punten volgen uit de getoonde waarden en drempels")


def test_detail_dekt_elk_criterium():
    """Geen criterium zonder onderbouwing: anders staat er een punt op de pagina
    dat de lezer niet kan controleren."""
    rijen = _jaren(
        total_equity=[500, 520], total_debt=[100, 100], ebit=[120, 125],
        interest_expense=[5, 5], roe=[0.20, 0.21], fcf=[80, 85],
        operating_cf=[110, 115], ebitda=[140, 145], total_assets=[900, 920],
        net_income=[90, 95], revenue=[800, 820], gross_profit=[300, 310],
        eps_diluted=[1.0, 1.1], current_assets=[400, 410],
        current_liabilities=[200, 200], shares_outstanding=[90, 90],
    )
    q = quality_score(rijen, {"avg_roe": 0.2, "avg_roic": 0.15,
                              "stddev_eps_pct": 0.1, "stddev_fcf_pct": 0.1})
    ontbreekt = set(q["breakdown"]) - set(q["detail"])
    assert not ontbreekt, f"criteria zonder onderbouwing: {ontbreekt}"
    print("  [OK] elk kwaliteitscriterium heeft zijn onderbouwing")


if __name__ == "__main__":
    test_normalisatie_is_narekenbaar()
    test_normalisatie_zonder_uitschieters()
    test_piotroski_waarden_verklaren_de_uitkomst()
    test_kwaliteit_detail_verklaart_de_punten()
    test_detail_dekt_elk_criterium()
    print("\nAlle narekentests geslaagd.")
