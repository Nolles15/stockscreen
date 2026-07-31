"""Test NULL-bescherming en storm-guard zonder database of netwerk."""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 1. NULL-bescherming in de upsert-clausule ---
from engine.db import _update_clause
c = _update_clause("financials", {"ticker":1,"period_type":1,"fiscal_year":1,"revenue":1,"fetched_date":1},
                   keys=("ticker","period_type","fiscal_year"), always=("fetched_date",))
print("clausule:", c)
assert "revenue=COALESCE(excluded.revenue, financials.revenue)" in c, "omzet moet beschermd zijn"
assert "fetched_date=excluded.fetched_date" in c, "tijdstempel moet altijd overschrijven"
assert "ticker" not in c.split("fetched_date")[0].replace("financials.",""), "sleutel hoort niet in SET"
print("  [OK] data-kolom beschermd, tijdstempel overschrijft altijd\n")

# --- 2. Storm-guard ---
import engine.refresh as R

bumps, suspends = [], []
class FakeDB:
    def get_refresh_queue(self, n): return [f"T{i}" for i in range(n)]
    def log_activity(self, *a, **k): self.last = a
    def bump_failure_counter(self, t): bumps.append(t)
    def count_annual_rows(self, t): return 3
FakeDB.last = None

def maak(faalkans):
    fake = FakeDB()
    R.db = fake
    R.screener = types.SimpleNamespace(run_ticker=lambda t, c: {"signal":"BUY"})
    def fetch(t, count_failure=True):
        idx = int(t[1:])
        if idx < faalkans: raise RuntimeError("Yahoo down")
    R.data_fetcher = types.SimpleNamespace(fetch_and_store=fetch)
    return fake

# Storm: 80 van 100 mislukt
bumps.clear()
fake = maak(80)
res = R.refresh_fundamentals_batch(100, config={})
print(f"storm-scenario: {len(res['failed'])}/100 mislukt, storm_detected={res['storm_detected']}, tellers opgehoogd={len(bumps)}")
assert res["storm_detected"] is True, "moet als storm herkend worden"
assert len(bumps) == 0, "bij een storm mag GEEN teller omhoog"
assert fake.last[0] == "storm_detected"
print("  [OK] storing bij de bron leidt niet tot suspensies\n")

# Normaal: 10 van 100 mislukt
bumps.clear()
fake = maak(10)
res = R.refresh_fundamentals_batch(100, config={})
print(f"normaal scenario: {len(res['failed'])}/100 mislukt, storm_detected={res['storm_detected']}, tellers opgehoogd={len(bumps)}")
assert res["storm_detected"] is False
assert len(bumps) == 10, "echte losse fouten moeten wel tellen"
print("  [OK] losse fouten tellen wel gewoon mee")
print("\nALLE TESTS GESLAAGD")
