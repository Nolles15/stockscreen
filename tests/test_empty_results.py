"""Lege opbrengst moet meetellen als mislukking én in de storm-detectie."""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine.refresh as R

bumps=[]
class FakeDB:
    def get_refresh_queue(self,n): return [f"T{i}" for i in range(n)]
    def log_activity(self,*a,**k): self.last=a
    def bump_failure_counter(self,t): bumps.append(t)
    def count_annual_rows(self,t): return 0 if int(t[1:])<self.leeg else 3
FakeDB.last=None

def maak(leeg):
    f=FakeDB(); f.leeg=leeg; R.db=f
    R.screener=types.SimpleNamespace(run_ticker=lambda t,c:{"signal":"BUY"})
    R.data_fetcher=types.SimpleNamespace(fetch_and_store=lambda t,count_failure=True: None)
    return f

# 10 van 100 leveren niets op: echte dode tickers, tellers moeten omhoog
bumps.clear(); f=maak(10)
r=R.refresh_fundamentals_batch(100, config={})
print(f"10 leeg: empty={len(r['empty'])} storm={r['storm_detected']} tellers={len(bumps)}")
assert len(r['empty'])==10 and not r['storm_detected'] and len(bumps)==10
print("  [OK] dode tickers gaan nu wel richting het archief")

# 90 van 100 leveren niets op: dat is een storing, niemand straffen
bumps.clear(); f=maak(90)
r=R.refresh_fundamentals_batch(100, config={})
print(f"90 leeg: empty={len(r['empty'])} storm={r['storm_detected']} tellers={len(bumps)}")
assert r['storm_detected'] and len(bumps)==0
print("  [OK] massale leegte wordt als storing herkend, geen tellers omhoog")
print("\nGESLAAGD")
