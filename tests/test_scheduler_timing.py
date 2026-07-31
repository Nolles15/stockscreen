"""Test de tijdlogica van scheduler v2 zonder DB of netwerk."""
import os, sys, types
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

AMS = ZoneInfo("Europe/Amsterdam")
STATE = {}

# Kopie van de logica uit app.py (_ran_today / _days_since)
def ran_today(key, now_local):
    raw = STATE.get(key)
    if not raw: return False
    last = datetime.fromisoformat(raw)
    if last.tzinfo is None: last = last.replace(tzinfo=timezone.utc)
    return last.astimezone(AMS).date() == now_local.date()

def should_run_prices(now_local):
    return (now_local.hour, now_local.minute) >= (18, 30) and not ran_today("last_price_refresh_at", now_local)

cases = [
    ("17:00, nog niet gedraaid", datetime(2026,7,31,17,0,tzinfo=AMS), None, False),
    ("18:29, nog niet gedraaid", datetime(2026,7,31,18,29,tzinfo=AMS), None, False),
    ("18:30, nog niet gedraaid", datetime(2026,7,31,18,30,tzinfo=AMS), None, True),
    ("20:00, al gedraaid 18:31", datetime(2026,7,31,20,0,tzinfo=AMS),
        datetime(2026,7,31,18,31,tzinfo=AMS).astimezone(timezone.utc).isoformat(), False),
    ("volgende dag 19:00, gisteren gedraaid", datetime(2026,8,1,19,0,tzinfo=AMS),
        datetime(2026,7,31,18,31,tzinfo=AMS).astimezone(timezone.utc).isoformat(), True),
    ("herstart 18:45 na run 18:31", datetime(2026,7,31,18,45,tzinfo=AMS),
        datetime(2026,7,31,18,31,tzinfo=AMS).astimezone(timezone.utc).isoformat(), False),
]
fails = 0
for naam, now, state, verwacht in cases:
    STATE.clear()
    if state: STATE["last_price_refresh_at"] = state
    got = should_run_prices(now)
    ok = got == verwacht
    fails += not ok
    print(f"  [{'OK ' if ok else 'FOUT'}] {naam}: draait={got} (verwacht {verwacht})")
print("FALEND:", fails)
sys.exit(1 if fails else 0)
