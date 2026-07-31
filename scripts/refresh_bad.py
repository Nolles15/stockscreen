"""
Force-refresh alle INSUFFICIENT DATA tickers.

Gebruik:
    CRON_TOKEN=... python scripts/refresh_bad.py
"""
import json
import os
import sys
import time
import urllib.request

APP = "https://stockscreen-janco.fly.dev"
TOKEN = os.environ.get("CRON_TOKEN")
if not TOKEN:
    sys.exit("CRON_TOKEN env var vereist")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIST_PATH = os.path.join(SCRIPT_DIR, "bad_tickers.txt")
LOG_PATH = os.path.join(SCRIPT_DIR, "refresh_log.txt")

with open(LIST_PATH, "r", encoding="utf-8") as f:
    tickers = [t.strip() for t in f if t.strip()]

total = len(tickers)
ok = 0
fail = 0
log = open(LOG_PATH, "w", encoding="utf-8")


def refresh_one(t: str) -> tuple[bool, dict]:
    req = urllib.request.Request(
        f"{APP}/api/cron/refresh-one/{t}",
        method="POST",
        headers={"X-Cron-Token": TOKEN},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read().decode("utf-8")
            return True, json.loads(body)
    except Exception as e:
        return False, {"error": str(e)}


for idx, t in enumerate(tickers, 1):
    got, resp = refresh_one(t)
    if got and resp.get("ok"):
        ok += 1
        line = (
            f"[{idx:3}/{total}] OK   {t:14} sig={resp.get('signal','?'):20} "
            f"fv={resp.get('combined_fv','?')}  ({resp.get('elapsed_s','?')}s)"
        )
    else:
        fail += 1
        err = resp.get("error") if isinstance(resp, dict) else "?"
        line = f"[{idx:3}/{total}] FAIL {t:14} {err}"
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()

summary = f"=== Klaar: ok={ok} fail={fail} totaal={total} ==="
print(summary, flush=True)
log.write(summary + "\n")
log.close()
