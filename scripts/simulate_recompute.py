"""
Simuleer /api/data-quality/recompute LOKAAL en read-only.

Haalt per ticker de reeds opgeslagen financials/market/stock op via de live
/api/stock/<T> endpoint en draait de HUIDIGE (lokale) data_quality.evaluate()
erover. Vergelijkt de uitkomst met de status die nu in de DB staat
(/api/data-quality). Zo meet je de impact van een gate-wijziging (Route A)
zonder te hoeven deployen.

Read-only: doet uitsluitend GET-calls.

Gebruik:
    python scripts/simulate_recompute.py
    python scripts/simulate_recompute.py --workers 16 --show-rescued 40
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from engine import data_quality  # noqa: E402

APP = os.environ.get("STOCKSCREEN_URL", "https://stockscreen-janco.fly.dev")
_CTX = ssl._create_unverified_context()


def _get(path: str) -> object | None:
    try:
        with urllib.request.urlopen(APP + path, timeout=90, context=_CTX) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def simulate_one(rec: dict) -> dict | None:
    """rec = data-quality record. Pull stored data, run new evaluate()."""
    t = rec["ticker"]
    d = _get(f"/api/stock/{t}")
    if not d:
        return None
    annual = d.get("annual") or []
    market = d.get("market") or {}
    stock = d.get("stock") or {}
    fetched = annual[0].get("fetched_date") if annual else None
    fetch_ok = bool(annual) or bool(market.get("price"))

    new = data_quality.evaluate(
        t, annual, market, stock,
        fetch_success=fetch_ok,
        prev_consecutive_failures=rec.get("consecutive_failures") or 0,
        fetched_date=fetched,
    )
    return {
        "ticker": t,
        "before": rec.get("data_status"),
        "after": new["data_status"],
        "new_issues": new.get("issues") or [],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--show-rescued", type=int, default=30)
    args = p.parse_args()

    dq = _get("/api/data-quality")
    if not isinstance(dq, list):
        print("FOUT: kon /api/data-quality niet ophalen")
        return 1
    print(f"Bron: {APP}  |  {len(dq)} tickers  |  {args.workers} workers (read-only)")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = [r for r in ex.map(simulate_one, dq) if r]

    missing = len(dq) - len(results)
    before = Counter(r["before"] for r in results)
    after = Counter(r["after"] for r in results)
    trans = Counter(f"{r['before']} -> {r['after']}" for r in results)

    def line(c: Counter) -> str:
        return "  ".join(f"{k}={v}" for k, v in sorted(c.items(), key=lambda kv: str(kv[0])))

    print(f"\nGesimuleerd: {len(results)}  (niet opgehaald: {missing})")
    print(f"\nVOOR  : {line(before)}")
    print(f"NA    : {line(after)}")

    blocked_before = sum(v for k, v in before.items() if k in ("bad", "missing"))
    blocked_after = sum(v for k, v in after.items() if k in ("bad", "missing"))
    print(f"\nGeblokkeerd (bad+missing):  voor={blocked_before}  na={blocked_after}  "
          f"verschil={blocked_before - blocked_after:+d}")

    print("\nTRANSITIES (alleen waar status verandert):")
    for k, v in trans.most_common():
        b, a = k.split(" -> ")
        if b != a:
            print(f"  {k:<28} {v}")

    rescued = [r for r in results
               if r["before"] in ("bad", "missing") and r["after"] not in ("bad", "missing")]
    regress = [r for r in results
               if r["before"] not in ("bad", "missing", None) and r["after"] in ("bad", "missing")]

    print(f"\nGERED (was geblokkeerd -> nu waardeerbaar): {len(rescued)}")
    for r in rescued[:args.show_rescued]:
        print(f"  {r['ticker']:<14} {r['before']} -> {r['after']}")
    if len(rescued) > args.show_rescued:
        print(f"  ... +{len(rescued) - args.show_rescued} meer")

    if regress:
        print(f"\nLET OP — REGRESSIE (was ok/warning -> nu geblokkeerd): {len(regress)}")
        for r in regress:
            print(f"  {r['ticker']:<14} {r['before']} -> {r['after']}  {(r['new_issues'][:1] or [''])[0][:70]}")
    else:
        print("\nGeen regressies (niets dat eerst ok/warning was werd nu geblokkeerd).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
