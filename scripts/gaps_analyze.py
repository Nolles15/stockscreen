"""
Stockscreen gaps-analyse — Fase 0 diagnose-instrument.

Categoriseert de hele tickeruniverse in primaire blocker-buckets en print:
  • bucket-verdeling (aantal + % per primary_blocker)
  • kruistabellen: blocker × exchange-suffix, × markt, × currency
  • co-occurrence van blockers (hoeveel split-tickers hebben óók een ander blok)
  • gestratificeerde steekproef per bucket (met bewijstekst) voor handmatige check

Read-only: doet alleen GET-calls, wijzigt niets.

Bronnen (in volgorde van voorkeur):
  1. GET /api/gaps-report   — rijkste bron (markt/currency/blockers/evidence)
  2. GET /api/data-quality  — fallback vóór deploy; classificeert lokaal via
                              engine.data_quality.classify_blockers (geen markt/currency)

Gebruik:
    python scripts/gaps_analyze.py
    python scripts/gaps_analyze.py --sample 15
    python scripts/gaps_analyze.py --bucket split_suspected --sample 30
    STOCKSCREEN_URL=http://localhost:5001 python scripts/gaps_analyze.py

Env vars:
    STOCKSCREEN_URL  (default: https://stockscreen-janco.fly.dev)
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict

# De app draait achter een cert dat lokaal niet altijd verifieert (vgl. `curl -k`
# in CLAUDE.md). Read-only diagnose tegen de eigen app → verificatie uitzetten.
_SSL_CTX = ssl._create_unverified_context()

# Windows-console (cp1252) struikelt over non-ASCII; forceer UTF-8 indien mogelijk.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# Repo-root op pad zodat de fallback engine.data_quality kan importeren.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

APP = os.environ.get("STOCKSCREEN_URL", "https://stockscreen-janco.fly.dev")


def _get(path: str) -> tuple[int, object]:
    url = f"{APP}{path}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def _suffix(ticker: str) -> str:
    """Exchange-suffix na de laatste punt; geen punt → 'US' (US-notatie)."""
    return ticker.rsplit(".", 1)[1] if "." in ticker else "US"


def load_rows() -> tuple[list[dict], str]:
    """Haal gaps-report op; val terug op data-quality + lokale classificatie."""
    status, data = _get("/api/gaps-report")
    if status == 200 and isinstance(data, list):
        return data, "gaps-report"

    # Fallback: data-quality heeft issues maar geen markt/currency.
    status, data = _get("/api/data-quality")
    if status != 200 or not isinstance(data, list):
        print(f"FOUT: kon noch /api/gaps-report noch /api/data-quality ophalen ({status})")
        sys.exit(1)

    from engine import data_quality  # lazy: alleen in fallback nodig
    rows = []
    for r in data:
        cls = data_quality.classify_blockers(r.get("issues"), r.get("data_status"))
        rows.append({
            **r,
            "markt": None, "currency": None, "sector": None,
            "primary_blocker": cls["primary_blocker"],
            "blockers": cls["blockers"],
            "info_flags": cls["info_flags"],
            "evidence": cls["evidence"],
            "issue_count": len(r.get("issues") or []),
        })
    return rows, "data-quality (fallback, geen markt/currency)"


def _bar(n: int, total: int, width: int = 30) -> str:
    filled = int(round(width * n / total)) if total else 0
    return "#" * filled + "." * (width - filled)


def print_distribution(rows: list[dict]) -> None:
    total = len(rows)
    by_status = Counter(r.get("data_status") for r in rows)
    by_blocker = Counter(r.get("primary_blocker") for r in rows)

    print(f"\n=== UNIVERSE: {total} tickers ===\n")
    print("data_status:")
    for st in ("missing", "bad", "warning", "ok", None):
        if by_status.get(st):
            print(f"  {str(st):<10} {by_status[st]:>4}")

    print("\nPRIMAIRE BLOCKER-BUCKET (elke ticker telt 1×):")
    print(f"  {'bucket':<22} {'n':>4} {'%':>6}  verdeling")
    print("  " + "-" * 72)
    for blocker, n in by_blocker.most_common():
        pct = 100.0 * n / total if total else 0
        print(f"  {str(blocker):<22} {n:>4} {pct:>5.1f}%  {_bar(n, total)}")


def print_crosstab(rows: list[dict], dim: str, label: str, top: int = 12) -> None:
    """Kruis primary_blocker tegen een dimensie (suffix/markt/currency)."""
    if dim == "suffix":
        keyfn = lambda r: _suffix(r["ticker"])
    else:
        keyfn = lambda r: r.get(dim) or "?"

    # Alleen geblokkeerde tickers zijn interessant voor de kruistabel.
    blocked = [r for r in rows if r.get("data_status") in ("bad", "missing")]
    if not blocked:
        return
    counts = Counter(keyfn(r) for r in blocked)
    print(f"\n--- {label}: geblokkeerde tickers ({len(blocked)}) ---")
    for key, n in counts.most_common(top):
        # dominante blocker binnen deze groep
        sub = Counter(r.get("primary_blocker") for r in blocked if keyfn(r) == key)
        top_blocker, top_n = sub.most_common(1)[0]
        print(f"  {str(key):<8} {n:>4}   meest: {top_blocker} ({top_n})")


def print_cooccurrence(rows: list[dict]) -> None:
    """Hoe vaak komt een blocker samen met andere blockers voor?"""
    has_multi = [r for r in rows if r.get("blockers")]
    if not has_multi:
        return
    print("\n--- BLOCKER CO-OCCURRENCE (op alle blockers, niet alleen primair) ---")
    solo = Counter()
    combo = Counter()
    for r in rows:
        bl = r.get("blockers") or []
        if len(bl) == 1:
            solo[bl[0]] += 1
        for b in bl:
            combo[b] += 1
    print(f"  {'blocker':<22} {'totaal':>7} {'solo':>6}  (solo = enige blocker -> zuiverste rescue-kandidaat)")
    for b, n in combo.most_common():
        print(f"  {b:<22} {n:>7} {solo.get(b, 0):>6}")


def print_samples(rows: list[dict], n: int, only_bucket: str | None) -> None:
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[r.get("primary_blocker")].append(r)

    buckets = [only_bucket] if only_bucket else sorted(
        by_bucket, key=lambda b: -len(by_bucket[b])
    )
    print("\n=== GESTRATIFICEERDE STEEKPROEF PER BUCKET ===")
    for bucket in buckets:
        sample = by_bucket.get(bucket) or []
        if not sample:
            print(f"\n[{bucket}] — geen tickers")
            continue
        # Spreid over markten: pak afwisselend per suffix.
        by_sfx: dict[str, list[dict]] = defaultdict(list)
        for r in sample:
            by_sfx[_suffix(r["ticker"])].append(r)
        spread: list[dict] = []
        sfx_lists = list(by_sfx.values())
        i = 0
        while len(spread) < min(n, len(sample)):
            lst = sfx_lists[i % len(sfx_lists)]
            if lst:
                spread.append(lst.pop(0))
            i += 1
            if all(not l for l in sfx_lists):
                break
        print(f"\n[{bucket}] — {len(sample)} tickers, steekproef {len(spread)}:")
        for r in spread:
            mk = r.get("markt") or "?"
            cc = r.get("currency") or "?"
            ev = (r.get("evidence") or "").strip()
            print(f"  {r['ticker']:<14} {mk:<4} {cc:<5} {(r.get('name') or '')[:28]:<28} {ev[:80]}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", type=int, default=12, help="steekproefgrootte per bucket (default 12)")
    p.add_argument("--bucket", help="toon alleen steekproef voor één bucket")
    p.add_argument("--no-samples", action="store_true", help="sla steekproef over")
    args = p.parse_args()

    rows, source = load_rows()
    print(f"Bron: {source}  |  {APP}")

    print_distribution(rows)
    print_crosstab(rows, "suffix",   "PER EXCHANGE-SUFFIX")
    print_crosstab(rows, "markt",    "PER MARKT (veld)")
    print_crosstab(rows, "currency", "PER CURRENCY")
    print_cooccurrence(rows)
    if not args.no_samples:
        print_samples(rows, args.sample, args.bucket)
    return 0


if __name__ == "__main__":
    sys.exit(main())
