"""
Stockscreen triage CLI — hulpmiddel om INSUFFICIENT DATA tickers af te handelen.

Gebruikt de bulk-endpoints van de live app — geen directe DB-verbinding nodig.

Voorbeelden:
    python scripts/triage_cli.py list-bad
    python scripts/triage_cli.py list-bad --status missing
    python scripts/triage_cli.py deactivate AFKL.AS NRJ.PA --reason "dual-listing remap"
    python scripts/triage_cli.py activate EXO.MI BMT.AS
    python scripts/triage_cli.py remap EXOR.AS EXO.MI
    python scripts/triage_cli.py refresh AKRBP.OL YAR.OL

Env vars:
    STOCKSCREEN_URL  (default: https://stockscreen-janco.fly.dev)
    CRON_TOKEN       (alleen nodig voor `refresh`)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

APP = os.environ.get("STOCKSCREEN_URL", "https://stockscreen-janco.fly.dev")


def _request(method: str, path: str, body: dict | None = None, headers: dict | None = None) -> tuple[int, dict | str]:
    url = f"{APP}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read().decode("utf-8")
            try:
                return r.status, json.loads(raw)
            except json.JSONDecodeError:
                return r.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def cmd_list_bad(args: argparse.Namespace) -> int:
    status, rows = _request("GET", "/api/dashboard")
    if status != 200 or not isinstance(rows, list):
        print(f"FOUT: kon dashboard niet ophalen ({status})")
        return 1
    bad = [r for r in rows if r.get("signal") == "INSUFFICIENT DATA"]
    if args.status:
        bad = [r for r in bad if (r.get("data_status") or "null") == args.status]

    print(f"{'Ticker':<14} {'Status':<8} {'Conf':<5} {'Fails':<6} Naam")
    print("-" * 80)
    for r in bad:
        ds = r.get("data_status") or "null"
        conf = r.get("fv_confidence") or "?"
        fails = r.get("consecutive_failures") or 0
        name = (r.get("name") or "")[:40]
        print(f"{r['ticker']:<14} {ds:<8} {conf:<5} {fails:<6} {name}")
    print(f"\nTotaal: {len(bad)}")
    return 0


def cmd_deactivate(args: argparse.Namespace) -> int:
    body = {"tickers": args.tickers, "reason": args.reason}
    status, resp = _request("POST", "/api/stocks/bulk-deactivate", body)
    if status != 200:
        print(f"FOUT ({status}): {resp}")
        return 1
    print(f"Gedeactiveerd: {resp.get('count')}")
    for t in resp.get("deactivated", []):
        print(f"  OK   {t}")
    for t in resp.get("skipped", []):
        print(f"  SKIP {t} (niet in DB)")
    for r in resp.get("rejected", []):
        print(f"  REJ  {r.get('ticker')} ({r.get('reason')})")
    return 0


def cmd_activate(args: argparse.Namespace) -> int:
    body = {"tickers": args.tickers}
    status, resp = _request("POST", "/api/stocks/bulk-activate", body)
    if status != 200:
        print(f"FOUT ({status}): {resp}")
        return 1
    print(f"Geactiveerd: {resp.get('count')}")
    for t in resp.get("activated", []):
        print(f"  OK   {t}")
    for t in resp.get("skipped", []):
        print(f"  SKIP {t} (niet in DB)")
    return 0


def cmd_remap(args: argparse.Namespace) -> int:
    body = {"from": args.src, "to": args.dst}
    status, resp = _request("POST", "/api/stocks/remap", body)
    if status != 200:
        print(f"FOUT ({status}): {resp}")
        return 1
    print(f"Remap: {resp['from']} → {resp['to']} (fetch gestart: {resp['fetch_started']})")
    return 0


def cmd_refresh(args: argparse.Namespace) -> int:
    token = os.environ.get("CRON_TOKEN")
    if not token:
        print("FOUT: CRON_TOKEN env var vereist voor refresh")
        return 1
    ok_count = fail_count = 0
    for t in args.tickers:
        status, resp = _request("POST", f"/api/cron/refresh-one/{t}", body={}, headers={"X-Cron-Token": token})
        if status == 200 and isinstance(resp, dict) and resp.get("ok"):
            ok_count += 1
            print(f"  OK   {t} sig={resp.get('signal')} fv={resp.get('combined_fv')}")
        else:
            fail_count += 1
            err = resp.get("error") if isinstance(resp, dict) else resp
            print(f"  FAIL {t} {err}")
    print(f"\nok={ok_count} fail={fail_count}")
    return 0 if fail_count == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(prog="triage_cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-bad", help="Toon alle INSUFFICIENT DATA tickers")
    p_list.add_argument("--status", choices=["missing", "bad", "ok", "null"], help="Filter op data_status")
    p_list.set_defaults(func=cmd_list_bad)

    p_deact = sub.add_parser("deactivate", help="Zet active=0 voor een lijst tickers")
    p_deact.add_argument("tickers", nargs="+")
    p_deact.add_argument("--reason", default="manual-triage")
    p_deact.set_defaults(func=cmd_deactivate)

    p_act = sub.add_parser("activate", help="Zet active=1 voor een lijst tickers")
    p_act.add_argument("tickers", nargs="+")
    p_act.set_defaults(func=cmd_activate)

    p_remap = sub.add_parser("remap", help="Atomair: deactiveer src, voeg dst toe (+fetch)")
    p_remap.add_argument("src")
    p_remap.add_argument("dst")
    p_remap.set_defaults(func=cmd_remap)

    p_ref = sub.add_parser("refresh", help="Force-refresh een lijst tickers (vereist CRON_TOKEN)")
    p_ref.add_argument("tickers", nargs="+")
    p_ref.set_defaults(func=cmd_refresh)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
