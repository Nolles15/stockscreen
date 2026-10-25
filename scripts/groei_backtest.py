"""
Backtest van het groeiprofiel op het eigen universum (fase B2 van docs/plan-groeiprofiel.md).

Vraag: scheiden de toetsen van `engine/groei_profiel.py` de snelle groeiers die het
volhielden van die het niet volhielden? Daarvoor wordt het profiel teruggerekend op
de cijfers zoals ze er op een vormingsdatum lagen, en vergeleken met wat er daarna
gebeurde — fundamenteel (winst + kwaliteit nu) en in de koers (ten opzichte van de
mediaan van het hele universum).

Gebruik:
    DATABASE_URL="..." python scripts/groei_backtest.py           # haalt op én schrijft de cache
    python scripts/groei_backtest.py                              # leest de cache
    python scripts/groei_backtest.py --out docs/groeiprofiel-ijking.md
    python scripts/groei_backtest.py --zelftest

De database wordt één keer geraakt (alleen lezen); daarna draait alles op
`data/groei_backtest_invoer.json`. `data/` valt buiten git en buiten de build.

Wat dit níet kan (en in het rapport staat):
- Er is geen historische `data_status` en geen historische kwaliteitsscore. De
  datapoort is een proxy (omzet, activa en aandelen gevuld op eerste en laatste
  rij) en de fundamentele uitkomst gebruikt de kwaliteitsscore van nu.
- De F-score wordt hier alleen over boekjaren berekend; productie telt de
  TTM-rij mee.
- Koersrendement is slotkoers op slotkoers, zonder dividend.
- Het 24-maandsvenster is één marktregime. Drempels niet aanscherpen op één cyclus.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import date, datetime
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import groei_profiel as gp  # noqa: E402

CACHE = "data/groei_backtest_invoer.json"
GROEI_DREMPEL = 0.15            # = screening.growth_lossmaker_cagr in config.yaml
KOERS_VENSTER_DAGEN = 14
NIET_FONDS = ("ETF", "MUTUALFUND")

# Vorming = de cijfers t/m boekjaar `fy`, gekend op `vorming` (≥ 6 maanden na het
# boekjaareinde). Uitkomsten 12 en 24 maanden later.
VORMINGEN = {
    "F2023": {"fy": 2023, "vorming": "2024-07-01", "uit12": "2025-07-01", "uit24": "2026-07-01"},
    "F2024": {"fy": 2024, "vorming": "2025-07-01", "uit12": "2026-07-01", "uit24": None},
}
UITKOMST_FY = 2025              # nieuwste volledige boekjaar = de fundamentele uitkomst
KWALITEIT_UITKOMST = 6.0

# Het drempelraster ligt vóóraf vast (plan B2). Per toets: de constante die
# bepaalt wanneer hij rood geeft, en de waarden die geprobeerd worden.
RASTER = {
    "T1": ("t1_rood", [3.0, 5.0, 8.0, 12.0]),
    "T2": ("t2_rood_gat", [10.0, 15.0, 20.0, 30.0]),
    "T4": ("t4_rood_niveau", [0.10, 0.15, 0.20]),
    "T6": ("t6_rood", [15.0, 25.0, 35.0]),
}
# Van streng naar ruim: de waarde die het minste rood geeft staat achteraan
# (T4 omgekeerd: een lagere vloer is ruimer).
RUIMSTE_EERST = {"T1": True, "T2": True, "T4": False, "T6": True}
MIN_N = 20
# Welke drempel wint als er meer slagen. Janco koos op 25 okt 2026 "strengste"
# (meer rood); "ruimste" geeft rood alleen aan de duidelijkste gevallen.
KEUZE = "strengste"


# ---------------------------------------------------------------------------
# Invoer
# ---------------------------------------------------------------------------

def _json_veilig(x):
    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, (date, datetime)):
        return x.isoformat()
    raise TypeError(type(x))


def _koersen_op(cur, datum: str) -> dict[str, float]:
    cur.execute(
        """SELECT DISTINCT ON (ticker) ticker, date, close FROM price_history
           WHERE date <= %s AND date >= (%s::date - %s)::text
           ORDER BY ticker, date DESC""",
        (datum, datum, KOERS_VENSTER_DAGEN),
    )
    return {r["ticker"]: float(r["close"]) for r in cur.fetchall() if r["close"]}


def haal_uit_database() -> dict:
    from engine import db

    rijen = db.jaarrijen_met_overrides_bulk()
    datums = sorted({d for v in VORMINGEN.values() for k, d in v.items() if k != "fy" and d})
    with db._cursor() as cur:
        cur.execute("""SELECT ticker, name, sector, quote_type, active,
                              auto_suspended_at, presumed_delisted_at FROM stocks""")
        stocks = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT ticker, quality_score, groei_niveau FROM calculated_scores")
        scores = {r["ticker"]: dict(r) for r in cur.fetchall()}
        cur.execute("SELECT ticker, data_status FROM data_quality")
        dq = {r["ticker"]: r["data_status"] for r in cur.fetchall()}
        koersen = {d: _koersen_op(cur, d) for d in datums}
    return {"opgehaald": datetime.now().isoformat(timespec="seconds"),
            "stocks": stocks, "rijen": rijen, "scores": scores,
            "data_status": dq, "koersen": koersen}


def laad(cache: str, vernieuw: bool) -> dict:
    if os.path.exists(cache) and not vernieuw:
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    if not os.environ.get("DATABASE_URL"):
        sys.exit(f"Geen cache in {cache} en geen DATABASE_URL: niets om op te rekenen.")
    data = haal_uit_database()
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(data, f, default=_json_veilig)
    print(f"Cache geschreven: {cache}", file=sys.stderr)
    return data


# ---------------------------------------------------------------------------
# Universum en uitkomsten
# ---------------------------------------------------------------------------

def _f(r, k):
    v = r.get(k)
    return float(v) if v is not None else None


def rijen_tot(rijen: list[dict], fy: int) -> list[dict]:
    return [r for r in rijen if r.get("fiscal_year") and 0 < int(r["fiscal_year"]) <= fy]


def in_universum(stock: dict, rijen_asof: list[dict], fy: int) -> tuple[bool, str]:
    """Dezelfde poort als de motor, met een proxy voor data_status."""
    if (stock.get("quote_type") or "").upper() in NIET_FONDS:
        return False, "fonds"
    if stock.get("sector") in gp.NIET_TOEGEPAST_SECTOREN:
        return False, "bank/verzekeraar"
    v = gp._venster(rijen_asof)
    if len(v) < gp.MIN_RIJEN:
        return False, "te weinig jaren"
    if int(v[-1]["fiscal_year"]) != fy:
        return False, "laatste boekjaar ontbreekt"
    for r in (v[0], v[-1]):
        if not all(_f(r, k) for k in ("revenue", "total_assets", "shares_outstanding")):
            return False, "datapoort (proxy)"
    cagr = gp.omzet_cagr(rijen_asof)
    if cagr is None:
        return False, "omzetbreuk"
    if cagr < GROEI_DREMPEL:
        return False, "geen snelle groeier"
    return True, ""


def _weg(stock: dict) -> bool:
    return bool(stock.get("presumed_delisted_at") or stock.get("auto_suspended_at"))


def fundamentele_uitkomst(stock, rijen, score, fy) -> dict:
    """Winst in FY2025 én kwaliteit ≥ 6 nu; en of de groei aanhield (≥ 10%/jr)."""
    jaren = {int(r["fiscal_year"]): r for r in rijen if r.get("fiscal_year")}
    nu = jaren.get(UITKOMST_FY)
    if nu is None:
        if _weg(stock):
            return {"fund": False, "groei": False, "weg": True}
        return {"fund": None, "groei": None, "weg": False}
    ni = _f(nu, "net_income")
    q = (score or {}).get("quality_score")
    if ni is None:
        fund = None
    elif ni <= 0:
        fund = False
    else:
        fund = None if q is None else q >= KWALITEIT_UITKOMST
    groei = None
    oud = jaren.get(fy)
    if oud and _f(oud, "revenue") and _f(nu, "revenue") and _f(oud, "revenue") > 0 and _f(nu, "revenue") > 0:
        g = (_f(nu, "revenue") / _f(oud, "revenue")) ** (1 / (UITKOMST_FY - fy)) - 1
        groei = g >= 0.10
    return {"fund": fund, "groei": groei, "weg": False}


def koersverhoudingen(koersen: dict, van: str, tot: str) -> dict[str, float]:
    a, b = koersen.get(van, {}), koersen.get(tot, {})
    uit = {}
    for t, p0 in a.items():
        p1 = b.get(t)
        if p0 and p1:
            r = p1 / p0
            if 0.01 < r < 100:      # daarbuiten: eenheids- of splitfout, geen rendement
                uit[t] = r
    return uit


def koersuitkomst(t, stock, koersen, van, tot, verh, mediaan) -> dict:
    if tot is None:
        return {"ok": None, "rel": None}
    r = verh.get(t)
    if r is None:
        if koersen.get(van, {}).get(t) and _weg(stock):
            r = 0.5               # verdwenen: conservatief −50%
        else:
            return {"ok": None, "rel": None}
    rel = r - mediaan
    return {"ok": rel > 0, "rel": rel}


# ---------------------------------------------------------------------------
# Rekenen
# ---------------------------------------------------------------------------

def bouw_cohort(data: dict, naam: str) -> tuple[list[dict], dict]:
    vf = VORMINGEN[naam]
    fy = vf["fy"]
    k = data["koersen"]
    v12 = koersverhoudingen(k, vf["vorming"], vf["uit12"])
    v24 = koersverhoudingen(k, vf["vorming"], vf["uit24"]) if vf["uit24"] else {}
    med12 = statistics.median(v12.values()) if v12 else 1.0
    med24 = statistics.median(v24.values()) if v24 else 1.0

    cohort, uitval = [], {}
    for s in data["stocks"]:
        t = s["ticker"]
        rijen = data["rijen"].get(t) or []
        asof = rijen_tot(rijen, fy)
        ok, reden = in_universum(s, asof, fy)
        if not ok:
            uitval[reden] = uitval.get(reden, 0) + 1
            continue
        cohort.append({
            "ticker": t, "stock": s, "asof": asof,
            "geprijsd": t in k.get(vf["vorming"], {}),
            **fundamentele_uitkomst(s, rijen, data["scores"].get(t), fy),
            "k12": koersuitkomst(t, s, k, vf["vorming"], vf["uit12"], v12, med12),
            "k24": koersuitkomst(t, s, k, vf["vorming"], vf["uit24"], v24, med24),
        })
    meta = {"fy": fy, "n": len(cohort), "uitval": uitval,
            "geprijsd": sum(1 for c in cohort if c["geprijsd"]),
            "universum_geprijsd": len(v12), "mediaan12": med12, "mediaan24": med24,
            "weg": sum(1 for c in cohort if c["weg"])}
    return cohort, meta


def profiel(c: dict, drempels: dict | None = None) -> dict:
    return gp.bouw_profiel(c["asof"], sector=c["stock"].get("sector"),
                           groei_drempel=GROEI_DREMPEL, drempels=drempels)


def _trefkans(groep: list[dict], sleutel) -> tuple[int, float | None]:
    w = [sleutel(c) for c in groep]
    w = [x for x in w if x is not None]
    return len(w), (sum(1 for x in w if x) / len(w) if w else None)


def _mediaan_rel(groep, k) -> float | None:
    w = [c[k]["rel"] for c in groep if c[k]["rel"] is not None]
    return statistics.median(w) if w else None


def maten(groep: list[dict]) -> dict:
    nf, f = _trefkans(groep, lambda c: c["fund"])
    ng, g = _trefkans(groep, lambda c: c["groei"])
    n12, k12 = _trefkans(groep, lambda c: c["k12"]["ok"])
    n24, k24 = _trefkans(groep, lambda c: c["k24"]["ok"])
    return {"n": len(groep), "nf": nf, "fund": f, "ng": ng, "groei": g,
            "n12": n12, "k12": k12, "rel12": _mediaan_rel(groep, "k12"),
            "n24": n24, "k24": k24, "rel24": _mediaan_rel(groep, "k24")}


def splits_op_toets(cohort, profielen, code) -> dict[str, list]:
    uit = {"rood": [], "niet-rood": [], "onbekend": []}
    for c, p in zip(cohort, profielen):
        u = next(t["uitkomst"] for t in p["toetsen"] if t["code"] == code)
        uit["rood" if u == "rood" else "onbekend" if u == "onbekend" else "niet-rood"].append(c)
    return uit


def scheidt(m_rood: dict, m_niet: dict) -> tuple[bool, str]:
    """Keuzeregel: n ≥ 20 in rood én strikt slechter op fundament én koers (12m)."""
    if m_rood["nf"] < MIN_N or m_rood["n12"] < MIN_N:
        return False, f"rood te klein (n fund {m_rood['nf']}, koers {m_rood['n12']})"
    for sl, naam in (("fund", "fundament"), ("k12", "koers 12m")):
        if m_rood[sl] is None or m_niet[sl] is None or not m_rood[sl] < m_niet[sl]:
            return False, f"rood niet slechter op {naam}"
    return True, "scheidt"


def _slaagt(per: dict) -> bool:
    """Slaagt in minstens één vorming en wordt in geen enkele tegengesproken.

    Een vorming waar de rode groep te klein is, beslist niet mee. Zonder die regel
    kan F2023 (33 groeiers, want de meeste tickers hebben pas cijfers vanaf FY2022)
    elke toets in zijn eentje laten zakken.
    """
    oordelen = [p["oordeel"] for p in per.values()]
    beslissend = [ok for ok, reden in oordelen if not reden.startswith("rood te klein")]
    return bool(beslissend) and all(beslissend)


def _drempels_voor(code: str, waarde: float) -> dict:
    sleutel, _ = RASTER[code]
    d = {sleutel: waarde}
    if code == "T1":
        d["t1_geel"] = min(gp.T1_GEEL, waarde)
    if code == "T2":
        d["t2_kan_rood"] = True       # het raster meet juist wat rood zou doen
    if code == "T6":
        d["t6_groen"] = min(gp.T6_GROEN, waarde)
    return d


def analyse(data: dict) -> dict:
    cohorts = {n: bouw_cohort(data, n) for n in VORMINGEN}
    res = {"cohorts": {n: m for n, (_, m) in cohorts.items()}, "raster": {}, "vast": {},
           "niveau": {}, "score": {}}

    huidig = {n: [profiel(c) for c in co] for n, (co, _) in cohorts.items()}

    for code, (sleutel, waarden) in RASTER.items():
        rij = []
        for w in waarden:
            per = {}
            for n, (co, _) in cohorts.items():
                pr = [profiel(c, _drempels_voor(code, w)) for c in co]
                s = splits_op_toets(co, pr, code)
                mr, mn = maten(s["rood"]), maten(s["niet-rood"])
                per[n] = {"rood": mr, "niet": mn, "onbekend": len(s["onbekend"]),
                          "oordeel": scheidt(mr, mn)}
            rij.append({"waarde": w, "per": per, "slaagt": _slaagt(per)})
        ruim_eerst = RUIMSTE_EERST[code]
        volgorde = sorted(rij, key=lambda r: r["waarde"],
                          reverse=ruim_eerst if KEUZE == "ruimste" else not ruim_eerst)
        gekozen = next((r["waarde"] for r in volgorde if r["slaagt"]), None)
        res["raster"][code] = {"sleutel": sleutel, "rij": rij, "gekozen": gekozen,
                               "huidig": gp.DREMPELS[sleutel]}

    for code in ("T3", "T5", "T7"):
        per = {}
        for n, (co, _) in cohorts.items():
            s = splits_op_toets(co, huidig[n], code)
            if code == "T7":
                # R&D kan niet rood geven: vergelijk groen met de rest.
                groen = [c for c, p in zip(co, huidig[n])
                         if next(t["uitkomst"] for t in p["toetsen"] if t["code"] == "T7") == "groen"]
                rest = [c for c in co if c not in groen]
                per[n] = {"rood": maten(rest), "niet": maten(groen), "onbekend": 0,
                          "oordeel": (None, "groen tegen de rest")}
            else:
                mr, mn = maten(s["rood"]), maten(s["niet-rood"])
                per[n] = {"rood": mr, "niet": mn, "onbekend": len(s["onbekend"]),
                          "oordeel": scheidt(mr, mn)}
        res["vast"][code] = {"per": per, "slaagt": None if code == "T7" else _slaagt(per)}

    for n, (co, _) in cohorts.items():
        groepen: dict[str, list] = {}
        scoregroep: dict[str, list] = {}
        for c, p in zip(co, huidig[n]):
            groepen.setdefault(p["niveau"], []).append(c)
            s = p["score"]
            b = "onbekend" if s is None else "< 5" if s < 5 else "5 – 7" if s < 7 else "≥ 7"
            scoregroep.setdefault(b, []).append(c)
        res["niveau"][n] = {k: maten(v) for k, v in groepen.items()}
        res["score"][n] = {k: maten(v) for k, v in scoregroep.items()}
        res["alle"] = res.get("alle", {})
        res["alle"][n] = maten(co)

    res["wat_als"] = wat_als(data, res)
    return res


def wat_als(data: dict, res: dict) -> dict:
    """Verdeling op het huidige universum, met de huidige en de gekozen drempels.

    Benadering: zonder dubbel_soort, met de F-score over alleen boekjaren. De
    echte telling komt uit /api/scores/recompute met dry_run.
    """
    gekozen = {}
    for code, r in res["raster"].items():
        if r["gekozen"] is not None:
            gekozen.update(_drempels_voor(code, r["gekozen"]))
    tel = {"huidig": {}, "gekozen": {}}
    for s in data["stocks"]:
        if not s.get("active") or s.get("presumed_delisted_at"):
            continue
        if (s.get("quote_type") or "").upper() in NIET_FONDS:
            continue
        rijen = data["rijen"].get(s["ticker"]) or []
        cagr = gp.omzet_cagr(rijen)
        if cagr is None or cagr < GROEI_DREMPEL:
            continue
        ds = data["data_status"].get(s["ticker"])
        for naam, d in (("huidig", None), ("gekozen", gekozen)):
            p = gp.bouw_profiel(rijen, sector=s.get("sector"), data_status=ds,
                                groei_drempel=GROEI_DREMPEL, drempels=d)
            tel[naam][p["niveau"]] = tel[naam].get(p["niveau"], 0) + 1
    tel["drempels"] = gekozen
    return tel


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------

def _pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def _rel(x):
    return "–" if x is None else f"{100 * x:+.0f} pp"


def _maatcellen(m):
    return (f"{m['n']} | {_pct(m['fund'])} ({m['nf']}) | {_pct(m['groei'])} | "
            f"{_pct(m['k12'])} ({m['n12']}) | {_rel(m['rel12'])} | "
            f"{_pct(m['k24'])} ({m['n24']}) | {_rel(m['rel24'])}")


KOP = ("| n | fundament ok (n) | groei hield | koers 12m > mediaan (n) | mediaan rel. 12m "
       "| koers 24m > mediaan (n) | mediaan rel. 24m |")
SCHEIDING = "|---:|---:|---:|---:|---:|---:|---:|"


def rapport(res: dict, opgehaald: str) -> str:
    L = ["# Groeiprofiel — ijking op het eigen universum (fase B2)", "",
         f"Gegenereerd {datetime.now():%Y-%m-%d %H:%M} met `scripts/groei_backtest.py`, "
         f"data opgehaald {opgehaald}.", "",
         "**Leeswijzer.** Per vorming is het profiel teruggerekend op de cijfers t/m dat "
         "boekjaar. *Fundament ok* = winst in FY2025 én kwaliteitsscore ≥ 6 nu. *Groei "
         "hield* = omzet groeide daarna nog ≥ 10%/jr. *Koers > mediaan* = slotkoers-op-"
         "slotkoers beter dan de mediaan van het hele geprijsde universum (geen dividend). "
         "Verdwenen tickers tellen als −50% en als fundamenteel mislukt.", "",
         f"**Keuzeregel (vooraf vastgelegd).** Een drempel mag rood geven als de rode groep "
         f"n ≥ {MIN_N} heeft en strikt slechter scoort dan niet-rood op fundament én koers "
         f"12m. Een vorming waarin de rode groep te klein is, beslist niet mee; in de andere mag geen tegenspraak zitten. Slagen er meerdere, dan wint de **{KEUZE}** "
         "(strengste = die het meest rood geeft). Slaagt er geen, dan wordt de "
         "toets geel-only.", "",
         "## Cohorten", "",
         "| vorming | snelle groeiers | met koers op vorming | verdwenen | mediaan universum 12m | 24m |",
         "|---|---:|---:|---:|---:|---:|"]
    for n, m in res["cohorts"].items():
        L.append(f"| {n} (t/m FY{m['fy']}, {VORMINGEN[n]['vorming']}) | {m['n']} | "
                 f"{m['geprijsd']} ({_pct(m['geprijsd'] / m['n'] if m['n'] else None)}) | "
                 f"{m['weg']} | {m['mediaan12']:.3f} | "
                 f"{m['mediaan24']:.3f} |" if VORMINGEN[n]["uit24"] else
                 f"| {n} (t/m FY{m['fy']}, {VORMINGEN[n]['vorming']}) | {m['n']} | "
                 f"{m['geprijsd']} ({_pct(m['geprijsd'] / m['n'] if m['n'] else None)}) | "
                 f"{m['weg']} | {m['mediaan12']:.3f} | – |")
    L.append("")
    for n, m in res["cohorts"].items():
        L.append(f"Uitval {n}: " + ", ".join(f"{k} {v}" for k, v in
                                             sorted(m["uitval"].items(), key=lambda x: -x[1])))
    L += ["", "## Basiskans: alle snelle groeiers", "", "| vorming " + KOP, "|---" + SCHEIDING]
    for n, m in res["alle"].items():
        L.append(f"| {n} | {_maatcellen(m)} |")

    L += ["", "## Eindoordeel met de huidige drempels", ""]
    for n in res["niveau"]:
        L += [f"**{n}**", "", "| niveau " + KOP, "|---" + SCHEIDING]
        for niv in ("groen", "geel", "rood", "grijs"):
            if niv in res["niveau"][n]:
                L.append(f"| {niv} | {_maatcellen(res['niveau'][n][niv])} |")
        L.append("")
        L += ["| score " + KOP, "|---" + SCHEIDING]
        for b in ("≥ 7", "5 – 7", "< 5", "onbekend"):
            if b in res["score"][n]:
                L.append(f"| {b} | {_maatcellen(res['score'][n][b])} |")
        L.append("")

    L += ["## Drempelraster per toets", ""]
    for code, r in res["raster"].items():
        L += [f"### {code} — `{r['sleutel']}` (nu {r['huidig']:g})", "",
              "| waarde | vorming | groep " + KOP + " oordeel |",
              "|---:|---|---" + SCHEIDING + "---|"]
        for rij in r["rij"]:
            for n, p in rij["per"].items():
                L.append(f"| {rij['waarde']:g} | {n} | rood | {_maatcellen(p['rood'])} | "
                         f"{p['oordeel'][1]} |")
                L.append(f"| | | niet-rood | {_maatcellen(p['niet'])} | onbekend {p['onbekend']} |")
        g = r["gekozen"]
        L += ["", f"**Gekozen:** {g:g}" if g is not None else
              "**Geen drempel slaagt → geel-only (kan geen rood meer geven).**", ""]

    L += ["## Toetsen zonder raster (huidige drempel)", ""]
    for code, v in res["vast"].items():
        per = v["per"]
        groep = ("rest", "groen") if code == "T7" else ("rood", "niet-rood")
        L += [f"### {code}", "", "| vorming | groep " + KOP + " oordeel |",
              "|---|---" + SCHEIDING + "---|"]
        for n, p in per.items():
            L.append(f"| {n} | {groep[0]} | {_maatcellen(p['rood'])} | {p['oordeel'][1]} |")
            L.append(f"| | {groep[1]} | {_maatcellen(p['niet'])} | onbekend {p['onbekend']} |")
        if v["slaagt"] is not None:
            L += ["", "**Scheidt met de huidige drempel.**" if v["slaagt"] else
                  "**Scheidt niet met de huidige drempel.**"]
        L.append("")

    w = res["wat_als"]
    L += ["## Wat als: het huidige universum", "",
          "Benadering (F-score zonder TTM-rij, zonder aandelenklasse-markering); de echte "
          "telling komt uit `/api/scores/recompute` met `dry_run`.", "",
          f"Gekozen drempels: `{json.dumps(w['drempels'])}`", "",
          "| niveau | huidig | gekozen |", "|---|---:|---:|"]
    for niv in ("groen", "geel", "rood", "grijs"):
        L.append(f"| {niv} | {w['huidig'].get(niv, 0)} | {w['gekozen'].get(niv, 0)} |")
    tot = sum(w["huidig"].values()) or 1
    L += ["", f"Groen nu {_pct(w['huidig'].get('groen', 0) / tot)}, met de gekozen drempels "
          f"{_pct(w['gekozen'].get('groen', 0) / tot)}.", "",
          "## Beperkingen", "",
          "- Geen historische `data_status` en kwaliteitsscore: de poort is een proxy, de "
          "fundamentele uitkomst leunt op de kwaliteitsscore van nu (milde look-ahead op de "
          "poort, geen op de uitkomst).",
          "- F-score alleen over boekjaren; productie telt de TTM-rij mee.",
          "- Koers zonder dividend; verhoudingen buiten 0,01–100 zijn weggelaten als datafout.",
          "- Eén marktregime (2024-07 → 2026-07); F2024 heeft alleen 12 maanden.",
          "- De meeste tickers hebben pas cijfers vanaf FY2021, dus F2023 is een venster van "
          "drie jaar en F2022 is niet te vormen.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Zelftest (zonder database)
# ---------------------------------------------------------------------------

def _zelftest() -> None:
    def rij(j, omzet, aandelen, ni=10.0, activa=None, gpw=None):
        return {"fiscal_year": j, "period_type": "annual", "revenue": omzet,
                "total_assets": activa or omzet, "shares_outstanding": aandelen,
                "net_income": ni, "gross_profit": gpw if gpw is not None else 0.4 * omzet,
                "operating_cf": ni * 1.2, "total_equity": omzet / 2, "total_debt": omzet / 5,
                "current_assets": omzet / 2, "current_liabilities": omzet / 4}

    stocks, rijen, k = [], {}, {d: {} for d in ("2024-07-01", "2025-07-01", "2026-07-01")}
    for i in range(60):
        t = f"T{i}.AS"
        verwatert = i % 2 == 0
        stocks.append({"ticker": t, "sector": "Technology", "quote_type": "EQUITY",
                       "active": 1, "auto_suspended_at": None, "presumed_delisted_at": None})
        aandelen = [100 * (1.15 ** n if verwatert else 1.0) for n in range(5)]
        rijen[t] = [rij(2021 + n, 100 * 1.3 ** n, aandelen[n],
                        ni=-5.0 if (verwatert and n == 4) else 10.0) for n in range(5)]
        for n, d in enumerate(k):
            k[d][t] = 10 * ((0.8 if verwatert else 1.3) ** n)
    stocks.append({"ticker": "ETF.AS", "sector": None, "quote_type": "ETF", "active": 1})
    data = {"stocks": stocks, "rijen": rijen, "koersen": k, "data_status": {},
            "scores": {s["ticker"]: {"quality_score": 7.0} for s in stocks}}

    co, meta = bouw_cohort(data, "F2023")
    assert meta["n"] == 60, meta
    assert meta["uitval"].get("fonds") == 1, meta
    assert all(len(gp._venster(c["asof"])) == 3 for c in co), "F2023 hoort alleen FY ≤ 2023 te zien"
    res = analyse(data)
    t1 = res["raster"]["T1"]
    assert t1["gekozen"] == (3.0 if KEUZE == "strengste" else 12.0), t1["gekozen"]
    assert rapport(res, "zelftest").startswith("# Groeiprofiel")
    print("Zelftest geslaagd.")


def main() -> None:
    global KEUZE
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--vernieuw", action="store_true", help="cache negeren, opnieuw ophalen")
    ap.add_argument("--out", help="rapport naar dit bestand (anders stdout)")
    ap.add_argument("--keuze", choices=("strengste", "ruimste"), default="strengste")
    ap.add_argument("--zelftest", action="store_true")
    a = ap.parse_args()
    KEUZE = a.keuze
    if a.zelftest:
        _zelftest()
        return
    data = laad(a.cache, a.vernieuw)
    for n, vf in VORMINGEN.items():
        co, meta = bouw_cohort(data, n)
        if meta["n"] and meta["geprijsd"] / meta["n"] < 0.6:
            sys.exit(f"{n}: slechts {meta['geprijsd']} van {meta['n']} groeiers hebben een koers "
                     f"op {vf['vorming']}. Eerst POST /api/price-history/backfill "
                     f"{{\"period\":\"5y\"}} draaien.")
    tekst = rapport(analyse(data), data.get("opgehaald", "?"))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(tekst)
        print(f"Rapport geschreven: {a.out}", file=sys.stderr)
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(tekst)


if __name__ == "__main__":
    main()
