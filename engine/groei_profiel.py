"""
Groeiprofiel: kan deze snelle groei gezond zijn?

Een zeef, geen koopoordeel. Het tabblad Groeiers selecteerde op het ene getal dat
het makkelijkst te kopen is met verlies (omzetgroei) en keek nergens naar wat de
groei kost. Dit profiel toetst zeven signalen uit de literatuur (zie
docs/plan-groeiprofiel.md voor de bronnen): verwatering, balansgroei ten opzichte
van omzetgroei, F-score, brutowinst op activa, accruals, stabiliteit van de groei
en R&D-intensiteit.

Groen betekent uitsluitend "de cijfers spreken niet tegen". Van bedrijven die drie
jaar hard groeien houdt maar een klein deel het volhouden; de ambitie is de
basiskans te verhogen, niet winnaars aan te wijzen.

De DREMPELS hieronder zijn VOORLOPIG tot de ijking (fase B van het plan) ze
vastzet; daarna verandert er alleen een constante. Niet voorlopig zijn de
datapoorten (MIN_RIJEN, data_status, omzetbreuk), de uitsluiting van de TTM-rij en
de regel dat een ontbrekende R&D-regel "onbekend" is en nooit nul.

Puur op meegegeven data, geen DB-toegang — zoals `moat_profile`.
"""

from __future__ import annotations

import statistics
from typing import Optional

from engine import exit_regels
from engine.moat_profile import MARGE_EROSIE_PP, _f, _trend, marge_reeks

# --- Datapoorten (niet voorlopig) ------------------------------------------
VENSTER_RIJEN = 4          # vier boekjaren = drie intervallen, als de omzetgroei
MIN_RIJEN = 3

# --- Voorlopige drempels ----------------------------------------------------
T1_ROOD = 5.0              # %/jaar groei van het aantal aandelen
T1_GEEL = 2.0
T1_AANDELENBREUK = 2.0     # sprong ≥ factor 2 tussen twee jaren = klassewissel/split
T2_ROOD_GAT = 15.0         # procentpunt: activa-CAGR minus omzet-CAGR
T2_ROOD_ACTIVA = 20.0      # én activa-CAGR zelf minstens zoveel %/jaar
T2_GEEL_GAT = 5.0
T3_ROOD = 3                # F-score ≤ 3
T3_GEEL = 5                # 4–5 geel, ≥ 6 groen
T3_MIN_BEKEND = 7          # van 9 criteria
T4_ROOD_NIVEAU = 0.10      # brutowinst / totale activa
T4_GROEN_NIVEAU = 0.33
T5_ROOD = 10.0             # accruals in % van de activa
T6_GROEN = 10.0            # stdev van de jaarlijkse omzetgroei, procentpunt
T6_ROOD = 25.0
T7_GROEN_ZONDER_SECTOR = 5.0   # R&D in % van de omzet
GROEN_SCORE = 7.0
GROEN_MIN_BEKEND = 4
MIN_BEKEND_SCORE = 3

# Gewicht per toets in de score. T5 en T7 tellen half: het bewijs is gemengd
# (accruals in Europa) respectievelijk asymmetrisch (R&D kan niet rood geven).
GEWICHT = {"T1": 1.5, "T2": 1.5, "T3": 1.0, "T4": 1.0, "T5": 0.5, "T6": 1.0, "T7": 0.5}
HARD = ("T1", "T2", "T3", "T4")
PUNTEN = {"groen": 1.0, "geel": 0.5, "rood": 0.0}

DREMPELS = {
    "venster_rijen": VENSTER_RIJEN, "min_rijen": MIN_RIJEN,
    "t1_rood": T1_ROOD, "t1_geel": T1_GEEL, "t1_aandelenbreuk": T1_AANDELENBREUK,
    "t2_rood_gat": T2_ROOD_GAT, "t2_rood_activa": T2_ROOD_ACTIVA, "t2_geel_gat": T2_GEEL_GAT,
    "t3_rood": T3_ROOD, "t3_geel": T3_GEEL, "t3_min_bekend": T3_MIN_BEKEND,
    "t4_rood_niveau": T4_ROOD_NIVEAU, "t4_groen_niveau": T4_GROEN_NIVEAU,
    "t4_marge_erosie_pp": MARGE_EROSIE_PP,
    "t5_rood": T5_ROOD,
    "t6_groen": T6_GROEN, "t6_rood": T6_ROOD,
    "t7_groen_zonder_sector": T7_GROEN_ZONDER_SECTOR,
    "groen_score": GROEN_SCORE, "groen_min_bekend": GROEN_MIN_BEKEND,
}


# ---------------------------------------------------------------------------
# Rijen en groei
# ---------------------------------------------------------------------------

def _venster(annual: list[dict]) -> list[dict]:
    """De laatste VENSTER_RIJEN boekjaren met omzet, oplopend.

    Zelfde regel als `screener._omzetvenster`: er is één getal voor "omzetgroei".
    De TTM-rij valt per constructie af (`fiscal_year` = 0) en niet-jaarrijen ook.
    """
    per_jaar: dict[int, dict] = {}
    for r in annual or []:
        if (r.get("period_type") or "annual") != "annual":
            continue
        jaar = r.get("fiscal_year")
        omzet = _f(r, "revenue")
        if not jaar or not omzet or omzet <= 0:
            continue
        per_jaar[int(jaar)] = r
    return [per_jaar[j] for j in sorted(per_jaar)][-VENSTER_RIJEN:]


def _cagr(eerste: Optional[float], laatste: Optional[float], jaren: int) -> Optional[float]:
    if eerste is None or laatste is None or eerste <= 0 or laatste <= 0 or jaren <= 0:
        return None
    return (laatste / eerste) ** (1 / jaren) - 1


def omzet_cagr(annual: list[dict]) -> Optional[float]:
    """Omzetgroei per jaar over het venster; None bij te weinig rijen of een omzetbreuk.

    Moet gelijk blijven aan `screener._calc_revenue_cagr` (bewaakt in de tests).
    """
    v = _venster(annual)
    if len(v) < 2 or exit_regels.omzetbreuk(v):
        return None
    return _cagr(_f(v[0], "revenue"), _f(v[-1], "revenue"),
                 v[-1]["fiscal_year"] - v[0]["fiscal_year"])


def _reeks(v: list[dict], sleutel: str, positief: bool = False) -> list[dict]:
    uit = []
    for r in v:
        w = _f(r, sleutel)
        if w is None or (positief and w <= 0):
            continue
        uit.append({"jaar": int(r["fiscal_year"]), "waarde": w})
    return uit


def _cagr_van(reeks: list[dict]) -> Optional[float]:
    if len(reeks) < 2:
        return None
    return _cagr(reeks[0]["waarde"], reeks[-1]["waarde"], reeks[-1]["jaar"] - reeks[0]["jaar"])


def _accruals(v: list[dict]) -> Optional[float]:
    """(netto winst − operationele kasstroom) / activa, gemiddeld, in procenten.

    Zelfde formule als `screener._calc_accruals`, alleen over boekjaren. Hier
    herimplementeerd: de screener importeert dit bestand, niet andersom.
    """
    r = []
    for rij in v:
        ni, ocf, ta = _f(rij, "net_income"), _f(rij, "operating_cf"), _f(rij, "total_assets")
        if ni is not None and ocf is not None and ta:
            r.append((ni - ocf) / ta)
    return 100.0 * sum(r) / len(r) if len(r) >= 2 else None


# ---------------------------------------------------------------------------
# Toetsen
# ---------------------------------------------------------------------------

def _toets(code, naam, uitkomst, waarde, drempel, bron, uitleg) -> dict:
    return {"code": code, "naam": naam, "uitkomst": uitkomst, "waarde": waarde,
            "drempel": drempel, "bron": bron, "uitleg": uitleg}


def _t1(v, dubbel_soort, d) -> tuple[dict, Optional[float]]:
    aandelen = _reeks(v, "shares_outstanding", positief=True)
    cagr = _cagr_van(aandelen)
    naam, bron = "Verwatering", "Pontiff & Woodgate 2008; McLean, Pontiff & Watanabe 2009"
    drempel = f"rood ≥ {d['t1_rood']:g}%/jr, geel ≥ {d['t1_geel']:g}%/jr"
    if len(aandelen) < 2:
        return _toets("T1", naam, "onbekend", None, drempel, bron,
                      "Te weinig jaren met een aandelenaantal"), None
    if dubbel_soort == "aandelenklasse":
        return _toets("T1", naam, "onbekend", None, drempel, bron,
                      "Aandelenklasse van een bedrijf met meer klassen — het aantal is niet "
                      "vergelijkbaar"), cagr
    for a, b in zip(aandelen, aandelen[1:]):
        f = b["waarde"] / a["waarde"]
        if f >= d["t1_aandelenbreuk"] or f <= 1 / d["t1_aandelenbreuk"]:
            return _toets("T1", naam, "onbekend", None, drempel, bron,
                          f"Aantal aandelen springt een factor {max(f, 1 / f):.1f} tussen "
                          f"{a['jaar']} en {b['jaar']} — klassewissel of splitsing, geen "
                          f"verwatering te meten"), None
    pct = 100 * cagr
    uit = "rood" if pct >= d["t1_rood"] else "geel" if pct >= d["t1_geel"] else "groen"
    return _toets("T1", naam, uit, round(pct, 1), drempel, bron,
                  f"Aantal aandelen groeit {pct:+.1f}% per jaar"), cagr


def _t2(v, omzet_g, d) -> tuple[dict, Optional[float]]:
    activa = _reeks(v, "total_assets", positief=True)
    naam, bron = "Balansgroei", "Cooper, Gulen & Schill 2008; Papanastasopoulos 2017"
    drempel = f"rood: gat ≥ {d['t2_rood_gat']:g} pp én activa ≥ {d['t2_rood_activa']:g}%/jr"
    if len(v) < 2 or len(activa) < 2 or activa[0]["jaar"] != int(v[0]["fiscal_year"]) \
            or activa[-1]["jaar"] != int(v[-1]["fiscal_year"]) or omzet_g is None:
        return _toets("T2", naam, "onbekend", None, drempel, bron,
                      "Balanstotaal ontbreekt in het eerste of laatste jaar"), None
    ag = _cagr_van(activa)
    gat = 100 * (ag - omzet_g)
    if gat >= d["t2_rood_gat"] and 100 * ag >= d["t2_rood_activa"]:
        uit = "rood"
    elif gat >= d["t2_geel_gat"]:
        uit = "geel"
    else:
        uit = "groen"
    return _toets("T2", naam, uit, round(gat, 1), drempel, bron,
                  f"Activa {100 * ag:+.1f}%/jr tegen omzet {100 * omzet_g:+.1f}%/jr "
                  f"(gat {gat:+.1f} pp)"), ag


def _t3(piotroski, d) -> dict:
    naam, bron = "F-score", "Piotroski 2000; Mohr 2012; Walkshäusl 2017"
    drempel = f"rood ≤ {d['t3_rood']}, geel ≤ {d['t3_geel']}, groen ≥ {d['t3_geel'] + 1}"
    p = piotroski or {}
    crit = p.get("criteria") or {}
    bekend = p.get("known_criteria")
    if bekend is None:
        bekend = sum(1 for x in crit.values() if x is not None)
    score = p.get("score")
    if score is None or bekend < d["t3_min_bekend"]:
        return _toets("T3", naam, "onbekend", None, drempel, bron,
                      f"Slechts {bekend} van 9 criteria te bepalen")
    uit = "rood" if score <= d["t3_rood"] else "geel" if score <= d["t3_geel"] else "groen"
    return _toets("T3", naam, uit, score, drempel, bron,
                  f"F-score {score} van 9 ({bekend} criteria bekend)")


def _t4(v, d) -> dict:
    naam, bron = "Brutowinstkracht", "Novy-Marx 2013"
    drempel = (f"rood: brutowinst/activa < {d['t4_rood_niveau']:.2f} of marge "
               f"{d['t4_marge_erosie_pp']:+g} pp; groen ≥ {d['t4_groen_niveau']:.2f}")
    laatste = v[-1]
    gp, ta = _f(laatste, "gross_profit"), _f(laatste, "total_assets")
    if gp is None or not ta or ta <= 0:
        return _toets("T4", naam, "onbekend", None, drempel, bron,
                      "Geen brutowinst of balanstotaal in het laatste jaar")
    niveau = gp / ta
    trend = _trend([w for _, w in marge_reeks(v, "gross_profit")])
    erosie = trend is not None and trend <= d["t4_marge_erosie_pp"]
    if niveau < d["t4_rood_niveau"] or erosie:
        uit = "rood"
    elif niveau >= d["t4_groen_niveau"]:
        uit = "groen"
    else:
        uit = "geel"
    tekst = f"Brutowinst is {100 * niveau:.0f}% van de activa"
    if trend is not None:
        tekst += f", brutomarge {trend:+.1f} pp over het venster"
    return _toets("T4", naam, uit, round(niveau, 3), drempel, bron, tekst)


def _t5(v, d) -> dict:
    naam, bron = "Accruals", "Sloan 1996; Walkshäusl 2022"
    drempel = f"rood ≥ +{d['t5_rood']:g}%, groen < 0"
    a = _accruals(v)
    if a is None:
        return _toets("T5", naam, "onbekend", None, drempel, bron,
                      "Minder dan twee jaren met winst, kasstroom en activa")
    uit = "rood" if a >= d["t5_rood"] else "geel" if a >= 0 else "groen"
    return _toets("T5", naam, uit, round(a, 1), drempel, bron,
                  f"Winst ligt gemiddeld {a:+.1f}% van de activa boven de kasstroom")


def _t6(v, d) -> tuple[dict, list[float]]:
    naam, bron = "Groeistabiliteit", "Mohanram 2005; Amor-Tapia & Tascón 2016"
    drempel = f"groen < {d['t6_groen']:g} pp, rood > {d['t6_rood']:g} pp (spreiding)"
    groei = []
    for a, b in zip(v, v[1:]):
        ra, rb = _f(a, "revenue"), _f(b, "revenue")
        if ra and rb and ra > 0:
            groei.append(100 * (rb / ra - 1))
    if len(groei) < 2:
        return _toets("T6", naam, "onbekend", None, drempel, bron,
                      "Minder dan twee jaar-op-jaar-groeicijfers"), groei
    s = statistics.pstdev(groei)
    uit = "groen" if s < d["t6_groen"] else "rood" if s > d["t6_rood"] else "geel"
    return _toets("T6", naam, uit, round(s, 1), drempel, bron,
                  "Jaarlijkse omzetgroei: " + ", ".join(f"{g:+.0f}%" for g in groei)), groei


def _t7(v, sector_mediaan_rd, d) -> tuple[dict, Optional[float]]:
    naam, bron = "R&D-intensiteit", "Chan, Lakonishok & Sougiannis 2001; Duqi e.a. 2015"
    grens = sector_mediaan_rd if sector_mediaan_rd is not None else d["t7_groen_zonder_sector"]
    drempel = f"groen ≥ {grens:g}% van de omzet; nooit rood"
    laatste = v[-1]
    rd, omzet = _f(laatste, "rd_expense"), _f(laatste, "revenue")
    if rd is None or rd <= 0 or not omzet:
        # Ontbrekend is niet nul: Yahoo geeft de regel alleen als het bedrijf hem meldt.
        return _toets("T7", naam, "onbekend", None, drempel, bron,
                      "R&D niet gerapporteerd — telt niet mee, ook niet negatief"), None
    pct = 100 * rd / omzet
    uit = "groen" if pct >= grens else "geel"
    return _toets("T7", naam, uit, round(pct, 1), drempel, bron,
                  f"R&D is {pct:.1f}% van de omzet"), pct


# ---------------------------------------------------------------------------
# Score en oordeel
# ---------------------------------------------------------------------------

def _score(toetsen: list[dict]) -> Optional[float]:
    bekend = [t for t in toetsen if t["uitkomst"] in PUNTEN]
    if len(bekend) < MIN_BEKEND_SCORE:
        return None
    noemer = sum(GEWICHT[t["code"]] for t in bekend)
    teller = sum(GEWICHT[t["code"]] * PUNTEN[t["uitkomst"]] for t in bekend)
    return round(10 * teller / noemer, 1)


def _oordeel(toetsen, score, aantal_rijen, data_status, breuk, cagr, groei_drempel,
             d) -> tuple[str, str]:
    """Geeft (niveau, kop). Eerste treffer wint."""
    if data_status in ("bad", "missing"):
        return "grijs", "Cijfers afgekeurd — geen groeiprofiel"
    if aantal_rijen < d["min_rijen"]:
        return "grijs", "Te weinig boekjaren om groei te beoordelen"
    if breuk:
        # Een definitiewissel bij Yahoo is een data-artefact, geen zwakte van het
        # bedrijf: grijs, niet rood (de Adyen-valse-vlag).
        return "grijs", "Definitiewissel in de omzetreeks — groei niet te meten"
    if cagr is None or cagr < groei_drempel:
        tekst = f" (omzet {100 * cagr:+.0f}%/jr)" if cagr is not None else ""
        return "grijs", f"Geen snelle groeier{tekst} — profiel ter informatie"
    uit = {t["code"]: t["uitkomst"] for t in toetsen}
    for code, kop in (("T1", "Groei wordt betaald met nieuwe aandelen"),
                      ("T2", "De balans groeit veel harder dan de omzet"),
                      ("T3", "Zwakke fundamentele trend (F-score ≤ 3)"),
                      ("T4", "Brutowinst te mager of brokkelt af")):
        if uit.get(code) == "rood":
            return "rood", kop
    zwak = [t["naam"] for t in toetsen if t["code"] in ("T5", "T6") and t["uitkomst"] == "rood"]
    if zwak:
        return "geel", "Eén zwak punt: " + " en ".join(n.lower() for n in zwak)
    bekend = sum(1 for t in toetsen if t["uitkomst"] in PUNTEN)
    if bekend >= d["groen_min_bekend"] and score is not None and score >= d["groen_score"]:
        return "groen", "De cijfers pleiten niet tegen de groei"
    return "geel", "De cijfers spreken zich niet uit"


def bouw_profiel(annual: list[dict], sector: str | None = None,
                 sector_mediaan_rd: float | None = None, *,
                 piotroski: dict | None = None, data_status: str | None = None,
                 groei_drempel: float = 0.15, drempels: dict | None = None,
                 omzetbreuk: str | None = None,
                 dubbel_soort: str | None = None) -> dict:
    """Bouwt het volledige groeiprofiel uit jaarrijen (nooit de TTM-rij).

    `piotroski` is `q_result["piotroski"]` uit `run_ticker`, zodat het profiel
    dezelfde F-score toont als de aandeelpagina. Zonder: berekend op de jaarrijen
    (backtest). Productie telt daarbij de TTM-rij mee, de backtest niet.
    `drempels` overschrijft constanten (alleen het backtestraster); wat werkelijk
    gebruikt is staat in de uitkomst.
    """
    d = {**DREMPELS, **(drempels or {})}
    v = _venster(annual)
    breuk = omzetbreuk if omzetbreuk is not None else exit_regels.omzetbreuk(v)
    cagr = omzet_cagr(annual)

    if piotroski is None and v:
        from engine.quality_score import piotroski_fscore
        piotroski = piotroski_fscore(list(reversed(v)))

    verlieslatend = bool(v) and (_f(v[-1], "net_income") or 0) < 0
    t1, aandelen_g = _t1(v, dubbel_soort, d) if v else (_toets("T1", "Verwatering", "onbekend", None, "", "", "Geen cijfers"), None)
    toetsen: list[dict] = []
    activa_g = None
    groei_per_jaar: list[float] = []
    rd_pct = None
    if v:
        t2, activa_g = _t2(v, cagr, d)
        t6, groei_per_jaar = _t6(v, d)
        t7, rd_pct = _t7(v, sector_mediaan_rd, d)
        toetsen = [t1, t2, _t3(piotroski, d), _t4(v, d), _t5(v, d), t6, t7]

    score = _score(toetsen)
    niveau, kop = _oordeel(toetsen, score, len(v), data_status, breuk, cagr, groei_drempel, d)

    omzet = _reeks(v, "revenue")
    aandelen = _reeks(v, "shares_outstanding", positief=True)
    per_aandeel = []
    for r in v:
        o, a = _f(r, "revenue"), _f(r, "shares_outstanding")
        if o and a and a > 0:
            per_aandeel.append({"jaar": int(r["fiscal_year"]), "waarde": o / a})
    # Omzet per aandeel is alleen betekenisvol zonder aandelenbreuk.
    opa_g = _cagr_van(per_aandeel) if t1["uitkomst"] != "onbekend" else None

    redenen = []
    if cagr is not None:
        redenen.append(f"Omzet groeit {100 * cagr:+.1f}% per jaar ({_venstertekst(v)})")
    elif breuk:
        redenen.append(breuk)
    for t in toetsen:
        if t["uitkomst"] != "onbekend":
            redenen.append(f"{t['naam']}: {t['uitleg']}")
    if opa_g is not None:
        redenen.append(f"Omzet per aandeel {100 * opa_g:+.1f}% per jaar")
    runway = None
    if v and verlieslatend:
        fcf, kas = _f(v[-1], "fcf"), _f(v[-1], "net_cash")
        if fcf is not None and fcf < 0:
            if kas is None or kas <= 0:
                redenen.append("Verlieslatend met negatieve kasstroom en geen nettokas")
            else:
                runway = round(kas / abs(fcf), 1)
                redenen.append(f"Kasbuffer dekt ongeveer {runway:.1f} jaar bij de huidige kasverbranding")
    if verlieslatend:
        redenen.append("Verlieslatend in het laatste boekjaar")

    marge_bruto = marge_reeks(v, "gross_profit")
    return {
        "niveau": niveau,
        "kop": kop,
        "score": score,
        "redenen": redenen,
        "toetsen": toetsen,
        "drempels": d,
        "series": {
            "omzet": omzet,
            "activa": _reeks(v, "total_assets"),
            "aandelen": aandelen,
            "brutomarge_pct": [{"jaar": j, "waarde": round(w, 1)} for j, w in marge_bruto],
            "rd_pct": [{"jaar": int(r["fiscal_year"]),
                        "waarde": round(100 * _f(r, "rd_expense") / _f(r, "revenue"), 1)}
                       for r in v if (_f(r, "rd_expense") or 0) > 0],
            "omzet_per_aandeel": [{"jaar": x["jaar"], "waarde": round(x["waarde"], 4)}
                                  for x in per_aandeel],
        },
        "omzet_cagr": cagr,
        "activa_cagr": activa_g,
        "aandelen_cagr": aandelen_g,
        "omzet_per_aandeel_cagr": opa_g,
        "groeier": cagr is not None and cagr >= groei_drempel,
        "verlieslatend": verlieslatend,
        "kasrunway_jaren": runway,
        "jaren": len(v),
        "venster": _venstertekst(v),
        "omzetbreuk": breuk,
        "rd_pct": rd_pct,
        "voorlopig": True,
    }


def _venstertekst(v: list[dict]) -> str:
    return f"{v[0]['fiscal_year']}–{v[-1]['fiscal_year']}" if v else ""


def compact(profiel: dict) -> dict:
    """Alleen scalars, voor `calculated_scores.groei_profiel` (json-baar, zonder series)."""
    return {
        "niveau": profiel["niveau"],
        "kop": profiel["kop"],
        "score": profiel["score"],
        "toetsen": {t["code"]: t["uitkomst"] for t in profiel["toetsen"]},
        "omzet_cagr": profiel["omzet_cagr"],
        "activa_cagr": profiel["activa_cagr"],
        "aandelen_cagr": profiel["aandelen_cagr"],
        "omzet_per_aandeel_cagr": profiel["omzet_per_aandeel_cagr"],
        "groeier": profiel["groeier"],
        "verlieslatend": profiel["verlieslatend"],
        "kasrunway_jaren": profiel["kasrunway_jaren"],
        "venster": profiel["venster"],
        "omzetbreuk": profiel["omzetbreuk"],
        "rd_pct": profiel["rd_pct"],
    }
