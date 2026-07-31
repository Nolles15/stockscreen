"""
Data quality evaluator.

Na elke fetch draait `evaluate()` om vast te leggen hoe bruikbaar de data is:
  • completeness_pct  — hoeveel kritieke velden aanwezig zijn
  • freshness_days    — hoe oud de laatste succesvolle fetch is
  • consistency       — controles op nonsens-waarden
  • data_status       — afgeleide vlag ('ok' | 'warning' | 'bad' | 'missing')
  • issues            — lijst met concrete probleemomschrijvingen

Het resultaat wordt opgeslagen in `data_quality` en gelezen door:
  • de screener — `bad`/`missing` → INSUFFICIENT_DATA ipv doorrekenen op garbage
  • het dashboard — kolom + filter zodat je meteen ziet welke tickers ruis zijn
"""

from __future__ import annotations

import logging
from datetime import datetime, date

log = logging.getLogger(__name__)


# Kritieke velden voor de normalize/valuate stap. Ontbreekt er te veel, dan
# is de berekening niet meer betrouwbaar.
_CRITICAL_ANNUAL_FIELDS = (
    "revenue",
    "ebit",
    "ebitda",
    "net_income",
    "eps_diluted",
    "operating_cf",
    "fcf",
    "total_equity",
    "total_debt",
    "shares_outstanding",
)

_CRITICAL_MARKET_FIELDS = (
    "price",
    "market_cap",
)

# Drempels
_MIN_YEARS_OK      = 3
_MIN_COMPLETENESS_OK      = 80.0   # >= 80% + geen blockers → ok
_MIN_COMPLETENESS_WARNING = 50.0   # 50-80% → warning
_STALE_DAYS = 120                  # > 120 dagen zonder verse data → warning

# Instrumenten waarvoor de fundamentele FV-pipeline niet werkt (geen
# inkomstenstroom / eigen vermogen per aandeel). Deze worden als 'missing'
# gemarkeerd zodat de screener ze overslaat — alleen common stock is zinvol.
_NON_EQUITY_QUOTE_TYPES = {"ETF", "MUTUALFUND", "INDEX", "CURRENCY", "CRYPTOCURRENCY", "FUTURE", "OPTION"}

# Structurele winstgevendheid — aanhoudend verlies of negatieve FCF
# ondermijnt alle cashflow-gebaseerde methodes (Graham, perpetuity, EV/FCF).
_LOSS_STREAK_YEARS = 3             # >=N aaneengesloten jaren negative net_income → bad
_NEG_FCF_STREAK_YEARS = 3          # >=N aaneengesloten jaren negative FCF → bad

# Stock-split detectie: een niet-verwerkte split deelt EPS mechanisch door een
# heel split-getal (2:1, 3:1, 10:1, ...). De oude check (elke YoY-ratio ≥3×)
# vuurde echter massaal vals op kleine EPS-bases, herstel-uit-verlies en
# cyclische winstswings (zie docs/DIAGNOSE_INSUFFICIENT_DATA.md). Daarom nu:
#   1. absolute EPS-floor — negeer ratio's waarbij de kleinste EPS minuscuul is
#   2. clean-factor — alleen flaggen als de ratio dicht bij een écht split-getal ligt
# Bovendien is split_suspected géén harde blocker meer (alleen een waarschuwing);
# de downstream FV-plausibiliteitsgate in screener.py vangt absurde FV's alsnog.
_SPLIT_EPS_RATIO = 3.0
_SPLIT_EPS_MIN_ABS = 0.30          # kleinste EPS in het paar moet ≥ dit (native ccy)
_SPLIT_CLEAN_FACTORS = (2.0, 3.0, 4.0, 5.0, 10.0)
_SPLIT_FACTOR_TOL = 0.06           # ratio moet binnen 6% van een split-getal liggen


def _clean_split_factor(ratio: float) -> float | None:
    """Geef het split-getal terug als `ratio` (of 1/ratio) er dicht bij ligt, anders None."""
    for f in _SPLIT_CLEAN_FACTORS:
        for cand in (f, 1.0 / f):
            if abs(ratio - cand) / cand <= _SPLIT_FACTOR_TOL:
                return f
    return None


def evaluate(
    ticker: str,
    annual_rows: list[dict],
    market_data: dict | None,
    stock_info: dict | None,
    fetch_success: bool,
    prev_consecutive_failures: int = 0,
    fetched_date: str | None = None,
    count_failure: bool = True,
) -> dict:
    """
    Evalueer data-kwaliteit voor één ticker. Return dict klaar voor
    `db.upsert_data_quality(**result)`.

    `count_failure=False` laat de failure-teller ongemoeid bij een mislukte
    fetch. De batch-refresh telt zelf, en alleen als blijkt dat het niet om een
    storing bij de bron ging — zie engine/refresh.py.
    """
    issues: list[str] = []

    # Bij count_failure=False blijft de teller staan waar hij stond.
    failed_count = prev_consecutive_failures + 1 if count_failure else prev_consecutive_failures

    # 1. Fetch slaagde niet (404, lege info, netwerk kapot na retries)
    if not fetch_success and not annual_rows and not market_data:
        return {
            "completeness_pct": 0.0,
            "years_available":  0,
            "latest_fy":        None,
            "freshness_days":   None,
            "fetch_success":    0,
            "consecutive_failures": failed_count,
            "data_status":      "missing",
            "issues":           ["Yahoo Finance gaf geen data terug — ticker lijkt ongeldig of onbereikbaar."],
            "last_checked":     datetime.utcnow().isoformat(),
        }

    # 1c. Non-equity instrument (ETF / fund / index / FX) — FV-pipeline werkt
    # hier niet op, dus direct als 'missing' classificeren. De screener slaat
    # zulke tickers dan automatisch over ipv SELL-signalen op een ETF te tonen.
    stk = stock_info or {}
    quote_type = (stk.get("quote_type") or "").upper()
    if quote_type in _NON_EQUITY_QUOTE_TYPES:
        return {
            "completeness_pct": 0.0,
            "years_available":  len(annual_rows or []),
            "latest_fy":        None,
            "freshness_days":   None,
            "fetch_success":    1 if fetch_success else 0,
            "consecutive_failures": 0,
            "data_status":      "missing",
            "issues":           [f"Instrument-type '{quote_type}' niet ondersteund door fundamentele screener — deactiveer of filter handmatig."],
            "last_checked":     datetime.utcnow().isoformat(),
        }

    # 1b. Fetch mislukte maar oude data staat nog in de DB — doorgaan met evaluatie
    # van wat we hebben, maar wel een issue + consecutive_failures bijhouden.
    fail_note: str | None = None
    if not fetch_success:
        fail_note = (
            f"Laatste fetch faalde (pogingen: {prev_consecutive_failures + 1}) — "
            f"data hieronder komt uit een eerdere succesvolle refresh."
        )

    # 2. Fetch slaagde (minstens gedeeltelijk) — tel completeness
    years_available = len(annual_rows)
    latest_fy = None
    if annual_rows:
        latest_fy = max((r.get("fiscal_year") or 0) for r in annual_rows) or None

    # Completeness over alle kritieke velden (gemiddelde vulgraad in laatste 3 jaar)
    filled, total = 0, 0
    recent_rows = sorted(
        [r for r in annual_rows if r.get("fiscal_year")],
        key=lambda r: r.get("fiscal_year") or 0,
        reverse=True,
    )[:3]
    for row in recent_rows:
        for f in _CRITICAL_ANNUAL_FIELDS:
            total += 1
            if row.get(f) is not None:
                filled += 1
    mkt = market_data or {}
    for f in _CRITICAL_MARKET_FIELDS:
        total += 1
        if mkt.get(f) is not None:
            filled += 1
    completeness_pct = (filled / total * 100.0) if total > 0 else 0.0

    # 3. Specifieke consistentie-checks → issues[]
    if years_available == 0:
        issues.append("Geen jaarcijfers gevonden — normalisatie en FV niet mogelijk.")
    elif years_available < _MIN_YEARS_OK:
        issues.append(f"Slechts {years_available} jaar historie — minimaal {_MIN_YEARS_OK} nodig voor betrouwbare waardering.")

    if not mkt.get("price"):
        issues.append("Geen koers beschikbaar — signaal kan niet berekend worden.")
    if not mkt.get("market_cap"):
        issues.append("Market cap ontbreekt — grootte-filter werkt niet.")

    # Financial currency mismatch detectie (ADR-problematiek)
    stk = stock_info or {}
    fin_ccy = stk.get("financial_currency")
    trd_ccy = stk.get("currency")
    if fin_ccy and trd_ccy and fin_ccy != trd_ccy:
        # Dit is op zich niet fout (BRK-B, TM, IBN), maar wel iets om te weten
        issues.append(f"ADR/dual-currency: handelt in {trd_ccy}, rapporteert in {fin_ccy} — check FX op FV.")

    # Negatief/nul eigen vermogen = insolvent of rare boekhouding
    latest_row = recent_rows[0] if recent_rows else None
    severe_unit_mismatch = False
    if latest_row:
        te = latest_row.get("total_equity")
        if te is not None and te <= 0:
            issues.append(f"Eigen vermogen FY{latest_row.get('fiscal_year')} = {te:.0f} (≤0) — P/B en Graham-FV niet bruikbaar.")
        rev = latest_row.get("revenue")
        if rev is not None and rev <= 0:
            issues.append(f"Omzet FY{latest_row.get('fiscal_year')} = {rev:.0f} (≤0) — fundamenteel onbetrouwbaar.")

        # Market-cap vs shares * price consistentie-check
        # Factor-2+ afwijking duidt bijna altijd op een schaal-bug (shares uit balance
        # sheet vs info.sharesOutstanding, pence/pound, etc.) → escalate naar bad.
        shares = latest_row.get("shares_outstanding")
        price = mkt.get("price")
        mc = mkt.get("market_cap")
        if shares and price and mc and shares > 0 and mc > 0:
            implied_mc = shares * price
            # Voor ADRs met andere financial_currency slaat deze check niet aan
            # omdat shares/mc in verschillende valuta's zitten. Skip dan.
            if not (fin_ccy and trd_ccy and fin_ccy != trd_ccy):
                ratio = max(implied_mc, mc) / min(implied_mc, mc)
                if ratio > 2.0:
                    severe_unit_mismatch = True
                    issues.append(
                        f"Market cap SEVERE mismatch: shares×price ≈ {implied_mc/1e6:.0f}M, "
                        f"Yahoo = {mc/1e6:.0f}M (factor {ratio:.2f}x) — "
                        f"vermoedelijke shares/pence/FX schaal-bug."
                    )
                elif ratio > 1.2:
                    issues.append(
                        f"Market cap inconsistent: shares×price ≈ {implied_mc/1e6:.0f}M, "
                        f"Yahoo = {mc/1e6:.0f}M (verschil {(ratio-1)*100:.0f}%)."
                    )

        # EV consistentie: enterprise_value ≈ market_cap + net_debt
        ev = mkt.get("enterprise_value")
        total_debt = latest_row.get("total_debt")
        total_cash = latest_row.get("total_cash")
        if ev and mc and ev > 0 and mc > 0 and total_debt is not None:
            net_debt = (total_debt or 0) - (total_cash or 0)
            implied_ev = mc + net_debt
            if implied_ev > 0:
                ev_ratio = max(implied_ev, ev) / min(implied_ev, ev)
                if ev_ratio > 1.5:
                    issues.append(
                        f"EV inconsistent: mcap+net_debt ≈ {implied_ev/1e6:.0f}M, "
                        f"Yahoo EV = {ev/1e6:.0f}M (factor {ev_ratio:.2f}x)."
                    )

    # 3b. Structurele winstgevendheid — streaks van verlies of negatieve FCF
    # maken de cashflow-methodes onbetrouwbaar ongeacht completeness.
    loss_streak = 0
    neg_fcf_streak = 0
    for row in recent_rows:
        ni = row.get("net_income")
        if ni is not None and ni < 0:
            loss_streak += 1
        else:
            break
    # Tel losjes over alle beschikbare jaren (niet alleen recent_rows=3 voor streak-lengte)
    full_rows = sorted(
        [r for r in annual_rows if r.get("fiscal_year")],
        key=lambda r: r.get("fiscal_year") or 0,
        reverse=True,
    )
    loss_streak_full = 0
    for row in full_rows:
        ni = row.get("net_income")
        if ni is not None and ni < 0:
            loss_streak_full += 1
        else:
            break
    neg_fcf_streak_full = 0
    for row in full_rows:
        fcf = row.get("fcf")
        if fcf is not None and fcf < 0:
            neg_fcf_streak_full += 1
        else:
            break

    structural_loss = loss_streak_full >= _LOSS_STREAK_YEARS
    structural_neg_fcf = neg_fcf_streak_full >= _NEG_FCF_STREAK_YEARS
    if structural_loss:
        issues.append(
            f"Structureel verlies: {loss_streak_full} aaneengesloten jaren negative net_income — "
            f"EPS-gebaseerde FV (Graham, P/E) niet bruikbaar."
        )
    if structural_neg_fcf:
        issues.append(
            f"Structureel negatieve FCF: {neg_fcf_streak_full} aaneengesloten jaren — "
            f"perpetuity en EV/FCF niet bruikbaar."
        )

    # 3c. Stock-split detectie via EPS YoY ratio (alleen waarschuwing, geen blocker)
    split_suspected = False
    for i in range(len(full_rows) - 1):
        eps_now = full_rows[i].get("eps_diluted")
        eps_prev = full_rows[i + 1].get("eps_diluted")
        if eps_now and eps_prev and eps_now > 0 and eps_prev > 0:
            ratio = eps_now / eps_prev
            # Filter 1: kleinste EPS moet boven de floor — kleine bases blazen
            # de ratio kunstmatig op (bv. 0.05 → 0.22 = 4.4x is gewoon groei).
            if min(eps_now, eps_prev) < _SPLIT_EPS_MIN_ABS:
                continue
            # Filter 2: ratio moet dicht bij een écht split-getal liggen. Cyclische
            # of herstel-swings (5.9x, 13x, 7.7x) zijn geen mechanische splits.
            factor = _clean_split_factor(ratio)
            if factor is not None:
                split_suspected = True
                issues.append(
                    f"Verdachte EPS-sprong FY{full_rows[i].get('fiscal_year')}→FY{full_rows[i+1].get('fiscal_year')}: "
                    f"{eps_prev:.2f} → {eps_now:.2f} (ratio {ratio:.1f}x ≈ {factor:.0f}:1) — mogelijk niet-verwerkte split; "
                    f"controleer normalisatie (waarschuwing, niet blokkerend)."
                )
                break

    # 4. Freshness
    freshness_days: int | None = None
    if fetched_date:
        try:
            d = datetime.fromisoformat(fetched_date[:10]).date()
            freshness_days = (date.today() - d).days
            if freshness_days > _STALE_DAYS:
                issues.append(f"Data is {freshness_days} dagen oud — een refresh kan nieuwe cijfers ophalen.")
        except (ValueError, TypeError):
            pass

    # 5. Afgeleide status
    #
    # Harde blockers = data is écht onbruikbaar (geen jaren, geen koers, omzet ≤0,
    # schaal-bug). De zachte heuristieken (structureel verlies/FCF, split, negatief
    # eigen vermogen) zijn GÉÉN harde blocker meer: ze laten vaak nog bruikbare
    # multiples-methodes (P/B, EV/EBITDA) toe. De screener heeft een downstream
    # vangnet (FV-plausibiliteitsgate + min. 2 valide methodes) dat de echt
    # onwaardeerbare gevallen alsnog op INSUFFICIENT DATA zet. Zie
    # docs/DIAGNOSE_INSUFFICIENT_DATA.md (Route A).
    #
    # Negatief eigen vermogen blokkeert alleen bij een verlieslatend bedrijf:
    # winstgevende bedrijven met negatief EV door aandeleninkoop (bv. HP, Edenred)
    # zijn gewoon waardeerbaar op EPS/EV-multiples.
    ni_latest = latest_row.get("net_income") if latest_row else None
    profitable = ni_latest is not None and ni_latest > 0
    insolvent_equity = (
        latest_row is not None
        and latest_row.get("total_equity") is not None
        and latest_row["total_equity"] <= 0
        and not profitable
    )
    has_blocker = (
        years_available == 0
        or not mkt.get("price")
        or insolvent_equity
        or (latest_row and latest_row.get("revenue") is not None and latest_row["revenue"] <= 0)
        or severe_unit_mismatch
    )
    if has_blocker or completeness_pct < _MIN_COMPLETENESS_WARNING:
        status = "bad"
    elif completeness_pct < _MIN_COMPLETENESS_OK or years_available < _MIN_YEARS_OK:
        status = "warning"
    elif issues:   # niet-blokkerend maar wel iets te melden
        status = "warning" if len(issues) > 1 else "ok"
    else:
        status = "ok"

    if fail_note:
        issues.insert(0, fail_note)
        # Na 3+ mislukte fetches in een rij klasseren we als 'bad' — deze ticker
        # is waarschijnlijk delisted of verkeerd gespeld ondanks dat er oude
        # data in de DB staat.
        if failed_count >= 3:
            status = "bad"

    return {
        "completeness_pct": round(completeness_pct, 1),
        "years_available":  years_available,
        "latest_fy":        latest_fy,
        "freshness_days":   freshness_days,
        "fetch_success":    0 if fail_note else 1,
        "consecutive_failures": failed_count if fail_note else 0,
        "data_status":      status,
        "issues":           issues,
        "last_checked":     datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Diagnostiek — issue-strings classificeren in één primaire blocker-categorie.
#
# Read-only hulpmiddel voor /api/gaps-report en scripts/gaps_analyze.py. Raakt
# de drempels of de gate in evaluate() NIET aan; het categoriseert alleen de al
# opgeslagen `issues`-strings zodat 500 tickers in ~10 telbare buckets vallen.
#
# De fingerprints hieronder zijn substrings van de teksten die evaluate() zelf
# genereert (zie hierboven) — blijven dus in sync zolang die teksten kloppen.
# ---------------------------------------------------------------------------

# Blokkerende categorieën, op volgorde van "harde, echte datafout" → "zachte
# heuristiek". De primaire blocker is de eerste match in deze volgorde, zodat de
# heuristiek-buckets (split / structureel) precies de tickers bevatten waarvan
# dát het énige obstakel is — de zuiverste rescue-kandidaten.
_BLOCKER_FINGERPRINTS: tuple[tuple[str, str], ...] = (
    ("non_equity",           "niet ondersteund door fundamentele screener"),
    ("no_data",              "gaf geen data terug"),
    ("no_years",             "Geen jaarcijfers gevonden"),
    ("no_price",             "Geen koers beschikbaar"),
    ("negative_equity",      "Eigen vermogen"),
    ("negative_revenue",     "Omzet FY"),
    ("unit_mismatch_severe", "Market cap SEVERE mismatch"),
    ("structural_neg_fcf",   "Structureel negatieve FCF"),
    ("structural_loss",      "Structureel verlies"),
    ("split_suspected",      "Verdachte EPS-sprong"),
)

# Niet-blokkerende signalen — wel diagnostisch nuttig, maar leiden op zichzelf
# niet tot status 'bad'. Bv. ADR/dual-currency wijst op een notatie/databron-fix.
_INFO_FINGERPRINTS: tuple[tuple[str, str], ...] = (
    ("ev_inconsistent",      "EV inconsistent"),
    ("unit_mismatch_light",  "Market cap inconsistent"),
    ("adr_dual_currency",    "ADR/dual-currency"),
    ("no_market_cap",        "Market cap ontbreekt"),
    ("few_years",            "jaar historie"),
    ("stale",                "dagen oud"),
    ("fetch_failed",         "Laatste fetch faalde"),
)


def classify_blockers(issues: list[str] | None, data_status: str | None) -> dict:
    """
    Leid uit de opgeslagen `issues` + `data_status` één primaire blocker-categorie
    af, plus alle matchende blocker/info-categorieën en de bewijstekst.

    Pure functie — geen DB, geen drempels. Return:
      {
        "primary_blocker": str,        # bv. "split_suspected", "ok", "low_completeness"
        "blockers":  [str, ...],       # alle blokkerende categorieën die matchen
        "info_flags": [str, ...],      # niet-blokkerende signalen
        "evidence":  str | None,       # issue-tekst die de primaire blocker triggerde
      }
    """
    issues = issues or []

    matched: list[tuple[str, str]] = []   # (categorie, bewijstekst) in prioriteitsvolgorde
    for cat, fp in _BLOCKER_FINGERPRINTS:
        hit = next((s for s in issues if fp in s), None)
        if hit:
            matched.append((cat, hit))
    info = [cat for cat, fp in _INFO_FINGERPRINTS if any(fp in s for s in issues)]

    # Alleen tickers die écht geblokkeerd zijn (bad/missing) krijgen een blocker-
    # bucket. Een 'ok'/'warning'-ticker kan na Route A nog wel een (niet-
    # blokkerende) heuristiek-waarschuwing dragen — die hoort bij info_flags, niet
    # bij blockers, zodat geredde tickers niet in een blocker-bucket blijven hangen.
    is_blocked = data_status in ("bad", "missing")

    if is_blocked and matched:
        blockers = [c for c, _ in matched]
        primary, evidence = matched[0]
    elif is_blocked:
        # 'bad'/'missing' zonder herkenbare blocker-issue = completeness < 50%.
        blockers, primary, evidence = [], "low_completeness", None
    else:
        # Niet geblokkeerd: matched-categorieën zijn niet-blokkerende waarschuwingen.
        blockers = []
        info = [c for c, _ in matched] + info
        primary, evidence = (data_status or "unknown"), None

    return {
        "primary_blocker": primary,
        "blockers": blockers,
        "info_flags": info,
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# Reden-categorisatie voor het dashboard — vertaalt het ene "INSUFFICIENT DATA"
# label naar drie heldere buckets zodat zichtbaar is WAAROM er geen signaal is.
# Pure functie; geen gate-/drempelwijziging.
#
#   geen_data  — bron leverde niets / niet-equity / geen koers / te leeg → Route B (databron)
#   databug    — schaal/eenheid/dual-listing-mismatch of FV 10x+ van koers → fixen
#   geen_fv    — data compleet & solvabel maar verlieslatend → <2 methodes; of insolvent
# ---------------------------------------------------------------------------

_REASON_LABELS = {
    "geen_data": "GEEN DATA",
    "databug":   "DATABUG",
    "geen_fv":   "GEEN FV (VERLIES)",
}
_REASON_COLORS = {
    "geen_data": "slate",
    "databug":   "purple",
    "geen_fv":   "amber",
}

# primary_blocker (uit classify_blockers) → reden-bucket
_BLOCKER_TO_REASON = {
    "no_data":              "geen_data",
    "non_equity":           "geen_data",
    "no_years":             "geen_data",
    "no_price":             "geen_data",
    "low_completeness":     "geen_data",
    "unit_mismatch_severe": "databug",
    "negative_revenue":     "databug",
    "negative_equity":      "geen_fv",   # insolvent = verlies-gedreven
}


def classify_signal_reason(
    signal: str | None,
    data_status: str | None,
    issues: list[str] | None,
    fv_ratio_oob: bool = False,
    fv_methods_used: int | None = None,
) -> dict:
    """
    Bepaal WAAROM een ticker geen bruikbaar signaal heeft. Voor normale signalen
    (BUY/HOLD/SELL/…) is reason_code None. Pure functie.

    Return: {"reason_code": str|None, "reason_label": str|None, "reason_color": str|None}
    """
    none = {"reason_code": None, "reason_label": None, "reason_color": None}
    if signal != "INSUFFICIENT DATA":
        return none

    if fv_ratio_oob:
        code = "databug"
    elif data_status in ("bad", "missing"):
        primary = classify_blockers(issues, data_status)["primary_blocker"]
        code = _BLOCKER_TO_REASON.get(primary, "geen_data")
    else:
        # Data passeert de gate, maar het screener-vangnet sloeg toe (<2 valide
        # methodes door verlies). Data is er; het bedrijf is (nog) niet te waarderen.
        code = "geen_fv"

    return {
        "reason_code":  code,
        "reason_label": _REASON_LABELS[code],
        "reason_color": _REASON_COLORS[code],
    }
