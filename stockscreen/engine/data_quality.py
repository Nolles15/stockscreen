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


def evaluate(
    ticker: str,
    annual_rows: list[dict],
    market_data: dict | None,
    stock_info: dict | None,
    fetch_success: bool,
    prev_consecutive_failures: int = 0,
    fetched_date: str | None = None,
) -> dict:
    """
    Evalueer data-kwaliteit voor één ticker. Return dict klaar voor
    `db.upsert_data_quality(**result)`.
    """
    issues: list[str] = []

    # 1. Fetch slaagde niet (404, lege info, netwerk kapot na retries)
    if not fetch_success and not annual_rows and not market_data:
        return {
            "completeness_pct": 0.0,
            "years_available":  0,
            "latest_fy":        None,
            "freshness_days":   None,
            "fetch_success":    0,
            "consecutive_failures": prev_consecutive_failures + 1,
            "data_status":      "missing",
            "issues":           ["Yahoo Finance gaf geen data terug — ticker lijkt ongeldig of onbereikbaar."],
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
        severe_unit_mismatch = False
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
    has_blocker = (
        years_available == 0
        or not mkt.get("price")
        or (latest_row and latest_row.get("total_equity") is not None and latest_row["total_equity"] <= 0)
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
        if prev_consecutive_failures + 1 >= 3:
            status = "bad"

    return {
        "completeness_pct": round(completeness_pct, 1),
        "years_available":  years_available,
        "latest_fy":        latest_fy,
        "freshness_days":   freshness_days,
        "fetch_success":    0 if fail_note else 1,
        "consecutive_failures": (prev_consecutive_failures + 1) if fail_note else 0,
        "data_status":      status,
        "issues":           issues,
        "last_checked":     datetime.utcnow().isoformat(),
    }
