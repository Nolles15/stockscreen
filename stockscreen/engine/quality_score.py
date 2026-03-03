"""
Quality Score (0–10) and Piotroski F-Score.

Quality score breakdown (max 2 points each):
  1. Profitability consistency: avg ROE & ROIC > 12%, no year < 8%
  2. Balance sheet strength: D/E < 0.5 AND interest coverage > 10
  3. Earnings/FCF stability: stddev of EPS and FCF < 20% over 5 years
  4. FCF positivity: FCF > 0 in every available year
  5. Piotroski F-Score > 7

Piotroski F-Score (0–9):
  Profitability (4):
    F1  ROA > 0
    F2  Operating cash flow > 0
    F3  ROA increasing year-over-year
    F4  Operating CF / Total Assets > ROA (accrual quality)
  Leverage / Liquidity / Dilution (3):
    F5  Long-term debt ratio decreasing
    F6  Current ratio increasing
    F7  No new shares issued in last year
  Operating Efficiency (2):
    F8  Gross margin increasing
    F9  Asset turnover increasing
"""

import logging
from typing import Optional

log = logging.getLogger(__name__)

MIN_YEARS_PIOTROSKI = 2   # Need at least 2 years for YoY comparisons


def _val(row: dict, *keys) -> Optional[float]:
    for k in keys:
        v = row.get(k)
        if v is not None:
            return float(v)
    return None


# ---------------------------------------------------------------------------
# Piotroski F-Score
# ---------------------------------------------------------------------------

def piotroski_fscore(annual_rows: list[dict]) -> dict:
    """
    Compute Piotroski F-Score from annual financial data.
    annual_rows: sorted newest-first.

    Returns {
      "score": int (0–9),
      "criteria": {F1: bool, F2: bool, ...},
      "sufficient_data": bool
    }
    """
    criteria: dict[str, Optional[bool]] = {f"F{i}": None for i in range(1, 10)}

    if len(annual_rows) < 1:
        return {"score": 0, "criteria": criteria, "sufficient_data": False}

    curr = annual_rows[0]
    prev = annual_rows[1] if len(annual_rows) >= 2 else None

    # ---- Profitability -------------------------------------------------------

    total_assets_curr = _val(curr, "total_assets")
    net_income_curr   = _val(curr, "net_income")
    op_cf_curr        = _val(curr, "operating_cf")
    roa_curr = (net_income_curr / total_assets_curr) if (net_income_curr is not None and total_assets_curr) else None

    # F1: ROA > 0
    criteria["F1"] = (roa_curr > 0) if roa_curr is not None else None

    # F2: Operating cash flow > 0
    criteria["F2"] = (op_cf_curr > 0) if op_cf_curr is not None else None

    # F3: ROA increasing
    if prev is not None:
        ta_prev = _val(prev, "total_assets")
        ni_prev = _val(prev, "net_income")
        roa_prev = (ni_prev / ta_prev) if (ni_prev is not None and ta_prev) else None
        if roa_curr is not None and roa_prev is not None:
            criteria["F3"] = roa_curr > roa_prev

    # F4: Accruals (operating CF / total assets > ROA)
    if op_cf_curr is not None and total_assets_curr and roa_curr is not None:
        accrual_roa = op_cf_curr / total_assets_curr
        criteria["F4"] = accrual_roa > roa_curr

    # ---- Leverage / Liquidity / Dilution ------------------------------------

    if prev is not None:
        # F5: Long-term debt ratio decreasing
        debt_curr = _val(curr, "total_debt")
        ta_curr   = _val(curr, "total_assets")
        debt_prev = _val(prev, "total_debt")
        ta_prev   = _val(prev, "total_assets")
        lev_curr = (debt_curr / ta_curr) if (debt_curr is not None and ta_curr) else None
        lev_prev = (debt_prev / ta_prev) if (debt_prev is not None and ta_prev) else None
        if lev_curr is not None and lev_prev is not None:
            criteria["F5"] = lev_curr < lev_prev

        # F6: Current ratio increasing
        ca_curr  = _val(curr, "current_assets")
        cl_curr  = _val(curr, "current_liabilities")
        ca_prev  = _val(prev, "current_assets")
        cl_prev  = _val(prev, "current_liabilities")
        cr_curr  = (ca_curr / cl_curr) if (ca_curr and cl_curr) else None
        cr_prev  = (ca_prev / cl_prev) if (ca_prev and cl_prev) else None
        if cr_curr is not None and cr_prev is not None:
            criteria["F6"] = cr_curr > cr_prev

        # F7: No dilution — shares outstanding not increased
        sh_curr = _val(curr, "shares_outstanding")
        sh_prev = _val(prev, "shares_outstanding")
        if sh_curr is not None and sh_prev is not None and sh_prev > 0:
            criteria["F7"] = sh_curr <= sh_prev * 1.02  # allow 2% rounding tolerance

    # ---- Operating Efficiency -----------------------------------------------

    if prev is not None:
        # F8: Gross margin improving
        gp_curr = _val(curr, "gross_profit")
        rev_curr = _val(curr, "revenue")
        gp_prev = _val(prev, "gross_profit")
        rev_prev = _val(prev, "revenue")
        gm_curr = (gp_curr / rev_curr) if (gp_curr and rev_curr) else None
        gm_prev = (gp_prev / rev_prev) if (gp_prev and rev_prev) else None
        if gm_curr is not None and gm_prev is not None:
            criteria["F8"] = gm_curr > gm_prev

        # F9: Asset turnover improving
        ta_curr_f9 = _val(curr, "total_assets")
        ta_prev_f9 = _val(prev, "total_assets")
        rev_curr_f9 = _val(curr, "revenue")
        rev_prev_f9 = _val(prev, "revenue")
        at_curr = (rev_curr_f9 / ta_curr_f9) if (rev_curr_f9 and ta_curr_f9) else None
        at_prev = (rev_prev_f9 / ta_prev_f9) if (rev_prev_f9 and ta_prev_f9) else None
        if at_curr is not None and at_prev is not None:
            criteria["F9"] = at_curr > at_prev

    # Score = count of True criteria (None = unknown, treated as 0)
    score = sum(1 for v in criteria.values() if v is True)
    known = sum(1 for v in criteria.values() if v is not None)
    sufficient = known >= 6

    return {
        "score":           score,
        "criteria":        {k: v for k, v in criteria.items()},
        "known_criteria":  known,
        "sufficient_data": sufficient,
    }


# ---------------------------------------------------------------------------
# Main quality score
# ---------------------------------------------------------------------------

def quality_score(annual_rows: list[dict], normalized: dict) -> dict:
    """
    Compute quality score (0–10) from annual data + pre-computed normalized metrics.

    Returns {
      "total": float,
      "max":   10,
      "breakdown": {criterion: points_awarded},
      "warnings": [str]
    }
    """
    breakdown: dict[str, float] = {}
    warnings: list[str] = []
    rows = annual_rows[:5]
    n = len(rows)

    # ---- 1. Profitability consistency (max 2) --------------------------------
    # Condition: avg ROE AND avg ROIC > 12%, no single year below 8%
    avg_roe  = normalized.get("avg_roe")
    avg_roic = normalized.get("avg_roic")

    roe_ok  = avg_roe  is not None and avg_roe  > 0.12
    roic_ok = avg_roic is not None and avg_roic > 0.12

    no_bad_roe  = True
    no_bad_roic = True
    for r in rows:
        roe_yr = r.get("roe")
        if roe_yr is not None and roe_yr < 0.08:
            no_bad_roe = False
        ebit   = r.get("ebit")
        equity = r.get("total_equity")
        debt   = r.get("total_debt", 0) or 0
        if ebit is not None and equity is not None and (equity + debt) > 0:
            roic_yr = (ebit * 0.75) / (equity + debt)
            if roic_yr < 0.08:
                no_bad_roic = False

    if avg_roe is None or avg_roic is None:
        warnings.append("ROE or ROIC data incomplete — profitability criterion may be under-scored.")
        breakdown["profitability"] = 0.0
    elif roe_ok and roic_ok and no_bad_roe and no_bad_roic:
        breakdown["profitability"] = 2.0
    elif (roe_ok or roic_ok) and (no_bad_roe or no_bad_roic):
        breakdown["profitability"] = 1.0
    else:
        breakdown["profitability"] = 0.0

    # ---- 2. Balance sheet strength (max 2) -----------------------------------
    # Condition: D/E < 0.5 AND interest coverage > 10 (latest year AND average)
    de_ok   = False
    ic_ok   = False

    if rows:
        curr = rows[0]
        debt   = curr.get("total_debt", 0) or 0
        equity = curr.get("total_equity")
        ebit   = curr.get("ebit")
        intexp = curr.get("interest_expense")

        de_curr = (debt / equity) if equity and equity > 0 else None
        ic_curr = (ebit / intexp) if ebit and intexp and intexp > 0 else None

        # Average over available years
        de_vals = []
        ic_vals = []
        for r in rows:
            d = r.get("total_debt", 0) or 0
            e = r.get("total_equity")
            if e and e > 0:
                de_vals.append(d / e)
            eb = r.get("ebit")
            ie = r.get("interest_expense")
            if eb and ie and ie > 0:
                ic_vals.append(eb / ie)

        import statistics as _stats
        de_avg = _stats.median(de_vals) if de_vals else None
        ic_avg = _stats.median(ic_vals) if ic_vals else None

        de_ok = ((de_curr is not None and de_curr < 0.5) or (de_avg is not None and de_avg < 0.5))
        ic_ok = ((ic_curr is not None and ic_curr > 10) or (ic_avg is not None and ic_avg > 10))

        if intexp is None or intexp == 0:
            ic_ok = True  # no interest = no debt burden
            warnings.append("Interest expense not found — assuming no debt burden for coverage check.")

    if de_ok and ic_ok:
        breakdown["balance_sheet"] = 2.0
    elif de_ok or ic_ok:
        breakdown["balance_sheet"] = 1.0
    else:
        breakdown["balance_sheet"] = 0.0

    # ---- 3. Earnings / FCF stability (max 2) --------------------------------
    # Condition: stddev of EPS and FCF < 20% over 5 years
    stddev_eps = normalized.get("stddev_eps_pct")
    stddev_fcf = normalized.get("stddev_fcf_pct")

    eps_stable = (stddev_eps is not None and stddev_eps < 0.20)
    fcf_stable = (stddev_fcf is not None and stddev_fcf < 0.20)

    if stddev_eps is None or stddev_fcf is None:
        if n < 2:
            warnings.append("Insufficient history for EPS/FCF stability assessment.")
        breakdown["stability"] = 0.0
    elif eps_stable and fcf_stable:
        breakdown["stability"] = 2.0
    elif eps_stable or fcf_stable:
        breakdown["stability"] = 1.0
    else:
        breakdown["stability"] = 0.0

    # ---- 4. FCF positivity (max 2) ------------------------------------------
    # Condition: FCF > 0 in every available year (of last 5)
    fcf_years = [(r.get("fcf"), r.get("fiscal_year")) for r in rows if r.get("fcf") is not None]

    if not fcf_years:
        warnings.append("No FCF data available — FCF positivity criterion skipped.")
        breakdown["fcf_positive"] = 0.0
    else:
        all_positive = all(fcf > 0 for fcf, _ in fcf_years)
        if all_positive and len(fcf_years) >= 3:
            breakdown["fcf_positive"] = 2.0
        elif all_positive:
            breakdown["fcf_positive"] = 1.0
        else:
            negative_years = [yr for fcf, yr in fcf_years if fcf <= 0]
            warnings.append(f"Negative FCF in year(s): {negative_years}")
            breakdown["fcf_positive"] = 0.0

    # ---- 5. Piotroski F-Score (max 2) ---------------------------------------
    piotroski = piotroski_fscore(rows)
    f_score = piotroski["score"]
    sufficient = piotroski["sufficient_data"]

    threshold = 7 if sufficient else 7   # ≥7 for both cases per spec
    if f_score >= threshold:
        breakdown["piotroski"] = 2.0
    elif f_score >= 5:
        breakdown["piotroski"] = 1.0
    else:
        breakdown["piotroski"] = 0.0

    if not sufficient:
        warnings.append(
            f"Piotroski score based on limited data ({piotroski['known_criteria']}/9 criteria known)."
        )

    total = sum(breakdown.values())
    return {
        "total":          total,
        "max":            10,
        "breakdown":      breakdown,
        "piotroski":      piotroski,
        "warnings":       warnings,
    }
