"""
Normalizer — produces stable, outlier-resistant 5-year averages.

Strategy:
  1. Collect up to 5 years of annual values.
  2. Winsorize: remove values > N×IQR beyond Q1/Q3 (default N=3).
  3. Return the median of the remaining values as the normalized figure.
     (Median is more robust than mean for small samples.)
"""

import logging
import statistics
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def winsorize(values: list[float], iqr_multiplier: float = 3.0) -> list[float]:
    """
    Remove extreme outliers beyond Q1 - N×IQR or Q3 + N×IQR.
    Returns the cleaned list (never shorter than 1 element if input is non-empty).
    """
    if len(values) < 3:
        return values
    arr = np.array(values, dtype=float)
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    cleaned = [v for v in values if lower <= v <= upper]
    return cleaned if cleaned else values  # never return empty


def safe_median(values: list[Optional[float]]) -> Optional[float]:
    """Median of non-None, non-NaN values. Returns None if no valid values."""
    clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not clean:
        return None
    return statistics.median(clean)


def normalize_metric(annual_rows: list[dict], field: str, iqr_multiplier: float = 3.0) -> Optional[float]:
    """
    Extract `field` from each annual row, winsorize, return median.
    annual_rows must be sorted newest-first.
    """
    raw = [r.get(field) for r in annual_rows[:5]]
    values = [v for v in raw if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not values:
        return None
    cleaned = winsorize(values, iqr_multiplier)
    return statistics.median(cleaned)


def normalize_all(annual_rows: list[dict], iqr_multiplier: float = 3.0) -> dict:
    """
    Compute all normalized metrics needed for valuation.

    Returns dict with:
      normalized_eps, normalized_ebitda, normalized_fcf,
      normalized_owner_earnings, avg_roe, avg_roic,
      stddev_eps, stddev_fcf, years_used
    """
    if not annual_rows:
        return {}

    rows = annual_rows[:5]  # max 5 years

    norm_eps    = normalize_metric(rows, "eps_diluted",  iqr_multiplier)
    norm_ebitda = normalize_metric(rows, "ebitda",       iqr_multiplier)
    norm_fcf    = normalize_metric(rows, "fcf",          iqr_multiplier)

    # Owner Earnings ≈ FCF (as per spec; add R&D capitalisation only if data is clear)
    norm_oe = norm_fcf

    # Per-share owner earnings for perpetuity formula
    # We'll use per-share FCF = FCF / shares_outstanding
    fcf_ps_values = []
    for r in rows:
        fcf = r.get("fcf")
        shares = r.get("shares_outstanding")
        if fcf is not None and shares and shares > 0:
            fcf_ps_values.append(fcf / shares)
    norm_oe_ps = safe_median(winsorize(fcf_ps_values, iqr_multiplier)) if fcf_ps_values else None

    # ROE and ROIC
    roe_values = []
    roic_values = []
    for r in rows:
        roe = r.get("roe")
        if roe is not None:
            roe_values.append(roe)

        # ROIC = EBIT × (1-t) / Invested Capital
        # Invested Capital = Equity + Debt - overtollige Cash
        # Overtollige cash aftrekken voorkomt dat cash-rijke bedrijven (Apple, MIPS)
        # onterecht laag scoren doordat idle cash de noemer opblaast.
        ebit     = r.get("ebit")
        equity   = r.get("total_equity")
        debt     = r.get("total_debt", 0) or 0
        net_cash = r.get("net_cash", 0) or 0
        excess_cash = max(0.0, net_cash)   # alleen positieve nettocash aftrekken
        invested_capital = (equity or 0) + debt - excess_cash
        if ebit is not None and equity is not None and invested_capital > 0:
            nopat = ebit * 0.75
            roic_values.append(nopat / invested_capital)

    avg_roe  = safe_median(roe_values)  if roe_values  else None
    avg_roic = safe_median(roic_values) if roic_values else None

    # StdDev of EPS and FCF (for quality score stability check)
    eps_vals = [r.get("eps_diluted") for r in rows if r.get("eps_diluted") is not None]
    fcf_vals = [r.get("fcf")         for r in rows if r.get("fcf") is not None]

    stddev_eps = (statistics.pstdev(eps_vals) / abs(statistics.mean(eps_vals))
                  if len(eps_vals) >= 2 and statistics.mean(eps_vals) != 0 else None)
    stddev_fcf = (statistics.pstdev(fcf_vals) / abs(statistics.mean(fcf_vals))
                  if len(fcf_vals) >= 2 and statistics.mean(fcf_vals) != 0 else None)

    return {
        "normalized_eps":          norm_eps,
        "normalized_ebitda":       norm_ebitda,
        "normalized_fcf":          norm_fcf,
        "normalized_owner_earnings": norm_oe,
        "normalized_oe_per_share": norm_oe_ps,
        "avg_roe":                 avg_roe,
        "avg_roic":                avg_roic,
        "stddev_eps_pct":          stddev_eps,   # as fraction (0.2 = 20%)
        "stddev_fcf_pct":          stddev_fcf,
        "years_used":              len(rows),
    }


def historical_median_multiple(historical_multiples: list[dict], field: str,
                                iqr_multiplier: float = 3.0) -> Optional[float]:
    """Return the median historical multiple (e.g. pe_ratio) from stored yearly data."""
    values = [r.get(field) for r in historical_multiples if r.get(field) is not None and r.get(field) > 0]
    if not values:
        return None
    cleaned = winsorize(values, iqr_multiplier)
    return statistics.median(cleaned)
