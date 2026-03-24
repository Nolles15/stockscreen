"""
Screener — orchestrates the full pipeline for one or all tickers.

Pipeline per ticker:
  1. Load annual financials + market data from DB
  2. Apply manual overrides
  3. Run normalizer
  4. Run quality score
  5. Run valuation
  6. Determine signal
  7. Persist calculated scores to DB
  8. Return result dict ready for the dashboard
"""

import logging
from datetime import datetime
from typing import Optional

from . import db
from .normalizer import normalize_all, historical_median_multiple
from .quality_score import quality_score as calc_quality
from .valuation import combined_fair_value
from .data_fetcher import get_eur_rate

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal logic
# ---------------------------------------------------------------------------

def determine_signal(
    price_eur: float,
    combined_fv_eur: float,
    quality: float,
    config: dict,
) -> dict:
    """
    Returns {"signal": str, "margin_of_safety": float, "price_vs_fv_pct": float}
    """
    sig_cfg  = config.get("signals", {})
    strong_buy_pct = sig_cfg.get("strong_buy_pct", 60) / 100.0
    buy_pct        = sig_cfg.get("buy_pct",        75) / 100.0
    hold_upper_pct = sig_cfg.get("hold_upper_pct", 115) / 100.0
    sell_pct       = sig_cfg.get("sell_pct",       130) / 100.0
    sell_q_floor   = sig_cfg.get("sell_quality_floor", 6)
    min_quality    = config.get("screening", {}).get("min_quality_score", 7)

    price_vs_fv = price_eur / combined_fv_eur     # < 1 = undervalued
    mos = (1 - price_vs_fv) * 100                 # positive = discount, negative = premium

    if quality < sell_q_floor:
        signal = "SELL"
    elif quality >= 8 and price_vs_fv <= strong_buy_pct:
        signal = "STRONG BUY"
    elif quality >= min_quality and price_vs_fv <= buy_pct:
        signal = "BUY"
    elif price_vs_fv <= hold_upper_pct:
        signal = "HOLD"
    elif price_vs_fv > sell_pct:
        signal = "SELL"
    else:
        signal = "HOLD"   # 115%–130% zone: cautious hold

    return {
        "signal":          signal,
        "margin_of_safety": round(mos, 1),
        "price_vs_fv_pct": round(price_vs_fv * 100, 1),
    }


# ---------------------------------------------------------------------------
# Helper calculations
# ---------------------------------------------------------------------------

def _relative_pct(current, median) -> Optional[float]:
    """Return % deviation of current from median. Negative = cheaper than history."""
    if current is None or median is None or median == 0:
        return None
    return round((current - median) / abs(median) * 100, 1)


def _calc_hist_relative(hist_mult: list[dict], market_data: dict, iqr_mult: float) -> dict:
    """Compare current EV/EBITDA, P/E, P/B to own historical medians."""
    med_eveb = historical_median_multiple(hist_mult, "ev_ebitda", iqr_mult)
    med_pe   = historical_median_multiple(hist_mult, "pe_ratio",  iqr_mult)
    med_pb   = historical_median_multiple(hist_mult, "pb_ratio",  iqr_mult)

    import math

    def _finite(v):
        """Vervang Infinity/NaN door None — browsers kunnen dit niet parsen als JSON."""
        return None if (v is not None and isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v

    mkt = market_data or {}
    curr_eveb = _finite(mkt.get("ev_ebitda_ttm"))
    curr_pe   = _finite(mkt.get("pe_ttm"))
    curr_pb   = _finite(mkt.get("pb_ratio"))

    return {
        "ev_ebitda_pct":     _relative_pct(curr_eveb, med_eveb),
        "pe_pct":            _relative_pct(curr_pe,   med_pe),
        "pb_pct":            _relative_pct(curr_pb,   med_pb),
        "median_ev_ebitda":  med_eveb,
        "median_pe":         med_pe,
        "median_pb":         med_pb,
        "current_ev_ebitda": curr_eveb,
        "current_pe":        curr_pe,
        "current_pb":        curr_pb,
        "years_available":   len(hist_mult),
    }


def _calc_accruals(rows: list[dict]) -> Optional[float]:
    """
    Accruals ratio = (net_income - operating_cf) / total_assets.
    Averaged over up to 5 most recent years. Returned as percentage.
    Negative = earnings backed by real cash (good). Positive = accrual-based earnings (warning).
    """
    ratios = []
    for row in rows[:5]:  # rows already sorted desc; max 5
        ni  = row.get("net_income")
        ocf = row.get("operating_cf")
        ta  = row.get("total_assets")
        if ni is not None and ocf is not None and ta and ta != 0:
            ratios.append((ni - ocf) / ta)
    if not ratios:
        return None
    return round(sum(ratios) / len(ratios) * 100, 2)


def _calc_revenue_cagr(annual_rows: list[dict], years: int = 3) -> Optional[float]:
    """CAGR van omzet over de laatste `years` jaar. Negatief = omzetkrimp."""
    rev_data = [
        (r.get("fiscal_year"), r.get("revenue"))
        for r in annual_rows
        if r.get("fiscal_year") and r.get("revenue") and r["revenue"] > 0
    ]
    rev_data.sort(key=lambda x: x[0])
    if len(rev_data) < 2:
        return None
    if len(rev_data) > years:
        rev_data = rev_data[-(years + 1):]
    oldest, newest = rev_data[0], rev_data[-1]
    n_years = newest[0] - oldest[0]
    if n_years <= 0:
        return None
    return (newest[1] / oldest[1]) ** (1 / n_years) - 1


# ---------------------------------------------------------------------------
# Per-ticker pipeline
# ---------------------------------------------------------------------------

def run_ticker(ticker: str, config: dict) -> dict:
    """
    Run the full calculation pipeline for one ticker.
    Returns a result dict (also persisted to calculated_scores table).
    """
    warnings: list[str] = []

    annual_rows = db.get_financials(ticker, "annual")
    ttm_rows    = db.get_financials(ticker, "ttm")
    market_data = db.get_market_data(ticker)
    stock_info  = db.get_stock(ticker)
    hist_mult   = db.get_historical_multiples(ticker)

    if not stock_info:
        return {"ticker": ticker, "error": "Not found in database", "signal": "N/A"}

    sector   = stock_info.get("sector") or "Default"
    currency = stock_info.get("currency") or "EUR"
    # Gebruik financial_currency voor de EUR-rate bij de valuatie:
    # ADR-tickers handelen in USD maar rapporteren cijfers in JPY/INR/etc.
    # De EUR-conversie moet op de rapportagevaluta gebaseerd zijn.
    financial_currency = stock_info.get("financial_currency") or currency
    eur_rate = get_eur_rate(financial_currency)

    # Load overrides early so we can synthesize rows for manually-entered years
    overrides = db.get_overrides(ticker)

    # Synthesize annual rows for years that only exist in overrides (no Yahoo Finance data)
    existing_fy_set = {row.get("fiscal_year") for row in annual_rows}
    override_only_years = sorted(
        {ov_yr for (_, ov_yr) in overrides if ov_yr is not None and ov_yr not in existing_fy_set},
        reverse=True,
    )
    for ov_yr in override_only_years:
        annual_rows.append({"fiscal_year": ov_yr})
    if override_only_years:
        annual_rows.sort(key=lambda r: r.get("fiscal_year") or 0, reverse=True)
        warnings.append(
            f"Jaarcijfers voor {', '.join(f'FY{y}' for y in override_only_years)} zijn uitsluitend handmatig ingevoerd — "
            f"geen Yahoo Finance-data beschikbaar."
        )

    # Minimum data check
    min_years = config.get("screening", {}).get("required_years", 3)
    if len(annual_rows) < min_years:
        warnings.append(
            f"Only {len(annual_rows)} year(s) of data — minimum {min_years} required for reliable scoring."
        )

    # Controle op verouderde jaarcijfers (bijv. FY2024 terwijl het 2026 is)
    current_year = datetime.utcnow().year
    if annual_rows:
        latest_fy = annual_rows[0].get("fiscal_year") or 0
        if latest_fy < current_year - 1:
            warnings.append(
                f"Recentste jaarcijfers in DB: FY{latest_fy} — FY{current_year - 1} mogelijk al beschikbaar. Klik Refresh."
            )

    # Data freshness check: waarschuw als laatste fetch > 90 dagen geleden is
    if annual_rows:
        fetched_date_str = annual_rows[0].get("fetched_date")
        if fetched_date_str:
            try:
                fetched_dt = datetime.fromisoformat(fetched_date_str[:10])
                days_old = (datetime.utcnow() - fetched_dt).days
                if days_old > 90:
                    warnings.append(
                        f"Jaarcijfers zijn {days_old} dagen oud — overweeg een Refresh om recentere data op te halen."
                    )
            except (ValueError, TypeError):
                pass

    # Revenue-trend check: waarschuw bij structureel dalende omzet (value trap indicator)
    rev_cagr = _calc_revenue_cagr(annual_rows)
    if rev_cagr is not None and rev_cagr < -0.02:
        warnings.append(
            f"Omzetkrimp: 3-jaars CAGR = {rev_cagr * 100:.1f}% — mogelijke value trap. Controleer concurrentiepositie."
        )

    if not annual_rows:
        db.upsert_scores(
            ticker,
            signal="INSUFFICIENT DATA",
            warnings=warnings,
            last_calculated=datetime.utcnow().isoformat(),
        )
        return {"ticker": ticker, "signal": "INSUFFICIENT DATA", "warnings": warnings}

    # Waarschuw als er overrides zijn voor een jaar waarvoor Yahoo Finance nu ook data heeft
    annual_years = {row.get("fiscal_year") for row in annual_rows if row.get("fiscal_year") not in override_only_years}
    override_years_with_data = {ov_yr for (_, ov_yr) in overrides if ov_yr in annual_years}
    for ov_yr in sorted(override_years_with_data):
        warnings.append(
            f"FY{ov_yr} heeft handmatige overrides én Yahoo Finance-data — "
            f"controleer of de overrides nog kloppen en verwijder ze indien Yahoo Finance nu de juiste cijfers heeft."
        )

    for row in annual_rows:
        yr = row.get("fiscal_year")
        for (field, ov_yr), entry in overrides.items():
            if ov_yr == yr or ov_yr is None:
                row[field] = entry["value"]

    # Prepend TTM row (fiscal_year=0) as "year 0" if available and more recent than latest annual
    ttm_row = ttm_rows[0] if ttm_rows else None
    if ttm_row and annual_rows and (annual_rows[0].get("fiscal_year") or 0) < datetime.utcnow().year:
        calc_rows = [ttm_row] + annual_rows
    else:
        calc_rows = annual_rows

    # Normalize
    iqr_mult  = config.get("valuation", {}).get("outlier_iqr_multiplier", 3.0)
    normalized = normalize_all(calc_rows, iqr_mult)

    # Extra quality indicators
    hist_relative  = _calc_hist_relative(hist_mult, market_data, iqr_mult)
    accruals_ratio = _calc_accruals(calc_rows)

    # Quality score
    q_result  = calc_quality(calc_rows, normalized)
    q_total   = q_result["total"]
    warnings += q_result.get("warnings", [])

    min_quality = config.get("screening", {}).get("min_quality_score", 7)
    below_threshold = q_total < min_quality

    # Valuation
    fv_result = combined_fair_value(
        normalized, hist_mult, calc_rows, sector, config, eur_rate
    )

    # Signals
    price_eur    = (market_data or {}).get("price_eur")
    combined_fv  = fv_result.get("combined_fv_eur")
    signal_data  = {}

    if price_eur and combined_fv and combined_fv > 0:
        signal_data = determine_signal(price_eur, combined_fv, q_total, config)
    else:
        signal_data = {
            "signal":           "N/A",
            "margin_of_safety": None,
            "price_vs_fv_pct":  None,
        }
        if not price_eur:
            warnings.append("Current price unavailable — signal cannot be calculated.")
        if not combined_fv:
            warnings.append("Fair value could not be calculated — check financial data.")

    # Market cap filter
    min_cap = config.get("screening", {}).get("min_market_cap_eur", 200)
    mc_eur = (market_data or {}).get("market_cap_eur")
    if min_cap > 0 and mc_eur is not None and mc_eur / 1e6 < min_cap:
        warnings.append(f"Market cap (€{mc_eur/1e6:.0f}M) is below the €{min_cap}M threshold.")

    result = {
        "ticker":           ticker,
        "name":             stock_info.get("name"),
        "sector":           sector,
        "market":           stock_info.get("market"),
        "currency":         currency,
        # Market data
        "price":            (market_data or {}).get("price"),
        "price_eur":        price_eur,
        "market_cap_eur":   mc_eur,
        "market_cap_m_eur": (mc_eur / 1e6) if mc_eur else None,
        # Fair values
        "combined_fv_eur":      combined_fv,
        "conservative_fv_eur":  fv_result.get("conservative_fv_eur"),
        "base_fv_eur":          fv_result.get("base_fv_eur"),
        "optimistic_fv_eur":    fv_result.get("optimistic_fv_eur"),
        "multiples_fv_eur":     fv_result.get("multiples_fv_eur"),
        "graham_fv_eur":        fv_result.get("graham_fv_eur"),
        "perpetuity_fv_eur":    fv_result.get("perpetuity_fv_eur"),
        # Quality
        "quality_score":    q_total,
        "quality_breakdown": q_result.get("breakdown"),
        "piotroski_score":  q_result.get("piotroski", {}).get("score"),
        "piotroski_breakdown": q_result.get("piotroski", {}).get("criteria"),
        # Normalized metrics
        "normalized_eps":        normalized.get("normalized_eps"),
        "normalized_ebitda":     normalized.get("normalized_ebitda"),
        "normalized_fcf":        normalized.get("normalized_fcf"),
        "normalized_owner_earn": normalized.get("normalized_owner_earnings"),
        # Signal
        **signal_data,
        # Extra indicators
        "hist_relative":   hist_relative,
        "accruals_ratio":  accruals_ratio,
        # Meta
        "below_quality_threshold": below_threshold,
        "warnings":         warnings,
        "last_calculated":  datetime.utcnow().isoformat(),
    }

    # Persist
    db.upsert_scores(
        ticker,
        quality_score=q_total,
        quality_breakdown=q_result.get("breakdown"),
        piotroski_score=result["piotroski_score"],
        piotroski_breakdown=result["piotroski_breakdown"],
        normalized_eps=result["normalized_eps"],
        normalized_ebitda=result["normalized_ebitda"],
        normalized_fcf=result["normalized_fcf"],
        normalized_owner_earn=result["normalized_owner_earn"],
        multiples_fv_eur=result["multiples_fv_eur"],
        graham_fv_eur=result["graham_fv_eur"],
        perpetuity_fv_eur=result["perpetuity_fv_eur"],
        combined_fv_eur=combined_fv,
        conservative_fv_eur=result["conservative_fv_eur"],
        base_fv_eur=result["base_fv_eur"],
        optimistic_fv_eur=result["optimistic_fv_eur"],
        signal=result["signal"],
        margin_of_safety=result.get("margin_of_safety"),
        warnings=warnings,
        last_calculated=result["last_calculated"],
        hist_relative=hist_relative,
        accruals_ratio=accruals_ratio,
    )

    return result


# ---------------------------------------------------------------------------
# Bulk screener
# ---------------------------------------------------------------------------

def run_all(config: dict, progress_cb=None) -> list[dict]:
    """
    Run pipeline for all active tickers. Returns list of result dicts.
    Stocks below quality threshold are included but flagged.
    """
    tickers = [s["ticker"] for s in db.get_all_stocks()]
    results = []
    total = len(tickers)

    for idx, ticker in enumerate(tickers):
        if progress_cb:
            progress_cb(ticker, idx, total)
        try:
            r = run_ticker(ticker, config)
            results.append(r)
        except Exception as e:
            log.exception("Pipeline error for %s", ticker)
            results.append({
                "ticker": ticker,
                "signal": "ERROR",
                "warnings": [str(e)],
                "error": str(e),
            })

    # Sort by margin of safety descending (biggest discount first)
    results.sort(
        key=lambda x: x.get("margin_of_safety") or -999,
        reverse=True,
    )
    return results
