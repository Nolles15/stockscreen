"""
Valuation engine — three methods, then combines them.

Method A (60% weight): Normalized Multiples
  4 multiples × fair value:
    P/E FV     = Normalized EPS        × max(hist median P/E,  sector P/E)
    EV/EBITDA  = Normalized EBITDA     × max(hist median EV/EBITDA, sector) → per share
    P/B FV     = Book value per share  × max(hist median P/B,  sector P/B)
    EV/FCF FV  = Normalized FCF/share  × max(hist median EV/FCF, sector)
  Multiples FV = average of the 4 (excluding None)

Method B (40% weight): Split evenly between:
  Graham IV   = Normalized EPS × (8.5 + 2 × g)
  Perpetuity  = Normalized Owner Earnings per share / (r – g)

Combined FV = 0.60 × Multiples FV + 0.40 × (Graham + Perpetuity) / 2

Conservative / Base / Optimistic:
  Use low/base/high growth + high/base/low required return from sector config.
"""

import logging
from typing import Optional

log = logging.getLogger(__name__)


def _sector_cfg(sector: str, config: dict) -> dict:
    """Look up sector config, fall back to Default."""
    sectors = config.get("sectors", {})
    return sectors.get(sector) or sectors.get("Default") or {
        "growth_base": 4, "growth_min": 2, "growth_max": 6,
        "required_return": 10,
        "pe": 18.0, "ev_ebitda": 11.0, "pb": 2.5, "ev_fcf": 16.0,
    }


def _cap_growth(g: float, config: dict, perpetuity: bool = False) -> float:
    hard_cap = config.get("valuation", {}).get(
        "max_perpetuity_growth" if perpetuity else "max_growth_rate", 8 if not perpetuity else 5
    )
    return min(g, hard_cap)


# ---------------------------------------------------------------------------
# Method A — Normalized Multiples
# ---------------------------------------------------------------------------

def multiples_fair_value(
    normalized: dict,
    historical_multiples: list[dict],
    annual_rows: list[dict],
    sector: str,
    config: dict,
    iqr_multiplier: float = 3.0,
) -> dict:
    """
    Returns {pe_fv, ev_ebitda_fv, pb_fv, ev_fcf_fv, avg_fv, detail}
    All values are per-share prices in the stock's native currency.
    """
    from .normalizer import historical_median_multiple

    sc = _sector_cfg(sector, config)

    eps       = normalized.get("normalized_eps")
    ebitda    = normalized.get("normalized_ebitda")
    fcf_ps    = normalized.get("normalized_oe_per_share")   # per share
    norm_fcf  = normalized.get("normalized_fcf")

    # Book value per share from most recent year.
    # NOTE: book_value_ps in the DB may store total equity (data bug), so we always
    # recompute from total_equity / shares_outstanding to guarantee per-share units.
    bvps = None
    shares = None
    for r in annual_rows:
        sh = r.get("shares_outstanding")
        te = r.get("total_equity")
        if sh and sh > 0:
            shares = sh
            if te is not None:
                bvps = te / sh
            break

    # Net debt (latest year) for EV → equity bridge
    net_debt_ps = None
    for r in annual_rows:
        debt = r.get("total_debt", 0) or 0
        nc   = r.get("net_cash", 0) or 0
        sh   = r.get("shares_outstanding")
        if sh and sh > 0:
            net_debt_ps = -nc / sh   # net_debt = -net_cash; positive = net debt, negative = net cash
            break

    # Historical median multiples
    hist_pe     = historical_median_multiple(historical_multiples, "pe_ratio",  iqr_multiplier)
    hist_eveb   = historical_median_multiple(historical_multiples, "ev_ebitda", iqr_multiplier)
    hist_pb     = historical_median_multiple(historical_multiples, "pb_ratio",  iqr_multiplier)
    hist_evfcf  = historical_median_multiple(historical_multiples, "ev_fcf",    iqr_multiplier)

    # Gewogen gemiddelde: 65% historisch median + 35% sectorgemiddelde.
    # Als er geen historische data is, valt het terug op het sectorgemiddelde.
    # Dit voorkomt dat bedrijven die historisch op hoge multiples handelden
    # (bijv. groeibedrijven) structureel te hoog gewaardeerd worden.
    def _blend(hist, sector_default, fallback):
        if hist is not None:
            return 0.65 * hist + 0.35 * sector_default
        return sector_default if sector_default is not None else fallback

    use_pe    = _blend(hist_pe,    sc.get("pe"),       18.0)
    use_eveb  = _blend(hist_eveb,  sc.get("ev_ebitda"), 11.0)
    use_pb    = _blend(hist_pb,    sc.get("pb"),         2.5)
    use_evfcf = _blend(hist_evfcf, sc.get("ev_fcf"),   16.0)

    # Fair values (per share)
    pe_fv: Optional[float] = (eps * use_pe) if eps else None

    ev_ebitda_fv: Optional[float] = None
    if ebitda is not None and net_debt_ps is not None and shares and shares > 0:
        # EV = EBITDA × multiple; Equity value = EV - Net Debt; divide by shares → per share
        total_ebitda_ev = ebitda * use_eveb
        equity_val = total_ebitda_ev - (net_debt_ps * shares)
        ev_ebitda_fv = equity_val / shares

    pb_fv: Optional[float] = (bvps * use_pb) if bvps else None

    ev_fcf_fv: Optional[float] = None
    if fcf_ps is not None and norm_fcf is not None:
        # EV/FCF × total FCF = EV; Equity = EV - Net Debt
        for r in annual_rows:
            sh = r.get("shares_outstanding")
            if sh and sh > 0:
                total_fcf_ev = norm_fcf * use_evfcf
                ev_fcf_fv = (total_fcf_ev - (net_debt_ps * sh if net_debt_ps else 0)) / sh
                break

    candidates = [v for v in [pe_fv, ev_ebitda_fv, pb_fv, ev_fcf_fv] if v is not None and v > 0]
    avg_fv = sum(candidates) / len(candidates) if candidates else None

    return {
        "pe_fv":       pe_fv,
        "ev_ebitda_fv": ev_ebitda_fv,
        "pb_fv":       pb_fv,
        "ev_fcf_fv":   ev_fcf_fv,
        "avg_fv":      avg_fv,
        "multiples_used": {"pe": use_pe, "ev_ebitda": use_eveb, "pb": use_pb, "ev_fcf": use_evfcf},
    }


# ---------------------------------------------------------------------------
# Method B1 — Graham Intrinsic Value
# ---------------------------------------------------------------------------

def graham_fair_value(normalized: dict, sector: str, config: dict, scenario: str = "base") -> Optional[float]:
    """
    Graham IV = Normalized EPS × (8.5 + 2 × g)
    g comes from sector config (base/min/max based on scenario).
    """
    eps = normalized.get("normalized_eps")
    if not eps or eps <= 0:
        return None

    sc = _sector_cfg(sector, config)
    g_key = {"conservative": "growth_min", "base": "growth_base", "optimistic": "growth_max"}.get(scenario, "growth_base")
    g_pct = sc.get(g_key, sc.get("growth_base", 4))
    g = _cap_growth(g_pct, config, perpetuity=False) / 100.0

    return eps * (8.5 + 2 * g * 100)   # g back to % for the formula (Greenblatt uses g as %)


def graham_fair_value_all_scenarios(normalized: dict, sector: str, config: dict) -> dict:
    return {
        "conservative": graham_fair_value(normalized, sector, config, "conservative"),
        "base":         graham_fair_value(normalized, sector, config, "base"),
        "optimistic":   graham_fair_value(normalized, sector, config, "optimistic"),
    }


# ---------------------------------------------------------------------------
# Method B2 — Owner Earnings Perpetuity
# ---------------------------------------------------------------------------

def perpetuity_fair_value(normalized: dict, sector: str, config: dict, scenario: str = "base") -> Optional[float]:
    """
    Perpetuity FV = Normalized Owner Earnings per share / (r – g)
    r = required return (from sector or default), g = conservative growth (max 5%).
    """
    oe_ps = normalized.get("normalized_oe_per_share")
    if not oe_ps or oe_ps <= 0:
        return None

    sc  = _sector_cfg(sector, config)
    val_cfg = config.get("valuation", {})

    g_key = {"conservative": "growth_min", "base": "growth_base", "optimistic": "growth_max"}.get(scenario, "growth_base")
    r_map = {"conservative": 0.01, "base": 0.0, "optimistic": -0.01}   # adjust r ± 1%

    g_raw  = sc.get(g_key, sc.get("growth_base", 4))
    g      = _cap_growth(g_raw, config, perpetuity=True) / 100.0

    r_base = sc.get("required_return", val_cfg.get("default_required_return", 10)) / 100.0
    r      = r_base + r_map.get(scenario, 0.0)

    if r <= g:
        log.warning("r (%.2f) ≤ g (%.2f) for perpetuity — skipping", r, g)
        return None

    return oe_ps / (r - g)


def perpetuity_fair_value_all_scenarios(normalized: dict, sector: str, config: dict) -> dict:
    return {
        "conservative": perpetuity_fair_value(normalized, sector, config, "conservative"),
        "base":         perpetuity_fair_value(normalized, sector, config, "base"),
        "optimistic":   perpetuity_fair_value(normalized, sector, config, "optimistic"),
    }


# ---------------------------------------------------------------------------
# Combined Fair Value
# ---------------------------------------------------------------------------

def combined_fair_value(
    normalized: dict,
    historical_multiples: list[dict],
    annual_rows: list[dict],
    sector: str,
    config: dict,
    eur_rate: float = 1.0,
) -> dict:
    """
    Compute all fair value scenarios and combine them.

    Returns {
      "multiples_fv_native", "multiples_fv_eur",
      "graham_fv_native", "perpetuity_fv_native",
      "combined_fv_native", "combined_fv_eur",
      "conservative_fv_eur", "base_fv_eur", "optimistic_fv_eur",
      "detail": {...}
    }
    """
    val_cfg = config.get("valuation", {})
    w_mult  = val_cfg.get("multiples_weight", 0.60)
    w_dcf   = val_cfg.get("dcf_weight",       0.40)

    mult_result = multiples_fair_value(
        normalized, historical_multiples, annual_rows, sector, config
    )

    graham_scenarios = graham_fair_value_all_scenarios(normalized, sector, config)
    perp_scenarios   = perpetuity_fair_value_all_scenarios(normalized, sector, config)

    def _combine(mult_fv, g_fv, p_fv):
        if mult_fv is None:
            # Fall back to DCF only
            dcf_inputs = [v for v in [g_fv, p_fv] if v]
            return (sum(dcf_inputs) / len(dcf_inputs)) if dcf_inputs else None
        dcf_inputs = [v for v in [g_fv, p_fv] if v]
        if not dcf_inputs:
            return mult_fv   # Only multiples available
        dcf_avg = sum(dcf_inputs) / len(dcf_inputs)
        return w_mult * mult_fv + w_dcf * dcf_avg

    # Base scenario
    combined_base = _combine(
        mult_result["avg_fv"],
        graham_scenarios["base"],
        perp_scenarios["base"],
    )

    # Conservative (lowest g, highest r)
    mult_cons = mult_result["avg_fv"]   # multiples don't have scenarios; use base
    combined_cons = _combine(mult_cons, graham_scenarios["conservative"], perp_scenarios["conservative"])

    # Optimistic
    combined_opt = _combine(mult_result["avg_fv"], graham_scenarios["optimistic"], perp_scenarios["optimistic"])

    def to_eur(v):
        return (v * eur_rate) if v is not None else None

    return {
        # Native currency (stock's reporting currency)
        "multiples_fv_native":   mult_result["avg_fv"],
        "graham_fv_native":      graham_scenarios["base"],
        "perpetuity_fv_native":  perp_scenarios["base"],
        "combined_fv_native":    combined_base,
        "conservative_fv_native": combined_cons,
        "optimistic_fv_native":  combined_opt,
        # EUR equivalents
        "multiples_fv_eur":      to_eur(mult_result["avg_fv"]),
        "graham_fv_eur":         to_eur(graham_scenarios["base"]),
        "perpetuity_fv_eur":     to_eur(perp_scenarios["base"]),
        "combined_fv_eur":       to_eur(combined_base),
        "base_fv_eur":           to_eur(combined_base),
        "conservative_fv_eur":   to_eur(combined_cons),
        "optimistic_fv_eur":     to_eur(combined_opt),
        # Detail for debugging
        "detail": {
            "multiples":  mult_result,
            "graham":     graham_scenarios,
            "perpetuity": perp_scenarios,
        },
    }
