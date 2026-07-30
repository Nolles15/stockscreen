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
import statistics
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Robustness helpers (Fase 4)
# ---------------------------------------------------------------------------

# Graham-multiplier werd in 1962 geijkt op een AAA bedrijfsobligatierendement van 4.4%.
# In een hoger-rente-omgeving is de klassieke formule te hoog; we schalen mee met de
# required_return zodat Graham een conservatieve benchmark blijft (nooit inflerend).
GRAHAM_REFERENCE_YIELD = 4.4

# Minimum r - g spread voor een stabiele Gordon Growth — onder de 2% explodeert de
# denominator en krijg je onzinnige FV's.
PERPETUITY_MIN_SPREAD = 0.02

# Single-method outlier-filters binnen multiples_fair_value.
MULTIPLE_OUTLIER_LOW  = 0.20
MULTIPLE_OUTLIER_HIGH = 5.0

# Cross-method outlier-filter in combined_fair_value (multiples / Graham / perpetuity).
METHOD_OUTLIER_LOW  = 0.33
METHOD_OUTLIER_HIGH = 3.0


def _filter_outliers(values: list[float], lo: float, hi: float) -> list[float]:
    """
    Filter waarden die buiten [lo × mediaan, hi × mediaan] vallen.
    Alleen actief bij ≥3 waarden (daaronder is mediaan niet betekenisvol).
    """
    clean = [v for v in values if v is not None and v > 0]
    if len(clean) < 3:
        return clean
    med = statistics.median(clean)
    if med <= 0:
        return clean
    return [v for v in clean if lo * med <= v <= hi * med]


def _spread_pct(values: list[float]) -> Optional[float]:
    """(max − min) / mediaan × 100 — hoe hoger, hoe meer disagreement tussen methodes."""
    clean = [v for v in values if v is not None and v > 0]
    if len(clean) < 2:
        return None
    med = statistics.median(clean)
    if med <= 0:
        return None
    return (max(clean) - min(clean)) / med * 100


def _confidence_label(spread: Optional[float], n_methods: int) -> str:
    """
    Confidence in de FV o.b.v. disagreement tussen methodes.
    - 1 methode beschikbaar: low (geen cross-validatie mogelijk)
    - spread < 30%: high
    - spread 30-60%: medium
    - spread > 60%: low
    """
    if n_methods < 2 or spread is None:
        return "low"
    if spread < 30:
        return "high"
    if spread < 60:
        return "medium"
    return "low"


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
        s = sector_default if sector_default is not None else fallback
        if hist is not None:
            return 0.65 * hist + 0.35 * s
        return s

    use_pe    = _blend(hist_pe,    sc.get("pe"),       18.0)
    use_eveb  = _blend(hist_eveb,  sc.get("ev_ebitda"), 11.0)
    use_pb    = _blend(hist_pb,    sc.get("pb"),         2.5)
    use_evfcf = _blend(hist_evfcf, sc.get("ev_fcf"),   16.0)

    # Fair values (per share) — expliciete pre-filter op inputs: een methode
    # moet positieve input (eps/bvps/ebitda/fcf) hebben anders slaat 'm nergens
    # op. Transparantie: elke drop komt met reden in drop_reasons.
    drop_reasons: list[str] = []

    if eps is not None and eps > 0:
        pe_fv: Optional[float] = eps * use_pe
    else:
        pe_fv = None
        if eps is None:
            drop_reasons.append("pe: normalized_eps ontbreekt")
        else:
            drop_reasons.append(f"pe: eps≤0 ({eps:.2f})")

    ev_ebitda_fv: Optional[float] = None
    if ebitda is None or ebitda <= 0:
        drop_reasons.append(
            "ev_ebitda: ebitda ontbreekt" if ebitda is None else f"ev_ebitda: ebitda≤0 ({ebitda:.0f})"
        )
    elif net_debt_ps is None or not (shares and shares > 0):
        drop_reasons.append("ev_ebitda: net_debt of shares ontbreekt")
    else:
        total_ebitda_ev = ebitda * use_eveb
        equity_val = total_ebitda_ev - (net_debt_ps * shares)
        ev_ebitda_fv = equity_val / shares

    if bvps is not None and bvps > 0:
        pb_fv: Optional[float] = bvps * use_pb
    else:
        pb_fv = None
        if bvps is None:
            drop_reasons.append("pb: book_value_per_share ontbreekt")
        else:
            drop_reasons.append(f"pb: bvps≤0 ({bvps:.2f} — negatief eigen vermogen)")

    ev_fcf_fv: Optional[float] = None
    if norm_fcf is None or norm_fcf <= 0:
        drop_reasons.append(
            "ev_fcf: normalized_fcf ontbreekt" if norm_fcf is None else f"ev_fcf: norm_fcf≤0 ({norm_fcf:.0f})"
        )
    else:
        for r in annual_rows:
            sh = r.get("shares_outstanding")
            if sh and sh > 0:
                total_fcf_ev = norm_fcf * use_evfcf
                ev_fcf_fv = (total_fcf_ev - (net_debt_ps * sh if net_debt_ps else 0)) / sh
                break
        if ev_fcf_fv is None:
            drop_reasons.append("ev_fcf: geen rij met shares_outstanding")

    # Combineer alleen de valide methodes (>0) en dan pas outlier-filter.
    # Bij ≤2 methodes slaat outlier-filter zichzelf over → geen cascade-risico.
    raw = [v for v in [pe_fv, ev_ebitda_fv, pb_fv, ev_fcf_fv] if v is not None and v > 0]
    kept = _filter_outliers(raw, MULTIPLE_OUTLIER_LOW, MULTIPLE_OUTLIER_HIGH)
    if len(raw) > len(kept):
        drop_reasons.append(f"outlier-filter binnen multiples: {len(raw) - len(kept)} methode(s) verworpen")

    if len(kept) >= 3:
        avg_fv = statistics.median(kept)
    elif kept:
        avg_fv = sum(kept) / len(kept)
    else:
        avg_fv = None

    return {
        "pe_fv":       pe_fv,
        "ev_ebitda_fv": ev_ebitda_fv,
        "pb_fv":       pb_fv,
        "ev_fcf_fv":   ev_fcf_fv,
        "avg_fv":      avg_fv,
        "n_methods":   len(kept),
        "n_dropped":   len(raw) - len(kept),
        "drop_reasons": drop_reasons,
        "multiples_used": {"pe": use_pe, "ev_ebitda": use_eveb, "pb": use_pb, "ev_fcf": use_evfcf},
    }


# ---------------------------------------------------------------------------
# Method B1 — Graham Intrinsic Value
# ---------------------------------------------------------------------------

def graham_fair_value(normalized: dict, sector: str, config: dict, scenario: str = "base") -> Optional[float]:
    """
    Gemoderniseerde Graham IV = EPS × (8.5 + 2g) × (4.4 / Y)
    waarbij Y = sector required_return (min. 4.4%).

    De klassieke formule is geijkt op AAA-bondyields van 4.4% (1962). In een hogere-rente-
    omgeving overwaardeert de pure formule; de Y-correctie dempt dat. We scalen alleen naar
    beneden (cap bij 1.0) zodat Graham nooit wordt opgeblazen in ultra-laag-rente-sectoren.
    """
    eps = normalized.get("normalized_eps")
    if not eps or eps <= 0:
        return None

    sc = _sector_cfg(sector, config)
    val_cfg = config.get("valuation", {})

    g_key = {"conservative": "growth_min", "base": "growth_base", "optimistic": "growth_max"}.get(scenario, "growth_base")
    g_pct = sc.get(g_key, sc.get("growth_base", 4))
    g = _cap_growth(g_pct, config, perpetuity=False)   # percentage

    required_return = sc.get("required_return", val_cfg.get("default_required_return", 10))
    yield_scaler = min(1.0, GRAHAM_REFERENCE_YIELD / max(required_return, GRAHAM_REFERENCE_YIELD))

    return eps * (8.5 + 2 * g) * yield_scaler


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

    # Stability-guard: te kleine r-g → denominator explodeert en FV wordt onzinnig.
    # Minimum spread van 2% zorgt dat de FV eindige, realistische waarden oplevert.
    if (r - g) < PERPETUITY_MIN_SPREAD:
        log.warning("perpetuity skip: r (%.2f%%) − g (%.2f%%) < %.0f%%",
                    r * 100, g * 100, PERPETUITY_MIN_SPREAD * 100)
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
) -> dict:
    """
    Compute all fair value scenarios and combine them.
    Alle bedragen zijn per aandeel in de native currency van het aandeel —
    er wordt geen valutaconversie gedaan.

    Returns {
      "multiples_fv", "graham_fv", "perpetuity_fv",
      "combined_fv", "base_fv" (= combined_fv),
      "conservative_fv", "optimistic_fv",
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

    def _sanity_filter_methods(mult_fv, g_fv, p_fv):
        """
        Cross-method outlier-filter: drop elke methode die >3× of <0.33× afwijkt van
        de mediaan van de andere twee. Voorkomt dat één wild afwijkende methode het
        gewogen gemiddelde scheeftrekt (bv. perpetuity bij kleine r-g, Graham bij
        extreem hoge EPS na uitschieter-jaar).
        """
        methods = {"mult": mult_fv, "graham": g_fv, "perp": p_fv}
        vals = [v for v in methods.values() if v is not None and v > 0]
        if len(vals) < 3:
            return mult_fv, g_fv, p_fv, []
        med = statistics.median(vals)
        if med <= 0:
            return mult_fv, g_fv, p_fv, []
        dropped = []
        filtered = {}
        for name, v in methods.items():
            if v is not None and v > 0 and not (METHOD_OUTLIER_LOW * med <= v <= METHOD_OUTLIER_HIGH * med):
                dropped.append(name)
                filtered[name] = None
            else:
                filtered[name] = v
        return filtered["mult"], filtered["graham"], filtered["perp"], dropped

    def _combine(mult_fv, g_fv, p_fv):
        mult_fv, g_fv, p_fv, dropped = _sanity_filter_methods(mult_fv, g_fv, p_fv)
        if mult_fv is None:
            dcf_inputs = [v for v in [g_fv, p_fv] if v]
            combined = (sum(dcf_inputs) / len(dcf_inputs)) if dcf_inputs else None
            return combined, dropped
        dcf_inputs = [v for v in [g_fv, p_fv] if v]
        if not dcf_inputs:
            return mult_fv, dropped
        dcf_avg = sum(dcf_inputs) / len(dcf_inputs)
        return w_mult * mult_fv + w_dcf * dcf_avg, dropped

    # Base scenario
    combined_base, dropped_base = _combine(
        mult_result["avg_fv"],
        graham_scenarios["base"],
        perp_scenarios["base"],
    )

    # Conservative (lowest g, highest r); multiples heeft geen scenario's
    combined_cons, _ = _combine(
        mult_result["avg_fv"], graham_scenarios["conservative"], perp_scenarios["conservative"]
    )

    # Optimistic
    combined_opt, _ = _combine(
        mult_result["avg_fv"], graham_scenarios["optimistic"], perp_scenarios["optimistic"]
    )

    # Confidence-score op basis van disagreement tussen methodes (base-scenario)
    base_methods = [
        mult_result["avg_fv"],
        graham_scenarios["base"],
        perp_scenarios["base"],
    ]
    base_clean = [v for v in base_methods if v is not None and v > 0]
    spread_pct = _spread_pct(base_clean)
    confidence = _confidence_label(spread_pct, len(base_clean))

    # Minimum 2 valide top-level methodes vereist. Eén enkele methode levert geen
    # cross-validatie en is daardoor te gevoelig voor een enkele schaal/data-bug.
    # Met <2 methodes geven we expliciet None terug → FV-gate in screener klasseert
    # dit als INSUFFICIENT DATA in plaats van een misleidende waarde.
    insufficient_methods = len(base_clean) < 2
    if insufficient_methods:
        combined_base = None
        combined_cons = None
        combined_opt = None

    # Verzamel alle drop-redenen (zowel per-multiples pre-filter als cross-method
    # outliers) zodat dashboard en /api/fv-diagnostics inzichtelijk tonen waarom
    # een FV onbetrouwbaar is.
    drop_reasons_all: list[str] = []
    drop_reasons_all.extend(mult_result.get("drop_reasons", []))
    for name in dropped_base:
        drop_reasons_all.append(f"cross-method outlier: {name} (>3× of <0.33× van mediaan)")
    if insufficient_methods:
        drop_reasons_all.append(
            f"onvoldoende valide FV-methodes ({len(base_clean)} < 2) — combined_fv niet berekend"
        )

    return {
        "multiples_fv":      mult_result["avg_fv"],
        "graham_fv":         graham_scenarios["base"],
        "perpetuity_fv":     perp_scenarios["base"],
        "combined_fv":       combined_base,
        "base_fv":           combined_base,
        "conservative_fv":   combined_cons,
        "optimistic_fv":     combined_opt,
        "fv_confidence":     confidence,
        "fv_spread_pct":     round(spread_pct, 1) if spread_pct is not None else None,
        "fv_methods_used":   len(base_clean),
        "fv_methods_dropped": drop_reasons_all,
        # Detail for debugging
        "detail": {
            "multiples":  mult_result,
            "graham":     graham_scenarios,
            "perpetuity": perp_scenarios,
        },
    }
