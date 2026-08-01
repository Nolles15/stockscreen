"""
Flask application — routes, background refresh job, REST API.

Start with:  python app.py
Dashboard:   http://localhost:<port>  (see app.port in config.yaml)
"""

import json
import logging
import math
import os
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import yaml
from flask import Flask, jsonify, render_template, request, redirect, url_for

from engine import db
from engine import data_quality
from engine import refresh
from engine import remap_rules
from engine.data_fetcher import (
    fetch_and_store,
    fetch_market_only,
    fetch_all_tickers,
)
from engine.screener import run_ticker, run_all, determine_signal

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "stockscreen-local-dev-only")

CONFIG_PATH = "config.yaml"

# Ticker format-validatie: optionele exchange-suffix na een punt (bv. ASML.AS,
# BRK-B, ADYEN.AS, BABA). Voorkomt dat malformed tickers als "NASDAQ:ICLR" of
# "foo bar" in de DB belanden en 50× faal-fetches forceren in de cron.
_VALID_TICKER_RE = re.compile(r"^[A-Z0-9]{1,6}(-[A-Z0-9]{1,3})?(\.[A-Z]{1,4})?$")


def _validate_ticker(raw: str) -> tuple[str | None, str | None]:
    """Returns (normalized_ticker, error_message). One of the two is None."""
    if not raw:
        return None, "ticker is leeg"
    t = raw.strip().upper()
    if len(t) > 12:
        return None, f"ticker te lang ({len(t)} chars, max 12)"
    if not _VALID_TICKER_RE.match(t):
        return None, (
            f"'{t}' voldoet niet aan ticker-format — verwacht: SYMBOOL "
            f"eventueel met -class en .EXCH suffix (bv. BRK-B, ASML.AS, SAND.ST)."
        )
    return t, None

# Tabellen aanmaken bij elke startup (CREATE TABLE IF NOT EXISTS — veilig idempotent)
db.init_db()


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Background job tracking
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}   # job_id → {status, progress, current, errors}
_jobs_lock = threading.Lock()
_startup_job_id: str | None = None   # job_id of the auto-refresh triggered at startup

# Bewaakt dat er maar één ronde jaarcijfers tegelijk draait. Twee batches
# tegelijk verdubbelen het aantal Yahoo-aanroepen per seconde en lopen linea
# recta in de rate-limits; de scheduler en de handmatige knop delen deze vlag.
_fundamentals_lock = threading.Lock()
_fundamentals_running = False

STALE_HEAVY_DAYS = 6   # dagen zonder zware refresh → opnieuw ophalen bij next run


def _new_job() -> str:
    jid = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[jid] = {"status": "pending", "progress": 0, "current": "", "errors": {}}
    return jid


def _update_job(jid: str, **kwargs):
    with _jobs_lock:
        if jid in _jobs:
            _jobs[jid].update(kwargs)


def _get_job(jid: str) -> dict:
    with _jobs_lock:
        return dict(_jobs.get(jid, {}))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    cfg = load_config()
    return render_template("index.html", config=cfg)


@app.route("/stock/<ticker>")
def stock_detail(ticker):
    cfg = load_config()
    stock   = db.get_stock(ticker)
    if not stock:
        return "Stock not found", 404
    annual    = db.get_financials(ticker, "annual")
    ttm_list  = db.get_financials(ticker, "ttm")
    market    = db.get_market_data(ticker)
    scores    = db.get_scores(ticker)
    overrides = db.get_overrides(ticker)
    hist_mult = db.get_historical_multiples(ticker)

    # Voeg synthetische rijen toe voor jaren die alleen in overrides voorkomen (geen Yahoo Finance-data)
    existing_fy_set = {row.get("fiscal_year") for row in annual}
    override_only_years = sorted(
        {ov_yr for (_, ov_yr) in overrides if ov_yr is not None and ov_yr not in existing_fy_set},
        reverse=True,
    )
    for ov_yr in override_only_years:
        annual.append({"fiscal_year": ov_yr})
    if override_only_years:
        annual.sort(key=lambda r: r.get("fiscal_year") or 0, reverse=True)

    # Pas overrides toe op alle rijen (inclusief synthetische)
    for row in annual:
        yr = row.get("fiscal_year")
        for (field, ov_yr), entry in overrides.items():
            if ov_yr == yr or ov_yr is None:
                row[field] = entry["value"]
        # Bereken ROE als het ontbreekt maar net_income + total_equity bekend zijn
        if not row.get("roe") and row.get("net_income") and row.get("total_equity") and row["total_equity"] > 0:
            row["roe"] = row["net_income"] / row["total_equity"]

    # Convert overrides key (field, year) → list for template
    override_list = [
        {"field": f, "year": y, "value": entry["value"], "note": entry["note"]}
        for (f, y), entry in overrides.items()
    ]

    # Set van "field:jaar" strings voor snelle lookup in template (gele cellen)
    override_set = {f"{f}:{y}" for (f, y) in overrides}

    return render_template(
        "stock.html",
        ticker=ticker,
        stock=stock,
        annual=annual,
        ttm=ttm_list[0] if ttm_list else None,
        market=market,
        scores=scores,
        overrides=override_list,
        override_set=override_set,
        hist_mult=hist_mult,
        config=cfg,
    )


@app.route("/settings")
def settings():
    cfg = load_config()
    return render_template("settings.html", config=cfg)


# ---------------------------------------------------------------------------
# API — Dashboard data
# ---------------------------------------------------------------------------

@app.route("/api/dashboard")
def api_dashboard():
    """Return alle aandelen met scores en marktdata. Filtering gebeurt client-side."""
    cfg = load_config()
    min_quality = cfg.get("screening", {}).get("min_quality_score", 7)
    new_days    = cfg.get("app", {}).get("new_ticker_days", 7)
    today       = datetime.now(timezone.utc).date()

    rows = []
    for r in db.get_dashboard_data():
        t        = r["ticker"]
        mc_m     = (r.get("market_cap") / 1e6) if r.get("market_cap") else None
        q_score  = r.get("quality_score")
        price    = r.get("price")
        fv       = r.get("combined_fv")

        # Signal + mos live herberekenen zodat verse market_data direct
        # goed matcht tegen de laatste FV/quality-snapshot (staleness-fix).
        # FV-plausibiliteitsgate: factor-10+ afwijking tussen FV en price =
        # schaal/eenheid/data-bug → INSUFFICIENT DATA ipv misleidend signal.
        fv_price_ratio = _fv_price_ratio(price, fv)
        fv_ratio_oob = _fv_ratio_oob(fv_price_ratio)
        signal = _effective_signal(price, fv, q_score, r, cfg)

        norm_fcf_raw = r.get("normalized_fcf")
        fcf_m        = (norm_fcf_raw / 1e6) if norm_fcf_raw is not None else None

        added_str = r.get("added_date")
        try:
            days_since_added = (today - datetime.fromisoformat(added_str).date()).days if added_str else None
        except (ValueError, TypeError):
            days_since_added = None

        # Markering voor client-side filtering — server filtert niet meer
        low_quality = q_score is not None and q_score < min_quality

        # Bij INSUFFICIENT DATA is de combined_fv per definitie onbetrouwbaar
        # (schaal-bug, sanity-gate-hit, <2 methodes). De korting/mos die we
        # eruit zouden rekenen is misleidend — en sorteert de rijen naar de
        # top omdat de factor-21×-FV als '94% korting' verschijnt. Daarom:
        # geen mos/price_vs_fv tonen voor INSUFFICIENT DATA.
        is_insufficient = signal == "INSUFFICIENT DATA"
        mos_val = None if is_insufficient else _margin_of_safety(price, fv)
        pvf_val = None if is_insufficient else _price_vs_fv(price, fv)

        # Reden WAAROM er geen signaal is: 3 heldere buckets i.p.v. één rood label.
        reason = data_quality.classify_signal_reason(
            signal, r.get("data_status"), r.get("data_issues") or [],
            fv_ratio_oob, r.get("fv_methods_used"),
        )
        # Verlieslatend-maar-groeiend markeren zodat kansen opvallen i.p.v. wegvallen.
        # data_status ok/warning sluit insolvente (negatief EV) gevallen uit.
        rev_cagr = r.get("revenue_cagr")
        growth_thr = cfg.get("screening", {}).get("growth_lossmaker_cagr", 0.15)
        is_growth_lossmaker = (
            reason["reason_code"] == "geen_fv"
            and r.get("data_status") not in ("bad", "missing")
            and rev_cagr is not None and rev_cagr >= growth_thr
        )

        rows.append({
            "ticker":               t,
            "name":                 r.get("name") or t,
            "sector":               r.get("sector"),
            "market":               r.get("market"),
            "currency":             r.get("currency"),
            "price":                r.get("price"),
            "market_cap_m":         mc_m,
            "combined_fv":          r.get("combined_fv"),
            "conservative_fv":      r.get("conservative_fv"),
            "base_fv":              r.get("base_fv"),
            "optimistic_fv":        r.get("optimistic_fv"),
            "fv_confidence":        r.get("fv_confidence"),
            "fv_spread_pct":        r.get("fv_spread_pct"),
            "fv_methods_used":      r.get("fv_methods_used"),
            "normalized_fcf_m":     fcf_m,
            "margin_of_safety":     mos_val,
            "price_vs_fv_pct":      pvf_val,
            "quality_score":        q_score,
            "piotroski_score":      r.get("piotroski_score"),
            "signal":               signal or "N/A",
            "last_updated":         r.get("last_updated"),
            "last_calculated":      r.get("last_calculated"),
            "warnings":             r.get("warnings") or [],
            "latest_fiscal_year":   r.get("latest_fy"),
            "hist_relative":        r.get("hist_relative") or {},
            "is_new":               days_since_added is not None and days_since_added <= new_days,
            "days_since_added":     days_since_added,
            # Markering voor client-side filtering
            "low_quality":          low_quality,
            # Data-kwaliteit (Fase 2)
            "data_status":          r.get("data_status"),
            "data_completeness":    r.get("completeness_pct"),
            "data_issues":          r.get("data_issues") or [],
            "data_fetch_success":   r.get("fetch_success"),
            "data_consecutive_failures": r.get("consecutive_failures") or 0,
            # FV-diagnose (Fase 1): ratio buiten [0.1, 10] = schaal-bug signaal
            "fv_price_ratio":       round(fv_price_ratio, 3) if fv_price_ratio is not None else None,
            "fv_ratio_oob":         fv_ratio_oob,
            # Reden-weergave (Plan 2): waarom geen signaal + groei-markering
            "reason_code":          reason["reason_code"],
            "reason_label":         reason["reason_label"],
            "reason_color":         reason["reason_color"],
            "fv_methods_dropped":   r.get("fv_methods_dropped") or [],
            "revenue_cagr":         rev_cagr,
            "is_growth_lossmaker":  is_growth_lossmaker,
        })

    _add_rank_scores(rows)
    rows.sort(key=lambda x: x.get("margin_of_safety") or -9999, reverse=True)
    return jsonify(_sanitize(rows))


def _add_rank_scores(rows: list[dict]) -> None:
    """
    Geef elke beoordeelde rij een `rank_score` van 0 tot 100.

    Bestaansreden: de absolute signalen hangen af van hoe goed de fair values
    gekalibreerd zijn, en dat blijft altijd een benadering. Een rangorde niet:
    die zegt alleen "dit aandeel is binnen de huidige selectie aantrekkelijker
    dan dat". Daardoor is er altijd een bruikbare shortlist, ook wanneer er
    volgens de absolute maatstaf nauwelijks koopjes zijn.

    Weging: korting weegt het zwaarst, daarna de kwaliteit van het bedrijf, en
    als laatste hoeveel vertrouwen we in de waardering zelf hebben.
    """
    kandidaten = [
        r for r in rows
        if r.get("margin_of_safety") is not None
        and r.get("data_status") in ("ok", "warning")
        and r.get("combined_fv")
    ]
    if len(kandidaten) < 2:
        return

    # Percentielpositie binnen de huidige selectie: 0 = de duurste, 1 = de goedkoopste.
    op_marge = sorted(kandidaten, key=lambda r: r["margin_of_safety"])
    laatste = len(op_marge) - 1
    positie = {id(r): i / laatste for i, r in enumerate(op_marge)}

    vertrouwen = {"high": 1.0, "medium": 0.6, "low": 0.3}
    for r in kandidaten:
        kwaliteit = (r.get("quality_score") or 0) / 10.0
        conf = vertrouwen.get(r.get("fv_confidence"), 0.3)
        score = 0.5 * positie[id(r)] + 0.3 * kwaliteit + 0.2 * conf
        r["rank_score"] = round(100 * score, 1)


def _fv_price_ratio(price, fv):
    return (fv / price) if (price and fv and fv > 0) else None


def _fv_ratio_oob(ratio):
    """
    Wijkt de fair value een factor tien af van de koers, dan klopt er iets niet
    met de schaal of de eenheid. Dan liever geen oordeel dan een misleidend
    oordeel.
    """
    return ratio is not None and (ratio < 0.1 or ratio > 10.0)


def _effective_signal(price, fv, q_score, row, cfg):
    """
    Het signaal zoals het nú geldt, niet zoals het bij de laatste herberekening
    was opgeslagen.

    Koersen worden dagelijks ververst, de fair value hooguit eens per negen dagen.
    Het opgeslagen signaal is dus al snel achterhaald; live herberekenen zorgt dat
    een verse koers meteen tegen de laatste waardering wordt gehouden.

    Deze functie is met opzet de enige plek waar die afleiding gebeurt. Toen het
    dashboard live herberekende en /api/health de opgeslagen waarde telde, gaven
    de twee schermen verschillende aantallen voor precies hetzelfde ("167 zonder
    oordeel" naast "179"). Dat soort verschil ondermijnt het vertrouwen in elk
    ander getal op de pagina.
    """
    fv_price_ratio = _fv_price_ratio(price, fv)
    fv_ratio_oob = _fv_ratio_oob(fv_price_ratio)
    if price and fv and fv > 0 and q_score is not None and not fv_ratio_oob:
        return determine_signal(price, fv, q_score, cfg).get("signal")
    if fv_ratio_oob:
        return "INSUFFICIENT DATA"
    return (row or {}).get("signal") or "N/A"


def _price_vs_fv(price, fv):
    if price and fv and fv > 0:
        return round(price / fv * 100, 1)
    return None


def _margin_of_safety(price, fv):
    """Mos = (1 - price/fv) * 100. Live berekend zodat verse market_data
    niet tegen oude calc-snapshots kan wrijven (staleness-bug)."""
    if price and fv and fv > 0:
        return round((1 - price / fv) * 100, 1)
    return None


def _sanitize(obj):
    """Vervang Infinity/NaN door None zodat de browser de JSON kan parsen."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Background refresh worker (shared by manual refresh + startup auto-refresh)
# ---------------------------------------------------------------------------

def _run_refresh_job(jid: str, tickers: list, cfg: dict) -> None:
    """Fetch + recalculate all tickers, log to activity DB, update job status."""
    total = len(tickers)
    errors: dict = {}
    _update_job(jid, status="running", total=total)

    def _progress(ticker: str, idx: int, tot: int):
        _update_job(jid, current=f"Fetching {ticker}…", progress=int(idx / tot * 60))

    # fetch_all_tickers doet rate-limited parallel fetch met retry in de lower layer;
    # FX rates worden intern al één keer ververst voordat tickers worden afgewerkt.
    try:
        fetch_results = fetch_all_tickers(tickers, progress_cb=_progress)
    except Exception as e:
        log.exception("Bulk fetch crashte")
        _update_job(jid, status="error", current=f"Bulk fetch crashte: {e}")
        return

    for ticker, warn in fetch_results.items():
        if warn:
            errors[ticker] = warn
            status = "error" if any("crashed" in str(w).lower() for w in warn) else "warning"
            db.log_activity("fetch", ticker, status, {"warnings": warn})
        else:
            db.log_activity("fetch", ticker, "ok", {"source": "Yahoo Finance"})

    # Scoreberekening: ververs de net-gefetchte tickers + bijvangen van ontbrekende scores
    scored_set = {r["ticker"] for r in db.get_all_scores()}
    gap_tickers = [s["ticker"] for s in db.get_all_stocks() if s["ticker"] not in scored_set]
    fetched_set = set(tickers)
    calc_tickers = tickers + [t for t in gap_tickers if t not in fetched_set]
    total_calc = len(calc_tickers)

    for idx, ticker in enumerate(calc_tickers):
        _update_job(jid, current=f"Calculating {ticker}…", progress=60 + int(idx / total_calc * 38))
        try:
            result = run_ticker(ticker, cfg)
            db.log_activity("recalculate", ticker, "ok", {
                "signal": result.get("signal"),
                "fv": result.get("combined_fv"),
                "warnings": result.get("warnings", []),
            })
        except Exception as e:
            log.exception("Calc failed for %s", ticker)
            errors.setdefault(ticker, []).append(f"Calculation: {e}")
            db.log_activity("recalculate", ticker, "error", {"error": str(e)})

    _update_job(jid, status="done", progress=100, current="Klaar", errors=errors)
    log.info("Refresh job %s complete. Fetched: %d, Calculated: %d, Errors: %d",
             jid, len(tickers), total_calc, len(errors))


# ---------------------------------------------------------------------------
# Smart refresh helpers
# ---------------------------------------------------------------------------

def _get_stale_tickers(all_tickers: list[str], max_age_days: int) -> list[str]:
    """Geeft tickers terug die ouder zijn dan max_age_days of nog nooit zijn opgehaald."""
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=max_age_days)).isoformat()
    fetched = db.get_latest_fetched_dates()
    return [t for t in all_tickers if fetched.get(t, "") < cutoff]


def _run_light_job(jid: str, tickers: list[str]) -> None:
    """Lichte refresh: marktdata voor alle tickers bijwerken (native, geen FX)."""
    total = len(tickers)
    _update_job(jid, status="running", total=total, current="Marktdata ophalen...")

    for idx, ticker in enumerate(tickers):
        _update_job(jid, current=f"Marktdata {ticker}...", progress=int((idx + 1) / total * 100))
        try:
            fetch_market_only(ticker)
        except Exception as e:
            log.warning("Light refresh mislukt voor %s: %s", ticker, e)
        time.sleep(0.3)   # voorkomt rate-limiting bij Yahoo Finance

    _update_job(jid, status="done", progress=100, current="Klaar")
    log.info("Light refresh klaar: %d tickers bijgewerkt", total)


def _last_market_update_age_hours() -> float | None:
    """
    Leest de nieuwste 'last_updated' uit market_data. Geeft uren sinds die update,
    of None als er nog geen data is. Bron van waarheid voor de scheduler zodat
    restarts geen dubbele refresh triggeren.
    """
    try:
        with db._cursor() as cur:
            cur.execute("SELECT MAX(last_updated) AS latest FROM market_data")
            row = cur.fetchone()
    except Exception:
        return None
    latest = (row or {}).get("latest") if row else None
    if not latest:
        return None
    try:
        ts = datetime.fromisoformat(latest.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None
    return (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0


# Scheduler v2. Elke beslissing komt uit de tabel `refresh_state`, nooit uit
# geheugen: een herstart van de machine mag geen run overslaan en ook geen
# dubbele run veroorzaken.
SCHEDULER_TICK_SECONDS = 900          # elk kwartier kijken of er werk is
PRICE_REFRESH_HOUR     = 18           # na sluiting van de Europese beurzen
PRICE_REFRESH_MINUTE   = 30
FUNDAMENTALS_HOUR      = 3            # 's nachts, buiten de drukke uren van Yahoo
AMSTERDAM              = ZoneInfo("Europe/Amsterdam")


def _local_now() -> datetime:
    """Nu in Amsterdamse tijd — de machine draait op UTC, de beurzen niet."""
    return datetime.now(AMSTERDAM)


def _ran_today(state_key: str) -> bool:
    """Heeft deze taak vandaag (Amsterdamse kalenderdag) al gedraaid?"""
    raw = db.get_refresh_state(state_key)
    if not raw:
        return False
    try:
        last = datetime.fromisoformat(raw)
    except ValueError:
        return False
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return last.astimezone(AMSTERDAM).date() == _local_now().date()


def _days_since(state_key: str) -> float:
    raw = db.get_refresh_state(state_key)
    if not raw:
        return 9999.0
    try:
        last = datetime.fromisoformat(raw)
    except ValueError:
        return 9999.0
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - last).total_seconds() / 86400.0


def _run_price_refresh(cfg: dict) -> None:
    tickers = [s["ticker"] for s in db.get_all_stocks()]
    db.log_activity("refresh_prices", None, "start", {"tickers": len(tickers)})
    result = refresh.refresh_prices_bulk(tickers, cfg)
    db.set_refresh_state("last_price_refresh_at", datetime.now(timezone.utc).isoformat())
    db.log_activity("refresh_prices", None, "ok", {
        "ok": result["ok"],
        "failed": len(result["failed"]),
        "split_suspects": result["split_suspects"],
        "chunks_failed": result["chunks_failed"],
    })
    log.info("Prijsrefresh klaar: %d bijgewerkt, %d mislukt, %d split-verdacht",
             result["ok"], len(result["failed"]), len(result["split_suspects"]))


def _run_fundamentals_refresh(cfg: dict) -> None:
    global _fundamentals_running
    limit = int(cfg.get("refresh", {}).get("fundamentals_per_night", 100))

    # Deelt de vlag met de handmatige knop: draait er al een ronde, dan slaan we
    # deze over in plaats van het aantal Yahoo-aanroepen te verdubbelen.
    with _fundamentals_lock:
        if _fundamentals_running:
            log.info("Nachtelijke ronde jaarcijfers overgeslagen: er loopt er al een")
            db.log_activity("refresh_fundamentals", None, "skipped",
                            {"reason": "handmatige ronde was nog bezig"})
            return
        _fundamentals_running = True

    try:
        _do_fundamentals_refresh(cfg, limit)
    finally:
        with _fundamentals_lock:
            _fundamentals_running = False


def _do_fundamentals_refresh(cfg: dict, limit: int) -> None:
    db.log_activity("refresh_fundamentals", None, "start", {"limit": limit})
    result = refresh.refresh_fundamentals_batch(limit, cfg)
    db.set_refresh_state("last_fundamentals_refresh_at", datetime.now(timezone.utc).isoformat())
    db.log_activity("refresh_fundamentals", None,
                    "warning" if result["storm_detected"] else "ok", {
                        "attempted": result["attempted"],
                        "ok": result["ok"],
                        "failed": len(result["failed"]),
                        "empty": len(result["empty"]),
                        "insufficient": result["insufficient"],
                        "storm_detected": result["storm_detected"],
                    })
    # Suspenderen gebeurt alleen buiten een storm, en alleen voor tickers die
    # aan alle drie de voorwaarden voldoen (zie refresh.maybe_auto_suspend).
    if not result["storm_detected"]:
        for ticker in result["failed"] + result["empty"]:
            try:
                refresh.maybe_auto_suspend(ticker)
            except Exception:
                log.exception("Suspend-check mislukt voor %s", ticker)
    log.info("Fundamentals-refresh klaar: %d/%d ok, storm=%s",
             result["ok"], result["attempted"], result["storm_detected"])


def _run_weekly_tasks(cfg: dict) -> None:
    db.log_activity("reprobe", None, "start", None)
    # 40 per week: met ruim honderd gesuspendeerde tickers krijgt elk er zo eens
    # per drie weken een nieuwe kans, in plaats van eens per zes.
    result = refresh.weekly_reprobe(40, cfg)
    db.set_refresh_state("last_reprobe_at", datetime.now(timezone.utc).isoformat())
    db.log_activity("reprobe", None, "ok", {
        "attempted": result["attempted"],
        "reactivated": result["reactivated"],
        "delisted": result["delisted"],
    })

    # Log opschonen: zonder dit groeit activity_log ongelimiteerd.
    try:
        with db._cursor() as cur:
            cur.execute(
                "DELETE FROM activity_log WHERE timestamp < %s",
                ((datetime.now(timezone.utc) - timedelta(days=90)).isoformat(),),
            )
        db.set_refresh_state("last_prune_at", datetime.now(timezone.utc).isoformat())
    except Exception:
        log.exception("Opschonen activity_log mislukt")


def _scheduler_loop(cfg: dict) -> None:
    """
    Achtergrondloop (daemon thread) die elk kwartier kijkt of er werk is.

    De vorige versie stuurde op de leeftijd van de marktdata. Dat ging mis zodra
    één ticker toevallig vers was: dan leek alles vers en bleef de refresh uit.
    Nu bepaalt uitsluitend `refresh_state` wat er moet gebeuren.
    """
    log.info("Scheduler v2 gestart (tick=%ds, prijzen %02d:%02d, fundamentals %02d:00 Amsterdam)",
             SCHEDULER_TICK_SECONDS, PRICE_REFRESH_HOUR, PRICE_REFRESH_MINUTE, FUNDAMENTALS_HOUR)
    while True:
        try:
            db.set_refresh_state("last_scheduler_tick_at", datetime.now(timezone.utc).isoformat())
            now = _local_now()

            past_price_time = (now.hour, now.minute) >= (PRICE_REFRESH_HOUR, PRICE_REFRESH_MINUTE)
            if past_price_time and not _ran_today("last_price_refresh_at"):
                _run_price_refresh(cfg)

            if now.hour >= FUNDAMENTALS_HOUR and not _ran_today("last_fundamentals_refresh_at"):
                _run_fundamentals_refresh(cfg)

            if _days_since("last_reprobe_at") >= 7:
                _run_weekly_tasks(cfg)

        except Exception:
            log.exception("Scheduler-tick crashte — loop blijft draaien")
        time.sleep(SCHEDULER_TICK_SECONDS)


# ---------------------------------------------------------------------------
# API — Refresh (background job)
# ---------------------------------------------------------------------------

@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """Start a background refresh: fetch data + recalculate all scores.

    Als geen specifieke tickers meegegeven worden, worden alleen verouderde
    tickers opgehaald (ouder dan STALE_HEAVY_DAYS). Stuur force=true mee
    om alle tickers te forceren.
    """
    data = request.get_json(silent=True) or {}
    tickers = data.get("tickers")   # optional: refresh only specific tickers
    force   = data.get("force", False)

    jid = _new_job()
    cfg = load_config()

    if not tickers:
        all_tickers = [s["ticker"] for s in db.get_all_stocks()]
        tickers = all_tickers if force else _get_stale_tickers(all_tickers, STALE_HEAVY_DAYS)

    threading.Thread(target=_run_refresh_job, args=(jid, tickers, cfg), daemon=True).start()
    return jsonify({"job_id": jid})


@app.route("/api/refresh/status")
def api_refresh_status():
    jid = request.args.get("job_id", "")
    job = _get_job(jid)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/startup_job")
def api_startup_job():
    """Return the job_id of the auto-refresh that was started at startup, if any."""
    return jsonify({"job_id": _startup_job_id})


# ---------------------------------------------------------------------------
# API — Cron batch refresh (externe scheduler, bv. GitHub Actions)
# ---------------------------------------------------------------------------

@app.route("/api/cron/refresh-batch", methods=["POST"])
def api_cron_refresh_batch():
    """Refresh de N oudste tickers. Aangeroepen door externe cron.

    Auth: header X-Cron-Token moet matchen met env var CRON_TOKEN.
    Param: ?limit=N (default 90) — aantal tickers deze batch.

    Spawnt een background job en retourneert het job_id. De cron-runner
    moet /api/refresh/status pollen om te zien wanneer de batch klaar is.
    """
    auth_err = _check_cron_auth()
    if auth_err is not None:
        return auth_err

    try:
        limit = max(1, min(500, int(request.args.get("limit", "90"))))
    except ValueError:
        limit = 90

    all_tickers = [s["ticker"] for s in db.get_all_stocks()]
    if not all_tickers:
        return jsonify({"job_id": None, "n_tickers": 0, "message": "Geen tickers in DB"})

    fetched = db.get_latest_fetched_dates()
    # Oudste eerst; nooit-gefetchte tickers krijgen voorrang (prefix "0")
    ordered = sorted(all_tickers, key=lambda t: fetched.get(t) or "0000-00-00")
    batch = ordered[:limit]

    cfg = load_config()
    jid = _new_job()
    threading.Thread(target=_run_refresh_job, args=(jid, batch, cfg), daemon=True).start()

    oldest_date = fetched.get(batch[0]) or "never"
    log.info("Cron refresh-batch gestart: job=%s, n=%d, oudste=%s (%s)",
             jid, len(batch), batch[0], oldest_date)
    return jsonify({
        "job_id": jid,
        "n_tickers": len(batch),
        "total_tickers": len(all_tickers),
        "oldest_ticker": batch[0],
        "oldest_date": oldest_date,
        "poll_url": f"/api/refresh/status?job_id={jid}",
    })


def _check_cron_auth():
    """Retourneert None bij geldig token, anders (response, status)-tuple."""
    expected = os.environ.get("CRON_TOKEN")
    provided = request.headers.get("X-Cron-Token", "")
    if not expected:
        return jsonify({"error": "CRON_TOKEN niet geconfigureerd op de server"}), 503
    if provided != expected:
        return jsonify({"error": "unauthorized"}), 401
    return None


@app.route("/api/cron/next-batch", methods=["GET"])
def api_cron_next_batch():
    """Geeft de N oudste tickers terug zonder ze op te halen.

    Auth: X-Cron-Token header.
    Param: ?limit=N (default 90, max 1000).

    Wordt door de externe cron-runner gebruikt om zelf een loop te draaien:
    voor elke ticker in deze lijst roept de runner vervolgens /api/cron/refresh-one
    aan. Zo hoeft de server geen lange achtergrond-state bij te houden.
    """
    auth_err = _check_cron_auth()
    if auth_err is not None:
        return auth_err

    try:
        limit = max(1, min(1000, int(request.args.get("limit", "90"))))
    except ValueError:
        limit = 90

    all_tickers = [s["ticker"] for s in db.get_all_stocks()]
    if not all_tickers:
        return jsonify({"tickers": [], "total": 0})

    # Sorteer op laatste POGING, niet op laatste geslaagde fetch. Tickers waar
    # Yahoo niets voor teruggeeft krijgen namelijk nooit een financials-rij; op
    # fetched_date sorteren zette die permanent vooraan, waardoor ze elke nacht
    # de hele batch opslokten en echte tickers nooit meer aan de beurt kwamen.
    attempts = db.get_last_attempt_dates()
    fetched = db.get_latest_fetched_dates()
    ordered = sorted(all_tickers, key=lambda t: attempts.get(t) or "0000-00-00")
    batch = ordered[:limit]

    return jsonify({
        "tickers": batch,
        "count": len(batch),
        "total": len(all_tickers),
        "oldest_date": fetched.get(batch[0]) or "never",
        "oldest_attempt": attempts.get(batch[0]) or "never",
    })


@app.route("/api/cron/refresh-one/<ticker>", methods=["POST"])
def api_cron_refresh_one(ticker):
    """Synchroon één ticker ophalen + herrekenen. Retourneert resultaat direct.

    Auth: X-Cron-Token header.

    Ontworpen om snel te zijn (<30s) zodat de call altijd binnen het gunicorn-
    request-timeout blijft. Een externe cron-runner roept deze endpoint in een
    loop aan voor elke ticker uit /api/cron/next-batch.

    Retourneert: {ticker, ok, signal, combined_fv, price, warnings, elapsed_s}
    """
    auth_err = _check_cron_auth()
    if auth_err is not None:
        return auth_err

    t = ticker.upper()
    if not db.get_stock(t):
        return jsonify({"ticker": t, "ok": False, "error": "ticker niet in DB"}), 404

    cfg = load_config()
    t0 = time.time()
    fetch_warnings: list[str] = []
    calc_result: dict = {}

    try:
        fetch_warnings = fetch_and_store(t) or []
        status = "warning" if fetch_warnings else "ok"
        db.log_activity("fetch", t, status, {
            "source": "Yahoo Finance",
            "warnings": fetch_warnings,
        })
    except Exception as e:
        log.exception("refresh-one fetch faalde voor %s", t)
        db.log_activity("fetch", t, "error", {"error": str(e)})
        return jsonify({
            "ticker": t, "ok": False, "phase": "fetch",
            "error": str(e), "elapsed_s": round(time.time() - t0, 1),
        }), 200  # 200 zodat de cron-runner doorgaat; ok=false zegt dat deze faalde

    try:
        calc_result = run_ticker(t, cfg)
        db.log_activity("recalculate", t, "ok", {
            "signal": calc_result.get("signal"),
            "fv": calc_result.get("combined_fv"),
        })
    except Exception as e:
        log.exception("refresh-one calc faalde voor %s", t)
        db.log_activity("recalculate", t, "error", {"error": str(e)})
        return jsonify({
            "ticker": t, "ok": False, "phase": "calc",
            "error": str(e), "elapsed_s": round(time.time() - t0, 1),
        }), 200

    # Self-heal: na N opeenvolgende fetch-failures auto-suspenden
    auto_suspended = _maybe_auto_suspend(t)

    return jsonify({
        "ticker":      t,
        "ok":          True,
        "signal":      calc_result.get("signal"),
        "combined_fv": calc_result.get("combined_fv"),
        "price":       calc_result.get("price"),
        "quality":     calc_result.get("quality_score"),
        "warnings":    fetch_warnings + (calc_result.get("warnings") or []),
        "auto_suspended": auto_suspended,
        "elapsed_s":   round(time.time() - t0, 1),
    })


AUTO_SUSPEND_THRESHOLD = int(os.environ.get("AUTO_SUSPEND_AFTER_FAILS", "7"))


def _maybe_auto_suspend(ticker: str) -> bool:
    """
    Zet active=0 + auto_suspended_at=now als consecutive_failures >= drempel.
    Returns True als suspension gebeurd is.
    """
    dq = db.get_data_quality(ticker) or {}
    fails = dq.get("consecutive_failures") or 0
    if fails < AUTO_SUSPEND_THRESHOLD:
        return False
    stock = db.get_stock(ticker)
    if not stock or stock.get("active") == 0:
        return False  # al gedeactiveerd
    reason = f"auto-suspend na {fails} opeenvolgende fetch-failures"
    db.upsert_stock(
        ticker,
        active=0,
        auto_suspended_at=datetime.utcnow().isoformat(),
        auto_suspend_reason=reason,
    )
    db.log_activity("remove", ticker, "ok", {"reason": reason, "auto": True, "fails": fails})
    log.info("Auto-suspended %s na %d fails", ticker, fails)
    return True


# ---------------------------------------------------------------------------
# API — Recalculate (no re-fetch)
# ---------------------------------------------------------------------------

@app.route("/api/recalculate", methods=["POST"])
def api_recalculate():
    """Recalculate scores from cached DB data (no network calls)."""
    cfg = load_config()
    data = request.get_json(silent=True) or {}
    tickers = data.get("tickers") or [s["ticker"] for s in db.get_all_stocks()]

    results = []
    for ticker in tickers:
        try:
            r = run_ticker(ticker, cfg)
            results.append({"ticker": ticker, "signal": r.get("signal"), "ok": True})
        except Exception as e:
            results.append({"ticker": ticker, "ok": False, "error": str(e)})

    return jsonify(results)


@app.route("/api/fv-diagnostics/<ticker>")
def api_fv_diagnostics(ticker: str):
    """
    Diagnose per ticker: alle inputs + outputs van de FV-pipeline.
    Bedoeld om snel te zien waarom een combined_fv er raar uitziet
    (schaal-bug, FX-mismatch, negatieve inputs, gedropte methodes).
    """
    t, err = _validate_ticker(ticker)
    if err:
        return jsonify({"error": err}), 400

    stock_info = db.get_stock(t)
    if not stock_info:
        return jsonify({"error": f"{t} niet in database"}), 404

    cfg = load_config()
    try:
        calc = run_ticker(t, cfg)
    except Exception as e:
        log.exception("fv-diagnostics calc failed %s", t)
        return jsonify({"error": f"berekening faalde: {e}"}), 500

    market_data = db.get_market_data(t) or {}
    dq          = db.get_data_quality(t) or {}
    annual_rows = db.get_financials(t, "annual")
    latest_fy   = annual_rows[0] if annual_rows else {}

    price        = market_data.get("price")
    market_cap   = market_data.get("market_cap")
    shares       = latest_fy.get("shares_outstanding")
    implied_mc   = (shares * price) if (shares and price) else None
    mc_consistency = None
    if market_cap and implied_mc and market_cap > 0:
        mc_consistency = round(abs(market_cap - implied_mc) / market_cap, 3)

    combined_fv = calc.get("combined_fv")
    fv_price_ratio = (combined_fv / price) if (price and combined_fv and combined_fv > 0) else None

    per_method = {
        "multiples_fv":  calc.get("multiples_fv"),
        "graham_fv":     calc.get("graham_fv"),
        "perpetuity_fv": calc.get("perpetuity_fv"),
    }

    return jsonify(_sanitize({
        "ticker":            t,
        "name":              stock_info.get("name"),
        "sector":            stock_info.get("sector"),
        "market":            stock_info.get("market"),
        "currency":          stock_info.get("currency"),
        "financial_currency": stock_info.get("financial_currency"),
        "quote_type":        stock_info.get("quote_type"),
        "price":             price,
        "market_cap":        market_cap,
        "shares_outstanding": shares,
        "implied_market_cap": implied_mc,
        "mc_consistency_ratio": mc_consistency,
        "normalized": {
            "eps":           calc.get("normalized_eps"),
            "ebitda":        calc.get("normalized_ebitda"),
            "fcf":           calc.get("normalized_fcf"),
            "owner_earnings": calc.get("normalized_owner_earn"),
        },
        "fair_values": {
            "per_method":    per_method,
            "conservative": calc.get("conservative_fv"),
            "base":         calc.get("base_fv"),
            "optimistic":   calc.get("optimistic_fv"),
            "combined":     combined_fv,
        },
        "fv_price_ratio":    round(fv_price_ratio, 3) if fv_price_ratio else None,
        "fv_ratio_oob":      bool(fv_price_ratio and (fv_price_ratio < 0.1 or fv_price_ratio > 10.0)),
        "fv_confidence":     calc.get("fv_confidence"),
        "fv_spread_pct":     calc.get("fv_spread_pct"),
        "fv_methods_used":   calc.get("fv_methods_used"),
        "fv_methods_dropped": calc.get("fv_methods_dropped") or [],
        "quality_score":     calc.get("quality_score"),
        "signal":            calc.get("signal"),
        "margin_of_safety":  calc.get("margin_of_safety"),
        "data_status":       dq.get("data_status"),
        "data_completeness": dq.get("completeness_pct"),
        "data_issues":       dq.get("issues") or [],
        "warnings":          calc.get("warnings") or [],
        "latest_fiscal_year": latest_fy.get("fiscal_year"),
    }))


# ---------------------------------------------------------------------------
# API — Stocks (watchlist management)
# ---------------------------------------------------------------------------

@app.route("/api/stocks", methods=["GET"])
def api_stocks():
    return jsonify(db.get_all_stocks())


@app.route("/api/stocks", methods=["POST"])
def api_add_stock():
    data = request.get_json() or {}
    ticker, err = _validate_ticker(data.get("ticker") or "")
    if err:
        log.warning("api_add_stock afgewezen: %s", err)
        return jsonify({"error": err}), 400

    if db.get_stock(ticker):
        return jsonify({"error": f"{ticker} staat al in de watchlist"}), 409

    remap = remap_rules.lookup(ticker)
    force = bool(data.get("force"))
    if remap and not force:
        primary, reason = remap
        return jsonify({
            "warning": "secondary_listing",
            "ticker": ticker,
            "suggested_primary": primary,
            "reason": reason,
            "hint": "POST opnieuw met \"force\": true om toch toe te voegen, of gebruik de suggested_primary.",
        }), 409

    db.upsert_stock(ticker, active=1, added_date=datetime.utcnow().date().isoformat())
    db.log_activity("add", ticker, "ok", {"source": "manual"})

    # Immediately fetch basic data
    cfg = load_config()
    jid = _new_job()

    def _fetch_one():
        _update_job(jid, status="running", current=f"Fetching {ticker}…")
        try:
            warn = fetch_and_store(ticker)
            result = run_ticker(ticker, cfg)
            status = "warning" if warn else "ok"
            db.log_activity("fetch", ticker, status, {
                "source": "Yahoo Finance",
                "warnings": warn or [],
                "signal": result.get("signal"),
            })
            _update_job(jid, status="done", progress=100, current="Done")
        except Exception as e:
            db.log_activity("fetch", ticker, "error", {"error": str(e)})
            _update_job(jid, status="done", errors={ticker: [str(e)]})

    threading.Thread(target=_fetch_one, daemon=True).start()
    return jsonify({"ticker": ticker, "job_id": jid}), 201


@app.route("/api/stocks/import", methods=["POST"])
def api_import_stocks():
    """
    Voeg tickers toe mét hun metadata, zónder direct op te halen.
    Body: {"stocks": [{"ticker","name","isin","market","currency","sector"}, ...]}

    Bedoeld voor `import_tickers.py --apply-via-api`. Anders dan
    `/api/stocks/bulk` start dit géén fetch: bij een onboarding-ronde van
    honderden tickers zou dat de machine (256MB) omleggen. De nachtelijke
    fundamentals-rotatie pakt ze vanzelf met voorrang op, want hun
    `financials.fetched_date` is NULL.

    Aannemen dat de aanroeper al geprobed heeft: symbolen die Yahoo niet kent
    horen hier niet binnen te komen.
    """
    data = request.get_json(silent=True) or {}
    rows = data.get("stocks") or []
    if not isinstance(rows, list):
        return jsonify({"error": "stocks moet een lijst zijn"}), 400
    if len(rows) > 500:
        return jsonify({"error": "max 500 per aanroep"}), 400

    added, skipped, rejected = [], [], []
    today = datetime.utcnow().date().isoformat()
    for row in rows:
        if not isinstance(row, dict):
            rejected.append({"ticker": row, "reason": "geen object"})
            continue
        ticker, err = _validate_ticker(row.get("ticker") or "")
        if err:
            rejected.append({"ticker": row.get("ticker"), "reason": err})
            continue
        if db.get_stock(ticker):
            skipped.append(ticker)
            continue
        fields = {"active": 1, "added_date": today}
        for key in ("name", "isin", "market", "currency", "sector"):
            if row.get(key):
                fields[key] = row[key]
        if row.get("currency"):
            # Aanname tot de eerste fetch het echte rapportagevaluta oplevert.
            fields.setdefault("financial_currency", row["currency"])
        try:
            db.upsert_stock(ticker, **fields)
            added.append(ticker)
        except Exception as e:  # noqa: BLE001 — één slechte rij mag de rest niet blokkeren
            log.exception("Import mislukt voor %s", ticker)
            rejected.append({"ticker": ticker, "reason": str(e)})

    if added:
        db.log_activity("import", None, "ok", {
            "added": len(added), "skipped": len(skipped),
            "source": data.get("source") or "import_tickers.py",
        })
    return jsonify({"added": added, "skipped": skipped, "rejected": rejected}), 201


@app.route("/api/stocks/bulk", methods=["POST"])
def api_add_stocks_bulk():
    """Add multiple tickers at once. Body: {tickers: ["AAPL", "ASML.AS", ...]}"""
    data = request.get_json() or {}
    raw = data.get("tickers") or []
    added = []
    skipped = []
    rejected = []
    for raw_ticker in raw:
        ticker, err = _validate_ticker(raw_ticker)
        if err:
            rejected.append({"ticker": raw_ticker, "reason": err})
            log.warning("bulk-add afgewezen: %r → %s", raw_ticker, err)
            continue
        if db.get_stock(ticker):
            skipped.append(ticker)
        else:
            db.upsert_stock(ticker, active=1, added_date=datetime.utcnow().date().isoformat())
            db.log_activity("add", ticker, "ok", {"source": "bulk import"})
            added.append(ticker)

    if not added:
        return jsonify({"added": [], "skipped": skipped, "rejected": rejected, "job_id": None})

    cfg = load_config()
    jid = _new_job()

    def _fetch_all():
        total = len(added)
        _update_job(jid, status="running", total=total)

        def _progress(ticker: str, idx: int, tot: int):
            _update_job(jid, current=f"Fetching {ticker}…", progress=int(idx / tot * 70))

        try:
            fetch_results = fetch_all_tickers(added, progress_cb=_progress)
        except Exception as e:
            log.exception("Bulk-add fetch crashte")
            _update_job(jid, status="error", current=f"Bulk fetch crashte: {e}")
            return

        for idx, ticker in enumerate(added):
            _update_job(jid, current=f"Berekenen {ticker}…", progress=70 + int(idx / total * 28))
            try:
                result = run_ticker(ticker, cfg)
                warn = fetch_results.get(ticker, [])
                status = "warning" if warn else "ok"
                db.log_activity("fetch", ticker, status, {
                    "source": "Yahoo Finance",
                    "warnings": warn,
                    "signal": result.get("signal"),
                })
            except Exception as e:
                log.exception("Bulk calc failed for %s", ticker)
                db.log_activity("fetch", ticker, "error", {"error": str(e)})
        _update_job(jid, status="done", progress=100, current=f"{len(added)} tickers toegevoegd")

    threading.Thread(target=_fetch_all, daemon=True).start()
    return jsonify({"added": added, "skipped": skipped, "rejected": rejected, "job_id": jid}), 201


@app.route("/api/stocks/<ticker>", methods=["DELETE"])
def api_delete_stock(ticker):
    t = ticker.upper()
    db.log_activity("remove", t, "ok")
    db.delete_stock(t)
    return jsonify({"deleted": t})


@app.route("/api/stocks/bulk-deactivate", methods=["POST"])
def api_bulk_deactivate():
    """Body: {"tickers": [...], "reason": "..."}. Zet active=0 voor alle meegegeven tickers."""
    data = request.get_json() or {}
    raw = data.get("tickers") or []
    reason = data.get("reason") or "bulk-deactivate"
    if not isinstance(raw, list):
        return jsonify({"error": "tickers moet een array zijn"}), 400

    deactivated, skipped, rejected = [], [], []
    for raw_t in raw:
        t, err = _validate_ticker(raw_t)
        if err:
            rejected.append({"ticker": raw_t, "reason": err})
            continue
        if not db.get_stock(t):
            skipped.append(t)
            continue
        db.upsert_stock(t, active=0)
        db.log_activity("remove", t, "ok", {"reason": reason, "bulk": True})
        deactivated.append(t)

    return jsonify({
        "deactivated": deactivated,
        "skipped": skipped,
        "rejected": rejected,
        "count": len(deactivated),
    })


@app.route("/api/stocks/bulk-activate", methods=["POST"])
def api_bulk_activate():
    """Body: {"tickers": [...]}. Heractiveer (active=1) bestaande tickers."""
    data = request.get_json() or {}
    raw = data.get("tickers") or []
    if not isinstance(raw, list):
        return jsonify({"error": "tickers moet een array zijn"}), 400

    activated, skipped, rejected = [], [], []
    for raw_t in raw:
        t, err = _validate_ticker(raw_t)
        if err:
            rejected.append({"ticker": raw_t, "reason": err})
            continue
        if not db.get_stock(t):
            skipped.append(t)
            continue
        db.upsert_stock(t, active=1)
        db.log_activity("add", t, "ok", {"reason": "bulk-activate", "bulk": True})
        activated.append(t)

    return jsonify({
        "activated": activated,
        "skipped": skipped,
        "rejected": rejected,
        "count": len(activated),
    })


@app.route("/api/stocks/suspended", methods=["GET"])
def api_suspended_stocks():
    """Lijst alle tickers die auto-suspended zijn (active=0 + auto_suspended_at not null)."""
    with db._cursor() as cur:
        cur.execute("""
            SELECT s.ticker, s.name, s.auto_suspended_at, s.auto_suspend_reason,
                   dq.consecutive_failures, dq.data_status, dq.last_checked
            FROM stocks s
            LEFT JOIN data_quality dq ON dq.ticker = s.ticker
            WHERE s.active = 0 AND s.auto_suspended_at IS NOT NULL
            ORDER BY s.auto_suspended_at DESC
        """)
        rows = cur.fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/stocks/unsuspend/<ticker>", methods=["POST"])
def api_unsuspend_stock(ticker):
    """Heractiveer een auto-suspended ticker + reset consecutive_failures."""
    t = ticker.upper()
    stock = db.get_stock(t)
    if not stock:
        return jsonify({"error": f"{t} niet in DB"}), 404
    db.upsert_stock(t, active=1, auto_suspended_at=None, auto_suspend_reason=None)
    # Reset failure counter — anders suspend hij meteen weer bij volgende cron
    with db._cursor() as cur:
        cur.execute(
            "UPDATE data_quality SET consecutive_failures = 0 WHERE ticker = %s",
            (t,),
        )
    db.log_activity("add", t, "ok", {"reason": "unsuspend", "manual": True})
    return jsonify({"ticker": t, "unsuspended": True})


@app.route("/api/stocks/remap", methods=["POST"])
def api_remap_stock():
    """
    Body: {"from": "EXOR.AS", "to": "EXO.MI"}.
    Atomair: deactiveer `from`, voeg `to` toe (of heractiveer), start fetch voor `to`.
    """
    data = request.get_json() or {}
    raw_from = data.get("from") or ""
    raw_to = data.get("to") or ""
    src, err1 = _validate_ticker(raw_from)
    dst, err2 = _validate_ticker(raw_to)
    if err1 or err2:
        return jsonify({"error": err1 or err2}), 400
    if src == dst:
        return jsonify({"error": "from en to zijn identiek"}), 400

    if not db.get_stock(src):
        return jsonify({"error": f"{src} niet in watchlist"}), 404

    # Bron deactiveren
    db.upsert_stock(src, active=0)
    db.log_activity("remove", src, "ok", {"reason": "remap", "remap_to": dst})

    # Doel toevoegen of heractiveren
    existing_dst = db.get_stock(dst)
    if existing_dst:
        db.upsert_stock(dst, active=1)
        db.log_activity("add", dst, "ok", {"reason": "remap", "remap_from": src})
        fetch_started = False
    else:
        db.upsert_stock(dst, active=1, added_date=datetime.utcnow().date().isoformat())
        db.log_activity("add", dst, "ok", {"reason": "remap", "remap_from": src})
        # Fetch new ticker in background zoals api_add_stock doet
        cfg = load_config()

        def _fetch_remap():
            try:
                warn = fetch_and_store(dst)
                result = run_ticker(dst, cfg)
                status = "warning" if warn else "ok"
                db.log_activity("fetch", dst, status, {
                    "source": "Yahoo Finance (remap)",
                    "warnings": warn or [],
                    "signal": result.get("signal"),
                })
            except Exception as e:
                db.log_activity("fetch", dst, "error", {"error": str(e)})

        threading.Thread(target=_fetch_remap, daemon=True).start()
        fetch_started = True

    return jsonify({
        "from": src,
        "to": dst,
        "fetch_started": fetch_started,
    })


# ---------------------------------------------------------------------------
# API — Overrides
# ---------------------------------------------------------------------------

@app.route("/api/overrides/<ticker>", methods=["GET"])
def api_get_overrides(ticker):
    ov = db.get_overrides(ticker.upper())
    return jsonify([{"field": f, "year": y, "value": v} for (f, y), v in ov.items()])


VALID_OVERRIDE_FIELDS = {
    "eps_diluted", "fcf", "ebitda", "net_income", "revenue",
    "operating_cf", "total_equity", "total_debt", "shares_outstanding",
    "book_value_ps", "roe", "interest_expense", "capex", "ebit",
    "total_assets", "current_assets", "current_liabilities", "gross_profit",
}

@app.route("/api/overrides/<ticker>", methods=["POST"])
def api_set_override(ticker):
    data = request.get_json()
    field = data.get("field")
    year  = data.get("year")
    value = data.get("value")
    note  = data.get("note", "")
    if not field or value is None:
        return jsonify({"error": "field and value required"}), 400
    if field not in VALID_OVERRIDE_FIELDS:
        return jsonify({"error": f"Onbekend veld '{field}'. Geldige velden: {sorted(VALID_OVERRIDE_FIELDS)}"}), 400
    try:
        float_value = float(value)
    except (ValueError, TypeError):
        return jsonify({"error": "Waarde moet een getal zijn"}), 400
    db.set_override(ticker.upper(), field, year, float_value, note)
    db.log_activity("override", ticker.upper(), "ok", {
        "field": field, "year": year, "value": float_value, "note": note
    })
    return jsonify({"ok": True})


@app.route("/api/price/<ticker>", methods=["POST"])
def api_set_manual_price(ticker):
    """Manually set the current price for a ticker (when Yahoo Finance is stale/unavailable)."""
    data  = request.get_json()
    price = data.get("price")
    note  = data.get("note", "Handmatig ingevoerd")
    if price is None:
        return jsonify({"error": "price required"}), 400
    try:
        price_float = float(price)
    except (ValueError, TypeError):
        return jsonify({"error": "price moet een getal zijn"}), 400

    t = ticker.upper()
    stock = db.get_stock(t)
    if not stock:
        return jsonify({"error": "Ticker niet gevonden"}), 404

    currency = stock.get("currency") or "EUR"

    db.upsert_market_data(t,
        price=price_float,
        last_updated=datetime.now(timezone.utc).isoformat(),
    )
    db.log_activity("manual_price", t, "ok", {
        "price": price_float,
        "currency": currency,
        "note": note,
    })
    # Recalculate signal with new price
    cfg = load_config()
    run_ticker(t, cfg)
    return jsonify({"ok": True, "price": price_float, "currency": currency})


@app.route("/api/overrides/<ticker>", methods=["DELETE"])
def api_delete_override(ticker):
    data = request.get_json()
    db.delete_override(ticker.upper(), data.get("field"), data.get("year"))
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# API — Settings
# ---------------------------------------------------------------------------

@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    return jsonify(load_config())


@app.route("/api/settings", methods=["POST"])
def api_save_settings():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    cfg = load_config()

    # Update screening + signals + valuation keys
    for section in ("screening", "signals", "valuation"):
        if section in data:
            cfg.setdefault(section, {}).update(data[section])

    # Update sectors
    if "sectors" in data:
        for sector_name, vals in data["sectors"].items():
            cfg.setdefault("sectors", {})[sector_name] = vals

    save_config(cfg)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# API — Data quality
# ---------------------------------------------------------------------------

@app.route("/api/data-quality")
def api_data_quality():
    """Return alle data_quality records als lijst, inclusief tickers zonder record."""
    dq_map = db.get_all_data_quality()
    stocks = db.get_all_stocks()
    out = []
    for s in stocks:
        t = s["ticker"]
        dq = dq_map.get(t, {})
        out.append({
            "ticker":                 t,
            "name":                   s.get("name"),
            "data_status":            dq.get("data_status"),
            "completeness_pct":       dq.get("completeness_pct"),
            "years_available":        dq.get("years_available"),
            "latest_fy":              dq.get("latest_fy"),
            "freshness_days":         dq.get("freshness_days"),
            "fetch_success":          dq.get("fetch_success"),
            "consecutive_failures":   dq.get("consecutive_failures") or 0,
            "issues":                 dq.get("issues") or [],
            "last_checked":           dq.get("last_checked"),
        })
    # Sorteer: slechtste status eerst zodat problemen bovenaan staan
    order = {"missing": 0, "bad": 1, "warning": 2, "ok": 3, None: 4}
    out.sort(key=lambda r: (order.get(r.get("data_status"), 4), -(r.get("consecutive_failures") or 0)))
    return jsonify(out)


@app.route("/api/refresh/prices", methods=["POST"])
def api_refresh_prices():
    """
    Start direct een koersronde. Normaal doet de scheduler dit na 18:30, maar
    handmatig starten is handig na een storing of om iets te controleren.

    Body (optioneel): {"tickers": [...]} om alleen die tickers te doen.
    Draait op de achtergrond — een volledige ronde duurt langer dan een
    HTTP-request mag duren.
    """
    data = request.get_json(silent=True) or {}
    tickers = data.get("tickers") or [s["ticker"] for s in db.get_all_stocks()]
    cfg = load_config()

    def _work():
        try:
            db.log_activity("refresh_prices", None, "start",
                            {"tickers": len(tickers), "trigger": "handmatig"})
            result = refresh.refresh_prices_bulk(tickers, cfg)
            db.set_refresh_state("last_price_refresh_at", datetime.now(timezone.utc).isoformat())
            db.log_activity("refresh_prices", None, "ok", {
                "ok": result["ok"],
                "failed": len(result["failed"]),
                "split_suspects": result["split_suspects"],
                "trigger": "handmatig",
            })
        except Exception:
            log.exception("Handmatige koersronde mislukt")
            db.log_activity("refresh_prices", None, "error", {"trigger": "handmatig"})

    threading.Thread(target=_work, daemon=True).start()
    return jsonify({"started": True, "tickers": len(tickers)})


@app.route("/api/refresh/fundamentals", methods=["POST"])
def api_refresh_fundamentals():
    """
    Draai direct een ronde jaarcijfers, bovenop de nachtelijke rotatie.

    Body (optioneel): {"limit": N}. Handig om een achterstand in te lopen na
    een grote import; de rotatie pakt sowieso de langst niet-geprobeerde
    tickers eerst, dus nieuwe tickers komen vanzelf bovenaan.

    Draait op de achtergrond en weigert een tweede ronde zolang er één loopt:
    twee batches tegelijk verdubbelen het aantal Yahoo-aanroepen per seconde
    en dat is precies waar de rate-limits op afgaan.
    """
    global _fundamentals_running
    data = request.get_json(silent=True) or {}
    cfg = load_config()
    default_limit = int((cfg.get("refresh") or {}).get("fundamentals_per_night", 100))
    try:
        limit = int(data.get("limit") or default_limit)
    except (TypeError, ValueError):
        return jsonify({"error": "limit moet een getal zijn"}), 400
    limit = max(1, min(limit, 500))

    with _fundamentals_lock:
        if _fundamentals_running:
            return jsonify({
                "started": False,
                "reason": "Er loopt al een ronde jaarcijfers. Wacht tot die klaar is.",
            }), 409
        _fundamentals_running = True

    def _work():
        global _fundamentals_running
        try:
            # Zelfde functie als de nachtelijke ronde, zodat storm-guard,
            # suspend-regels en logging niet uit elkaar kunnen gaan lopen.
            _do_fundamentals_refresh(cfg, limit)
        except Exception:
            log.exception("Handmatige ronde jaarcijfers mislukt")
            db.log_activity("refresh_fundamentals", None, "error", {"trigger": "handmatig"})
        finally:
            with _fundamentals_lock:
                _fundamentals_running = False

    threading.Thread(target=_work, daemon=True).start()
    return jsonify({"started": True, "limit": limit})


@app.route("/api/refresh/fundamentals/status")
def api_refresh_fundamentals_status():
    """Loopt er een ronde jaarcijfers? Gebruikt door de knop in Beheer."""
    return jsonify({"running": _fundamentals_running})


@app.route("/api/price-history/backfill", methods=["POST"])
def api_backfill_price_history():
    """
    Haal de koershistorie op die Yahoo nu nog heeft, voor tickers die er nog
    geen hebben. Body (optioneel): {"limit": N, "period": "5y"}.

    Waarom dit apart staat van de dagelijkse koersronde: die legt alleen vast
    wat er vanaf nu gebeurt. Alles van vóór vandaag is nu nog op te halen maar
    verdwijnt zodra de bron wegvalt, en een koersreeks staat nergens anders in
    de database — `market_data` houdt maar één koers per aandeel vast.

    Draait op de achtergrond; deelt het slot met de rondes jaarcijfers zodat we
    Yahoo niet vanuit twee kanten tegelijk bevragen.
    """
    global _fundamentals_running
    data = request.get_json(silent=True) or {}
    try:
        limit = int(data.get("limit") or 200)
    except (TypeError, ValueError):
        return jsonify({"error": "limit moet een getal zijn"}), 400
    limit = max(1, min(limit, 500))
    period = str(data.get("period") or "5y")
    if period not in ("1y", "2y", "5y", "10y", "max"):
        return jsonify({"error": "period moet 1y, 2y, 5y, 10y of max zijn"}), 400

    tickers = data.get("tickers") or db.tickers_without_price_history(limit)
    if not tickers:
        return jsonify({"started": False, "reason": "Alle actieve tickers hebben al koershistorie."})

    with _fundamentals_lock:
        if _fundamentals_running:
            return jsonify({
                "started": False,
                "reason": "Er loopt al een ophaalronde. Wacht tot die klaar is.",
            }), 409
        _fundamentals_running = True

    def _work():
        global _fundamentals_running
        try:
            db.log_activity("backfill_prices", None, "start",
                            {"tickers": len(tickers), "period": period})
            result = refresh.backfill_price_history(tickers, period)
            db.log_activity("backfill_prices", None, "ok", {
                "tickers_ok": result["tickers_ok"],
                "rows": result["rows"],
                "failed": len(result["failed"]),
                "period": period,
            })
        except Exception:
            log.exception("Backfill koershistorie mislukt")
            db.log_activity("backfill_prices", None, "error", {})
        finally:
            with _fundamentals_lock:
                _fundamentals_running = False

    threading.Thread(target=_work, daemon=True).start()
    return jsonify({"started": True, "tickers": len(tickers), "period": period})


@app.route("/api/price-history/stats")
def api_price_history_stats():
    """Omvang van het koersarchief + hoeveel tickers er nog op wachten."""
    stats = db.price_history_stats()
    stats["tickers_zonder_historie"] = len(db.tickers_without_price_history(100000))
    return jsonify(stats)


@app.route("/api/stocks/probe", methods=["POST"])
def api_probe_stocks():
    """
    Test kandidaat-tickers zonder ze toe te voegen. Body: {"tickers": [...]}.

    Bedoeld voor `import_tickers.py --probe`: de beurslijsten leveren het
    symbool van de beurs, en dat wijkt regelmatig af van wat Yahoo gebruikt.
    Dit draait op de Fly-machine omdat daar wél een werkende netwerkverbinding
    naar Yahoo is.

    Synchroon (max ~200 per call, zodat het binnen de gunicorn-timeout blijft).
    """
    data = request.get_json(silent=True) or {}
    raw = data.get("tickers") or []
    if not isinstance(raw, list):
        return jsonify({"error": "tickers moet een lijst zijn"}), 400
    if len(raw) > 200:
        return jsonify({"error": "max 200 tickers per aanroep"}), 400

    candidates, rejected = [], []
    for item in raw:
        ticker, err = _validate_ticker(item)
        if err:
            rejected.append({"ticker": item, "reason": err})
        else:
            candidates.append(ticker)

    result = refresh.probe_tickers(candidates)
    result["rejected"] = rejected
    db.log_activity("probe", None, "ok", {
        "candidates": len(candidates),
        "resolved": len(result["resolved"]),
        "unresolved": len(result["unresolved"]),
    })
    return jsonify(result)


@app.route("/api/health")
def api_health():
    """
    Eén blik op de gezondheid van de verversing — zonder token, want dit is
    juist het endpoint dat moet werken als er iets mis is.

    Het draait om de vraag die eerder onbeantwoord bleef: draait de motor nog,
    en hoe vers is wat ik zie? Zes weken stilstand hoort hier onmiddellijk
    zichtbaar te zijn.
    """
    state = db.get_all_refresh_state()
    now = datetime.now(timezone.utc)

    def _age_hours(key: str) -> float | None:
        raw = state.get(key)
        if not raw:
            return None
        try:
            ts = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (now - ts).total_seconds() / 3600.0

    with db._cursor() as cur:
        cur.execute("""
            SELECT
              COUNT(*) FILTER (WHERE active = 1 AND presumed_delisted_at IS NULL)  AS active,
              COUNT(*) FILTER (WHERE active = 0 AND auto_suspended_at IS NOT NULL
                                 AND presumed_delisted_at IS NULL)                 AS suspended,
              COUNT(*) FILTER (WHERE presumed_delisted_at IS NOT NULL)             AS presumed_delisted
            FROM stocks
        """)
        counts = dict(cur.fetchone())

        cur.execute("""
            SELECT COUNT(*) AS n FROM market_data md
            JOIN stocks s ON s.ticker = md.ticker
            WHERE s.active = 1 AND md.last_updated >= %s
        """, ((now - timedelta(hours=24)).isoformat(),))
        prices_fresh = cur.fetchone()["n"]

        cur.execute("""
            SELECT COUNT(DISTINCT f.ticker) AS n FROM financials f
            JOIN stocks s ON s.ticker = f.ticker
            WHERE s.active = 1 AND f.period_type = 'annual' AND f.fetched_date >= %s
        """, ((now - timedelta(days=30)).date().isoformat(),))
        fundamentals_fresh = cur.fetchone()["n"]

        cur.execute("""
            SELECT COUNT(*) AS n FROM activity_log
            WHERE action = 'storm_detected' AND timestamp >= %s
        """, ((now - timedelta(days=7)).isoformat(),))
        storms = cur.fetchone()["n"]

    # Zonder oordeel op dezelfde manier tellen als het dashboard het toont —
    # zie _effective_signal voor waarom dat via dezelfde functie moet lopen.
    cfg = load_config()
    no_verdict = 0
    for r in db.get_dashboard_data():
        sig = _effective_signal(r.get("price"), r.get("combined_fv"),
                                r.get("quality_score"), r, cfg)
        if sig in ("INSUFFICIENT DATA", "N/A"):
            no_verdict += 1

    active = counts.get("active") or 0
    price_age = _age_hours("last_price_refresh_at")
    fund_age = _age_hours("last_fundamentals_refresh_at")
    tick_age = _age_hours("last_scheduler_tick_at")

    return jsonify({
        "last_price_refresh":        state.get("last_price_refresh_at"),
        "last_fundamentals_refresh": state.get("last_fundamentals_refresh_at"),
        "last_reprobe":              state.get("last_reprobe_at"),
        "price_age_hours":           round(price_age, 1) if price_age is not None else None,
        "fundamentals_age_hours":    round(fund_age, 1) if fund_age is not None else None,
        "prices_fresh_24h_pct":      round(100.0 * prices_fresh / active, 1) if active else 0.0,
        "fundamentals_fresh_30d_pct": round(100.0 * fundamentals_fresh / active, 1) if active else 0.0,
        "active":                    active,
        "suspended":                 counts.get("suspended") or 0,
        "presumed_delisted":         counts.get("presumed_delisted") or 0,
        "assessed":                  active - no_verdict,
        "no_verdict":                no_verdict,
        "storm_last_7d":             storms,
        # Een tick hoort elk kwartier te komen; een half uur stilte betekent dat
        # de thread is omgevallen.
        "scheduler_alive":           tick_age is not None and tick_age < 0.5,
    })


@app.route("/api/gaps-report")
def api_gaps_report():
    """
    Diagnose-endpoint: categoriseert elke ticker in één primaire blocker-bucket.

    Read-only. Joint de opgeslagen data_quality-issues met markt/currency/sector
    uit `stocks`, zodat de hele universe in ~10 telbare buckets valt zonder
    per-ticker Yahoo-bevraging. Input voor scripts/gaps_analyze.py (Fase 0) en
    voor de uiteindelijke remediation-routing (gate-kalibratie vs databron-fix).

    Wijzigt niets aan de kwaliteits-gate; gebruikt data_quality.classify_blockers().
    """
    dq_map = db.get_all_data_quality()
    stocks = db.get_all_stocks()
    out = []
    for s in stocks:
        t = s["ticker"]
        dq = dq_map.get(t, {})
        issues = dq.get("issues") or []
        cls = data_quality.classify_blockers(issues, dq.get("data_status"))
        out.append({
            "ticker":               t,
            "name":                 s.get("name"),
            "markt":                s.get("markt"),
            "currency":             s.get("currency"),
            "sector":               s.get("sector"),
            "active":               s.get("active"),
            "data_status":          dq.get("data_status"),
            "completeness_pct":     dq.get("completeness_pct"),
            "years_available":      dq.get("years_available"),
            "consecutive_failures": dq.get("consecutive_failures") or 0,
            "fetch_success":        dq.get("fetch_success"),
            "primary_blocker":      cls["primary_blocker"],
            "blockers":             cls["blockers"],
            "info_flags":           cls["info_flags"],
            "evidence":             cls["evidence"],
            "issue_count":          len(issues),
        })
    # Slechtste eerst: bad/missing bovenaan, daarna op aantal fails
    order = {"missing": 0, "bad": 1, "warning": 2, "ok": 3, None: 4}
    out.sort(key=lambda r: (order.get(r.get("data_status"), 4), -(r.get("consecutive_failures") or 0)))
    return jsonify(out)


@app.route("/api/data-quality/recompute", methods=["POST"])
def api_data_quality_recompute():
    """
    Her-evalueer data_quality voor alle (of opgegeven) tickers vanuit de REEDS
    opgeslagen financials/market — geen netwerkcall naar Yahoo.

    Gebruikt om een gewijzigde gate (zie data_quality.evaluate / Route A) direct
    op de bestaande data toe te passen i.p.v. te wachten op de nachtelijke cron.
    Fetch-tracking velden (consecutive_failures, fetch_success, freshness_days)
    blijven behouden — een recompute is geen echte fetch.

    Body: {"dry_run": true, "tickers": [...]}  (dry_run default True)
    Geeft de before/after status-verdeling + transitie-matrix terug.
    """
    data = request.get_json(silent=True) or {}
    dry_run = bool(data.get("dry_run", True))
    only = data.get("tickers")

    prev_map = db.get_all_data_quality()
    stocks = db.get_all_stocks()
    if only:
        want = {t.upper() for t in only}
        stocks = [s for s in stocks if s["ticker"] in want]

    before_counts: dict = {}
    after_counts: dict = {}
    transitions: dict = {}
    rescued: list[dict] = []
    newly_blocked: list[str] = []

    for s in stocks:
        t = s["ticker"]
        prev = prev_map.get(t, {})
        annual = db.get_financials(t, "annual")
        market = db.get_market_data(t)
        fetch_succeeded = bool(annual) or bool((market or {}).get("price"))

        dq = data_quality.evaluate(
            t, annual, market, s,
            fetch_success=fetch_succeeded,
            prev_consecutive_failures=prev.get("consecutive_failures") or 0,
            fetched_date=None,
        )
        # Een recompute is geen fetch → tracking-velden niet vervalsen.
        dq["consecutive_failures"] = prev.get("consecutive_failures") or 0
        if prev.get("fetch_success") is not None:
            dq["fetch_success"] = prev.get("fetch_success")
        if prev.get("freshness_days") is not None:
            dq["freshness_days"] = prev.get("freshness_days")

        before = prev.get("data_status")
        after = dq["data_status"]
        before_counts[before] = before_counts.get(before, 0) + 1
        after_counts[after] = after_counts.get(after, 0) + 1
        key = f"{before} -> {after}"
        transitions[key] = transitions.get(key, 0) + 1

        was_blocked = before in ("bad", "missing")
        now_blocked = after in ("bad", "missing")
        if was_blocked and not now_blocked:
            rescued.append({"ticker": t, "before": before, "after": after})
        elif not was_blocked and now_blocked and before is not None:
            newly_blocked.append(t)

        if not dry_run:
            db.upsert_data_quality(t, **dq)

    return jsonify({
        "dry_run": dry_run,
        "evaluated": len(stocks),
        "before": before_counts,
        "after": after_counts,
        "transitions": dict(sorted(transitions.items(), key=lambda kv: -kv[1])),
        "rescued_count": len(rescued),
        "rescued": rescued,
        "newly_blocked": newly_blocked,
    })


@app.route("/api/data-quality/cleanup", methods=["POST"])
def api_data_quality_cleanup():
    """
    Deactiveer (active=0) alle tickers die consistent falen op Yahoo Finance.
    Default drempel: consecutive_failures >= 3 OF data_status = 'missing'.
    Body kan {"min_failures": N, "dry_run": true} bevatten.
    Geeft lijst van betrokken tickers terug.
    """
    data = request.get_json(silent=True) or {}
    min_failures = int(data.get("min_failures", 3))
    dry_run = bool(data.get("dry_run", False))

    dq_map = db.get_all_data_quality()
    targets = [
        t for t, dq in dq_map.items()
        if (dq.get("data_status") == "missing"
            or (dq.get("consecutive_failures") or 0) >= min_failures)
    ]

    # Alleen active tickers — anders dubbel werk
    active_tickers = {s["ticker"] for s in db.get_all_stocks()}
    targets = [t for t in targets if t in active_tickers]

    if dry_run:
        return jsonify({"candidates": targets, "would_deactivate": len(targets)})

    for t in targets:
        db.upsert_stock(t, active=0)
        db.log_activity("remove", t, "ok", {"reason": "data_quality cleanup", "auto": True})

    return jsonify({"deactivated": targets, "count": len(targets)})


# ---------------------------------------------------------------------------
# API — Activity log
# ---------------------------------------------------------------------------

@app.route("/api/log")
def api_log():
    ticker = request.args.get("ticker")
    limit  = int(request.args.get("limit", 200))
    return jsonify(db.get_activity_log(ticker=ticker, limit=limit))


@app.route("/log")
def activity_log_page():
    return render_template("log.html")


@app.route("/triage")
def triage_page():
    return render_template("triage.html")


# ---------------------------------------------------------------------------
# API — Stock detail
# ---------------------------------------------------------------------------

@app.route("/api/stock/<ticker>")
def api_stock_detail(ticker):
    t = ticker.upper()
    stock  = db.get_stock(t)
    annual = db.get_financials(t, "annual")
    market = db.get_market_data(t)
    scores = db.get_scores(t)
    hist   = db.get_historical_multiples(t)
    # Data-kwaliteit hoort erbij: het eerste blok van de beslisboom op de
    # detailpagina beantwoordt de vraag of je op deze cijfers kunt bouwen.
    dq = db.get_data_quality(t)
    if dq and isinstance(dq.get("issues"), str):
        try:
            dq["issues"] = json.loads(dq["issues"])
        except (json.JSONDecodeError, TypeError):
            dq["issues"] = [dq["issues"]]
    return jsonify({
        "stock":  stock,
        "annual": annual,
        "market": market,
        "scores": scores,
        "data_quality": dq,
        "historical_multiples": hist,
        "price_history": db.get_price_history(ticker),
    })


# ---------------------------------------------------------------------------
# Startup — draait zowel onder Gunicorn als met python app.py
# ---------------------------------------------------------------------------

_startup_done = False


def _on_startup() -> None:
    """
    Eenmalige startup-taken:
      1. Watchlist seeden vanuit config.yaml
      2. Lichte refresh (marktdata) voor alle tickers direct uitvoeren
      3. Zware refresh voor verouderde tickers direct uitvoeren
      4. Dagelijkse + wekelijkse schedulers starten
    """
    global _startup_job_id, _startup_done
    if _startup_done:
        return
    _startup_done = True

    cfg = load_config()

    # Seed watchlist alleen bij een lege DB (eerste start); daarna is de DB leidend
    if not db.get_all_stocks():
        for ticker in cfg.get("watchlist", []):
            db.upsert_stock(ticker, active=1, added_date=datetime.now(timezone.utc).date().isoformat())
        log.info("Lege DB geseed met %d watchlist-tickers uit config.yaml", len(cfg.get("watchlist", [])))

    # De scheduler draait nu op de machine zelf (die staat altijd aan). GitHub
    # Actions is nog slechts een handmatige noodknop, geen dagelijkse motor meer.
    # De oude gate zette de scheduler uit zodra CRON_TOKEN bestond — daardoor lag
    # de verversing zes weken stil toen GitHub de workflow uitschakelde, zonder
    # dat iets dat opving.
    if os.environ.get("SCHEDULER_ENABLED", "1") != "1":
        log.info("Scheduler uitgezet via SCHEDULER_ENABLED=0")
        return

    # Automatische refresh bij opstart is standaard UIT (config-flag stuurt dit).
    # Bij true wordt direct een refresh gestart; de scheduler hieronder verzorgt
    # daarna de periodieke dagelijkse/stale refreshes op achtergrond.
    auto_refresh = cfg.get("app", {}).get("auto_refresh_on_startup", False)
    if auto_refresh:
        all_tickers = [s["ticker"] for s in db.get_all_stocks()]
        if all_tickers:
            stale = _get_stale_tickers(all_tickers, STALE_HEAVY_DAYS)
            if stale:
                heavy_jid = _new_job()
                _startup_job_id = heavy_jid
                threading.Thread(target=_run_refresh_job, args=(heavy_jid, stale, cfg), daemon=True).start()
                log.info("Startup zware refresh gestart (%d/%d verouderde tickers)", len(stale), len(all_tickers))
            else:
                light_jid = _new_job()
                _startup_job_id = light_jid
                threading.Thread(target=_run_light_job, args=(light_jid, all_tickers), daemon=True).start()
                log.info("Startup lichte refresh gestart (%d tickers, alles vers)", len(all_tickers))
    else:
        log.info("Auto-refresh bij opstart staat uit (config.app.auto_refresh_on_startup=false)")

    # Periodieke scheduler: daemon-thread, stuurt zichzelf via DB-state (restart-safe)
    threading.Thread(target=_scheduler_loop, args=(cfg,), daemon=True).start()


# Wordt aangeroepen bij module-import (Gunicorn) én bij python app.py
_on_startup()


if __name__ == "__main__":
    cfg = load_config()
    port = int(os.environ.get("PORT", cfg.get("app", {}).get("port", 5001)))
    log.info("Starting Stock Screener on http://localhost:%s", port)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
