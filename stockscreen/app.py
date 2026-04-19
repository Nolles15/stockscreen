"""
Flask application — routes, background refresh job, REST API.

Start with:  python app.py
Dashboard:   http://localhost:<port>  (see app.port in config.yaml)
"""

import json
import logging
import math
import os
import threading
import time
import uuid
from datetime import datetime, timezone

import yaml
from flask import Flask, jsonify, render_template, request, redirect, url_for

from engine import db
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

# Dashboard-cache: wordt gevuld door api_dashboard(), geleegd na elke refresh/recalculate.
_dashboard_cache: dict = {"data": None, "expires": 0.0}
_DASHBOARD_CACHE_TTL = 90  # seconden

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
    # Geef gecachede response terug als die nog vers is (max 90s)
    now = time.time()
    if _dashboard_cache["data"] is not None and now < _dashboard_cache["expires"]:
        return jsonify(_dashboard_cache["data"])

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
        fv_price_ratio = (fv / price) if (price and fv and fv > 0) else None
        fv_ratio_oob = (
            fv_price_ratio is not None
            and (fv_price_ratio < 0.1 or fv_price_ratio > 10.0)
        )
        if price and fv and fv > 0 and q_score is not None and not fv_ratio_oob:
            signal = determine_signal(price, fv, q_score, cfg).get("signal")
        elif fv_ratio_oob:
            signal = "INSUFFICIENT DATA"
        else:
            signal = r.get("signal") or "N/A"

        norm_fcf_raw = r.get("normalized_fcf")
        fcf_m        = (norm_fcf_raw / 1e6) if norm_fcf_raw is not None else None

        added_str = r.get("added_date")
        try:
            days_since_added = (today - datetime.fromisoformat(added_str).date()).days if added_str else None
        except (ValueError, TypeError):
            days_since_added = None

        # Markering voor client-side filtering — server filtert niet meer
        low_quality = (
            q_score is not None
            and q_score < min_quality
            and signal != "INSUFFICIENT DATA"
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
            "margin_of_safety":     _margin_of_safety(r.get("price"), r.get("combined_fv")),
            "price_vs_fv_pct":      _price_vs_fv(r.get("price"), r.get("combined_fv")),
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
        })

    rows.sort(key=lambda x: x.get("margin_of_safety") or -9999, reverse=True)
    result = _sanitize(rows)

    # Sla op in cache
    _dashboard_cache["data"] = result
    _dashboard_cache["expires"] = now + _DASHBOARD_CACHE_TTL

    return jsonify(result)


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

    # Dashboard-cache legen zodat verse scores direct zichtbaar zijn
    _dashboard_cache["data"] = None


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


# Minimum-intervals tussen automatische refreshes. Fail-safe tegen restart-storms:
# zelfs als de app elke 10 min crasht en herstart, doen we niet telkens opnieuw werk.
SCHEDULER_TICK_SECONDS    = 3600     # elk uur bekijken of er werk is
LIGHT_REFRESH_INTERVAL_H  = 20       # dagelijkse marktdata (rekening houdend met drift)
HEAVY_REFRESH_INTERVAL_H  = 24       # één keer per etmaal stale-ticker check


def _scheduler_loop(cfg: dict) -> None:
    """
    Achtergrondloop (daemon thread) die elk uur kijkt of er een scheduled refresh
    uitgevoerd moet worden. De DB is leidend voor 'wanneer draaide de laatste
    refresh?' — dus restarts resetten het ritme niet.
    """
    log.info("Scheduler-loop gestart (tick=%ds, light=%dh, heavy=%dh)",
             SCHEDULER_TICK_SECONDS, LIGHT_REFRESH_INTERVAL_H, HEAVY_REFRESH_INTERVAL_H)
    while True:
        try:
            age = _last_market_update_age_hours()
            all_tickers = [s["ticker"] for s in db.get_all_stocks()]

            if all_tickers and (age is None or age >= HEAVY_REFRESH_INTERVAL_H):
                stale = _get_stale_tickers(all_tickers, STALE_HEAVY_DAYS)
                if stale:
                    jid = _new_job()
                    log.info("Scheduler: zware refresh (%d/%d stale tickers)",
                             len(stale), len(all_tickers))
                    threading.Thread(target=_run_refresh_job,
                                     args=(jid, stale, cfg), daemon=True).start()
                    # Zware refresh werkt ook marktdata bij — geen aparte lichte nodig
                elif age is None or age >= LIGHT_REFRESH_INTERVAL_H:
                    jid = _new_job()
                    log.info("Scheduler: lichte refresh (%d tickers, leeftijd=%.1fu)",
                             len(all_tickers), age or -1)
                    threading.Thread(target=_run_light_job,
                                     args=(jid, all_tickers), daemon=True).start()
            elif all_tickers and age is not None and age >= LIGHT_REFRESH_INTERVAL_H:
                jid = _new_job()
                log.info("Scheduler: lichte refresh (%d tickers, leeftijd=%.1fu)",
                         len(all_tickers), age)
                threading.Thread(target=_run_light_job,
                                 args=(jid, all_tickers), daemon=True).start()
        except Exception:
            log.exception("Scheduler-loop tick crashte — ga door")
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

    fetched = db.get_latest_fetched_dates()
    ordered = sorted(all_tickers, key=lambda t: fetched.get(t) or "0000-00-00")
    batch = ordered[:limit]

    return jsonify({
        "tickers": batch,
        "count": len(batch),
        "total": len(all_tickers),
        "oldest_date": fetched.get(batch[0]) or "never",
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

    # Cache legen zodat het dashboard direct de nieuwe waardes ziet
    _dashboard_cache["data"] = None

    return jsonify({
        "ticker":      t,
        "ok":          True,
        "signal":      calc_result.get("signal"),
        "combined_fv": calc_result.get("combined_fv"),
        "price":       calc_result.get("price"),
        "quality":     calc_result.get("quality_score"),
        "warnings":    fetch_warnings + (calc_result.get("warnings") or []),
        "elapsed_s":   round(time.time() - t0, 1),
    })


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

    _dashboard_cache["data"] = None  # cache legen na herberekening
    return jsonify(results)


# ---------------------------------------------------------------------------
# API — Stocks (watchlist management)
# ---------------------------------------------------------------------------

@app.route("/api/stocks", methods=["GET"])
def api_stocks():
    return jsonify(db.get_all_stocks())


@app.route("/api/stocks", methods=["POST"])
def api_add_stock():
    data = request.get_json()
    ticker = (data.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400

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


@app.route("/api/stocks/bulk", methods=["POST"])
def api_add_stocks_bulk():
    """Add multiple tickers at once. Body: {tickers: ["AAPL", "ASML.AS", ...]}"""
    data = request.get_json()
    raw = data.get("tickers") or []
    added = []
    skipped = []
    for ticker in raw:
        ticker = ticker.strip().upper()
        if not ticker:
            continue
        if db.get_stock(ticker):
            skipped.append(ticker)
        else:
            db.upsert_stock(ticker, active=1, added_date=datetime.utcnow().date().isoformat())
            db.log_activity("add", ticker, "ok", {"source": "bulk import"})
            added.append(ticker)

    if not added:
        return jsonify({"added": [], "skipped": skipped, "job_id": None})

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
    return jsonify({"added": added, "skipped": skipped, "job_id": jid}), 201


@app.route("/api/stocks/<ticker>", methods=["DELETE"])
def api_delete_stock(ticker):
    t = ticker.upper()
    db.log_activity("remove", t, "ok")
    db.delete_stock(t)
    return jsonify({"deleted": t})


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

    # Cache legen zodat dashboard direct verandert
    _dashboard_cache["data"] = None
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
    return jsonify({
        "stock":  stock,
        "annual": annual,
        "market": market,
        "scores": scores,
        "historical_multiples": hist,
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

    # Externe cron (bv. GitHub Actions) is leidend zodra CRON_TOKEN gezet is:
    # we slaan dan startup-refresh én in-process scheduler over om dubbelwerk
    # en rate-limit clashes te voorkomen.
    external_cron = bool(os.environ.get("CRON_TOKEN"))
    if external_cron:
        log.info("Externe cron actief (CRON_TOKEN gezet) — in-process scheduler uit")
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
