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
from engine.data_fetcher import fetch_and_store, fetch_market_only, refresh_exchange_rates, get_eur_rate
from engine.screener import run_ticker, run_all

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
    """Return all stocks with latest calculated scores, merged with market data."""
    stocks      = db.get_all_stocks()
    scores      = {s["ticker"]: s for s in db.get_all_scores()}
    markets     = {s["ticker"]: db.get_market_data(s["ticker"]) for s in stocks}
    latest_fys  = db.get_latest_fiscal_years()

    cfg = load_config()
    min_quality   = cfg.get("screening", {}).get("min_quality_score", 7)
    min_cap       = cfg.get("screening", {}).get("min_market_cap_eur", 200)
    new_days      = cfg.get("app", {}).get("new_ticker_days", 7)
    show_all      = request.args.get("show_all",    "false").lower() == "true"
    hide_no_data  = request.args.get("hide_no_data", "false").lower() == "true"

    today = datetime.now(timezone.utc).date()

    rows = []
    for stock in stocks:
        t   = stock["ticker"]
        sc  = scores.get(t, {})
        mkt = markets.get(t) or {}

        mc_eur_m = (mkt.get("market_cap_eur") / 1e6) if mkt.get("market_cap_eur") else None
        q_score  = sc.get("quality_score")
        signal   = sc.get("signal")

        # Verberg aandelen zonder data als dat gevraagd wordt.
        if hide_no_data and signal == "INSUFFICIENT DATA":
            continue

        # Apply filters unless show_all.
        # Tickers with INSUFFICIENT DATA are always shown (no data ≠ low quality).
        if not show_all and signal != "INSUFFICIENT DATA":
            if q_score is not None and q_score < min_quality:
                continue
            if min_cap > 0 and mc_eur_m is not None and mc_eur_m < min_cap:
                continue

        # FCF in millions EUR
        norm_fcf_raw = sc.get("normalized_fcf")
        eur_rate = get_eur_rate(stock.get("currency") or "EUR")
        fcf_m_eur = (norm_fcf_raw * eur_rate / 1e6) if norm_fcf_raw is not None else None

        # "New" ticker badge
        added_str = stock.get("added_date")
        try:
            days_since_added = (today - datetime.fromisoformat(added_str).date()).days if added_str else None
        except (ValueError, TypeError):
            days_since_added = None
        is_new = days_since_added is not None and days_since_added <= new_days

        rows.append({
            "ticker":           t,
            "name":             stock.get("name") or t,
            "sector":           stock.get("sector"),
            "market":           stock.get("market"),
            "currency":         stock.get("currency"),
            "price":            mkt.get("price"),
            "price_eur":        mkt.get("price_eur"),
            "market_cap_m_eur": mc_eur_m,
            "combined_fv_eur":  sc.get("combined_fv_eur"),
            "conservative_fv_eur": sc.get("conservative_fv_eur"),
            "base_fv_eur":      sc.get("base_fv_eur"),
            "optimistic_fv_eur": sc.get("optimistic_fv_eur"),
            "normalized_fcf_m_eur": fcf_m_eur,
            "margin_of_safety": sc.get("margin_of_safety"),
            "price_vs_fv_pct":  _price_vs_fv(mkt.get("price_eur"), sc.get("combined_fv_eur")),
            "quality_score":    q_score,
            "piotroski_score":  sc.get("piotroski_score"),
            "signal":           sc.get("signal") or "N/A",
            "last_updated":        mkt.get("last_updated"),
            "last_calculated":     sc.get("last_calculated"),
            "warnings":            sc.get("warnings") or [],
            "latest_fiscal_year":  latest_fys.get(t),
            "hist_relative":       sc.get("hist_relative") or {},
            "is_new":              is_new,
            "days_since_added":    days_since_added,
        })

    # Sort by margin_of_safety descending
    rows.sort(key=lambda x: x.get("margin_of_safety") or -9999, reverse=True)
    return jsonify(_sanitize(rows))


def _price_vs_fv(price_eur, fv_eur):
    if price_eur and fv_eur and fv_eur > 0:
        return round(price_eur / fv_eur * 100, 1)
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

    _update_job(jid, current="Wisselkoersen ophalen…")
    try:
        refresh_exchange_rates()
    except Exception as e:
        log.warning("FX refresh failed: %s", e)

    for idx, ticker in enumerate(tickers):
        _update_job(jid, current=f"Fetching {ticker}…", progress=int(idx / total * 60))
        try:
            warn = fetch_and_store(ticker)
            if warn:
                errors[ticker] = warn
                db.log_activity("fetch", ticker, "warning", {"warnings": warn})
            else:
                db.log_activity("fetch", ticker, "ok", {"source": "Yahoo Finance"})
        except Exception as e:
            log.exception("Fetch failed for %s", ticker)
            errors[ticker] = [str(e)]
            db.log_activity("fetch", ticker, "error", {"error": str(e)})

    for idx, ticker in enumerate(tickers):
        _update_job(jid, current=f"Calculating {ticker}…", progress=60 + int(idx / total * 38))
        try:
            result = run_ticker(ticker, cfg)
            db.log_activity("recalculate", ticker, "ok", {
                "signal": result.get("signal"),
                "fv_eur": result.get("combined_fv_eur"),
                "warnings": result.get("warnings", []),
            })
        except Exception as e:
            log.exception("Calc failed for %s", ticker)
            errors.setdefault(ticker, []).append(f"Calculation: {e}")
            db.log_activity("recalculate", ticker, "error", {"error": str(e)})

    _update_job(jid, status="done", progress=100, current="Klaar", errors=errors)
    log.info("Refresh job %s complete. Errors: %s", jid, len(errors))


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
    """Lichte refresh: wisselkoersen + marktdata voor alle tickers bijwerken."""
    total = len(tickers)
    _update_job(jid, status="running", total=total, current="Wisselkoersen ophalen...")
    try:
        refresh_exchange_rates()
    except Exception as e:
        log.warning("FX refresh mislukt: %s", e)

    for idx, ticker in enumerate(tickers):
        _update_job(jid, current=f"Marktdata {ticker}...", progress=int((idx + 1) / total * 100))
        try:
            fetch_market_only(ticker)
        except Exception as e:
            log.warning("Light refresh mislukt voor %s: %s", ticker, e)
        time.sleep(0.3)   # voorkomt rate-limiting bij Yahoo Finance

    _update_job(jid, status="done", progress=100, current="Klaar")
    log.info("Light refresh klaar: %d tickers bijgewerkt", total)


def _schedule_light(cfg: dict) -> None:
    """Dagelijkse lichte refresh: alleen marktdata voor alle tickers."""
    tickers = [s["ticker"] for s in db.get_all_stocks()]
    if tickers:
        jid = _new_job()
        log.info("Geplande lichte refresh: %d tickers", len(tickers))
        threading.Thread(target=_run_light_job, args=(jid, tickers), daemon=True).start()
    threading.Timer(24 * 3600, _schedule_light, args=(cfg,)).start()


def _schedule_heavy(cfg: dict) -> None:
    """Wekelijkse zware refresh: alleen verouderde tickers opnieuw ophalen."""
    all_tickers = [s["ticker"] for s in db.get_all_stocks()]
    stale = _get_stale_tickers(all_tickers, STALE_HEAVY_DAYS)
    if stale:
        jid = _new_job()
        log.info("Geplande zware refresh: %d/%d verouderde tickers", len(stale), len(all_tickers))
        threading.Thread(target=_run_refresh_job, args=(jid, stale, cfg), daemon=True).start()
    threading.Timer(7 * 24 * 3600, _schedule_heavy, args=(cfg,)).start()


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
        if not tickers:
            # Alles is al recent bijgewerkt — stuur gelijk-klare job terug
            _update_job(jid, status="done", progress=100, current="Alles is al recent bijgewerkt", errors={})
            return jsonify({"job_id": jid, "skipped": True})

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
            refresh_exchange_rates()
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
        refresh_exchange_rates()
        for idx, ticker in enumerate(added):
            _update_job(jid, current=f"Fetching {ticker}…", progress=int(idx / total * 70))
            try:
                warn = fetch_and_store(ticker)
                result = run_ticker(ticker, cfg)
                status = "warning" if warn else "ok"
                db.log_activity("fetch", ticker, status, {
                    "source": "Yahoo Finance",
                    "warnings": warn or [],
                    "signal": result.get("signal"),
                })
            except Exception as e:
                log.exception("Bulk fetch failed for %s", ticker)
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
    eur_rate = get_eur_rate(currency)
    price_eur = price_float * eur_rate

    db.upsert_market_data(t,
        price=price_float,
        price_eur=price_eur,
        last_updated=datetime.now(timezone.utc).isoformat(),
    )
    db.log_activity("manual_price", t, "ok", {
        "price": price_float,
        "currency": currency,
        "price_eur": price_eur,
        "note": note,
    })
    # Recalculate signal with new price
    cfg = load_config()
    run_ticker(t, cfg)
    return jsonify({"ok": True, "price_eur": price_eur})


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

    # Seed watchlist
    for ticker in cfg.get("watchlist", []):
        if not db.get_stock(ticker):
            db.upsert_stock(ticker, active=1, added_date=datetime.now(timezone.utc).date().isoformat())
            log.info("Watchlist ticker toegevoegd: %s", ticker)

    all_tickers = [s["ticker"] for s in db.get_all_stocks()]
    if all_tickers:
        stale = _get_stale_tickers(all_tickers, STALE_HEAVY_DAYS)
        if stale:
            # Zware refresh voor verouderde tickers (update ook marktdata → geen aparte lichte nodig)
            heavy_jid = _new_job()
            _startup_job_id = heavy_jid
            threading.Thread(target=_run_refresh_job, args=(heavy_jid, stale, cfg), daemon=True).start()
            log.info("Startup zware refresh gestart (%d/%d verouderde tickers)", len(stale), len(all_tickers))
        else:
            # Alles is vers: alleen marktdata even bijwerken
            light_jid = _new_job()
            _startup_job_id = light_jid
            threading.Thread(target=_run_light_job, args=(light_jid, all_tickers), daemon=True).start()
            log.info("Startup lichte refresh gestart (%d tickers, alles vers)", len(all_tickers))

    # Schedulers: licht dagelijks, zwaar wekelijks
    threading.Timer(24 * 3600, _schedule_light, args=(cfg,)).start()
    threading.Timer(7 * 24 * 3600, _schedule_heavy, args=(cfg,)).start()
    log.info("Schedulers actief: licht elke 24u, zwaar elke 7d")


# Wordt aangeroepen bij module-import (Gunicorn) én bij python app.py
_on_startup()


if __name__ == "__main__":
    cfg = load_config()
    port = int(os.environ.get("PORT", cfg.get("app", {}).get("port", 5001)))
    log.info("Starting Stock Screener on http://localhost:%s", port)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
