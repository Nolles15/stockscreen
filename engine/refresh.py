"""
Refresh-motor v2.

Vervangt het model waarin een externe cron per ticker een HTTP-call deed. Drie
redenen waarom dat model niet betrouwbaar was:

1. De rotatie sorteerde op de laatste *geslaagde* fetch, waardoor tickers die
   niets opleveren de wachtrij permanent blokkeerden.
2. Een storing bij Yahoo liet honderden tickers tegelijk falen; elke failure
   telde individueel mee richting auto-suspend, dus één slechte avond kon een
   groot deel van de universe onzichtbaar maken.
3. Koersen werden per ticker opgehaald (~4s elk), wat bij 900+ tickers niet
   binnen een nacht paste — laat staan bij de 3.000 die het doel is.

Deze module haalt koersen in bulk op (honderden per call), rouleert de trage
fundamentals in batches, en telt failures op batch-niveau zodat een storing
nooit tot massa-suspensies leidt.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from . import data_fetcher, db, screener

log = logging.getLogger(__name__)

# Aantal tickers per yf.download-call. Yahoo accepteert er meer, maar bij grote
# chunks is één rate-limit-fout duurder: dan mislukt de hele chunk.
PRICE_CHUNK_SIZE = 200

# Prijsafwijking waarboven we een aandelensplitsing vermoeden in plaats van een
# koersbeweging. Onder deze grens slaan we gewoon op; erboven weigeren we de
# prijs en halen we de ticker volledig opnieuw op (inclusief shares en EPS).
SPLIT_SUSPECT_RATIO = 0.40

# Boven dit aandeel mislukte fetches in één batch gaan we uit van een storing
# aan de kant van Yahoo in plaats van problemen met de tickers zelf.
STORM_THRESHOLD = 0.25


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_config() -> dict:
    """Lees config.yaml. Lokaal geïmporteerd om een cirkelimport met app.py te vermijden."""
    import yaml
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Koersen (snel, in bulk)
# ---------------------------------------------------------------------------

def _extract_last_close(frame: pd.DataFrame, ticker: str, multi: bool) -> float | None:
    """Haal de laatste niet-lege slotkoers uit een yf.download-resultaat."""
    try:
        series = frame[ticker]["Close"] if multi else frame["Close"]
    except (KeyError, TypeError):
        return None
    series = series.dropna()
    if series.empty:
        return None
    value = float(series.iloc[-1])
    return value if value > 0 else None


def refresh_prices_bulk(tickers: list[str], config: dict | None = None) -> dict:
    """
    Werk de koersen van alle meegegeven tickers bij via bulk-downloads.

    Returns {"ok": int, "failed": [ticker], "split_suspects": [ticker],
             "chunks_failed": int}
    """
    result = {"ok": 0, "failed": [], "split_suspects": [], "chunks_failed": 0}
    if not tickers:
        return result

    # Bestaande prijzen vooraf ophalen: nodig voor de split-guard.
    previous = {t: (md or {}).get("price") for t, md in db.get_all_market_data().items()}
    shares = db.get_latest_shares_outstanding()
    eps = db.get_latest_eps_ttm()

    for start in range(0, len(tickers), PRICE_CHUNK_SIZE):
        chunk = tickers[start:start + PRICE_CHUNK_SIZE]
        try:
            frame = data_fetcher._yf_retry(
                lambda: yf.download(
                    tickers=" ".join(chunk),
                    period="5d",
                    interval="1d",
                    group_by="ticker",
                    threads=False,
                    progress=False,
                    auto_adjust=False,
                ),
                label=f"bulk-prijzen chunk {start // PRICE_CHUNK_SIZE + 1}",
                attempts=3,
            )
        except Exception:
            log.exception("Bulk-prijschunk mislukt (%d tickers)", len(chunk))
            result["chunks_failed"] += 1
            result["failed"].extend(chunk)
            continue

        if frame is None or frame.empty:
            result["chunks_failed"] += 1
            result["failed"].extend(chunk)
            continue

        multi = isinstance(frame.columns, pd.MultiIndex)
        for ticker in chunk:
            price = _extract_last_close(frame, ticker, multi)
            if price is None:
                result["failed"].append(ticker)
                continue

            # Yahoo geeft Londense koersen in pence terwijl de cijfers in ponden
            # staan; zonder deling is de koers 100x te hoog en lijkt elk UK-aandeel
            # peperduur. Zelfde regel als in data_fetcher.fetch_ticker.
            if ticker.endswith(".L"):
                price = price / 100.0

            old = previous.get(ticker)
            if old and old > 0 and abs(price - old) / old > SPLIT_SUSPECT_RATIO:
                # Waarschijnlijk een split of een herdenominatie: de koers is dan
                # niet vergelijkbaar met de opgeslagen shares/EPS. Prijs niet
                # opslaan, maar de ticker volledig opnieuw ophalen.
                result["split_suspects"].append(ticker)
                continue

            fields = {"price": price, "last_updated": _now_iso()}
            n_shares = shares.get(ticker)
            if n_shares:
                fields["market_cap"] = price * n_shares
            per_share = eps.get(ticker)
            if per_share and per_share > 0:
                fields["pe_ttm"] = price / per_share

            try:
                db.upsert_market_data(ticker, **fields)
                result["ok"] += 1
            except Exception:
                log.exception("Opslaan koers mislukt voor %s", ticker)
                result["failed"].append(ticker)

    # Vermoedelijke splits volledig opnieuw ophalen zodat koers, shares en EPS
    # weer bij elkaar horen.
    cfg = config if config is not None else _load_config()
    for ticker in result["split_suspects"]:
        try:
            data_fetcher.fetch_and_store(ticker)
            screener.run_ticker(ticker, cfg)
        except Exception:
            log.exception("Her-fetch na split-vermoeden mislukt voor %s", ticker)

    return result


# ---------------------------------------------------------------------------
# Fundamentals (traag, in rotatie)
# ---------------------------------------------------------------------------

def refresh_fundamentals_batch(limit: int = 100, config: dict | None = None) -> dict:
    """
    Ververs de `limit` langst niet-geprobeerde tickers.

    Failures worden pas ná afloop van de batch geteld: als meer dan een kwart
    van de batch faalt gaan we uit van een storing bij Yahoo en laten we alle
    tellers ongemoeid. Anders zou één slechte avond honderden tickers richting
    auto-suspend duwen.
    """
    tickers = db.get_refresh_queue(limit)
    outcome = {
        "attempted": len(tickers),
        "ok": 0,
        "failed": [],
        "insufficient": 0,
        "storm_detected": False,
    }
    if not tickers:
        return outcome

    cfg = config if config is not None else _load_config()
    for ticker in tickers:
        try:
            # count_failure=False: de teller gaat pas omhoog als na afloop blijkt
            # dat dit géén storm was.
            data_fetcher.fetch_and_store(ticker, count_failure=False)
            calc = screener.run_ticker(ticker, cfg)
            if (calc or {}).get("signal") == "INSUFFICIENT DATA":
                outcome["insufficient"] += 1
            outcome["ok"] += 1
        except Exception as exc:
            log.warning("Refresh mislukt voor %s: %s", ticker, exc)
            outcome["failed"].append(ticker)

    failure_rate = len(outcome["failed"]) / len(tickers)
    if failure_rate > STORM_THRESHOLD:
        outcome["storm_detected"] = True
        db.log_activity("storm_detected", None, "warning", {
            "failure_rate": round(failure_rate, 3),
            "attempted": len(tickers),
            "failed": len(outcome["failed"]),
            "note": "Tellers niet opgehoogd — dit lijkt een storing bij de bron.",
        })
        log.warning("Storm gedetecteerd (%.0f%% mislukt) — failure-tellers ongemoeid gelaten",
                    failure_rate * 100)
    else:
        for ticker in outcome["failed"]:
            db.bump_failure_counter(ticker)

    return outcome


# ---------------------------------------------------------------------------
# Suspend-beleid
# ---------------------------------------------------------------------------

# Een ticker mag pas verdwijnen als het écht structureel is: veel pogingen,
# gespreid over ruim een maand, en geen enkel jaarcijfer in de database.
SUSPEND_MIN_FAILURES = 10
SUSPEND_MIN_DAYS = 30
DELISTED_AFTER_DAYS = 90


def maybe_auto_suspend(ticker: str) -> bool:
    """Suspendeer een ticker alleen als hij aan alle drie de voorwaarden voldoet."""
    dq = db.get_data_quality(ticker) or {}
    if (dq.get("consecutive_failures") or 0) < SUSPEND_MIN_FAILURES:
        return False

    first = dq.get("first_failure_at")
    if not first:
        return False
    try:
        first_dt = datetime.fromisoformat(first)
        if first_dt.tzinfo is None:
            first_dt = first_dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    if datetime.now(timezone.utc) - first_dt < timedelta(days=SUSPEND_MIN_DAYS):
        return False

    # Hebben we al bruikbare cijfers, dan is een reeks mislukte fetches geen
    # reden om de ticker te verbergen — de oude cijfers blijven waardevol.
    if db.count_annual_rows(ticker) > 0:
        return False

    db.upsert_stock(
        ticker,
        active=0,
        auto_suspended_at=_now_iso(),
        auto_suspend_reason=(
            f"{dq.get('consecutive_failures')} mislukte pogingen sinds {first[:10]}, "
            "geen enkel jaarcijfer beschikbaar"
        ),
    )
    db.log_activity("auto_suspend", ticker, "warning", {
        "consecutive_failures": dq.get("consecutive_failures"),
        "first_failure_at": first,
    })
    return True


def weekly_reprobe(limit: int = 20, config: dict | None = None) -> dict:
    """
    Geef gesuspendeerde tickers periodiek een nieuwe kans.

    Zonder dit blijft een ticker die tijdelijk onbereikbaar was voorgoed weg —
    precies de stille data-verdwijning die dit project moest oplossen.
    """
    tickers = db.get_reprobe_candidates(limit)
    outcome = {"attempted": len(tickers), "reactivated": [], "delisted": []}
    cfg = config if config is not None else _load_config()

    for ticker in tickers:
        recovered = False
        try:
            data_fetcher.fetch_and_store(ticker, count_failure=False)
            recovered = db.count_annual_rows(ticker) > 0
        except Exception as exc:
            log.info("Re-probe zonder resultaat voor %s: %s", ticker, exc)

        if recovered:
            db.upsert_stock(ticker, active=1, auto_suspended_at=None, auto_suspend_reason=None)
            db.reset_failure_counter(ticker)
            try:
                screener.run_ticker(ticker, cfg)
            except Exception:
                log.exception("Herberekening na re-probe mislukt voor %s", ticker)
            db.log_activity("reactivated", ticker, "ok", {"via": "weekly_reprobe"})
            outcome["reactivated"].append(ticker)
            continue

        # Al heel lang stil en nog steeds niets: waarschijnlijk van de beurs.
        # We gooien niets weg — de ticker blijft zichtbaar in het beheerscherm.
        stock = db.get_stock(ticker) or {}
        suspended_at = stock.get("auto_suspended_at")
        if suspended_at and not stock.get("presumed_delisted_at"):
            try:
                since = datetime.fromisoformat(suspended_at)
                if since.tzinfo is None:
                    since = since.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if datetime.now(timezone.utc) - since > timedelta(days=DELISTED_AFTER_DAYS):
                db.upsert_stock(ticker, presumed_delisted_at=_now_iso())
                db.log_activity("presumed_delisted", ticker, "warning", {
                    "suspended_since": suspended_at,
                })
                outcome["delisted"].append(ticker)

    return outcome
