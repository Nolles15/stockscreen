"""
Data fetcher — wraps yfinance and currency conversion.

Responsibilities:
  • Fetch raw financial statements (income, balance sheet, cash flow)
  • Fetch current market data (price, market cap, EV, trailing ratios)
  • Fetch historical per-year multiples for the last 5 years
  • Fetch EUR exchange rates via Yahoo Finance currency pairs
  • Apply manual overrides from the DB before returning data
  • Return structured dicts ready for the engine pipeline
"""

import logging
from datetime import datetime, date
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from . import db

log = logging.getLogger(__name__)

# Map market suffix → native currency
MARKET_CURRENCIES = {
    ".WA": "PLN",   # Warsaw
    ".ST": "SEK",   # Stockholm
    ".BR": "EUR",   # Brussels (already EUR)
    ".AS": "EUR",   # Amsterdam
    ".OL": "NOK",   # Oslo
    ".DE": "EUR",   # Frankfurt
    ".FI": "EUR",   # Helsinki
    ".L":  "GBP",   # London
}

# Yahoo Finance tickers for EUR FX rates
FX_TICKERS = {
    "USD": "EURUSD=X",
    "PLN": "EURPLN=X",
    "SEK": "EURSEK=X",
    "NOK": "EURNOK=X",
    "GBP": "EURGBP=X",
    "CHF": "EURCHF=X",
    "DKK": "EURDKK=X",
}


def _safe_get(obj, *keys, default=None):
    """Safely walk a dict/object by multiple key options."""
    for key in keys:
        try:
            val = obj[key]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                return float(val)
        except (KeyError, TypeError, ValueError):
            continue
    return default


def _df_value(df: pd.DataFrame, row_keys: list[str], col_idx: int = 0, default=None):
    """Get a cell from a yfinance statement DataFrame by trying multiple row names."""
    if df is None or df.empty:
        return default
    cols = list(df.columns)
    if col_idx >= len(cols):
        return default
    for key in row_keys:
        if key in df.index:
            val = df.loc[key].iloc[col_idx]
            if pd.notna(val):
                return float(val)
    return default


def _col_year(col) -> int:
    """Extract fiscal year from a pandas Timestamp column."""
    if isinstance(col, pd.Timestamp):
        return col.year
    try:
        return int(str(col)[:4])
    except Exception:
        return 0


def infer_currency(ticker: str) -> str:
    for suffix, ccy in MARKET_CURRENCIES.items():
        if ticker.upper().endswith(suffix.upper()):
            return ccy
    return "USD"


# ---------------------------------------------------------------------------
# Exchange rates
# ---------------------------------------------------------------------------

def refresh_exchange_rates() -> dict[str, float]:
    """Fetch latest EUR rates from Yahoo Finance and cache in DB."""
    rates = {"EUR": 1.0}
    for ccy, yf_ticker in FX_TICKERS.items():
        try:
            info = yf.Ticker(yf_ticker).info
            rate = _safe_get(info, "regularMarketPrice", "previousClose", "ask")
            if rate and rate > 0:
                # Yahoo gives EUR/USD etc. — we want units of foreign per 1 EUR
                # EURUSD=X means 1 EUR = X USD, so to convert USD→EUR: divide by X
                # We store rate_to_eur = how many EUR per 1 unit of foreign currency
                rates[ccy] = 1.0 / rate
                db.upsert_exchange_rate(ccy, 1.0 / rate)
        except Exception as e:
            log.warning("FX fetch failed for %s: %s", ccy, e)
    return rates


def get_eur_rate(currency: str) -> float:
    """Return how many EUR 1 unit of `currency` equals. Cached in DB."""
    if currency == "EUR":
        return 1.0
    cached = db.get_exchange_rates()
    if currency in cached:
        return cached[currency]
    # Try live fetch
    yf_ticker = FX_TICKERS.get(currency)
    if yf_ticker:
        try:
            info = yf.Ticker(yf_ticker).info
            rate = _safe_get(info, "regularMarketPrice", "previousClose")
            if rate and rate > 0:
                eur_rate = 1.0 / rate
                db.upsert_exchange_rate(currency, eur_rate)
                return eur_rate
        except Exception:
            pass
    return 1.0  # fallback: assume EUR


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

def fetch_ticker(ticker: str) -> dict[str, Any]:
    """
    Fetch all available data for a single ticker from Yahoo Finance.
    Returns a structured dict:
      {
        "meta": {name, sector, currency, market_cap, ...},
        "market": {price, pe_ttm, ev_ebitda, pb, ...},
        "annual": [{fiscal_year, ebit, ...}, ...],  # last 5 years, newest first
        "warnings": ["..."]
      }
    """
    warnings: list[str] = []
    result: dict[str, Any] = {"ticker": ticker, "annual": [], "market": {}, "meta": {}, "warnings": warnings}

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception as e:
        warnings.append(f"Could not connect to Yahoo Finance: {e}")
        return result

    if not info:
        warnings.append("No data returned from Yahoo Finance — ticker may be invalid.")
        return result

    # ---- Meta ---------------------------------------------------------------
    currency = info.get("currency") or infer_currency(ticker)
    result["meta"] = {
        "name":        info.get("longName") or info.get("shortName") or ticker,
        "sector":      info.get("sector") or "Unknown",
        "currency":    currency,
        "market":      _detect_market(ticker),
        "description": info.get("longBusinessSummary"),
    }

    # ---- Current market data ------------------------------------------------
    price_raw = _safe_get(info, "currentPrice", "regularMarketPrice", "previousClose")
    market_cap_raw = _safe_get(info, "marketCap")
    ev_raw = _safe_get(info, "enterpriseValue")
    eur_rate = get_eur_rate(currency)

    analyst_target_raw = _safe_get(info, "targetMeanPrice")
    result["market"] = {
        "price":                price_raw,
        "price_eur":            (price_raw * eur_rate) if price_raw else None,
        "market_cap":           market_cap_raw,
        "market_cap_eur":       (market_cap_raw * eur_rate) if market_cap_raw else None,
        "enterprise_value":     ev_raw,
        "enterprise_value_eur": (ev_raw * eur_rate) if ev_raw else None,
        "pe_ttm":               _safe_get(info, "trailingPE"),
        "ev_ebitda_ttm":        _safe_get(info, "enterpriseToEbitda"),
        "pb_ratio":             _safe_get(info, "priceToBook"),
        "last_updated":         datetime.utcnow().isoformat(),
        # Analyst consensus
        "analyst_target_raw":   analyst_target_raw,
        "analyst_target_eur":   (analyst_target_raw * eur_rate) if analyst_target_raw else None,
        "analyst_consensus":    info.get("recommendationKey"),
        "analyst_n":            info.get("numberOfAnalystOpinions"),
    }

    if not price_raw:
        warnings.append("Current price unavailable — check ticker symbol.")

    # ---- Annual financial statements ----------------------------------------
    try:
        inc = t.income_stmt        # columns = dates newest-first
        bal = t.balance_sheet
        cf  = t.cashflow
    except Exception as e:
        warnings.append(f"Financial statements unavailable: {e}")
        return result

    years_found = 0
    if inc is not None and not inc.empty:
        for col_idx, col in enumerate(inc.columns[:5]):   # max 5 years
            yr = _col_year(col)
            if yr < 2010:
                continue

            # Income statement
            ebit = _df_value(inc, ["EBIT", "Operating Income", "Ebit"], col_idx)
            ebitda = _df_value(inc, ["EBITDA", "Ebitda"], col_idx) or _calc_ebitda(inc, bal, col_idx)
            net_income = _df_value(inc, ["Net Income", "Net Income Common Stockholders", "NetIncome"], col_idx)
            eps_diluted = _df_value(inc, ["Diluted EPS", "Basic EPS", "EPS"], col_idx)
            revenue = _df_value(inc, ["Total Revenue", "Revenue"], col_idx)
            gross_profit = _df_value(inc, ["Gross Profit", "GrossProfit"], col_idx)
            interest_exp = _df_value(inc, [
                "Interest Expense", "Interest Expense Non Operating",
                "Net Interest Income", "InterestExpense"
            ], col_idx)
            if interest_exp is not None:
                interest_exp = abs(interest_exp)

            # Balance sheet (same year column if available, else nearest)
            bal_idx = _match_col_index(bal, yr) if bal is not None and not bal.empty else col_idx
            total_assets    = _df_value(bal, ["Total Assets", "TotalAssets"], bal_idx)
            total_equity    = _df_value(bal, [
                "Total Equity Gross Minority Interest", "Stockholders Equity",
                "Total Stockholders Equity", "TotalEquity"
            ], bal_idx)
            total_debt      = _df_value(bal, ["Total Debt", "Long Term Debt", "TotalDebt"], bal_idx)
            current_assets  = _df_value(bal, ["Current Assets", "Total Current Assets"], bal_idx)
            current_liab    = _df_value(bal, ["Current Liabilities", "Total Current Liabilities"], bal_idx)
            net_ppe         = _df_value(bal, ["Net PPE", "Net Property Plant And Equipment", "NetPPE"], bal_idx)
            bvps            = None  # yfinance "Book Value" = total equity, niet per aandeel; fallback hieronder
            shares          = _df_value(bal, [
                "Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"
            ], bal_idx) or _safe_get(info, "sharesOutstanding")
            inventory       = _df_value(bal, ["Inventory", "Inventories"], bal_idx)
            cash            = _df_value(bal, ["Cash And Cash Equivalents", "Cash", "CashAndCashEquivalents"], bal_idx)
            if total_debt is not None and cash is not None:
                net_debt = total_debt - cash
                net_cash = -net_debt
            else:
                net_cash = None

            # Cash flow
            cf_idx = _match_col_index(cf, yr) if cf is not None and not cf.empty else col_idx
            op_cf   = _df_value(cf, ["Operating Cash Flow", "Cash From Operating Activities"], cf_idx)
            capex   = _df_value(cf, ["Capital Expenditure", "Capex", "PurchaseOfPPE"], cf_idx)
            if capex is not None:
                capex = -abs(capex)  # capex stored as negative (outflow)
            fcf_direct = _df_value(cf, ["Free Cash Flow", "FreeCashFlow"], cf_idx)
            fcf = fcf_direct if fcf_direct is not None else (
                (op_cf + capex) if op_cf is not None and capex is not None else None
            )

            # Derived ratios
            roe = (net_income / total_equity) if net_income and total_equity else None
            if bvps is None and total_equity and shares and shares > 0:
                bvps = total_equity / shares

            annual_row = {
                "fiscal_year":       yr,
                "revenue":           revenue,
                "ebit":              ebit,
                "ebitda":            ebitda,
                "net_income":        net_income,
                "eps_diluted":       eps_diluted,
                "operating_cf":      op_cf,
                "capex":             capex,
                "fcf":               fcf,
                "total_assets":      total_assets,
                "total_equity":      total_equity,
                "total_debt":        total_debt,
                "current_assets":    current_assets,
                "current_liabilities": current_liab,
                "net_ppe":           net_ppe,
                "book_value_ps":     bvps,
                "roe":               roe,
                "gross_profit":      gross_profit,
                "interest_expense":  interest_exp,
                "shares_outstanding": shares,
                "net_cash":          net_cash,
                "inventory":         inventory,
            }
            result["annual"].append(annual_row)
            years_found += 1

    if years_found < 3:
        warnings.append(
            f"Only {years_found} year(s) of financial history available — "
            "quality score and normalization may be limited."
        )
    if years_found == 0:
        warnings.append("No annual financial data found — manual entry required for valuation.")

    return result


# ---------------------------------------------------------------------------
# TTM (Trailing Twelve Months) — som van laatste 4 kwartalen
# ---------------------------------------------------------------------------

def _fetch_ttm_row(t: "yf.Ticker", info: dict) -> dict | None:
    """
    Bouw een TTM-rij op uit de laatste 4 kwartaalrapporten.
    Flow-items (omzet, EBIT, FCF, …) = som van 4 kwartalen.
    Balansitems (schuld, eigen vermogen, aandelen) = meest recente kwartaal.
    Geeft None terug als er minder dan 4 kwartalen beschikbaar zijn.
    """
    try:
        q_inc = t.quarterly_income_stmt
        q_bal = t.quarterly_balance_sheet
        q_cf  = t.quarterly_cashflow
    except Exception as e:
        log.warning("Quarterly statements unavailable: %s", e)
        return None

    if q_inc is None or q_inc.empty or len(q_inc.columns) < 4:
        return None

    def q_sum(df, keys):
        """Som van de laatste 4 kwartalen voor een reeks kolomnamen."""
        if df is None or df.empty:
            return None
        total = 0.0
        found = False
        for col_idx in range(4):
            if col_idx >= len(df.columns):
                break
            v = _df_value(df, keys, col_idx)
            if v is not None:
                total += v
                found = True
        return total if found else None

    def q_last(df, keys):
        """Meest recente kwartaal voor balansitems."""
        if df is None or df.empty:
            return None
        return _df_value(df, keys, 0)

    revenue     = q_sum(q_inc, ["Total Revenue", "Revenue"])
    ebit        = q_sum(q_inc, ["EBIT", "Operating Income", "Ebit"])
    ebitda_raw  = q_sum(q_inc, ["EBITDA", "Ebitda"])
    net_income  = q_sum(q_inc, ["Net Income", "Net Income Common Stockholders", "NetIncome"])
    eps_diluted = q_sum(q_inc, ["Diluted EPS", "Basic EPS"])
    gross_profit = q_sum(q_inc, ["Gross Profit", "GrossProfit"])
    interest_exp = q_sum(q_inc, [
        "Interest Expense", "Interest Expense Non Operating",
        "Net Interest Income", "InterestExpense",
    ])
    if interest_exp is not None:
        interest_exp = abs(interest_exp)

    da = q_sum(q_inc, ["Reconciled Depreciation", "Depreciation And Amortization",
                        "Depreciation Amortization Depletion"])
    ebitda = ebitda_raw if ebitda_raw is not None else (
        (ebit + abs(da)) if ebit is not None and da is not None else None
    )

    op_cf  = q_sum(q_cf, ["Operating Cash Flow", "Cash From Operating Activities"])
    capex  = q_sum(q_cf, ["Capital Expenditure", "Capex", "PurchaseOfPPE"])
    if capex is not None:
        capex = -abs(capex)
    fcf_direct = q_sum(q_cf, ["Free Cash Flow", "FreeCashFlow"])
    fcf = fcf_direct if fcf_direct is not None else (
        (op_cf + capex) if op_cf is not None and capex is not None else None
    )

    total_equity   = q_last(q_bal, ["Total Equity Gross Minority Interest", "Stockholders Equity",
                                    "Total Stockholders Equity", "TotalEquity"])
    total_debt     = q_last(q_bal, ["Total Debt", "Long Term Debt", "TotalDebt"])
    total_assets   = q_last(q_bal, ["Total Assets", "TotalAssets"])
    current_assets = q_last(q_bal, ["Current Assets", "Total Current Assets"])
    current_liab   = q_last(q_bal, ["Current Liabilities", "Total Current Liabilities"])
    net_ppe        = q_last(q_bal, ["Net PPE", "Net Property Plant And Equipment", "NetPPE"])
    inventory      = q_last(q_bal, ["Inventory", "Inventories"])
    cash           = q_last(q_bal, ["Cash And Cash Equivalents", "Cash", "CashAndCashEquivalents"])
    shares         = q_last(q_bal, ["Ordinary Shares Number", "Share Issued",
                                    "Common Stock Shares Outstanding"]) \
                     or _safe_get(info, "sharesOutstanding")

    net_cash = None
    if total_debt is not None and cash is not None:
        net_cash = cash - total_debt

    roe  = (net_income / total_equity) if net_income and total_equity and total_equity > 0 else None
    bvps = (total_equity / shares)     if total_equity and shares and shares > 0 else None

    return {
        "fiscal_year":         None,   # TTM heeft geen vast fiscaal jaar
        "revenue":             revenue,
        "ebit":                ebit,
        "ebitda":              ebitda,
        "net_income":          net_income,
        "eps_diluted":         eps_diluted,
        "operating_cf":        op_cf,
        "capex":               capex,
        "fcf":                 fcf,
        "total_assets":        total_assets,
        "total_equity":        total_equity,
        "total_debt":          total_debt,
        "current_assets":      current_assets,
        "current_liabilities": current_liab,
        "net_ppe":             net_ppe,
        "book_value_ps":       bvps,
        "roe":                 roe,
        "gross_profit":        gross_profit,
        "interest_expense":    interest_exp,
        "shares_outstanding":  shares,
        "net_cash":            net_cash,
        "inventory":           inventory,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_market(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith(".WA"):  return "PL"
    if t.endswith(".ST"):  return "SE"
    if t.endswith(".BR"):  return "BE"
    if t.endswith(".AS"):  return "NL"
    if t.endswith(".OL"):  return "NO"
    if t.endswith(".DE"):  return "DE"
    if t.endswith(".FI"):  return "FI"
    if t.endswith(".L"):   return "UK"
    return "US"


def _match_col_index(df: pd.DataFrame, year: int) -> int:
    """Find column index in df whose year matches `year`, or 0 as fallback."""
    if df is None or df.empty:
        return 0
    for i, col in enumerate(df.columns):
        if _col_year(col) == year:
            return i
    return 0


def _calc_ebitda(inc: pd.DataFrame, bal: pd.DataFrame, col_idx: int) -> float | None:
    """Approximate EBITDA = EBIT + D&A when EBITDA row is missing."""
    ebit = _df_value(inc, ["EBIT", "Operating Income"], col_idx)
    da   = _df_value(inc, ["Reconciled Depreciation", "Depreciation And Amortization",
                            "Depreciation Amortization Depletion"], col_idx)
    if ebit is not None and da is not None:
        return ebit + abs(da)
    return None


# ---------------------------------------------------------------------------
# Persist fetched data to DB
# ---------------------------------------------------------------------------

def fetch_and_store(ticker: str) -> list[str]:
    """
    Full pipeline: fetch from Yahoo Finance → apply overrides → persist to DB.
    Returns list of warning strings.
    """
    log.info("Fetching %s", ticker)
    data = fetch_ticker(ticker)
    warnings = data.get("warnings", [])

    # Upsert stock meta
    meta = data.get("meta", {})
    if meta:
        db.upsert_stock(
            ticker,
            name=meta.get("name"),
            sector=meta.get("sector"),
            market=meta.get("market"),
            currency=meta.get("currency"),
            description=meta.get("description"),
            active=1,
        )

    # Upsert market data
    mkt = data.get("market", {})
    if mkt and mkt.get("price") is not None:
        db.upsert_market_data(ticker, **mkt)

    # Upsert annual financials
    overrides = db.get_overrides(ticker)
    today = datetime.utcnow().date().isoformat()

    for row in data.get("annual", []):
        yr = row["fiscal_year"]
        # Apply overrides
        applied = dict(row)
        for field, value in {
            k: v for (f, y), v in overrides.items()
            if (k := f) and (y == yr or y is None)
        }.items():
            if field in applied:
                applied[field] = value
                log.debug("Override applied: %s %s %s = %s", ticker, yr, field, value)
        applied["fetched_date"] = today
        db.upsert_financials(ticker, "annual", yr, **{k: v for k, v in applied.items() if k != "fiscal_year"})

    # Fetch and store TTM (Trailing Twelve Months) row
    # fiscal_year=0 is used as sentinel (NULL doesn't work with SQLite UNIQUE constraints)
    try:
        t_obj = yf.Ticker(ticker)
        ttm_row = _fetch_ttm_row(t_obj, data.get("meta", {}))
        if ttm_row:
            ttm_stored = dict(ttm_row)
            ttm_stored["fetched_date"] = today
            db.upsert_financials(ticker, "ttm", 0, **{k: v for k, v in ttm_stored.items() if k != "fiscal_year"})
    except Exception as e:
        log.warning("TTM fetch failed for %s: %s", ticker, e)

    # Compute and store historical per-year multiples
    _store_historical_multiples(ticker, data)

    return warnings


def _store_historical_multiples(ticker: str, data: dict) -> None:
    """Calculate and store price/earnings multiples per historical year."""
    mkt = data.get("market", {})
    price = mkt.get("price")
    if not price:
        return

    # For proper historical multiples we'd need historical prices — yfinance provides these
    # As a pragmatic approximation: store current TTM multiples for the latest year,
    # and use historical stock price / historical EPS for prior years.
    try:
        t = yf.Ticker(ticker)
        hist_price = t.history(period="5y", interval="1mo")
    except Exception:
        return

    for row in data.get("annual", []):
        yr = row["fiscal_year"]
        eps = row.get("eps_diluted")
        bvps = row.get("book_value_ps")
        fcf = row.get("fcf")
        ebitda = row.get("ebitda")
        total_assets = row.get("total_assets")
        total_debt = row.get("total_debt")
        net_cash = row.get("net_cash")

        # Find approximate year-end price
        yr_price = _historical_year_end_price(hist_price, yr)
        if yr_price is None:
            continue

        shares = row.get("shares_outstanding")
        ev = None
        if yr_price and shares and total_debt is not None:
            market_cap_hist = yr_price * shares
            cash = (-net_cash + total_debt) if net_cash is not None else 0
            ev = market_cap_hist + total_debt - (total_debt - (net_cash or 0) * -1)
            # Simplified: EV ≈ market_cap + net_debt
            net_debt = (total_debt or 0) - max(0, -(net_cash or 0))
            ev = market_cap_hist + net_debt

        pe   = (yr_price / eps)     if eps   and eps > 0   else None
        pb   = (yr_price / bvps)    if bvps  and bvps > 0  else None
        ev_ebitda = (ev / ebitda)   if ev    and ebitda and ebitda > 0 else None
        ev_fcf    = (ev / fcf) if ev and fcf and fcf > 0 else None

        db.upsert_historical_multiples(ticker, yr, pe_ratio=pe, ev_ebitda=ev_ebitda, pb_ratio=pb, ev_fcf=ev_fcf)


def _historical_year_end_price(hist: pd.DataFrame, year: int) -> float | None:
    """Find the closing price closest to Dec 31 of a given year."""
    if hist is None or hist.empty:
        return None
    try:
        yr_data = hist[hist.index.year == year]
        if yr_data.empty:
            return None
        return float(yr_data["Close"].iloc[-1])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Bulk operations
# ---------------------------------------------------------------------------

def fetch_all_tickers(tickers: list[str], progress_cb=None) -> dict[str, list[str]]:
    """
    Fetch all tickers, updating progress via callback(ticker, idx, total).
    Returns {ticker: [warnings]}.
    """
    # Refresh FX rates once before processing
    refresh_exchange_rates()

    results = {}
    total = len(tickers)
    for idx, ticker in enumerate(tickers):
        if progress_cb:
            progress_cb(ticker, idx, total)
        try:
            warnings = fetch_and_store(ticker)
            results[ticker] = warnings
        except Exception as e:
            log.exception("Unexpected error fetching %s", ticker)
            results[ticker] = [str(e)]
    return results
