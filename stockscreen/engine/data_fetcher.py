"""
Data fetcher — wraps yfinance.

Responsibilities:
  • Fetch raw financial statements (income, balance sheet, cash flow)
  • Fetch current market data (price, market cap, EV, trailing ratios)
  • Fetch historical per-year multiples for the last 5 years
  • Apply manual overrides from the DB before returning data
  • Return structured dicts ready for the engine pipeline

Alle bedragen blijven in de native currency van het aandeel — er vindt geen
valutaconversie plaats. Vergelijking tussen aandelen gebeurt via relatieve
maatstaven (margin of safety, P/E) die valuta-onafhankelijk zijn.
"""

import logging
import time
from datetime import datetime, date
from typing import Any, Callable

import numpy as np
import pandas as pd
import yfinance as yf

from . import db, data_quality

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry helper — yfinance is flaky onder load (rate limits, timeouts, 5xx)
# ---------------------------------------------------------------------------

def _yf_retry(
    fn: Callable,
    *args,
    attempts: int = 3,
    initial_delay: float = 2.0,
    backoff: float = 2.0,
    label: str = "yfinance",
    **kwargs,
):
    """
    Voer een yfinance-oproep uit met exponential backoff.
    Rate-limit fouten (HTTP 429 / 'Too Many Requests') krijgen een langere wachttijd
    omdat Yahoo pas na minimaal een paar seconden weer responseert.
    Gooit bij falen de laatste exception opnieuw.
    """
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt >= attempts - 1:
                break
            msg = str(e).lower()
            is_rate_limit = any(s in msg for s in ("429", "too many", "rate limit", "ratelimit"))
            delay = initial_delay * (backoff ** attempt)
            if is_rate_limit:
                delay *= 3.0
            log.warning("%s mislukt (poging %d/%d): %s — retry over %.1fs",
                        label, attempt + 1, attempts, e, delay)
            time.sleep(delay)
    assert last_err is not None
    raise last_err

# Map market suffix → native currency
MARKET_CURRENCIES = {
    ".WA": "PLN",   # Warsaw
    ".ST": "SEK",   # Stockholm
    ".BR": "EUR",   # Brussels
    ".AS": "EUR",   # Amsterdam
    ".OL": "NOK",   # Oslo
    ".DE": "EUR",   # Frankfurt
    ".FI": "EUR",   # Helsinki
    ".L":  "GBP",   # London
}


# Money-velden die mee-schalen bij FX-conversie. Alles in "financial currency" bij
# dual-currency tickers (bv. .OL aandeel met USD-rapportage) wordt hiermee omgezet
# naar trading currency zodat per-share-metrics consistent zijn met de koers.
_MONEY_FIELDS_PER_ROW = (
    "revenue", "ebit", "ebitda", "net_income", "eps_diluted",
    "operating_cf", "capex", "fcf",
    "total_assets", "total_equity", "total_debt",
    "current_assets", "current_liabilities", "net_ppe",
    "book_value_ps", "gross_profit", "interest_expense",
    "net_cash", "inventory",
)


# FX-cache per proces (yfinance {FROM}{TO}=X tickers zijn rate-limited).
_FX_CACHE: dict[tuple[str, str], float | None] = {}


def _fx_rate(from_ccy: str, to_ccy: str) -> float | None:
    """
    Haal FX-koers op via yfinance `{FROM}{TO}=X`. In-memory cached per proces.
    Geeft None terug als yfinance geen koers levert — caller moet beslissen of
    data onveranderd blijft + waarschuwing, of niet opgeslagen wordt.
    """
    if not from_ccy or not to_ccy:
        return None
    # Normaliseer pence-varianten (hoort eigenlijk nooit tot hier te komen)
    if from_ccy in ("GBp", "GBX"):
        from_ccy = "GBP"
    if to_ccy in ("GBp", "GBX"):
        to_ccy = "GBP"
    if from_ccy == to_ccy:
        return 1.0
    key = (from_ccy, to_ccy)
    if key in _FX_CACHE:
        return _FX_CACHE[key]
    pair = f"{from_ccy}{to_ccy}=X"
    rate: float | None = None
    try:
        info = _yf_retry(lambda: yf.Ticker(pair).info or {}, label=f"fx {pair}", attempts=2)
        rate = _safe_get(info, "regularMarketPrice", "previousClose")
    except Exception as e:
        log.warning("FX %s→%s fetch faalde: %s", from_ccy, to_ccy, e)
    if rate is None:
        # Fallback: probeer de omgekeerde koers en inverteer
        inv_pair = f"{to_ccy}{from_ccy}=X"
        try:
            info_inv = _yf_retry(lambda: yf.Ticker(inv_pair).info or {}, label=f"fx {inv_pair}", attempts=2)
            inv_rate = _safe_get(info_inv, "regularMarketPrice", "previousClose")
            if inv_rate and inv_rate > 0:
                rate = 1.0 / inv_rate
        except Exception:
            pass
    _FX_CACHE[key] = rate
    return rate


def _apply_fx_to_row(row: dict, fx: float) -> None:
    """Vermenigvuldig alle money-velden in `row` met `fx`. In-place mutatie."""
    for f in _MONEY_FIELDS_PER_ROW:
        v = row.get(f)
        if v is not None:
            row[f] = v * fx


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
        info = _yf_retry(lambda: t.info or {}, label=f"info {ticker}")
    except Exception as e:
        warnings.append(f"Yahoo Finance niet bereikbaar na retries: {e}")
        return result

    if not info:
        warnings.append("Geen data terug van Yahoo Finance — ticker mogelijk ongeldig.")
        return result

    # ---- Meta ---------------------------------------------------------------
    currency = info.get("currency") or infer_currency(ticker)
    # financialCurrency = valuta van de jaarrekening (kan afwijken van handelvaluta bij ADRs)
    # Bijv. TM handelt in USD maar rapporteert in JPY; WIT/IBN rapporteren in INR
    financial_currency = info.get("financialCurrency") or currency
    result["meta"] = {
        "name":              info.get("longName") or info.get("shortName") or ticker,
        "sector":            info.get("sector") or "Unknown",
        "currency":          currency,
        "financial_currency": financial_currency,
        "market":            _detect_market(ticker),
        "description":       info.get("longBusinessSummary"),
        # quoteType laat de screener ETFs/funds/indices etc. vroeg afvangen —
        # voor die instrumenten werkt de fundamentele FV-pipeline niet.
        "quote_type":        info.get("quoteType") or info.get("typeDisp"),
    }

    # ---- Current market data ------------------------------------------------
    price_raw = _safe_get(info, "currentPrice", "regularMarketPrice", "previousClose")
    market_cap_raw = _safe_get(info, "marketCap")
    ev_raw = _safe_get(info, "enterpriseValue")
    analyst_target_raw = _safe_get(info, "targetMeanPrice")

    # GBp/GBX (pence) is een Yahoo-quirk voor .L tickers: prijs wordt in pence
    # gegeven terwijl marketCap en financials in pounds (GBP) staan. Zonder
    # conversie is price 100× te hoog tov fair value → misleidende SELL signals
    # op elke UK-ticker. Normaliseer alles naar GBP.
    if currency in ("GBp", "GBX"):
        if price_raw is not None:
            price_raw = price_raw / 100.0
        if analyst_target_raw is not None:
            analyst_target_raw = analyst_target_raw / 100.0
        # market_cap en enterprise_value blijven ongewijzigd — Yahoo levert die al in GBP
        currency = "GBP"
        result["meta"]["currency"] = "GBP"
        # Vlag voor _store_historical_multiples: historische prijzen uit yf.history()
        # komen óók in pence — die moeten ook door 100.
        result["meta"]["_price_pence_scale"] = True

    result["market"] = {
        "price":                price_raw,
        "market_cap":           market_cap_raw,
        "enterprise_value":     ev_raw,
        "pe_ttm":               _safe_get(info, "trailingPE"),
        "ev_ebitda_ttm":        _safe_get(info, "enterpriseToEbitda"),
        "pb_ratio":             _safe_get(info, "priceToBook"),
        "last_updated":         datetime.utcnow().isoformat(),
        "analyst_target_raw":   analyst_target_raw,
        "analyst_consensus":    info.get("recommendationKey"),
        "analyst_n":            info.get("numberOfAnalystOpinions"),
    }

    if not price_raw:
        warnings.append("Current price unavailable — check ticker symbol.")

    # ---- Annual financial statements ----------------------------------------
    try:
        inc = _yf_retry(lambda: t.income_stmt, label=f"inc {ticker}")
        bal = _yf_retry(lambda: t.balance_sheet, label=f"bal {ticker}")
        cf  = _yf_retry(lambda: t.cashflow, label=f"cf {ticker}")
    except Exception as e:
        warnings.append(f"Jaarrekening onbereikbaar na retries: {e}")
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
            bvps            = None
            inventory       = _df_value(bal, ["Inventory", "Inventories"], bal_idx)
            cash            = _df_value(bal, ["Cash And Cash Equivalents", "Cash", "CashAndCashEquivalents"], bal_idx)

            # Shares: bij A/B-aandelenstructuren (bijv. BRK-B) geeft de balans soms
            # het A-equivalent aantal terug, terwijl de koers die van het B-aandeel is.
            # Ook bij Nordic tickers blijkt de balance-sheet-waarde regelmatig ~10×
            # af te wijken van info.sharesOutstanding — en dan is info correct
            # (factor drukt FV per aandeel 10× omlaag → misleidende STRONG BUY).
            # Drempel factor-2: bij divergentie wint info.sharesOutstanding.
            bal_shares  = _df_value(bal, [
                "Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"
            ], bal_idx)
            info_shares = _safe_get(info, "sharesOutstanding")
            if bal_shares and info_shares:
                ratio = max(bal_shares, info_shares) / min(bal_shares, info_shares)
                if ratio > 2:
                    log.warning(
                        "%s FY%s shares-mismatch: balance=%s vs info=%s (factor %.2fx) — info gekozen",
                        ticker, yr, bal_shares, info_shares, ratio
                    )
                    shares = info_shares
                else:
                    shares = bal_shares
            else:
                shares = bal_shares or info_shares

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
            bvps = None
            if total_equity and shares and shares > 0:
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

    # Valuta-consistentie: als financiële cijfers in andere ccy staan dan de
    # trading currency (bv. AUTO.OL rapporteert in USD maar handelt in NOK),
    # converteer alle money-velden naar trading ccy zodat per-share metrics
    # consistent zijn met de koers. Shares blijven een telling (geen conversie).
    if financial_currency and currency and financial_currency != currency:
        fx = _fx_rate(financial_currency, currency)
        if fx and fx > 0:
            for row in result["annual"]:
                _apply_fx_to_row(row, fx)
            result["meta"]["fx_rate_applied"] = round(fx, 6)
            result["meta"]["fx_from"] = financial_currency
            result["meta"]["fx_to"] = currency
            log.info("FX %s→%s × %.4f toegepast op %s", financial_currency, currency, fx, ticker)
            warnings.append(
                f"Financials geconverteerd van {financial_currency} naar {currency} (×{fx:.4f})."
            )
        else:
            warnings.append(
                f"FX-koers {financial_currency}→{currency} niet beschikbaar — "
                f"financials niet geconverteerd; FV mogelijk onbetrouwbaar."
            )

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
        q_inc = _yf_retry(lambda: t.quarterly_income_stmt, label="q_inc")
        q_bal = _yf_retry(lambda: t.quarterly_balance_sheet, label="q_bal")
        q_cf  = _yf_retry(lambda: t.quarterly_cashflow, label="q_cf")
    except Exception as e:
        log.warning("Kwartaalcijfers onbereikbaar na retries: %s", e)
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
    bal_shares_ttm  = q_last(q_bal, ["Ordinary Shares Number", "Share Issued",
                                      "Common Stock Shares Outstanding"])
    info_shares_ttm = _safe_get(info, "sharesOutstanding")
    if bal_shares_ttm and info_shares_ttm:
        _r = max(bal_shares_ttm, info_shares_ttm) / min(bal_shares_ttm, info_shares_ttm)
        shares = info_shares_ttm if _r > 2 else bal_shares_ttm
    else:
        shares = bal_shares_ttm or info_shares_ttm

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
            financial_currency=meta.get("financial_currency"),
            description=meta.get("description"),
            quote_type=meta.get("quote_type"),
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
        applied = dict(row)
        # overrides: {(field_name, fiscal_year): {"value": v, "note": n}}
        for (field, ov_yr), entry in overrides.items():
            if (ov_yr == yr or ov_yr is None) and field in applied:
                applied[field] = entry["value"]
                log.debug("Override toegepast: %s FY%s %s = %s", ticker, yr, field, entry["value"])
        applied["fetched_date"] = today
        db.upsert_financials(ticker, "annual", yr, **{k: v for k, v in applied.items() if k != "fiscal_year"})

    # Fetch and store TTM (Trailing Twelve Months) row
    # fiscal_year=0 is used as sentinel (NULL doesn't work with SQLite UNIQUE constraints)
    try:
        t_obj = yf.Ticker(ticker)
        ttm_row = _fetch_ttm_row(t_obj, data.get("meta", {}))
        if ttm_row:
            # Pas dezelfde FX-conversie toe die op annual rows is toegepast,
            # zodat de "year 0" TTM-rij consistent in trading currency staat.
            fx_applied = meta.get("fx_rate_applied") if meta else None
            if fx_applied:
                _apply_fx_to_row(ttm_row, fx_applied)
            ttm_stored = dict(ttm_row)
            ttm_stored["fetched_date"] = today
            db.upsert_financials(ticker, "ttm", 0, **{k: v for k, v in ttm_stored.items() if k != "fiscal_year"})
    except Exception as e:
        log.warning("TTM fetch failed for %s: %s", ticker, e)

    # Compute and store historical per-year multiples
    _store_historical_multiples(ticker, data)

    # Evalueer data-kwaliteit (draait ook bij gefaalde fetch zodat we altijd
    # een record hebben — zie data_quality.evaluate voor de missing-path).
    try:
        prev_dq = db.get_data_quality(ticker) or {}
        prev_fails = prev_dq.get("consecutive_failures") or 0

        annual_persisted = db.get_financials(ticker, "annual")
        market_persisted = db.get_market_data(ticker)
        stock_persisted  = db.get_stock(ticker)

        fetch_succeeded = bool(data.get("meta")) and bool(
            data.get("annual") or data.get("market", {}).get("price")
        )

        dq = data_quality.evaluate(
            ticker,
            annual_persisted,
            market_persisted,
            stock_persisted,
            fetch_success=fetch_succeeded,
            prev_consecutive_failures=prev_fails,
            fetched_date=today,
        )
        db.upsert_data_quality(ticker, **dq)
    except Exception:
        log.exception("Data-quality evaluatie mislukt voor %s", ticker)

    return warnings


def _store_historical_multiples(ticker: str, data: dict) -> None:
    """Calculate and store price/earnings multiples per historical year."""
    mkt = data.get("market", {})
    meta = data.get("meta", {}) or {}
    price = mkt.get("price")
    if not price:
        return

    # For proper historical multiples we'd need historical prices — yfinance provides these
    # As a pragmatic approximation: store current TTM multiples for the latest year,
    # and use historical stock price / historical EPS for prior years.
    try:
        t = yf.Ticker(ticker)
        hist_price = _yf_retry(lambda: t.history(period="5y", interval="1mo"),
                                label=f"history {ticker}")
    except Exception:
        return

    # GBp-tickers: historische prijzen uit yf.history() komen in pence; annual
    # cijfers staan al in GBP → price moet ook naar GBP om multiples kloppend
    # te maken.
    pence_scale = bool(meta.get("_price_pence_scale"))

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
        if pence_scale:
            yr_price = yr_price / 100.0

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

def fetch_market_only(ticker: str) -> None:
    """
    Lichte fetch: alleen koers + marktdata bijwerken, geen financiële statements.
    Veel sneller dan fetch_and_store — geschikt voor dagelijks draaien.
    """
    try:
        info = _yf_retry(lambda: yf.Ticker(ticker).info or {}, label=f"light {ticker}")
    except Exception as e:
        log.warning("Light fetch definitief mislukt voor %s: %s", ticker, e)
        return

    if not info:
        return

    price_raw = _safe_get(info, "currentPrice", "regularMarketPrice", "previousClose")
    if not price_raw:
        return

    db.upsert_market_data(ticker,
        price                = price_raw,
        market_cap           = _safe_get(info, "marketCap"),
        enterprise_value     = _safe_get(info, "enterpriseValue"),
        pe_ttm               = _safe_get(info, "trailingPE"),
        ev_ebitda_ttm        = _safe_get(info, "enterpriseToEbitda"),
        pb_ratio             = _safe_get(info, "priceToBook"),
        last_updated         = datetime.utcnow().isoformat(),
        analyst_target_raw   = _safe_get(info, "targetMeanPrice"),
        analyst_consensus    = info.get("recommendationKey"),
        analyst_n            = info.get("numberOfAnalystOpinions"),
    )
    log.debug("Light refresh OK: %s @ %s", ticker, price_raw)


def fetch_all_tickers(
    tickers: list[str],
    progress_cb=None,
    max_workers: int = 3,
    jitter_seconds: float = 0.8,
) -> dict[str, list[str]]:
    """
    Fetch all tickers concurrent met rate-limiting:
      • max_workers threads parallel (default 3 — conservatief voor Yahoo)
      • Per worker een willekeurige sleep tussen tickers (jitter_seconds)
      • Retry-logic zit in de onderliggende yfinance-calls via _yf_retry

    Returns {ticker: [warnings]}.
    """
    import concurrent.futures
    import random
    from threading import Lock

    results: dict[str, list[str]] = {}
    total = len(tickers)
    idx_lock = Lock()
    current_idx = 0

    def _fetch_worker(ticker: str):
        nonlocal current_idx
        # Willekeurige jitter om burst-rate te dempen (alle workers kloppen niet
        # tegelijk op Yahoo aan). Alleen relevant bij bulk-refresh.
        if jitter_seconds > 0:
            time.sleep(random.uniform(0, jitter_seconds))

        try:
            warnings = fetch_and_store(ticker)
        except Exception as e:
            log.exception("Onverwachte fout bij fetchen %s", ticker)
            warnings = [str(e)]

        with idx_lock:
            current_idx += 1
            if progress_cb:
                progress_cb(ticker, current_idx - 1, total)

        return ticker, warnings

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(_fetch_worker, t): t for t in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            t = future_to_ticker[future]
            try:
                t_res, warnings_res = future.result()
                results[t_res] = warnings_res
            except Exception as e:
                log.exception("Worker crashed voor %s", t)
                results[t] = [f"Worker crashed: {e}"]

    return results
