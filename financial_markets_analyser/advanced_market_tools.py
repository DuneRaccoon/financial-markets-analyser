
# ------------------------------------------------------------------------------------------------------
# Advanced Market Analysis Tools
# ------------------------------------------------------------------------------------------------------

import json
import logging
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np


# Import from server (will be set when this module is imported by server.py)
logger = None
make_request = None
make_sync_request = None
FMP_API_KEY = None
FMP_BASE_URL = None
ALPHA_VANTAGE_API_KEY = None
ALPHA_VANTAGE_BASE_URL = None
get_current_stock_price = None
get_historical_stock_prices = None
get_historical_crypto_prices = None


# ------------------------------------------------------------------------------------------------------
# Market Indices and Sector Performance
# ------------------------------------------------------------------------------------------------------

async def get_market_indices() -> str:
    """Get current prices and performance of major market indices.

    Returns data for S&P 500, Dow Jones, NASDAQ, Russell 2000, and international indices.
    """
    logger.info("Getting major market indices")

    try:
        indices = {
            "S&P 500": "^GSPC",
            "Dow Jones": "^DJI",
            "NASDAQ": "^IXIC",
            "Russell 2000": "^RUT",
            "VIX": "^VIX",
            "FTSE 100": "^FTSE",
            "DAX": "^GDAXI",
            "Nikkei 225": "^N225",
            "Hang Seng": "^HSI",
            "Shanghai Composite": "000001.SS",
        }

        results = {}

        for name, ticker in indices.items():
            try:
                index = yf.Ticker(ticker)
                info = index.info
                hist = index.history(period="5d")

                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price

                    # Calculate changes
                    day_change = current_price - prev_close
                    day_change_pct = (day_change / prev_close) * 100 if prev_close else 0

                    # Get 52-week high/low
                    hist_year = index.history(period="1y")
                    year_high = hist_year['High'].max() if not hist_year.empty else current_price
                    year_low = hist_year['Low'].min() if not hist_year.empty else current_price

                    results[name] = {
                        "ticker": ticker,
                        "price": current_price,
                        "change": day_change,
                        "change_percent": day_change_pct,
                        "previous_close": prev_close,
                        "52_week_high": year_high,
                        "52_week_low": year_low,
                        "distance_from_high": ((current_price / year_high) - 1) * 100,
                        "distance_from_low": ((current_price / year_low) - 1) * 100,
                    }
            except Exception as e:
                logger.warning(f"Error fetching {name}: {str(e)}")
                continue

        # Market breadth analysis
        market_summary = {
            "indices": results,
            "timestamp": datetime.now().isoformat(),
        }

        # Calculate market trend
        sp500_trend = "Bullish" if results.get("S&P 500", {}).get("change_percent", 0) > 0 else "Bearish"
        vix_level = results.get("VIX", {}).get("price", 0)

        market_sentiment = "Low"
        if vix_level > 30:
            market_sentiment = "Extreme Fear"
        elif vix_level > 20:
            market_sentiment = "High"
        elif vix_level > 15:
            market_sentiment = "Moderate"

        market_summary["analysis"] = {
            "sp500_trend": sp500_trend,
            "vix_level": vix_level,
            "volatility_sentiment": market_sentiment,
        }

        logger.info("Successfully retrieved market indices")
        return json.dumps(market_summary, indent=2)

    except Exception as e:
        logger.error(f"Error getting market indices: {str(e)}")
        return json.dumps({"Error": f"Failed to get market indices: {str(e)}"}, indent=2)


async def get_sector_performance(period: str = "1d") -> str:
    """Get performance of major market sectors.

    Args:
        period: Time period for performance (1d, 5d, 1mo, 3mo, 6mo, 1y, ytd)
    """
    logger.info(f"Getting sector performance for period: {period}")

    try:
        # Sector ETFs as proxies
        sectors = {
            "Technology": "XLK",
            "Healthcare": "XLV",
            "Financials": "XLF",
            "Consumer Discretionary": "XLY",
            "Industrials": "XLI",
            "Energy": "XLE",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Materials": "XLB",
            "Consumer Staples": "XLP",
            "Communication Services": "XLC",
        }

        results = {}

        for sector, etf in sectors.items():
            try:
                ticker = yf.Ticker(etf)
                hist = ticker.history(period="1y")  # Get full year for calculations

                if not hist.empty:
                    # Get the appropriate time window
                    if period == "1d":
                        if len(hist) >= 2:
                            start_price = hist['Close'].iloc[-2]
                            end_price = hist['Close'].iloc[-1]
                        else:
                            continue
                    elif period == "5d":
                        if len(hist) >= 5:
                            start_price = hist['Close'].iloc[-6]
                            end_price = hist['Close'].iloc[-1]
                        else:
                            continue
                    elif period == "1mo":
                        start_price = hist['Close'].iloc[-22] if len(hist) >= 22 else hist['Close'].iloc[0]
                        end_price = hist['Close'].iloc[-1]
                    elif period == "3mo":
                        start_price = hist['Close'].iloc[-66] if len(hist) >= 66 else hist['Close'].iloc[0]
                        end_price = hist['Close'].iloc[-1]
                    elif period == "6mo":
                        start_price = hist['Close'].iloc[-132] if len(hist) >= 132 else hist['Close'].iloc[0]
                        end_price = hist['Close'].iloc[-1]
                    elif period == "1y":
                        start_price = hist['Close'].iloc[0]
                        end_price = hist['Close'].iloc[-1]
                    elif period == "ytd":
                        # Find the first trading day of the year
                        current_year = datetime.now().year
                        ytd_start = hist[hist.index.year == current_year].iloc[0]['Close'] if len(hist[hist.index.year == current_year]) > 0 else hist['Close'].iloc[0]
                        start_price = ytd_start
                        end_price = hist['Close'].iloc[-1]
                    else:
                        start_price = hist['Close'].iloc[0]
                        end_price = hist['Close'].iloc[-1]

                    # Calculate performance
                    performance = ((end_price - start_price) / start_price) * 100

                    # Calculate volatility
                    returns = hist['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252) * 100  # Annualized

                    results[sector] = {
                        "etf": etf,
                        "performance": performance,
                        "volatility": volatility,
                        "current_price": end_price,
                    }
            except Exception as e:
                logger.warning(f"Error fetching {sector}: {str(e)}")
                continue

        # Sort by performance
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1]["performance"], reverse=True))

        # Identify leaders and laggards
        sector_list = list(sorted_results.items())
        leaders = sector_list[:3] if len(sector_list) >= 3 else sector_list
        laggards = sector_list[-3:] if len(sector_list) >= 3 else sector_list

        sector_summary = {
            "period": period,
            "sectors": sorted_results,
            "leaders": {name: data for name, data in leaders},
            "laggards": {name: data for name, data in laggards},
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("Successfully retrieved sector performance")
        return json.dumps(sector_summary, indent=2)

    except Exception as e:
        logger.error(f"Error getting sector performance: {str(e)}")
        return json.dumps({"Error": f"Failed to get sector performance: {str(e)}"}, indent=2)


# ------------------------------------------------------------------------------------------------------
# Economic Indicators
# ------------------------------------------------------------------------------------------------------

async def get_economic_indicators() -> str:
    """Get key economic indicators including interest rates, inflation, and GDP data.

    Returns current and historical data for major economic metrics.
    """
    logger.info("Getting economic indicators")

    try:
        results = {}

        # Try Alpha Vantage if API key available
        if ALPHA_VANTAGE_API_KEY:
            # Federal Funds Rate
            try:
                url = f"{ALPHA_VANTAGE_BASE_URL}?function=FEDERAL_FUNDS_RATE&apikey={ALPHA_VANTAGE_API_KEY}"
                data = await make_request(url)
                if not data.get("Error") and "data" in data:
                    latest = data["data"][0] if len(data["data"]) > 0 else {}
                    results["federal_funds_rate"] = {
                        "value": float(latest.get("value", 0)),
                        "date": latest.get("date", ""),
                        "unit": "percent"
                    }
            except Exception as e:
                logger.warning(f"Error fetching federal funds rate: {str(e)}")

            # CPI (Inflation)
            try:
                url = f"{ALPHA_VANTAGE_BASE_URL}?function=CPI&apikey={ALPHA_VANTAGE_API_KEY}"
                data = await make_request(url)
                if not data.get("Error") and "data" in data:
                    latest = data["data"][0] if len(data["data"]) > 0 else {}
                    prev = data["data"][1] if len(data["data"]) > 1 else {}

                    # Calculate YoY inflation
                    prev_12mo = data["data"][12] if len(data["data"]) > 12 else {}
                    current_cpi = float(latest.get("value", 0))
                    prev_year_cpi = float(prev_12mo.get("value", current_cpi))
                    inflation_yoy = ((current_cpi - prev_year_cpi) / prev_year_cpi) * 100 if prev_year_cpi else 0

                    results["cpi"] = {
                        "value": current_cpi,
                        "yoy_change": inflation_yoy,
                        "date": latest.get("date", ""),
                        "unit": "index"
                    }
            except Exception as e:
                logger.warning(f"Error fetching CPI: {str(e)}")

            # Unemployment Rate
            try:
                url = f"{ALPHA_VANTAGE_BASE_URL}?function=UNEMPLOYMENT&apikey={ALPHA_VANTAGE_API_KEY}"
                data = await make_request(url)
                if not data.get("Error") and "data" in data:
                    latest = data["data"][0] if len(data["data"]) > 0 else {}
                    results["unemployment_rate"] = {
                        "value": float(latest.get("value", 0)),
                        "date": latest.get("date", ""),
                        "unit": "percent"
                    }
            except Exception as e:
                logger.warning(f"Error fetching unemployment: {str(e)}")

            # GDP
            try:
                url = f"{ALPHA_VANTAGE_BASE_URL}?function=REAL_GDP&apikey={ALPHA_VANTAGE_API_KEY}"
                data = await make_request(url)
                if not data.get("Error") and "data" in data:
                    latest = data["data"][0] if len(data["data"]) > 0 else {}
                    prev = data["data"][1] if len(data["data"]) > 1 else {}

                    # Calculate QoQ growth
                    current_gdp = float(latest.get("value", 0))
                    prev_gdp = float(prev.get("value", current_gdp))
                    gdp_growth = ((current_gdp - prev_gdp) / prev_gdp) * 100 if prev_gdp else 0

                    results["gdp"] = {
                        "value": current_gdp,
                        "qoq_growth": gdp_growth,
                        "date": latest.get("date", ""),
                        "unit": "billions"
                    }
            except Exception as e:
                logger.warning(f"Error fetching GDP: {str(e)}")

        # Get Treasury yields via Yahoo Finance
        try:
            treasury_tickers = {
                "3_month": "^IRX",
                "2_year": "^2YR",  # Note: These might not work, fallback needed
                "10_year": "^TNX",
                "30_year": "^TYX",
            }

            treasury_yields = {}
            for name, ticker in treasury_tickers.items():
                try:
                    t = yf.Ticker(ticker)
                    hist = t.history(period="5d")
                    if not hist.empty:
                        current_yield = hist['Close'].iloc[-1]
                        treasury_yields[name] = {
                            "value": current_yield,
                            "unit": "percent"
                        }
                except Exception:
                    continue

            if treasury_yields:
                results["treasury_yields"] = treasury_yields

                # Calculate yield curve slope (10Y - 2Y)
                if "10_year" in treasury_yields and "2_year" in treasury_yields:
                    slope = treasury_yields["10_year"]["value"] - treasury_yields["2_year"]["value"]
                    results["yield_curve"] = {
                        "slope": slope,
                        "inverted": slope < 0,
                        "interpretation": "Inverted (recession signal)" if slope < 0 else "Normal"
                    }
        except Exception as e:
            logger.warning(f"Error fetching treasury yields: {str(e)}")

        # Add economic interpretation
        interpretation = []

        if "federal_funds_rate" in results:
            rate = results["federal_funds_rate"]["value"]
            if rate > 4.5:
                interpretation.append("Federal funds rate is elevated, indicating tight monetary policy")
            elif rate < 2.0:
                interpretation.append("Federal funds rate is low, indicating accommodative monetary policy")

        if "cpi" in results:
            inflation = results["cpi"]["yoy_change"]
            if inflation > 3.0:
                interpretation.append(f"Inflation at {inflation:.2f}% is above the Fed's 2% target")
            elif inflation < 2.0:
                interpretation.append(f"Inflation at {inflation:.2f}% is below the Fed's 2% target")

        if "yield_curve" in results and results["yield_curve"]["inverted"]:
            interpretation.append("Yield curve is inverted, historically a recession indicator")

        results["interpretation"] = interpretation
        results["timestamp"] = datetime.now().isoformat()

        logger.info("Successfully retrieved economic indicators")
        return json.dumps(results, indent=2)

    except Exception as e:
        logger.error(f"Error getting economic indicators: {str(e)}")
        return json.dumps({"Error": f"Failed to get economic indicators: {str(e)}"}, indent=2)


# ------------------------------------------------------------------------------------------------------
# Options Data and Implied Volatility
# ------------------------------------------------------------------------------------------------------


async def get_options_chain(ticker: str, expiration_date: Optional[str] = None) -> str:
    """Get options chain data for a stock including calls and puts.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL)
        expiration_date: Specific expiration date (YYYY-MM-DD), or None for nearest expiration
    """
    logger.info(f"Getting options chain for {ticker}")

    try:
        stock = yf.Ticker(ticker)

        # Get available expiration dates
        expirations = stock.options

        if not expirations:
            return json.dumps({"Error": "No options data available for this ticker"}, indent=2)

        # Use provided expiration or the nearest one
        if expiration_date:
            if expiration_date not in expirations:
                return json.dumps({"Error": f"Expiration date {expiration_date} not available. Available: {expirations}"}, indent=2)
            selected_exp = expiration_date
        else:
            selected_exp = expirations[0]

        # Get options chain
        opt_chain = stock.option_chain(selected_exp)

        # Process calls
        calls = opt_chain.calls
        calls_data = calls[[
            'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest',
            'impliedVolatility', 'inTheMoney'
        ]].to_dict('records')

        # Process puts
        puts = opt_chain.puts
        puts_data = puts[[
            'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest',
            'impliedVolatility', 'inTheMoney'
        ]].to_dict('records')

        # Calculate put/call ratio
        total_call_oi = calls['openInterest'].sum()
        total_put_oi = puts['openInterest'].sum()
        put_call_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0

        # Find at-the-money options
        current_price_data = json.loads(await get_current_stock_price(ticker))
        current_price = current_price_data.get("price", 0)

        # Find ATM call and put
        atm_call = min(calls_data, key=lambda x: abs(x['strike'] - current_price)) if calls_data else None
        atm_put = min(puts_data, key=lambda x: abs(x['strike'] - current_price)) if puts_data else None

        results = {
            "ticker": ticker,
            "current_price": current_price,
            "expiration_date": selected_exp,
            "available_expirations": list(expirations),
            "calls": calls_data,
            "puts": puts_data,
            "summary": {
                "total_call_volume": int(calls['volume'].sum()),
                "total_put_volume": int(puts['volume'].sum()),
                "total_call_open_interest": int(total_call_oi),
                "total_put_open_interest": int(total_put_oi),
                "put_call_ratio": put_call_ratio,
                "atm_call_iv": atm_call['impliedVolatility'] if atm_call else None,
                "atm_put_iv": atm_put['impliedVolatility'] if atm_put else None,
            },
            "interpretation": []
        }

        # Add interpretation
        if put_call_ratio > 1.0:
            results["interpretation"].append(f"Put/call ratio of {put_call_ratio:.2f} indicates bearish sentiment")
        elif put_call_ratio < 0.7:
            results["interpretation"].append(f"Put/call ratio of {put_call_ratio:.2f} indicates bullish sentiment")
        else:
            results["interpretation"].append(f"Put/call ratio of {put_call_ratio:.2f} is neutral")

        logger.info(f"Successfully retrieved options chain for {ticker}")
        return json.dumps(results, indent=2)

    except Exception as e:
        logger.error(f"Error getting options chain: {str(e)}")
        return json.dumps({"Error": f"Failed to get options chain: {str(e)}"}, indent=2)



async def get_implied_volatility_surface(ticker: str) -> str:
    """Get implied volatility surface across multiple strikes and expirations.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL)
    """
    logger.info(f"Getting IV surface for {ticker}")

    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options

        if not expirations:
            return json.dumps({"Error": "No options data available"}, indent=2)

        # Get current price
        current_price_data = json.loads(await get_current_stock_price(ticker))
        current_price = current_price_data.get("price", 0)

        iv_surface = []

        # Limit to first 6 expirations for performance
        for exp_date in expirations[:6]:
            try:
                opt_chain = stock.option_chain(exp_date)
                calls = opt_chain.calls

                # Calculate days to expiration
                exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
                days_to_exp = (exp_dt - datetime.now()).days

                # Get IV for different moneyness levels
                for _, row in calls.iterrows():
                    strike = row['strike']
                    iv = row['impliedVolatility']
                    moneyness = strike / current_price

                    iv_surface.append({
                        "expiration": exp_date,
                        "days_to_expiration": days_to_exp,
                        "strike": strike,
                        "moneyness": moneyness,
                        "implied_volatility": iv,
                    })
            except Exception as e:
                logger.warning(f"Error processing expiration {exp_date}: {str(e)}")
                continue

        # Calculate average IV by expiration
        df = pd.DataFrame(iv_surface)
        if not df.empty:
            iv_by_expiration = df.groupby('expiration')['implied_volatility'].mean().to_dict()

            # Calculate term structure (volatility smile)
            term_structure = []
            for exp in sorted(iv_by_expiration.keys()):
                exp_dt = datetime.strptime(exp, "%Y-%m-%d")
                days = (exp_dt - datetime.now()).days
                term_structure.append({
                    "expiration": exp,
                    "days_to_expiration": days,
                    "average_iv": iv_by_expiration[exp]
                })

            results = {
                "ticker": ticker,
                "current_price": current_price,
                "iv_surface": iv_surface,
                "term_structure": term_structure,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            results = {"Error": "Unable to calculate IV surface"}

        logger.info(f"Successfully retrieved IV surface for {ticker}")
        return json.dumps(results, indent=2)

    except Exception as e:
        logger.error(f"Error getting IV surface: {str(e)}")
        return json.dumps({"Error": f"Failed to get IV surface: {str(e)}"}, indent=2)


# ------------------------------------------------------------------------------------------------------
# Insider Trading and Institutional Ownership
# ------------------------------------------------------------------------------------------------------


async def get_insider_transactions(ticker: str, limit: int = 20) -> str:
    """Get recent insider trading transactions for a stock.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL)
        limit: Number of transactions to return (default: 20)
    """
    logger.info(f"Getting insider transactions for {ticker}")

    try:
        # Try FMP API if available
        if FMP_API_KEY:
            url = f"{FMP_BASE_URL}/insider-trading?symbol={ticker}&limit={limit}&apikey={FMP_API_KEY}"
            data = await make_request(url)

            if not data.get("Error") and isinstance(data, list):
                # Process the data
                transactions = []
                for transaction in data:
                    transactions.append({
                        "filing_date": transaction.get("filingDate"),
                        "transaction_date": transaction.get("transactionDate"),
                        "insider_name": transaction.get("reportingName"),
                        "transaction_type": transaction.get("transactionType"),
                        "securities_owned": transaction.get("securitiesOwned"),
                        "securities_transacted": transaction.get("securitiesTransacted"),
                        "price": transaction.get("price"),
                        "type_of_owner": transaction.get("typeOfOwner"),
                    })

                # Calculate summary statistics
                buys = [t for t in transactions if 'P-Purchase' in t.get('transaction_type', '')]
                sells = [t for t in transactions if 'S-Sale' in t.get('transaction_type', '')]

                total_buy_value = sum(t.get('securities_transacted', 0) * t.get('price', 0) for t in buys if t.get('price'))
                total_sell_value = sum(abs(t.get('securities_transacted', 0)) * t.get('price', 0) for t in sells if t.get('price'))

                summary = {
                    "total_transactions": len(transactions),
                    "buys": len(buys),
                    "sells": len(sells),
                    "total_buy_value": total_buy_value,
                    "total_sell_value": total_sell_value,
                    "net_insider_activity": total_buy_value - total_sell_value,
                }

                # Interpretation
                interpretation = []
                if len(buys) > len(sells) * 2:
                    interpretation.append("Strong insider buying activity (bullish signal)")
                elif len(sells) > len(buys) * 2:
                    interpretation.append("Heavy insider selling activity (bearish signal)")
                else:
                    interpretation.append("Mixed insider activity")

                results = {
                    "ticker": ticker,
                    "transactions": transactions,
                    "summary": summary,
                    "interpretation": interpretation,
                    "timestamp": datetime.now().isoformat(),
                }

                logger.info(f"Successfully retrieved insider transactions for {ticker}")
                return json.dumps(results, indent=2)

        # Fallback to yfinance
        stock = yf.Ticker(ticker)
        insider_txns = stock.insider_transactions

        if insider_txns is not None and not insider_txns.empty:
            # Limit results
            insider_txns = insider_txns.head(limit)
            transactions = insider_txns.to_dict('records')

            results = {
                "ticker": ticker,
                "transactions": transactions,
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(f"Successfully retrieved insider transactions from yfinance for {ticker}")
            return json.dumps(results, indent=2)

        return json.dumps({"Error": "No insider transaction data available"}, indent=2)

    except Exception as e:
        logger.error(f"Error getting insider transactions: {str(e)}")
        return json.dumps({"Error": f"Failed to get insider transactions: {str(e)}"}, indent=2)



async def get_institutional_ownership(ticker: str) -> str:
    """Get institutional ownership data for a stock.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL)
    """
    logger.info(f"Getting institutional ownership for {ticker}")

    try:
        stock = yf.Ticker(ticker)

        # Get institutional holders
        inst_holders = stock.institutional_holders

        if inst_holders is not None and not inst_holders.empty:
            holders = inst_holders.to_dict('records')

            # Calculate summary
            total_shares_held = inst_holders['Shares'].sum() if 'Shares' in inst_holders.columns else 0

            # Get total shares outstanding
            info = stock.info
            shares_outstanding = info.get('sharesOutstanding', 0)
            institutional_pct = (total_shares_held / shares_outstanding) * 100 if shares_outstanding else 0

            results = {
                "ticker": ticker,
                "institutional_holders": holders,
                "summary": {
                    "total_shares_held_by_institutions": total_shares_held,
                    "shares_outstanding": shares_outstanding,
                    "institutional_ownership_percent": institutional_pct,
                    "number_of_institutions": len(holders),
                },
                "timestamp": datetime.now().isoformat(),
            }

            # Interpretation
            interpretation = []
            if institutional_pct > 80:
                interpretation.append("Very high institutional ownership (>80%), indicating strong institutional confidence")
            elif institutional_pct > 60:
                interpretation.append("High institutional ownership (60-80%), indicating good institutional support")
            elif institutional_pct < 30:
                interpretation.append("Low institutional ownership (<30%), may indicate retail-driven stock")

            results["interpretation"] = interpretation

            logger.info(f"Successfully retrieved institutional ownership for {ticker}")
            return json.dumps(results, indent=2)

        return json.dumps({"Error": "No institutional ownership data available"}, indent=2)

    except Exception as e:
        logger.error(f"Error getting institutional ownership: {str(e)}")
        return json.dumps({"Error": f"Failed to get institutional ownership: {str(e)}"}, indent=2)


# ------------------------------------------------------------------------------------------------------
# Market Sentiment Indicators
# ------------------------------------------------------------------------------------------------------


async def get_market_sentiment_indicators() -> str:
    """Get comprehensive market sentiment indicators including Fear & Greed, VIX, Put/Call ratios.

    Returns multiple sentiment metrics to gauge overall market mood.
    """
    logger.info("Getting market sentiment indicators")

    try:
        results = {}

        # VIX (Volatility Index)
        try:
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period="5d")
            if not vix_hist.empty:
                current_vix = vix_hist['Close'].iloc[-1]
                prev_vix = vix_hist['Close'].iloc[-2] if len(vix_hist) > 1 else current_vix
                vix_change = current_vix - prev_vix

                # VIX interpretation
                vix_sentiment = "Extreme Fear"
                if current_vix < 12:
                    vix_sentiment = "Complacency"
                elif current_vix < 20:
                    vix_sentiment = "Low Volatility"
                elif current_vix < 30:
                    vix_sentiment = "Elevated Volatility"

                results["vix"] = {
                    "value": current_vix,
                    "change": vix_change,
                    "sentiment": vix_sentiment,
                }
        except Exception as e:
            logger.warning(f"Error fetching VIX: {str(e)}")

        # Put/Call Ratio (using SPY as proxy)
        try:
            spy = yf.Ticker("SPY")
            spy_options = spy.options
            if spy_options:
                exp = spy_options[0]  # Nearest expiration
                opt_chain = spy.option_chain(exp)

                total_call_volume = opt_chain.calls['volume'].sum()
                total_put_volume = opt_chain.puts['volume'].sum()
                put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0

                # Interpretation
                pc_sentiment = "Neutral"
                if put_call_ratio > 1.15:
                    pc_sentiment = "Bearish (High Put Buying)"
                elif put_call_ratio < 0.85:
                    pc_sentiment = "Bullish (High Call Buying)"

                results["put_call_ratio"] = {
                    "value": put_call_ratio,
                    "sentiment": pc_sentiment,
                    "call_volume": int(total_call_volume),
                    "put_volume": int(total_put_volume),
                }
        except Exception as e:
            logger.warning(f"Error fetching put/call ratio: {str(e)}")

        # Market Breadth - Advance/Decline
        try:
            # Use NYSE advance/decline line
            adv = yf.Ticker("^NYAD")  # NYSE Advance-Decline
            adv_hist = adv.history(period="5d")
            if not adv_hist.empty:
                current_ad = adv_hist['Close'].iloc[-1]
                prev_ad = adv_hist['Close'].iloc[-2] if len(adv_hist) > 1 else current_ad

                ad_sentiment = "Positive" if current_ad > prev_ad else "Negative"

                results["advance_decline"] = {
                    "value": current_ad,
                    "change": current_ad - prev_ad,
                    "sentiment": ad_sentiment,
                }
        except Exception as e:
            logger.warning(f"Error fetching advance/decline: {str(e)}")

        # High Yield Spread (Credit spread as risk indicator)
        try:
            # Use HYG (high yield) vs TLT (treasuries) as spread proxy
            hyg = yf.Ticker("HYG")
            hyg_hist = hyg.history(period="1mo")

            if not hyg_hist.empty:
                hyg_returns = hyg_hist['Close'].pct_change()
                hyg_volatility = hyg_returns.std() * np.sqrt(252) * 100

                credit_sentiment = "Risk-On"
                if hyg_volatility > 15:
                    credit_sentiment = "Risk-Off (High Credit Volatility)"

                results["credit_market"] = {
                    "hyg_volatility": hyg_volatility,
                    "sentiment": credit_sentiment,
                }
        except Exception as e:
            logger.warning(f"Error fetching credit spread: {str(e)}")

        # Calculate composite sentiment score
        sentiment_scores = []

        if "vix" in results:
            # Lower VIX = bullish (invert the score)
            vix_score = max(0, min(100, 100 - (results["vix"]["value"] * 3)))
            sentiment_scores.append(vix_score)

        if "put_call_ratio" in results:
            # Lower P/C = bullish
            pc_ratio = results["put_call_ratio"]["value"]
            pc_score = max(0, min(100, (1 - pc_ratio) * 100))
            sentiment_scores.append(pc_score)

        if "advance_decline" in results:
            # Positive change = bullish
            ad_score = 60 if results["advance_decline"]["sentiment"] == "Positive" else 40
            sentiment_scores.append(ad_score)

        if sentiment_scores:
            composite_score = sum(sentiment_scores) / len(sentiment_scores)

            overall_sentiment = "Neutral"
            if composite_score > 70:
                overall_sentiment = "Extreme Greed"
            elif composite_score > 55:
                overall_sentiment = "Greed"
            elif composite_score < 30:
                overall_sentiment = "Extreme Fear"
            elif composite_score < 45:
                overall_sentiment = "Fear"

            results["composite_sentiment"] = {
                "score": composite_score,
                "sentiment": overall_sentiment,
            }

        results["timestamp"] = datetime.now().isoformat()

        logger.info("Successfully retrieved market sentiment indicators")
        return json.dumps(results, indent=2)

    except Exception as e:
        logger.error(f"Error getting market sentiment: {str(e)}")
        return json.dumps({"Error": f"Failed to get market sentiment: {str(e)}"}, indent=2)


# ------------------------------------------------------------------------------------------------------
# Analyst Ratings and Price Targets
# ------------------------------------------------------------------------------------------------------


async def get_analyst_ratings(ticker: str) -> str:
    """Get analyst ratings, price targets, and recommendations for a stock.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL)
    """
    logger.info(f"Getting analyst ratings for {ticker}")

    try:
        # Try FMP if API key available
        if FMP_API_KEY:
            # Get analyst ratings
            url = f"{FMP_BASE_URL}/grade/{ticker}?limit=10&apikey={FMP_API_KEY}"
            ratings_data = await make_request(url)

            # Get price targets
            targets_url = f"{FMP_BASE_URL}/price-target?symbol={ticker}&apikey={FMP_API_KEY}"
            targets_data = await make_request(url)

            if not ratings_data.get("Error") and isinstance(ratings_data, list):
                # Process ratings
                recent_ratings = ratings_data[:10]

                # Count recommendations
                buy_count = sum(1 for r in recent_ratings if 'buy' in r.get('gradeToGrade', '').lower() or 'outperform' in r.get('gradeToGrade', '').lower())
                hold_count = sum(1 for r in recent_ratings if 'hold' in r.get('gradeToGrade', '').lower() or 'neutral' in r.get('gradeToGrade', '').lower())
                sell_count = sum(1 for r in recent_ratings if 'sell' in r.get('gradeToGrade', '').lower() or 'underperform' in r.get('gradeToGrade', '').lower())

                results = {
                    "ticker": ticker,
                    "recent_ratings": recent_ratings,
                    "consensus": {
                        "buy": buy_count,
                        "hold": hold_count,
                        "sell": sell_count,
                        "total": len(recent_ratings),
                    },
                    "timestamp": datetime.now().isoformat(),
                }

                logger.info(f"Successfully retrieved analyst ratings for {ticker}")
                return json.dumps(results, indent=2)

        # Fallback to yfinance
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get recommendations
        recommendations = stock.recommendations

        results = {
            "ticker": ticker,
            "info": {
                "target_mean_price": info.get('targetMeanPrice'),
                "target_high_price": info.get('targetHighPrice'),
                "target_low_price": info.get('targetLowPrice'),
                "recommendation_mean": info.get('recommendationMean'),
                "recommendation_key": info.get('recommendationKey'),
                "number_of_analyst_opinions": info.get('numberOfAnalystOpinions'),
            },
            "timestamp": datetime.now().isoformat(),
        }

        if recommendations is not None and not recommendations.empty:
            # Get recent recommendations
            recent = recommendations.tail(20).to_dict('records')
            results["recent_recommendations"] = recent

        # Get current price for comparison
        current_price_data = json.loads(await get_current_stock_price(ticker))
        current_price = current_price_data.get("price", 0)

        if info.get('targetMeanPrice') and current_price:
            upside = ((info['targetMeanPrice'] - current_price) / current_price) * 100
            results["price_target_upside"] = upside

        logger.info(f"Successfully retrieved analyst data for {ticker}")
        return json.dumps(results, indent=2)

    except Exception as e:
        logger.error(f"Error getting analyst ratings: {str(e)}")
        return json.dumps({"Error": f"Failed to get analyst ratings: {str(e)}"}, indent=2)


# ------------------------------------------------------------------------------------------------------
# Earnings Calendar and Analysis
# ------------------------------------------------------------------------------------------------------


async def get_earnings_calendar(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
) -> str:
    """Get upcoming earnings announcements calendar.

    Args:
        from_date: Start date (YYYY-MM-DD), defaults to today
        to_date: End date (YYYY-MM-DD), defaults to 7 days from now
    """
    logger.info("Getting earnings calendar")

    try:
        if not from_date:
            from_date = datetime.now().strftime("%Y-%m-%d")
        if not to_date:
            to_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

        # Try FMP if API key available
        if FMP_API_KEY:
            url = f"{FMP_BASE_URL}/earning_calendar?from={from_date}&to={to_date}&apikey={FMP_API_KEY}"
            data = await make_request(url)

            if not data.get("Error") and isinstance(data, list):
                # Sort by date
                sorted_data = sorted(data, key=lambda x: x.get('date', ''))

                results = {
                    "from_date": from_date,
                    "to_date": to_date,
                    "earnings_announcements": sorted_data,
                    "total_companies": len(sorted_data),
                    "timestamp": datetime.now().isoformat(),
                }

                logger.info(f"Successfully retrieved earnings calendar")
                return json.dumps(results, indent=2)

        return json.dumps({"Error": "Earnings calendar requires FMP API key"}, indent=2)

    except Exception as e:
        logger.error(f"Error getting earnings calendar: {str(e)}")
        return json.dumps({"Error": f"Failed to get earnings calendar: {str(e)}"}, indent=2)



async def get_earnings_history(ticker: str, limit: int = 4) -> str:
    """Get historical earnings data including estimates, actuals, and surprises.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL)
        limit: Number of earnings reports to return (default: 4)
    """
    logger.info(f"Getting earnings history for {ticker}")

    try:
        stock = yf.Ticker(ticker)

        # Get earnings dates and estimates
        earnings = stock.earnings_dates

        if earnings is not None and not earnings.empty:
            # Limit to recent earnings
            recent_earnings = earnings.head(limit).to_dict('index')

            # Convert to list format
            earnings_list = []
            for date, data in recent_earnings.items():
                earnings_list.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "eps_estimate": data.get('EPS Estimate'),
                    "eps_actual": data.get('Reported EPS'),
                    "surprise": data.get('Surprise(%)'),
                })

            # Calculate average surprise
            surprises = [e['surprise'] for e in earnings_list if e['surprise'] is not None]
            avg_surprise = sum(surprises) / len(surprises) if surprises else 0

            # Count beats vs misses
            beats = sum(1 for s in surprises if s > 0)
            misses = sum(1 for s in surprises if s < 0)

            results = {
                "ticker": ticker,
                "earnings_history": earnings_list,
                "summary": {
                    "average_surprise_percent": avg_surprise,
                    "beats": beats,
                    "misses": misses,
                    "total_reports": len(earnings_list),
                },
                "timestamp": datetime.now().isoformat(),
            }

            # Add interpretation
            interpretation = []
            if avg_surprise > 5:
                interpretation.append("Company consistently beats estimates (positive track record)")
            elif avg_surprise < -5:
                interpretation.append("Company frequently misses estimates (concerning)")

            if beats > misses * 2:
                interpretation.append("Strong earnings execution")

            results["interpretation"] = interpretation

            logger.info(f"Successfully retrieved earnings history for {ticker}")
            return json.dumps(results, indent=2)

        return json.dumps({"Error": "No earnings data available"}, indent=2)

    except Exception as e:
        logger.error(f"Error getting earnings history: {str(e)}")
        return json.dumps({"Error": f"Failed to get earnings history: {str(e)}"}, indent=2)


# ------------------------------------------------------------------------------------------------------
# Advanced Technical Analysis
# ------------------------------------------------------------------------------------------------------


async def detect_chart_patterns(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None
) -> str:
    """Detect common chart patterns like head and shoulders, double tops/bottoms, triangles.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL)
        start_date: Start date for analysis (YYYY-MM-DD)
        end_date: End date for analysis (default: today)
    """
    logger.info(f"Detecting chart patterns for {ticker}")

    try:
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Get historical data
        is_crypto = "-" in ticker
        if is_crypto:
            history_data = json.loads(await get_historical_crypto_prices(ticker, start_date, end_date, "day"))
        else:
            history_data = json.loads(await get_historical_stock_prices(ticker, start_date, end_date, "1d"))

        if isinstance(history_data, dict) and history_data.get("Error"):
            return json.dumps({"Error": "Unable to get price data"}, indent=2)

        # Convert to DataFrame
        df = pd.DataFrame(history_data)
        if is_crypto:
            if "price" in df.columns:
                df["close"] = df["price"]
        else:
            df.columns = [col.lower() for col in df.columns]

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        patterns_detected = []

        # 1. Double Top Pattern
        # Find local maxima
        prices = df["close"].values
        window = 10

        local_maxima = []
        for i in range(window, len(prices) - window):
            if prices[i] == max(prices[i-window:i+window+1]):
                local_maxima.append((i, prices[i]))

        # Check for double tops (two similar peaks)
        for i in range(len(local_maxima) - 1):
            idx1, price1 = local_maxima[i]
            idx2, price2 = local_maxima[i + 1]

            # Check if peaks are similar in price (within 2%)
            if abs(price1 - price2) / price1 < 0.02 and (idx2 - idx1) > 10:
                # Check if there's a trough between them
                valley_idx = idx1 + np.argmin(prices[idx1:idx2])
                valley_price = prices[valley_idx]

                if price1 > valley_price * 1.03:  # At least 3% drop
                    patterns_detected.append({
                        "pattern": "Double Top",
                        "type": "Bearish Reversal",
                        "first_peak_date": df.index[idx1].strftime("%Y-%m-%d"),
                        "second_peak_date": df.index[idx2].strftime("%Y-%m-%d"),
                        "peak_price": (price1 + price2) / 2,
                        "support_level": valley_price,
                        "signal": "Bearish - Potential reversal if support breaks",
                    })

        # 2. Double Bottom Pattern
        local_minima = []
        for i in range(window, len(prices) - window):
            if prices[i] == min(prices[i-window:i+window+1]):
                local_minima.append((i, prices[i]))

        for i in range(len(local_minima) - 1):
            idx1, price1 = local_minima[i]
            idx2, price2 = local_minima[i + 1]

            if abs(price1 - price2) / price1 < 0.02 and (idx2 - idx1) > 10:
                peak_idx = idx1 + np.argmax(prices[idx1:idx2])
                peak_price = prices[peak_idx]

                if peak_price > price1 * 1.03:
                    patterns_detected.append({
                        "pattern": "Double Bottom",
                        "type": "Bullish Reversal",
                        "first_bottom_date": df.index[idx1].strftime("%Y-%m-%d"),
                        "second_bottom_date": df.index[idx2].strftime("%Y-%m-%d"),
                        "bottom_price": (price1 + price2) / 2,
                        "resistance_level": peak_price,
                        "signal": "Bullish - Potential reversal if resistance breaks",
                    })

        # 3. Support and Resistance Levels
        support_resistance = []

        # Find significant levels (price levels tested multiple times)
        price_levels = {}
        for price in prices:
            # Round to nearest significant level
            level = round(price, 2)
            if level in price_levels:
                price_levels[level] += 1
            else:
                price_levels[level] = 1

        # Find levels tested at least 3 times
        significant_levels = [(level, count) for level, count in price_levels.items() if count >= 3]
        significant_levels.sort(key=lambda x: x[1], reverse=True)

        current_price = prices[-1]

        for level, count in significant_levels[:10]:  # Top 10 levels
            level_type = "Support" if level < current_price else "Resistance"
            distance_pct = abs((level - current_price) / current_price) * 100

            support_resistance.append({
                "level": level,
                "type": level_type,
                "times_tested": count,
                "distance_from_current_price_pct": distance_pct,
            })

        # 4. Trend Analysis
        # Calculate trend using linear regression
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)

        trend = "Uptrend" if slope > 0 else "Downtrend"
        trend_strength = abs(slope) / np.mean(prices) * 100  # Normalized slope

        results = {
            "ticker": ticker,
            "period": f"{start_date} to {end_date}",
            "current_price": current_price,
            "patterns_detected": patterns_detected,
            "support_resistance_levels": sorted(support_resistance, key=lambda x: x["distance_from_current_price_pct"]),
            "trend_analysis": {
                "direction": trend,
                "strength_pct": trend_strength,
            },
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Successfully detected patterns for {ticker}")
        return json.dumps(results, indent=2)

    except Exception as e:
        logger.error(f"Error detecting patterns: {str(e)}")
        return json.dumps({"Error": f"Failed to detect patterns: {str(e)}"}, indent=2)



async def calculate_fibonacci_levels(
    ticker: str,
    trend_type: str = "uptrend",
    lookback_days: int = 90
) -> str:
    """Calculate Fibonacci retracement and extension levels for a stock.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL)
        trend_type: Type of trend ("uptrend" or "downtrend")
        lookback_days: Number of days to look back for swing high/low (default: 90)
    """
    logger.info(f"Calculating Fibonacci levels for {ticker}")

    try:
        # Get historical data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        is_crypto = "-" in ticker
        if is_crypto:
            history_data = json.loads(await get_historical_crypto_prices(ticker, start_date, end_date, "day"))
        else:
            history_data = json.loads(await get_historical_stock_prices(ticker, start_date, end_date, "1d"))

        if isinstance(history_data, dict) and history_data.get("Error"):
            return json.dumps({"Error": "Unable to get price data"}, indent=2)

        # Convert to DataFrame
        df = pd.DataFrame(history_data)
        if is_crypto:
            if "price" in df.columns:
                df["close"] = df["price"]
                df["high"] = df["price"]
                df["low"] = df["price"]
        else:
            df.columns = [col.lower() for col in df.columns]

        # Find swing high and low
        swing_high = df["high"].max()
        swing_low = df["low"].min()
        current_price = df["close"].iloc[-1]

        # Fibonacci retracement levels
        diff = swing_high - swing_low

        if trend_type.lower() == "uptrend":
            # Retracement levels (from high)
            fib_levels = {
                "0.0% (High)": swing_high,
                "23.6% Retracement": swing_high - (0.236 * diff),
                "38.2% Retracement": swing_high - (0.382 * diff),
                "50.0% Retracement": swing_high - (0.500 * diff),
                "61.8% Retracement": swing_high - (0.618 * diff),
                "78.6% Retracement": swing_high - (0.786 * diff),
                "100.0% (Low)": swing_low,
                "161.8% Extension": swing_high + (0.618 * diff),
                "261.8% Extension": swing_high + (1.618 * diff),
            }
        else:
            # Retracement levels (from low)
            fib_levels = {
                "0.0% (Low)": swing_low,
                "23.6% Retracement": swing_low + (0.236 * diff),
                "38.2% Retracement": swing_low + (0.382 * diff),
                "50.0% Retracement": swing_low + (0.500 * diff),
                "61.8% Retracement": swing_low + (0.618 * diff),
                "78.6% Retracement": swing_low + (0.786 * diff),
                "100.0% (High)": swing_high,
                "161.8% Extension": swing_low - (0.618 * diff),
                "261.8% Extension": swing_low - (1.618 * diff),
            }

        # Identify nearest level to current price
        nearest_level = min(fib_levels.items(), key=lambda x: abs(x[1] - current_price))

        # Identify next support and resistance
        support_levels = [item for item in fib_levels.items() if item[1] < current_price]
        resistance_levels = [item for item in fib_levels.items() if item[1] > current_price]

        nearest_support = max(support_levels, key=lambda x: x[1]) if support_levels else None
        nearest_resistance = min(resistance_levels, key=lambda x: x[1]) if resistance_levels else None

        results = {
            "ticker": ticker,
            "current_price": current_price,
            "trend_type": trend_type,
            "swing_high": swing_high,
            "swing_low": swing_low,
            "price_range": diff,
            "fibonacci_levels": fib_levels,
            "analysis": {
                "nearest_level": {
                    "name": nearest_level[0],
                    "price": nearest_level[1],
                    "distance_pct": abs((nearest_level[1] - current_price) / current_price) * 100,
                },
                "nearest_support": {
                    "name": nearest_support[0] if nearest_support else None,
                    "price": nearest_support[1] if nearest_support else None,
                } if nearest_support else None,
                "nearest_resistance": {
                    "name": nearest_resistance[0] if nearest_resistance else None,
                    "price": nearest_resistance[1] if nearest_resistance else None,
                } if nearest_resistance else None,
            },
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Successfully calculated Fibonacci levels for {ticker}")
        return json.dumps(results, indent=2)

    except Exception as e:
        logger.error(f"Error calculating Fibonacci levels: {str(e)}")
        return json.dumps({"Error": f"Failed to calculate Fibonacci levels: {str(e)}"}, indent=2)
