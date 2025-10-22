import json
import os
import httpx
import logging
import sys
import asyncio
import requests
from mcp.server.fastmcp import FastMCP
from typing import Dict, List, Union, Any, Optional
import yfinance as yf
from datetime import datetime, timedelta

# Configure logging to write to stderr
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("financial-markets-analyser")

# Initialize FastMCP server
mcp = FastMCP("financial-markets-analyser")

# API Constants
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# Helper function to make API requests with httpx
async def make_request(url: str, headers: Optional[Dict] = None) -> Dict[str, Any]:
    """Make an async request to an API with proper error handling."""
    if headers is None:
        headers = {}
        
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error making request to {url}: {str(e)}")
            return {"Error": str(e)}

# Helper function for synchronous requests
def make_sync_request(url: str, headers: Optional[Dict] = None) -> Dict[str, Any]:
    """Make a synchronous request to an API with proper error handling."""
    if headers is None:
        headers = {}
        
    try:
        response = requests.get(url, headers=headers, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error making request to {url}: {str(e)}")
        return {"Error": str(e)}

# Helper function to convert CoinGecko IDs
def convert_ticker_to_coingecko_id(ticker: str) -> str:
    """Convert ticker symbol to CoinGecko ID format."""
    # Common mappings - definitely need a better way to do this
    mapping = {
        "BTC-USD": "bitcoin",
        "ETH-USD": "ethereum",
        "SOL-USD": "solana",
        "ADA-USD": "cardano",
        "XRP-USD": "ripple",
        "DOT-USD": "polkadot",
        "DOGE-USD": "dogecoin",
        "AVAX-USD": "avalanche-2",
        "MATIC-USD": "matic-network",
        "LINK-USD": "chainlink",
    }
    
    # Try direct mapping first
    if ticker in mapping:
        return mapping[ticker]
    
    # If not found, use first part of hyphenated ticker
    if "-" in ticker:
        base_ticker = ticker.split("-")[0].lower()
        return base_ticker
    
    return ticker.lower()

@mcp.tool()
async def get_income_statements(
    ticker: str,
    period: str = "annual",
    limit: int = 4,
) -> str:
    """Get income statements for a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
        period: Period of the income statement (annual or quarterly)
        limit: Number of income statements to return (default: 4)
    """
    logger.info(f"Getting {period} income statements for {ticker}, limit: {limit}")
    
    # Try Alpha Vantage first if API key is available
    if ALPHA_VANTAGE_API_KEY:
        period_map = {"annual": "annual", "quarterly": "quarterly"}
        av_period = period_map.get(period, "annual")
        
        url = f"{ALPHA_VANTAGE_BASE_URL}?function=INCOME_STATEMENT&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        data = await make_request(url)
        
        if not data.get("Error") and av_period + "Reports" in data:
            statements = data.get(f"{av_period}Reports", [])[:limit]
            if statements:
                logger.info(f"Successfully retrieved income statements from Alpha Vantage for {ticker}")
                return json.dumps(statements, indent=2)
    
    # Try Financial Modeling Prep if API key is available
    if FMP_API_KEY:
        period_map = {"annual": "annual", "quarterly": "quarter"}
        fmp_period = period_map.get(period, "annual")
        
        url = f"{FMP_BASE_URL}/income-statement/{ticker}?period={fmp_period}&limit={limit}&apikey={FMP_API_KEY}"
        data = await make_request(url)
        
        if not data.get("Error") and isinstance(data, list) and len(data) > 0:
            logger.info(f"Successfully retrieved income statements from FMP for {ticker}")
            return json.dumps(data[:limit], indent=2)
    
    # Fall back to Yahoo Finance
    try:
        stock = yf.Ticker(ticker)
        
        # Choose the right financials based on period
        if period == "quarterly":
            income_stmt = stock.quarterly_income_stmt
        else:
            income_stmt = stock.income_stmt
            
        # Convert to dict for consistent output format
        if income_stmt is not None and not income_stmt.empty:
            # Convert Pandas DataFrame to dict and limit results
            statements = income_stmt.T.to_dict('records')[:limit]
            logger.info(f"Successfully retrieved income statements from Yahoo Finance for {ticker}")
            return json.dumps(statements, indent=2)
    except Exception as e:
        logger.error(f"Error retrieving income statements from Yahoo Finance: {str(e)}")
    
    return json.dumps({"Error": "Unable to fetch income statements or no income statements found."}, indent=2)

@mcp.tool()
async def get_balance_sheets(
    ticker: str,
    period: str = "annual",
    limit: int = 4,
) -> str:
    """Get balance sheets for a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
        period: Period of the balance sheet (annual or quarterly)
        limit: Number of balance sheets to return (default: 4)
    """
    logger.info(f"Getting {period} balance sheets for {ticker}, limit: {limit}")
    
    # Try Alpha Vantage first if API key is available
    if ALPHA_VANTAGE_API_KEY:
        period_map = {"annual": "annual", "quarterly": "quarterly"}
        av_period = period_map.get(period, "annual")
        
        url = f"{ALPHA_VANTAGE_BASE_URL}?function=BALANCE_SHEET&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        data = await make_request(url)
        
        if not data.get("Error") and av_period + "Reports" in data:
            statements = data.get(f"{av_period}Reports", [])[:limit]
            if statements:
                logger.info(f"Successfully retrieved balance sheets from Alpha Vantage for {ticker}")
                return json.dumps(statements, indent=2)
    
    # Try Financial Modeling Prep if API key is available
    if FMP_API_KEY:
        period_map = {"annual": "annual", "quarterly": "quarter"}
        fmp_period = period_map.get(period, "annual")
        
        url = f"{FMP_BASE_URL}/balance-sheet-statement/{ticker}?period={fmp_period}&limit={limit}&apikey={FMP_API_KEY}"
        data = await make_request(url)
        
        if not data.get("Error") and isinstance(data, list) and len(data) > 0:
            logger.info(f"Successfully retrieved balance sheets from FMP for {ticker}")
            return json.dumps(data[:limit], indent=2)
    
    # Fall back to Yahoo Finance
    try:
        stock = yf.Ticker(ticker)
        
        # Choose the right financials based on period
        if period == "quarterly":
            balance_sheet = stock.quarterly_balance_sheet
        else:
            balance_sheet = stock.balance_sheet
            
        # Convert to dict for consistent output format
        if balance_sheet is not None and not balance_sheet.empty:
            # Convert Pandas DataFrame to dict and limit results
            statements = balance_sheet.T.to_dict('records')[:limit]
            logger.info(f"Successfully retrieved balance sheets from Yahoo Finance for {ticker}")
            return json.dumps(statements, indent=2)
    except Exception as e:
        logger.error(f"Error retrieving balance sheets from Yahoo Finance: {str(e)}")
    
    return json.dumps({"Error": "Unable to fetch balance sheets or no balance sheets found."}, indent=2)

@mcp.tool()
async def get_cash_flow_statements(
    ticker: str,
    period: str = "annual",
    limit: int = 4,
) -> str:
    """Get cash flow statements for a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
        period: Period of the cash flow statement (annual or quarterly)
        limit: Number of cash flow statements to return (default: 4)
    """
    logger.info(f"Getting {period} cash flow statements for {ticker}, limit: {limit}")
    
    # Try Alpha Vantage first if API key is available
    if ALPHA_VANTAGE_API_KEY:
        period_map = {"annual": "annual", "quarterly": "quarterly"}
        av_period = period_map.get(period, "annual")
        
        url = f"{ALPHA_VANTAGE_BASE_URL}?function=CASH_FLOW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        data = await make_request(url)
        
        if not data.get("Error") and av_period + "Reports" in data:
            statements = data.get(f"{av_period}Reports", [])[:limit]
            if statements:
                logger.info(f"Successfully retrieved cash flow statements from Alpha Vantage for {ticker}")
                return json.dumps(statements, indent=2)
    
    # Try Financial Modeling Prep if API key is available
    if FMP_API_KEY:
        period_map = {"annual": "annual", "quarterly": "quarter"}
        fmp_period = period_map.get(period, "annual")
        
        url = f"{FMP_BASE_URL}/cash-flow-statement/{ticker}?period={fmp_period}&limit={limit}&apikey={FMP_API_KEY}"
        data = await make_request(url)
        
        if not data.get("Error") and isinstance(data, list) and len(data) > 0:
            logger.info(f"Successfully retrieved cash flow statements from FMP for {ticker}")
            return json.dumps(data[:limit], indent=2)
    
    # Fall back to Yahoo Finance
    try:
        stock = yf.Ticker(ticker)
        
        # Choose the right financials based on period
        if period == "quarterly":
            cash_flow = stock.quarterly_cashflow
        else:
            cash_flow = stock.cashflow
            
        # Convert to dict for consistent output format
        if cash_flow is not None and not cash_flow.empty:
            # Convert Pandas DataFrame to dict and limit results
            statements = cash_flow.T.to_dict('records')[:limit]
            logger.info(f"Successfully retrieved cash flow statements from Yahoo Finance for {ticker}")
            return json.dumps(statements, indent=2)
    except Exception as e:
        logger.error(f"Error retrieving cash flow statements from Yahoo Finance: {str(e)}")
    
    return json.dumps({"Error": "Unable to fetch cash flow statements or no cash flow statements found."}, indent=2)

@mcp.tool()
async def get_current_stock_price(ticker: str) -> str:
    """Get the current / latest price of a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
    """
    logger.info(f"Getting current stock price for {ticker}")
    
    # Try Yahoo Finance first (most reliable for current prices)
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract the most relevant price information
        snapshot = {
            "ticker": ticker,
            "price": info.get("currentPrice", info.get("regularMarketPrice")),
            "currency": info.get("currency"),
            "previousClose": info.get("previousClose"),
            "open": info.get("open"),
            "dayHigh": info.get("dayHigh"),
            "dayLow": info.get("dayLow"),
            "volume": info.get("volume"),
            "marketCap": info.get("marketCap"),
            "timestamp": datetime.now().isoformat(),
        }
        
        logger.info(f"Successfully retrieved current stock price from Yahoo Finance for {ticker}")
        return json.dumps(snapshot, indent=2)
    except Exception as e:
        logger.error(f"Error retrieving current stock price from Yahoo Finance: {str(e)}")
    
    # Try Alpha Vantage if YF fails and API key is available
    if ALPHA_VANTAGE_API_KEY:
        url = f"{ALPHA_VANTAGE_BASE_URL}?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        data = await make_request(url)
        
        if not data.get("Error") and "Global Quote" in data:
            quote = data.get("Global Quote", {})
            if quote:
                snapshot = {
                    "ticker": ticker,
                    "price": float(quote.get("05. price", 0)),
                    "previousClose": float(quote.get("08. previous close", 0)),
                    "open": float(quote.get("02. open", 0)),
                    "high": float(quote.get("03. high", 0)),
                    "low": float(quote.get("04. low", 0)),
                    "volume": int(quote.get("06. volume", 0)),
                    "timestamp": quote.get("07. latest trading day"),
                }
                logger.info(f"Successfully retrieved current stock price from Alpha Vantage for {ticker}")
                return json.dumps(snapshot, indent=2)
    
    # Try FMP as last resort if API key is available
    if FMP_API_KEY:
        url = f"{FMP_BASE_URL}/quote/{ticker}?apikey={FMP_API_KEY}"
        data = await make_request(url)
        
        if not data.get("Error") and isinstance(data, list) and len(data) > 0:
            quote = data[0]
            snapshot = {
                "ticker": ticker,
                "price": quote.get("price"),
                "previousClose": quote.get("previousClose"),
                "open": quote.get("open"),
                "dayHigh": quote.get("dayHigh"),
                "dayLow": quote.get("dayLow"),
                "volume": quote.get("volume"),
                "marketCap": quote.get("marketCap"),
                "timestamp": datetime.now().isoformat(),
            }
            logger.info(f"Successfully retrieved current stock price from FMP for {ticker}")
            return json.dumps(snapshot, indent=2)
    
    return json.dumps({"Error": "Unable to fetch current price or no current price found."}, indent=2)

@mcp.tool()
async def get_historical_stock_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    interval_multiplier: int = 1,
) -> str:
    """Gets historical stock prices for a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
        start_date: Start date of the price data (e.g. 2020-01-01)
        end_date: End date of the price data (e.g. 2020-12-31)
        interval: Interval of the price data (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        interval_multiplier: Not used with Yahoo Finance, kept for API compatibility
    """
    logger.info(f"Getting historical prices for {ticker} from {start_date} to {end_date}, interval: {interval}")
    
    # Map interval to Yahoo Finance format
    yf_interval_map = {
        "minute": "1m",
        "hour": "1h",
        "day": "1d",
        "week": "1wk",
        "month": "1mo",
    }
    
    # If the interval is in the original format, convert it
    if interval in yf_interval_map:
        yf_interval = yf_interval_map[interval]
    else:
        # Assume it's already in YF format
        yf_interval = interval
        
    # Try Yahoo Finance (most reliable for historical data)
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date, interval=yf_interval)
        
        if not hist.empty:
            # Convert to the expected output format
            prices = []
            for date, row in hist.iterrows():
                prices.append({
                    "date": date.strftime("%Y-%m-%d") if yf_interval in ["1d", "5d", "1wk", "1mo", "3mo"] else date.strftime("%Y-%m-%d %H:%M:%S"),
                    "open": row.get("Open"),
                    "high": row.get("High"),
                    "low": row.get("Low"),
                    "close": row.get("Close"),
                    "volume": row.get("Volume"),
                })
            
            logger.info(f"Successfully retrieved {len(prices)} historical prices from Yahoo Finance for {ticker}")
            return json.dumps(prices, indent=2)
    except Exception as e:
        logger.error(f"Error retrieving historical prices from Yahoo Finance: {str(e)}")
    
    # Try Alpha Vantage if YF fails and API key is available
    if ALPHA_VANTAGE_API_KEY:
        # Map interval to Alpha Vantage format
        av_interval_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "60m": "60min",
            "1h": "60min",
            "1d": "daily",
            "1wk": "weekly",
            "1mo": "monthly",
        }
        
        av_interval = av_interval_map.get(yf_interval, "daily")
        
        # Determine the right Alpha Vantage function based on interval
        if av_interval in ["1min", "5min", "15min", "30min", "60min"]:
            function = "TIME_SERIES_INTRADAY"
            url = f"{ALPHA_VANTAGE_BASE_URL}?function={function}&symbol={ticker}&interval={av_interval}&apikey={ALPHA_VANTAGE_API_KEY}"
        elif av_interval == "daily":
            function = "TIME_SERIES_DAILY"
            url = f"{ALPHA_VANTAGE_BASE_URL}?function={function}&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        elif av_interval == "weekly":
            function = "TIME_SERIES_WEEKLY"
            url = f"{ALPHA_VANTAGE_BASE_URL}?function={function}&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        else:  # monthly
            function = "TIME_SERIES_MONTHLY"
            url = f"{ALPHA_VANTAGE_BASE_URL}?function={function}&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        
        data = await make_request(url)
        
        if not data.get("Error"):
            # Extract the right time series key
            time_series_key = f"Time Series ({av_interval})" if av_interval in ["1min", "5min", "15min", "30min", "60min"] else {
                "daily": "Time Series (Daily)",
                "weekly": "Weekly Time Series",
                "monthly": "Monthly Time Series"
            }.get(av_interval)
            
            if time_series_key in data:
                time_series = data[time_series_key]
                
                # Filter by date range and format
                prices = []
                for date, values in time_series.items():
                    date_obj = datetime.strptime(date.split(" ")[0], "%Y-%m-%d")
                    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
                    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
                    
                    if start_date_obj <= date_obj <= end_date_obj:
                        prices.append({
                            "date": date,
                            "open": float(values.get("1. open", 0)),
                            "high": float(values.get("2. high", 0)),
                            "low": float(values.get("3. low", 0)),
                            "close": float(values.get("4. close", 0)),
                            "volume": int(values.get("5. volume", 0)),
                        })
                
                if prices:
                    logger.info(f"Successfully retrieved {len(prices)} historical prices from Alpha Vantage for {ticker}")
                    return json.dumps(prices, indent=2)
    
    return json.dumps({"Error": "Unable to fetch historical prices or no prices found."}, indent=2)

@mcp.tool()
async def get_company_news(ticker: str, limit: int = 10) -> str:
    """Get news for a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
        limit: Number of news articles to return (default: 10)
    """
    logger.info(f"Getting news for {ticker}, limit: {limit}")
    
    # Try Yahoo Finance
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if news:
            # # Format the news articles
            # articles = []
            # for article in news[:limit]:
            #     articles.append({
            #         "title": article.get("title"),
            #         "publisher": article.get("publisher"),
            #         "link": article.get("link"),
            #         "published_at": datetime.fromtimestamp(article.get("providerPublishTime", 0)).isoformat() if article.get("providerPublishTime") else None,
            #         "type": article.get("type"),
            #         "thumbnail": article.get("thumbnail", {}).get("resolutions", [{}])[0].get("url") if article.get("thumbnail") else None,
            #     })
            
            logger.info(f"Successfully retrieved {len(news)} news articles from Yahoo Finance for {ticker}")
            return json.dumps(news, indent=2)
    except Exception as e:
        logger.error(f"Error retrieving news from Yahoo Finance: {str(e)}")
    
    # Try FMP as fallback if API key is available
    if FMP_API_KEY:
        url = f"{FMP_BASE_URL}/stock_news?tickers={ticker}&limit={limit}&apikey={FMP_API_KEY}"
        data = await make_request(url)
        
        if not data.get("Error") and isinstance(data, list) and len(data) > 0:
            logger.info(f"Successfully retrieved {len(data)} news articles from FMP for {ticker}")
            return json.dumps(data, indent=2)
    
    return json.dumps({"Error": "Unable to fetch news or no news found."}, indent=2)

@mcp.tool()
async def get_available_crypto_tickers() -> str:
    """
    Gets all available crypto tickers from CoinGecko.
    """
    logger.info("Getting available crypto tickers")
    
    # Use CoinGecko API to get list of all coins
    url = f"{COINGECKO_BASE_URL}/coins/list"
    tickers = await make_request(url)
    
    if tickers and isinstance(tickers, list):
        logger.info(f"Successfully retrieved {len(tickers)} crypto tickers from CoinGecko")
        return json.dumps(tickers, indent=2)
    
    return json.dumps({"Error": "Unable to fetch available crypto tickers."}, indent=2)

@mcp.tool()
async def get_historical_crypto_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "day",
    interval_multiplier: int = 1,
) -> str:
    """Gets historical prices for a crypto currency.

    Args:
        ticker: Ticker symbol of the crypto currency (e.g. BTC-USD)
        start_date: Start date of the price data (e.g. 2020-01-01)
        end_date: End date of the price data (e.g. 2020-12-31)
        interval: Interval of the price data (e.g. day, hour, minute)
        interval_multiplier: Not used with CoinGecko, kept for API compatibility
    """
    logger.info(f"Getting historical crypto prices for {ticker} from {start_date} to {end_date}, interval: {interval}")
    
    # Convert ticker to CoinGecko format
    coin_id = convert_ticker_to_coingecko_id(ticker)
    
    # Convert dates to UNIX timestamps (required by CoinGecko)
    try:
        start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        # Add a day to end_date to include the full day
        end_timestamp = int((datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).timestamp())
    except ValueError as e:
        return json.dumps({"Error": f"Invalid date format: {str(e)}"}, indent=2)
    
    # Map interval to CoinGecko format
    cg_interval_map = {
        "minute": "minutely",
        "hour": "hourly",
        "day": "daily",
    }
    
    cg_interval = cg_interval_map.get(interval, "daily")
    
    # Use CoinGecko API to get historical prices
    url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart/range?vs_currency=usd&from={start_timestamp}&to={end_timestamp}"
    data = await make_request(url)
    
    if not data.get("Error") and "prices" in data:
        prices_data = data["prices"]
        volumes_data = data.get("total_volumes", [])
        market_caps_data = data.get("market_caps", [])
        
        # Create a dict to map timestamp to volume and market cap
        volumes_dict = {int(item[0]/1000): item[1] for item in volumes_data}
        market_caps_dict = {int(item[0]/1000): item[1] for item in market_caps_data}
        
        # Format the prices
        prices = []
        for item in prices_data:
            timestamp = int(item[0] / 1000)  # Convert from milliseconds to seconds
            date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            
            price_obj = {
                "date": date_str,
                "price": item[1],
                "volume": volumes_dict.get(timestamp, None),
                "market_cap": market_caps_dict.get(timestamp, None),
            }
            prices.append(price_obj)
        
        logger.info(f"Successfully retrieved {len(prices)} historical crypto prices from CoinGecko for {ticker}")
        return json.dumps(prices, indent=2)
    
    # Fallback to Yahoo Finance for cryptocurrencies it covers
    try:
        crypto = yf.Ticker(ticker)
        hist = crypto.history(start=start_date, end=end_date)
        
        if not hist.empty:
            # Convert to the expected output format
            prices = []
            for date, row in hist.iterrows():
                prices.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": row.get("Open"),
                    "high": row.get("High"),
                    "low": row.get("Low"),
                    "close": row.get("Close"),
                    "volume": row.get("Volume"),
                })
            
            logger.info(f"Successfully retrieved {len(prices)} historical crypto prices from Yahoo Finance for {ticker}")
            return json.dumps(prices, indent=2)
    except Exception as e:
        logger.error(f"Error retrieving historical crypto prices from Yahoo Finance: {str(e)}")
    
    return json.dumps({"Error": "Unable to fetch historical crypto prices or no prices found."}, indent=2)

@mcp.tool()
async def get_current_crypto_price(ticker: str) -> str:
    """Get the current / latest price of a crypto currency.

    Args:
        ticker: Ticker symbol of the crypto currency (e.g. BTC-USD)
    """
    logger.info(f"Getting current crypto price for {ticker}")

    # Convert ticker to CoinGecko format
    coin_id = convert_ticker_to_coingecko_id(ticker)

    # Use CoinGecko API to get current price
    url = f"{COINGECKO_BASE_URL}/coins/{coin_id}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false"
    data = await make_request(url)

    if not data.get("Error") and "market_data" in data:
        market_data = data["market_data"]

        snapshot = {
            "ticker": ticker,
            "name": data.get("name"),
            "price": market_data.get("current_price", {}).get("usd"),
            "market_cap": market_data.get("market_cap", {}).get("usd"),
            "total_volume": market_data.get("total_volume", {}).get("usd"),
            "high_24h": market_data.get("high_24h", {}).get("usd"),
            "low_24h": market_data.get("low_24h", {}).get("usd"),
            "price_change_24h": market_data.get("price_change_24h"),
            "price_change_percentage_24h": market_data.get("price_change_percentage_24h"),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Successfully retrieved current crypto price from CoinGecko for {ticker}")
        return json.dumps(snapshot, indent=2)

    # Fallback to Yahoo Finance for cryptocurrencies it covers
    try:
        crypto = yf.Ticker(ticker)
        info = crypto.info

        if info:
            snapshot = {
                "ticker": ticker,
                "name": info.get("shortName", ticker),
                "price": info.get("regularMarketPrice"),
                "market_cap": info.get("marketCap"),
                "volume": info.get("regularMarketVolume"),
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(f"Successfully retrieved current crypto price from Yahoo Finance for {ticker}")
            return json.dumps(snapshot, indent=2)
    except Exception as e:
        logger.error(f"Error retrieving current crypto price from Yahoo Finance: {str(e)}")

    return json.dumps({"Error": "Unable to fetch current crypto price or no price found."}, indent=2)


# Import and register financial analysis methods
try:
    import financial_markets_analyser.financial_methods as fin_methods

    # Set up module-level references
    fin_methods.logger = logger
    fin_methods.get_income_statements = get_income_statements
    fin_methods.get_balance_sheets = get_balance_sheets
    fin_methods.get_cash_flow_statements = get_cash_flow_statements
    fin_methods.get_current_stock_price = get_current_stock_price
    fin_methods.get_historical_stock_prices = get_historical_stock_prices
    fin_methods.get_historical_crypto_prices = get_historical_crypto_prices
    fin_methods.get_company_news = get_company_news

    # Register financial analysis tools with MCP
    mcp.tool()(fin_methods.get_financial_ratios)
    mcp.tool()(fin_methods.perform_dcf_valuation)
    mcp.tool()(fin_methods.get_technical_indicators)
    mcp.tool()(fin_methods.analyze_portfolio_risk)
    mcp.tool()(fin_methods.compare_peers)
    mcp.tool()(fin_methods.analyze_news_sentiment)

    logger.info("Successfully registered financial analysis tools")
except Exception as e:
    logger.warning(f"Could not load financial analysis tools: {str(e)}")


# Import and register advanced tools
try:
    import financial_markets_analyser.advanced_market_tools as advanced_tools

    # Set up module-level references
    advanced_tools.logger = logger
    advanced_tools.make_request = make_request
    advanced_tools.make_sync_request = make_sync_request
    advanced_tools.FMP_API_KEY = FMP_API_KEY
    advanced_tools.FMP_BASE_URL = FMP_BASE_URL
    advanced_tools.ALPHA_VANTAGE_API_KEY = ALPHA_VANTAGE_API_KEY
    advanced_tools.ALPHA_VANTAGE_BASE_URL = ALPHA_VANTAGE_BASE_URL
    advanced_tools.get_current_stock_price = get_current_stock_price
    advanced_tools.get_historical_stock_prices = get_historical_stock_prices
    advanced_tools.get_historical_crypto_prices = get_historical_crypto_prices

    # Register advanced tools with MCP
    mcp.tool()(advanced_tools.get_market_indices)
    mcp.tool()(advanced_tools.get_sector_performance)
    mcp.tool()(advanced_tools.get_economic_indicators)
    mcp.tool()(advanced_tools.get_options_chain)
    mcp.tool()(advanced_tools.get_implied_volatility_surface)
    mcp.tool()(advanced_tools.get_insider_transactions)
    mcp.tool()(advanced_tools.get_institutional_ownership)
    mcp.tool()(advanced_tools.get_market_sentiment_indicators)
    mcp.tool()(advanced_tools.get_analyst_ratings)
    mcp.tool()(advanced_tools.get_earnings_calendar)
    mcp.tool()(advanced_tools.get_earnings_history)
    mcp.tool()(advanced_tools.detect_chart_patterns)
    mcp.tool()(advanced_tools.calculate_fibonacci_levels)

    logger.info("Successfully registered advanced market analysis tools")
except Exception as e:
    logger.warning(f"Could not load advanced market tools: {str(e)}")


def main():
    """Main entry point for the server."""
    # Log server startup
    logger.info("Starting Financial Markets Analyser MCP Server")

    # Initialize and run the server
    mcp.run(transport="stdio")

    # This line won't be reached during normal operation
    logger.info("Server stopped")

if __name__ == "__main__":
    main()