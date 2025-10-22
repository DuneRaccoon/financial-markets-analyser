# Financial Markets Analyser

A comprehensive FastMCP server that provides advanced financial market analysis tools for making informed investment decisions. This server combines multiple free data sources with sophisticated analytics to give agents access to institutional-grade market intelligence.

## Features

### 📊 Core Market Data
- **Stock Market Data**: Real-time and historical prices, financial statements (income, balance sheet, cash flow)
- **Cryptocurrency Data**: Current and historical prices for thousands of cryptocurrencies
- **Company News**: Latest news articles with sentiment analysis
- **Multi-Source Fallback System**: Automatically tries multiple data sources to ensure reliability

### 📈 Advanced Market Analysis
- **Market Indices**: Track major indices (S&P 500, NASDAQ, Dow, Russell 2000, VIX, international markets)
- **Sector Performance**: Real-time sector rotation analysis and performance tracking
- **Economic Indicators**: Federal funds rate, CPI/inflation, GDP, unemployment, Treasury yields, yield curve analysis
- **Market Sentiment**: Comprehensive sentiment indicators including VIX, Put/Call ratios, advance/decline, Fear & Greed metrics

### 💼 Fundamental Analysis
- **Financial Ratios**: Profitability, liquidity, leverage, efficiency, and valuation metrics
- **DCF Valuation**: Discounted Cash Flow models with customizable parameters
- **Peer Comparison**: Compare companies against industry peers across key metrics
- **Analyst Ratings**: Consensus ratings, price targets, and recommendation trends
- **Earnings Analysis**: Earnings history, surprises, estimates, and upcoming earnings calendar

### 📉 Technical Analysis
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV with trading signals
- **Chart Pattern Detection**: Identify double tops/bottoms, support/resistance levels
- **Fibonacci Levels**: Calculate retracement and extension levels for trend analysis
- **Trend Analysis**: Automated trend detection and strength measurement

### 🔍 Advanced Market Intelligence
- **Options Data**: Full options chains, implied volatility, IV surface analysis
- **Insider Trading**: Track insider transactions and identify patterns
- **Institutional Ownership**: Monitor institutional holdings and changes
- **Portfolio Risk Analysis**: Calculate Sharpe ratio, VaR, beta, correlation, drawdown analysis
- **News Sentiment**: AI-powered sentiment analysis of company news

## Free Data Sources Used

This package uses a combination of these free financial data APIs:

1. **Yahoo Finance** (via yfinance): No API key required, no rate limits
2. **Alpha Vantage**: Free tier with 5 API calls per minute, 500 calls per day
3. **Financial Modeling Prep (FMP)**: Free tier with ~250-300 API calls per day
4. **CoinGecko**: Free tier with rate limiting (10-50 calls per minute)

## Prerequisites
- Python 3.10 or higher
- [uv](https://pypi.org/project/uv/)

## Configuration and Installation

1. Install UV globally using Homebrew in Terminal (if you haven't already done so):
```bash
brew install uv
```

or with curl
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
curl -LsSf https://astral.sh/uv/install.ps1 | powershell
```

2. Clone and install the repo
```bash
# Clone the repository
git clone https://github.com/duneraccoon/financial-markets-analyser.git
cd financial-markets-analyser

uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
# Install using uv
uv pip install -e .
## or in development mode
# uv pip install -e ".[dev]"
```

3. Install Claude Desktop (if you haven't already done so):
   - Download and install [Claude Desktop](https://www.anthropic.com/claude-desktop) for your OS.
   - Follow the installation instructions provided on the website.

3. Create claude_desktop_config.json (if it doesn't exist):
    - For MacOS: Open directory ~/Library/Application Support/Claude/ and create the file inside it
    - For Windows: Open directory %APPDATA%/Claude/ and create the file inside it

4. Add the server config to the config JSON. Use .env.example as a guide for the env arg:

```json
{
  "mcpServers": {
    "financial-markets-analyser": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/financial-markets-analyser",
        "run",
        "entrypoint.py"
      ],
      "env": {
        "ALPHA_VANTAGE_API_KEY": "your_api_key",
        "FMP_API_KEY": "your_api_secret"
      }
    }
  }
}
```
You can obtain free API keys from:
- Alpha Vantage: https://www.alphavantage.co/support/#api-key
- Financial Modeling Prep: https://site.financialmodelingprep.com/developer/docs/

## Usage

### Starting the MCP Server

```bash
python server.py
```

Or use the installed script:

```bash
financial-markets-analyser
```

### Available Methods

#### Core Stock Market Data

| Method | Description | Parameters |
|--------|-------------|------------|
| `get_income_statements` | Get income statements for a company | `ticker`: Symbol (e.g., AAPL)<br>`period`: "annual" or "quarterly"<br>`limit`: Number of statements (default: 4) |
| `get_balance_sheets` | Get balance sheets for a company | `ticker`: Symbol (e.g., AAPL)<br>`period`: "annual" or "quarterly"<br>`limit`: Number of statements (default: 4) |
| `get_cash_flow_statements` | Get cash flow statements for a company | `ticker`: Symbol (e.g., AAPL)<br>`period`: "annual" or "quarterly"<br>`limit`: Number of statements (default: 4) |
| `get_current_stock_price` | Get latest stock price data | `ticker`: Symbol (e.g., AAPL) |
| `get_historical_stock_prices` | Get historical stock prices | `ticker`: Symbol (e.g., AAPL)<br>`start_date`: Start date (YYYY-MM-DD)<br>`end_date`: End date (YYYY-MM-DD)<br>`interval`: Time interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo) |
| `get_company_news` | Get latest news for a company | `ticker`: Symbol (e.g., AAPL)<br>`limit`: Number of news articles (default: 10) |

#### Cryptocurrency Data

| Method | Description | Parameters |
|--------|-------------|------------|
| `get_available_crypto_tickers` | Get list of available cryptocurrencies | None |
| `get_current_crypto_price` | Get latest price for a cryptocurrency | `ticker`: Symbol (e.g., BTC-USD) |
| `get_historical_crypto_prices` | Get historical cryptocurrency prices | `ticker`: Symbol (e.g., BTC-USD)<br>`start_date`: Start date (YYYY-MM-DD)<br>`end_date`: End date (YYYY-MM-DD)<br>`interval`: Time interval (minute, hour, day) |

#### Market Overview & Indices

| Method | Description | Parameters |
|--------|-------------|------------|
| `get_market_indices` | Get current prices and performance of major market indices (S&P 500, NASDAQ, Dow, VIX, etc.) | None |
| `get_sector_performance` | Get performance of major market sectors | `period`: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, ytd) |
| `get_economic_indicators` | Get key economic indicators (interest rates, inflation, GDP, unemployment, yield curve) | None |
| `get_market_sentiment_indicators` | Get comprehensive market sentiment (VIX, Put/Call ratio, Fear & Greed, advance/decline) | None |

#### Fundamental Analysis

| Method | Description | Parameters |
|--------|-------------|------------|
| `get_financial_ratios` | Calculate comprehensive financial ratios (profitability, liquidity, leverage, efficiency, valuation) | `ticker`: Symbol<br>`period`: "annual" or "quarterly"<br>`limit`: Number of periods (default: 4) |
| `perform_dcf_valuation` | Perform Discounted Cash Flow valuation analysis | `ticker`: Symbol<br>`forecast_years`: Years to forecast (default: 5)<br>`terminal_growth_rate`: Long-term growth rate (default: 0.02)<br>`discount_rate`: WACC or required return (default: 0.09) |
| `compare_peers` | Compare a company with industry peers on key metrics | `ticker`: Symbol<br>`metrics`: List of metrics to compare |
| `get_analyst_ratings` | Get analyst ratings, price targets, and recommendations | `ticker`: Symbol |
| `get_earnings_history` | Get historical earnings with estimates, actuals, and surprises | `ticker`: Symbol<br>`limit`: Number of reports (default: 4) |
| `get_earnings_calendar` | Get upcoming earnings announcements | `from_date`: Start date (default: today)<br>`to_date`: End date (default: +7 days) |

#### Technical Analysis

| Method | Description | Parameters |
|--------|-------------|------------|
| `get_technical_indicators` | Calculate technical indicators (SMA, EMA, RSI, MACD, BB, ATR, OBV) with trading signals | `ticker`: Symbol<br>`start_date`: Start date<br>`end_date`: End date<br>`indicators`: List of indicators |
| `detect_chart_patterns` | Detect chart patterns (double tops/bottoms, support/resistance) | `ticker`: Symbol<br>`start_date`: Start date<br>`end_date`: End date |
| `calculate_fibonacci_levels` | Calculate Fibonacci retracement and extension levels | `ticker`: Symbol<br>`trend_type`: "uptrend" or "downtrend"<br>`lookback_days`: Days to analyze (default: 90) |

#### Options & Derivatives

| Method | Description | Parameters |
|--------|-------------|------------|
| `get_options_chain` | Get full options chain with calls and puts | `ticker`: Symbol<br>`expiration_date`: Specific expiration or None for nearest |
| `get_implied_volatility_surface` | Get IV surface across strikes and expirations | `ticker`: Symbol |

#### Market Intelligence

| Method | Description | Parameters |
|--------|-------------|------------|
| `get_insider_transactions` | Get recent insider trading transactions | `ticker`: Symbol<br>`limit`: Number of transactions (default: 20) |
| `get_institutional_ownership` | Get institutional ownership data | `ticker`: Symbol |
| `analyze_news_sentiment` | Analyze sentiment of recent news articles | `ticker`: Symbol<br>`days`: Lookback period (default: 7) |
| `analyze_portfolio_risk` | Comprehensive portfolio risk analysis (Sharpe, beta, VaR, correlation, diversification) | `tickers`: List of symbols<br>`weights`: Portfolio weights<br>`period`: Analysis period (default: 5y) |

### Example Client Code

```python
import asyncio
from mcp.client.localclient import LocalClient

async def main():
    client = LocalClient(["python", "server.py"])
    await client.start()
    
    # Get current price of Apple stock
    result = await client.call("get_current_stock_price", {"ticker": "AAPL"})
    print(result)
    
    # Get historical prices for Tesla
    result = await client.call(
        "get_historical_stock_prices", 
        {
            "ticker": "TSLA", 
            "start_date": "2023-01-01", 
            "end_date": "2023-12-31",
            "interval": "1mo"
        }
    )
    print(result)
    
    await client.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## Development

### Installing Development Dependencies

```bash
# With uv
uv pip install -e ".[dev]"
```

### Running Tests

```bash
# Run tests with pytest
pytest
```

## API Rate Limits

Be mindful of these rate limits for free tiers:

| API | Rate Limit |
|-----|------------|
| Yahoo Finance | No official limits (use responsibly) |
| Alpha Vantage | 5 API calls per minute, 500 per day |
| Financial Modeling Prep | ~250-300 calls per day, 500MB bandwidth/month |
| CoinGecko | 10-50 calls per minute |

The server implements a fallback system to manage these limits efficiently.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
