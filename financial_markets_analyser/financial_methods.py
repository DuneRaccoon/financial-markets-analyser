
# ------------------------------------------------------------------------------------------------------
# Financial Aanalysis
# ------------------------------------------------------------------------------------------------------

@mcp.tool()
async def get_financial_ratios(
    ticker: str,
    period: str = "annual",
    limit: int = 4,
) -> str:
    """Calculate key financial ratios for investment analysis.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
        period: Period of the calculation (annual or quarterly)
        limit: Number of periods to analyze (default: 4)
    """
    logger.info(f"Calculating financial ratios for {ticker}, period: {period}, limit: {limit}")
    
    try:
        # Get required financial data
        income_data = json.loads(await get_income_statements(ticker, period, limit))
        balance_data = json.loads(await get_balance_sheets(ticker, period, limit))
        cashflow_data = json.loads(await get_cash_flow_statements(ticker, period, limit))
        
        # Handle error responses
        if isinstance(income_data, dict) and income_data.get("Error"):
            return json.dumps({"Error": "Unable to calculate ratios - insufficient income statement data"}, indent=2)
        if isinstance(balance_data, dict) and balance_data.get("Error"):
            return json.dumps({"Error": "Unable to calculate ratios - insufficient balance sheet data"}, indent=2)
        
        # Calculate ratios for each period
        results = []
        
        for i in range(min(len(income_data), len(balance_data), limit)):
            income = income_data[i]
            balance = balance_data[i]
            cashflow = cashflow_data[i] if i < len(cashflow_data) else {}
            
            # Get period date for reference
            fiscal_date = income.get("fiscalDateEnding", balance.get("fiscalDateEnding", "Unknown date"))
            
            # Extract required values, with fallbacks for different API naming conventions
            total_revenue = float(income.get("totalRevenue", income.get("revenue", 0)) or 0)
            net_income = float(income.get("netIncome", income.get("netIncome", 0)) or 0)
            ebit = float(income.get("ebit", income.get("operatingIncome", 0)) or 0)
            total_assets = float(balance.get("totalAssets", 0) or 0)
            total_liabilities = float(balance.get("totalLiabilities", 0) or 0)
            total_equity = float(balance.get("totalShareholderEquity", balance.get("totalEquity", 0)) or 0)
            current_assets = float(balance.get("totalCurrentAssets", 0) or 0)
            current_liabilities = float(balance.get("totalCurrentLiabilities", 0) or 0)
            inventory = float(balance.get("inventory", 0) or 0)
            operating_cash_flow = float(cashflow.get("operatingCashflow", cashflow.get("cashflowFromOperations", 0)) or 0)
            
            # Get current market data
            current_data = json.loads(await get_current_stock_price(ticker))
            current_price = current_data.get("price", 0)
            market_cap = current_data.get("marketCap", 0)
            
            # Calculate ratios
            ratio_data = {
                "ticker": ticker,
                "date": fiscal_date,
                "profitability": {
                    "grossMargin": income.get("grossProfit", 0) / total_revenue if total_revenue else None,
                    "operatingMargin": ebit / total_revenue if total_revenue else None,
                    "netMargin": net_income / total_revenue if total_revenue else None,
                    "ROE": net_income / total_equity if total_equity else None,
                    "ROA": net_income / total_assets if total_assets else None,
                    "ROIC": ebit * (1 - 0.21) / (total_assets - current_liabilities) if (total_assets - current_liabilities) else None,
                },
                "liquidity": {
                    "currentRatio": current_assets / current_liabilities if current_liabilities else None,
                    "quickRatio": (current_assets - inventory) / current_liabilities if current_liabilities else None,
                    "cashRatio": balance.get("cashAndCashEquivalents", 0) / current_liabilities if current_liabilities else None,
                    "operatingCashFlowRatio": operating_cash_flow / current_liabilities if current_liabilities else None,
                },
                "leverage": {
                    "debtToEquity": total_liabilities / total_equity if total_equity else None,
                    "debtToAssets": total_liabilities / total_assets if total_assets else None,
                    "interestCoverage": ebit / income.get("interestExpense", 1) if income.get("interestExpense") else None,
                },
                "efficiency": {
                    "assetTurnover": total_revenue / total_assets if total_assets else None,
                    "inventoryTurnover": income.get("costOfRevenue", 0) / inventory if inventory else None,
                    "receivablesTurnover": total_revenue / balance.get("currentNetReceivables", 1) if balance.get("currentNetReceivables") else None,
                },
                "valuation": {
                    "PE": current_price / (net_income / income.get("commonStockSharesOutstanding", 1)) if net_income and income.get("commonStockSharesOutstanding") else None,
                    "PS": market_cap / total_revenue if total_revenue else None,
                    "PB": market_cap / total_equity if total_equity else None,
                    "EV_EBITDA": (market_cap + float(balance.get("longTermDebt", 0)) - float(balance.get("cashAndCashEquivalents", 0))) / (ebit + float(income.get("depreciation", 0))) if ebit is not None else None,
                    "earningsYield": (net_income / income.get("commonStockSharesOutstanding", 1)) / current_price if current_price and net_income and income.get("commonStockSharesOutstanding") else None,
                    "freeCashFlowYield": (operating_cash_flow - cashflow.get("capitalExpenditures", 0)) / market_cap if market_cap else None,
                }
            }
            
            # Clean up None values for JSON serialization
            ratio_data = {k: {ki: vi if vi is not None else "N/A" for ki, vi in v.items()} if isinstance(v, dict) else v for k, v in ratio_data.items()}
            results.append(ratio_data)
        
        logger.info(f"Successfully calculated financial ratios for {ticker}")
        return json.dumps(results, indent=2)
    
    except Exception as e:
        logger.error(f"Error calculating financial ratios: {str(e)}")
        return json.dumps({"Error": f"Failed to calculate ratios: {str(e)}"}, indent=2)


@mcp.tool()
async def perform_dcf_valuation(
    ticker: str,
    forecast_years: int = 5,
    terminal_growth_rate: float = 0.02,
    discount_rate: float = 0.09,
) -> str:
    """Perform a Discounted Cash Flow (DCF) valuation for a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
        forecast_years: Number of years to forecast (default: 5)
        terminal_growth_rate: Long-term growth rate after forecast period (default: 0.02)
        discount_rate: Weighted Average Cost of Capital (WACC) or required rate of return (default: 0.09)
    """
    logger.info(f"Performing DCF valuation for {ticker}")
    
    try:
        # Get financial data
        income_data = json.loads(await get_income_statements(ticker, "annual", 5))
        balance_data = json.loads(await get_balance_sheets(ticker, "annual", 1))
        cashflow_data = json.loads(await get_cash_flow_statements(ticker, "annual", 5))
        stock_data = json.loads(await get_current_stock_price(ticker))
        
        # Handle error responses
        if isinstance(income_data, dict) and income_data.get("Error"):
            return json.dumps({"Error": "Unable to perform DCF - insufficient income statement data"}, indent=2)
        if isinstance(balance_data, dict) and balance_data.get("Error"):
            return json.dumps({"Error": "Unable to perform DCF - insufficient balance sheet data"}, indent=2)
        
        # Calculate average free cash flow growth rate from historical data
        fcf_values = []
        for cf in cashflow_data:
            operating_cf = float(cf.get("operatingCashflow", cf.get("cashflowFromOperations", 0)) or 0)
            capex = float(cf.get("capitalExpenditures", 0) or 0)
            fcf = operating_cf - abs(capex)
            fcf_values.append(fcf)
        
        # Calculate growth rate if we have enough data points
        if len(fcf_values) >= 2:
            growth_rates = [(fcf_values[i] / fcf_values[i+1]) - 1 for i in range(len(fcf_values)-1)] 
            avg_growth_rate = sum(growth_rates) / len(growth_rates)
            # Cap growth rate to be reasonable
            avg_growth_rate = min(max(avg_growth_rate, 0.01), 0.30)
        else:
            # Default growth rate if insufficient data
            avg_growth_rate = 0.05
        
        # Use most recent FCF as base
        base_fcf = fcf_values[0] if fcf_values else 0
        
        # If FCF is negative or very low, use operating cashflow with a conservative estimate
        if base_fcf <= 0 and len(cashflow_data) > 0:
            base_fcf = float(cashflow_data[0].get("operatingCashflow", 0) or 0) * 0.7
        
        # If still no valid data, return error
        if base_fcf <= 0:
            return json.dumps({"Error": "Cannot perform DCF with negative or zero free cash flow"}, indent=2)
        
        # Now project future cash flows
        projected_fcf = []
        for year in range(1, forecast_years + 1):
            # Apply declining growth rate model (reverts to terminal growth rate)
            year_growth = avg_growth_rate - ((avg_growth_rate - terminal_growth_rate) * (year / forecast_years))
            year_fcf = base_fcf * (1 + year_growth) ** year
            pv_factor = 1 / ((1 + discount_rate) ** year)
            projected_fcf.append({
                "year": year,
                "fcf": year_fcf,
                "growth_rate": year_growth,
                "present_value": year_fcf * pv_factor
            })
        
        # Calculate terminal value
        terminal_fcf = projected_fcf[-1]["fcf"] * (1 + terminal_growth_rate)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
        discounted_terminal_value = terminal_value / ((1 + discount_rate) ** forecast_years)
        
        # Sum PV of projected FCF and terminal value
        sum_pv_fcf = sum(year["present_value"] for year in projected_fcf)
        enterprise_value = sum_pv_fcf + discounted_terminal_value
        
        # Get balance sheet items for equity value calculation
        cash = float(balance_data[0].get("cashAndCashEquivalents", 0) or 0) if isinstance(balance_data, list) else 0
        total_debt = float(balance_data[0].get("shortTermDebt", 0) or 0) + float(balance_data[0].get("longTermDebt", 0) or 0) if isinstance(balance_data, list) else 0
        
        # Calculate equity value
        equity_value = enterprise_value + cash - total_debt
        
        # Get shares outstanding
        shares_outstanding = float(income_data[0].get("commonStockSharesOutstanding", stock_data.get("marketCap", 0) / stock_data.get("price", 1)) or 0)
        
        # Calculate fair value per share
        fair_value_per_share = equity_value / shares_outstanding if shares_outstanding else 0
        
        # Current stock price for comparison
        current_price = stock_data.get("price", 0)
        
        # Calculate upside/downside and margin of safety
        upside_pct = ((fair_value_per_share / current_price) - 1) * 100 if current_price else 0
        margin_of_safety = (1 - (current_price / fair_value_per_share)) * 100 if fair_value_per_share > 0 else 0
        
        # Prepare result
        result = {
            "ticker": ticker,
            "dcf_valuation": {
                "inputs": {
                    "base_fcf": base_fcf,
                    "historical_growth_rate": avg_growth_rate,
                    "forecast_years": forecast_years,
                    "terminal_growth_rate": terminal_growth_rate,
                    "discount_rate": discount_rate
                },
                "projected_cash_flows": projected_fcf,
                "terminal_value": {
                    "terminal_fcf": terminal_fcf,
                    "terminal_value": terminal_value,
                    "discounted_terminal_value": discounted_terminal_value
                },
                "valuation_summary": {
                    "sum_pv_projected_fcf": sum_pv_fcf,
                    "enterprise_value": enterprise_value,
                    "cash": cash,
                    "total_debt": total_debt,
                    "equity_value": equity_value,
                    "shares_outstanding": shares_outstanding,
                    "fair_value_per_share": fair_value_per_share,
                    "current_share_price": current_price,
                    "upside_downside_percent": upside_pct,
                    "margin_of_safety_percent": margin_of_safety
                },
                "valuation_rating": "Undervalued" if upside_pct > 15 else ("Fair Valued" if abs(upside_pct) <= 15 else "Overvalued")
            }
        }
        
        logger.info(f"Successfully performed DCF valuation for {ticker}")
        return json.dumps(result, indent=2)
    
    except Exception as e:
        logger.error(f"Error performing DCF valuation: {str(e)}")
        return json.dumps({"Error": f"Failed to perform DCF valuation: {str(e)}"}, indent=2)


@mcp.tool()
async def get_technical_indicators(
    ticker: str,
    start_date: str,
    end_date: str = None,
    indicators: List[str] = None,
) -> str:
    """Calculate technical indicators for a stock or cryptocurrency.

    Args:
        ticker: Ticker symbol of the security (e.g. AAPL, BTC-USD)
        start_date: Start date for the analysis (e.g. 2020-01-01)
        end_date: End date for the analysis (default: today)
        indicators: List of indicators to calculate (default: all available)
    """
    logger.info(f"Calculating technical indicators for {ticker}")
    
    try:
        import numpy as np
        import pandas as pd
        from datetime import datetime
        
        # Set default end date to today if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Set default indicators if not provided
        if not indicators:
            indicators = ["SMA", "EMA", "RSI", "MACD", "BB", "ATR", "OBV"]
        
        # Get historical prices
        is_crypto = "-" in ticker
        if is_crypto:
            history_data = json.loads(await get_historical_crypto_prices(ticker, start_date, end_date, "day"))
        else:
            history_data = json.loads(await get_historical_stock_prices(ticker, start_date, end_date, "1d"))
        
        # Handle error response
        if isinstance(history_data, dict) and history_data.get("Error"):
            return json.dumps({"Error": f"Unable to calculate technical indicators - {history_data.get('Error')}"}, indent=2)
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(history_data)
        
        # Rename and ensure columns for consistent processing
        if is_crypto:
            if "price" in df.columns:
                df["close"] = df["price"]
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        else:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            # Ensure lower case column names
            df.columns = [col.lower() for col in df.columns]
        
        # Ensure we have required price columns
        required_columns = ["close", "high", "low", "volume"]
        for col in required_columns:
            if col not in df.columns and col == "close" and "price" in df.columns:
                df["close"] = df["price"]
            elif col not in df.columns:
                df[col] = df["close"]  # Use close as a substitute if missing
        
        # Make sure data is sorted chronologically
        df.sort_index(inplace=True)
        
        # Calculate requested indicators
        result = {"ticker": ticker, "indicators": {}}
        
        # 1. Simple Moving Averages (SMA)
        if "SMA" in indicators:
            result["indicators"]["SMA"] = {
                "SMA20": df["close"].rolling(window=20).mean().dropna().tolist(),
                "SMA50": df["close"].rolling(window=50).mean().dropna().tolist(),
                "SMA200": df["close"].rolling(window=200).mean().dropna().tolist(),
            }
        
        # 2. Exponential Moving Averages (EMA)
        if "EMA" in indicators:
            result["indicators"]["EMA"] = {
                "EMA12": df["close"].ewm(span=12, adjust=False).mean().dropna().tolist(),
                "EMA26": df["close"].ewm(span=26, adjust=False).mean().dropna().tolist(),
                "EMA50": df["close"].ewm(span=50, adjust=False).mean().dropna().tolist(),
            }
        
        # 3. Relative Strength Index (RSI)
        if "RSI" in indicators:
            delta = df["close"].diff().dropna()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            result["indicators"]["RSI"] = {
                "RSI14": rsi.dropna().tolist(),
            }
        
        # 4. Moving Average Convergence Divergence (MACD)
        if "MACD" in indicators:
            ema12 = df["close"].ewm(span=12, adjust=False).mean()
            ema26 = df["close"].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line
            result["indicators"]["MACD"] = {
                "MACD_line": macd_line.dropna().tolist(),
                "signal_line": signal_line.dropna().tolist(),
                "histogram": histogram.dropna().tolist(),
            }
        
        # 5. Bollinger Bands (BB)
        if "BB" in indicators:
            sma20 = df["close"].rolling(window=20).mean()
            std20 = df["close"].rolling(window=20).std()
            upper_band = sma20 + (2 * std20)
            lower_band = sma20 - (2 * std20)
            result["indicators"]["BB"] = {
                "middle_band": sma20.dropna().tolist(),
                "upper_band": upper_band.dropna().tolist(),
                "lower_band": lower_band.dropna().tolist(),
            }
        
        # 6. Average True Range (ATR)
        if "ATR" in indicators:
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift())
            low_close = np.abs(df["low"] - df["close"].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(window=14).mean()
            result["indicators"]["ATR"] = {
                "ATR14": atr.dropna().tolist(),
            }
        
        # 7. On-Balance Volume (OBV)
        if "OBV" in indicators:
            obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
            result["indicators"]["OBV"] = {
                "OBV": obv.dropna().tolist(),
            }
        
        # Include reference dates for each indicator
        result["dates"] = df.index.strftime("%Y-%m-%d").tolist()
        result["prices"] = df["close"].tolist()
        
        # Generate signals and insights
        signals = []
        latest_close = df["close"].iloc[-1]
        
        # SMA signals
        if "SMA" in indicators and len(result["indicators"]["SMA"]["SMA20"]) > 0 and len(result["indicators"]["SMA"]["SMA50"]) > 0:
            sma20_latest = result["indicators"]["SMA"]["SMA20"][-1]
            sma50_latest = result["indicators"]["SMA"]["SMA50"][-1]
            sma20_prev = result["indicators"]["SMA"]["SMA20"][-2] if len(result["indicators"]["SMA"]["SMA20"]) > 1 else None
            sma50_prev = result["indicators"]["SMA"]["SMA50"][-2] if len(result["indicators"]["SMA"]["SMA50"]) > 1 else None
            
            if sma20_latest > sma50_latest and sma20_prev and sma50_prev and sma20_prev <= sma50_prev:
                signals.append("Golden Cross detected: SMA20 crossed above SMA50 (bullish)")
            elif sma20_latest < sma50_latest and sma20_prev and sma50_prev and sma20_prev >= sma50_prev:
                signals.append("Death Cross detected: SMA20 crossed below SMA50 (bearish)")
            
            if latest_close > sma200[-1] if "SMA200" in result["indicators"]["SMA"] and len(result["indicators"]["SMA"]["SMA200"]) > 0 else 0:
                signals.append("Price above SMA200 (bullish long-term trend)")
            else:
                signals.append("Price below SMA200 (bearish long-term trend)")
        
        # RSI signals
        if "RSI" in indicators and len(result["indicators"]["RSI"]["RSI14"]) > 0:
            rsi_latest = result["indicators"]["RSI"]["RSI14"][-1]
            if rsi_latest > 70:
                signals.append(f"RSI is overbought at {rsi_latest:.2f} (potential reversal)")
            elif rsi_latest < 30:
                signals.append(f"RSI is oversold at {rsi_latest:.2f} (potential reversal)")
        
        # MACD signals
        if "MACD" in indicators and len(result["indicators"]["MACD"]["MACD_line"]) > 1 and len(result["indicators"]["MACD"]["signal_line"]) > 1:
            macd_latest = result["indicators"]["MACD"]["MACD_line"][-1]
            signal_latest = result["indicators"]["MACD"]["signal_line"][-1]
            macd_prev = result["indicators"]["MACD"]["MACD_line"][-2]
            signal_prev = result["indicators"]["MACD"]["signal_line"][-2]
            
            if macd_latest > signal_latest and macd_prev <= signal_prev:
                signals.append("MACD crossed above signal line (bullish)")
            elif macd_latest < signal_latest and macd_prev >= signal_prev:
                signals.append("MACD crossed below signal line (bearish)")
        
        # Bollinger Band signals
        if "BB" in indicators and len(result["indicators"]["BB"]["upper_band"]) > 0 and len(result["indicators"]["BB"]["lower_band"]) > 0:
            upper_latest = result["indicators"]["BB"]["upper_band"][-1]
            lower_latest = result["indicators"]["BB"]["lower_band"][-1]
            
            if latest_close > upper_latest:
                signals.append("Price above upper Bollinger Band (overbought)")
            elif latest_close < lower_latest:
                signals.append("Price below lower Bollinger Band (oversold)")
        
        # Add signals to result
        result["signals"] = signals
        
        logger.info(f"Successfully calculated technical indicators for {ticker}")
        return json.dumps(result, indent=2)
    
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        return json.dumps({"Error": f"Failed to calculate technical indicators: {str(e)}"}, indent=2)


@mcp.tool()
async def analyze_portfolio_risk(
    tickers: List[str],
    weights: Optional[List[float]] = None,
    period: str = "5y",
) -> str:
    """Analyze risk metrics for a portfolio of securities.

    Args:
        tickers: List of ticker symbols (e.g. ["AAPL", "GOOGL", "MSFT"])
        weights: List of portfolio weights (default: equal weighting)
        period: Lookback period for risk calculation (default: 5y)
    """
    logger.info(f"Analyzing portfolio risk for {tickers}")
    
    try:
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Validate inputs
        if not tickers:
            return json.dumps({"Error": "No tickers provided"}, indent=2)
        
        # Default to equal weighting if weights not provided
        if not weights or len(weights) != len(tickers):
            weights = [1.0 / len(tickers)] * len(tickers)
        
        # Normalize weights to ensure they sum to 1
        total_weight = sum(weights)
        if total_weight != 1.0:
            weights = [w / total_weight for w in weights]
        
        # Calculate end date (today) and start date based on period
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Map period string to days
        period_map = {
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "3y": 3 * 365,
            "5y": 5 * 365,
            "10y": 10 * 365,
        }
        days_back = period_map.get(period, 5 * 365)  # Default to 5 years
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        # Get historical prices for each ticker
        price_data = {}
        for ticker in tickers:
            # Determine if it's crypto or stock
            is_crypto = "-" in ticker
            
            if is_crypto:
                history = json.loads(await get_historical_crypto_prices(ticker, start_date, end_date, "day"))
            else:
                history = json.loads(await get_historical_stock_prices(ticker, start_date, end_date, "1d"))
            
            # Handle error
            if isinstance(history, dict) and history.get("Error"):
                return json.dumps({"Error": f"Unable to get price data for {ticker}: {history.get('Error')}"}, indent=2)
            
            # Extract date and closing price
            dates = []
            prices = []
            for item in history:
                if is_crypto:
                    dates.append(item.get("date").split(" ")[0])  # Just the date part
                    prices.append(item.get("price", 0))
                else:
                    dates.append(item.get("date"))
                    prices.append(item.get("close", 0))
            
            # Save data
            price_data[ticker] = {"dates": dates, "prices": prices}
        
        # Create a unified DataFrame with all securities' prices
        all_dates = set()
        for ticker in price_data:
            all_dates.update(price_data[ticker]["dates"])
        
        all_dates = sorted(list(all_dates))
        
        # Initialize DataFrame with dates
        df = pd.DataFrame(index=all_dates)
        
        # Add price data for each ticker
        for ticker in price_data:
            # Create a Series with dates as index and prices as values
            series = pd.Series(index=price_data[ticker]["dates"], data=price_data[ticker]["prices"])
            # Add to main DataFrame
            df[ticker] = series
        
        # Ensure chronological order
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        # Forward fill missing values (use previous day's price)
        df.fillna(method="ffill", inplace=True)
        
        # Calculate daily returns
        returns = df.pct_change().dropna()
        
        # Calculate portfolio metrics
        # 1. Expected returns (annualized)
        expected_returns = returns.mean() * 252  # 252 trading days in a year
        portfolio_return = sum(expected_returns * weights)
        
        # 2. Covariance matrix and portfolio volatility
        cov_matrix = returns.cov() * 252  # Annualized
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # 3. Sharpe Ratio (assuming risk-free rate of 0.03)
        risk_free_rate = 0.03
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        # 4. Individual security volatilities
        volatilities = returns.std() * np.sqrt(252)
        
        # 5. Beta calculation (using S&P 500 as benchmark if available)
        benchmark_ticker = "SPY"
        try:
            benchmark_data = json.loads(await get_historical_stock_prices(benchmark_ticker, start_date, end_date, "1d"))
            benchmark_dates = [item.get("date") for item in benchmark_data]
            benchmark_prices = [item.get("close", 0) for item in benchmark_data]
            
            # Create Series and align with our DataFrame's dates
            benchmark_series = pd.Series(index=benchmark_dates, data=benchmark_prices)
            benchmark_series.index = pd.to_datetime(benchmark_series.index)
            benchmark_series = benchmark_series.reindex(df.index)
            
            # Calculate benchmark returns
            benchmark_returns = benchmark_series.pct_change().dropna()
            
            # Calculate betas for each security
            betas = {}
            for ticker in tickers:
                # Align returns
                aligned_returns = returns[[ticker]].dropna()
                aligned_benchmark = benchmark_returns.reindex(aligned_returns.index).dropna()
                aligned_returns = aligned_returns.reindex(aligned_benchmark.index)
                
                # Calculate beta
                covariance = aligned_returns[ticker].cov(aligned_benchmark)
                benchmark_variance = aligned_benchmark.var()
                beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
                betas[ticker] = beta
            
            # Calculate portfolio beta
            portfolio_beta = sum(betas.get(ticker, 0) * weights[i] for i, ticker in enumerate(tickers))
        except Exception as e:
            logger.warning(f"Unable to calculate beta against {benchmark_ticker}: {str(e)}")
            betas = {ticker: "N/A" for ticker in tickers}
            portfolio_beta = "N/A"
        
        # 6. Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns) / rolling_max - 1
        max_drawdowns = drawdowns.min()
        
        # Calculate portfolio drawdown
        portfolio_returns = returns.dot(weights)
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        portfolio_rolling_max = portfolio_cumulative.cummax()
        portfolio_drawdowns = portfolio_cumulative / portfolio_rolling_max - 1
        portfolio_max_drawdown = portfolio_drawdowns.min()
        
        # 7. Value at Risk (VaR) - 95% confidence
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(1)  # 1-day VaR
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()  # Conditional VaR
        
        # 8. Calculate correlation matrix
        correlation_matrix = returns.corr().to_dict()
        
        # Prepare the result
        result = {
            "portfolio": {
                "tickers": tickers,
                "weights": weights,
                "period": period,
                "metrics": {
                    "expected_annual_return": portfolio_return,
                    "annual_volatility": portfolio_volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "beta": portfolio_beta,
                    "max_drawdown": portfolio_max_drawdown,
                    "value_at_risk_95": var_95,
                    "conditional_var_95": cvar_95
                }
            },
            "securities": {
                ticker: {
                    "weight": weights[i],
                    "metrics": {
                        "expected_annual_return": expected_returns[ticker],
                        "annual_volatility": volatilities[ticker],
                        "beta": betas.get(ticker, "N/A"),
                        "max_drawdown": max_drawdowns[ticker],
                        "contribution_to_risk": weights[i] * volatilities[ticker] / portfolio_volatility if portfolio_volatility else 0
                    }
                } for i, ticker in enumerate(tickers)
            },
            "correlation_matrix": correlation_matrix
        }
        
        # Risk assessment
        risk_level = "Low"
        if portfolio_volatility > 0.25:
            risk_level = "Very High"
        elif portfolio_volatility > 0.20:
            risk_level = "High"
        elif portfolio_volatility > 0.15:
            risk_level = "Medium-High"
        elif portfolio_volatility > 0.10:
            risk_level = "Medium"
        elif portfolio_volatility > 0.05:
            risk_level = "Medium-Low"
        
        result["portfolio"]["risk_assessment"] = {
            "risk_level": risk_level,
            "diversification_score": 1 - np.sqrt(sum([w**2 for w in weights])),  # Closer to 1 is better diversified
            "recommendations": []
        }
        
        # Add recommendations based on analysis
        recommendations = []
        
        # Diversification recommendation
        if result["portfolio"]["risk_assessment"]["diversification_score"] < 0.5:
            recommendations.append("Portfolio is not well-diversified. Consider adding more uncorrelated assets.")
        
        # Correlation-based recommendation
        high_correlations = False
        for i, ticker1 in enumerate(tickers):
            for j, ticker2 in enumerate(tickers):
                if i != j and correlation_matrix[ticker1][ticker2] > 0.8:
                    high_correlations = True
                    recommendations.append(f"High correlation ({correlation_matrix[ticker1][ticker2]:.2f}) between {ticker1} and {ticker2}. Consider replacing one to improve diversification.")
        
        # Risk-return recommendation
        if portfolio_return / portfolio_volatility < 0.5:
            recommendations.append("Low risk-adjusted return. Consider rebalancing to improve the risk-return profile.")
        
        # Beta recommendation
        if portfolio_beta != "N/A" and portfolio_beta > 1.2:
            recommendations.append(f"Portfolio beta ({portfolio_beta:.2f}) is high, indicating higher sensitivity to market movements. Consider reducing exposure to high-beta stocks for more stability.")
        
        # Max drawdown recommendation
        if portfolio_max_drawdown < -0.3:
            recommendations.append(f"High maximum drawdown ({portfolio_max_drawdown:.2%}). Consider adding defensive assets to reduce drawdown risk.")
        
        result["portfolio"]["risk_assessment"]["recommendations"] = recommendations
        
        logger.info(f"Successfully analyzed portfolio risk for {tickers}")
        return json.dumps(result, indent=2)
    
    except Exception as e:
        logger.error(f"Error analyzing portfolio risk: {str(e)}")
        return json.dumps({"Error": f"Failed to analyze portfolio risk: {str(e)}"}, indent=2)


@mcp.tool()
async def compare_peers(
    ticker: str,
    metrics: Optional[List[str]] = None,
) -> str:
    """Compare a company with its industry peers on key financial metrics.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
        metrics: List of metrics to compare (default: all key metrics)
    """
    logger.info(f"Comparing {ticker} with industry peers")
    
    try:
        # Define peer groups manually or fetch from an API if available
        peer_groups = {
            # Technology
            "AAPL": ["MSFT", "GOOGL", "AMZN", "META", "AAPL"],
            "MSFT": ["AAPL", "GOOGL", "AMZN", "META", "MSFT"],
            "GOOGL": ["AAPL", "MSFT", "AMZN", "META", "GOOGL"],
            "GOOG": ["AAPL", "MSFT", "AMZN", "META", "GOOG"],
            "META": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "AMZN": ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
            
            # Semiconductors
            "NVDA": ["AMD", "INTC", "TSM", "MU", "NVDA"],
            "AMD": ["NVDA", "INTC", "TSM", "MU", "AMD"],
            "INTC": ["NVDA", "AMD", "TSM", "MU", "INTC"],
            
            # Banking & Finance
            "JPM": ["BAC", "C", "WFC", "GS", "JPM"],
            "BAC": ["JPM", "C", "WFC", "GS", "BAC"],
            "GS": ["JPM", "BAC", "C", "MS", "GS"],
            
            # Automotive
            "TSLA": ["F", "GM", "TM", "RIVN", "TSLA"],
            "F": ["GM", "TM", "TSLA", "STLA", "F"],
            
            # Retail
            "WMT": ["TGT", "COST", "AMZN", "KR", "WMT"],
            "TGT": ["WMT", "COST", "AMZN", "KR", "TGT"],
            
            # Healthcare
            "JNJ": ["PFE", "MRK", "ABT", "UNH", "JNJ"],
            "PFE": ["JNJ", "MRK", "ABT", "MRNA", "PFE"],
            
            # Oil & Gas
            "XOM": ["CVX", "BP", "SHEL", "COP", "XOM"],
            "CVX": ["XOM", "BP", "SHEL", "COP", "CVX"],
        }
        
        # Default metrics if not specified
        if not metrics:
            metrics = [
                "Revenue Growth", "Profit Margin", "P/E Ratio", "EV/EBITDA", 
                "ROE", "ROA", "Debt to Equity", "Current Ratio", "Dividend Yield"
            ]
        
        # Get peer group or use sector average if peer group not defined
        if ticker in peer_groups:
            peers = peer_groups[ticker]
        else:
            # If peer group not defined, get current data and use a generic set of peers
            ticker_data = json.loads(await get_current_stock_price(ticker))
            if isinstance(ticker_data, dict) and not ticker_data.get("Error"):
                # Use a default set of large index stocks as peers
                peers = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ", "PG", "V"]
                # Make sure the ticker itself is included
                if ticker not in peers:
                    peers.append(ticker)
            else:
                return json.dumps({"Error": f"Unable to determine peer group for {ticker}"}, indent=2)
        
        # Collect metrics for all peers
        results = {"ticker": ticker, "peer_comparison": {}}
        
        for peer in peers:
            # Get financial ratios
            ratios_data = json.loads(await get_financial_ratios(peer, "annual", 1))
            
            # Skip if error
            if isinstance(ratios_data, dict) and ratios_data.get("Error"):
                logger.warning(f"Unable to get financial ratios for {peer}: {ratios_data.get('Error')}")
                continue
            
            # Get current price data
            price_data = json.loads(await get_current_stock_price(peer))
            
            # Skip if error
            if isinstance(price_data, dict) and price_data.get("Error"):
                logger.warning(f"Unable to get current price for {peer}: {price_data.get('Error')}")
                continue
            
            # Extract the latest data point
            latest_ratios = ratios_data[0] if isinstance(ratios_data, list) and len(ratios_data) > 0 else {}
            
            # Create a metrics object for this peer
            peer_metrics = {
                "ticker": peer,
                "current_price": price_data.get("price", 0),
                "market_cap": price_data.get("marketCap", 0),
                "metrics": {}
            }
            
            # Extract relevant metrics based on the requested list
            for metric in metrics:
                if metric == "Revenue Growth":
                    # Would need income statements from multiple periods to calculate growth
                    income_data = json.loads(await get_income_statements(peer, "annual", 2))
                    if isinstance(income_data, list) and len(income_data) >= 2:
                        current_revenue = float(income_data[0].get("totalRevenue", income_data[0].get("revenue", 0)) or 0)
                        prev_revenue = float(income_data[1].get("totalRevenue", income_data[1].get("revenue", 0)) or 0)
                        growth = (current_revenue - prev_revenue) / prev_revenue if prev_revenue else 0
                        peer_metrics["metrics"][metric] = growth
                    else:
                        peer_metrics["metrics"][metric] = "N/A"
                
                elif metric == "Profit Margin":
                    if isinstance(latest_ratios, dict) and "profitability" in latest_ratios:
                        peer_metrics["metrics"][metric] = latest_ratios["profitability"].get("netMargin", "N/A")
                    else:
                        peer_metrics["metrics"][metric] = "N/A"
                
                elif metric == "P/E Ratio":
                    if isinstance(latest_ratios, dict) and "valuation" in latest_ratios:
                        peer_metrics["metrics"][metric] = latest_ratios["valuation"].get("PE", "N/A")
                    else:
                        peer_metrics["metrics"][metric] = "N/A"
                
                elif metric == "EV/EBITDA":
                    if isinstance(latest_ratios, dict) and "valuation" in latest_ratios:
                        peer_metrics["metrics"][metric] = latest_ratios["valuation"].get("EV_EBITDA", "N/A")
                    else:
                        peer_metrics["metrics"][metric] = "N/A"
                
                elif metric == "ROE":
                    if isinstance(latest_ratios, dict) and "profitability" in latest_ratios:
                        peer_metrics["metrics"][metric] = latest_ratios["profitability"].get("ROE", "N/A")
                    else:
                        peer_metrics["metrics"][metric] = "N/A"
                
                elif metric == "ROA":
                    if isinstance(latest_ratios, dict) and "profitability" in latest_ratios:
                        peer_metrics["metrics"][metric] = latest_ratios["profitability"].get("ROA", "N/A")
                    else:
                        peer_metrics["metrics"][metric] = "N/A"
                
                elif metric == "Debt to Equity":
                    if isinstance(latest_ratios, dict) and "leverage" in latest_ratios:
                        peer_metrics["metrics"][metric] = latest_ratios["leverage"].get("debtToEquity", "N/A")
                    else:
                        peer_metrics["metrics"][metric] = "N/A"
                
                elif metric == "Current Ratio":
                    if isinstance(latest_ratios, dict) and "liquidity" in latest_ratios:
                        peer_metrics["metrics"][metric] = latest_ratios["liquidity"].get("currentRatio", "N/A")
                    else:
                        peer_metrics["metrics"][metric] = "N/A"
                
                elif metric == "Dividend Yield":
                    # Would need to get this from current price data or another endpoint
                    peer_metrics["metrics"][metric] = "N/A"
            
            # Add to results
            results["peer_comparison"][peer] = peer_metrics
        
        # Calculate industry averages
        industry_avgs = {"ticker": "Industry Average", "metrics": {}}
        
        for metric in metrics:
            values = []
            for peer, data in results["peer_comparison"].items():
                metric_value = data["metrics"].get(metric, "N/A")
                if metric_value != "N/A" and metric_value is not None:
                    try:
                        values.append(float(metric_value))
                    except (ValueError, TypeError):
                        pass
            
            # Calculate average if we have values
            if values:
                industry_avgs["metrics"][metric] = sum(values) / len(values)
            else:
                industry_avgs["metrics"][metric] = "N/A"
        
        # Add industry average to results
        results["industry_average"] = industry_avgs
        
        # Generate insights and rankings
        insights = []
        rankings = {}
        
        for metric in metrics:
            # Create a ranking for this metric
            valid_peers = []
            for peer, data in results["peer_comparison"].items():
                metric_value = data["metrics"].get(metric, "N/A")
                if metric_value != "N/A" and metric_value is not None:
                    try:
                        valid_peers.append((peer, float(metric_value)))
                    except (ValueError, TypeError):
                        pass
            
            # Sort based on metric (some metrics higher is better, some lower is better)
            if metric in ["Revenue Growth", "Profit Margin", "ROE", "ROA", "Current Ratio"]:
                # Higher is better
                valid_peers.sort(key=lambda x: x[1], reverse=True)
            elif metric in ["P/E Ratio", "EV/EBITDA", "Debt to Equity"]:
                # Lower is better
                valid_peers.sort(key=lambda x: x[1])
            elif metric == "Dividend Yield":
                # Higher usually better, but context matters
                valid_peers.sort(key=lambda x: x[1], reverse=True)
            
            # Create the ranking
            rankings[metric] = {peer: idx + 1 for idx, (peer, _) in enumerate(valid_peers)}
            
            # Generate insights for the main ticker
            if ticker in rankings[metric]:
                rank = rankings[metric][ticker]
                percentile = (len(valid_peers) - rank + 1) / len(valid_peers) * 100 if len(valid_peers) > 1 else 50
                
                insight = f"{ticker} ranks {rank} out of {len(valid_peers)} peers for {metric} "
                if metric in ["Revenue Growth", "Profit Margin", "ROE", "ROA", "Current Ratio"]:
                    if percentile > 75:
                        insight += f"(top {percentile:.1f}%, excellent)"
                    elif percentile > 50:
                        insight += f"(top {percentile:.1f}%, above average)"
                    elif percentile > 25:
                        insight += f"(bottom {100-percentile:.1f}%, below average)"
                    else:
                        insight += f"(bottom {100-percentile:.1f}%, poor)"
                elif metric in ["P/E Ratio", "EV/EBITDA", "Debt to Equity"]:
                    if percentile > 75:
                        insight += f"(bottom {100-percentile:.1f}%, potentially overvalued)"
                    elif percentile > 50:
                        insight += f"(bottom {100-percentile:.1f}%, slightly expensive)"
                    elif percentile > 25:
                        insight += f"(top {percentile:.1f}%, reasonably valued)"
                    else:
                        insight += f"(top {percentile:.1f}%, potentially undervalued)"
                
                insights.append(insight)
        
        # Add overall ranking across all metrics
        overall_scores = {}
        for peer in results["peer_comparison"]:
            score = 0
            count = 0
            for metric, ranks in rankings.items():
                if peer in ranks:
                    # Normalize the score (1 is best, 0 is worst)
                    normalized_score = 1 - ((ranks[peer] - 1) / (len(results["peer_comparison"]) - 1)) if len(results["peer_comparison"]) > 1 else 0.5
                    score += normalized_score
                    count += 1
            
            if count > 0:
                overall_scores[peer] = score / count
        
        # Sort by overall score
        sorted_peers = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        results["overall_ranking"] = {peer: idx + 1 for idx, (peer, _) in enumerate(sorted_peers)}
        
        # Add overall insight
        if ticker in results["overall_ranking"]:
            overall_rank = results["overall_ranking"][ticker]
            total_peers = len(results["overall_ranking"])
            percentile = (total_peers - overall_rank + 1) / total_peers * 100 if total_peers > 1 else 50
            
            overall_insight = f"{ticker} ranks {overall_rank} out of {total_peers} peers overall "
            if percentile > 75:
                overall_insight += f"(top {percentile:.1f}%, excellent relative performance)"
            elif percentile > 50:
                overall_insight += f"(top {percentile:.1f}%, above average performance)"
            elif percentile > 25:
                overall_insight += f"(bottom {100-percentile:.1f}%, below average performance)"
            else:
                overall_insight += f"(bottom {100-percentile:.1f}%, poor relative performance)"
            
            insights.insert(0, overall_insight)
        
        # Add insights to results
        results["insights"] = insights
        
        logger.info(f"Successfully compared {ticker} with peers")
        return json.dumps(results, indent=2)
    
    except Exception as e:
        logger.error(f"Error comparing peers: {str(e)}")
        return json.dumps({"Error": f"Failed to compare peers: {str(e)}"}, indent=2)


@mcp.tool()
async def analyze_news_sentiment(
    ticker: str,
    days: int = 7,
) -> str:
    """Analyze sentiment of recent news articles for a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
        days: Number of days to look back for news (default: 7)
    """
    logger.info(f"Analyzing news sentiment for {ticker}, days: {days}")
    
    try:
        # Get company news
        news_data = json.loads(await get_company_news(ticker, limit=20))
        
        # Handle error response
        if isinstance(news_data, dict) and news_data.get("Error"):
            return json.dumps({"Error": f"Unable to get news: {news_data.get('Error')}"}, indent=2)
        
        # Define sentiment analysis function (basic version without external API)
        def analyze_sentiment(title, summary=None):
            # Define positive and negative word lists
            positive_words = [
                "up", "rise", "rising", "rose", "gain", "gains", "positive", "profit", "profits", 
                "growth", "grow", "grew", "increase", "increased", "beat", "beats", "exceeds", 
                "exceeded", "outperform", "outperforms", "outperformed", "strong", "stronger",
                "strongest", "improve", "improved", "improving", "improvement", "higher", "surge",
                "surged", "rally", "rallied", "bullish", "optimistic", "upgrade", "upgraded",
                "buy", "buying", "success", "successful", "win", "winning", "innovation"
            ]
            
            negative_words = [
                "down", "fall", "falling", "fell", "loss", "losses", "negative", "decline", 
                "declined", "drop", "dropped", "decrease", "decreased", "miss", "missed", 
                "underperform", "underperforms", "underperformed", "weak", "weaker", "weakest",
                "lower", "plunge", "plunged", "sink", "sank", "bearish", "pessimistic", "downgrade",
                "downgraded", "sell", "selling", "fail", "failed", "failure", "cut", "cuts", 
                "concern", "concerns", "worried", "worry", "risk", "risky", "warn", "warning",
                "layoff", "layoffs", "lawsuit", "investigation", "probe", "fine", "penalty",
                "default", "bankruptcy", "recession", "crisis", "scandal", "controversy"
            ]
            
            # Count positive and negative words in title and summary
            text = (title + " " + (summary or "")).lower()
            positive_count = sum(1 for word in positive_words if word in text.split())
            negative_count = sum(1 for word in negative_words if word in text.split())
            
            # Calculate sentiment score (-1 to 1)
            total_count = positive_count + negative_count
            if total_count > 0:
                score = (positive_count - negative_count) / total_count
            else:
                score = 0  # Neutral if no sentiment words found
            
            # Categorize sentiment
            if score > 0.3:
                sentiment = "Positive"
            elif score < -0.3:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            return {
                "score": score,
                "sentiment": sentiment,
                "positive_count": positive_count,
                "negative_count": negative_count
            }
        
        # Process news articles
        articles = []
        total_score = 0
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        # Get current time for date filtering
        from datetime import datetime, timedelta
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(days=days)
        
        # Process each news article
        for article in news_data:
            # Extract publication time
            pub_time = None
            if "providerPublishTime" in article:
                pub_time = datetime.fromtimestamp(article["providerPublishTime"])
            
            # Skip articles older than the cutoff
            if pub_time and pub_time < cutoff_time:
                continue
            
            # Analyze sentiment
            title = article.get("title", "")
            summary = article.get("summary", "")
            sentiment_analysis = analyze_sentiment(title, summary)
            
            # Count by sentiment category
            if sentiment_analysis["sentiment"] == "Positive":
                positive_count += 1
            elif sentiment_analysis["sentiment"] == "Negative":
                negative_count += 1
            else:
                neutral_count += 1
            
            # Add to running total
            total_score += sentiment_analysis["score"]
            
            # Add processed article
            articles.append({
                "title": title,
                "published": pub_time.isoformat() if pub_time else "Unknown",
                "source": article.get("publisher", "Unknown"),
                "sentiment": sentiment_analysis["sentiment"],
                "score": sentiment_analysis["score"],
                "url": article.get("link", "")
            })
        
        # Calculate overall sentiment
        total_articles = len(articles)
        if total_articles > 0:
            average_score = total_score / total_articles
            if average_score > 0.3:
                overall_sentiment = "Positive"
            elif average_score < -0.3:
                overall_sentiment = "Negative"
            else:
                overall_sentiment = "Neutral"
            
            sentiment_distribution = {
                "positive": positive_count / total_articles,
                "neutral": neutral_count / total_articles,
                "negative": negative_count / total_articles
            }
        else:
            average_score = 0
            overall_sentiment = "Neutral"
            sentiment_distribution = {
                "positive": 0,
                "neutral": 0,
                "negative": 0
            }
        
        # Generate insights
        insights = []
        
        if total_articles > 0:
            # Overall sentiment insight
            insights.append(f"Overall news sentiment for {ticker} is {overall_sentiment.lower()} with an average score of {average_score:.2f}.")
            
            # Distribution insight
            insights.append(f"{positive_count} positive, {neutral_count} neutral, and {negative_count} negative articles in the last {days} days.")
            
            # Trend or noticeable pattern
            if positive_count > (negative_count + neutral_count):
                insights.append(f"Media coverage is predominantly positive, which may indicate favorable market perception.")
            elif negative_count > (positive_count + neutral_count):
                insights.append(f"Media coverage is predominantly negative, which may indicate concerning market perception.")
            
            # Recent sentiment (last 3 articles)
            if len(articles) >= 3:
                recent_articles = articles[:3]
                recent_sentiments = [a["sentiment"] for a in recent_articles]
                if all(s == "Positive" for s in recent_sentiments):
                    insights.append("Recent news trend is strongly positive.")
                elif all(s == "Negative" for s in recent_sentiments):
                    insights.append("Recent news trend is strongly negative.")
            
            # Check for highly positive or negative articles
            strong_positive = [a for a in articles if a["score"] > 0.6]
            strong_negative = [a for a in articles if a["score"] < -0.6]
            
            if strong_positive:
                insights.append(f"Found {len(strong_positive)} strongly positive articles that could drive bullish sentiment.")
            if strong_negative:
                insights.append(f"Found {len(strong_negative)} strongly negative articles that could drive bearish sentiment.")
        else:
            insights.append(f"No recent news found for {ticker} in the past {days} days.")
        
        # Prepare the result
        result = {
            "ticker": ticker,
            "analysis_period": f"Last {days} days",
            "total_articles": total_articles,
            "overall_sentiment": {
                "score": average_score,
                "category": overall_sentiment,
                "distribution": sentiment_distribution
            },
            "insights": insights,
            "articles": articles
        }
        
        logger.info(f"Successfully analyzed news sentiment for {ticker}")
        return json.dumps(result, indent=2)
    
    except Exception as e:
        logger.error(f"Error analyzing news sentiment: {str(e)}")
        return json.dumps({"Error": f"Failed to analyze news sentiment: {str(e)}"}, indent=2)
