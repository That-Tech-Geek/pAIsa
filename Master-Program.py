import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize
from io import BytesIO
from datetime import datetime

# Function Definitions
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(data, window=20):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return rolling_mean, upper_band, lower_band

def analyze_stock(ticker, period='1y', interval='1d'):
    stock_data = yf.download(ticker, period=period, interval=interval)
    if stock_data.empty:
        return pd.DataFrame()

    stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['Daily Return'] = stock_data['Close'].pct_change() * 100
    stock_data['Cumulative Return'] = (1 + stock_data['Daily Return'] / 100).cumprod()
    stock_data['Volatility'] = stock_data['Daily Return'].rolling(window=20).std()
    stock_data['RSI'] = calculate_rsi(stock_data)
    stock_data['MACD'], stock_data['Signal'] = calculate_macd(stock_data)
    stock_data['BB_Middle'], stock_data['BB_Upper'], stock_data['BB_Lower'] = calculate_bollinger_bands(stock_data)
    stock_data['Sharpe Ratio'] = stock_data['Daily Return'].mean() / stock_data['Daily Return'].std()
    
    return stock_data

def fetch_financial_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    history = stock.history(start=start, end=end)
    financials = stock.financials
    balance_sheet = stock.balance_sheet
    cashflow = stock.cashflow
    valuation_measures = stock.info
    return history, financials, balance_sheet, cashflow, valuation_measures

def extract_key_metrics(financials, balance_sheet, cashflow):
    metrics = {}
    try:
        metrics['Revenue'] = financials.loc['Total Revenue'].mean()
        metrics['EBITDA'] = financials.loc['Operating Income'].mean()
        metrics['Net Income'] = financials.loc['Net Income'].mean()
        metrics['Total Debt'] = balance_sheet.loc['Long Term Debt'].mean()
        metrics['Total Assets'] = balance_sheet.loc['Total Assets'].mean()
        metrics['Free Cash Flow'] = cashflow.loc['Free Cash Flow'].mean()
    except KeyError as e:
        st.error(f"Key Error: {e}. Some metrics might be missing.")
    return metrics

def calculate_synergies(acquirer_metrics, target_metrics):
    synergies = {}
    synergies['Projected Revenue'] = acquirer_metrics['Revenue'] + target_metrics['Revenue']
    synergies['Projected EBITDA'] = acquirer_metrics['EBITDA'] + target_metrics['EBITDA']
    synergies['Cost Synergies'] = 0.05 * synergies['Projected Revenue']
    synergies['Projected Net Income'] = acquirer_metrics['Net Income'] + target_metrics['Net Income'] + synergies['Cost Synergies']
    return synergies

def calculate_dcf(fcf, growth_rate, discount_rate, terminal_growth_rate, projection_years=5):
    fcf_projections = [fcf * (1 + growth_rate)**i for i in range(1, projection_years + 1)]
    terminal_value = fcf_projections[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    discounted_fcfs = [fcf / (1 + discount_rate)**i for i, fcf in enumerate(fcf_projections, 1)]
    discounted_terminal_value = terminal_value / (1 + discount_rate)**projection_years
    dcf_value = sum(discounted_fcfs) + discounted_terminal_value
    return dcf_value

class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.0):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.data = self.get_data()
        self.mean_returns = self.data.pct_change().mean()
        self.cov_matrix = self.data.pct_change().cov()

    def get_data(self):
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False)['Adj Close']
        return data

    def portfolio_performance(self, weights):
        returns = np.sum(self.mean_returns * weights)
        std_dev = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return returns, std_dev

    def negative_sharpe_ratio(self, weights):
        returns, std_dev = self.portfolio_performance(weights)
        return -(returns - self.risk_free_rate) / std_dev

    def check_sum(self, weights):
        return np.sum(weights) - 1

    def optimize_portfolio(self):
        num_assets = len(self.tickers)
        initial_weights = num_assets * [1. / num_assets]
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = ({'type': 'eq', 'fun': self.check_sum})
        optimal = minimize(self.negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return optimal.x

    def simulate_portfolios(self, num_portfolios=10000):
        results = np.zeros((num_portfolios, len(self.tickers) + 2))
        for i in range(num_portfolios):
            weights = np.random.random(len(self.tickers))
            weights /= np.sum(weights)
            portfolio_return, portfolio_std_dev = self.portfolio_performance(weights)
            results[i, :-2] = weights
            results[i, -2] = portfolio_return
            results[i, -1] = portfolio_std_dev
        return results

# Streamlit App
def main():
    st.title("Financial Analysis Dashboard")

    service = st.selectbox("Select Service", ['Equity Analysis', 'Mergers and Acquisitions', 'Portfolio Optimisation'])

    if service == 'Equity Analysis':
        ticker = st.text_input("Enter Yahoo Finance Ticker Symbol", "AAPL")
        period = st.selectbox("Select Data Period", ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y'])
        interval = st.selectbox("Select Data Interval", ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1d', '5d', '1wk', '1mo'])

        if st.button("Analyze"):
            with st.spinner("Fetching and analyzing data..."):
                stock_data = analyze_stock(ticker, period, interval)
                if stock_data.empty:
                    st.error("No data available.")
                else:
                    st.write(f"Basic Statistics for {ticker}:")
                    st.dataframe(stock_data.describe())

                    fig, ax = plt.subplots(3, 1, figsize=(16, 12))

                    ax[0].plot(stock_data['Close'], label='Close Price', color='blue')
                    ax[0].plot(stock_data['MA20'], label='20-Day MA', color='green', linestyle='--')
                    ax[0].plot(stock_data['MA50'], label='50-Day MA', color='red', linestyle='--')
                    ax[0].plot(stock_data['BB_Upper'], label='Upper Bollinger Band', color='orange', linestyle='--')
                    ax[0].plot(stock_data['BB_Lower'], label='Lower Bollinger Band', color='orange', linestyle='--')
                    ax[0].set_title(f"{ticker} Stock Price Analysis with MAs and Bollinger Bands")
                    ax[0].set_xlabel('Date')
                    ax[0].set_ylabel('Price')
                    ax[0].legend()

                    ax[1].plot(stock_data['MACD'], label='MACD', color='purple')
                    ax[1].plot(stock_data['Signal'], label='Signal Line', color='red', linestyle='--')
                    ax[1].axhline(0, color='black', linestyle='--')
                    ax[1].set_title('MACD and Signal Line')
                    ax[1].set_xlabel('Date')
                    ax[1].set_ylabel('MACD Value')
                    ax[1].legend()

                    ax[2].plot(stock_data['RSI'], label='RSI', color='darkblue')
                    ax[2].axhline(30, color='red', linestyle='--')
                    ax[2].axhline(70, color='red', linestyle='--')
                    ax[2].set_title('Relative Strength Index (RSI)')
                    ax[2].set_xlabel('Date')
                    ax[2].set_ylabel('RSI Value')
                    ax[2].legend()

                    plt.tight_layout()
                    st.pyplot(fig)

    elif service == 'Mergers and Acquisitions':
        acquirer_ticker = st.text_input("Enter Acquirer Ticker", 'GOOGL')
        target_ticker = st.text_input("Enter Target Ticker", 'MSFT')
        start_date = st.date_input("Start Date", datetime(2023, 1, 1))
        end_date = st.date_input("End Date", datetime.now())

        if st.button("Analyze Synergies"):
            with st.spinner("Fetching and analyzing financial data..."):
                acquirer_data, acquirer_financials, acquirer_bs, acquirer_cf, acquirer_valuation = fetch_financial_data(acquirer_ticker, start_date, end_date)
                target_data, target_financials, target_bs, target_cf, target_valuation = fetch_financial_data(target_ticker, start_date, end_date)

                acquirer_metrics = extract_key_metrics(acquirer_financials, acquirer_bs, acquirer_cf)
                target_metrics = extract_key_metrics(target_financials, target_bs, target_cf)

                synergies = calculate_synergies(acquirer_metrics, target_metrics)
                st.write("Projected Synergies:")
                st.write(synergies)

    elif service == 'Portfolio Optimisation':
        st.write("Enter comma-separated ticker symbols for portfolio optimization.")
        tickers_input = st.text_input("Tickers", "AAPL, MSFT, GOOG")
        tickers = [ticker.strip() for ticker in tickers_input.split(',')]
        start_date = st.date_input("Start Date", datetime(2022, 1, 1))
        end_date = st.date_input("End Date", datetime.now())
        risk_free_rate = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 0.0) / 100

        if st.button("Optimize Portfolio"):
            with st.spinner("Optimizing portfolio..."):
                optimizer = PortfolioOptimizer(tickers, start_date, end_date, risk_free_rate)
                optimal_weights = optimizer.optimize_portfolio()

                st.write("Optimal Portfolio Weights:")
                weights_dict = {tickers[i]: round(optimal_weights[i], 4) for i in range(len(tickers))}
                st.write(weights_dict)

                simulated_portfolios = optimizer.simulate_portfolios()
                st.write("Simulated Portfolio Results:")
                st.dataframe(pd.DataFrame(simulated_portfolios, columns=[*tickers, 'Expected Return', 'Volatility']).head(10))

if __name__ == "__main__":
    main()
