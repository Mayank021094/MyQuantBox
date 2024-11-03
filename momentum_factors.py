# ------------------Import Libraries ------------#
import pandas as pd
import sqlite3
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import seaborn as sns
from pykalman import KalmanFilter
import datetime as dt
from ta.momentum import StochRSIIndicator
from weighting_strategy import equi_wt, cap_wt

# ---------------------CONSTANTS------------------#
DB = sqlite3.connect("./raw_datasets/head_database.db")

# --------------------MAIN CODE-------------------#
class Momentum:
    def __init__(self, strat, univ, wt_strat, DB):
        # Load the columns from the database into DataFrames
        symbol_yfinance_df = pd.read_sql_query("SELECT symbol_yfinance FROM table_symbol_yfinance", DB)
        symbol_nse_df = pd.read_sql_query("SELECT symbol_nse FROM table_symbol_nse", DB)
        # Combine both columns into a single DataFrame assuming index positions are the mapping
        symbol_mapping = pd.concat([symbol_yfinance_df, symbol_nse_df], axis=1)

        # Store Universe
        query = "SELECT * FROM table_symbol_yfinance"
        symbols = pd.read_sql_query(query, DB)
        if univ == 'All':
            self.tickers = symbols['symbol_yfinance'].tolist()
        elif univ == 'Nifty 50':
            pass
        elif univ == 'Nifty 500':
            self.tickers = symbols['symbol_yfinance'].head(500).tolist()
        elif univ == 'Nifty Next 50':
            pass

        # Calculate Scores based on strategy
        if strat == 'RSI':
            pass
        elif strat == 'Price Momentum (12-1)':
            scores = self.price_momentum_1_12()
        elif strat == 'Price Momentum (12-3)':
            scores = self.price_momentum_3_12()
        elif strat == 'Price Acceleration':
            scores = self.price_acceleration()

        scores = scores.sort_values(by='scores', ascending=False)
        # Calculate Weights based on weighting strategy
        if wt_strat == 'equal_wt':
            self.wts = equi_wt(scores, symbol_mapping)
        elif wt_strat == 'cap_wtd':
            symbol_list = scores['symbol_yfinance'].tolist()
            mkt_cap = [self.extract_market_cap(ticker) for ticker in symbol_list]
            scores['mkt_cap'] = mkt_cap
            self.wts = cap_wt(scores, symbol_mapping)

    # Method to return the weights DataFrame
    def get_wts(self):
        return self.wts

    # Function to extract market capitalization with error handling
    def extract_market_cap(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            market_cap = stock.info.get('marketCap')
            return market_cap if market_cap else 'N/A'
        except Exception as e:
            print(f"Error fetching market capitalization for {ticker}: {e}")
            return 'N/A'

    def extract_stock_data(self, ticker, **kwargs):
        stock_data = yf.download(ticker, **kwargs)
        return stock_data

    def rsi(self, tickers):
        rsi_dict = {}
        for ticker in tickers:
            temp = self.extract_stock_data(ticker, period='1y', interval='1d')
            stoch_rsi = StochRSIIndicator(close=temp['Adj Close'].squeeze())
            rsi_dict[ticker] = stoch_rsi.stochrsi()

        rsi_df = pd.DataFrame(rsi_dict).tail(5)
        return rsi_df

    def calculate_monthly_returns_for_lags(self, monthly_prices, lags):
        data = pd.DataFrame(index=monthly_prices.index)
        for lag in lags:
            data[f'return_{lag}m'] = (
                monthly_prices
                .pct_change(lag)
                .add(1)
                .pow(1 / lag)
                .sub(1)
            )
        return data

    def price_momentum_1_12(self):
        mom_dict = {}
        for ticker in self.tickers:
            temp = self.extract_stock_data(ticker, period='2y', interval='1d')
            prices = temp['Adj Close']
            monthly_prices = prices.resample('ME').last()
            lags = [1, 12]

            data = self.calculate_monthly_returns_for_lags(monthly_prices, lags)
            data['momentum_1_12'] = data['return_12m'].sub(data['return_1m'])
            mom_dict[ticker] = data['momentum_1_12'].tail(1).values

        mom_df = pd.DataFrame(list(mom_dict.items()), columns=['symbol_yfinance', 'scores'])
        return mom_df

    def price_momentum_3_12(self):
        mom_dict = {}
        for ticker in self.tickers:
            temp = self.extract_stock_data(ticker, period='2y', interval='1d')
            prices = temp['Adj Close']
            monthly_prices = prices.resample('ME').last()
            lags = [3, 12]

            data = self.calculate_monthly_returns_for_lags(monthly_prices, lags)
            data['momentum_3_12'] = data['return_12m'].sub(data['return_3m'])
            mom_dict[ticker] = data['momentum_3_12'].tail(1).values

        mom_df = pd.DataFrame(list(mom_dict.items()), columns=['symbol_yfinance', 'scores'])
        return mom_df

    def price_acceleration(self):
        acc_dict = {}

# Close the database connection
DB.close()
