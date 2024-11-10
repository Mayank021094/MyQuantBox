# ------------------Import Libraries ------------#
import pandas as pd
import sqlite3
import numpy as np
from numpy.random import random, uniform, dirichlet, choice
from numpy.linalg import inv
from scipy.optimize import minimize
import yfinance as yf
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import seaborn as sns
from pykalman import KalmanFilter
import datetime as dt
from ta.momentum import RSIIndicator
from weighting_strategy import equi_wt, cap_wt
from sklearn.linear_model import LinearRegression
from extract_data import Extract_Data


# ---------------------CONSTANTS------------------#
def get_db_connection():
    return sqlite3.connect("./raw_datasets/head_database.db")


# --------------------MAIN CODE-------------------#
class Optimization:
    def __init__(self, univ):

        self.periods_per_year = None
        self.rf = None
        self.wts = None
        # Use the function to create a new connection
        self.DB = get_db_connection()

        # Load the columns from the database into DataFrames
        symbol_yfinance_df = pd.read_sql_query("SELECT symbol_yfinance FROM table_symbol_yfinance", self.DB)
        symbol_nse_df = pd.read_sql_query("SELECT symbol_nse FROM table_symbol_nse", self.DB)
        # Combine both columns into a single DataFrame assuming index positions are the mapping
        symbol_mapping = pd.concat([symbol_yfinance_df, symbol_nse_df], axis=1)

        # Store Universe
        query = "SELECT * FROM table_symbol_yfinance"
        symbols = pd.read_sql_query(query, self.DB)
        if univ == 'All':
            self.tickers = symbols['symbol_yfinance'].tolist()
        elif univ == 'Nifty 50':
            symbol_nse_df = pd.read_sql_query("SELECT * FROM table_symbol_nse", self.DB)
            flag = symbol_nse_df['nifty_50'].tolist()
            self.tickers = symbols['symbol_yfinance'][pd.Series(flag).astype(bool)].tolist()
        elif univ == 'Nifty 500':
            self.tickers = symbols['symbol_yfinance'].head(500).tolist()
        elif univ == 'Nifty Next 50':
            pass

        # Calculate Scores based on strategy
        # if strat == 'RSI':
        #     scores = self.rsi()
        # elif strat == 'Price Momentum (12-1)':
        #     scores = self.price_momentum_1_12()
        # elif strat == 'Price Momentum (12-3)':
        #     scores = self.price_momentum_3_12()
        # elif strat == 'Price Acceleration':
        #     scores = self.price_acceleration()

    def __del__(self):
        # Close the connection when the object is deleted
        self.DB.close()

    # Method to return the weights DataFrame
    def get_wts(self):
        return self.wts

    # -----------------------STRATEGIES----------------------#

    def max_sharpe(self):
        ret_dict = {}
        for ticker in self.tickers:
            data = Extract_Data(ticker)
            temp = data.extract_stock_data(period='10y', interval='1d')
            prices = temp['Close']
            weekly_returns = prices.resample('W').last().pct_change().dropna()
            ret_dict[ticker] = weekly_returns

        # Determine the maximum length and align date range
        max_length = max(len(v) for v in ret_dict.values())
        end = max(series.index[-1] for series in ret_dict.values())
        start = end - pd.DateOffset(weeks=max_length - 1)

        aligned_dict = {}
        for ticker, weekly_returns in ret_dict.items():
            aligned_series = weekly_returns[start:end]
            aligned_series = aligned_series.reindex(pd.date_range(start=start, end=end, freq='W'),
                                                    method='ffill')
            aligned_dict[ticker] = aligned_series.tolist()

        # Create DataFrame from aligned returns
        ret_df = pd.DataFrame(aligned_dict)
        ret_df.index = pd.date_range(start=start, end=end, freq='W')

        # Calculate average number of weekly periods per year
        ret_df['Year'] = ret_df.index.year
        periods_per_year = ret_df.groupby('Year').size().mean()
        self.periods_per_year = round(periods_per_year)
        ret_df.drop(['Year'], axis=1, inplace=True)

        # Compute mean returns, covariance, and precision matrix
        mean_returns = ret_df.mean()
        cov_matrix = ret_df.cov()
        precision_matrix = pd.DataFrame(inv(cov_matrix), index=ret_df.columns, columns=ret_df.columns)

        # Fetch risk-free rate (using '^TNX' as proxy for 10-year rate)
        rf_data = yf.Ticker("^TNX")
        rf = rf_data.history(start=start, end=end)['Close'].resample('W').last()
        rf = rf.div(self.periods_per_year).div(100).mean()
        self.rf = rf

        # Define the constraint that weights sum to 1
        weight_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        # Initial guess for weights
        x0 = np.random.dirichlet(np.ones(len(ret_df.columns)), size=1)[0]

        # Optimize the portfolio to maximize the Sharpe Ratio
        wts = minimize(
            fun=self.neg_sharpe_ratio,
            x0=x0,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=[(0, 1) for _ in range(len(ret_df.columns))],
            constraints=weight_constraint,
            options={'maxiter': 1e4}
        )
        # Store and display the optimal weights
        self.wts = pd.DataFrame({'Symbol_NSE':ret_df.columns, 'Weights':wts.x})
        print(self.wts)

    def min_vol(self):
        pass

    def risk_parity(self):
        pass

    def kelly(self):
        pass

    # Portfolio metrics
    def portfolio_std(self, weights, cov=None):
        return np.sqrt(weights @ cov @ weights * self.periods_per_year)

    def portfolio_returns(self, weights, rt=None):
        return (weights @ rt + 1) ** self.periods_per_year - 1

    def portfolio_performance(self, weights, rt, cov):
        r = self.portfolio_returns(weights, rt=rt)
        sd = self.portfolio_std(weights, cov=cov)
        return r, sd

    def neg_sharpe_ratio(self, weights, mean_ret, cov):
        """Calculate the negative Sharpe ratio given portfolio weights."""
        r, sd = self.portfolio_performance(weights, mean_ret, cov)
        return -(r - self.rf) / sd


temp = Optimization(univ='Nifty 50')
temp.max_sharpe()
