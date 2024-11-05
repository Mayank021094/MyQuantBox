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
import datetime as dt
from weighting_strategy import equi_wt, cap_wt
import statistics

# ---------------------CONSTANTS------------------#
def get_db_connection():
    return sqlite3.connect("./raw_datasets/head_database.db")

# --------------------MAIN CODE-------------------#
class Quality:
    def __init__(self, strat, univ, wt_strat):
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
    def __del__(self):
        # Close the connection when the object is deleted
        self.DB.close()

    def get_wts(self):
        return self.wts

    def extract_line_item(self, ticker, key):
        try:
            stock = yf.Ticker(ticker)
            line_item = stock.info.get(key)
            if line_item is not None:
                return line_item
            else:
                return 'N/A'
        except Exception as e:
            print(f"Error fetching {key} for {ticker}: {e}")
            return 'N/A'

    def extract_balance_sheet_item(self, ticker, key):
        try:
            stock = yf.Ticker(ticker)
            bs = stock.balance_sheet
            stock.f
            line_item = bs.loc[key].iloc[0]
            if line_item is not None:
                return line_item
            else:
                return 'N/A'
        except Exception as e:
            print(f"Error fetching {key} for {ticker}: {e}")
            return 'N/A'

    def extract_pnl_item(self, ticker, key):
        try:
            stock = yf.Ticker(ticker)
            income_statement = stock.financials
            line_item = income_statement.loc[key].iloc[0]
            if line_item is not None:
                return line_item
            else:
                return 'N/A'
        except Exception as e:
            print(f"Error fetching {key} for {ticker}: {e}")
            return 'N/A'
    def extract_stock_data(self, ticker, **kwargs):
        try:
            stock_data = yf.download(ticker, **kwargs)
            if stock_data.empty:
                raise ValueError(f"No data found for {ticker}")
            return stock_data
        except Exception as e:
            print(f"Error fetching stock data for {ticker}: {e}")
            return pd.DataFrame()
    def asset_turnover(self):
        dict_asset_turnover = {}
        for ticker in self.tickers:
            rev = self.extract_pnl_item(ticker, 'Total Revenue')
            asset = self.extract_balance_sheet_item(ticker, 'Total Assets')
            dict_asset_turnover[ticker] = rev/asset
        df_asset_turnover = pd.DataFrame(list(dict_asset_turnover.items()), columns=['symbol_yfinance', 'scores'])
        return df_asset_turnover
    def current_ratio(self):
        dict_cr = {}
        for ticker in self.tickers:
            curr_assets = self.extract_balance_sheet_item(ticker, 'Current Assets')
            curr_debt = self.extract_balance_sheet_item(ticker, 'Current Debt')
            dict_cr[ticker] = curr_assets/curr_debt
        df_cr = pd.DataFrame(list(dict_cr.items()), columns=['symbol_yfinance', 'scores'])
        return df_cr
    def interest_coverage(self):
        dict_ic = {}
        for ticker in self.tickers:
            ebit = self.extract_pnl_item(ticker, 'EBIT' )
            interest_expense = self.extract_pnl_item(ticker, 'Interest Expense')
            dict_ic[ticker] = ebit/interest_expense
        df_ic = pd.DataFrame(list(dict_ic.items()), columns=['symbol_yfinance', 'scores'])
        return df_ic

    def leverage(self):
        dict_leverage = {}
        for ticker in self.tickers:
            dict_leverage[ticker] = self.extract_line_item(ticker, 'debtToEquity')
        df_leverage = pd.DataFrame(list(dict_leverage.items()), columns=['symbol_yfinance', 'scores'])
        return df_leverage
    def payout_ratio(self):
        dict_payout = {}
        for ticker in self.tickers:
            dict_payout[ticker] = self.extract_line_item(ticker, 'payoutRatio')
        df_payout = pd.DataFrame(list(dict_payout.items()), columns=['symbol_yfinance', 'scores'])
        return df_payout
    def roe(self):
        dict_roe = {}
        for ticker in self.tickers:
            net_income = self.extract_pnl_item(ticker, 'Net Income')
            total_assets = self.extract_balance_sheet_item(ticker, 'Total Assets')
            total_debt = self.extract_balance_sheet_item(ticker, 'Total Liabilities Net Minority Interest')
            equity = total_assets - total_debt
            dict_roe[ticker] = net_income/equity
        df_roe = pd.DataFrame(list(dict_roe.items()), columns=['symbol_yfinance', 'scores'])
        return df_roe
