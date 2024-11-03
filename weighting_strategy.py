# -----------------IMPORTING LIBRARIES-------------#
import pandas as pd

# -------------------MAIN CODE----------------------#
def equi_wt(scores, symbol_mapping):
    scores = scores.merge(symbol_mapping, on='symbol_yfinance', how='left')
    if len(scores) > 30:
        scores = scores.head(30)
    wt = 1 / len(scores)
    symbols = scores['symbol_nse'].values.tolist()
    wts_df = pd.DataFrame(symbols, columns=['symbol_nse'])
    wts_df['wts'] = wt
    return wts_df

def cap_wt(scores, symbol_mapping):
    scores = scores.merge(symbol_mapping, on='symbol_yfinance', how='left')
    if len(scores) > 30:
        scores = scores.head(30)
    symbols = scores['symbol_nse'].values.tolist()
    wts_df = pd.DataFrame(symbols, columns=['symbol_nse'])
    wts = [w / sum(scores['mkt_cap']) for w in scores['mkt_cap'].values.tolist()]
    wts_df['wts'] = wts
    return wts_df
