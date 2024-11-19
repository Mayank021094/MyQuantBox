# ------------------Import Libraries -------------#
from arch import arch_model
from mgarch import MGARCH
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from sklearn.metrics import mean_squared_error

# ---------------------CONSTANTS------------------#


# --------------------MAIN CODE-------------------#

class Expected_Volatility:
    def __init__(self, prices_df):

        self.prices = prices_df
        #Initialize Returns
        self.returns = np.log(self.prices).diff().dropna()
        #Test if the returns series is stationary
        test_results = self.test_unit_root(self.returns)


    def mgarch(self):

        model = MGARCH(dist='t')
        fitted_parameters = model.fit(self.returns)
        self.cov = model.predict()
        return self.cov

    def garch(self):
        trainsize = 0.5 * len(self.returns)
        T = len(self.returns)
        results = {}
        for p in range(1, 5):
            for q in range(1, 5):
                result = []
                for s, t in enumerate(range(trainsize, T - 1)):
                    train_set = self.returns.iloc[s: t]
                    test_set = self.returns.iloc[t + 1]  # 1-step ahead forecast
                    model = arch_model(y=train_set, p=p, q=q).fit(disp='off')
                    forecast = model.forecast(horizon=1)
                    mu = forecast.mean.iloc[-1, 0]
                    var = forecast.variance.iloc[-1, 0]
                    result.append([(test_set - mu) ** 2, var])
                df = pd.DataFrame(result, columns=['y_true', 'y_pred'])
                results[(p, q)] = np.sqrt(mean_squared_error(df.y_true, df.y_pred))

#----------------------Tests on Volatility-----------#
    def test_unit_root(self, df):
        return df.apply(lambda x: f'{pd.Series(adfuller(x)).iloc[1]:.2%}').to_frame('p-value')







