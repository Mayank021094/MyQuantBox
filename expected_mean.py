# ------------------Import Libraries -------------#
import pandas as pd
import arch
import numpy as np
from numpy.linalg import LinAlgError
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from scipy.stats import probplot, moment
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# ---------------------CONSTANTS------------------#

# --------------------MAIN CODE-------------------#

class Expected_Mean:

    def __init__(self, prices_df):
        self.prices = prices_df
        self.returns = np.log(self.prices).diff().dropna()

    def arima(self):
        train_size = int(0.5 * len(self.returns))
        ret_predictions = {}

        for column in self.returns.columns:
            returns_series = self.returns[column].mul(100)
            results = {}

            # Step 1: Evaluate all (p, q) combinations
            for p in range(1, 4):
                for q in range(1, 4):
                    aic, bic = [], []
                    convergence_error = 0
                    stationarity_error = 0
                    y_pred = []

                    # Rolling window forecast
                    for T in range(train_size, len(returns_series)):
                        train_set = returns_series[T - train_size: T]

                        # Ensure train_set is valid
                        if len(train_set) < 2:
                            print(f"Insufficient data for p={p}, q={q} at T={T}")
                            continue

                        try:
                            # Fit ARIMA model
                            model = ARIMA(endog=train_set, order=(p, 0, q)).fit()
                            forecast = model.forecast(steps=1)  # Get one-step forecast
                            y_pred.append(forecast[0])  # Append forecasted value
                            aic.append(model.aic)  # Append AIC
                            bic.append(model.bic)  # Append BIC
                            print(y_pred[-1])
                        except LinAlgError:
                            # Handle convergence error
                            convergence_error += 1
                            print(f"LinAlgError for p={p}, q={q} at T={T}")
                            continue
                        except ValueError:
                            # Handle stationarity error
                            stationarity_error += 1
                            print(f"ValueError for p={p}, q={q} at T={T}")
                            continue

                    # Skip if no predictions were made
                    if not y_pred:
                        print(f"No predictions for p={p}, q={q}. Skipping.")
                        continue

                    # Align y_true with y_pred
                    y_true_series = returns_series.iloc[train_size:train_size + len(y_pred)]

                    # Calculate evaluation metrics
                    rmse = np.sqrt(mean_squared_error(y_true=y_true_series, y_pred=y_pred))
                    mean_aic = np.mean(aic) if aic else np.inf
                    mean_bic = np.mean(bic) if bic else np.inf

                    # Store results
                    results[(p, q)] = {
                        'rmse': rmse,  # Root Mean Squared Error
                        'aic': mean_aic,  # Average AIC
                        'bic': mean_bic,  # Average BIC
                        'convergence_error': convergence_error,  # Count of convergence errors
                        'stationarity_error': stationarity_error  # Count of stationarity errors
                    }

            # Step 2: Select the best (p, q) combination
            # Criteria: Minimize RMSE, AIC, BIC, and avoid errors
            best_pq = min(
                results.keys(),
                key=lambda k: (
                    results[k]['rmse'],  # Prioritize RMSE
                    results[k]['aic'],  # Then prioritize AIC
                    results[k]['bic'],  # Then prioritize BIC
                    results[k]['convergence_error'],  # Minimize convergence errors
                    results[k]['stationarity_error']  # Minimize stationarity errors
                )
            )
            best_p, best_q = best_pq

            # Step 3: Refit the model with the best (p, q) on the entire series
            best_model = ARIMA(returns_series, order=(best_p, 0, best_q)).fit()

            # Step 4: Forecast expected returns
            forecast = best_model.forecast(steps=len(self.returns) - train_size)

            # Step 5: Store the forecasted returns
            ret_predictions[column] = forecast.values

        # Create a DataFrame from the predictions
        self.exp_ret_df = pd.DataFrame.from_dict(ret_predictions, orient='index', columns=['expected_returns'])

        # Return the expected returns DataFrame
        return self.exp_ret_df

