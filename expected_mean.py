# ------------------Import Libraries -------------#
import pandas as pd
import arch
import numpy as np

# ---------------------CONSTANTS------------------#

# --------------------MAIN CODE-------------------#

class Expected_Mean:

    def __init__(self, prices_df):

        self.prices = prices_df
        self.ret_df = np.log(self.prices).diff().dropna()



