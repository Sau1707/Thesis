import pandas as pd
import matplotlib.pyplot as plt
from src.api import Eodhd
from src.constants import TICKERS
from src.utils import get_data

# Plot and visualize the total return of the SMI, SMIM and equally weighted benchmark
if __name__ == '__main__':
    bm, df = get_data(benchmark="^SSMI")

    # stock = "VALN"
    # data = df[stock]
    # returns = data.pct_change(fill_method=None)
    # total_return = (returns + 1).cumprod() - 1
    # total_return.plot(label=stock)
    # plt.legend()
    # plt.show()

    # SMI.SW
    bm = bm / bm.iloc[0]
    bm_returns = bm.pct_change()
    bm_total_return = (bm_returns + 1).cumprod() - 1
    bm_total_return.plot(label="SMI")

    # SMIM.SW
    eod = Eodhd(expire_after=60)
    # smin = eod.get_eod_data("SMIM.SW")
    smin = eod.get_benchmark("SMIM.SW")
    smin = smin.loc["1995":]
    smin = smin / smin.iloc[0]
    smin_returns = smin.pct_change()
    smin_total_return = (smin_returns + 1).cumprod() - 1
    smin_total_return.plot(label="SMIM")

    # Filter out any stocks that have value higher than 1000
    returns = df.pct_change(fill_method=None)
    total_return = (returns + 1).cumprod() - 1
    total_return.mean(axis=1).plot(label="Portfolio")
    plt.legend()
    plt.show()
