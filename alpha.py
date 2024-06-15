import tqdm
from datetime import datetime
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from src.api import Eodhd


eod = Eodhd(expire_after=60)
if __name__ == '__main__' and 0:
    df_current = eod.get_symbols("SW")
    df_current["delisted"] = 0
    df_delisted = eod.get_symbols("SW", delisted=1)
    df_delisted["delisted"] = 1
    df = pd.concat([df_current, df_delisted])

    df = df[df["Type"] == "Common Stock"]
    df.set_index("Code", inplace=True)
    
    dfs = []
    for code in tqdm.tqdm(df.index, desc="Downloading data"):
        df_eod = eod.get_eod_data(f"{code}.SW")
        adj_close = df_eod["adjusted_close"] # adjusted_close
        adj_close.name = code
        dfs.append(adj_close)

    df_combined = pd.concat(dfs, axis=1)
    df_combined.to_csv("data/eod_data.csv")


if __name__ == '__main__' and 1:
    df = pd.read_csv("data/eod_data.csv", index_col=0, parse_dates=True)

    # Filter out any stocks that have value higher than 1000
    df = df[df.columns[df.max() < 10_000]]

    # Perform the ffill till the last traded date
    columns = []
    for column in df.columns:
        last_traded_date = df[column].last_valid_index()
        first_traded_dates = df[column].first_valid_index()
        total_trading_days = (last_traded_date - first_traded_dates).days

        # Count the number of NaNs
        nans = df[column].loc[first_traded_dates:last_traded_date].isna().sum()
        if nans / total_trading_days < 0.2:
            columns.append(column)
        df[column] = df[column].loc[:last_traded_date].ffill()
    df = df[columns]

    # Download the benchmark data
    smi = eod.get_benchmark("^SSMI")
    smi = smi.loc["2000":]
    smi = smi / smi.iloc[0]
    smi.plot()

    print(len(df.columns))

    # Plot an equal weighted portfolio for each day with the stocks that are traded (not NaN)
    df = df.loc[smi.index]
    df.fillna(0, inplace=True)
    portfolio = df.mean(axis=1)
    portfolio = portfolio / portfolio.iloc[0]
    portfolio.plot()

    plt.legend(["SMI", "Portfolio"])
    plt.show()
