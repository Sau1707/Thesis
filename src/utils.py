import os
import pandas as pd
from src.ticker import ITicker
from src.constants import SMI, TICKERS



def get_data(start_year=2000, end_year=2200, normalize=False):
    """Return the closing prices of the SMI and the stocks in TICKERS as a pandas DataFrame."""
    dfs = []

    smi = ITicker(SMI)
    smi = smi.get_close()
    dfs.append(smi)

    for ticker in TICKERS:
        t = ITicker(ticker)
        df = t.get_close()
        dfs.append(df)

    df = pd.concat(dfs, axis=1)
    df = df.dropna()
    # df.to_csv("data/stocks.csv")
    df.index = pd.to_datetime(df.index)

    # Remove the years that are not in the range
    df = df.loc[f"{start_year}":f"{end_year}"]

    # Normalize the data
    if normalize:
        df = df / df.iloc[0]

    smi = df.pop(SMI)
    return smi, df
