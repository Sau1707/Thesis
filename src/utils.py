import os
import datetime
import pandas as pd
from tqdm import tqdm
from src.api import Eodhd
import pickle


def get_data(exchange = "SW", start = "1995", benchmark = "^SSMI") -> tuple[pd.Series, pd.DataFrame]:
    """Download data from EODHD and return the SMI and the data of all the stocks listed on the exchange"""
    os.makedirs("data", exist_ok=True)

    filename = f"data/data_{exchange}_{start}_{benchmark}.pkl"
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    # Download the benchmark
    eod = Eodhd(expire_after=60)
    bm = eod.get_benchmark(benchmark)
    bm = bm.loc[start:]

    # Get all the symbols listed and not
    df_current = eod.get_symbols(exchange)
    df_current["delisted"] = 0
    df_delisted = eod.get_symbols(exchange, delisted=1)
    df_delisted["delisted"] = 1
    df = pd.concat([df_current, df_delisted])

    # Filter out only stocks
    df = df[df["Type"] == "Common Stock"]
    df.set_index("Code", inplace=True)

    dfs = []
    for code in tqdm(df.index, desc="Downloading data"):
        df_eod = eod.get_eod_data(f"{code}.{exchange}")
        adj_close = df_eod["adjusted_close"] # adjusted_close
        adj_close.name = code

        # Create a serie with the same index as the benchmark
        serie = pd.Series(index=bm.index, name=code)
        valid_index = adj_close.index.intersection(bm.index)
        serie.loc[valid_index] = adj_close[valid_index]

        # Analitycs     
        first_traded = serie.first_valid_index()
        last_traded = serie.last_valid_index()
        total_years = (last_traded - first_traded).days / 365 * 252
        missing_days = serie.loc[first_traded:last_traded].isna().sum()

        # Filter out stocks that are not traded for more than 30% of the time
        if missing_days / total_years > 0.3:
            print(f"[Warning] {code} has more than 30% missing data")
            continue
        else:
            serie.loc[first_traded:last_traded] = serie.loc[first_traded:last_traded].ffill()

            if last_traded < datetime.datetime.now() - datetime.timedelta(days=7):
                serie.loc[last_traded:] = 0
        
        dfs.append(serie)

    # Combine all the data
    df = pd.concat(dfs, axis=1) 
    df = df[df.columns[df.max() < 200_000]]

    # Save the data
    with open(filename, "wb") as f:
        pickle.dump((bm, df), f)
    
    return bm, df