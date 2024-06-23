import os
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from src.api import Eodhd


class Dataset:
    def __init__(self, exchange: str, start: str, benchmark: str):
        os.makedirs("data", exist_ok=True)

        self._exchange = exchange
        self._start = start
        self._benchmark = benchmark

        # Check if the data is already downloaded
        filename = f"data/data_{exchange}_{start}_{benchmark}.pkl"
        if os.path.exists(filename):
            self._bm, self._stocks = pickle.load(open(filename, "rb"))
            return

        # Download the benchmark
        eod = Eodhd(expire_after=60)
        self._bm = eod.get_benchmark(benchmark).loc[start:]

        # Get all the symbols listed and not
        df_current = eod.get_symbols(exchange)
        df_current["delisted"] = 0
        df_delisted = eod.get_symbols(exchange, delisted=1)
        df_delisted["delisted"] = 1
        df = pd.concat([df_current, df_delisted])

        # Filter out only stocks
        df = df[df["Type"] == "Common Stock"]
        df.set_index("Code", inplace=True)

        # Download the data
        dfs = []
        for code in tqdm(df.index, desc="Downloading data"):
            df_eod = eod.get_eod_data(f"{code}.{exchange}")
            adj_close = df_eod["adjusted_close"] # adjusted_close
            adj_close.name = code

            # Create a serie with the same index as the benchmark
            serie = pd.Series(index=self._bm.index, name=code)
            valid_index = adj_close.index.intersection(self._bm.index)
            serie.loc[valid_index] = adj_close[valid_index]
            dfs.append(serie)

        self._stocks = pd.concat(dfs, axis=1)

        # Save the data
        with open(filename, "wb") as f:
            pickle.dump((self._bm, self._stocks), f)

    def get_data(self, liquidity: float = 0.3) -> pd.DataFrame:
        """Get the data for the specified range
            liquidity:
            - 1: All the stocks must be traded every day
            - 0: All the stocks must be traded at least once
        """
        # df = df[df.columns[df.max() < 200_000]]
        stocks = self._stocks.copy()

        dfs = []
        for code in stocks.columns:
            serie = stocks[code]
            
            first_traded = serie.first_valid_index()
            last_traded = serie.last_valid_index()

            total_days = (last_traded - first_traded).days
            if total_days < 252:
                # print(f"[Warning] {code} is traded for less than a year")
                continue

            total_years = (last_traded - first_traded).days / 365 * 252
            missing_days = serie.loc[first_traded:last_traded].isna().sum()

            # Filter out stocks that are not traded for more than 30% of the time
            if missing_days and missing_days / total_years > 1 - liquidity:
                # print(f"[Warning] {code} has more than {liquidity * 100}% missing data")
                continue
            else:
                serie.loc[first_traded:last_traded] = serie.loc[first_traded:last_traded].ffill()

                if last_traded < datetime.datetime.now() - datetime.timedelta(days=7):
                    serie.loc[last_traded:] = 0

            dfs.append(serie)

        # Return the combined data
        return pd.concat(dfs, axis=1).iloc[:-1]

    def get_benchmark(self) -> pd.Series:
        """Get the benchmark for the specified range"""
        return self._bm
    
    ##################################################################################
    # Plot
    ##################################################################################
    def plot(self, liquidity: float | list[float] = 0.3):
        """Plot the total return of the portfolio for different liquidity levels"""
        if isinstance(liquidity, float):
            liquidity = [liquidity]
        
        bm = self.get_benchmark()
        bm = (1 + bm.pct_change()).cumprod() - 1
        
        plt.figure(figsize=(12, 6))

        # Plot the returns
        plt.subplot(1, 2, 1)
        for liq in liquidity:
            df = self.get_data(liq)
            returns = df.pct_change(fill_method=None)
            total_return = (returns + 1).cumprod() - 1
            total_return.mean(axis=1).plot(label=f"Mean Stocks (Liquidity>{liq * 100}%)")

        bm.plot(label="Benchmark")
        plt.xlabel("Date")
        plt.ylabel("Total Return")
        plt.legend()

        # Plot the number of stocks
        plt.subplot(1, 2, 2)
        for liq in liquidity:
            df = self.get_data(liq)
            total_stocks = df.apply(lambda x: x.notna() & (x != 0)).sum(axis=1)
            total_stocks.plot(label=f"Total Stocks (Liquidity>{liq * 100}%)")

        plt.xlabel("Date")
        plt.ylabel("Number of Stocks")
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    data =  Dataset("SW", "1995", "^SSMI")
    data.get_data(liquidity=0.99).to_csv("data.csv")
    data.plot(liquidity=[0.0, 0.5, 0.9, 0.99])