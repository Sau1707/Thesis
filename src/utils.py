import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from src.api import Eodhd
from pypfopt import expected_returns


class Stocks:
    def __init__(self, stocks: pd.DataFrame):
        self._stocks = stocks.iloc[:-1]  # Never use the last date as it can be incomplete
    
    def get_start_date(self) -> pd.Timestamp:
        """Return the start date of the stocks"""
        return self._stocks.index.min()
    
    def get_end_date(self) -> pd.Timestamp:
        """Return the end date of the stocks"""
        return self._stocks.index.max()
    
    def get_historical(self, *, start: pd.Timestamp = None, end: pd.Timestamp = None, valid: bool = True):
        """Get the historical data of the stocks
            start: Start date
            end: End date
            valid: Only return the valid data
        """
        df = self._stocks.loc[start:end]
        if valid:
            df = df.dropna(axis=1)
            print(df)
            df = df.loc[:, df.iloc[-1] != 0]
        return df
    
    def get_stocks(self, *, date: pd.Timestamp = None) -> np.array:
        """Return all the valid stocks at the specified date"""
        if date is None:
            return self._stocks.columns.to_numpy()
        
        # If the date is not in the index, return the closest date
        if date not in self._stocks.index:
            date = self._stocks.index[abs(self._stocks.index - date).argmin()]
        return self._stocks.loc[date].dropna().index.to_numpy()
    
    def get_returns(self, *, start: pd.Timestamp = None, end: pd.Timestamp = None, valid: bool = True) -> pd.DataFrame:
        """Get the returns of the stocks
            start: Start date
            end: End date
        """
        stocks = self.get_historical(start=start, end=end, valid=valid)
        return stocks.pct_change()

    def get_total_return(self, *, start: pd.Timestamp = None, end: pd.Timestamp = None, valid: bool = True) -> pd.Series:
        """Get the total return of the stocks
            start: Start date
            end: End date
        """
        returns = self.get_returns(start=start, end=end, valid=valid)
        total_return = (1 + returns).cumprod() - 1
        return total_return
    
    def get_mean_returns(self) -> pd.Series:
        """Get the mean returns of the stocks"""
        return expected_returns.mean_historical_return(self._stocks)




if __name__ == "__main__":
    from src.dataset import Dataset
    data =  Dataset("SW", "2000", "^SSMI")
    df = data.get_data(liquidity=0.99)
    bm = data.get_benchmark()
    np.random.seed(0)

    stocks = Stocks(df)
    historical = stocks.get_historical(valid=False)
    returns = stocks.get_returns(valid=False)
    total_return = stocks.get_total_return(valid=False)

    print(len(historical.columns), len(returns.columns), len(total_return.columns))

    # Plot the total return of the portfolio
    total_return.mean(axis=1).plot(label="Mean Stocks")
    bm_returns = (1 + bm.pct_change()).cumprod() - 1
    bm_returns.plot(label="Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Total Return")
    plt.legend()
    plt.show()