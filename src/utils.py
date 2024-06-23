import os
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from src.api import Eodhd


class Stocks:
    def __init__(self, stocks: pd.DataFrame):
        self._stocks = stocks 
    
    def get_historical(self, *, start: pd.Timestamp = None, end: pd.Timestamp = None, valid: bool = True):
        """Get the historical data of the stocks
            start: Start date
            end: End date
            valid: Only return the valid data
        """
        df = self._stocks.loc[start:end]
        if valid:
            df = df.dropna(axis=1)
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
        return (1 + returns).cumprod().iloc[-1] - 1




if __name__ == "__main__":
    from src.dataset import Dataset
    data =  Dataset("SW", "1995", "^SSMI")
    df = data.get_data(liquidity=0.99)

    stocks = Stocks(df)
    print(stocks.get_historical(valid=True)) # .to_csv("stocks.csv")
    print(stocks.get_stocks(date=pd.Timestamp("1995-01-01")))
    print(stocks.get_returns(start=pd.Timestamp("1995-01-01"), end=pd.Timestamp("1996-01-01")))
    print(stocks.get_total_return(start=pd.Timestamp("1995-01-01"), end=pd.Timestamp("1996-01-01"), valid=False))