import os
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf


DATA_PATH = "data/tickers"
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)


class ITicker():
    def __init__(self, ticker_symbol, data_path=DATA_PATH):
        self.ticker = ticker_symbol
        self.data_path = data_path
        self.file_path = os.path.join(data_path, f"{ticker_symbol}.csv")
        os.makedirs(self.data_path, exist_ok=True)

    def get_data(self, period="max", max_age_days=1):
        def is_data_old(file_path, max_age_days):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            return datetime.now() - file_mod_time > timedelta(days=max_age_days)
        
        if not os.path.exists(self.file_path) or is_data_old(self.file_path, max_age_days):
            # Download the data if it doesn't exist or is old
            df = yf.download(self.ticker, start="1990-01-01", end=datetime.now().strftime("%Y-%m-%d"))
            df.to_csv(self.file_path)
        else:
            # Load the data if it exists and is not old
            df = pd.read_csv(self.file_path, index_col=0, parse_dates=True)
        return df
    
    def get_close(self):
        """Return the closing prices as a pandas Series."""
        df = self.get_data()
        series = df["Adj Close"]
        series.name = self.ticker
        return series
    