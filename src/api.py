import os
import datetime        
import requests_cache
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
load_dotenv()


class Eodhd:
    def __init__(self, expire_after: int = 1):
        self.base_url = 'https://eodhd.com/api'
        self.session = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after=datetime.timedelta(days=expire_after))
    
    def _get_data(self, path: str, ticker: str = None, **kwargs) -> pd.DataFrame:
        params = {
            'api_token': os.getenv('EOD_API_KEY'),
            'fmt': 'json',
            **kwargs
        }

        if ticker is None:
            url = f'{self.base_url}/{path}'
        else:
            url = f'{self.base_url}/{path}/{ticker}'
        response = self.session.get(url, params=params)

        if response.status_code == 200:
            obj = response.json()
            df = pd.DataFrame(obj)
            return df
        else:
            raise Exception(f"Request failed with status code {response.status_code}")
    
    def get_exchanges(self) -> pd.DataFrame:
        """https://eodhd.com/api/exchanges-list/?api_token={}&fmt=json"""
        return self._get_data("exchanges-list")
    
    def get_symbols(self, exchange: str, delisted: int = 0) -> pd.DataFrame:
        """https://eodhd.com/api/exchange-symbol-list/{EXCHANGE_CODE}?api_token={}&delisted=0&fmt=json"""
        return self._get_data("exchange-symbol-list", exchange, delisted=delisted)
    
    def get_eod_data(self, ticker: str, period: str = 'd') -> pd.DataFrame:
        df = self._get_data("eod", ticker, period=period)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')
    
    def get_benchmark(self, ticker: str) -> pd.DataFrame:
        df = yf.download(ticker, session=self.session)
        return df["Adj Close"]
