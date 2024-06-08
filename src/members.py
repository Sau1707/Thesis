import os
import json
import pandas as pd
from selenium import webdriver
from .ticker import Ticker


class Members:
    def __init__(self, name: str, url: str) -> None:
        # Try to load the members from the data
        self.url = url
        self.name = name

        if not os.path.exists("./data/members"):
            os.makedirs("./data/members")

        if os.path.exists(f"./data/members/{name}.csv"):
            self._df = pd.read_csv(f"./data/members/{name}.csv")
        else:
            self._df = None

    def has_data(self) -> bool:
        """Check if the data is already loaded or not"""
        return self._df is not None

    def get_df(self) -> pd.DataFrame:
        """Return the data"""
        return self._df
    
    def download_data(self) -> pd.DataFrame:
        """Get the data"""
        options = webdriver.ChromeOptions()
        browser = webdriver.Chrome(options=options)
        browser.set_script_timeout(120)

        scroll_to_bottom = open("./scripts/expand.js").read()
        table_data = open("./scripts/table.js").read()

        browser.get(self.url)
        browser.execute_async_script(scroll_to_bottom)
        table_data = browser.execute_script(table_data)
        browser.quit()

        table_data = json.loads(table_data)
        self._df = pd.DataFrame(table_data)
        self._df.columns = [col.upper() for col in self._df.columns]

        # Clean the data
        self._df["WEIGHT"] = self._df["WEIGHT"].str.replace("%", "")
        self._df["WEIGHT"] = self._df["WEIGHT"].apply(pd.to_numeric, errors="coerce")
        self._df.dropna(axis=0, inplace=True)
        
        # Save the data
        self._df.to_csv(f"./data/members/{self.name}.csv", index=False)

        return self._df
    
    def has_tickers(self):
        """Check if the tickers are already loaded or not"""
        return "TICKER" in self._df.columns
        

    def download_tickers(self):
        """Download the tickers"""
        get_ticker = lambda name: Ticker.from_quote(name).simbol
        self._df["TICKER"] = self._df["HREF"].apply(get_ticker)

        # Save the data
        self._df.to_csv(f"./data/members/{self.name}.csv", index=False)

        return self._df
    
