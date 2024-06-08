import os
import json
import logging
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

# logging.getLogger("selenium.webdriver.remote.remote_connection").setLevel(logging.FATAL)

URL = "https://www.marketscreener.com{quote}"

# SETUP
path = os.path.join("data", "tickers")
if not os.path.exists(path):
    os.makedirs(path)


mapping = {
    "LHN": "41O",
}

class Ticker:
    def __init__(self, simbol: str) -> None:
        self.simbol = simbol
        self.df = None

        # Open the old file if exists
        self.path = os.path.join("data", "tickers", f"{self.simbol}.csv")
        if os.path.exists(self.path):
            df = pd.read_csv(self.path, index_col="Date")
            if not df.empty:
                self.df = df 
                return
        
        if simbol in mapping:
            simbol = mapping[simbol]
    
        extensions = ["", ".SW", ".SI"]
        for ext in extensions:
            try:
                self._tiker = yf.Ticker(simbol + ext)
                assert len(self._tiker.info) > 20
                break
            except Exception as e:
                pass
        else:
            raise ValueError(f"Could not create Ticker for {simbol}")
        

    @classmethod
    def from_quote(cls, quote: str) -> "Ticker":
        """Get the simbol from the quote."""
        options = webdriver.ChromeOptions()
        prefs = {"profile.managed_default_content_settings.images": 2}
        options.add_experimental_option("prefs", prefs)
        options.add_argument("--headless")
        options.add_argument("--disable-javascript")
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--disable-infobars")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-notifications")
        options.add_argument("--incognito")
        options.add_argument("--log-level=3")
        options.add_experimental_option("excludeSwitches", ["enable-logging"])

        service = Service(service_log_path="NUL")
        browser = webdriver.Chrome(options=options, service=service)
        browser.set_script_timeout(120)
        browser.get(URL.format(quote=quote))
        
        # Define the script to get the simbol
        script = """return $(".m-0.badge.txt-b5.txt-s1")[0].innerText;"""
        simbol = browser.execute_script(script)
        browser.quit()

        return cls(simbol)

    def __str__(self) -> str:
        return f'Ticker("{self.simbol}")'
    
    def get_data(self) -> pd.DataFrame:
        """Get the data for the ticker on a daily basis for the last year."""
        if self.df is not None:
            return self.df
                    
        # Get the data
        try:            
            self.df = self._tiker.history(period="10y")
        except Exception as e:
            logging.error(f"Error getting data for {self.simbol}: {e}")
            return None

        # Remove the timezone
        zurich_timezone = 'Europe/Zurich'
        self.df.index = self.df.index.tz_convert(zurich_timezone)
        self.df.index = self.df.index.tz_localize(None).normalize()

        # Save the data
        self.df.to_csv(self.path)
        return self.df

    
if __name__ == "__main__":
    ticker = Ticker("SREN")
    df = ticker.get_data()
    print(ticker.get_volatility())
    df.plot(y="Close")
    plt.show()

