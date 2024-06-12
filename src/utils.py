import pandas as pd
from tqdm import tqdm
from src.ticker import ITicker
from src.constants import SMI, TICKERS



def get_data(tickers: list[str], benchmark: str, start_year=2000, end_year=2200, normalize=False):
    """Return the closing prices of the SMI and the stocks in TICKERS as a pandas DataFrame."""
    
    # Try to open the data from the file and check if all the tickers are in the data and the range is correct
    try:
        df = pd.read_csv("data/stocks.csv", index_col=0, parse_dates=True)
        #  TODO: fix this 
        assert benchmark in df.columns, "The benchmark is not in the data"
        assert df.index[0].year <= start_year, "The start year is not in the data"
        assert df.index[-1].year >= end_year, "The end year is not in the data"
    except Exception as e:  
        print(f"Error reading data: {e}")
        dfs = []

        bm = ITicker(benchmark)
        bm = bm.get_close()
        dfs.append(bm)

        for ticker in tqdm(tickers, desc="Downloading data"):
            try:
                t = ITicker(ticker)
                df = t.get_close()

                # Check that the start year is in the data
                if df.index[0].year > start_year:
                    continue
                
                dfs.append(df)
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")

        df = pd.concat(dfs, axis=1)
        df = df.dropna()
        df.index = pd.to_datetime(df.index)
        df.to_csv("data/stocks.csv")

        
        
    # Remove the years that are not in the range
    df = df.loc[f"{start_year}":f"{end_year}"]

    # Normalize the data
    if normalize:
        df = df / df.iloc[0]

    # Remove the benchmark from the stocks
    bm = df.pop(benchmark)
    bm.name = benchmark

    return bm, df
