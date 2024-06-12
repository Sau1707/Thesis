import os
import pickle
import hashlib
import pandas as pd
from tqdm import tqdm
from src.ticker import ITicker


def generate_hash(tickers, benchmark, start_year, end_year, normalize):
    hash_input = ''.join(tickers) + benchmark + str(start_year) + str(end_year) + str(normalize)
    return hashlib.md5(hash_input.encode()).hexdigest()


def save_data_to_file(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_data_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_data(tickers: list[str], benchmark: str, start_year=2000, end_year=2200, normalize=False):
    """Return the closing prices of the SMI and the stocks in TICKERS as a pandas DataFrame."""
    
    # Ensure the cache directory exists
    os.makedirs('data/cache', exist_ok=True)
    
    # Generate the filename based on the hash
    data_hash = generate_hash(tickers, benchmark, start_year, end_year, normalize)
    filename = os.path.join('data/cache', f"{data_hash}.pkl")

    # Try to load the data from the file
    if os.path.exists(filename):
        return load_data_from_file(filename)
    
    # If file doesn't exist, fetch the data
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

    # Remove the years that are not in the range
    df = df.loc[f"{start_year}":f"{end_year}"]

    # Normalize the data
    if normalize:
        df = df / df.iloc[0]

    # Remove the benchmark from the stocks
    bm = df.pop(benchmark)
    bm.name = benchmark

    result = (bm, df)

    # Save the data to file
    save_data_to_file(result, filename)

    return result
