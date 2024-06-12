import os
import pickle
import hashlib
import pandas as pd
from tqdm import tqdm
from src.ticker import ITicker


def generate_hash(tickers, benchmark):
    hash_input = ''.join(tickers) + benchmark
    return hashlib.md5(hash_input.encode()).hexdigest()


def save_data_to_file(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_data_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_data(tickers: list[str], benchmark: str):
    """Return the closing prices of the SMI and the stocks in TICKERS as a pandas DataFrame."""
    # Ensure the cache directory exists
    os.makedirs('data/cache', exist_ok=True)
    
    # Generate the filename based on the hash
    data_hash = generate_hash(tickers, benchmark)
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
            dfs.append(df)
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

    df = pd.concat(dfs, axis=1)
    df.index = pd.to_datetime(df.index)

    # Remove the benchmark from the stocks
    bm = df.pop(benchmark)
    bm.name = benchmark
    
    # Fill the missing data in the benchmark with the previous value
    bm = bm.fillna(method='ffill')
    result = (bm, df)

    # Save the data to file
    save_data_to_file(result, filename)
    
    return result
