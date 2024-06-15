from src.utils import get_data
from src.constants import TICKERS_SW, SMI
import matplotlib.pyplot as plt

from src.api import Eodhd
eod = Eodhd(expire_after=60)

bm, df = get_data(tickers=TICKERS_SW, benchmark=SMI)

smi = eod.get_benchmark("^SSMI")
smi = smi.loc["2000":"2024-06-01"]
smi = smi / smi.iloc[0]
smi.plot(label="SMI")


 # Filter out any stocks that have value higher than 1000
df = df[df.columns[df.max() < 10_000]]

# Perform the ffill till the last traded date
columns = []
for column in df.columns:
    last_traded_date = df[column].last_valid_index()
    first_traded_dates = df[column].first_valid_index()
    total_trading_days = (last_traded_date - first_traded_dates).days

    # Count the number of NaNs
    nans = df[column].loc[first_traded_dates:last_traded_date].isna().sum()
    if nans / total_trading_days < 0.2:
        columns.append(column)
    df[column] = df[column].loc[:last_traded_date].ffill()
df = df[columns]

# Plot an equal weighted portfolio for each day with the stocks that are traded (not NaN)
df = df.loc[smi.index]
df.fillna(0, inplace=True)
stocks = df.mean(axis=1)
stocks = stocks / stocks.iloc[0]

stocks.plot(label="Portfolio")

plt.legend()
plt.show()
