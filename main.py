import pandas as pd
import matplotlib.pyplot as plt
from src.members import Members
from src.members import Ticker
from src.utils import utils


# We start by collecting the data for the SMI index.
index = Members("smi", "https://www.marketscreener.com/quote/index/SMI-7508/components/")
if not index.has_data(): index.download_data()

df = index.get_df()
print(f"{df["WEIGHT"].sum()}%")

# We can now download the tickers
if not index.has_tickers(): index.download_tickers()

# We can now download the data for the tickers
dfs = []
for ticker in index.get_df()["TICKER"]:
    t = Ticker(ticker)
    # Get the data, than save the closing price as the ticker name
    _df = t.get_data()
    columns = [col for col in _df.columns]
    _df[ticker] = _df["Close"]
    _df.drop(columns=columns, inplace=True)
    # _df = _df.resample('D').ffill()
    dfs.append(_df)

# Merge the data
data = pd.concat(dfs, axis=1)
data.index = pd.to_datetime(data.index)
data = data["2019":]

# Fill missing values
data = data.resample('D').ffill()



data.to_csv("./data/smi.csv")
data.dropna(inplace=True, axis=0)

# Plot the data
data.plot(figsize=(15, 10))
plt.show()
exit()

# Calculate the returns
returns = data.pct_change()
# rets.plot(figsize=(15, 10))
# plt.show()

# With cumulative returns
cum_rets = (1+returns).cumprod()
# cum_rets.plot(figsize=(15, 10))
# plt.show()

# Daily risk free rate
rf = (1.02**(1/len(data)))-1  # TODO: find the real risk free rate for CHF

# Calculate the shape ratios
volatility = returns.std() # volatility
print(volatility)
exit()
expected_returns = returns.mean() # excpected returns

# Calculate the shape ratios
shape_ratios = (expected_returns - rf)/volatility
print(shape_ratios)
exit()


# Plot the data
# data.plot(figsize=(15, 10))
# plt.show()

# Calculate the returns
rets = data.pct_change()
print(rets)
plot = rets["NESN"].plot(figsize=(15, 10))
# plt.show()


# Plot the shape ratios
shape_ratios = utils.sharpe_ratio(data, 0.03, len(data))
print(shape_ratios)
shape_ratios.plot.bar(figsize=(15, 10))
plt.show()
exit()

# Create the covariance matrix
cov = data.pct_change().cov()
print(cov)