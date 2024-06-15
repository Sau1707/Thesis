import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from src.constants import SMI, TICKERS_SW

#function for importing data
def get_data(stocks, start, end):
    stockData = yf.download(stocks, start, end)
    return stockData['Adj Close']

#selecting stocks and date ranges
stockList = [SMI] + TICKERS_SW
stocks = [stock for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365*10)

#running function to pull data on selected stocks
stocks = get_data(stocks, startDate, endDate)

# Plot the total returns
returns = stocks.pct_change()
total_returns = (1 + returns).cumprod() - 1

smi = total_returns[SMI]
stocks = total_returns.drop(SMI, axis=1)

# Equally weighted portfolio
weights = np.ones(len(stocks.columns)) / len(stocks.columns)
portfolio_eq = (weights * stocks).sum(axis=1)

# Random weighed portfolio
weights = np.random.dirichlet(np.ones(len(stocks.columns)))
portfolio_rd = (weights * stocks).sum(axis=1)

#plotting the simulation
plt.plot(smi, label=SMI)
plt.plot(portfolio_eq, label="Equally Weighted")
plt.plot(portfolio_rd, label="Random Weights")
plt.legend()
plt.show()