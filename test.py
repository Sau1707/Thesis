import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.constants import SMI, TICKERS, TICKERS_SW
from src.utils import get_data
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint, NonlinearConstraint
from scipy.optimize import minimize


bm, stocks = get_data(tickers=TICKERS_SW, benchmark=SMI, start_year=2000, end_year=2024, normalize=True)

weights = np.random.dirichlet(np.ones(len(stocks.columns)), size=1)[0]
weights = pd.Series(weights, index=stocks.columns)

START = 100

# Calculate the returns of the benchmark
bm_returns = bm.pct_change()
bm_returns = bm_returns.loc["2000":]
bm_returns = (1 + bm_returns).cumprod() * START

# Calculate the returns of the stocks
stocks_returns = stocks.pct_change()
stocks_returns = stocks_returns.loc["2000":]
stocks_returns = (1 + stocks_returns).cumprod() * START

# Simulate some random portfolios
PORTFOLIOS = 100
weights = np.random.dirichlet(np.ones(len(stocks.columns)), size=PORTFOLIOS)
portfolio_returns = stocks_returns.values @ weights.T
portfolio_returns = pd.DataFrame(portfolio_returns, index=stocks_returns.index)

# Calculate the efficient frontier of the portfolio
correlation_matrix = stocks.corr()
mean_returns = stocks.pct_change().mean()

fn_constraint = lambda w: w.T @ correlation_matrix @ w
fn = lambda w: -w.T @ mean_returns  # Minimize the negative return

bounds = Bounds(0, 1)
linear = LinearConstraint(np.ones(len(stocks.columns)), 1, 1)
constraint = NonlinearConstraint(fn_constraint, -np.inf, 0)
w0 = np.ones(len(stocks.columns)) / len(stocks.columns)
w0 /= np.sum(w0)

# Optimize the weights, calculate the returns and volatility
res = minimize(fn, w0, method='SLSQP', constraints=[constraint, linear], bounds=bounds)
assert round(sum(res.x), 4) == 1, "Weights don't sum to 1"
frontier_return = (res.x * stocks_returns).sum(axis=1)

# Plot the returns
plt.plot(portfolio_returns, linewidth=0.1)
plt.plot(frontier_return, label="Frontier", color="red", linewidth=2)
plt.plot(bm_returns, label="Benchmark", color="black", linestyle="--", linewidth=2)
plt.legend()
plt.show()

exit()
