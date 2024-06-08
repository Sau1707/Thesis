import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from src import IPlot, get_data
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint, NonlinearConstraint
from scipy.optimize import minimize
np.random.seed(0)

RISK_FREE_RATE = 0.03
# Get the data from 2000 to 2020, leave the rest for testing
smi, stocks = get_data(start_year=2000, end_year=2020, normalize=True)

plot = IPlot(stocks, smi)
# plot.returns()
# plot.cumulative_returns()
# plot.average_returns()
# plot.standard_deviation()
# plot.correlation()
# plot.sharpe_ratio()


def portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculate the returns and volatility of the portfolio"""
    returns = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, volatility

# Calculate the correlation matrix
mean_returns = stocks.pct_change().mean()
corr = stocks.corr()

# Define the optimization problem
fn_constraint = lambda w: w.T @ corr @ w
fn = lambda w: -w.T @ mean_returns # Minimize the negative return

# Define the bounds
bounds = Bounds(0, 1)
w0 = np.ones(len(stocks.columns)) / len(stocks.columns)
w0 /= np.sum(w0)

portfolios = pd.DataFrame(columns=["Returns", "Volatility", "Sharpe Ratio"])
for i in tqdm(range(20)):
    constraint = NonlinearConstraint(fn_constraint, -np.inf, 0.01 + i / 20)
    linear = LinearConstraint(np.ones(len(stocks.columns)), 1, 1)
    
    # Optimize the weights
    res = minimize(fn, w0, method='SLSQP', constraints=[constraint, linear], bounds=bounds)
    w = res.x
    assert round(sum(w), 4) == 1, "Weights don't sum to 1"

    # Calculate the returns and volatility
    ret, vol = portfolio_performance(w, mean_returns, corr)
    sharpe = (ret - RISK_FREE_RATE) / vol
    portfolios.loc[i] = [ret, vol, sharpe]


plt.figure(figsize=(20,10))
plt.scatter(portfolios["Volatility"], portfolios["Returns"], c=portfolios["Sharpe Ratio"], cmap='viridis')

# Fit the efficient frontier from the points
z = np.polyfit(portfolios["Volatility"], portfolios["Returns"], 2)
p = np.poly1d(z)
plt.plot(portfolios["Volatility"], p(portfolios["Volatility"]), "r--")

# Generate random portfolios
for i in tqdm(range(1_000)):
    w = np.random.dirichlet(np.ones(len(stocks.columns)), size=1)[0]
    ret, vol = portfolio_performance(w, mean_returns, corr)
    sharpe = (ret - RISK_FREE_RATE) / vol
    plt.scatter(vol, ret, c=sharpe, cmap='viridis')

plt.savefig("./data/plots/frontier.png")