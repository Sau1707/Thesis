import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint, NonlinearConstraint
from scipy.optimize import minimize


# Read and reindex the dataframe to include all dates in the range
df = pd.read_csv("./data/smi.csv", index_col="Date", parse_dates=True)
data_full = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='D'))

# Fill NaN values again with forward fill method, remove all the rows with NaN values at the start
# We suppose that the days with missing values are not trading days, therefore the price don't change
df = data_full.ffill()
df = df[~df.isna().any(axis=1)]

# Normalize the data, make all the values start at 1
df = df / df.iloc[0]

# Conver the data to the right format Y / M / D
df = df.resample('YE').last()

# Calculate the returns
returns = df.pct_change()
returns.plot(figsize=(15, 10), title="Returns")
plt.savefig("./data/plots/returns.png")
plt.clf()

# With cumulative returns
cum_rets = (1+returns).cumprod()
cum_rets.plot(figsize=(15, 10), title="Cumulative Returns")
plt.savefig("./data/plots/cumulative_returns.png")
plt.clf()

# Calculate the average returns
mean_returns = returns.mean()
mean_returns.sort_values(ascending=False, inplace=True)
mean_returns.plot(kind="bar", figsize=(15, 10), title="Annual Returns")
plt.savefig("./data/plots/annual_returns.png")
plt.clf()

# Total returns sorted value
total_returns = cum_rets.iloc[-1, :]
total_returns.sort_values(ascending=False, inplace=True)
total_returns.plot(kind="bar", figsize=(15, 10), title="Total Returns")
plt.savefig("./data/plots/total_returns.png")
plt.clf()

# Calculate the volatility
volatility = returns.std() / np.sqrt(len(df))
volatility.sort_values(ascending=False, inplace=True)
volatility.plot(kind="bar", figsize=(15, 10), title="Volatility")
plt.savefig("./data/plots/volatility.png")
plt.clf()


def portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculate the returns and volatility of the portfolio"""
    returns = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, volatility


# Calculate the shape ratios
# https://corporatefinanceinstitute.com/resources/career-map/sell-side/risk-management/sharpe-ratio-definition-formula/#:~:text=Sharpe%20Ratio%20%3D%20(Rx%20%E2%80%93%20Rf,portfolio%20return%20(or%2C%20volatility) 
# < 1 Bad, 1 < x < 2 Acceptable, 2 < x < 3 Good, 3 < 4 Excellent
RISK_FREE_RATE = 0.03
shape_ratios = (total_returns - RISK_FREE_RATE) / volatility
shape_ratios.sort_values(ascending=False, inplace=True)
shape_ratios.plot(kind="bar", figsize=(15, 10), title="Shape Ratios")
plt.savefig("./data/plots/shape_ratios.png")
plt.clf()

# Create the covariance matrix and normalize it
cov_matrix = returns.cov()
cov_matrix = cov_matrix / cov_matrix.max().max()
plt.matshow(cov_matrix, cmap='hot')
plt.colorbar()
plt.savefig("./data/plots/covariance_matrix.png")

#############################################################
# Calculate the higher return for the lowest risk possible
#############################################################
fn_constraint = lambda w: w.T @ cov_matrix @ w
fn = lambda w: -w.T @ mean_returns # Minimize the negative return

performance = []
volatility = []

for i in tqdm.tqdm(range(100)):
    bounds = Bounds(0, 1)
    constraint = NonlinearConstraint(fn_constraint, -np.inf, 0.01 + i / 100)
    linear = LinearConstraint(np.ones(len(df.columns)), 1, 1)

    # Set the first weights to be equal
    weights = np.ones(len(df.columns))
    weights /= np.sum(weights)

    # Define the optimization problem
    res = minimize(fn, weights, method='trust-constr', constraints=[constraint, linear], bounds=bounds)
    res = res.x
    assert round(sum(res), 4) == 1, "Weights don't sum to 1"
    # res = res.x / np.sum(res.x)
    perf, vol = portfolio_performance(res, mean_returns, cov_matrix)
    performance.append(perf)
    volatility.append(vol)


plt.figure(figsize=(20,10))
plt.scatter(volatility, performance, c="blue")
plt.savefig("./data/plots/frontier.png")
# exit()


# Simuate the efficient frontier
N_PORTFOLIOS = 100_000_000


def generate_portfolios(num_portfolios, mean_returns, cov_matrix):
    results = np.zeros((3, num_portfolios))
    
    # Add some random portfolios
    for i in tqdm.tqdm(range(num_portfolios)):
        weights = np.random.dirichlet(np.ones(len(mean_returns)))
        assert round(sum(weights), 4), "Weights don't sum to 1"
    
        portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_return
        results[1,i] = portfolio_volatility
        results[2,i] = results[0,i] / results[1,i]  # Sharpe Ratio

    return results


results = generate_portfolios(N_PORTFOLIOS, mean_returns, cov_matrix)

columns = ['Return', 'Volatility', 'Sharpe Ratio']
portfolios = pd.DataFrame(results.T, columns=columns)

# Find the portfolio with the highest Sharpe Ratio
max_sharpe = portfolios.iloc[portfolios['Sharpe Ratio'].idxmax()]

# Find the portfolio with the lowest volatility
min_volatility = portfolios.iloc[portfolios['Volatility'].idxmin()]


# Plot the efficient frontier
plt.figure(figsize=(15, 10))
plt.scatter(portfolios['Volatility'], portfolios['Return'], c=portfolios['Sharpe Ratio'], cmap='viridis')
plt.scatter(max_sharpe.iloc[1], max_sharpe.iloc[0], marker='x', color='r', s=100, label='Max Sharpe Ratio')
plt.scatter(min_volatility.iloc[1], min_volatility.iloc[0], marker='x', color='g', s=100, label='Min Volatility')

# Plot the optimal portfolio
plt.scatter(volatility, performance, c="blue", label="Optimal Portfolio")

plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier')
plt.savefig("./data/plots/efficient_frontier.png")
plt.clf()