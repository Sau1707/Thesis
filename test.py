import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


# Simuate the efficient frontier
N_PORTFOLIOS = 1_000_000


def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, volatility

def generate_portfolios(num_portfolios, mean_returns, cov_matrix):
    results = np.zeros((4, num_portfolios))
    weights_record = []
    
    for i in tqdm.tqdm(range(num_portfolios)):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_return
        results[1,i] = portfolio_volatility
        results[2,i] = results[0,i] / results[1,i]  # Sharpe Ratio
        results[3,i] = i  # Portfolio ID
    
    return results, weights_record


results, weights_record = generate_portfolios(N_PORTFOLIOS, mean_returns, cov_matrix)

columns = ['Return', 'Volatility', 'Sharpe Ratio', 'Portfolio ID']
portfolios = pd.DataFrame(results.T, columns=columns)

# Plot the efficient frontier
plt.figure(figsize=(15, 10))
plt.scatter(portfolios['Volatility'], portfolios['Return'], c=portfolios['Sharpe Ratio'], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier')
plt.savefig("./data/plots/efficient_frontier.png")
plt.clf()