import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint, NonlinearConstraint
from scipy.optimize import minimize




class IOptimizer:
    def __init__(self, stocks: pd.DataFrame, smi: pd.DataFrame, risk_free_rate: float = 0.03):
        self.stocks = stocks
        self.smi = smi
        self.risk_free_rate = risk_free_rate

        # Calculate the correlation matrix
        self.mean_returns = self.stocks.pct_change().mean()
        self.correlation_matrix = self.stocks.corr()

        # Create the plot 
        plt.figure(figsize=(20,10))
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.title('Efficient Frontier')

    def portfolio_performance(self, weights: pd.DataFrame):
        """Calculate the returns and volatility of the portfolio"""
        returns = weights @ self.mean_returns
        volatility = np.sqrt(weights.T @ self.correlation_matrix @ weights)
        return returns, volatility

    def efficient_frontier(self, count: int = 20, step: float = 0.1):
        """Calculate the efficient frontier of the portfolio"""
        # Define the optimization problem
        fn_constraint = lambda w: w.T @ self.correlation_matrix @ w
        fn = lambda w: -w.T @ self.mean_returns  # Minimize the negative return

        # Define the bounds
        bounds = Bounds(0, 1)
        w0 = np.ones(len(self.stocks.columns)) / len(self.stocks.columns)
        w0 /= np.sum(w0)
        linear = LinearConstraint(np.ones(len(self.stocks.columns)), 1, 1)

        portfolios = pd.DataFrame(columns=["Returns", "Volatility", "Sharpe Ratio"])
        for i in tqdm(range(count), desc="Calculating Efficient Frontier"):
            constraint = NonlinearConstraint(fn_constraint, -np.inf, i * step)
            
            # Optimize the weights
            res = minimize(fn, w0, method='SLSQP', constraints=[constraint, linear], bounds=bounds)
            w = res.x
            assert round(sum(w), 4) == 1, "Weights don't sum to 1"

            # Calculate the returns and volatility
            ret, vol = self.portfolio_performance(w)
            sharpe = (ret - self.risk_free_rate) / vol
            portfolios.loc[i] = [ret, vol, sharpe]

        # Plot the results
        plt.scatter(portfolios["Volatility"], portfolios["Returns"], c=portfolios["Sharpe Ratio"], cmap='viridis')

        # Fit the efficient frontier from the points
        z = np.polyfit(portfolios["Volatility"], portfolios["Returns"], 2)
        p = np.poly1d(z)
        plt.plot(portfolios["Volatility"], p(portfolios["Volatility"]), "r--")
        plt.colorbar(label='Sharpe Ratio')
        plt.savefig("./data/plots/frontier.png")

    def random_portfolios(self, n_portfolios: int = 1_000_000, chunk_size: int = 100_000):
        """Generate random portfolios"""
        portfolios = pd.DataFrame(columns=["Returns", "Volatility", "Sharpe Ratio"], index=range(n_portfolios))

        # Generate and process portfolios in chunks
        for chunk_start in tqdm(range(0, n_portfolios, chunk_size), desc="Generating Portfolios"):
            chunk_end = min(chunk_start + chunk_size, n_portfolios)
            weights = np.random.dirichlet(np.ones(len(self.stocks.columns)), chunk_end - chunk_start)

            # Compute portfolio returns and volatilities for the chunk
            returns = np.dot(weights, self.mean_returns)
            volatilities = np.sqrt(np.einsum('ij,jk,ik->i', weights, self.correlation_matrix, weights))
            sharpe_ratios = (returns - self.risk_free_rate) / volatilities

            # Save the results
            portfolios.loc[chunk_start:chunk_end] = pd.DataFrame({"Returns": returns, "Volatility": volatilities, "Sharpe Ratio": sharpe_ratios})

        # Plot the results
        plt.scatter(portfolios["Volatility"], portfolios["Returns"], c=portfolios["Sharpe Ratio"], cmap='viridis')
        plt.savefig("./data/plots/frontier.png")

    def add_point(self, returns: float, volatility: float):
        """Add a point to the efficient frontier"""
        plt.scatter(volatility, returns, color='red', marker='x')
        plt.savefig("./data/plots/frontier.png")