import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.constants import SMI, TICKERS, TICKERS_SW
from src.utils import get_data
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint, NonlinearConstraint
from scipy.optimize import minimize


class EfficientFrontier:
    def __init__(self, stocks: pd.DataFrame, benchmark: pd.Series):
        # Normalize the data
        stocks = stocks / stocks.iloc[0]
        benchmark = benchmark / benchmark.iloc[0]

        # Save the data
        self.stocks = stocks
        self.benchmark = benchmark

        # Calculate the returns of the benchmark
        self.benchmark_returns = benchmark.pct_change().ffill()
        self.benchmark_returns = (1 + self.benchmark_returns).cumprod()

        # Calculate the returns of the stocks
        self.stocks_returns = stocks.pct_change().ffill()
        self.stocks_returns = (1 + self.stocks_returns).cumprod()
        self.stocks_returns.iloc[0] = 1

    def get_benchmark_returns(self) -> pd.Series:
        return self.benchmark_returns
    
    def random_portfolios(self, n_portfolios: int) -> pd.DataFrame:
        """Return the weights of the random portfolios"""
        weights = np.random.dirichlet(np.ones(len(self.stocks.columns)), size=n_portfolios)
        portfolio_returns = self.stocks_returns.values @ weights.T
        return pd.DataFrame(portfolio_returns, index=self.stocks_returns.index)

    def mean_variance_portfolio(self, variance: float) -> pd.Series:
        """Calculate the efficient frontier of the portfolio
        Using:
        - Each stock can have a weight between 0 and 1
        - The sum of the weights must be 1
        - The estimated risk must not exceed a prespecified maximal admissible level of variance 
        - The objective is to minimize the negative return
        """
        correlation_matrix = self.stocks.corr()
        mean_returns = self.stocks_returns.mean()

        fn_constraint = lambda w: w.T @ correlation_matrix @ w
        fn = lambda w: -w.T @ mean_returns  # Minimize the negative return

        bounds = Bounds(0, 1)
        linear = LinearConstraint(np.ones(len(self.stocks.columns)), 1, 1)
        constraint = NonlinearConstraint(fn_constraint, -np.inf, variance)
        w0 = np.ones(len(self.stocks.columns)) / len(self.stocks.columns)
        w0 /= np.sum(w0)

        # Optimize the weights, calculate the returns and volatility
        res = minimize(fn, w0, method='SLSQP', constraints=[constraint, linear], bounds=bounds)

        # Normalize the weights to sum to 1
        weights = res.x / sum(res.x)
        frontier_return = (weights * self.stocks_returns).sum(axis=1)
        return frontier_return
    


bm, stocks = get_data(tickers=TICKERS_SW, benchmark=SMI)
stocks = stocks.loc["2000":].dropna(axis=1)
bm = bm.loc["2000":]

ef = EfficientFrontier(stocks, bm)
portfolio_returns = ef.random_portfolios(100)
frontier_return_0 = ef.mean_variance_portfolio(0.0)
frontier_return_1 = ef.mean_variance_portfolio(0.1)
frontier_return_2 = ef.mean_variance_portfolio(0.2)
frontier_return_3 = ef.mean_variance_portfolio(0.3)
frontier_return_4 = ef.mean_variance_portfolio(0.4)
frontier_return_5 = ef.mean_variance_portfolio(0.5)
frontier_return_6 = ef.mean_variance_portfolio(0.6)
bm_returns = ef.get_benchmark_returns()

# Plot the returns
plt.plot(portfolio_returns, alpha=0.1)
plt.plot(frontier_return_0, color='red', label='Frontier 0')
plt.plot(frontier_return_1, color='blue', label='Frontier 0.1')
plt.plot(frontier_return_2, color='black', label='Frontier 0.2')
plt.plot(frontier_return_3, color='yellow', label='Frontier 0.3')
plt.plot(frontier_return_4, color='purple', label='Frontier 0.4')
plt.plot(frontier_return_5, color='orange', label='Frontier 0.5')
plt.plot(frontier_return_6, color='brown', label='Frontier 0.6')
plt.plot(bm_returns, color='green', label='Benchmark')
plt.legend()
plt.show()

exit()
