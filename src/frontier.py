import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint, NonlinearConstraint
from scipy.optimize import minimize


class Frontier:
    # TODO: Create plot with random vs best frontier
    # TODO: Plot with sharp ratio
    # TODO: best years for data

    def __init__(self, stocks: pd.DataFrame, years: int = 10):
        # Make sure that the stocks don't contain columns with only NaN values
        assert not stocks.isnull().all().any(), "The stocks contain columns with only NaN values"
        last_date = stocks.index[-1]
        first_date = last_date - pd.DateOffset(years=years)
        # print(f"[Frontier] Using data from {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")

        # Save the columns of the stocks
        self.columns = stocks.columns
        # print(f"[Frontier] Dataset has {len(self.columns)} stocks")

        # Use the last X years of data for the stocks, filter out the stocks that don't have enough data
        stocks = stocks.loc[first_date:last_date]
        stocks = stocks.dropna(axis=1)
        stocks = stocks / stocks.iloc[0]
        self.stocks = stocks
        # print(f"[Frontier] Using {len(stocks.columns)} stocks")
    
        # Calculate the returns of the stocks
        self.stocks_returns = stocks.pct_change()
        self.stocks_returns = (1 + self.stocks_returns).cumprod()
        self.stocks_returns.iloc[0] = 1
    
    def random_portfolios(self, n_portfolios: int) -> pd.DataFrame:
        """Return the weights of the random portfolios"""
        weights = np.random.dirichlet(np.ones(len(self.columns)), size=n_portfolios)
        return pd.DataFrame(weights.T, index=self.columns)

    def mean_variance_portfolio(self, variance: float) -> pd.Series:
        """Calculate the efficient frontier of the portfolio
        Using:
        - Each stock can have a weight between 0 and 1
        - The sum of the weights must be 1
        - The estimated risk must not exceed a prespecified maximal admissible level of variance 
        - The objective is to minimize the negative return
        """
        # We use the stocks that have more than X years of data
        correlation_matrix = self.stocks.corr()
        mean_returns = self.stocks_returns.mean()

        fn_constraint = lambda w: w.T @ correlation_matrix @ w
        fn = lambda w: -w.T @ mean_returns  # Minimize the negative return

        bounds = Bounds(0, 1)
        linear = LinearConstraint(np.ones(len(self.stocks_returns.columns)), 1, 1)
        constraint = NonlinearConstraint(fn_constraint, -np.inf, variance)
        w0 = np.ones(len(self.stocks_returns.columns)) / len(self.stocks_returns.columns)
        w0 /= np.sum(w0)

        # Optimize the weights, calculate the returns and volatility
        res = minimize(fn, w0, method='SLSQP', constraints=[constraint, linear], bounds=bounds)
        weights = res.x / sum(res.x)
        weights = pd.Series(weights, index=self.stocks_returns.columns)

        # Add the missing stocks as 0
        weights = weights.reindex(self.columns, fill_value=0)
        return weights
    
    def mean_ENC_portfolio(self):
        """Calculate the efficient frontier of the portfolio
        Using:
        
        """
    

if __name__ == "__main__":
    from src.constants import TICKERS_SW, SMI
    from src.utils import get_data
    bm, stocks = get_data(tickers=TICKERS_SW, benchmark=SMI)
    ef = Frontier(stocks.loc["2019":, :])
    print(ef.mean_variance_portfolio(0.6))
    print(ef.random_portfolios(10))
