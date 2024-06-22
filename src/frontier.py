import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint, NonlinearConstraint
from scipy.optimize import minimize


class Frontier:
    # TODO: Create plot with random vs best frontier
    # TODO: Plot with sharp ratio
    # TODO: best years for data ? 

    def __init__(self, stocks: pd.DataFrame, years: int = 10):
        """
            stocks: 
            - A DataFrame the the stocks closing adjusted prices as columns
            - The index is a DatetimeIndex
            - Each stock can start and end in a different date

            years:
            - Every time the historical data is requested, only the companies that have data for the last X years are used
        """
        # Make sure that all the stocks have data
        assert not stocks.empty, "No stocks to invest in"
        self._columns = stocks.columns

        # Get the simulation range
        last_date = stocks.index[-2] # Never use the last date as it can be incomplete
        first_date = last_date - pd.DateOffset(years=years)
  
        # Use the last X years of data for the stocks, and remove the ones that have no data
        stocks = stocks.loc[first_date:last_date]
        stocks = stocks.replace(0, np.nan)
        stocks = stocks.dropna(axis=1)
        self._stocks = stocks
 
        # Calculate the returns of the stocks
        stocks = stocks.fillna(0)
        self._returns = stocks.pct_change()
        self._returns = (1 + self._returns).cumprod() - 1
        self._returns.iloc[0] = 1

        # Some utilities
        self._correlation_matrix = self._stocks.corr()
        self._mean_returns = self._returns.mean()

    ##################################################################################
    # Utilities
    ##################################################################################
    def returns(self, weights: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        Calculate the returns of the portfolio
            weights: 
            - A Series with the weights of the stocks
        """
        return weights.T @ self._mean_returns
    
    def volatility(self, weights: pd.Series) -> float:
        """
        Calculate the volatility of the portfolio
            weights: 
            - A Series with the weights of the stocks
        """
        return np.sqrt(weights @ self._correlation_matrix @ weights)

    def sharp_ratio(self, weights: pd.Series, risk_free_rate: float) -> float:
        """
        Calculate the sharp ratio of the portfolio
            weights: 
            - A Series with the weights of the stocks
            risk_free_rate:
            - The risk free rate
        """
        return (self.returns(weights) - risk_free_rate) / self.volatility(weights)
    
    ##################################################################################
    # Portfolios generation
    ##################################################################################
    def random_portfolios(self, n_portfolios: int, use_years = True) -> pd.DataFrame:
        """
        Generate some random portfolios using the Dirichlet distribution
            n_portfolios: 
            - Cpunt of portfolios to generate
            use_years:
            - If True, the stocks that have more than X years of data are used
        """
        if use_years:
            weights = np.random.dirichlet(np.ones(len(self._stocks.columns)), size=n_portfolios)
            df = pd.DataFrame(weights.T, index=self._stocks.columns)
            df = df.reindex(self._columns, fill_value=0)
        else:
            weights = np.random.dirichlet(np.ones(len(self._columns)), size=n_portfolios)
            df = pd.DataFrame(weights.T, index=self._columns)
        return df
    
    def mean_variance_portfolio(self, variance: float | list[float]) -> pd.Series:
        """
        Calculate the efficient frontier of the portfolio
            Using:
            - Each stock can have a weight between 0 and 1
            - The sum of the weights must be 1
            - The estimated risk must not exceed a prespecified maximal admissible level of variance 
            - The objective is to minimize the negative return
        """
        if isinstance(variance, float):
            variance = [variance]

        # We use the stocks that have more than X years of data
        fn_constraint = lambda w: w.T @ self._correlation_matrix @ w

        bounds = Bounds(0, 1)
        linear = LinearConstraint(np.ones(len(self._returns.columns)), 1, 1)

        w0 = np.ones(len(self._returns.columns)) / len(self._returns.columns)
        w0 /= np.sum(w0)
        
        df = pd.DataFrame(index=self._columns)
        for v in tqdm.tqdm(variance, desc="Calculating Mean Variance Portfolios..."):
            constraint = NonlinearConstraint(fn_constraint, -np.inf, v**2)

            # Optimize the weights, calculate the returns and volatility
            res = minimize(lambda x: - self.returns(x), w0, method='SLSQP', constraints=[constraint, linear], bounds=bounds) #  options={'disp': True}
            assert res.success, f"Optimization failed: {res.message}"
            weights = res.x / sum(res.x)
            weights = pd.Series(weights, index=self._returns.columns)
            df[f"variance_{v}"] = weights

        return df.fillna(0)
    
    def mean_ENC_portfolio(self, N: int | list[int]):
        """Calculate the efficient frontier of the portfolio
        Using:
        TODO: 
        """
        if isinstance(N, int):
            N = [N]

        w0 = np.ones(len(self._returns.columns)) / len(self._returns.columns)
        w0 /= np.sum(w0)

        df = pd.DataFrame(index=self._columns)
        for n in tqdm.tqdm(N, desc="Calculating Mean ENC Portfolios..."):
            gamma = 0.5

            # Add the constraint that the number of constituents must be N
            fn_constraint = lambda w: (w > 0.0001).sum() - n
            constraint = NonlinearConstraint(fn_constraint, 0, 0)

            
            fn = lambda x: - (1 - gamma) * x @ self._mean_returns + gamma * (n * x.T @ self._correlation_matrix @ x + 1 / n)

            res = minimize(fn, w0, method='SLSQP', options={'disp': True}, constraints=[constraint])
            weights = res.x / sum(res.x)
            weights = pd.Series(weights, index=self._returns.columns)
            df[f"N_{n}"] = weights
        #for gamma in tqdm.tqdm([i/10 for i in range(1, 10)], desc="Calculating Mean ENC Portfolios..."):
        #    
        #
        #    res = minimize(fn, w0, method='BFGS', options={'disp': True})
        #    weights = res.x / sum(res.x)
        #    weights = pd.Series(weights, index=self._returns.columns)
        #    df[f"gamma_{gamma}"] = weights

        return df.fillna(0)
    
    ##################################################################################
    # Plots
    ##################################################################################
    def plot(self, weights: pd.DataFrame):
        """
        Plot the efficient frontier and number of constituents
            weights: 
            - A DataFrame with the weights of the stocks
        """
        df = weights.loc[self._returns.columns]
        returns = df.apply(self.returns)
        volatility = df.apply(self.volatility)
        sharp_ratio = df.apply(lambda x: self.sharp_ratio(x, 0.0659))  # 10 year treasury bond rate

        # Efficient frontier plot
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(volatility, returns, c=sharp_ratio, cmap='viridis')
        plt.xlabel("Volatility")
        plt.ylabel("Returns")
        plt.colorbar(label='Sharpe Ratio')
        plt.legend(df.columns)
        plt.title("Efficient Frontier")

        # Number of constituents vs returns
        num_constituents = df.apply(lambda x: (x > 0.0001).sum())

        # Filter out all the portfolios that have more then 80% of the stocks
        num_constituents = num_constituents #[num_constituents < 0.8 * len(df.columns)]
        returns = returns[num_constituents.index]
        sharp_ratio = sharp_ratio[num_constituents.index]
        
        plt.subplot(1, 2, 2)
        plt.scatter(num_constituents, returns, c=sharp_ratio, cmap='viridis')
        plt.xlabel("Number of Constituents")
        plt.ylabel("Returns")
        plt.colorbar(label='Sharpe Ratio')
        plt.title("Number of Constituents vs Returns")

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    from src.utils import get_data
    pd.set_option('display.max_rows', 10)
    # pd.set_option('display.max_columns', None)
    np.random.seed(0)

    bm, stocks = get_data()
    # Get 20 random columns
    # stocks = stocks.sample(n=10, axis=1)

    ef = Frontier(stocks)
    dfs = [
        # ef.mean_variance_portfolio([i/5 for i in range(1, 5)]),
        ef.mean_ENC_portfolio([i for i in range(1, 20)]),
        #ef.random_portfolios(5, use_years=True)
    ] 

    df = pd.concat(dfs, axis=1)
    print(df)

    ef.plot(df)
    # 
    # print(ef.random_portfolios(10, use_years=False))
