import numpy as np
import pandas as pd
import gurobipy as gp
import matplotlib.pyplot as plt


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
        
        df.columns = [f"random_{i}" for i in range(n_portfolios)]
        return df
    
    def mean_ENC_portfolio(self, N: int | list[int]):
        """Calculate the efficient frontier of the portfolio
        Using:
        TODO: 
        """
        if isinstance(N, int):
            N = [N]

        return df.fillna(0)

    def mean_variance_portfolio(self, variance: float | list[float]) -> pd.DataFrame:
        """"""
        if isinstance(variance, float):
            variance = [variance]

        # Initialize the model
        m = gp.Model()
        m.params.OutputFlag = 0

        # Setup the optimization problem
        x = m.addMVar(len(self._mean_returns), lb=0, ub=1, name="x") # 0 <= x[i] <= 1
        m.addConstr(x.sum() == 1, name="Budget_Constraint") # all investments sum up to 1
        m.setObjective(self._mean_returns.to_numpy() @ x, gp.GRB.MAXIMIZE)

        df = pd.DataFrame(index=self._columns)
        variance_constr = None
        for v in variance:
            # Remove previous variance constraint if it exists
            if variance_constr is not None:
                m.remove(variance_constr)

            # Limit on variance
            variance_constr = m.addConstr(x @ self._correlation_matrix.to_numpy() @ x <= v**2, name="Variance")
            m.optimize()

            # Print investments (with non-negligible values, i.e., > 1e-5)
            positions = pd.Series(name="Position", data=x.X, index=self._mean_returns.index)
            df[f"max_returns_{v}"] = positions[positions > 1e-5]

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
        plt.xlabel("Standard Deviation ($\sigma$)")
        plt.ylabel("Returns ($\mu$)")
        plt.colorbar(label='Sharpe Ratio')

        columns = [col for col in weights.columns if not col.startswith("random")]
        volatility = volatility[columns]
        df = df[columns]
        num_constituents = df.apply(lambda x: (x > 0.0001).sum())
        sharp_ratio = sharp_ratio[columns]
        
        plt.subplot(1, 2, 2)
        plt.scatter(volatility, num_constituents, c=sharp_ratio, cmap='viridis')
        plt.xlabel("Standard Deviation ($\sigma$)")
        plt.ylabel("Number of Constituents")
        plt.colorbar(label='Sharpe Ratio')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from src.utils import Dataset
    data =  Dataset("SW", "1995", "^SSMI")
    bm = data.get_benchmark()
    stocks = data.get_data(liquidity=0.9)
    np.random.seed(0)

    ef = Frontier(stocks)
    df = pd.concat([
        ef.mean_variance_portfolio([i/10 for i in range(1, 10)]),
        ef.random_portfolios(1_000, use_years=True)
    ], axis=1)

    print(df)

    ef.plot(df)
