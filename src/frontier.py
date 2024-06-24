import numpy as np
import pandas as pd
import gurobipy as gp
import matplotlib.pyplot as plt
from src.utils import Stocks


class Frontier:
    def __init__(self, stocks: pd.DataFrame, years: int = 20):
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
        self._data = Stocks(stocks)
        
        # Get the range of the calculations
        self._end_date = self._data.get_end_date()
        self._start_date = self._end_date - pd.DateOffset(years=years)

        # Save the current stocks (columns of the dataframe)
        self._columns = self._data.get_stocks(date=self._end_date)

        # Get the data for the calculations
        self._stocks = self._data.get_historical(start=self._start_date, end=self._end_date, valid=True)
        self._returns = self._data.get_returns(start=self._start_date, end=self._end_date, valid=True)
        self._total_return = self._data.get_total_return(start=self._start_date, end=self._end_date,  valid=True)
        
        # Some utilities
        self._mean_returns = (1 + self._returns).prod() ** (252 / self._returns.count()) - 1
        self._correlation_matrix = self._stocks.corr()


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
    
    def mean_ENC_portfolio(self, N: int, u: float = 0.15, l: float = 0.005, max_returns = 1) -> pd.DataFrame:
        """Calculate the efficient frontier of the portfolio"""
        df = pd.DataFrame(index=self._columns)
        constrain = None

        returns = 0
        step = 0.01

        # Initialize the model
        m = gp.Model()
        m.params.OutputFlag = 0

        # Setup the optimization problem
        x = m.addMVar(len(self._mean_returns), lb=0, ub=u, name="x")
        b = m.addMVar(len(self._mean_returns), vtype=gp.GRB.BINARY, name="b")
        m.addConstr(x.sum() == 1, name="Budget_Constraint")
        m.addConstr(x <= b, name="Indicator")
        m.addConstr(x >= l * b, name="Minimal_Position")
        m.addConstr(b.sum() >= N, name="Diversification")
        m.setObjective(x @ self._correlation_matrix.to_numpy() @ x, gp.GRB.MINIMIZE)

        best_weights = None
        while returns < max_returns and step > 1e-5:
            # Remove previous variance constraint if it exists
            if constrain is not None:
                m.remove(constrain)

            # print(f"Returns: {returns:.6f}")
            # Add the constraint with the current returns
            constrain = m.addConstr(self._mean_returns.to_numpy() @ x >= returns, name="Minimal_Return")
            m.optimize()

            if m.Status == gp.GRB.OPTIMAL:
                best_weights = x.X
                returns += step
            else:
                returns -= step
                step /= 10
        # Print investments (with non-negligible value, i.e. >1e-5)
        positions = pd.Series(name="Position", data=best_weights, index=self._mean_returns.index)
        df[f"ENC"] = positions[positions > 1e-5]

        return df.fillna(0)

    def mean_variance_portfolio(self, min_variance: float = 0) -> pd.DataFrame:
        """Return the minimum variance portfolio"""
        df = pd.DataFrame(index=self._columns)
        constrain = None

        # Keep track of the current variance and the current step, we start from 1 and go down
        variance = 1
        step = 0.1

        # Initialize the model
        m = gp.Model()
        m.params.OutputFlag = 0

        # Setup the optimization problem
        x = m.addMVar(len(self._mean_returns), lb=0, ub=1, name="x")         # 0 <= x[i] <= 1
        m.addConstr(x.sum() == 1, name="Budget_Constraint")                  # All investments sum up to 1
        m.setObjective(self._mean_returns.to_numpy() @ x, gp.GRB.MAXIMIZE)   # Objective function

        best_weights = None
        while variance > min_variance and step > 1e-5:
            # Remove previous variance constraint if it exists
            if constrain is not None:
                m.remove(constrain)

            # Add the constraint with the current variance
            # print(f"Variance: {variance:.6f}")
            constrain = m.addConstr(x @ self._correlation_matrix.to_numpy() @ x <= variance**2, name="Variance")
            m.optimize()

            if m.Status == gp.GRB.OPTIMAL:
                best_weights = x.X
                variance -= step
            else:
                variance += step
                step /= 10
        
        # Save the results
        positions = pd.Series(name="Position", data=best_weights, index=self._mean_returns.index)
        df[f"min_variance"] = positions[positions > 1e-5]
            
        return df.fillna(0)
    
    ##################################################################################
    # Plots
    ##################################################################################
    def plot(self, weights: pd.DataFrame):
        """
        Plot the efficient frontier and number of constituents
            weights: 
            - A DataFrame with the weights of the stocks

        TODO: transform this in a table
        """
        df = weights.loc[self._returns.columns]
        returns = df.apply(self.returns)
        volatility = df.apply(self.volatility)
        sharp_ratio = df.apply(lambda x: self.sharp_ratio(x, 0.0659))  # 10 year treasury bond rate

        # Efficient frontier plot
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(volatility, returns, c=sharp_ratio, cmap='viridis')
        plt.xlabel(r"Standard Deviation ($\sigma$)")
        plt.ylabel(r"Returns ($\mu$)")
        plt.colorbar(label='Sharpe Ratio')

        columns = [col for col in weights.columns if not col.startswith("random")]
        volatility = volatility[columns]
        df = df[columns]
        num_constituents = df.apply(lambda x: (x > 0.0001).sum())
        sharp_ratio = sharp_ratio[columns]
        
        plt.subplot(1, 2, 2)
        plt.scatter(volatility, num_constituents, c=sharp_ratio, cmap='viridis')
        plt.xlabel(r"Standard Deviation")
        plt.ylabel(r"Number of Constituents")
        plt.colorbar(label='Sharpe Ratio')

        plt.tight_layout()
        plt.show()

    def plot_returns(self, weights: pd.DataFrame):
        """Plot the returns of the portfolios"""
        plt.figure(figsize=(12, 6))

        total_return = self._data.get_total_return(start=self._start_date, end=self._end_date, valid=False)
        total_return = total_return.fillna(0)

        # Plot the returns of the random portfolios
        returns = weights.apply(lambda x: total_return @ x)
        returns_random = returns[[col for col in returns.columns if col.startswith("random")]]
        returns_random = returns_random.mean(axis=1)
        plt.plot(returns_random, label="Random Portfolios")

        # Plot the returns of the efficient portfolios
        returns = returns[[col for col in returns.columns if not col.startswith("random")]]
        for col in returns.columns:
            print(col, len(weights[col][weights[col] > 1e-5]))
            plt.plot(returns[col], label=col)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    from src.dataset import Dataset
    data =  Dataset("SW", "1995", "^SSMI")
    bm = data.get_benchmark()
    stocks = data.get_data(liquidity=0.99)
    # np.random.seed(0)

    ef = Frontier(stocks)
    
    df = pd.concat([
        ef.mean_ENC_portfolio(50),
        ef.mean_variance_portfolio(0.6),
        ef.random_portfolios(20, use_years=True)
    ], axis=1)

    ef.plot_returns(df)
    # ef.plot(df)
