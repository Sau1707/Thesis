import os
import pandas as pd
import matplotlib.pyplot as plt

# Create the plot folder if it does not exist
PLOTS = "data/benchmark"
if not os.path.exists(PLOTS):
    os.makedirs(PLOTS)


class Benchmark:
    def __init__(self, df: pd.DataFrame):
        assert isinstance(df.index, pd.DatetimeIndex), "Index must be a DatetimeIndex"
        self._df = df

    # Tested on https://www.1stock1.com/1stock1_763.htm
    def get_returns(self, time_period: str = "y") -> pd.Series:
        """Return the total returns of the benchmark."""
        df = self._df.pct_change()
        if time_period == "y":
            df = df.resample("YE").apply(lambda x: (x + 1).prod() - 1)
        elif time_period == "m":
            df = df.resample("M").apply(lambda x: (x + 1).prod() - 1)
        elif time_period == "d":
            df = df
        else:
            raise ValueError("time_period must be 'y', 'm' or 'd'")
        return df

    def get_average_returns(self, time_period: str = "y") -> pd.Series:
        """Return the average returns of the benchmark."""
        df = self.get_returns(time_period)
        return df.mean()
    
    def get_cumulative_returns(self, time_period: str = "y") -> pd.Series:
        """Return the cumulative returns of the benchmark."""
        df = self.get_returns(time_period)
        return (1 + df).cumprod() - 1

    ##############################################################
    # Plotting methods
    ##############################################################
    def plot_returns(self, time_period: str = "y", path: str = "data/benchmark/returns.png"):
        """Plot the returns of the benchmark."""
        df = self.get_returns(time_period)
        df.plot()
        plt.savefig(path, dpi=300)
    
    def plot_cumulative_returns(self, time_period: str = "y", path: str = "data/benchmark/cumulative_returns.png"):
        """Plot the cumulative returns of the benchmark."""
        df = self.get_cumulative_returns(time_period)
        df.plot()
        plt.savefig(path, dpi=300)