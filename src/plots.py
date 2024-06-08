import os
import pandas as pd
import matplotlib.pyplot as plt


PLOTS = "data/plots"
if not os.path.exists(PLOTS):
    os.makedirs(PLOTS)


class IPlot:
    def __init__(self, stocks: pd.DataFrame, benchmark: pd.Series):
        self.stocks = stocks
        self.benchmark = benchmark

    def returns(self, benchmark=True, show=False):
        """Plot the returns of the stocks and the benchmark."""
        returns = self.stocks.pct_change()
        returns.plot(figsize=(15, 10), title="Returns")

        # Plot the benchmark
        if benchmark:
            benchmark_returns = self.benchmark.pct_change()
            benchmark_returns.plot(label="Benchmark", color="black", linewidth=2.0)

        if show:
            plt.show()
        plt.savefig(f"{PLOTS}/returns.png", dpi=300)
        plt.clf()
    
    def cumulative_returns(self, benchmark=True, show=False):
        """Plot the cumulative returns of the stocks."""
        cum_rets = (1 + self.stocks.pct_change()).cumprod()
        cum_rets.plot(figsize=(15, 10), title="Cumulative Returns")

        # Plot the benchmark
        if benchmark:
            cum_benchmark = (1 + self.benchmark.pct_change()).cumprod()
            cum_benchmark.plot(label="Benchmark", color="black", linewidth=4.0)
        
        if show:
            plt.show()
        plt.savefig(f"{PLOTS}/cumulative_returns.png", dpi=300)
        plt.clf()

    def average_returns(self, benchmark=True, show=False):
        """Plot the average yearly returns of the stocks."""
        avg_rets = self.stocks.pct_change().mean() * 252
        avg_rets.sort_values(ascending=False, inplace=True)
        avg_rets.plot(kind="bar", figsize=(15, 10), title="Average Returns")

        # Plot the benchmark as a line
        if benchmark:
            avg_benchmark = self.benchmark.pct_change().mean() * 252
            plt.axhline(y=avg_benchmark, color="red", linestyle="--", label="Benchmark")
        
        if show:
            plt.show()
        plt.savefig(f"{PLOTS}/average_returns.png", dpi=300)
        plt.clf()
    
    def standard_deviation(self, benchmark=True, show=False):
        """Plot the standard deviation of the returns of the stocks."""
        std_rets = self.stocks.pct_change().std() * (252 ** 0.5)
        std_rets.sort_values(ascending=False, inplace=True)
        std_rets.plot(kind="bar", figsize=(15, 10), title="Standard Deviation of Returns")

        # Plot the benchmark
        if benchmark:
            std_benchmark = self.benchmark.pct_change().std() * (252 ** 0.5)
            plt.axhline(y=std_benchmark, color="red", linestyle="--", label="Benchmark")
        
        if show:
            plt.show()

        plt.savefig(f"{PLOTS}/std_returns.png", dpi=300)
        plt.clf()
        