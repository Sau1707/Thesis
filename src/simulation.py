import pandas as pd
import matplotlib.pyplot as plt


class Simulation:
    """Simulate the portfolio over time"""

    def __init__(self, benchmark: pd.Series, stocks: pd.DataFrame): 
        self.benchmark = benchmark
        self.stocks = stocks

        # Get the max and the min date
        self.start = self.stocks.index.min()
        self.end = self.stocks.index.max()

        # Keep track of the events and the portfolio value
        self.events = {}

    def add_event(self, date: pd.Timestamp, event: callable):
        """Add an event to the simulation"""
        assert date > self.start, "The event is before the start date"
        assert date < self.end, "The event is after the end date"
        self.events[date] = event

    def run(self, weights: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """Simulate the portfolio over time"""
        assert start_date >= self.start, "The start date is before the simulation start date"
        assert end_date <= self.end, "The end date is after the simulation end date"
        assert start_date < end_date, "The start date is after the end date"

        # Get the requited data range and normalize it to the start date
        benchmark = self.benchmark.loc[start_date:end_date]
        benchmark: pd.Series = benchmark / benchmark.iloc[0]
        stocks = self.stocks.loc[start_date:end_date]
        stocks: pd.DataFrame  = stocks / stocks.iloc[0]

        # Sort the events by date and add the last date
        events = {k: v for k, v in sorted(self.events.items(), key=lambda item: item[0])}
        events[end_date] = None

        # Create the portfolio, where the column is the total return
        portfolios = pd.DataFrame(index=stocks.index, columns=weights.columns)

        for date, event in events.items():
            # When an event is triggered, get the new weights using the historical data
            if event is not None:
                historical_stocks = self.stocks.loc[:date]
                weights = event(historical_stocks)

            # Get the returns of the stocks in the period
            returns = stocks.loc[:date].pct_change()
            total_returns = (1 + returns).cumprod()

            # Save the total return of the portfolio
            for portfolio in weights.columns:
                portfolios.loc[:date, portfolio] = (total_returns @ weights[portfolio])

        # Calculate the returns of the benchmark
        bm_returns = benchmark.pct_change()
        bm_returns = (1 + bm_returns).cumprod()

        return bm_returns, portfolios
    
    def plot(self, benchmark: pd.Series, portfolios: pd.DataFrame):
        """Plot the returns of the portfolio"""
        # Plot vertical lines for the events
        for i, date in enumerate(self.events.keys()):
            if i == 0:
                plt.axvline(date, color="red", linestyle="--", label="Event")
            else:
                plt.axvline(date, color="red", linestyle="--")
        
        # Plot the returns of the portfolios
        for portfolio in portfolios.columns:
            plt.plot(portfolios[portfolio], label=portfolio)

        # Plot the returns of the benchmark
        plt.plot(benchmark, label="Benchmark")

        plt.legend()
        plt.show()
   