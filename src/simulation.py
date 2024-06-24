import pandas as pd
import matplotlib.pyplot as plt
from src.utils import Stocks


class Simulation:
    """Simulate the portfolio over time"""
    # TODO: remove benchmark?
    # TODO: number of positions
    # TODO: sharp ratio

    def __init__(self, benchmark: pd.Series, stocks: pd.DataFrame): 
        self.benchmark = benchmark
        self.stocks = stocks
        self._data = Stocks(stocks)

        # Get the max and the min date
        self._start = self.stocks.index.min()
        self._end = self.stocks.index.max()
        print(f"[Simulation] {self._start.strftime('%Y-%m-%d')} -> {self._end.strftime('%Y-%m-%d')}")

        # Keep track of the events and the portfolio value
        self._events = {}

    def add_event(self, date: pd.Timestamp, event: callable):
        """Add an event to the simulation"""
        assert date > self._start, "The event is before the start date"
        assert date < self._end, "The event is after the end date"
        self._events[date] = event
        print(f"[Simulation] Event added at {date.strftime('%Y-%m-%d')}")

    def get_stocks(self, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None):
        """Return all the stocks in the range of dates"""
        return self._data.get_historical(start=start_date, end=end_date, valid=True)
    
    def run(self, weights: pd.DataFrame, *, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """Simulate the portfolio over time"""
        assert start_date >= self._start, "The start date is before the simulation start date"
        assert end_date <= self._end, "The end date is after the simulation end date"
        assert start_date < end_date, "The start date is after the end date"

        # Get the requited data range and normalize it to the start date
        benchmark = self.benchmark.loc[start_date:end_date]
        benchmark: pd.Series = benchmark / benchmark.iloc[0]
        stocks = self.stocks.loc[start_date:end_date]
        stocks: pd.DataFrame  = stocks / stocks.iloc[0]

        # Sort the events by date and add the last date
        events = {k: v for k, v in sorted(self._events.items(), key=lambda item: item[0])}
        events[end_date] = None

        # Create the portfolio, where the column is the total return
        portfolios = pd.DataFrame(index=stocks.index, columns=weights.columns)
        previous_date = start_date
 
        for date, event in events.items():
            # Get the returns of the stocks in the period
            returns = self._data.get_returns(start=previous_date, end=date, valid=True)

            # Save the total return of the portfolio
            for portfolio in weights.columns:
                assert round(sum(weights[portfolio]), 2) == 1, f"The weights of the portfolio do not sum to 1, {sum(weights[portfolio])}"
                returns = returns[weights[portfolio].index]
                portfolios.loc[previous_date:date, portfolio] = (returns @ weights[portfolio])
            previous_date = date

            # The weights of the portfolio can be updated by the event
            if event is not None:
                historical_stocks = self._data.get_historical(end=date, valid=False)
                print(historical_stocks)
                weights = event(historical_stocks, weights)

        # Calculate the returns of the benchmark
        bm_returns = benchmark.pct_change()
        bm_returns = (1 + bm_returns).cumprod()

        # Calculate the returns of the portfolios
        portfolios = (1 + portfolios).cumprod()

        return bm_returns, portfolios
    
    def plot(self, benchmark: pd.Series, portfolios: pd.DataFrame):
        """Plot the returns of the portfolio"""
        # Plot vertical lines for the events
        plt.figure(figsize=(12, 8))

        for i, date in enumerate(self._events.keys()):
            if i == 0:
                plt.axvline(date, color="red", linestyle="--", label="Event")
            else:
                plt.axvline(date, color="red", linestyle="--")
        
        # Plot the returns of the portfolios
        for portfolio in portfolios.columns:
            plt.plot(portfolios[portfolio], label=portfolio)

        # Plot the returns of the benchmark
        plt.plot(benchmark, label="Benchmark")
        plt.xlabel("Date")
        plt.ylabel("Returns")
        plt.legend()
    
    def plot_sharp_ratio(self):
        """https://www.investopedia.com/terms/s/sharperatio.asp"""

    def plot_number_of_positions(self):
        """Plot the number of positions in the portfolio"""
        # https://gurobi-finance.readthedocs.io/en/latest/modeling_notebooks/basic_model_maxmu.html