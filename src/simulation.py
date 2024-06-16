import tqdm
import pandas as pd
import matplotlib.pyplot as plt


class Simulation:
    """Simulate the portfolio over time"""
    # TODO: remove benchmark?
    # TODO: number of positions
    # TODO: sharp ratio

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

    def get_stocks(self, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None):
        """Return the stocks data, without the future data and the missing data"""
        return self.stocks.loc[start_date:end_date].dropna(axis=1, how='all')

    def run(self, weights: pd.DataFrame, *, start_date: pd.Timestamp, end_date: pd.Timestamp):
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
        previous_date = start_date
        for date, event in tqdm.tqdm(events.items(), desc="Running simulation"):
            # Get the historical data of the stocks for this period
            historical_stocks = self.get_stocks(previous_date, date)

            # Get the returns of the stocks in the period and filter out the stocks that are not in the portfolio
            historical_stocks = historical_stocks.ffill()
            returns = historical_stocks.pct_change()
            print(returns)
            returns = returns[weights.index]

            # Save the total return of the portfolio
            for portfolio in weights.columns:
                assert round(sum(weights[portfolio]), 2) == 1, f"The weights of the portfolio do not sum to 1, {sum(weights[portfolio])}"
                portfolios.loc[previous_date:date, portfolio] = (returns @ weights[portfolio])
            previous_date = date

            # The weights of the portfolio can be updated by the event
            if event is not None:
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
        plt.xlabel("Date")
        plt.ylabel("Returns")
        plt.legend()
    
    def plot_sharp_ratio(self):
        """https://www.investopedia.com/terms/s/sharperatio.asp"""

    def plot_number_of_positions(self):
        """Plot the number of positions in the portfolio"""
        # https://gurobi-finance.readthedocs.io/en/latest/modeling_notebooks/basic_model_maxmu.html