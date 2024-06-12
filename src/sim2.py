import pandas as pd
from tqdm import tqdm

class Simulation2:
    """Simulate the time evolution of a portfolio."""

    def __init__(self, benchmark: pd.Series, returns: pd.DataFrame):
        # Calculate the total return of the benchmark
        self.benchmark = (1 + benchmark).cumprod() - 1
        self.returns = returns
        self.events = {}

    def add_event(self, date: pd.Timestamp, event: callable):
        """Change the weights of the portfolio."""
        self.events[date] = event

    def run(self, weights: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """Run the simulation from start_date to end_date."""

        # Setup the dataframe to store the results
        date_range = self.benchmark.loc[start_date:end_date].index
        df_performance = pd.DataFrame(index=date_range, columns=["benchmark", "portfolio"])

        # Calculate the returns of the benchmark
        df_performance["benchmark"] = self.benchmark.loc[date_range]

        # Prepare a vector for portfolio performance
        portfolio_returns = []

        # Loop over the days of the simulation
        for current_date in date_range:

            # Check if something happens on this day
            if current_date in self.events:
                weights = self.events[current_date](current_date)
                assert weights.sum() == 1, "The weights should sum to 1"

            # Save the day if it is a trading day
            if current_date in self.returns.index:
                performance = (weights * self.returns.loc[current_date]).sum()
                portfolio_returns.append(performance)
            else:
                portfolio_returns.append(0)

        # Assign portfolio returns to the dataframe
        df_performance["portfolio"] = portfolio_returns

        # Convert the performance to a cumulative performance
        df_performance["portfolio"] = (1 + df_performance["portfolio"]).cumprod()

        return df_performance