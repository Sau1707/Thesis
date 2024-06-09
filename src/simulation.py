from dataclasses import dataclass
import pandas as pd

@dataclass
class Day:
    day : pd.Timestamp
    weights: pd.Series
    returns: pd.Series


class Simulation:
    """Simulate the time evolution of a portfolio."""

    def __init__(self, weights: pd.DataFrame, returns: pd.DataFrame):
        self.weights = weights
        self.returns = returns

        self.events = {}

    def add_event(self, date: pd.Timestamp, event: function):
        """Change the weights of the portfolio."""
        self.events[date] = event

    def run(self, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """Run the simulation from start_date to end_date."""

        

        # Run the simulation
        for date, event in self.events:
            if date < start_date:
                continue
            if date > end_date:
                break

        return portfolio