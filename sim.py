import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.constants import SMI, TICKERS, TICKERS_SW
from src.utils import get_data
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint, NonlinearConstraint
from scipy.optimize import minimize
from src.simulation import Simulation


def event(historical_stocks: pd.DataFrame, current_weights: pd.DataFrame):
    """Event that rebalances the portfolio"""
    n = len(historical_stocks.columns)
    df = pd.DataFrame(np.random.dirichlet(np.ones(n), size=2).T, index=historical_stocks.columns, columns=['Portfolio 1', 'Portfolio 2'])
    df["Portfolio 2"] = current_weights["Portfolio 2"]
    df["Portfolio 2"] = df["Portfolio 2"].fillna(0)
    return df


if __name__ == "__main__":
    bm, stocks = get_data(tickers=TICKERS_SW, benchmark=SMI)
    # Create the simulation
    sim = Simulation(bm, stocks)
    stocks = sim.get_stocks(pd.Timestamp("2000-01-01"))
    start_date = pd.Timestamp("2000-01-01")
    end_date = pd.Timestamp("2024-01-01")

    # Add an event every 6 months
    # for date in pd.date_range(start_date, end_date, freq='3YE'):
    #     sim.add_event(date, event)

    # Run the simulation
    portfolios_names = ['Portfolio 1', 'Portfolio 2']
    initial_weights = pd.DataFrame(np.random.dirichlet(np.ones(len(stocks.columns)), size=2).T, index=stocks.columns, columns=portfolios_names)
    bm_returns, portfolio_values = sim.run(initial_weights, start_date=start_date, end_date=end_date)

    sim.plot(bm_returns, portfolio_values)
