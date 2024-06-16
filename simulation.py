import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import get_data
from src.frontier import Frontier
from src.simulation import Simulation

np.random.seed(0)

# Set up argument parser
parser = argparse.ArgumentParser(description="Generate a plot with specified years and variance.")
parser.add_argument('--years', type=int, help='Number of years', default=5)
parser.add_argument('--variance', type=float, help='Variance value', default=0.1)

args = parser.parse_args()
YEARS = args.years
VARIANCE = args.variance


def event(historical_stocks: pd.DataFrame, current_weights: pd.DataFrame, start = False):
    """Event that rebalances the portfolio"""
    n = len(historical_stocks.columns)
    portfolios = ['Random - Start', 'Random - Update', 'Frontier - Start', 'Frontier - Update']
    df = pd.DataFrame(index=historical_stocks.columns, columns=portfolios)

    # Generate the random update
    df["Random - Update"] = np.random.dirichlet(np.ones(n))

    # Update the efficient frontier
    frontier = Frontier(historical_stocks, years=YEARS)
    frontier_weights = frontier.mean_variance_portfolio(VARIANCE)
    df["Frontier - Update"] = frontier_weights

    return df


if __name__ == "__main__":
    os.makedirs("data/simulations", exist_ok=True)

    # Get the data and initialize the simulation
    bm, stocks = get_data(benchmark="^SSMI")
    sim = Simulation(bm, stocks)

    # Setup the simulation range
    start_date = pd.Timestamp("2000-01-01")
    end_date = pd.Timestamp("2024-01-01")
    
    # Add an event every 6 months
    for date in pd.date_range(start_date, end_date, freq='1YE'):
        sim.add_event(date, event)

    # Run the simulation
    stocks = sim.get_stocks(end_date=start_date)
    initial_weights = event(stocks, None, start=True)
    bm_returns, portfolio_values = sim.run(initial_weights, start_date=start_date, end_date=end_date)

    sim.plot(bm_returns, portfolio_values)
    plt.title(f"{YEARS} years, {VARIANCE} variance")
    plt.savefig(f"data/simulations/Y{YEARS}-V{VARIANCE}.png", dpi=300)
    