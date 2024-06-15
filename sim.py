import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.constants import SMI, TICKERS_SW
from src.utils import get_data
from src.simulation import Simulation
from src.frontier import Frontier

np.random.seed(0)

# Set up argument parser
parser = argparse.ArgumentParser(description="Generate a plot with specified years and variance.")
parser.add_argument('--years', type=int, required=True, help='Number of years')
parser.add_argument('--variance', type=float, required=True, help='Variance value')

args = parser.parse_args()
YEARS = args.years
VARIANCE = args.variance


def event(historical_stocks: pd.DataFrame, current_weights: pd.DataFrame, start = False):
    """Event that rebalances the portfolio"""
    n = len(historical_stocks.columns)
    portfolios = ['Random - Start', 'Random - Update', 'Frontier - Start', 'Frontier - Update']
    df = pd.DataFrame(index=historical_stocks.columns, columns=portfolios)

    frontier = Frontier(historical_stocks, years=YEARS)
    # Generate the random start
    if start:
        df["Random - Start"] = np.random.dirichlet(np.ones(n))
    else:
        df["Random - Start"] = current_weights["Random - Start"]
        df["Random - Start"] = df["Random - Start"].fillna(0)

    # Generate the random update
    df["Random - Update"] = np.random.dirichlet(np.ones(n))

    # Generate the efficient frontier start
    frontier_weights = frontier.mean_variance_portfolio(VARIANCE)
    if start:
        df["Frontier - Start"] = frontier_weights
    else:
        df["Frontier - Start"] = current_weights["Frontier - Start"]
        df["Frontier - Start"] = df["Frontier - Start"].fillna(0)

    # Generate the efficient frontier update
    df["Frontier - Update"] = frontier_weights

    return df


if __name__ == "__main__":
    bm, stocks = get_data(tickers=TICKERS_SW, benchmark=SMI)
    # Create the simulation
    sim = Simulation(bm, stocks)

    # Setup the simulation range
    start_date = pd.Timestamp("2010-01-01")
    end_date = pd.Timestamp("2024-01-01")
    
    # Add an event every 6 months
    for date in pd.date_range(start_date, end_date, freq='1YE'):
        sim.add_event(date, event)

    # Run the simulation
    stocks = sim.get_stocks(start_date)
    initial_weights = event(stocks, None, start=True)
    bm_returns, portfolio_values = sim.run(initial_weights, start_date=start_date, end_date=end_date)

    sim.plot(bm_returns, portfolio_values)
    plt.title(f"{YEARS} years, {VARIANCE} variance")
    plt.savefig(f"data/simulations/Y{YEARS}-V{VARIANCE}.png", dpi=300)
    