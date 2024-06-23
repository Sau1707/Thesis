import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import Dataset
from src.simulation import Simulation
from src.frontier import Frontier

np.random.seed(0)

# Set up argument parser
parser = argparse.ArgumentParser(description="Generate a plot with specified years and variance.")
parser.add_argument('--years', type=int, default=5, help='Number of years to simulate')
parser.add_argument('--variance', type=float, default=0.1, help='Variance for the efficient frontier')

args = parser.parse_args()
YEARS = args.years
VARIANCE = args.variance


def event(historical_stocks: pd.DataFrame, current_weights: pd.DataFrame):
    """Event that rebalances the portfolio"""
    n = len(historical_stocks.columns)
    portfolios = ['Random - Update', 'Frontier - Update']
    df = pd.DataFrame(index=historical_stocks.columns, columns=portfolios)

    frontier = Frontier(historical_stocks, years=YEARS)

    # Generate the random
    df["Random - Update"] = np.random.dirichlet(np.ones(n))

    # Generate the efficient frontier
    frontier_weights = frontier.mean_variance_portfolio(0.5)
    df["Frontier - Update"] = frontier_weights

    return df


if __name__ == "__main__":
    data =  Dataset("SW", "1995", "^SSMI")
    bm = data.get_benchmark()
    stocks = data.get_data(liquidity=0.9)

    # Get 20 random stocks
    stocks = stocks.sample(20, axis=1)
    
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
    initial_weights = event(stocks, None)
    bm_returns, portfolio_values = sim.run(initial_weights, start_date=start_date, end_date=end_date)

    sim.plot(bm_returns, portfolio_values)
    plt.title(f"{YEARS} years, {VARIANCE} variance")
    plt.savefig(f"data/simulations/Y{YEARS}-V{VARIANCE}.png", dpi=300)
    