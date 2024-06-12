import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.constants import SMI, TICKERS, TICKERS_SW
from src.utils import get_data
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint, NonlinearConstraint
from scipy.optimize import minimize
from src.simulation import Simulation


def event(historical_stocks: pd.DataFrame, current_weights: pd.DataFrame, start = False):
    """Event that rebalances the portfolio"""
    n = len(historical_stocks.columns)
    historical_stocks = historical_stocks.ffill()

    portfolios = ['Random - Start', 'Random - Update', 'Frontier - Start', 'Frontier - Update']
    df = pd.DataFrame(index=historical_stocks.columns, columns=portfolios)

    # Generate the random start
    if start:
        df["Random - Start"] = np.random.dirichlet(np.ones(n))
    else:
        df["Random - Start"] = current_weights["Random - Start"]
        df["Random - Start"] = df["Random - Start"].fillna(0)

    # Generate the random update
    df["Random - Update"] = np.random.dirichlet(np.ones(n))

    # Define efficient frontier parameters
    correlation_matrix = historical_stocks.corr()
    mean_returns = historical_stocks.pct_change().mean()
    fn_constraint = lambda w: w.T @ correlation_matrix @ w
    fn_objective = lambda w: -w.T @ mean_returns  # Minimize the negative return
    bounds = Bounds(0, 1)
    linear_constraint = LinearConstraint(np.ones(n), 1, 1)
    non_linear_constraint = NonlinearConstraint(fn_constraint, -np.inf, 0)

    # Generate the efficient frontier start
    if start:
        # Make the initial guess equ
        w0 = np.ones(n) / n
        w0 = w0 / np.sum(w0)
        res = minimize(fn_objective, w0, bounds=bounds, constraints=[linear_constraint, non_linear_constraint])
        df["Frontier - Start"] = res.x
    else:
        df["Frontier - Start"] = current_weights["Frontier - Start"]
        df["Frontier - Start"] = df["Frontier - Start"].fillna(0)

    # Generate the efficient frontier update
    w0 = np.ones(n) / n
    w0 = w0 / np.sum(w0)
    res = minimize(fn_objective, w0, bounds=bounds, constraints=[linear_constraint, non_linear_constraint])
    df["Frontier - Update"] = res.x

    return df


if __name__ == "__main__":
    bm, stocks = get_data(tickers=TICKERS_SW, benchmark=SMI)
    # Create the simulation
    sim = Simulation(bm, stocks)

    # Setup the simulation range
    start_date = pd.Timestamp("2000-01-01")
    end_date = pd.Timestamp("2024-01-01")
    
    # Add an event every 6 months
    for date in pd.date_range(start_date, end_date, freq='3YE'):
        sim.add_event(date, event)

    # Run the simulation
    stocks = sim.get_stocks(start_date)
    initial_weights = event(stocks, None, start=True)
    bm_returns, portfolio_values = sim.run(initial_weights, start_date=start_date, end_date=end_date)

    sim.plot(bm_returns, portfolio_values)
