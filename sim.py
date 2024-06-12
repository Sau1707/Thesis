import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.constants import SMI, TICKERS, TICKERS_SW
from src.utils import get_data
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint, NonlinearConstraint
from scipy.optimize import minimize
from src.simulation import Simulation


if __name__ == "__main__":
    bm, stocks = get_data(tickers=TICKERS_SW, benchmark=SMI, start_year=2000, end_year=2024, normalize=True)

    # Create the simulation
    sim = Simulation(bm, stocks)

    # Add an event
    sim.add_event(pd.Timestamp("2020-06-06"), lambda x: pd.DataFrame(np.ones((len(x.columns), 2)) / len(x.columns), index=x.columns, columns=['Portfolio 1', 'Portfolio 2']))
    sim.add_event(pd.Timestamp("2020-12-06"), lambda x: pd.DataFrame(np.random.dirichlet(np.ones(len(x.columns)), size=2).T, index=x.columns, columns=['Portfolio 1', 'Portfolio 2']))

    # Run the simulation
    initial_weights = pd.DataFrame(np.random.dirichlet(np.ones(len(stocks.columns)), size=2).T, index=stocks.columns, columns=['Portfolio 1', 'Portfolio 2'])
    bm_returns, portfolio_values = sim.run(initial_weights, pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01"))

    print(portfolio_values)
    sim.plot(bm_returns, portfolio_values)
