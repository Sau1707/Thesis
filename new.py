import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import IPlot, IOptimizer, get_data
np.random.seed(0)


RISK_FREE_RATE = 0.03
# Get the data from 2000 to 2020, leave the rest for testing
smi, stocks = get_data(start_year=2000, end_year=2020, normalize=True)

# Plot basic statistics
plot = IPlot(stocks, smi)
# plot.returns()
# plot.cumulative_returns()
# plot.average_returns()
# plot.standard_deviation()
# plot.correlation()
# plot.sharpe_ratio()

# Calculate the efficient frontier
optimizer = IOptimizer(stocks, smi, risk_free_rate=RISK_FREE_RATE)
optimizer.efficient_frontier()
optimizer.random_portfolios(n_portfolios=10_000, chunk_size=1_000)

# Add a point to the plot
smi_returns = smi.pct_change().mean()
smi_volatility = smi.pct_change().std() * np.sqrt(252)
print(smi_volatility)
optimizer.add_point(smi_returns, smi_volatility)
exit()