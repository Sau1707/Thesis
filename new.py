import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import IPlot, IOptimizer, get_data
from src.benchmark import Benchmark
np.random.seed(0)


RISK_FREE_RATE = 0.03
# Get the data from 2000 to 2020, leave the rest for testing
smi, stocks = get_data(start_year=2000, end_year=2024, normalize=True)

bm = Benchmark(smi)
bm.plot_returns("y")
bm.plot_cumulative_returns("y")
print(bm.get_average_returns("d") * 100)
exit()
# Plot the df
df.plot.bar()
plt.show()
exit()

# As a point of comparison we plot the benchmark total return
total_return = (smi.iloc[-1] - smi.iloc[0]) / smi.iloc[0]


# Plot the smi returns and volatility
smi_returns = smi.pct_change().mean()
smi_volatility = smi.pct_change().std() * np.sqrt(252)
plt.scatter(smi_volatility, smi_returns, color='red', s=100, label='SMI')

# Add the cap weighted returns and volatility
for ticker in stocks.columns:
    returns = stocks[ticker].pct_change().mean()
    volatility = stocks[ticker].pct_change().std() * np.sqrt(252)
    plt.scatter(volatility, returns, label=ticker)

plt.legend()
plt.show()
exit()

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