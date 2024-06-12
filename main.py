from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import get_data
from src.constants import SMI, TICKERS_SW
from src.simulation import Simulation
from src.sim2 import Simulation2

bm, stocks = get_data(tickers=TICKERS_SW, benchmark=SMI, start_year=2019, end_year=2024, normalize=True)


s2 = Simulation2(benchmark=bm.pct_change(), returns=stocks.pct_change())
weights = np.random.dirichlet(np.ones(len(stocks.columns)), size=1)[0]
weights = pd.Series(weights, index=stocks.columns)
data = s2.run(weights=weights, start_date=pd.Timestamp("2000"), end_date=pd.Timestamp("2024"))
data.plot()
plt.show()
exit()

# s1 = Simulation(benchmark=bm, returns=stocks.pct_change())
# data = s2.run(weights=weights, start_date=pd.Timestamp("2000"), end_date=pd.Timestamp("2024"))
# data.plot()
# plt.show()
# exit()

# s1.add_event(pd.Timestamp("2020-01-01"), lambda x: pd.Series(np.random.dirichlet(np.ones(len(stocks.columns)), size=1)[0], index=stocks.columns))
# s1.add_event(pd.Timestamp("2021-01-01"), lambda x: pd.Series(np.random.dirichlet(np.ones(len(stocks.columns)), size=1)[0], index=stocks.columns))
# s1.add_event(pd.Timestamp("2022-01-01"), lambda x: pd.Series(np.random.dirichlet(np.ones(len(stocks.columns)), size=1)[0], index=stocks.columns))
# s1.add_event(pd.Timestamp("2023-01-01"), lambda x: pd.Series(np.random.dirichlet(np.ones(len(stocks.columns)), size=1)[0], index=stocks.columns))



total = {}
for _ in tqdm(range(100), desc="Running simulations"):
    # Inizialize the weights
    weights = np.random.dirichlet(np.ones(len(stocks.columns)), size=1)[0]
    weights = pd.Series(weights, index=stocks.columns)

    data = s2.run(weights=weights, start_date=pd.Timestamp("2000"), end_date=pd.Timestamp("2024"))

    final = round(data["portfolio"].iloc[-1], 2)
    if final in total:
        total[final] += 1
    else:
        total[final] = 1

benchmark = round(data["benchmark"].iloc[-1], 2)
print(f"Final benchmark: {benchmark}")
plt.hist(total)
plt.show()


# # Add the event dates
# for event in s1.events.keys():
#     plt.axvline(event, color="black", linestyle="-", alpha=0.2)
# plt.show()