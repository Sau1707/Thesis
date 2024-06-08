import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.utils import get_data
from src.plots import IPlot

# Get the data from 2000 to 2020, leave the rest for testing
smi, stocks = get_data(start_year=2000, end_year=2020, normalize=True)

plot = IPlot(stocks, smi)
plot.returns()
plot.cumulative_returns()
plot.average_returns()
plot.standard_deviation()