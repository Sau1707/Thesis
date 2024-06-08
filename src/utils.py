import pandas as pd


class Utils:
    def standard_deviation(self, series: pd.Series):
        return series.std()

    def ptc_change(self, series: pd.Series):
        return series.pct_change()

    def annualize_rets(self, r: pd.Series, periods_per_year: int):
        """returns the annualized return of a set of returns"""

        # Check if the data is stock prices or returns
        if (r > 5).any().any(): # Suppose that is impossible to have a return of 5
            r = self.ptc_change(r)
        
        compounded_growth = (1+r).prod()
        n_periods = r.shape[0]
        return compounded_growth**(periods_per_year/n_periods)-1

    def annualize_vol(self, r: pd.Series, periods_per_year: int):
        """Annualizes the vol of a set of returns"""
        return r.std()*(periods_per_year**0.5)
    
    def sharpe_ratio(self, r: pd.Series, riskfree_rate: float, periods_per_year: int):
        """Computes the annualized sharpe ratio of a set of returns"""
        # convert the annual riskfree rate to per period
        rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
        excess_ret = r - rf_per_period
        ann_ex_ret = self.annualize_rets(excess_ret, periods_per_year)
        ann_vol = self.annualize_vol(r, periods_per_year)
        return ann_ex_ret/ann_vol


utils = Utils()