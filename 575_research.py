#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:09:01 2026

@author: michael
"""

import lseg.data as ld
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen 

# obtaining raw oil and gas future prices via API
ld.open_session()
prices_df = ld.get_history(
    universe=["LCOc1", "NGc1"],
    fields=["SETTLE"],
    start="1991-03-31",
    end="2026-03-31",
    interval="daily" # try weekly or monthly frequency
)
ld.close_session()

# data cleaning
cleaned_prices = prices_df.dropna()

# (cleaned_prices < 0).any().any() 
# run in console to test for negative values

# export to csv
cleaned_prices.to_csv("raw_energy_prices.csv")



# plot the spread
plt.xlabel('Time')
plt.ylabel('Log Price Difference')
plt.title('LCOc1-NGc1 Spread')
diff = np.log(cleaned_prices.LCOc1) - np.log(cleaned_prices.NGc1)
diff_plot = plt.plot(diff)
plt.savefig('spread_plot.png')



# johansen test
# .values extracts an array of just the values from the dataframe
# astype(float) was needed for the function to work
log_prices = np.log(cleaned_prices)
results = coint_johansen(
    endog=log_prices.values.astype(float), 
    det_order=0, 
    k_ar_diff=1
    )

# ???
def get_johansen_report(results, column_names):
    cv_idx = 1 
    
    stats = {
        'Trace Stat': results.lr1,
        'Trace CV (95%)': results.cvt[:, cv_idx],
        'Max Eigen Stat': results.lr2,
        'Max Eigen CV (95%)': results.cvm[:, cv_idx]
    }
    
    report = pd.DataFrame(stats)
    report.index.name = 'r <= (Rank)'
    
    report['Trace Reject'] = report['Trace Stat'] > report['Trace CV (95%)']
    report['Max Eigen Reject'] = report['Max Eigen Stat'] > report['Max Eigen CV (95%)']
    
    return report
report = get_johansen_report(results, log_prices.columns)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(report)
report.to_csv('johansen_results.csv')



# $$Spread = \text{LogPrice}_A \times w_1 + \text{LogPrice}_B \times w_2$$
weights = results.evec[:, 0]
print(f"Weight for Asset 1: {weights[0]}")
print(f"Weight for Asset 2: {weights[1]}")
