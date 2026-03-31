#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:35:53 2026

@author: michael
"""

"""
- get as much data as you can, get 20 yrs, not just 100
- beware of negative values because of the logs
- regarding missing values, weekly mean may be better than daily frequency

- Brent Crude was released in 2011
- Henry Hub was released in 1990

- can model the spread itself with exponential smoothing
- model difference betweeen log prices using exp. smoothing
"""

import lseg.data as ld

# obtaining raw oil and gas future prices via API
ld.open_session()
prices_df = ld.get_history(
    universe=["LCOc1", "NGc1"],
    fields=["SETTLE"],
    start="1991-03-31",
    end="2026-03-31",
    interval="daily"
)
ld.close_session()

# data cleaning
cleaned_prices = prices_df.dropna()

# export to csv
cleaned_prices.to_csv("raw_energy_prices.csv")

# (cleaned_prices < 0).any().any() 
# run in console to test for negative values
