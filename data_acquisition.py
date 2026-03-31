#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:35:53 2026

@author: michael
"""

import lseg.data as ld

# obtaining raw oil and gas future prices via API
ld.open_session()
prices_df = ld.get_history(
    universe=["LCOc1", "NGc1"],
    fields=["SETTLE"],
    count=120,
    interval="daily"
)
ld.close_session()

# data cleaning
prices_df = prices_df[-100:]

# export to csv
prices_df.to_csv("raw_energy_prices.csv")
