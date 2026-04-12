import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen 
from statsmodels.tsa.holtwinters import SimpleExpSmoothing



# load in an already cleaned dataset of oil and gas future prices
df = pd.read_csv(
    "https://github.com/MichaelKA-J/575-research/"
    "blob/main/raw_energy_prices.csv?raw=true",
    )

# descriptive statistics
df.iloc[:, 1:].describe().T

# plotting
diff = np.log(df.LCOc1) - np.log(df.NGc1) # log difference future prices
df['Date'] = pd.to_datetime(df['Date']) # 

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


# plot raw price levels of both series
def plot_levels():
    plt.plot(df.Date, df.iloc[:, 1:])
    plt.ylabel('Raw Prices')
    plt.title('Fig. 1: LCOc1 & NGc1 Raw Prices')
    plt.gcf().autofmt_xdate() 
    plt.autoscale(tight = 'x') 

plot_levels()

# plot log price levels of both series
def plot_levels():
    plt.plot(df.Date, np.log(df.iloc[:, 1:]))
    plt.ylabel('Log Prices')
    plt.title('Fig. 2: LCOc1 & NGc1 Log Prices')
    plt.gcf().autofmt_xdate() 
    plt.autoscale(tight = 'x') 

plot_levels()

# plot the (log) spread
def plot_spread():
    plt.plot(df.Date, diff)
    plt.ylabel('Log Price Difference')
    plt.title('Fig. 3: LCOc1-NGc1 Log Price Spread')
    plt.gcf().autofmt_xdate() 
    plt.autoscale(tight = 'x') 

plot_spread()



# johansen test
log_prices = np.log(df.iloc[:, 1:])
results = coint_johansen(
    endog = log_prices.values.astype(float), 
    det_order=0, 
    k_ar_diff=1
    )

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
    report['Max Eigen Reject'] = (
        report['Max Eigen Stat'] > report['Max Eigen CV (95%)']
        )    
    return report

report = get_johansen_report(results, log_prices.columns)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(report)



weights = results.evec[:, 0] 
print(f"Weight for Asset 1: {weights[0]}") 
print(f"Weight for Asset 2: {weights[1]}")

# construct cointegration spread using first eigenvector
coint_spread = log_prices.values @ weights

# plot cointegration spread
def plot_coint_spread():
    plt.figure()
    plt.plot(df.Date, coint_spread)
    plt.ylabel('Cointegrating Spread')
    plt.title('Fig. 4: Cointegrating Vector Spread (Johansen)')
    plt.gcf().autofmt_xdate()
    plt.autoscale(tight='x')

plot_coint_spread()

# plot the spread with exponential smoothing
recent_spread = diff[-60:]

es_model = SimpleExpSmoothing(recent_spread).fit()
es_fitted = es_model.fittedvalues

# plot
def plot_es_spread():
    plt.figure()
    plt.plot(df.Date[-60:], recent_spread, label='Log Price Spread')
    plt.plot(df.Date[-60:], es_fitted, label='Exp. Smoothed Spread')
    plt.ylabel('Spread')
    plt.title('Fig. 4: Exponential Smoothing of Log Price Spread')
    plt.gcf().autofmt_xdate()
    plt.autoscale(tight='x')
    plt.legend()

plot_es_spread()



# you need not run this, it is just to show how we acquired and cleaned data
'''
import os
os.chdir('/Users/michael/Documents/ECON 575')

# pull raw prices using LSEG's API
import lseg.data as ld
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
(cleaned_prices < 0).any().any() # run in console to test for negative values

plt.savefig('log_spread_plot.png')
'''

