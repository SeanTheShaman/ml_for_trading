"""Analyze a portfolio.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from util import get_data, plot_data

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    ### Get daily portfolio value
    port_val = prices_SPY # add code here to compute daily portfolio values
    norm = prices/prices.ix[0] # Normalize
    #print "Normalized portfolio: " + str(norm.ix[0:5])
    alloc = norm*allocs # Apply allocations
    #print "Allocated amounts: " + str(alloc.ix[0:5]) 
    # Apply starting value
    port_val_ind = alloc*sv # calculate the actual daily portfolio amount by multiplying by starting value
    #print "Portfolio Value by Day: " + str(port_val_ind.ix[0:5])
    port_val = port_val_ind.sum(axis=1) # sum across rows to get combined portfolio values
    #print "Portfolio Value (total): " + str(port_val)
    ### Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = compute_portfolio_stats(port_val, allocs, rfr, sf)

    ### Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        print "SPY PRICES: " + str(prices_SPY)
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1) # Add SPY to Portfolio DF
        df_norm = df_temp/df_temp.ix[0]
        df_norm.plot()
        plt.title("Portfolio Value vs SPY")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price")
        #plt.show()
        plt.savefig('Portfolio vs SPY.pdf', format='pdf')
        pass

    # Add code here to properly compute end value
    ev = port_val[-1]
    #print "Ending Portfolio Value: " + str(ev)

    return cr, adr, sddr, sr, ev


# Given a dataframe of summed portfolio value, initial allocations, risk free return rate, and sampling frequency,
# this returns statistics of the portfolio (cumul ret, avg daily ret, std daily ret, and sharpe ratio.
def compute_portfolio_stats(prices, allocs, rfr, sf):
    daily_returns = (prices / prices.shift(1)) # calculate the daily returns for each day
    daily_returns = daily_returns[1:] # Remove the 0th row because it doesn't contain a daily return

    #print "Daily Returns: " + str(daily_returns)
    cr = get_cumul_rets(prices)
    adr = get_avg_daily_rets(daily_returns)
    sddr = get_std_daily_rets(daily_returns)
    sr = get_sharpe_ratio(daily_returns, rfr, sf)

    return [cr, adr, sddr, sr]

def get_cumul_rets(daily_rets):
    return (daily_rets.ix[-1]/daily_rets.ix[0])-1 # Return the last value by the first value to generate cumulative returns

def get_avg_daily_rets(daily_rets):
    return daily_rets.mean()-1 # Return the mean-1 for daily returns 

def get_std_daily_rets(daily_rets): 
    return daily_rets.std() # Return the std dev for daily returns 


# Given a dataframe of portfolio values, risk free return rate, and sampling frequency this function returns Sharpe Ratio
def get_sharpe_ratio(daily_rets, rfr, sf):
    mean = get_avg_daily_rets(daily_rets-rfr) # Get mean for daily_rets-rfr 
    std = get_std_daily_rets(daily_rets-rfr) # Get std for daily_rets-rfr
    sr = (mean/std)*np.sqrt(sf) # Adjust ratio for sampling frequency
    return sr

def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']
    allocations = [0.0, 0.0, 0.0, 0.1]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    test_code()
