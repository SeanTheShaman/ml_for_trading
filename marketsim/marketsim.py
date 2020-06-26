"""MC2-P1: Market simulator.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def author():
    return "scox43"

def compute_portfolio_stats(prices, rfr, sf):
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


def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input

    # TODO: Check to see whether orders_file is a file object or string and generate dfOrders appropriately
    
    dfOrders = pd.read_csv(orders_file,index_col="Date",parse_dates=True, names=['Date','Symbol','Type','Quantity'], skiprows=[0])
    dfOrders.sort_index() # Sort the df by Date so that it will be more efficient to process

    #print dfOrders
    # CURRENT STATE: dfOrders is a pandas dataframe in format:
    # Date, Symbol, BUY/SELL, Quantity. 

    # Build prices df which contains prices for each stock on trading days in range
    dfPrices = build_prices_df(dfOrders['Symbol'].drop_duplicates().values.tolist(), dfOrders.index.min(), dfOrders.index.max())
    dfPrices.to_csv('prices-12.csv')
    #print dfPrices 
    # Build trades df
    tradesDf = build_trades_df(dfOrders, dfPrices, commission, impact) 
    #print "DF TRADES:\n" + str(tradesDf)

    holdingsDf = build_holdings_df(start_val, tradesDf)

    #print "HOLDINGS: \n" + str(holdingsDf)

    valuesDf = build_values_df(dfPrices, holdingsDf)

    #print "VALUES: \n" + str(valuesDf)

    # The values dataframe can now be used to calculate port val for each day
    portvals = valuesDf.sum(axis=1) 

    #print "PORTVALS: \n" + str(portvals)

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months

    rv = pd.DataFrame(index=portvals.index, data=portvals.as_matrix())

    return rv


### Given a df of orders and df of prices, return 
def build_trades_df(orders_df, prices_df, commission, impact):
    dfTrades = prices_df.copy() # Intiialize trades df to prices df
    dfTrades = dfTrades.drop('Cash', axis=1)
    dfTrades['Commission'] = 0.0 # Add col to track comm fees
    dfTrades.iloc[:,:] = dfTrades.iloc[:,:]*0 # Set all values  to 0 
    dfTradesAdjusted = dfTrades.copy() # Trades adjusted for impact

    # Iterate through orders file by date and update trades df
    for index, row in orders_df.iterrows():
    
        # Set the price in the trades df:
        quantity = row.Quantity
        if row.Type == "SELL": # If selling then reverse symbol
            quantity *= -1
            quantityAdj = (quantity-(impact*quantity))
        else: # it's a buy
            quantityAdj = quantity+(impact*quantity)
        

 
        dfTradesAdjusted.ix[index, row.Symbol] = dfTradesAdjusted.ix[index, row.Symbol]+(quantityAdj) # Set the adjusted quantity
        dfTrades.ix[index, row.Symbol] = dfTrades.ix[index, row.Symbol]+quantity # Set the quantity
        dfTrades.ix[index, 'Commission'] = dfTrades.ix[index,'Commission']+commission


    dfTesting =  dfTradesAdjusted*prices_df 
    dfTrades['Cash'] = dfTesting.sum(axis=1)*-1 # Sum across row to get cash for each day 
  
    dfTrades['Cash'] = dfTrades['Cash']-dfTrades['Commission'] # Subtract comm fees 
    return dfTrades 

### Given a list of symbols, start date, and end date this
# function returns a pandas dataframe representing adj close stock price with columns:
# Date, SYM1, SYM2, SYMN, CASH
def build_prices_df(symbols, startdate, enddate):

    # Pass in symbols, startdate and enddate to get_data function
    dfData = get_data(symbols, pd.date_range(pd.to_datetime(startdate), pd.to_datetime(enddate)))  
    dfData = dfData.drop('SPY', axis=1)# Remove SPY column
    dfData['Cash'] = 1.0   # Add cash column with value set to 1.0 


    return dfData

def build_holdings_df(starting_cash, trades_df):
    holdings_df = trades_df.copy()
    holdings_df.ix[0,'Cash'] = holdings_df.ix[0,'Cash']+starting_cash # Set starting cash
    holdings_df = holdings_df.cumsum() # Cumul. sum the values for each col
    return holdings_df 

def build_values_df(prices_df, holdings_df):
    values_df = prices_df*holdings_df
    return values_df 

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters
    
    of = "./orders/orders-12.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv,impact=0.005)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats:
    start_date = portvals.index.min()
    end_date = portvals.index.max()
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(portvals, 0.0, 252)
    
    spy_vals = get_data( [], pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date) ) )
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = compute_portfolio_stats(spy_vals, 0.0, 252)

    print "AUTHOR: " + author()
    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()



