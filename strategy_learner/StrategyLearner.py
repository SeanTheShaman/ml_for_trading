"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import pandas as pd
import util as ut
import random
import indicators as ind 
import QLearner as ql
import numpy as np 

class StrategyLearner(object):


    dfIndicators = None 
    n_days = 5 # Days window between X->Y
    min_iters = 10 # Min number of iterations
    max_iters = 1000 # Max number of iterations
    q_learner = None 

    # These help with discretize function:
    bollinger_buckets = None
    macd_buckets = None
 

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.q_learner = ql.QLearner(999, 3)

    # Discretization function for technical indicators
    def discretize(self, tdate):


        # Discretize based on pre-calculated buckets for these technical indicators:
        ubb_int = self.bollinger_buckets[tdate]
        macd_int = self.macd_buckets[tdate]
        rsi_int = self.rsi_buckets[tdate]

        

        self.printv('UBB INT: %i. MACD INT: %i. RSI Int: %i' %(ubb_int, macd_int,rsi_int))
        disc_value = (ubb_int*100) + (macd_int*10) + rsi_int
        self.printv('Discretized Values: %i' %(disc_value))
        return int(disc_value)

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000): 


       # Get the data for the date range (in addition go back n-days plus 30 for tech indicators to work)
        dates = pd.date_range(sd - dt.timedelta(days=(self.n_days+30 ) ) , ed)
        df_prices = ut.get_data(['IBM'],dates) # Get data including SPY
        df_prices = df_prices.rename(columns={'IBM' : 'Adj Close'})

        indic = ind.Indicators(df_prices, self.verbose)
        df_indicators = indic.get_indicators()
        adj_sd_pos = np.searchsorted(df_indicators.index, sd.date()) #Position of adjusted start date
        df_indicators = df_indicators.ix[adj_sd_pos:-1,:] # Trim anything before the start date
        #self.printv('Trimmed Indicators DF: \n%s' %(df_indicators))

        # Normalize the data, required so that discretization of test data will work same as training data 
        #df_indicators = df_indicators/(df_indicators.ix[0])
        #self.printv('Normalized DF Indicators: %s' %(df_indicators))

        # Set buckets for indicator values, this helps to discretize the values for use with QLearner
        self.bollinger_buckets = pd.qcut(df_indicators['UBB'], 10, labels=np.arange(0,10))
        self.macd_buckets = pd.qcut(df_indicators['MACD'], 10, labels=np.arange(0,10))
        self.rsi_buckets = pd.qcut(df_indicators['RSI'], 10, labels=np.arange(0,10))

        #self.printv('MACD Buckets: %s. \n Bollinger Buckets: %s' %(self.macd_buckets, self.bollinger_buckets))




        # Start Training 
        converged = False 
        prev_df_trades = None # df_trades of last iteration, used to check convergence 
        current_iter = 1 # Start out at iteration 1, used to ensure enough iterations are ran but not too many leading to timeout


        while not converged:
            
            df_trades = pd.Series(index=df_indicators.index) # Create DF to track trades 
            df_trades[:] = 0 # Init to zero 
            holding = 0 # Starts out with 0 holdings 

            # Iterate throughout the indicators/price data frame
            for index, row in df_indicators.iterrows():
                current_row = df_indicators.index.get_loc(index) # Get the current row number
                s = self.discretize(index) # Get the discretized state for this day
                if current_row==0: # First day so just set-up the state
                    a = self.q_learner.querysetstate(s)
                    action = self.process_action(a, holding) # Determines buy/sell action based on a and current holding
                    holding = holding+action
                else: # Not first day, so adjust q-table

                    # Calculate reward as percentage change multiplied by holding
                    prev_close = df_indicators['Adj Close'].iloc[current_row-1]
                    today_close = row['Adj Close']
                    r = ((today_close/prev_close)-1) * holding 
                    self.printv('Reward for %s is %s' %(index, r))
                    a = self.q_learner.query(s, r)
                    action = self.process_action(a, holding)
                    holding = holding + action
                df_trades[index] = action
            self.printv(df_trades)

            # Check for convergence
            if current_iter>self.max_iters: converged=True
            
            if current_iter>self.min_iters and prev_df_trades.all()==df_trades.all(): converged=True

            prev_df_trades = df_trades # Track history of last df trades for comparison for convergence
            current_iter = current_iter+1 # Increase iteration

        if self.verbose == True: df_trades.to_csv('df_trades.csv')
        


    def process_action(self, a, holding):
        # Holding is current holding, used to adjust action

        if a==0: # Go to short 1000
            if holding > 0: return -2000 
            elif holding == 0: return -1000
            else: return 0
        elif a==1: # Do nothing
            return 0 
        else: # Go to long 1000
            if holding <0: return 2000
            elif holding ==0: return 1000
            else: return 0 



    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):


        dates = pd.date_range(sd - dt.timedelta(days=(self.n_days+30 ) ) , ed)
        df_prices = ut.get_data(['IBM'],dates) # Get data including SPY
        df_prices = df_prices.rename(columns={'IBM' : 'Adj Close'})

        indic = ind.Indicators(df_prices, self.verbose)
        df_indicators = indic.get_indicators()
        adj_sd_pos = np.searchsorted(df_indicators.index, sd.date()) #Position of adjusted start date
        df_indicators = df_indicators.ix[adj_sd_pos:-1,:] # Trim anything before the start date
        #self.printv('Trimmed Indicators DF: \n%s' %(df_indicators))

        # Normalize the data, required so that discretization of test data will work same as training data 
        #df_indicators = df_indicators/(df_indicators.ix[0])
        #self.printv('Normalized DF Indicators: %s' %(df_indicators))

        # Set buckets for indicator values, this helps to discretize the values for use with QLearner
        self.bollinger_buckets = pd.cut(df_indicators['UBB'], 10, labels=np.arange(0,10))
        self.macd_buckets = pd.cut(df_indicators['MACD'], 10, labels=np.arange(0,10))
        self.rsi_buckets = pd.cut(df_indicators['RSI'], 10, labels=np.arange(0,10))

        df_trades = pd.DataFrame(index=df_indicators.index) # Create DF to track trades 
        df_trades["Trade"] = 0
        df_trades[:] = 0 # Init to zero 
        holding = 0 # Starts out with 0 holdings 

            # Iterate throughout the indicators/price data frame
        for index, row in df_indicators.iterrows():
            current_row = df_indicators.index.get_loc(index) # Get the current row number
            s = self.discretize(index) # Get the discretized state for this day
            a = self.q_learner.querysetstate(s)
            action = self.process_action(a, holding) # Determines buy/sell action based on a and current holding
            holding = holding+action
            df_trades.ix[index,'Trade'] = action

        if self.verbose == True: df_trades.to_csv('test_df_trades.csv')

        return df_trades

    def printv(self, some_string):
        if self.verbose:
            print some_string

if __name__=="__main__":
    print "One does not simply think up a strategy"
