### This code is used to calculate various technical indicators throughout a given timeframe. 

import pandas as pd 
import numpy as np 

class Indicators():
    
    df_prices = None 
    #start_date = None
    #end_date = None 
    verbose = False


    def __init__(self, df_prices, verbose=False):
        self.df_prices = df_prices 
        #self.start_date = start_date
        #self.end_date = end_date 
        self.verbose = verbose


    # Returns a data frame in which the rows are the date and the columns are the technical indicators 
    # Expects price for each trading day to be in column named Adj Close
    def get_indicators(self):
        df_stats = self.df_prices.copy() # Started with prices data frame 

        # Build SMA stats to produce Bollinger Bands:
        df_stats['SMA'] = self.df_prices.rolling(14).mean()['Adj Close'] # Get rolling mean
        df_stats['STD'] = self.df_prices.rolling(14).std()['Adj Close'] # Get Std Dev
        df_stats['UBB'] = df_stats['SMA'] + (df_stats['STD']*2) # Upper Bollinger
        df_stats['LBB'] = df_stats['SMA'] - (df_stats['STD']*2) # Lower Bollinger

        # Build Average Gains/Losses to Produce RSI:
        df_stats['Change'] = self.df_prices['Adj Close']-self.df_prices['Adj Close'].shift(1) # Get change since prev day
        df_stats['Gain'] = df_stats.query('Change > 0')['Change']
        df_stats['Loss'] = df_stats.query('Change < 0')['Change']
        df_stats['Gain'] = df_stats['Gain'].fillna(0.00)
        df_stats['Loss'] = df_stats['Loss'].fillna(0.00)*-1
        df_stats['Avg Gain'] = df_stats.rolling(14).mean()['Gain']
        df_stats['Avg Loss'] = df_stats.rolling(14).mean()['Loss']
        df_stats['RS'] = df_stats['Avg Gain'] / df_stats['Avg Loss']
        df_stats['RSI'] = 100 - (100 / (1 + df_stats['RS']))


        # Add MACD column
        df_stats['26EMA'] = df_stats['Adj Close'].ewm(span=26).mean() # Add 26 day Exponential Moving Average
        df_stats['12EMA'] = df_stats['Adj Close'].ewm(span=12).mean() # Add 12 day EMA
        df_stats['MACD'] = df_stats['26EMA']-df_stats['12EMA'] # Calculate MACD

        
        #self.printv("DF_STATS: %s" %(df_stats))
        return df_stats

    def printv(self, some_string):
        if self.verbose == True:
            print some_string


    

        



