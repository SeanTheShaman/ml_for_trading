import StrategyLearner as sl 
import indicators as ind 
import datetime as dt 
import util as ut
import pandas as pd 
import matplotlib.pyplot as plt

def main():
   

    #### FOR SOLELY TESTING THE INDICATORS FUNCTIONALITY: 
    # Need to adjust startdate to not only account for n days but for 30 days prior to account for averages used in features
    #dates = pd.date_range(dt.datetime(2008,1,1),dt.datetime(2009, 12, 31))
    #df_prices = ut.get_data(['IBM'],dates) # Get data including SPY
    #df_prices = df_prices.rename(columns={'IBM' : 'Adj Close'})
    #i = ind.Indicators(df_prices, False)
    #df_stats = i.get_indicators()
    #print('Stats Dataframe: %s' %(df_stats))

    #df_stats.to_csv('stats.csv')
    #plt.plot(df_stats)
    #plt.show()


    ### TEST TRAINING FUNCTIONALITY
    learner = sl.StrategyLearner(verbose=True, impact=0.0) # constructor
    learner.addEvidence(symbol = "AAPL", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # training phase
    df_trades = learner.testPolicy(symbol = "AAPL", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000) # testing phase
    print df_trades.shape 
    return 

if __name__ == '__main__':
    main()
