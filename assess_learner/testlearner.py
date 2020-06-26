"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import DTLearner as dtl
import RTLearner as rtl
import BagLearner as bl 
import InsaneLearner as il 
import sys

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])

    start_col = 0 # Define the starting column for the data - this can be used to exclude time series column if included 
    start_row = 1 # Define the starting row for data - can be used to exclude header row - assumed that data has header row

    if "Istanbul" in sys.argv[1]:
        start_col = 1

    data = np.array([map(float,s.strip().split(',')[start_col:]) for s in inf.readlines()[start_row:]])

    # compute how much of the data is training and testing
    train_rows = int(0.6* data.shape[0])
    #train_rows = int(data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    print trainX.shape
    #trainY = trainY.reshape(8,1)

    # create a learner and train it
    learner = il.InsaneLearner()
    learner.addEvidence(trainX, trainY)
    print learner.author()

    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    print "PRED Y Shape: " + str(predY.shape)
    print "TRAINX SHAPE: " + str(trainX.shape)
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0,1]

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]
