"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np
import math

# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees
def best4LinReg(seed=1489683273):
    np.random.seed(seed)

    # Linear Regression models are best used when the data fits a line so build a dataset with linear relationship to Y
    X = np.random.random(size=(100,2))
    Y = np.sum(X, axis=1) 
    return X, Y

def best4DT(seed=1489683273):

    # Return a dataset (X and Y) that will return better for DT
    np.random.seed(seed)
    X = np.random.random(size=(100,2))

    # DT does better with classification problems so build something that resembles a classifier with no real linear relationship between factors
    Y = np.max(X, axis=1)
    return X, Y

def author():
    return 'scox43' #Change this to your user ID

if __name__=="__main__":
    print "they call me the Midnight Cowboy."
