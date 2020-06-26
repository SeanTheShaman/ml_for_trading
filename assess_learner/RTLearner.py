
##################################### RTLearner Class
##### Notes 
### This class represents a Random Tree Learner.
##### API Specification:
### import RTLearner as rt
### learner = dt.RTLearner(leaf_size = 1, verbose = False) # constructor
### learner.addEvidence(Xtrain, Ytrain) # training step
### Y = learner.query(Xtest) # query
#####################################

import numpy as np 

class RTLearner():
    leaf_size = 1 
    verbose = False
    tree_table = np.array([], dtype=float)

    def __init__(self, leaf_size=1, verbose=False): 
        self.leaf_size = leaf_size 
        self.verbose = verbose


    def addEvidence(self, Xtrain, Ytrain):
        self.printv("*************************** BUILDING THE TREE *********************************")
        self.printv('X-Train: %s. Y-Train: %s' %(Xtrain, Ytrain))
        self.printv('X-Train Shape: %s. Y-Train Shape: %s' %(Xtrain.shape, Ytrain.shape))
        self.tree_table = self.build_tree(Xtrain,  Ytrain) # Build the tree

        self.printv('**** FINAL TREE TABLE **** \n %s' %(self.tree_table))
        self.printv('*************************** TREE BUILDING ACTIVITIES ARE OVER  *********************************')

    def build_tree(self, Xtrain, Ytrain):
        
        # STOPPING CONDITIONS 

        # 1. Number of elements in data set is <= leaf_size
        # 2. All elements of Y are the same
        # 3. No rows

        self.printv('X-Train Shape: %s Leaf Size: %s. Min Y-Train: %s. Max Y-Train: %s' %(Xtrain.shape, self.leaf_size, np.min(Ytrain), np.max(Ytrain)))
        if Xtrain.shape[0]<= self.leaf_size: # Case 1
            #print np.concatenate(([-1], Ytrain, [-1, -1]))
            #return np.concatenate(([-1], Ytrain, [-1, -1]))
            return np.array([-1, np.mean(Ytrain), -1, -1]) 
        elif np.min(Ytrain)==np.max(Ytrain): # Case 2
            return np.array([-1, Ytrain[0], -1, -1])
        elif Xtrain.shape[0] == 0: # Case 3
            return 
        else:
           

            # Use the sorted array to determine columns, should choose first column in case of tie
            best_factor = -1 # This will be used to determine when the best factor has been found - not just based off of correlation but also whether factor ensures splits to both branches
            failed_attempts = 0
            while best_factor < 0: # Keep searching until the best factor is found
                best_factor = np.random.randint(0, Xtrain.shape[1]-1) # grab a random column


                if failed_attempts > 20:
                    self.printv('FAILED ATTEMPTS ARE GREATER THAN 20. SET SPLIT VAL TO MEAN OF COLUMN')
                    splitVal = np.mean(Xtrain[:, best_factor])
                    self.printv('Looking at factor %s. Split value is %s' %(best_factor, splitVal))

                else:
                    self.printv("Taking the median of %s to determine split val" %Xtrain[:, best_factor])
                    splitVal = np.median(Xtrain[:, best_factor]) # The split value becomes the median of the column which currently interested in
                    self.printv('Looking at factor %s. Split value is %s' %(best_factor, splitVal))

                # Determine which indices will go in which direction
                to_the_left_ix = np.where(Xtrain[:, best_factor] <= splitVal) # Indices of Xtrain that will be going to the left (less than or equal to the split value)
                to_the_right_ix = np.where(Xtrain[:, best_factor] > splitVal) # Indices of Xtrain that will be going to the right (greater than the split value)
                self.printv('Indices going to the left %s' %to_the_left_ix)
                self.printv('Indices going to the right %s' %to_the_right_ix)

                # Pull the corresponding rows from the data 
                to_the_left_data = Xtrain[to_the_left_ix] #,:]
                to_the_right_data = Xtrain[to_the_right_ix] #, :]
                self.printv('Data going to the left %s \n Data going to the right %s' %(to_the_left_data, to_the_right_data))
                
                # Check to make sure using this factor will not result in empty branches:
                if to_the_left_data.shape[0]==0 or to_the_right_data.shape[0]==0: 
                    best_factor = -1
                    failed_attempts += 1 # Increment failed attempts by one

            
            # Current state here is that a best factor has been found and data has been split up based on which branch it will become a part of. 
            left_tree = self.build_tree(to_the_left_data, Ytrain[to_the_left_ix]) # Recursively pass the Xtrain/Ytrain data which will be going to the left branch
            right_tree = self.build_tree(to_the_right_data, Ytrain[to_the_right_ix]) # Recursively pass the Xtrain/Ytrain data which will be going to the right branch

            self.printv('LEFT TREE %s' %left_tree)
            self.printv('LEFT TREE SHAPE[1] %s' %str(left_tree.shape))
            self.printv('LEFT TREE SHAPE[1] %s' %len(left_tree.shape))
            # If 2D array then shape[0] works because it is the number of rows
            if len(left_tree.shape)==2:
                root = np.array([best_factor, splitVal, 1, left_tree.shape[0]+1]) # Set the root node
            else:
                root = np.array([best_factor, splitVal, 1, 2]) # Else, when left_tree is 1D then shape[0] becomes the number of columns then just add 1
            self.printv('Root Shape: %s \n Left Tree Shape: %s \n Right Tree Shape: %s' %(root.shape, left_tree.shape, right_tree.shape))
            return (np.vstack( (root, left_tree, right_tree)).astype(float))  ## Return the appended tree




            
        

        

     
       

    def query(self, xtest):
        # Each row in X-Test is a series of factors, use with tree to generate the predicted value.
        queryY = np.array([], dtype=np.float) # This is the array that will store predicted values for Y

        for row in xtest:
            queryY = np.append(queryY, self.get_prediction(row) ) 

        self.printv('QUERY Y IS: ' + str(queryY))
        return queryY  # Return a ndarray for which each row represents the predicted value for Y


    # Given an ordered set of factors - X1,X2,X3,XN - returns the predicted value for Y based on the decision tree
    def get_prediction(self, xfactors):
        current_location = 0
        while self.tree_table[int(current_location),0] >= 0: # Until a leaf node is found, keep searching
            current_factor, split_val,left_inc, right_inc = self.tree_table[int(current_location)]
            self.printv('Current location: %s Current Factor: %s Split Val: %s Left Inc %s Right Inc %s' %(current_location, current_factor, split_val, left_inc, right_inc))
            x_val = xfactors[int(float(current_factor))]

            if x_val <= split_val:
                if left_inc > 0:
                    current_location = current_location+left_inc
            else:
                if right_inc > 0:
                    current_location = current_location+right_inc

        self.printv('Final location is row %s. With values: %s' %(current_location, self.tree_table[int(current_location)]))
        return self.tree_table[int(current_location), 1] # Return the Y val

        



    def author(self):
        return "scox43"

    def printv(self, somestring): # Used so that debug messages are only printed when in verbose mode 
        if self.verbose:
            print somestring 

if __name__ == "__main__":
    print ""
