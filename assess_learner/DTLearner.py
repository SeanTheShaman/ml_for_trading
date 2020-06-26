
##################################### DTLearner Class
##### Notes 
### This class represents a Decision Tree Learner.
##### API Specification:
### import DTLearner as dt
### learner = dt.DTLearner(leaf_size = 1, verbose = False) # constructor
### learner.addEvidence(Xtrain, Ytrain) # training step
### Y = learner.query(Xtest) # query
#####################################

import numpy as np 

class DTLearner():
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
            return np.array([-1, np.mean(Ytrain), -1, -1]) ## TODO: Come back and modify for the case where leaf_size > 1
        elif np.min(Ytrain)==np.max(Ytrain): # Case 2
            return np.array([-1, Ytrain[0], -1, -1])
        elif Xtrain.shape[0] == 0: # Case 3
            return 
        else:
            # Find correlations 
            # Find best column
            # Ensure that split is done so that neither branches are empty 
            correl = np.nan_to_num(np.corrcoef(x=Xtrain, y=Ytrain, rowvar=0)[-1,0:-1]) # Generate the correl matrix bt Xtrain/Ytrain, only select last row and all except last column as this is only relevant data 
            sorted_correl = np.flipud(np.sort(np.abs(correl))) # Sorted correlation is an ordered list (from largest at [0] to smallest at [n-1] of values of correlations

            # Use the sorted array to determine columns, should choose first column in case of tie
            best_factor = -1 # This will be used to determine when the best factor has been found - not just based off of correlation but also whether factor ensures splits to both branches
            current_index = 0 # Start with the highest correlated factor from the sorted_correl array
            self.printv('Correlation values: %s' %(correl))
            self.printv('Sorted correlation: %s' %(sorted_correl) )
            no_split_found = False 

            while best_factor < 0: # Keep searching until the best factor is found


                if no_split_found:
                    target_cols = np.where(np.abs(correl)==sorted_correl[0])
                    best_factor = target_cols[0][0] # Best correl value is in sorted_correl[0], find value and look up in correl array to determine factor
                    splitVal = np.mean(Xtrain[:,best_factor])
                    self.printv("Couldn't find a decent split val, going to use the mean of highest correlated column which is %s" %(splitVal))
                else:
                    target_cols = np.where(np.abs(correl)==sorted_correl[current_index])
                    best_factor = target_cols[0][0] # Best correl value is in sorted_correl[0], find value and look up in correl array to determine factor
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
                self.printv('Data going to right shape[0] is %s' %to_the_right_data.shape[0])
                # Check to make sure using this factor will not result in empty branches:
                if to_the_left_data.shape[0]==0 or to_the_right_data.shape[0]==0: 
                    self.printv("THIS WOULD BE A BAD SPLIT - CURRENT INDEX %s" %(current_index))
                    best_factor = -1
                    current_index = current_index+1 # Move to the next highest correlated factor
                    if current_index > sorted_correl.shape[0]-1: # If we are on the last possible factor and it creates a split which is empty set no split found to true
                        no_split_found = True


                
            
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

        return queryY  # Return a ndarray for which each row represents the predicted value for Y


    # Given an ordered set of factors - X1,X2,X3,XN - returns the predicted value for Y based on the decision tree
    def get_prediction(self, xfactors):
        current_location = 0
        self.printv("TREE TABLE IS CURRENTLY \n %s \n" %self.tree_table)
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
