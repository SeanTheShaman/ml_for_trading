import numpy as np 
import math 
class BagLearner():

    verbose = False
    learners = [] # A list of learner instances  
    bag_count = 20 
    bag_size = 0
    bags = []
    boost = False 
    
    def __init__(self, learner, kwargs={}, bags=20, boost=False, verbose=False):
        self.bag_count = bags
        self.bag_size = 0
        self.boost = boost
        self.verbose = verbose
        learners = [] 
        # Instantiate the learners:
        for i in range(0, bags):
            learners.append(learner(**kwargs))
        self.learners = learners 

    def addEvidence(self, Xtrain, Ytrain):
        self.printv('X-TRAIN: %s \n Y-TRAIN %s' %(Xtrain, Ytrain))
        data = np.hstack( (Xtrain, Ytrain.reshape(Ytrain.shape[0], 1))) # Concatenate the X/Y data 
        self.printv('Concatenated Data: ' + str(data))
        self.printv('*************** CREATING BAGS **********************\n')
        # Determine how many elements should be in each bag - using replacement so #elements*#bags can be > than # elements in Xtrain data
        self.bag_size = int(math.ceil(Xtrain.shape[0]/float(self.bag_count))) # Cast to float and round up.
        self.printv('Number of X Train Elements %s. \n Number of Bags: %s \n Number of elements in each bag: %s' %(Xtrain.shape[0], self.bag_count, self.bag_size ))
        
        # Split training data randomly into m (bag_count) bags of n (bag_size) elements each
        bags_created_count = 0
        while bags_created_count < self.bag_count:
            # Use numpy to generate random array of indexes - these indexes correspond to elements in Xtrain - don't care about duplicates b/c with replacement
            indices_to_grab = np.random.randint(0, data.shape[0]-1, (int(self.bag_size)))  # Create random index array the length of bag size
            self.printv('Random index array for bag no %s is %s' %(bags_created_count, indices_to_grab ))
            this_bag = data[indices_to_grab]
            self.printv('Values grabbed for this bag are %s' %(this_bag))
            self.bags.append(this_bag)  # Append created bag to bags list
            bags_created_count += 1

        self.printv('**************** BAGS HAVE BEEN CREATED. COUNT IS %s **********************\n' %(len(self.bags)) )
        # Current State: All bags have been created.

        self.printv("*************** TRAINING LEARNERS ON CREATED BAGS ********************")
        self.printv("LENGTH OF SELF.LEARNERS " + str(len(self.learners)) )
        for i in range(0, len(self.learners)): # Next run created bags through learners 
            self.learners[i].addEvidence(self.bags[i][:,0:-1], self.bags[i][:,-1]) # pass the ith bag to the ith learner - splitting out the Xtrain and Ytrain values
        self.printv("****************** TRAINING LEARNERS COMPLETE ************************")

        return 

    def query(self, Xtest):
        self.printv('************** QUERYING FOR %s *******************\n' %(Xtest))

        # Get predictions from each of the learners 
        results = np.zeros( (Xtest.shape[0],0) )
        for learner in self.learners: 
            this_result = learner.query(Xtest) # Query the learner
            self.printv('THIS RESULT IS %s \n' %(this_result))
            results = np.hstack( (results, this_result.reshape(this_result.shape[0], 1)) )


        self.printv('*************** QUERYING COMPLETE. \n RESULTS SHAPE: %s \n RESULTS: \n %s' %(results.shape, results))


        # Calculate and return the mean
        mean_results = np.mean(results, axis=1)
        self.printv('** MEAN RESULTS ARRAY: \n %s \n' %(mean_results))
        return mean_results

    def author(self):
        return 'scox43'

    def printv(self, some_string):
        if self.verbose:
            print some_string
        
