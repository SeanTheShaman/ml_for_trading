import numpy as np 
import BagLearner as bl 
import LinRegLearner as lrl 

class InsaneLearner():

    learners = []
    verbose = False 

    def __init__(self, verbose=False):
        self.verbose = verbose

        learners = []
        for i in range(20): # Create 20 BagLearners with 20 LinRegLearners.
            learners.append(bl.BagLearner(lrl.LinRegLearner, verbose=self.verbose) )
            
        self.learners = learners
        
        # Current State: self.learners contains array of BagLearner instances which in turn contain arrays of LinRegLearner instances

        

    def addEvidence(self, Xtrain, Ytrain):
        # Call the addEvidence function in each of the "sub-learners"
        for learner in self.learners: 
            learner.addEvidence(Xtrain, Ytrain)
        

    def query(self, Xtest):
        results = np.zeros( (Xtest.shape[0],0) )

        for learner in self.learners: 
            this_result = learner.query(Xtest) # Query the learner
            results = np.hstack( (results, this_result.reshape(this_result.shape[0], 1)) )
        mean_results = np.mean(results, axis=1)
        return mean_results

    def author(self):
        return "scox43"

    
