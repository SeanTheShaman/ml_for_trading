"""
Template for implementing QLearner  (c) 2015 Tucker Balch
Additional code written by Sean Cox - July 2018
"""

import numpy as np
import random as rand

class QLearner(object):

    def printv(self, some_string):
        """
        @summary: Prints a string if verbose mode is true
        @param some_string: The string to print
        """
        if self.verbose:
            print some_string
    
    def author(self): # The good ole author function
        return "scox43"

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0
        self.a = 0
        self.q_table = np.zeros((num_states, num_actions)) # Q-table is numpy array where rows represent states, columns represent actions, and cell values represent reward
        self.alpha = alpha 
        self.gamma = gamma 
        self.rar = rar 
        self.radr = radr 
        self.dyna = dyna 

        if self.dyna > 0: # If running dyna then set-up structure for experience replay
   
            self.replay_memory = [] # Create replay memory array in which each row will represent s, a, r, s' 

            


    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        random_val = rand.random() # Roll the dice

        if random_val > self.rar: # If the random value is greater than the RAR then choose the best action
            action = np.argmax(self.q_table[s, :])
        else:
            action = np.random.randint(0, self.num_actions) # Less than RAR so choose a random action

        self.a = action # Remember current action for next iteration
        self.s = s# Remember current state for next iteration
        return action

    def q(self, s, a, s_prime, r):
        q_value = ""
        #self.printv('Q-Function. Alpha: %s. S: %s. A:%s' %(self.alpha, s, a))
        part1 = (1-self.alpha)*self.q_table[s, a]
        part2 = self.alpha*(r + self.gamma*np.max(self.q_table[s_prime, :]) )
        q_value = part1 + part2
        #self.printv("P1: %s. P2: %f.  Q-VALUE: %f" %(part1, part2, q_value))
        return q_value

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The reward for the last action taken
        @returns: The selected action
        """


        # UPDATE RULE
        # Update the Q-Table with this transition:
        self.q_table[self.s, self.a] = self.q(self.s, self.a, s_prime, r)
        
        # PROCESS DYNA - using experience replay
        if self.dyna > 0:
            self.replay_memory.append((self.s, self.a, s_prime, r))# Add this real-world experience to replay memory
            for i in range(0, self.dyna):
                mem_s, mem_a, mem_sprime, mem_r = rand.choice(self.replay_memory) # Choose random experience tuple from replay memory
                #self.printv('Mem S: %s Mem A: %s Mem SPrime: %s Mem R: %s' %(mem_s, mem_a, mem_sprime, mem_r))
                self.q_table[mem_s, mem_a] = self.q(mem_s, mem_a, mem_sprime, mem_r) # Update the Q-Table with "hallucinated" values


        # QUERY / DETERMINE BEST ACTION

        # Should take random action or not? 
        random_val = rand.random() # Roll the dice
    
        if random_val > self.rar: # If the random value is greater than the RAR then choose the best action
            action = np.argmax(self.q_table[s_prime, :])
        else:
            action = np.random.randint(0, self.num_actions) # Less than RAR so choose a random action

        self.rar = self.rar*self.radr # Decay the Random Action Rate (rar) by the decay rate (radr)
        self.a = action # Remember current action for next iteration
        self.s = s_prime # Remember current state for next iteration
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
