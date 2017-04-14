#!/usr/bin/env python 

from util import * 
from numpy import *
from math import log
import copy
import sys
inf = float("inf")

# Pretty printing for 1D/2D numpy arrays
MAX_PRINTING_SIZE = 30

def format_array(arr):
    s = shape(arr)
    if s[0] > MAX_PRINTING_SIZE or (len(s) == 2 and s[1] > MAX_PRINTING_SIZE):
        return "[  too many values (%s)   ]" % s

    if len(s) == 1:
        return  "[  " + (
            " ".join(["%.6f" % float(arr[i]) for i in range(s[0])])) + "  ]"
    else:
        lines = []
        for i in range(s[0]):
            lines.append("[  " + "  ".join(["%.6f" % float(arr[i,j]) for j in range(s[1])]) + "  ]")
        return "\n".join(lines)



def format_array_print(arr):
    print format_array(arr)


def string_of_model(model, label):
    (initial, tran_model, obs_model) = model
    return """
Model: %s 
initial: 
%s

transition: 
%s

observation: 
%s
""" % (label, 
       format_array(initial),
       format_array(tran_model),
       format_array(obs_model))

    
def check_model(model):
    """Check that things add to one as they should"""
    (initial, tran_model, obs_model) = model
    for state in range(len(initial)):
        assert((abs(sum(tran_model[state,:]) - 1)) <= 0.01)
        assert((abs(sum(obs_model[state,:]) - 1)) <= 0.01)
        assert((abs(sum(initial) - 1)) <= 0.01)


def print_model(model, label):
    check_model(model)
    print string_of_model(model, label)    

def max_delta(model, new_model):
    """Return the largest difference between any two corresponding 
    values in the models"""
    return max( [(abs(model[i] - new_model[i])).max() for i in range(len(model))] )


class HMM:
    """ HMM Class that defines the parameters for HMM """
    def __init__(self, states, outputs):
        """If the hmm is going to be trained from data with labeled states,
        states should be a list of the state names.  If the HMM is
        going to trained using EM, states can just be range(num_states)."""
        self.states = states
        self.outputs = outputs
        n_s = len(states)
        n_o = len(outputs)
        self.num_states = n_s
        self.num_outputs = n_o
        self.initial = zeros(n_s)
        self.transition = zeros([n_s,n_s])
        self.observation = zeros([n_s, n_o])

    def set_hidden_model(self, init, trans, observ):
        """ Debugging function: set the model parameters explicitly """
        self.num_states = len(init)
        self.num_outputs = len(observ[0])
        self.initial = array(init)
        self.transition = array(trans)
        self.observation = array(observ)
        
    def get_model(self):
        return (self.initial, self.transition, self.observation)

    def compute_logs(self):
        """Compute and store the logs of the model (helper)"""
        raise Exception("Not implemented")

    def __repr__(self):
        return """states = %s
observations = %s
%s
""" % (" ".join(array_to_string(self.states)), 
       " ".join(array_to_string(self.outputs)), 
       string_of_model((self.initial, self.transition, self.observation), ""))

     
    # declare the @ decorator just before the function, invokes print_timing()
    @print_timing
    def learn_from_labeled_data(self, state_seqs, obs_seqs):
        """
        Learn the parameters given state and observations sequences. 
        The ordering of states in states[i][j] must correspond with observations[i][j].
        Use Laplacian smoothing to avoid zero probabilities.
        Implement for (a).
        """
        print self.states
        # Fill this in...
        n_s = self.num_states
        n_o = self.num_outputs
        N = len(state_seqs)
        N_1k = [0. for i in range(n_s)]
        N_k = [0. for i in range(n_s)]
        N__nk = [0. for i in range(n_s)]
        N_kj = [[0. for i in range(n_o)] for j in range(n_s)]
        N_kl = [[0. for i in range(n_s)] for j in range(n_s)]

        for states, obs in zip(state_seqs, obs_seqs):
            prev_state = None
            seq_len = len(states)
            for t in range(seq_len):
                if t == 0:
                    # Update initial counts (theta)
                    N_1k[states[t]] += 1
                # Update observation in a state count (pi)
                N_kj[states[t]][obs[t]] += 1
                N_k[states[t]] += 1

                # Update transition count (t)
                if t != (seq_len - 1):
                    N_kl[states[t]][states[t+1]] += 1
                    N__nk[states[t]] += 1

        # Normalize Counts
        for k in range(n_s):
            self.initial[k] = (N_1k[k] + 1.0) / (N + n_s)

        for k in range(n_s):
            for l in range(n_s):
                self.transition[k][l] = (N_kl[k][l] + 1) / (N__nk[k] + n_s)

        for k in range(n_s):
            for j in range(n_o):
                self.observation[k][j] = (N_kj[k][j] + 1) / (N_k[k] + n_o)
        

    def most_likely_states(self, sequence, debug=False):
        """Return the most like sequence of states given an output sequence.
        Uses Viterbi algorithm to compute this.
        Implement for (b) and (c).
        """
        theta = self.initial
        transition = self.transition
        pi = self.observation
        num_states = self.num_states
        n = len(sequence)

        v = [[-3 for i in range(num_states)] for j in range(n)]
        z = [[-3 for i in range(num_states)] for j in range(n)]
        """
        for t in range(n):
            # s = s_{t}
            for s in range(num_states):
                if t == 0:
                    v[t][s] = pi[s][sequence[t]] * theta[s]
                    z[t][s] = -1
                else:
                    biggest = -5
                    # i = s_{t-1}
                    for i in range(num_states):
                        if transition[i][s] * v[t - 1][i] > biggest:
                            biggest = transition[i][s] * v[t - 1][i]
                            z[t][s] = i
                    v[t][s] = pi[s][sequence[t]] * biggest

        final = v[n-1].index(max(v[n - 1]))
        best_states = [final]

        prev = final
        for i in range(n-1,0,-1):
            best_states.append(z[i][prev])
            prev = z[i][prev]

        best_states = best_states[::-1]
        """

        for t in range(n):
            # s = s_{t}
            for s in range(num_states):
                if t == 0:
                    v[t][s] = log(pi[s][sequence[t]]) + log(theta[s])
                    z[t][s] = -1
                else:
                    biggest = -inf
                    # i = s_{t-1}
                    for i in range(num_states):
                        if log(transition[i][s]) + v[t - 1][i] > biggest:
                            biggest = log(transition[i][s]) + v[t - 1][i]
                            z[t][s] = i
                    v[t][s] = log(pi[s][sequence[t]]) + biggest

        final = v[n-1].index(max(v[n - 1]))
        best_states = [final]

        prev = final
        for i in range(n-1,0,-1):
            best_states.append(z[i][prev])
            prev = z[i][prev]

        best_states = best_states[::-1]

        return best_states

    
def get_wikipedia_model():
    # From the rainy/sunny example on wikipedia (viterbi page)
    hmm = HMM(['Rainy','Sunny'], ['walk','shop','clean'])
    init = [0.6, 0.4]
    trans = [[0.7,0.3], [0.4,0.6]]
    observ = [[0.1,0.4,0.5], [0.6,0.3,0.1]]
    hmm.set_hidden_model(init, trans, observ)
    return hmm

def get_toy_model():
    hmm = HMM(['h1','h2'], ['A','B'])
    init = [0.6, 0.4]
    trans = [[0.7,0.3], [0.4,0.6]]
    observ = [[0.1,0.9], [0.9,0.1]]
    hmm.set_hidden_model(init, trans, observ)
    return hmm
    

