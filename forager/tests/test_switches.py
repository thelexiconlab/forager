import pytest
import pandas as pd
import numpy as np
from forager.cues import create_history_variables
from forager.switch import *

'''
Runs baseline tests for switch methods.
'''


# Example data
ex_list_1 = ['cat','dog','mouse','rat','giraffe','lion']



# Import Default Data (e.g. semantic_similarity item(s), norms)


#forager Test Data 
parentfolder = "../../data/"
norms = pd.read_csv(parentfolder + "norms/troyernorms.csv", encoding="unicode-escape")

freq_matrix = pd.read_csv(parentfolder + "lexical_data/frequencies.csv", header = None) 
labels = list(freq_matrix[0])
freq_matrix = np.array(freq_matrix[1])
sim_matrix = pd.read_csv(parentfolder + 'lexical_data/similaritymatrix.csv',delimiter=' ',header = None).values
phon_matrix= pd.read_csv(parentfolder + 'lexical_data/phonmatrix.csv',header = None).values

phon_matrix[phon_matrix <= 0] = .0001
sim_matrix[sim_matrix <= 0] = .0001

history_vars = create_history_variables(ex_list_1, labels, sim_matrix, freq_matrix, phon_matrix)

def test_troyer():
    '''
    Test Conditions:
        Generic Case(s):
            Fluency list is of size > 2 , returns vec of [2,...,...,...]
        Boundary Case(s):
            Fluency list is of size <= 2 : this should raise an exception
    '''
    assert switch_troyer(ex_list_1,norms) == [2, 0, 1, 0, 1, 0] # Generic Case 1 
    
def test_simdrop():
    '''
    Test Conditions:
        Generic Case(s):
            Fluency list is of size > 3 , returns vec of [2,...,...,2,2]
        Boundary Case(s):
            Fluency list is of size <= 3 : this should raise an exception
    '''     
    assert switch_simdrop(ex_list_1,history_vars[0]) == [2, 0, 1, 0, 1, 2] # Generic Case 1 


def test_multimodalsimdrop():
    '''
    Test Conditions:
        Generic Case(s):
            Fluency list is of size > 3 , returns vec of [2,...,...,2,2]   
                alpha = 1 should give same result as switch_simdrop
                alpha = 0 is when there is fully phonological cue 
        Boundary Case(s):
            Fluency list is of size <= 3 : this should raise an exception
            Check if alpha within range
    '''
    assert switch_multimodal(ex_list_1, history_vars[0], history_vars[4],1) == switch_simdrop(ex_list_1, history_vars[0]) #Generic Case 1 : fully similarity based
    assert switch_multimodal(ex_list_1, history_vars[0], history_vars[4],0) == [2, 0, 0, 0, 0, 2]
    assert switch_multimodal(ex_list_1, history_vars[0], history_vars[4],0.5) == [2, 0, 1, 0 , 0, 2]

def test_delta():
    '''
    Test Conditions:
        Generic Case(s):

        Boundary Case(s):
            Fluency list is of size <= 2 : this should raise an exception
            Check if rise_thresh, fall_thresh within range
    '''
    assert switch_delta(ex_list_1,history_vars[0],0.5,0.5) == [2, 0, 1, 0, 1, 0]
    assert switch_delta(ex_list_1,history_vars[0],0,0) == [2, 0, 1, 0, 1, 0]
    assert switch_delta(ex_list_1,history_vars[0],1,1) == [2, 0, 1, 0, 0, 0]


