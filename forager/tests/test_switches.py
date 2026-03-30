import os
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
ex_times_1 = [2.0, 4.5, 7.0, 11.0, 18.0, 25.0] # cumulative response times in seconds



# Import Default Data (e.g. semantic_similarity item(s), norms)


#forager Test Data
parentfolder = os.path.join(os.path.dirname(__file__), "../../data/")
norms = pd.read_csv(parentfolder + "norms/animals_snafu_scheme_vocab.csv", encoding="unicode-escape")

lexical_folder = parentfolder + "lexical_data/animals/"
freq_matrix = pd.read_csv(lexical_folder + "USE_frequencies.csv", header=None)
labels = list(freq_matrix[0])
freq_matrix = np.array(freq_matrix[1])
sim_matrix = pd.read_csv(lexical_folder + 'USE_semantic_matrix.csv', delimiter=',', header=None).values
phon_matrix = pd.read_csv(lexical_folder + 'USE_phonological_matrix.csv', header=None).values

phon_matrix[phon_matrix <= 0] = .0001
sim_matrix[sim_matrix <= 0] = .0001

history_vars = create_history_variables(ex_list_1, labels, sim_matrix, freq_matrix, phon_matrix)

def test_norms_associative():
    '''
    Test Conditions:
        Generic Case(s):
            Fluency list is of size > 2 , returns vec of [2,...,...,...]
        Boundary Case(s):
            Fluency list is of size <= 2 : this should raise an exception
    '''
    assert switch_norms_associative(ex_list_1,norms) == [2, 0, 1, 0, 1, 0] # Generic Case 1
    
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
    assert switch_delta(ex_list_1,history_vars[0],1,1) == [2, 0, 0, 0, 1, 0] # updated: USE_semantic_matrix has different similarity values than old similaritymatrix

def test_slope_difference():
    '''
    Test Conditions:
        Generic Case(s):
            Fluency list is of size > 2, returns (decisions, slope_diffs)
        Boundary Case(s):
            Fluency list is of size <= 2: returns all boundary markers
    '''
    decisions, slope_diffs = switch_slope_difference(ex_list_1, ex_times_1)
    assert decisions == [2, 1, 0, 1, 1, 0]
    assert len(slope_diffs) == len(ex_list_1) - 1

    # Boundary: short list
    short_decisions, short_diffs = switch_slope_difference(['cat', 'dog'], [2.0, 4.5])
    assert short_decisions == [2, 2]
    assert len(short_diffs) == 0

def test_pei():
    '''
    Test Conditions:
        Generic Case(s):
            With semantic similarity only
            With both semantic and phonological similarity
        Boundary Case(s):
            Fluency list is of size <= 2: returns all boundary markers
            Missing similarity raises ValueError
    '''
    # Precompute slope differences
    _, slope_diffs = switch_slope_difference(ex_list_1, ex_times_1)

    # Semantic only
    assert switch_pei(ex_list_1, ex_times_1, semantic_similarity=history_vars[0], slope_diffs=slope_diffs) == [2, 0, 0, 0, 1, 0]

    # Both modalities
    assert switch_pei(ex_list_1, ex_times_1, semantic_similarity=history_vars[0],
                      phonological_similarity=history_vars[4], slope_diffs=slope_diffs) == [2, 0, 0, 0, 1, 0]

    # Boundary: short list
    assert switch_pei(['cat', 'dog'], [2.0, 4.5], semantic_similarity=[0, 0.5]) == [2, 2]

    # Missing similarity raises error
    with pytest.raises(ValueError):
        switch_pei(ex_list_1, ex_times_1, slope_diffs=slope_diffs)
