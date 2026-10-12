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

def test_pei_probabilities():
    '''
    Test Conditions:
        Generic Case(s):
            return_probabilities=True returns (decisions, probabilities) with the same decisions
            Probabilities are in (0, 1) and decisions[i] == int(prob[i] < 0.5) for non-boundary items
            First item is NaN
            Reference values for participant 50001 in the Lundin et al. (2023) sample (alpha 0.8, beta 0.9, prior 0.5)
        Boundary Case(s):
            Fluency list is of size <= 2, or no slope differences: all probabilities are NaN
    '''
    _, slope_diffs = switch_slope_difference(ex_list_1, ex_times_1)
    kwargs = dict(semantic_similarity=history_vars[0], phonological_similarity=history_vars[4],
                  slope_diffs=slope_diffs, alpha=0.8, beta=0.9, prior_probability=0.5)

    decisions, probs = switch_pei(ex_list_1, ex_times_1, return_probabilities=True, **kwargs)
    assert decisions == switch_pei(ex_list_1, ex_times_1, **kwargs)
    assert len(probs) == len(ex_list_1)
    assert np.isnan(probs[0])
    for d, p in zip(decisions[1:], probs[1:]):
        assert 0 < p < 1
        assert d == int(p < 0.5)

    # Boundary: short list
    short_decisions, short_probs = switch_pei(['cat', 'dog'], [2.0, 4.5], semantic_similarity=[0, 0.5],
                                              return_probabilities=True)
    assert short_decisions == [2, 2]
    assert len(short_probs) == 2 and np.isnan(short_probs).all()

    # Boundary: no slope differences
    _, no_slope_probs = switch_pei(ex_list_1, ex_times_1, semantic_similarity=history_vars[0],
                                   slope_diffs=np.array([]), return_probabilities=True)
    assert np.isnan(no_slope_probs).all()

    # Reference values: participant 50001, words 2-15
    data = pd.read_csv(parentfolder + "fluency_lists/animals_timings_Lundin_et_al_2023.csv")
    subj = data[data['SID'] == 50001]
    words, times = subj['entry'].tolist(), subj['time'].tolist()
    hv = create_history_variables(words, labels, sim_matrix, freq_matrix, phon_matrix)
    _, sd = switch_slope_difference(words, times)
    decisions, probs = switch_pei(words, times, hv[0], hv[4], slope_diffs=sd, alpha=0.8, beta=0.9,
                                  prior_probability=0.5, return_probabilities=True)
    expected = [.55, .30, .48, .80, .60, .57, .61, .63, .81, .19, .80, .55, .35, .51]
    assert np.round(probs[1:15], 2).tolist() == expected
    assert decisions == [2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]

def test_cluster_lengths():
    '''
    Test Conditions:
        Generic Case(s):
            A switch starts a new cluster; the switch word is not counted in the previous cluster
            A list ending in a switch ends with a cluster of size 1
        Boundary Case(s):
            No switches: a single cluster spanning the list
    '''
    # PEI codes for participant 50001 (alpha 0.8, beta 0.9, prior 0.5)
    pei_50001 = [2,0,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,1,1,1,0,1,1,1,1]
    assert cluster_lengths(pei_50001) == [2,1,7,3,5,2,1,1,1,2,1,1,1,1]
    assert sum(cluster_lengths(pei_50001)) == len(pei_50001)
    assert cluster_lengths([2, 0, 1, 0, 1, 2]) == [2, 2, 2]
    assert cluster_lengths([2, 0, 0, 0]) == [4]
    assert cluster_lengths([2]) == [1]
