import pytest
import pandas as pd
import numpy as np
from semforage.cues import create_history_variables
from semforage.switch import *
'''
Runs baseline tests for switch methods. Switch methods are verified on data from psyrev & fmri data

#TODO: Boundary cases & Exception handling in pytest
https://miguendes.me/how-to-check-if-an-exception-is-raised-or-not-with-pytest




'''


# Example data
ex_list_1 = ['cat','dog','mouse','rat','giraffe','lion']


# Psych Review Test Data 
# Import Default Data (e.g. semantic_similarity item(s), norms)

norms = pd.read_csv("psyrev_files/troyernorms.csv", encoding="unicode-escape")

sim_matrix = pd.read_csv("psyrev_files/similaritymatrix.csv",header=None).dropna(axis=1).values
labels = list(pd.read_csv("psyrev_files/similaritylabels.csv",header=None).squeeze().values)
freq_data = pd.read_csv("psyrev_files/frequencies.csv",header=None).dropna(axis=1)
#reorder freq_matrix to match sim_matrix
freq_matrix = []
for label in labels:
    freq_matrix.append(freq_data[freq_data[0] == label].squeeze().to_dict())
freq_matrix = np.array(pd.DataFrame(freq_matrix)[1])

sim_matrix[sim_matrix <= 0] = .0001

history_vars = create_history_variables(ex_list_1,labels,sim_matrix,freq_matrix)

#Semforager Test Data 
parentfolder = "regards_files/"
freq_matrix_regards = pd.read_csv(parentfolder + "regards_final_frequencies.csv", header = None) 
labels_regards = list(freq_matrix_regards[0])
freq_matrix_regards = np.array(freq_matrix_regards[1])
sim_matrix_regards = pd.read_csv(parentfolder + 'word2vec_sim_matrix.csv',header = None).values
phon_matrix_regards= pd.read_csv(parentfolder + 'simlabels_phon_matrix.csv',header = None).values


phon_matrix_regards[phon_matrix_regards <= 0] = .0001
sim_matrix_regards[sim_matrix_regards <= 0] = .0001

print(ex_list_1)
print(history_vars[0])
print(switch_simdrop(ex_list_1,history_vars[0]))    
history_vars_regards = create_history_variables(ex_list_1, labels_regards, sim_matrix_regards, freq_matrix_regards, phon_matrix_regards)

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
    assert switch_simdrop(ex_list_1,history_vars[0]) == [2, 0, 1, 0, 2, 2] # Generic Case 1 


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
    assert switch_multimodal(ex_list_1, history_vars_regards[0], history_vars_regards[4],1) == switch_simdrop(ex_list_1, history_vars_regards[0]) #Generic Case 1 : fully similarity based
    assert switch_multimodal(ex_list_1, history_vars_regards[0], history_vars_regards[4],0) == [2, 0, 0, 0, 2, 2]
    assert switch_multimodal(ex_list_1, history_vars_regards[0], history_vars_regards[4],0.5) == [2, 0, 1, 0 , 2, 2]

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


