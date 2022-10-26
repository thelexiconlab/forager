import pytest
from forager.cues import *
import numpy as np
import pandas as pd

#TODO: update paths for test files
parentfolder = '../../data/'
path_to_embeddings = parentfolder + "lexical_data/semantic_embeddings.csv"
path_to_frequencies = parentfolder + "lexical_data/archive/psyrev/psyrev-frequencies.csv"
embeddings = pd.read_csv(path_to_embeddings, encoding="unicode-escape").transpose().values
labels = list(pd.read_csv(parentfolder + "lexical_data/frequencies.csv",header=None).values.tolist())

toy_semantic_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
toy_phonological_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
toy_freq_matrix = np.array([1,0,0,0])
toy_list = ['cat','dog','mouse','rat']

def test_create_history_variables():
    histvars_1 = create_history_variables(toy_list, toy_list, toy_semantic_matrix, toy_freq_matrix, phon_matrix = None)
    histvars_1 = create_history_variables(toy_list, toy_list, toy_semantic_matrix, toy_freq_matrix, toy_phonological_matrix)
    
    sim_list, sim_history, freq_list, freq_history,phon_list, phon_history = histvars_1
    assert len(sim_list) == len(sim_history)
    assert len(freq_list) == len(freq_history)
    assert len(phon_list) == len(phon_history)

def test_create_semantic_matrix(path = path_to_embeddings):
    
    semantic_matrix = create_semantic_matrix(path)
    diagonal = np.diag(semantic_matrix)
    # check that the diagonal is all 1s and the range of the matrix is [-1,1]
    assert diagonal.sum() == len(diagonal) and np.all(semantic_matrix) <= 1 and np.all(semantic_matrix) >= -1

def test_normalized_edit_distance():
    w1 = "hello"
    w2 = "world"
    assert phonology_funcs.normalized_edit_distance(w1, w2) == 0.2

def test_wordbreak():
    w = "hello"
    assert phonology_funcs.wordbreak(w)[0] == ['HH', 'AH0', 'L', 'OW1']

def test_create_phonological_matrix():
    phon_matrix = phonology_funcs.create_phonological_matrix(labels[:5])
    diagonal = np.diag(phon_matrix)
    # check that diagonal is 1 and range of matrix is [0,1]
    assert diagonal.sum() == len(diagonal) and np.all(phon_matrix) <= 1 and np.all(phon_matrix) >= 0


def test_get_labels_and_frequencies():
    labels, freq_matrix = get_labels_and_frequencies(path_to_frequencies)
    assert len(labels) == len(freq_matrix)
    assert np.all(freq_matrix >= 0)


