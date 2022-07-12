import numpy as np

'''

Generate Cue History for each type of cue (similarity, frequency, phonology). Includes
    additional functions to support the creation of similarity matrices and frequency lists.

    Functions
        (1) create_history_variables: creates similarity, frequency, and phonology list and history
        variables to be used by foraging methods in foraging.py
        (2) create_semantic_matrix: converts a word embedding space into a similarity matrix
        (3) phonology_funcs: class to execute the creation of a phonological similarity matrix
'''


def create_history_variables(fluency_list, labels, sim_matrix, freq_matrix, phon_matrix = None):
    '''
        Args:
            (1) sim_matrix: semantic similarity matrix (NxN np.array)
            (2) phon_matrix: phonological similarity matrix (NxN np.array)
            (3) freq_matrix: frequencies array (Nx1 array)
            (4) labels: the space of words (list of length N)
            (5) fluency_list: items produced by a participant (list of size L)

        Returns: 
            (1) sim_list: semantic similarities between each item in fluency_list (list of size L)
            (2) sim_history: semantic similarities of each word with all items in labels
                 (list of L arrays of size N)
            (3) phon_list: phonological similarities between each item in fluency_list
                 (list of size L)
            (4) phon_history: phonological similarities of each word with all items in labels
                 (list of L arrays of size N)
            (5) freq_list: frequencies of each item in fluency_list (list of size L)
            (6) freq_history: frequencies of all items in labels repeated L items
                 (list of L arrays of size N)


    '''
    if phon_matrix is not None:
        phon_matrix[phon_matrix <= 0] = .0001
    sim_matrix[sim_matrix <= 0] = .0001

    freq_list = []
    freq_history = []

    sim_list = []
    sim_history = []

    phon_list = []
    phon_history = []

    for i in range(0,len(fluency_list)):
        word = fluency_list[i]
        currentwordindex = labels.index(word)

        freq_list.append(freq_matrix[currentwordindex])
        freq_history.append(freq_matrix)

        if i > 0: # get similarity between this word and preceding word
            prevwordindex = labels.index(fluency_list[i-1])
            sim_list.append(sim_matrix[prevwordindex, currentwordindex] )
            sim_history.append(sim_matrix[prevwordindex,:])
            if phon_matrix is not None:
                phon_list.append(phon_matrix[prevwordindex, currentwordindex] )
                phon_history.append(phon_matrix[prevwordindex,:])
        else: # first word
            sim_list.append(0.0001)
            sim_history.append(sim_matrix[currentwordindex,:])
            if phon_matrix is not None:
                phon_list.append(0.0001)
                phon_history.append(phon_matrix[currentwordindex,:])

    return sim_list, sim_history, freq_list, freq_history,phon_list, phon_history

'''
TODO: Abhilasha
- Add Sim + Freq functions
'''