import numpy as np
import scipy
import pandas as pd
import nltk
from functools import lru_cache
from itertools import product as iterprod
import re
from tqdm import tqdm
import time
import pickle  # for saving and loading the wordbreak_labels


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
            (1) sim_list (list, size: L): semantic similarities between each item in fluency_list 
            (2) sim_history(list, size: L arrays of size N): semantic similarities of each word in fluency_list with all items in labels
            (3) phon_list (list, size: L): phonological similarities between each item in fluency_list 
            (4) phon_history (list, size: L arrays of size N): phonological similarities of each word in fluency_list with all items in labels
            (5) freq_list (list, size: L): frequencies of each item in fluency_list (list of size L)
            (6) freq_history  (list, size: L arrays of size N): frequencies of all words in labels repeated L times


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

def get_labels_and_frequencies(path_to_frequencies):
    '''
        Description:
            Returns search space of words and their log-frequencies
        Args:
            (1) path_to_frequencies (str): path to a .csv file containing N words (the full search space) and its log-frequencies (Nx2 array)
        Returns: 
            (1) labels (list, size: N): the space of words
            (2) freq_matrix (np.array, Nx1): a np.array of the frequencies of each word in labels
    '''

    freq_matrix = pd.read_csv(path_to_frequencies, header = None)
    labels = list(freq_matrix[0])
    freq_matrix = np.array(freq_matrix[1])

    return labels, freq_matrix

def create_semantic_matrix(path_to_embeddings, path_for_lexical_data):
    '''
        Description:
            Takes in N word embeddings and returns a semantic similarity matrix (NxN np.array)
        Args:
            (1) path_to_embeddings (str): path to a .csv file containing N word embeddings of size D each (DxN array)
        Returns: 
            (1) semantic_matrix: semantic similarity matrix (NxN np.array)
    '''
    embeddings = pd.read_csv(path_to_embeddings, encoding="unicode-escape").transpose().values
    N = len(embeddings)
    
    semantic_matrix = 1-scipy.spatial.distance.cdist(embeddings, embeddings, 'cosine').reshape(-1)
    semantic_matrix = semantic_matrix.reshape((N,N))
    # convert to dataframe without header or index
    semantic_matrix_df = pd.DataFrame(semantic_matrix)
    semantic_matrix_df.to_csv(path_for_lexical_data + '/USE_semantic_matrix.csv', header=False, index=False)
    return semantic_matrix

class phonology_funcs:
    '''
        Description: 
            This class contains functions to generate phonemes from a list of words and create a phonological similarity matrix.
            Code has been adapted from the following link: https://stackoverflow.com/questions/33666557/get-phonemes-from-any-word-in-python-nltk-or-other-modules
        Functions:
            (1) load_arpabet(): loads and returns the arpabet dictionary from the NLTK CMU dictionary
            (2) wordbreak(s, arpabet): takes in a word (str) and an arpabet dictionary and returns a list of phonemes
            (3) normalized_edit_distance(w1, w2): takes in two strings (w1, w2) and returns the normalized edit distance between them
            (3) create_phonological_matrix: takes in a list of labels (size N) and returns a phonological similarity matrix (NxN np.array)
    '''
    @lru_cache()
    def wordbreak(s):
        '''
            Description:
                Takes in a word (str) and an arpabet dictionary and returns a list of phonemes
            Args:
                (1) s (str): string to be broken into phonemes
            Returns:
                (1) phonemes (list, size: variable): list of phonemes in s 
        '''
        try:
            arpabet = nltk.corpus.cmudict.dict()
        except LookupError:
            nltk.download('cmudict')
            arpabet = nltk.corpus.cmudict.dict()
                
        s = s.lower()
        if s in arpabet:
            return arpabet[s]
        middle = len(s)/2
        partition = sorted(list(range(len(s))), key=lambda x: (x-middle)**2-x)
        for i in partition:
            pre, suf = (s[:i], s[i:])
            if pre in arpabet and phonology_funcs.wordbreak(suf) is not None:
                return [x+y for x,y in iterprod(arpabet[pre], phonology_funcs.wordbreak(suf))]
        return None

    def normalized_edit_distance(w1, w2):
        '''
            Description: 
                Takes in two strings (w1, w2) and returns the normalized edit distance between them
            Args:
                (1) w1 (str): first word
                (2) w2 (str): second word
            Returns:
                (1) normalized_edit_distance (float): normalized edit distance between w1 and w2
        '''
        return round(1-nltk.edit_distance(w1,w2)/(max(len(w1), len(w2))),4)

    def update_wordbreak_labels(new_labels, path_for_lexical_data):
        wordbreak_labels_file = path_for_lexical_data + '/wordbreak_labels.pkl'
        try:
            # Try to load the existing wordbreak_labels from the file
            with open(wordbreak_labels_file, 'rb') as f:
                wordbreak_labels = pickle.load(f)
        except FileNotFoundError:
            # If the file doesn't exist, create a new empty list
            wordbreak_labels = []

        start_time = time.time()
        # Compute the wordbreak for only the new labels
        new_wordbreak_labels = [phonology_funcs.wordbreak(l) for l in new_labels if l not in [label for sublist in wordbreak_labels for label in sublist]]
        wordbreak_labels.extend(new_wordbreak_labels)
        print(f"Time taken to update wordbreak_labels: {time.time() - start_time:.6f} seconds")

        start_time = time.time()
        # Save the updated wordbreak_labels to the file
        with open(wordbreak_labels_file, 'wb') as f:
            pickle.dump(wordbreak_labels, f)
        print(f"Time taken to save updated wordbreak_labels: {time.time() - start_time:.6f} seconds")

        return wordbreak_labels
    def create_phonological_matrix(labels, path_for_lexical_data):
        '''
            Description:
                Takes in a list of labels (size N) and returns a phonological similarity matrix (NxN np.array)
            Args:
                (1) labels: a list of words matching the size of your search space (list of length N)
            Returns: 
                (1) phonological_matrix: phonological similarity matrix (NxN np.array)
        '''
        start_time = time.time()
        labels = [re.sub('[^a-zA-Z]+', '', str(v)) for v in labels]
        print(f"Time taken to strip labels: {time.time() - start_time:.6f} seconds")

        wordbreak_labels_file = path_for_lexical_data + '/wordbreak_labels.pkl'
        wordbreak_labels = {}
        start_time = time.time()
        for i in tqdm(range(len(labels))):
            label = labels[i]
            wordbreak_labels[label] = phonology_funcs.wordbreak(label)
        print(f"Time taken for wordbreaks: {time.time() - start_time:.6f} seconds")

        with open(wordbreak_labels_file, 'wb') as f:
            pickle.dump(wordbreak_labels, f)
        print(f"Time taken to save wordbreak_labels: {time.time() - start_time:.6f} seconds")

        start_time = time.time()
        sim = np.zeros((len(labels), len(labels)))
        for i in tqdm(range(len(labels))):
            for j in range(i):
                sim[i, j] = phonology_funcs.normalized_edit_distance(wordbreak_labels[labels[i]][0], wordbreak_labels[labels[j]][0])
        sim = sim + sim.T
        np.fill_diagonal(sim, 1)
        print(f"Time taken for similarity calculation: {time.time() - start_time:.6f} seconds")

        start_time = time.time()
        phon_matrix_df = pd.DataFrame(sim)
        phon_matrix_df.to_csv(path_for_lexical_data + '/USE_phon_matrix.csv', header=False, index=False)
        print(f"Time taken to save the matrix: {time.time() - start_time:.6f} seconds")

        return sim


    def update_phonological_matrix(existing_labels, new_labels, path_for_lexical_data):
        wordbreak_labels_file = path_for_lexical_data + '/wordbreak_labels.pkl'
        try:
            # Load the existing wordbreak_labels from the file
            with open(wordbreak_labels_file, 'rb') as f:
                wordbreak_labels = pickle.load(f)
        except FileNotFoundError:
            # If the file doesn't exist, create a new dictionary
            wordbreak_labels = {}

        # Compute the wordbreak for the new labels and update the dictionary
        start_time = time.time()
        for label in new_labels:
            if label not in wordbreak_labels:
                wordbreak_labels[label] = phonology_funcs.wordbreak(label)
        print(f"Time taken to update wordbreak_labels: {time.time() - start_time:.6f} seconds")

        start_time = time.time()
        # Save the updated wordbreak_labels to the file
        with open(wordbreak_labels_file, 'wb') as f:
            pickle.dump(wordbreak_labels, f)
        print(f"Time taken to save updated wordbreak_labels: {time.time() - start_time:.6f} seconds")

        all_labels = list(wordbreak_labels.keys())
        start_time = time.time()
        # Compute the new phonological similarity matrix
        sim = np.zeros((len(all_labels), len(all_labels)))
        for i in tqdm(range(len(all_labels))):
            for j in range(i):
                sim[i, j] = phonology_funcs.normalized_edit_distance(wordbreak_labels[all_labels[i]][0], wordbreak_labels[all_labels[j]][0])
        sim = sim + sim.T
        np.fill_diagonal(sim, 1)
        print(f"Time taken for similarity calculation: {time.time() - start_time:.6f} seconds")

        start_time = time.time()
        phon_matrix_df = pd.DataFrame(sim)
        phon_matrix_df.to_csv(path_for_lexical_data + '/USE_new_phon_matrix.csv', header=False, index=False)
        print(f"Time taken to save the matrix: {time.time() - start_time:.6f} seconds")

        return sim
    
    def get_phonological_similarity(word1, word2):
        '''
            Description:
                Takes in two words and returns their phonological similarity
            Args:
                (1) word1 (str): first word
                (2) word2 (str): second word
            Returns:
                (1) phonological_similarity (float): phonological similarity between word1 and word2
        '''
        
        word1 = re.sub('[^a-zA-Z]+', '', str(word1))
        word2 = re.sub('[^a-zA-Z]+', '', str(word2))
        print(f"word1={word1}, word2={word2}")
        phon1 = phonology_funcs.wordbreak(word1)[0]
        phon2 = phonology_funcs.wordbreak(word2)[0]
        print(f"phon1={phon1}, phon2={phon2}")
        print(phonology_funcs.normalized_edit_distance(phon1, phon2))
    
    def check_phon_matrix(path_for_lexical_data):
        phon_matrix = np.loadtxt(path_for_lexical_data +'USE_new_phon_matrix.csv',delimiter=',')
        print("shape of phon_matrix: ", phon_matrix.shape)
        labels = pd.read_csv(path_for_lexical_data +'vocab_semantigories.csv')['word'].values.tolist()
        # generate 5 random pairs of words
        for i in range(5):
            word1 = labels[np.random.randint(0, len(labels))]
            word2 = labels[np.random.randint(0, len(labels))]
            matrix_sim= np.round(phon_matrix[labels.index(word1), labels.index(word2)], 2)
            word1 = re.sub('[^a-zA-Z]+', '', str(word1))
            word2 = re.sub('[^a-zA-Z]+', '', str(word2))
            phon1 = phonology_funcs.wordbreak(word1)[0]
            phon2 = phonology_funcs.wordbreak(word2)[0]
            phon_similarity = np.round(phonology_funcs.normalized_edit_distance(phon1, phon2), 2)
            print(f"Direct Phonological similarity for {word1} and {word2} is {phon_similarity}")
            print(f"Matrix Phonological similarity for {word1} and {word2} is {matrix_sim}")
            assert phon_similarity == matrix_sim
            # print message if all assertions pass
            print("assertion passed")
        
        if phon_matrix.shape[0] != phon_matrix.shape[1]:
            raise ValueError("Phonological matrix is not square")
    

### SAMPLE RUN CODE ###
#create_semantic_matrix('../data/lexical_data/USE_embeddings.csv')
# v = pd.read_csv('../data/lexical_data/animals/vocab_semantigories.csv')['word'].values.tolist()
# print("length of v: ", len(v))
# phonology_funcs.create_phonological_matrix(v, '../data/lexical_data/animals')

# phonology_funcs.get_phonological_similarity('able seaman', 'accounts payable')
# phonology_funcs.check_phon_matrix('../data/lexical_data/occupations/')