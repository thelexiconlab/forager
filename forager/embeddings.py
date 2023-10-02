import pandas as pd 
import tensorflow as tf

import tensorflow_hub as hub
import numpy as np
import os
import re

class embeddings:
    '''
        Description: 
            This class contains functions that create the embeddings.csv file from a list of words
            using the Universal Sentence Encoder.
        
        Args:
            path_to_words: path to the csv file containing the list of words with 'vocab' as the header.
            
        Functions: 
            (1) __init__: creates USE_embeddings.csv file
            (2) test_embeddings: tests the similarity of two words using cosine similarity from scipy.
    
    '''
    def __init__(self, words, path_for_lexical_data): 
        #words = pd.read_csv(path_to_words)['vocab'].values.tolist()
        self.path = path_for_lexical_data + '/USE_embeddings.csv'
        # keep only unique words
        self.words = list(set(words))
        # convert to lowercase and sort alphabetically

        self.words = [w.lower() for w in self.words]
        self.words.sort()
        # write to vocab.csv with column header 'vocab'

        pd.DataFrame(self.words).to_csv(path_for_lexical_data + '/vocab.csv', index=False, header=['vocab'])

        # load USE model
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
        model = hub.load(module_url)
        print ("module %s loaded" % module_url)
        
        embeddings = []
        
        for v in self.words:
            embeddings.append(model([v]).numpy()[0])
        
        # create a dictionary of words and their embeddings without loop
        self.dict = dict(zip(self.words, embeddings))
        # convert dictionary to dataframe with column names as words and each column is the embedding

        self.df = pd.DataFrame(self.dict)
        # save dataframe as csv file
        self.df.to_csv(self.path, index=False)
        
#### SAMPLE RUN CODE ####
# embeddings('../data/lexical_data/vocab.csv') 