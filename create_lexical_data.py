#### Run data.py to get tab-delimited txt file and lexical data files to use for forager
## Files created : data/input_files/word_list.txt
## Files created: data/lexical_data/frequencies.csv
#                 data/lexical_data/phonmatrix.csv
#                 data/lexical_data/similaritymatrix.csv


import pandas as pd 

from forager.embeddings import embeddings
from forager.frequency import get_frequencies
from forager.cues import get_labels_and_frequencies
from forager.cues import phonology_funcs
from forager.cues import create_semantic_matrix
import os 

class data: 
    '''
        Description: 
            data class contains functions that help with creating the lexical data files.     
    '''
    
    def __init__(self, words):
        self.path = 'data/lexical_data/NEW'

        # Check whether the specified path exists or not
        isExist = os.path.exists(self.path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.path)
        
        #creating embeddings 
        embeddings(words, self.path)
        print("\nCreated and saved embeddings as USE_embeddings.csv inside " + self.path) 
        
        #get frequencies 
        get_frequencies(self.path + '/USE_embeddings.csv', self.path)
        print("\nCreated and saved word frequencies as USE_frequencies.csv inside " + self.path)  
        
        
        # get semantic matrix 
        create_semantic_matrix(self.path + '/USE_embeddings.csv', self.path)
        print("\nCreated and saved semantic similarity matrix as USE_semantic_matrix.csv inside " + self.path)  
        
        # get phonological matrix 
        labels, freq_matrix = get_labels_and_frequencies(self.path + '/USE_frequencies.csv')
        phonology_funcs.create_phonological_matrix(labels, self.path)
        print("\nCreated and saved phonological similarity matrix as USE_phon_matrix.csv inside " + self.path) 

#### SAMPLE RUN CODE ####
# data(['apple', 'mango'])