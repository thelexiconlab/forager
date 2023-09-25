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
import difflib 
import os 
import re
from alive_progress import alive_bar 


class data: 
    '''
        Description: 
            Embeddings class contains functions that help with creating the lexical data files. 
            Files 
    
    
    '''
    
    def __init__(self, words):
        self.path = '../data/lexical_data/NEW/'

        # Check whether the specified path exists or not
        isExist = os.path.exists(self.path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.path)
        
        #creating embeddings 
        embeddings(words, self.path)
        print("created embeddings") 
        
        #get frequencies 
        get_frequencies(self.path + '/USE_embeddings.csv')
        print("created frequencies") 
        
        
        # get semantic matrix 
        create_semantic_matrix(self.path + '/USE_embeddings.csv')
        print("created semantic matrix") 
        
        # get phonological matrix 
        labels, freq_matrix = get_labels_and_frequencies(self.path + '/frequencies.csv')
        phonology_funcs.create_phonological_matrix(labels)
        print("created phon matrix")