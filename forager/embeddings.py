#### NOTE: If using pymagnitude gives you an error, find the path to params.py inside the pymagnitude folder
### and in the file, change the line "from collections import MutableMapping, OrderedDict" to the following:
### import sys
# if sys.version_info[:2] >= (3, 8):
#     from collections.abc import MutableMapping
# else:
#     from collections import MutableMapping
# from collections import OrderedDict

### CODE WRITTEN BY: Mingi Kang and Abhilasha Kumar (Bowdoin College)

from pymagnitude import * 
from pymagnitude import MagnitudeUtils 
import pandas as pd 
import pandas as pd

class embeddings:
    '''
        Description: 
            This class contains functions that create the embeddings.csv file from a list of words
            using PyMagnitude's vectors based on the word2vec model trained on Google News dataset.
            
        Functions: 
            (1) __init__: creates embeddings.csv file
            (2) collect_words: preprocesses the list of words.
            (3) word_checker: checks if word is in PyMagnitude's vectors. If not, gets the most similar word.
    
    '''
    def __init__(self, list_of_words): 
        self.words = embeddings.collect_words(list_of_words) 
        v = Magnitude(MagnitudeUtils.download_model('word2vec/medium/GoogleNews-vectors-negative300'))

        self.updated_words = [] 
        self.embeddings = [] 
        
        for word in self.words: 
            if word not in v: 
                vector = v.query(embeddings.word_checker(word))
                vector = vector.tolist() 
                self.embeddings += [vector] 
                self.updated_words += [embeddings.word_checker(word)]

            else: 
                vector = v.query(word) 
                vector = vector.tolist()
                self.embeddings += [vector]
                self.updated_words += [word]
                
        self.dict = {} 
        i = 0
        while i < len(self.updated_words): 
            self.dict[self.updated_words[i]] = self.embeddings[i]
            i += 1
            
        self.df = pd.DataFrame(self.dict)
        self.to_csv('data/lexical data/embeddings.csv', index=False)
    
    def collect_words(list_of_words):
        '''
            Description: 
                Preprocesses the list of words. The words are turned into lowercase, add an underscore for spaces, 
                removes all unnecessary characters. The column of excel file is named " spellcheck". To get words from a different 
                column, change "spellcheck" to column name". Used fovacs_animals.xlsx files to gather words. 
            
            Args: 
                (1) filename: name of excel file (.xlsx) 
            
            Returns: 
                List of adjusted words from the excel file. 
        '''
            
        
        words = [x.lower() for x in list_of_words]
        words = [x.strip() for x in words]
        words = [x.replace(" ", "_") for x in words]
        words = [x.replace("[", "") for x in words]
        words = [x.replace("-", "_") for x in words]
        words = [x.replace("//", "") for x in words]
        words = [x.replace(".", "") for x in words]
        words = [x.replace("\\", '') for x in words]
        return words 


    def word_checker(x): 
        '''
            Description: 
                Takes word (x) and if the word is not in PyMagnitude's vectors, then gets the most similar word from 
                a list of potential replacement words by PyMagnitude's most_similar_to_given function. 
            Args: 
                (1) x (str): word to check if it is in vectors. 
            Returns: 
                (1) replacement (str): the replacement word for the original word 
        ''' 
        
        v = Magnitude(MagnitudeUtils.download_model('word2vec/medium/GoogleNews-vectors-negative300'))
        original_word = []
        original_word.append(x) 
        z = x.replace("_", "") 
        y = [] 
        if "_" in x: 
            if z in v:
                return z
            else: 
                x = x.replace("_", " ")
                x = x.split()
                for words in x: 
                    if words in v: 
                        y.append(words) 
        else: 
            idx = 0
            while idx < len(x): 
                i = idx +1
                while i < len(x):
                    if x[idx:i+2] in v: 
                        y.append(x[idx:i+2])
                        i +=1 
                    else: 
                        i += 1
                idx += 1 
        replacement = v.most_similar_to_give(str(original_word), y) 
        return replacement 
            
#### SAMPLE RUN CODE ####
# a = embeddings(['apple', 'mango']) 
# print(a.df)
