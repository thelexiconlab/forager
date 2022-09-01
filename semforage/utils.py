import numpy as np
from scipy import stats
import statistics
import pandas as pd

import difflib
import nltk

def prepareData(path,delimiter = '\t'):
    ### LOAD BEHAVIORAL DATA ###
    df = pd.read_csv(path, header=None, names=['SID', 'entry'], delimiter=delimiter)

    # load similarity labels
    labels = pd.read_csv("../data/similarity_labels.csv")

    # get values from df 
    values = df['entry'].values

    # loop through values to find which ones are not in file
    oov = [w for w in values if w not in labels['word'].values]

    if len(oov) > 0:
        print("There are " + str(len(oov)) + " items from your data that are out of the vocabulary set (OOV). The default policy is to replace any OOV item with the closest available word if the Levenshtein edit-distance is 2 or lower. Otherwise, the fluency list is truncated before the OOV item.")
        while True:
            choice = input("Type 'd' to use the default policy. \n Type 'r' to review each OOV item and choose between replacing the word and truncating the list. \n Then, press enter.")
            if choice == "d":
                replacements = {}
                for word in set(oov):
                    # get closest match in vocab and check edit distance
                    closest_word = difflib.get_close_matches(word, labels, 1)
                    if nltk.edit_distance(word, closest_word[0]) <= 2:
                        replacements[word] = closest_word[0]
                    else: 
                        # truncate fluency list before instance of OOV item
                        indices = df[df['entry'] == word].index.values # indices of rows of OOV item 
                        for i in indices:
                            sid = df['SID'].iloc[i]
                            sid_rows = df[df['SID'] == sid].index.values
                            j = sid_rows[-1] + 1
                            df.drop(df.index[i:j], inplace=True)
                # replace words within edit distance threshold
                df.replace(replacements, inplace=True)
                break
            elif choice == "r":
                for x in range(len(oov)):
                    # offer user top 3 matches and option to truncate
                    closest_words = difflib.get_close_matches(oov[x], labels, 3)
                    print("OOV item #"+str(x)+": "+oov[x])
                    print("The top three closest matches are:")
                    print("(1)", closest_words[0])
                    print("(2)", closest_words[1])
                    print("(3)", closest_words[2])
                    while True:
                        c = input("To replace this item with one of the words above, please enter '1', '2', or '3' accordingly. \n Enter 't' to truncate this participant's list before the item.")
                        if c == "1" or c == "2" or c == "3":
                            y = int(c) - 1
                            df.replace(oov[x], closest_words[y], inplace=True)
                            break
                        elif c == "t":
                            i = df[df['entry'] == word].index.values[0] # index of row of OOV item (repeats will be deleted as it goes)
                            sid = df['SID'].iloc[i]
                            sid_rows = df[df['SID'] == sid].index.values
                            j = sid_rows[-1] + 1
                            df.drop(df.index[i:j], inplace=True)
                            break
                        else:
                            print("Entry invalid. Try again.")
                            continue
                break
            else:
                print("Entry invalid. Try again.") 
                continue 
    
    print("Data preparation complete.")
    return df 
