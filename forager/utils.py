import numpy as np
from scipy import stats
import pandas as pd

import difflib
import nltk

def trunc(word, df):
    # function to truncate fluency list at word
    i = df[df['entry'] == word].index.values[0]
    sid = df.iloc[i]['SID']
    sid_rows = df[df['SID'] == sid].index.values
    j = sid_rows[-1] + 1
    df.drop(df.index[i:j], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return None

def prepareData(path,delimiter = '\t'):
    ### LOAD BEHAVIORAL DATA ###
    df = pd.read_csv(path, header=None, names=['SID', 'entry'], delimiter=delimiter)
    # load labels
    labels = pd.read_csv("data/lexical_data/frequencies.csv", names=['word', 'logct', 'ct']) 

    # get values from df 
    values = df['entry'].values
    
    # loop through values to find which ones are not in file
    oov = [w for w in values if w not in labels['word'].values]
    if len(oov) > 0:
        print("There are " + str(len(oov)) + " items from your data that are out of the vocabulary set (OOV). The default policy is to replace any OOV item with the closest available word if the Levenshtein edit-distance is 2 or lower. Otherwise, the fluency list is truncated before the OOV item.")
        rct = 0 # number of replacements
        tct = 0 # number of truncations
        while True:
            choice = input("Type 'd' to use the default policy. \nType 'r' to review each OOV item and choose between replacing the word and truncating the list. \nThen, press enter. \n")
            if choice == "d":
                replacements = {}
                for word in set(oov):
                    # get closest match in vocab and check edit distance
                    closest_word = difflib.get_close_matches(word, labels['word'].values,1)

                    if len(closest_word)>0 and nltk.edit_distance(word, closest_word[0]) <= 2:
                        replacements[word] = closest_word[0]
                    else: 
                        # truncate fluency list before instance of OOV item
                        while word in df.values:
                            tct += 1
                            trunc(word, df)
                # replace words within edit distance threshold
                rct += len(replacements)
                df.replace(replacements, inplace=True)
                break
            elif choice == "r":
                for x in range(len(oov)):
                    # offer user top 3 matches and option to truncate
                    closest_words = difflib.get_close_matches(oov[x], labels['word'].values, 3)
                    print("OOV item #"+str(x)+": "+oov[x])
                    if len(closest_words)==0:
                        print("No close matches found. Truncating list before OOV item.")
                        tct += 1
                        trunc(oov[x], df)
                        continue
                    # Sometimes theres fewer than 3 closest words
                    print("The top (three) closest matches are:")
                    for i,word in enumerate(closest_words):
                        print("({i}) {word}".format(i=i+1,word=word))
                    while True:
                        c = input("To replace this item with one of the words above, please enter '1', '2', or '3' accordingly. \nEnter 't' to truncate this participant's list before the item.")
                        if c == "1" or c == "2" or c == "3":
                            y = int(c) - 1
                            if y > len(closest_words):
                                raise Exception("Out-of-vocabulary word may have fewer than three close matches, please choose the closest valid replacement item")
                            rct += 1
                            df.replace(oov[x], closest_words[y], inplace=True)
                            break
                        elif c == "t":
                            tct += 1
                            trunc(oov[x], df)
                            break
                        else:
                            print("Entry invalid. Try again.")
                            continue
                break
            else:
                print("Entry invalid. Try again.") 
                continue 
        print("There were", rct, "replacements and", tct, "truncations.")
    print("Data preparation complete.")
    
    # Stratify data into fluency lists
    data = []
    for subj in df['SID'].unique():
        subj_df = df[df['SID'] == subj]
        subj_data = (subj,subj_df['entry'].values.tolist())
        data.append(subj_data)
    return data
