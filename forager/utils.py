import numpy as np
from scipy import stats
import pandas as pd
import os
import difflib
import nltk
import zipfile

def trunc(word, df):
    # function to truncate fluency list at word

    i = df[df['entry'] == word].index.values[0]
    sid = df.iloc[i]['SID']
    if 'timepoint' in df.columns:
        tp = df.iloc[i]['timepoint']
        list_rows = df[(df['SID'] == sid) & (df['timepoint'] == tp)].index.values
    else:
        list_rows = df[df['SID'] == sid].index.values
    j = list_rows[-1] + 1
    df.drop(df.index[i:j], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return None

def exclude(word,df):
    # function to exclude all instances of word from df
    df.drop(df[df['entry'] == word].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return None

# takes in a path to a data file to be read as a CSV, the first row will be assumed as a header 
# accepted delimiters include commas, tabs, semicolons, pipes, and spaces
def prepareData(path, fp):
    ### LOAD BEHAVIORAL DATA ###
    df = pd.read_csv(path, header=0, engine='python', sep=None, encoding='utf-8-sig')
    if len(df.columns) > 2:
        three_col = input("Use the third column of this data file as a time point? Type 'y' for yes or 'n' for no: ")
        if three_col == "y":
            df = df.iloc[:, :4]
            df.columns = ['SID', 'entry', 'timepoint']
        else:
            df = df.iloc[:, :3]
            df.columns = ['SID', 'entry']
    else:
        df.columns = ['SID', 'entry']

    # load labels
    labels = pd.read_csv(os.path.join(fp,"data/lexical_data/USE_frequencies.csv"), names=['word', 'logct', 'ct']) 

    # set all replacements to actual word for all words in labels as the default
    replacements = {word: word for word in labels['word'].values}

    # get values from df 
    values = df['entry'].values
    
    # loop through values to find which ones are not in file
    oov = [w for w in values if w not in labels['word'].values]
    if len(oov) > 0:
        replacement_df = df.copy()
        print("We did not find exact matches for " + str(len(oov)) + " items in our vocabulary. Any items for which we find a reasonable match will be automatically replaced. For all other OOV items, you may:")
        while True:
            choice = input("type 'e' to exclude these words from the fluency lists but continue with the rest of the list, \ntype 't' to truncate each fluency list at the first occurrence of such a word, \nor type 'r' to assign a mean semantic/phonological vector and random frequency to any such word and continue with the rest of the list. \nThen, press enter. \n")
            choices = ['e', 't', 'r']
            if choice not in choices:
                print("Entry invalid. Try again.") 
                continue

            for word in set(oov):
                # get closest match in vocab and check edit distance
                closest_word = difflib.get_close_matches(word, labels['word'].values,1)

                if len(closest_word)>0 and nltk.edit_distance(word, closest_word[0]) <= 2:
                    replacements[word] = closest_word[0]
                elif choice == "e":
                    # exclude this word from the list
                    exclude(word, df)
                    replacements[word] = "EXCLUDE"

                elif choice == "r":
                    # change all occurrences of word to "UNK"
                    replacements[word] = "UNK"
                else: 
                    # truncate fluency list before instance of OOV item
                    while word in df.values:
                        trunc(word, df)
                    replacements[word] = "TRUNCATE"
            break        
        df.replace(replacements, inplace=True)
        
        # add an extra column to orig_df with the replacement word

        replacement_df['evaluation'] = replacement_df['entry'].map(replacements)
        # create a new column 'replacement' that is a copy of 'evaluation'
        replacement_df['replacement'] = replacement_df['evaluation']
        # now replace all instances in evalution where the entry doesn't match the replacement AND isn't within
        # ['UNK', 'EXCLUDE', 'TRUNCATE'] with 'REPLACE'
        replacement_df.loc[(replacement_df['entry'] != replacement_df['evaluation']) & (~replacement_df['evaluation'].isin(['UNK', 'EXCLUDE', 'TRUNCATE'])), 'evaluation'] = 'REPLACE'
        # also for the column 'evaluation', if entry matches evaluation, replace with 'found'
        replacement_df.loc[(replacement_df['entry'] == replacement_df['evaluation']), 'evaluation'] = 'FOUND'
        
        exclude_count = (replacement_df["evaluation"] == "EXCLUDE").sum()
        unk_count = (replacement_df["evaluation"] == "UNK").sum()
        trunc_count = (replacement_df["evaluation"] == "TRUNCATE").sum()
        replacement_count = (replacement_df["evaluation"] == "REPLACE").sum()
        print("We have found reasonable replacements for " + str(replacement_count)+ " item(s) in your data. \n\n")
        if exclude_count>0:
            print(str(exclude_count) + " items were excluded across all lists.\n")
        elif unk_count>0: 
            print(str(unk_count) + " items were assigned a random vector across all lists.\n")
        elif trunc_count>0:
            print("Lists were truncated at " + str(trunc_count) + " items across all lists.\n")
        
        # Stratify data into fluency lists
        data = []
        if 'timepoint' in df.columns:
            lists = df.groupby(["SID", "timepoint"])
        else: 
            lists = df.groupby("SID")
        
        for sub, frame in lists:
            list = frame["entry"].values.tolist()
            subj_data = (sub, list)
            data.append(subj_data)    
        return data, replacement_df, df
    
    else:
        print("Success! We have found exact matches for all items in your data. \n\n")
        replacement_df = df.copy()
        replacement_df['evaluation'] = "FOUND"
        # Add the column corresponding to the replacement column , set it all to the same value   
        data = []
        if 'timepoint' in df.columns:
            lists = df.groupby(["SID", "timepoint"])
        else: 
            lists = df.groupby("SID")
        
        for sub, frame in lists:
            list = frame["entry"].values.tolist()
            subj_data = (sub, list)
            data.append(subj_data)
        
        return data, replacement_df, df

