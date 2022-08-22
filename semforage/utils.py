import numpy as np
from scipy import stats
import statistics
import pandas as pd

import difflib

def prepareData(path,delimiter = '\t', oov_choice = "truncate"):
    ### LOAD BEHAVIORAL DATA ###
    df = pd.read_csv(path, header=None, names=['SID', 'entry'], delimiter=delimiter)

    # load similarity labels
    labels = pd.read_csv("../data/similarity_labels.csv")

    # get values from df 
    values = df['entry'].values

    # loop through values to find which ones are not in file
    for word in values:
        if word not in labels['word'].values:
            # find closest word in vocabulary to this
            closest_words = difflib.get_close_matches(word, labels)
            print("word not found:", word, "closest words:", closest_words)
            if oov_choice == "truncate":
                print("truncating list at first oov word")
            elif oov_choice == "replace":
                print("replacing oov word with closest word")