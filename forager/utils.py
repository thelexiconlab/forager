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

# takes in a path to a data file to be read as a CSV or TXT, the first row will be assumed as a header
# accepted delimiters include commas, tabs, semicolons, pipes, and spaces
# columns are detected by header name (case-insensitive): SID, entry, timepoint (optional), time (optional)
def prepareData(path, domain, time_type='cumulative', time_units='s'):
    ### LOAD BEHAVIORAL DATA ###
    df = pd.read_csv(path, header=0, engine='python', sep=None, encoding='utf-8-sig')
    col_map = _detect_columns(df)
    # Rename detected columns to canonical names
    df = df.rename(columns={v: k for k, v in col_map.items()})
    col_map = {k: k for k in col_map}

    detected = list(col_map.keys())
    print(f"Detected columns: {detected}")
    if 'timepoint' in col_map:
        print("  -> Will group lists by 'timepoint'.")
    if 'time' in col_map:
        print(f"  -> Timing data detected (time_type='{time_type}', time_units='{time_units}').")

    # load labels
    labels = pd.read_csv('data/lexical_data/' + domain + '/USE_frequencies.csv', names=['word', 'logct'], header=None) 

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
        data = _build_data_list(df, col_map, time_type, time_units)
        return data, replacement_df, df

    else:
        print("Success! We have found exact matches for all items in your data. \n\n")
        replacement_df = df.copy()
        replacement_df['evaluation'] = "FOUND"
        data = _build_data_list(df, col_map, time_type, time_units)
        return data, replacement_df, df

def prepareData_colab(path, time_type='cumulative', time_units='s'):
    ### LOAD BEHAVIORAL DATA ###
    df = pd.read_csv(path, header=0, engine='python', sep=None, encoding='utf-8-sig')
    col_map = _detect_columns(df)
    # Rename detected columns to canonical names
    df = df.rename(columns={v: k for k, v in col_map.items()})
    col_map = {k: k for k in col_map}

    detected = list(col_map.keys())
    print(f"Detected columns: {detected}")
    if 'timepoint' in col_map:
        print("  -> Will group lists by 'timepoint'.")
    if 'time' in col_map:
        print(f"  -> Timing data detected (time_type='{time_type}', time_units='{time_units}').")

    # load labels
    labels = pd.read_csv("data/lexical_data/USE_frequencies.csv", names=['word', 'logct', 'ct']) 

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
        data = _build_data_list(df, col_map, time_type, time_units)
        return data, replacement_df, df

    else:
        print("Success! We have found exact matches for all items in your data. \n\n")
        replacement_df = df.copy()
        replacement_df['evaluation'] = "FOUND"
        data = _build_data_list(df, col_map, time_type, time_units)
        return data, replacement_df, df


_ALIASES = {
    'SID': ['sid', 'id', 'subject', 'participant'],
    'entry': ['entry', 'item', 'word', 'response'],
    'timepoint': ['timepoint'],
    'time': ['time', 'rt', 'response_time'],
}

def _detect_columns(df):
    '''
        Map DataFrame column names to canonical roles using case-insensitive matching.
        Works with any delimited file format (CSV, TXT, TSV, etc.) as long as a header
        row is present. Supports common aliases for each column role.

        Args:
            df (DataFrame): Raw DataFrame from pd.read_csv.

        Returns:
            dict: Maps canonical names ('SID', 'entry', and optionally 'timepoint', 'time')
                to the actual column name strings found in df.

        Raises:
            ValueError: If required columns 'SID' or 'entry' are not found.
    '''
    lowered = {col.strip().lower(): col for col in df.columns}
    found = {}
    for canonical, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                found[canonical] = lowered[alias]
                break

    missing = [c for c in ('SID', 'entry') if c not in found]
    if missing:
        raise ValueError(
            f"Required column(s) not found in data file: {missing}. "
            f"Columns present: {list(df.columns)}")
    return found


def _build_data_list(df, col_map, time_type='cumulative', time_units='s'):
    '''
        Build the data list from a processed DataFrame.

        Args:
            df (DataFrame): Processed DataFrame with canonical column names.
            col_map (dict): Mapping from canonical names to column names in df.
            time_type (str): Passed to prepare_times if 'time' column is present.
            time_units (str): Passed to prepare_times if 'time' column is present.

        Returns:
            list: Tuples of (subj, word_list) or (subj, word_list, time_list).
    '''
    has_timepoint = 'timepoint' in col_map
    has_time = 'time' in col_map

    if has_timepoint:
        groups = df.groupby([col_map['SID'], col_map['timepoint']])
    else:
        groups = df.groupby(col_map['SID'])

    data = []
    for sub, frame in groups:
        word_list = frame[col_map['entry']].values.tolist()
        if has_time:
            raw_times = frame[col_map['time']].values.tolist()
            time_list = prepare_times(raw_times, time_type, time_units).tolist()
            data.append((sub, word_list, time_list))
        else:
            data.append((sub, word_list))
    return data


def prepare_times(times, time_type="cumulative", time_units="s"):
    '''
        Convert timing data to cumulative seconds.

        Args:
            times (list or array): Response times for each word.
            time_type (str): "cumulative" or "irt" (inter-response time). Default: "cumulative".
            time_units (str): "s" (seconds) or "ms" (milliseconds). Default: "s".

        Returns:
            np.ndarray: Cumulative times in seconds.

        Raises:
            ValueError: If time_type or time_units are invalid.
    '''
    if time_type not in ("cumulative", "irt"):
        raise ValueError(f"time_type must be 'cumulative' or 'irt', got '{time_type}'")
    if time_units not in ("s", "ms"):
        raise ValueError(f"time_units must be 's' or 'ms', got '{time_units}'")

    times = np.array(times, dtype=float)

    if time_units == "ms":
        times = times / 1000.0

    if time_type == "irt":
        times = np.cumsum(times)

    return times
