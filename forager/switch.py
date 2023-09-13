import numpy as np
from scipy import stats
import statistics
import difflib
import pandas as pd

'''
Methods for calculating switches in Semantic Foraging methods.
    Current Methods:
        (1) Similarity Drop (simdrop): Switch Heuristic used in Hills TT, Jones MN, Todd (2012), where a switch is 
            predicted within a series of items A,B,C,D after B if S(A,B) > S(B,C) and S(B,C) < S(C,D)

        (2) Norms_Associative: Switch Method based on norms developed in Troyer, AK, Moscovitch, M, & Winocur, G (1997) 
            and extended by Zemla & Austerweil (2018) via SNAFU.
            Switches are predicted when current item does not share categories with preceding item.
            
        (3) Norms_Categorical: Switch Method based on norms developed in Troyer, AK, Moscovitch, M, & Winocur, G (1997) 
            and extended by Zemla & Austerweil (2018) via SNAFU.
            Switches are predicted when current item does not share categories with all items in preceding cluster.
            See Hills et al. (2015) for more details.

        (4) Multimodal Simdrop: An extension of the Similarity Drop Method to include phonological similarity in the heuristic

        (5) Delta Similarity: A method for predicting switches proposed by Nancy Lundin in her dissertation to bypass limitations
            of simdrop model, and allow for consecutive switches, and accounts for small dips in similarity that simdrop may
            deem a switch, which may actually be due to "noise" 

    Output Format: 
        Each switch method should preserve the same length/general format for returing switch values, 
        which can then be used in the foraging models, passed as a parameter

        length: each switch list should be the same length as the fluency_list
        coding: 
            0 - no switch
            1 - switch predicted by method
            2 - boundary case
    
'''


def switch_simdrop(fluency_list, semantic_similarity):
    '''
        Similarity Drop Switch Method from Hills TT, Jones MN, Todd (2012).
        
        Args:
            fluency_list (list, size = L): fluency list to predict switches on
            semantic_similarity (list, size = L): a list of semantic similarities between items in the fluency list, obtained via create_history_variables

        Returns:
            a list, size L, of switches, where 0 = no switch, 1 = switch, 2 = boundary case
    '''
    simdrop = []
    for k in range(len(fluency_list)):
        if (k > 0 and k < (len(fluency_list)-1)): 
            # simdrop
            if (semantic_similarity[k+1] > semantic_similarity[k]) and (semantic_similarity[k-1] > semantic_similarity[k]):
                simdrop.append(1)
            
            else:
                simdrop.append(0)

        else:
            simdrop.append(2)

    return simdrop

def switch_norms_associative(fluency_list,norms):
    '''
        Switch Method Based on Troyer Norms from Troyer, A. K., Moscovitch, M., & Winocur, G. (1997).

        Args:
            fluency_list (list, size = L): fluency list to predict switches on
            norms (dataframe, size = L x 2): dataframe of norms data matching animals to a categorical classification

        Returns:
            troyer (list, size = L): a list of switches, where 0 = no switch, 1 = switch, 2 = boundary case
    '''
    norm_designation = []

    for k in range(len(fluency_list)):
        if k > 0:
            item1 = fluency_list[k]
            item2 = fluency_list[k-1]
            items_in_norms = norms['Item'].values.tolist()
            # find closest match to item1 and item2 in norms
            # often, this will be an exact match, but if not, we want to find the closest match
            # so that we can assign the category of the closest match to item1 and item2
            # e.g., grapes -> grape
            # if difflib returns an empty list, then there is no close match, and we just use the original item
            
            item1 = difflib.get_close_matches(item1, items_in_norms, n=1)[0] if len(difflib.get_close_matches(item1, items_in_norms, n = 1)) > 0 else item1
            item2 = difflib.get_close_matches(item2, items_in_norms, n=1)[0] if len(difflib.get_close_matches(item2, items_in_norms, n = 1)) > 0 else item2

            

            category1 = norms[norms['Item'] == item1]['Category'].values.tolist()
            category2 = norms[norms['Item'] == item2]['Category'].values.tolist()

            if len(list(set(category1) & set(category2)))== 0:
                norm_designation.append(1)
            else:
                norm_designation.append(0)

        else:
            norm_designation.append(2)
    
    return norm_designation

def switch_norms_categorical(fluency_list,norms):
    '''
    Hills et al. 2015 categorical switches where a switch is predicted if the category of the current word is different than the shared category of the previous two words

    Args:
        fluency_list (list, size = L): fluency list to predict switches on
        norms (dataframe, size = L x 2): dataframe of norms data matching animals to a categorical classification
    Returns:
        categorical (list, size = L): a list of switches, where 0 = no switch, 1 = switch, 2 = boundary case
    '''
    df = pd.DataFrame({'item': fluency_list, 'designation': [-1] * len(fluency_list)})

    items_in_norms = norms['Item'].values.tolist()

    def find_most_recent_one_index(lst, index):
        for i in range(index - 1, -1, -1):
            if lst[i] == 1 or lst[i] == 2:
                return i
        return index # if no 1s or 2s are found, return the current index

    for i, row in df.iterrows():
        if i == 0:
            # first word
            df.at[i, 'designation'] = 2
        elif i == 1:
            # find category of previous word
            prev_word = df.loc[i - 1, 'item']
            closest_match = difflib.get_close_matches(prev_word, items_in_norms, n=1)
            prev_word = closest_match[0] if len(closest_match) > 0 else prev_word
            
            prev_word_cats = norms[norms['Item'] == prev_word]['Category'].tolist() if prev_word in items_in_norms else 'notinnorms'            
            # find category of current word
            current_word = row['item']
            closest_match = difflib.get_close_matches(current_word, items_in_norms, n=1)
            current_word = closest_match[0] if len(closest_match) > 0 else current_word
            current_word_cats = norms[norms['Item'] == current_word]['Category'].tolist() if current_word in items_in_norms else 'notinnorms'            
            # check if they share a category
            if any(cat in current_word_cats for cat in prev_word_cats):
                df.at[i, 'designation'] = 0
            else:
                df.at[i, 'designation'] = 1
        else:
            
            prev_one = find_most_recent_one_index(df['designation'], i)
            cluster = df.loc[prev_one:i, :] if prev_one != i else df.loc[i, :]
            prev_words = norms[norms['Item'].isin(cluster['item'])]
            prev_cats = prev_words.groupby('Item', group_keys=False)['Category'].apply(list).to_dict()
            all_shared_cats = set.intersection(*[set(cats) for cats in prev_cats.values()]) if len(prev_cats) > 0 else set()
            
            # find category of current word
            current_word = row['item']
            closest_match = difflib.get_close_matches(current_word, items_in_norms, n=1)
            current_word = closest_match[0] if len(closest_match) > 0 else current_word
            current_word_cats = norms[norms['Item'] == current_word]['Category'].tolist() if current_word in items_in_norms else 'notinnorms'
            
            
            if any(cat in current_word_cats for cat in all_shared_cats):
                df.at[i, 'designation'] = 0
                
            else:
                df.at[i, 'designation'] = 1
                
    
    # return the designations
    categorical_designations = df['designation'].values.tolist()
    return categorical_designations

def switch_multimodal(fluency_list,semantic_similarity,phonological_similarity,alpha):
    '''
        Multimodal Similarity Drop based on semantic and phonological cues, extending Hills TT, Jones MN, Todd (2012).
        Args:
            fluency_list (list, size = L): fluency list to predict switches on
            semantic_similarity (list, size = L): a list of semantic similarities between items in the fluency list, obtained via create_history_variables
            phonological_similarity (list, size = L): a list of phonological similarities between items in the fluency list obtained via create_history_variables
            alpha (float): alpha parameter that dictates the weight of semantic vs. phonological cue, between 0 and 1
        Returns:
            multimodalsimdrop (list, size = L): a list of switches, where 0 = no switch, 1 = switch, 2 = boundary case
        
        Raises:
            Exception: if alpha is not between 0 and 1
    '''    
    #Check if alpha is between 0 and 1
    
    if alpha > 1 or alpha < 0:
        raise Exception("Alpha parameter must be within range [0,1]")
    simphon = alpha * np.array(semantic_similarity) + (1 - alpha) * np.array(phonological_similarity)
    multimodalsimdrop = []

    for k in range(len(fluency_list)):
        if (k > 0 and k < (len(fluency_list) - 1)): 
            if (simphon[k + 1] > simphon[k]) and (simphon[k - 1] > simphon[k]):
                multimodalsimdrop.append(1)
            else:
                multimodalsimdrop.append(0)

        else:
            multimodalsimdrop.append(2)

    return multimodalsimdrop

def switch_delta(fluency_list, semantic_similarity, rise_thresh, fall_thresh):
    '''
        Delta Similarity Switch Method proposed by Nancy Lundin & Peter Todd. 
        
        Args:
            fluency_list (list, size = L): fluency list to predict switches on
            semantic_similarity (list, size = L): a list of semantic similarities between items in the fluency list, obtained via create_history_variables
            rise_thresh (float): after a switch occurs, the threshold that the increase in z-scored similarity must exceed to be a cluster  
            fall_thresh (float): while in a cluster, the threshold that the decrease in z-scored similarity must exceed to be a switch

        Returns:
            a list, size L, of switches, where 0 = no switch, 1 = switch, 2 = boundary case
    '''
    if rise_thresh > 1 or rise_thresh < 0:
        raise Exception("Rise Threshold parameter must be within range [0,1]")

    if fall_thresh > 1 or fall_thresh < 0:
        raise Exception("Fall Threshold parameter must be within range [0,1]")

    switchVector = [2] # first item designated with 2

    # obtain consecutive semantic similarities b/w responses
    # z-score similarities within participant
    similaritiesZ = stats.zscore(semantic_similarity[1:])
    medianSim = statistics.median(similaritiesZ)
    meanSim = 0
    similaritiesZ = np.concatenate(([np.nan], similaritiesZ))

    # define subject level threshold = median (zscored similarities)
    # firstSwitchSimThreshold = meanSim
    firstSwitchSimThreshold = medianSim
    # for second item, if similarity < median, then switch, else cluster
    if similaritiesZ[1] < firstSwitchSimThreshold:
        switchVector.append(1)
    else:
        switchVector.append(0)

    currentState = switchVector[1]
    previousState = currentState

    # for all other items:
    for n in range(1,len(fluency_list)-1):
    #   consider n-1, n, n+1 items
        
        simPrecedingToCurrentWord = similaritiesZ[n]
        
        simCurrentToNextWord = similaritiesZ[n+1]
        if previousState == 0: #if previous state was a cluster
            if fall_thresh < (simPrecedingToCurrentWord - simCurrentToNextWord): # similarity diff fell more than threshold
                currentState = 1 # switch
            else:
                currentState = 0 # cluster
        else: # previous state was a switch
            if rise_thresh < (simCurrentToNextWord - simPrecedingToCurrentWord): # similarity diff is greater than our rise threshold
                currentState = 0 # cluster
            else:
                currentState = 1 # switch

        switchVector.append(currentState)
        previousState = currentState

    return switchVector

### SAMPLE RUN CODE ###
# normspath =  '../data/norms/animals_snafu_scheme_vocab.csv'
# norms = pd.read_csv(normspath, encoding="unicode-escape")
# fluency_list = pd.read_csv('../data/fluency_lists/51.txt',header=0, engine='python', sep=None, encoding='utf-8-sig')
# print(fluency_list.head())
# fluency_list = fluency_list['entry'].values.tolist()

# switch_norms_associative(fluency_list, norms)