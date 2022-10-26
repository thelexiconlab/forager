import pytest
import pandas as pd
import numpy as np
# from scipy.optimize import fmin
from forager.foraging import *
from forager.cues import create_history_variables
from forager.switch import switch_simdrop

#Import Default Data from Psych Review

parentfolder = '../../data/'
sim_matrix = pd.read_csv(parentfolder + "lexical_data/archive/psyrev/psyrev-similaritymatrix.csv",header=None).dropna(axis=1).values
labels = list(pd.read_csv(parentfolder + "lexical_data/archive/psyrev/psyrev-similaritylabels.csv",header=None).squeeze().values)
freq_data = pd.read_csv(parentfolder + "lexical_data/archive/psyrev/psyrev-frequencies.csv",header=None).dropna(axis=1)
psyrev_data = pd.read_csv(parentfolder + "fluency_lists/data-psyrev.txt",sep='\t',header=None)

print(psyrev_data)
#reorder freq_matrix to match sim_matrix
freq_matrix = []
for label in labels:
    freq_matrix.append(freq_data[freq_data[0] == label].squeeze().to_dict())
freq_matrix = np.array(pd.DataFrame(freq_matrix)[1])
sim_matrix[sim_matrix <= 0] = .0001

# Import Example Data
subj = 51
ex_list = psyrev_data[psyrev_data[0] == subj][1].values.tolist()

#Create History Variables for fitting Model
history_vars = create_history_variables(ex_list, labels, sim_matrix,freq_matrix)

# Get comparison value for untuned model
results_df = pd.read_csv(parentfolder + "misc/fullfits_nophon.csv")
comparison_nll_static = results_df[results_df['sid'] == subj]['merr-SR'][0]
comparison_nll_dynamic = results_df[results_df['sid'] == subj]['merr-DR'][0]


def test_model_static():
    nll = forage.model_static(beta = [0,0], freql = history_vars[2], freqh = history_vars[3], siml = history_vars[0], simh = history_vars[1])
    assert nll == comparison_nll_static

def test_model_dynamic():
    switches = switch_simdrop(ex_list,history_vars[0])
    nll = forage.model_dynamic(beta = [0,0], freql = history_vars[2], freqh = history_vars[3], siml = history_vars[0], simh = history_vars[1], switchvals = switches)
    assert nll == comparison_nll_dynamic
