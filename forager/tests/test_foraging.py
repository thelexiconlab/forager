import os
import pytest
import pandas as pd
import numpy as np
from forager.foraging import forage
from forager.cues import create_history_variables
from forager.switch import switch_simdrop

# Load current lexical data (animals domain)
parentfolder = os.path.join(os.path.dirname(__file__), "../../data/")
lexical_folder = parentfolder + "lexical_data/animals/"

sim_matrix = np.loadtxt(lexical_folder + "USE_semantic_matrix.csv", delimiter=',')
phon_matrix = np.loadtxt(lexical_folder + "USE_phonological_matrix.csv", delimiter=',')
freq_data = pd.read_csv(lexical_folder + "USE_frequencies.csv", header=None)
labels = freq_data[0].values.tolist()
freq_matrix = np.array(freq_data[1])

# Example fluency list from animals_sample.txt (subject 51)
ex_list = ['dog', 'cat', 'bird', 'fish', 'horse', 'cow', 'pig', 'goat',
           'sheep', 'chicken', 'duck', 'turkey', 'deer', 'bear', 'wolf',
           'fox', 'rabbit', 'mouse', 'rat', 'snake', 'frog', 'turtle']

# Create history variables
history_vars = create_history_variables(ex_list, labels, sim_matrix, freq_matrix, phon_matrix)
switches = switch_simdrop(ex_list, history_vars[0])


def test_model_static_untuned():
    """Untuned static model (beta=[0,0]) returns a finite positive NLL."""
    nll = forage.model_static([0, 0], history_vars[2], history_vars[3],
                               history_vars[0], history_vars[1])
    assert np.isfinite(nll)
    assert nll > 0


def test_model_static_deterministic():
    """Static model produces the same NLL for the same inputs."""
    nll1 = forage.model_static([0.5, 0.5], history_vars[2], history_vars[3],
                                history_vars[0], history_vars[1])
    nll2 = forage.model_static([0.5, 0.5], history_vars[2], history_vars[3],
                                history_vars[0], history_vars[1])
    assert nll1 == nll2


def test_model_dynamic_untuned():
    """Untuned dynamic model (beta=[0,0]) returns a finite positive NLL."""
    nll = forage.model_dynamic([0, 0], history_vars[2], history_vars[3],
                                history_vars[0], history_vars[1], switches)
    assert np.isfinite(nll)
    assert nll > 0


def test_model_dynamic_deterministic():
    """Dynamic model produces the same NLL for the same inputs."""
    nll1 = forage.model_dynamic([0.5, 0.5], history_vars[2], history_vars[3],
                                 history_vars[0], history_vars[1], switches)
    nll2 = forage.model_dynamic([0.5, 0.5], history_vars[2], history_vars[3],
                                 history_vars[0], history_vars[1], switches)
    assert nll1 == nll2


def test_model_static_phon():
    """Phonological static model returns a finite positive NLL."""
    nll = forage.model_static_phon([0.5, 0.5, 0.5], history_vars[2], history_vars[3],
                                    history_vars[0], history_vars[1],
                                    history_vars[4], history_vars[5])
    assert np.isfinite(nll)
    assert nll > 0


def test_model_dynamic_phon_global():
    """Phonological dynamic model (global) returns a finite positive NLL."""
    nll = forage.model_dynamic_phon([0.5, 0.5, 0.5], history_vars[2], history_vars[3],
                                     history_vars[0], history_vars[1],
                                     history_vars[4], history_vars[5],
                                     switches, 'global')
    assert np.isfinite(nll)
    assert nll > 0


def test_model_dynamic_phon_local():
    """Phonological dynamic model (local) returns a finite positive NLL."""
    nll = forage.model_dynamic_phon([0.5, 0.5, 0.5], history_vars[2], history_vars[3],
                                     history_vars[0], history_vars[1],
                                     history_vars[4], history_vars[5],
                                     switches, 'local')
    assert np.isfinite(nll)
    assert nll > 0


def test_model_dynamic_phon_switch():
    """Phonological dynamic model (switch) returns a finite positive NLL."""
    nll = forage.model_dynamic_phon([0.5, 0.5, 0.5], history_vars[2], history_vars[3],
                                     history_vars[0], history_vars[1],
                                     history_vars[4], history_vars[5],
                                     switches, 'switch')
    assert np.isfinite(nll)
    assert nll > 0


def test_model_dynamic_phon_invalid_cue():
    """Phonological dynamic model raises on invalid phoncue."""
    with pytest.raises(Exception):
        forage.model_dynamic_phon([0.5, 0.5, 0.5], history_vars[2], history_vars[3],
                                   history_vars[0], history_vars[1],
                                   history_vars[4], history_vars[5],
                                   switches, 'invalid')


def test_static_vs_dynamic_untuned():
    """With beta=[0,0] the static model equals the dynamic model (no similarity contribution)."""
    nll_static = forage.model_static([0, 0], history_vars[2], history_vars[3],
                                      history_vars[0], history_vars[1])
    nll_dynamic = forage.model_dynamic([0, 0], history_vars[2], history_vars[3],
                                        history_vars[0], history_vars[1], switches)
    assert nll_static == nll_dynamic


def test_report_matches_optimized():
    """The _report variants should return the same total NLL as the optimized versions."""
    beta = [0.3, 0.7]
    nll_opt = forage.model_static(beta, history_vars[2], history_vars[3],
                                   history_vars[0], history_vars[1])
    nll_rep, _ = forage.model_static_report(beta, history_vars[2], history_vars[3],
                                             history_vars[0], history_vars[1])
    assert np.isclose(nll_opt, nll_rep, rtol=1e-10)

    nll_dyn_opt = forage.model_dynamic(beta, history_vars[2], history_vars[3],
                                        history_vars[0], history_vars[1], switches)
    nll_dyn_rep, _ = forage.model_dynamic_report(beta, history_vars[2], history_vars[3],
                                                  history_vars[0], history_vars[1], switches)
    assert np.isclose(nll_dyn_opt, nll_dyn_rep, rtol=1e-10)
