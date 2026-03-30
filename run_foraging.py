import argparse
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
from forager.foraging import forage
from forager.switch import *
from forager.cues import create_history_variables
from forager.utils import prepareData
import pandas as pd
import numpy as np
import os, sys
from tqdm import tqdm
import warnings
import zipfile

warnings.simplefilter('ignore')

"""
Workflow: 
1. Evaluate data
    a. "Prepare Data" 
        - takes path of fluency list ; 

2. Select use case (lexical, switches, models) via --pipeline
    a. Lexical: returns similarity & frequency values for each word in fluency list
    b. Switches: returns switch values for each word in fluency list + lexical values
    c. Models: returns model outputs for each word in fluency list + lexical values + switch values
"""
# Global Path Variables
fp = "/".join(sys.argv[0].split('/')[:-1])



# if sys.argv[0] != 'run_foraging.py': # check the reference path
#     normspath = os.path.join(fp,'data/norms/animals_snafu_scheme_vocab.csv')
#     similaritypath =  os.path.join(fp,'data/lexical_data/USE_semantic_matrix.csv')
#     frequencypath =  os.path.join(fp,'data/lexical_data/USE_frequencies.csv')
#     phonpath = os.path.join(fp,'data/lexical_data/USE_phonological_matrix.csv')
#     vocabpath = os.path.join(fp,'data/lexical_data/vocab.csv')
# else:
#     normspath = 'data/norms/animals_snafu_scheme_vocab.csv'
#     similaritypath =  'data/lexical_data/USE_semantic_matrix.csv'
#     frequencypath =  'data/lexical_data/USE_frequencies.csv'
#     phonpath = 'data/lexical_data/USE_phonological_matrix.csv'
#     vocabpath = 'data/lexical_data/vocab.csv'

# Global Variables
models = ['static','dynamic','pstatic','pdynamic','all']
switch_methods = ['simdrop','multimodal','norms_associative','norms_categorical', 'delta', 'multimodaldelta', 'slope_difference', 'pei', 'all']


#Methods
def retrieve_data(path, domain, time_type='cumulative', time_units='s'):
    """
    1. Verify that data path exists

    """
    if os.path.exists(path) == False:
        ex_str = "Provided path to data \"{path}\" does not exist. Please specify a proper path".format(path=path)
        raise Exception(ex_str)
    data = prepareData(path, domain, time_type=time_type, time_units=time_units)
    return data

def get_lexical_data(domain):

    animalnormspath =  'data/norms/animals_snafu_scheme_vocab.csv'
    foodnormspath =  'data/norms/foods_snafu_scheme_vocab.csv'
    similaritypath =  'data/lexical_data/' + domain + '/USE_semantic_matrix.csv'
    frequencypath =  'data/lexical_data/' + domain + '/USE_frequencies.csv'
    phonpath = 'data/lexical_data/' + domain + '/USE_phonological_matrix.csv'

    animalnorms = pd.read_csv(animalnormspath, encoding="unicode-escape")
    foodnorms = pd.read_csv(foodnormspath, encoding="unicode-escape")
    norms = [animalnorms, foodnorms]
    similarity_matrix = np.loadtxt(similaritypath,delimiter=',')
    frequency_list = np.array(pd.read_csv(frequencypath,header=None,encoding="unicode-escape")[1])
    phon_matrix = np.loadtxt(phonpath,delimiter=',')
    labels = pd.read_csv(frequencypath,header=None)[0].values.tolist()
    
    
    return norms, similarity_matrix, phon_matrix, frequency_list,labels

def calculate_model(model, history_vars, switch_names, switch_vecs):
    """
    1. Check if specified model is valid
    2. Return a set of model functions to pass

    Deduplicates identical switch vectors across all methods/parameters so that
    minimize() is only called once per unique binary sequence.
    """
    model_name = []
    model_results = []
    if model not in models:
        ex_str = "Specified model is invalid. Model must be one of the following: {models}".format(models=models)
        raise Exception(ex_str)

    # Cache for deduplicating identical switch vectors.
    # Keys: tuple(switch_vec); Values: dict of model_type -> result tuple
    _switch_cache = {}

    if model == models[0] or model == models[4]:
        r1 = np.random.rand()
        r2 = np.random.rand()

        result = minimize(forage.model_static, [r1,r2], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1]))
        beta_df = float(result.x[0]) # Optimized weight for frequency cue
        beta_ds = float(result.x[1]) # Optimized weight for similarity cue
        model_name.append('forage_static')
        model_results.append((beta_df, beta_ds, float(result.fun)))
    if model == models[1] or model == models[4]:
        for i, switch_vec in enumerate(switch_vecs):
            key = tuple(switch_vec)
            if key not in _switch_cache:
                _switch_cache[key] = {}
            if 'dynamic' not in _switch_cache[key]:
                r1 = np.random.rand()
                r2 = np.random.rand()
                result = minimize(forage.model_dynamic, [r1,r2], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1], switch_vec))
                beta_df = float(result.x[0])
                beta_ds = float(result.x[1])
                _switch_cache[key]['dynamic'] = (beta_df, beta_ds, float(result.fun))
            model_name.append('forage_dynamic_' + switch_names[i])
            model_results.append(_switch_cache[key]['dynamic'])
    if model == models[2] or model == models[4]:
        r1 = np.random.rand()
        r2 = np.random.rand()
        r3 = np.random.rand()
        result = minimize(forage.model_static_phon, [r1,r2,r3], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1], history_vars[4],history_vars[5]))
        beta_df = float(result.x[0]) # Optimized weight for frequency cue
        beta_ds = float(result.x[1]) # Optimized weight for similarity cue
        beta_dp = float(result.x[2]) # Optimized weight for phonological cue
        model_name.append('forage_phonologicalstatic')
        model_results.append((beta_df, beta_ds, beta_dp, float(result.fun)))
    if model == models[3] or model == models[4]:
        for i, switch_vec in enumerate(switch_vecs):
            key = tuple(switch_vec)
            if key not in _switch_cache:
                _switch_cache[key] = {}

            # Global Dynamic Phonological Model
            if 'pdynamic_global' not in _switch_cache[key]:
                r1 = np.random.rand()
                r2 = np.random.rand()
                r3 = np.random.rand()
                result = minimize(forage.model_dynamic_phon, [r1,r2,r3], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5], switch_vec,'global'))
                beta_df = float(result.x[0])
                beta_ds = float(result.x[1])
                beta_dp = float(result.x[2])
                _switch_cache[key]['pdynamic_global'] = (beta_df, beta_ds, beta_dp, float(result.fun))
            model_name.append('forage_phonologicaldynamicglobal_' + switch_names[i])
            model_results.append(_switch_cache[key]['pdynamic_global'])

            # Local Dynamic Phonological Model
            if 'pdynamic_local' not in _switch_cache[key]:
                r1 = np.random.rand()
                r2 = np.random.rand()
                r3 = np.random.rand()
                result = minimize(forage.model_dynamic_phon, [r1,r2,r3], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5], switch_vec,'local'))
                beta_df = float(result.x[0])
                beta_ds = float(result.x[1])
                beta_dp = float(result.x[2])
                _switch_cache[key]['pdynamic_local'] = (beta_df, beta_ds, beta_dp, float(result.fun))
            model_name.append('forage_phonologicaldynamiclocal_' + switch_names[i])
            model_results.append(_switch_cache[key]['pdynamic_local'])

            # Switch Dynamic Phonological Model
            if 'pdynamic_switch' not in _switch_cache[key]:
                r1 = np.random.rand()
                r2 = np.random.rand()
                r3 = np.random.rand()
                result = minimize(forage.model_dynamic_phon, [r1,r2,r3], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5], switch_vec,'switch'))
                beta_df = float(result.x[0])
                beta_ds = float(result.x[1])
                beta_dp = float(result.x[2])
                _switch_cache[key]['pdynamic_switch'] = (beta_df, beta_ds, beta_dp, float(result.fun))
            model_name.append('forage_phonologicaldynamicswitch_' + switch_names[i])
            model_results.append(_switch_cache[key]['pdynamic_switch'])

    # Unoptimized Model
    model_name.append('forage_random_baseline')
    nll_baseline = forage.model_static([0,0], history_vars[2], history_vars[3], history_vars[0], history_vars[1])
    model_results.append((0, 0, float(nll_baseline)))
    return model_name, model_results

def calculate_switch(switch, fluency_list, semantic_similarity, phon_similarity, norms, domain, times=None, alpha = np.arange(0, 1.1, 0.1), rise = np.arange(0, 1.25, 0.25), fall = np.arange(0, 1.25, 0.25), pei_beta = np.arange(0, 1.1, 0.1), pei_prior = np.arange(0.1, 1.0, 0.1)):
    '''
    1. Check if specified switch model is valid
    2. Return set of switches, including parameter value, if required

    switch_methods = ['simdrop','multimodal','norms_associative', 'norms_categorical', 'delta','multimodaldelta', 'slope_difference', 'pei', 'all']
    '''
    switch_names = []
    switch_vecs = []

    if switch not in switch_methods:
        ex_str = "Specified switch method is invalid. Switch method must be one of the following: {switch}".format(switch=switch_methods)
        raise Exception(ex_str)

    if switch == switch_methods[0] or switch == switch_methods[8]:
        switch_names.append(switch_methods[0])
        switch_vecs.append(switch_simdrop(fluency_list, semantic_similarity))

    if switch == switch_methods[1] or switch == switch_methods[8]:
        for i, a in enumerate(alpha):
            a = round(a, 1)
            switch_names.append('multimodal_alpha={alpha}'.format(alpha=a))
            switch_vecs.append(switch_multimodal(fluency_list, semantic_similarity, phon_similarity, a))

    if (switch == switch_methods[2] or switch == switch_methods[8]) and domain in ['animals','foods']:

        if domain == 'animals':
            switch_names.append("norms_associative")
            switch_vecs.append(switch_norms_associative(fluency_list,norms[0]))
        else:
            switch_names.append("norms_associative")
            switch_vecs.append(switch_norms_associative(fluency_list,norms[1]))

    if switch == switch_methods[3] or switch == switch_methods[8] and domain in ['animals','foods']:

        if domain == 'animals':
            switch_names.append("norms_categorical")
            switch_vecs.append(switch_norms_categorical(fluency_list,norms[0]))
        else:
            switch_names.append("norms_categorical")
            switch_vecs.append(switch_norms_categorical(fluency_list,norms[1]))

    if switch == switch_methods[4] or switch == switch_methods[8]:
        for i, r in enumerate(rise):
            for j, f in enumerate(fall):
                r = round(r, 1)
                f = round(f, 1)
                switch_names.append("delta_rise={rise}_fall={fall}".format(rise=r,fall=f))
                switch_vecs.append(switch_delta(fluency_list, semantic_similarity, r, f))

    if switch == switch_methods[5] or switch == switch_methods[8]:
        for i, a in enumerate(alpha):
            for i, r in enumerate(rise):
                for j, f in enumerate(fall):
                    # round a, r, f to 1 decimal places
                    a = round(a, 1)
                    r = round(r, 1)
                    f = round(f, 1)
                    switch_names.append("multimodaldelta_alpha={alpha}_rise={rise}_fall={fall}".format(alpha=a,rise=r,fall=f))
                    switch_vecs.append(switch_multimodaldelta(fluency_list, semantic_similarity, phon_similarity, r, f, a))

    # Precompute slope differences once (used by slope_difference, PEI, and lexical output)
    precomputed_slope_diffs = None
    if times is not None:
        _, precomputed_slope_diffs = switch_slope_difference(fluency_list, times)

    if switch == switch_methods[6] or switch == switch_methods[8]:
        if times is None:
            if switch == switch_methods[6]:
                raise Exception("Slope difference method requires timing data. Please provide cumulative response times in seconds via the 'times' parameter.")
            # skip when running 'all' without timing data
        else:
            switch_names.append("slope_difference")
            decisions = [2]
            for i in range(len(precomputed_slope_diffs)):
                decisions.append(1 if precomputed_slope_diffs[i] < 0 else 0)
            switch_vecs.append(decisions)

    if switch == switch_methods[7] or switch == switch_methods[8]:
        if times is None:
            if switch == switch_methods[7]:
                raise Exception("PEI method requires timing data. Please provide cumulative response times in seconds via the 'times' parameter.")
            # skip when running 'all' without timing data
        else:
            for i, a in enumerate(alpha):
                for j, b in enumerate(pei_beta):
                    for k, p in enumerate(pei_prior):
                        a = round(a, 1)
                        b = round(b, 1)
                        p = round(p, 1)
                        switch_names.append("pei_alpha={alpha}_beta={beta}_prior={prior}".format(alpha=a, beta=b, prior=p))
                        switch_vecs.append(switch_pei(fluency_list, times, semantic_similarity, phon_similarity, slope_diffs=precomputed_slope_diffs, alpha=a, beta=b, prior_probability=p))

    return switch_names, switch_vecs, precomputed_slope_diffs

def _physical_cpu_count():
    """Return the number of physical CPU cores (cross-platform)."""
    try:
        import subprocess
        if sys.platform == 'linux':
            out = subprocess.check_output(
                ['lscpu', '-p=Core'], text=True)
            # Count unique core IDs (lines not starting with '#')
            cores = set(line.strip() for line in out.splitlines()
                        if line.strip() and not line.startswith('#'))
            if cores:
                return len(cores)
        elif sys.platform == 'darwin':
            out = subprocess.check_output(
                ['sysctl', '-n', 'hw.physicalcpu'], text=True)
            return int(out.strip())
        elif sys.platform == 'win32':
            out = subprocess.check_output(
                ['wmic', 'cpu', 'get', 'NumberOfCores'], text=True)
            cores = [int(x) for x in out.split() if x.isdigit()]
            if cores:
                return sum(cores)
    except Exception:
        pass
    # Fallback: assume hyperthreading (2 threads per core)
    logical = os.cpu_count() or 1
    return max(1, logical // 2)


def _process_subject(item, labels, similarity_matrix, frequency_list, phon_matrix,
                     norms, domain, switch_type, model_type):
    """Process a single subject: compute lexical, switch, and model results.

    This is a top-level function so it can be pickled for multiprocessing.
    """
    subj, fl_list = item[0], item[1]
    fl_times = item[2] if len(item) == 3 else None

    # Compute history variables once per subject
    history_vars = create_history_variables(fl_list, labels, similarity_matrix, frequency_list, phon_matrix)

    # Lexical results
    lexical_df = pd.DataFrame()
    lexical_df['Subject'] = len(fl_list) * [subj]
    lexical_df['Fluency_Item'] = fl_list
    lexical_df['Semantic_Similarity'] = history_vars[0]
    lexical_df['Frequency_Value'] = history_vars[2]
    lexical_df['Phonological_Similarity'] = history_vars[4]

    # Add timing columns if timing data is available
    if fl_times is not None:
        cumulative = np.array(fl_times)
        lexical_df['Cumulative_IRT'] = cumulative
        irt = np.empty(len(cumulative))
        irt[0] = cumulative[0]
        irt[1:] = np.diff(cumulative)
        lexical_df['IRT'] = irt

    switch_dfs = []
    model_dicts = []

    # Switch results (computed once, reused by models)
    if switch_type is not None:
        switch_names, switch_vecs, slope_diffs = calculate_switch(
            switch_type, fl_list, history_vars[0], history_vars[4],
            norms, domain, times=fl_times)

        # Add slope differences to lexical results (from precomputed values)
        if fl_times is not None:
            sd_col = np.empty(len(fl_list))
            sd_col[0] = np.nan
            if slope_diffs is not None and len(slope_diffs) > 0:
                sd_col[1:] = slope_diffs
            else:
                sd_col[1:] = np.nan
            lexical_df['Slope_Difference'] = sd_col

        for j, switch in enumerate(switch_vecs):
            df = pd.DataFrame()
            df['Subject'] = len(switch) * [subj]
            df['Fluency_Item'] = fl_list
            df['Switch_Value'] = switch
            df['Switch_Method'] = switch_names[j]
            switch_dfs.append(df)

        # Model results (reuse switch_names/switch_vecs from above)
        if model_type is not None:
            model_names, model_results = calculate_model(
                model_type, history_vars, switch_names, switch_vecs)

            for k, model in enumerate(model_names):
                model_dict = dict()
                model_dict['Subject'] = subj
                model_dict['Model'] = model
                model_dict['Beta_Frequency'] = model_results[k][0]
                model_dict['Beta_Semantic'] = model_results[k][1]
                if len(model_results[k]) == 3:
                    model_dict['Beta_Phonological'] = None
                    model_dict['Negative_Log_Likelihood_Optimized'] = model_results[k][2]
                if len(model_results[k]) == 4:
                    model_dict['Beta_Phonological'] = model_results[k][2]
                    model_dict['Negative_Log_Likelihood_Optimized'] = model_results[k][3]
                model_dicts.append(model_dict)

    return lexical_df, switch_dfs, model_dicts


def run_pipeline(data, domain, switch_type=None, model_type=None, parallel=False, n_workers=None):
    """Single-pass pipeline that computes lexical, switch, and model results.

    Loads lexical data once and processes subjects in parallel using
    multiprocessing when --parallel is enabled and there are multiple subjects.

    Args:
        data: list of tuples (subj, fl_list[, fl_times]) from retrieve_data.
        domain: lexical domain (e.g. 'animals').
        switch_type: switch method name, or None to skip switches.
        model_type: model name, or None to skip models.
        parallel: if True, process subjects in parallel using multiprocessing.
        n_workers: number of parallel workers. None = number of physical CPUs.

    Returns:
        tuple: (lexical_results, switch_results, model_results) as DataFrames.
            switch_results and model_results are None when their corresponding
            type argument is None.
    """
    norms, similarity_matrix, phon_matrix, frequency_list, labels = get_lexical_data(domain)

    lexical_list = []
    switch_list = []
    model_list = []

    use_parallel = parallel and len(data) > 1

    if use_parallel:
        if n_workers is None:
            n_workers = min(_physical_cpu_count(), len(data))
        print(f"Processing {len(data)} subjects using {n_workers} parallel workers...")

        from functools import partial
        from concurrent.futures import as_completed
        worker_fn = partial(_process_subject,
                            labels=labels, similarity_matrix=similarity_matrix,
                            frequency_list=frequency_list, phon_matrix=phon_matrix,
                            norms=norms, domain=domain,
                            switch_type=switch_type, model_type=model_type)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {executor.submit(worker_fn, item): idx
                             for idx, item in enumerate(data)}
            results = [None] * len(data)
            for future in tqdm(as_completed(future_to_idx), total=len(data)):
                idx = future_to_idx[future]
                results[idx] = future.result()

        for lexical_df, switch_dfs, model_dicts in results:
            lexical_list.append(lexical_df)
            switch_list.extend(switch_dfs)
            model_list.extend(model_dicts)
    else:
        for i, item in enumerate(tqdm(data)):
            lexical_df, switch_dfs, model_dicts = _process_subject(
                item, labels, similarity_matrix, frequency_list, phon_matrix,
                norms, domain, switch_type, model_type)
            lexical_list.append(lexical_df)
            switch_list.extend(switch_dfs)
            model_list.extend(model_dicts)

    lexical_results = pd.concat(lexical_list, ignore_index=True)
    switch_results = pd.concat(switch_list, ignore_index=True) if switch_list else None
    forager_results = pd.DataFrame(model_list) if model_list else None

    return lexical_results, switch_results, forager_results


def indiv_desc_stats(lexical_results, switch_results = None):
    metrics = lexical_results[['Subject', 'Semantic_Similarity', 'Frequency_Value', 'Phonological_Similarity']]
    # replace first row of each subject with NaN for Semantic_Similarity and Phonological_Similarity
    metrics.loc[metrics.groupby('Subject').head(1).index, ['Semantic_Similarity', 'Phonological_Similarity']] = np.nan
    # ungroup the DataFrame
    metrics = metrics.reset_index(drop=True)
    # group the DataFrame by Subject and calculate the mean and standard deviation of each column
    grouped = metrics.groupby('Subject').agg(['mean', 'std'])
    grouped.columns = ['{}_{}'.format(col[0], col[1]) for col in grouped.columns]
    grouped.reset_index(inplace=True)
    num_items = lexical_results.groupby('Subject')['Fluency_Item'].size()
    grouped['#_of_Items'] = num_items[grouped['Subject']].values
    # create column for each switch method per subject and get number of switches, mean cluster size, and sd of cluster size for each switch method
    if switch_results is not None:
        # count the number of unique values in the Switch_Method column of the switch_results DataFrame
        n_rows = len(switch_results['Switch_Method'].unique())
        new_df = pd.DataFrame(np.nan, index=np.arange(len(grouped) * (n_rows)), columns=grouped.columns)

        # Insert the original DataFrame into the new DataFrame but repeat the value in 'Subject' column n_rows-1 times

        new_df.iloc[(slice(None, None, n_rows)), :] = grouped
        new_df['Subject'] = new_df['Subject'].ffill()

        switch_methods = []
        num_switches_arr = []
        cluster_size_mean = []
        cluster_size_sd = []
        for sub, fl_list in switch_results.groupby(["Subject", "Switch_Method"]):
            switch_method = sub[1]
            cluster_lengths = []
            num_switches = 0
            ct = 0
            for x in fl_list['Switch_Value'].values:
                ct += 1
                if x == 1:
                    num_switches += 1
                    ct = ct-1
                    cluster_lengths.append(ct)
                    ct = 1
            if ct != 0:
                cluster_lengths.append(ct)
            
            avg = sum(cluster_lengths) / len(cluster_lengths)
            sd = np.std(cluster_lengths)
            switch_methods.append(switch_method)
            num_switches_arr.append(num_switches)
            cluster_size_mean.append(avg)
            cluster_size_sd.append(sd)

        new_df['Switch_Method'] = switch_methods
        new_df['Number_of_Switches'] = num_switches_arr
        new_df['Cluster_Size_mean'] = cluster_size_mean
        new_df['Cluster_Size_std'] = cluster_size_sd
        grouped = new_df
        
    return grouped

def agg_desc_stats(switch_results, model_results=None):
    agg_df = pd.DataFrame()
    # get number of switches per subject for each switch method
    switches_per_method = {}
    for sub, fl_list in switch_results.groupby(["Subject", "Switch_Method"]):
        method = sub[1]
        if method not in switches_per_method:
            switches_per_method[method] = []
        if 1 in fl_list['Switch_Value'].values:
            switches_per_method[method].append(fl_list['Switch_Value'].value_counts()[1])
        else: 
            switches_per_method[method].append(0)
    agg_df['Switch_Method'] = switches_per_method.keys()
    agg_df['Switches_per_Subj_mean'] = [np.average(switches_per_method[k]) for k in switches_per_method.keys()]
    agg_df['Switches_per_Subj_SD'] = [np.std(switches_per_method[k]) for k in switches_per_method.keys()]
    
    if model_results is not None:
        betas = model_results.drop(columns=['Subject', 'Negative_Log_Likelihood_Optimized'])
        betas.drop(betas[betas['Model'] == 'forage_random_baseline'].index, inplace=True)
        grouped = betas.groupby('Model').agg(['mean', 'std'])
        grouped.columns = ['{}_{}'.format(col[0], col[1]) for col in grouped.columns]
        grouped.reset_index(inplace=True)

        # add a column to the grouped dataframe that contains the switch method used for each model
        grouped.loc[grouped['Model'].str.contains('static'), 'Model'] += ' none'
        # if the model name starts with 'forage_dynamic_', ''forage_phonologicaldynamicglobal_', 'forage_phonologicaldynamiclocal_', or 'forage_phonologicaldynamicswitch_', replace the second underscore with a space
        switch_models = ['forage_dynamic_', 'forage_phonologicaldynamicglobal_', 'forage_phonologicaldynamiclocal_', 'forage_phonologicaldynamicswitch_']
        for model in switch_models:
            # replace only the second underscore with a space
            grouped.loc[grouped['Model'].str.contains(model), 'Model'] = grouped.loc[grouped['Model'].str.contains(model), 'Model'].str.replace('_', ' ', 2)
            grouped.loc[grouped['Model'].str.contains("forage "), 'Model'] = grouped.loc[grouped['Model'].str.contains("forage "), 'Model'].str.replace(' ', '_', 1)
        
        # split the Model column on the space
        grouped[['Model', 'Switch_Method']] = grouped['Model'].str.rsplit(' ', n=1, expand=True)

        # merge the two dataframes on the Switch_Method column 
        agg_df = pd.merge(agg_df, grouped, how='outer', on='Switch_Method')


    return agg_df
 
parser = argparse.ArgumentParser(description='Execute Semantic Foraging Code.')
parser.add_argument('--data', type=str,  help='specifies path to fluency lists')
parser.add_argument('--pipeline',type=str, help='specifies which part of pipeline (lexical, switches, models) to execute')
parser.add_argument('--model', type=str, help='specifies foraging model to use')
parser.add_argument('--switch', type=str, help='specifies switch model to use')
parser.add_argument('--domain', type=str, help='specifies domain to use')
parser.add_argument('--time_type', type=str, default='cumulative', help="type of timing data: 'cumulative' or 'irt' (inter-response time). Default: cumulative")
parser.add_argument('--time_units', type=str, default='s', help="units of timing data: 's' (seconds) or 'ms' (milliseconds). Default: s")
parser.add_argument('--parallel', action='store_true', help='enable multiprocessing to process subjects in parallel (switches and models pipelines)')
parser.add_argument('--n_workers', type=int, default=None, help='number of parallel workers (default: number of physical CPU cores). Requires --parallel')

args = parser.parse_args()

if os.path.exists('output') == False:
    os.mkdir('output')

if args.data == None:
    parser.error("Please specify a data file for which you would like to run the forager pipeline for")

if args.pipeline == None:
    parser.error("Please specify which part of the forager pipeline you would like to execute for your data (e.g. \'lexical\', \'switches\',\'model\')")

args.data = os.path.join(os.getcwd(),args.data)
#oname = 'output/' + args.data.split('/')[-1].split('.')[0] + '_forager_results.zip'

output_dir = 'output'
file_name = args.domain + '_forager_results.zip'
oname = os.path.join(output_dir, file_name)


if args.pipeline == 'evaluate_data':
    data, replacement_df, processed_df = retrieve_data(args.data, args.domain, time_type=args.time_type, time_units=args.time_units)
    with zipfile.ZipFile(oname, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Save the first DataFrame as a CSV file inside the zip
        with zipf.open('evaluation_results.csv', 'w') as csvf:
            replacement_df.to_csv(csvf, index=False)

        # Save the second DataFrame as a CSV file inside the zip
        with zipf.open('processed_data.csv', 'w') as csvf:
            processed_df.to_csv(csvf, index=False)
        
        # Save vocab as a CSV file inside the zip
        with zipf.open('forager_vocab.csv', 'w') as csvf:
            vocabpath = 'data/lexical_data/' + args.domain + '/vocab.csv'
            vocab = pd.read_csv(vocabpath, encoding="unicode-escape")
            vocab.to_csv(csvf, index=False)

        print(f"File 'evaluation_results.csv' detailing the changes made to the dataset has been saved in '{oname}'")
        print(f"File 'processed_data.csv' containing the processed dataset used in the forager pipeline saved in '{oname}'")
        print(f"File 'forager_vocab.csv' containing the full vocabulary used by forager saved in '{oname}'")

elif args.pipeline == 'lexical':
    # Retrieve the Data for Getting Lexical Info
    data, replacement_df, processed_df = retrieve_data(args.data, args.domain, time_type=args.time_type, time_units=args.time_units)
    # Run subroutine for getting strictly the similarity & frequency values
    lexical_results, _, _ = run_pipeline(data, args.domain)
    ind_stats = indiv_desc_stats(lexical_results)
    with zipfile.ZipFile(oname, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Save the first DataFrame as a CSV file inside the zip
        with zipf.open('evaluation_results.csv', 'w') as csvf:
            replacement_df.to_csv(csvf, index=False)

        # Save the second DataFrame as a CSV file inside the zip
        with zipf.open('processed_data.csv', 'w') as csvf:
            processed_df.to_csv(csvf, index=False)
        # Save vocab as a CSV file inside the zip
        with zipf.open('forager_vocab.csv', 'w') as csvf:
            vocabpath = 'data/lexical_data/' + args.domain + '/vocab.csv'
            vocab = pd.read_csv(vocabpath, encoding="unicode-escape")
            vocab.to_csv(csvf, index=False)
        # save lexical results
        with zipf.open('lexical_results.csv','w') as csvf:
            lexical_results.to_csv(csvf, index=False) 
        # save individual descriptive statistics
        with zipf.open('individual_descriptive_stats.csv', 'w') as csvf:
            ind_stats.to_csv(csvf, index=False)

        print(f"File 'evaluation_results.csv' detailing the changes made to the dataset has been saved in '{oname}'")
        print(f"File 'processed_data.csv' containing the processed dataset used in the forager pipeline saved in '{oname}'")
        print(f"File 'forager_vocab.csv' containing the full vocabulary used by forager saved in '{oname}'")
        print(f"File 'lexical_results.csv' containing similarity and frequency values of fluency list data saved in '{oname}'")
        print(f"File 'individual_descriptive_stats.csv' containing individual-level statistics saved in '{oname}'")

        
elif args.pipeline == 'switches':
    # Check if switches, then there is a switch method specified
    if args.switch == None:
        parser.error(f"Please specify a switch method (e.g. {switch_methods})")
    if args.switch not in switch_methods:
        parser.error(f"Please specify a proper switch method (e.g. {switch_methods})")
    # Run subroutine for getting strictly switch outputs 
    # Run subroutine for getting model outputs
    print("Checking Data ...")
    data, replacement_df, processed_df = retrieve_data(args.data, args.domain, time_type=args.time_type, time_units=args.time_units)
    print("Running Pipeline ...")
    lexical_results, switch_results, _ = run_pipeline(data, args.domain, switch_type=args.switch,
        parallel=args.parallel, n_workers=args.n_workers)
    ind_stats = indiv_desc_stats(lexical_results, switch_results)
    agg_stats = agg_desc_stats(switch_results)
    with zipfile.ZipFile(oname, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Save the first DataFrame as a CSV file inside the zip
        with zipf.open('evaluation_results.csv', 'w') as csvf:
            replacement_df.to_csv(csvf, index=False)

        # Save the second DataFrame as a CSV file inside the zip
        with zipf.open('processed_data.csv', 'w') as csvf:
            processed_df.to_csv(csvf, index=False)

        # Save vocab as a CSV file inside the zip
        with zipf.open('forager_vocab.csv', 'w') as csvf:
            vocabpath = 'data/lexical_data/' + args.domain + '/vocab.csv'
            vocab = pd.read_csv(vocabpath, encoding="unicode-escape")
            vocab.to_csv(csvf, index=False)

        # save lexical results

        with zipf.open('lexical_results.csv','w') as csvf:
            lexical_results.to_csv(csvf, index=False)

        # save switch results
        with zipf.open('switch_results.csv', 'w') as csvf:
            switch_results.to_csv(csvf, index=False)
        print(f"File 'switch_results.csv' containing switch results saved in '{oname}'")

        # save individual descriptive statistics
        with zipf.open('individual_descriptive_stats.csv', 'w') as csvf:
            ind_stats.to_csv(csvf, index=False)

        # save aggregate descriptive statistics
        with zipf.open('aggregate_descriptive_stats.csv', 'w') as csvf:
            agg_stats.to_csv(csvf, index=False)

        print(f"File 'evaluation_results.csv' detailing the changes made to the dataset has been saved in '{oname}'")
        print(f"File 'processed_data.csv' containing the processed dataset used in the forager pipeline saved in '{oname}'")
        print(f"File 'forager_vocab.csv' containing the full vocabulary used by forager saved in '{oname}'")
        print(f"File 'lexical_results.csv' containing similarity and frequency values of fluency list data saved in '{oname}'")
        print(f"File 'individual_descriptive_stats.csv' containing individual-level statistics saved in '{oname}'")
        print(f"File 'aggregate_descriptive_stats.csv' containing the overall group-level statistics saved in '{oname}'")

elif args.pipeline == 'models':
    # Check for model and switch parameters
    if args.model == None:
        parser.error(f"Please specify a forager model (e.g. {models})")
    if args.model not in models:
        parser.error(f"Please specify a proper forager model (e.g. {models})")
    if args.switch == None:
        parser.error(f"Please specify a switch method (e.g. {switch_methods})")
    if args.switch not in switch_methods:
        parser.error(f"Please specify a proper switch method (e.g. {switch_methods})")
    # Run subroutine for getting model outputs
    print("Checking Data ...")
    data, replacement_df, processed_df = retrieve_data(args.data, args.domain, time_type=args.time_type, time_units=args.time_units)
    print("Running Pipeline ...")
    lexical_results, switch_results, forager_results = run_pipeline(
        data, args.domain, switch_type=args.switch, model_type=args.model,
        parallel=args.parallel, n_workers=args.n_workers)

    ind_stats = indiv_desc_stats(lexical_results, switch_results)
    agg_stats = agg_desc_stats(switch_results, forager_results)
    with zipfile.ZipFile(oname, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Save the first DataFrame as a CSV file inside the zip
        with zipf.open('evaluation_results.csv', 'w') as csvf:
            replacement_df.to_csv(csvf, index=False)

        # Save the second DataFrame as a CSV file inside the zip
        with zipf.open('processed_data.csv', 'w') as csvf:
            processed_df.to_csv(csvf, index=False)
        
        # Save vocab as a CSV file inside the zip
        with zipf.open('forager_vocab.csv', 'w') as csvf:
            vocabpath = 'data/lexical_data/' + args.domain + '/vocab.csv'
            vocab = pd.read_csv(vocabpath, encoding="unicode-escape")
            vocab.to_csv(csvf, index=False)
        # save lexical results
        with zipf.open('lexical_results.csv','w') as csvf:
            lexical_results.to_csv(csvf, index=False) 
        # save switch results
        with zipf.open('switch_results.csv', 'w') as csvf:
            switch_results.to_csv(csvf, index=False)
        print(f"File 'switch_results.csv' containing switch results saved in '{oname}'")
        # save model results
        with zipf.open('model_results.csv','w') as csvf:
            forager_results.to_csv(csvf, index=False)
        # save individual descriptive statistics
        with zipf.open('individual_descriptive_stats.csv', 'w') as csvf:
            ind_stats.to_csv(csvf, index=False)
        # save aggregate descriptive statistics
        with zipf.open('aggregate_descriptive_stats.csv', 'w') as csvf:
            agg_stats.to_csv(csvf, index=False)

        print(f"File 'evaluation_results.csv' detailing the changes made to the dataset has been saved in '{oname}'")
        print(f"File 'processed_data.csv' containing the processed dataset used in the forager pipeline saved in '{oname}'")
        print(f"File 'forager_vocab.csv' containing the full vocabulary used by forager saved in '{oname}'")
        print(f"File 'lexical_results.csv' containing similarity and frequency values of fluency list data saved in '{oname}'")
        print(f"File 'model_results.csv' containing model level NLL results of provided fluency data saved in '{oname}'")
        print(f"File 'individual_descriptive_stats.csv' containing individual-level statistics saved in '{oname}'")
        print(f"File 'aggregate_descriptive_stats.csv' containing the overall group-level statistics saved in '{oname}'")

else:
    parser.error("Please specify a proper pipeline option (e.g. \'evaluate_data\', \'lexical\', \'switches\',\'models\')")


 
#### SAMPLE RUN CODE ####
## Sample execution to evaluate data file ##
# python run_foraging.py --data data/fluency_lists/psyrev_data.txt --pipeline evaluate_data

## Sample execution to obtain lexical metrics (semantic similarity, phonological similarity, frequency) ##
# python run_foraging.py --data data/fluency_lists/psyrev_data.txt --pipeline lexical

## Sample execution to obtain switch designations + lexical metrics (semantic similarity, phonological similarity, frequency) ##
## 'all' switch method will run all switch methods ##
## other possible arguments for --switch include: 'simdrop', 'multimodal', 'norms_associative','norms_categorical', 'delta' ##

# python run_foraging.py --data data/fluency_lists/psyrev_data.txt --pipeline switches --switch all

## Sample execution to obtain model results ##
## 'all' model will run all models ##
## other possible arguments for --model include: 'static', 'dynamic', 'pstatic', 'pdynamic' ##

# python run_foraging.py --data data/fluency_lists/psyrev_data.txt --pipeline models --model all
