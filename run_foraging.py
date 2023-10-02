import argparse
from scipy.optimize import fmin
from forager.foraging import forage
from forager.switch import switch_delta, switch_multimodal, switch_simdrop, switch_troyer
from forager.cues import create_history_variables
from forager.utils import prepareData
import pandas as pd
import numpy as np
from scipy.optimize import fmin, minimize
import os, sys
from tqdm import tqdm
import warnings 
import zipfile

warnings.simplefilter('ignore')

"""
Workflow: 
1. Validate input(s)
    a. "Prepare Data" - does this also get required freq/sim data?
        - takes path of fluency list ; replace/truncated fluency list

2. Run model(s)
    a. Model Optimization: Currently, the code base doesn't do optimization implicity. We have to include that now.
        Question: Do we want to do the same and report optimized & unoptimized model results?; along with param values?
    b. Running through switch method(s)

3. Outputs:
    a. Results
    b. Optimized Parameters
    c. Runtime
    d. Best model(s)/switching?

4. Extras & Reporting/Comparison?:
    a. visualization(s)
    X b. statistical test(s) & reporting

"""
# Global Path Variabiles
normspath =  'data/norms/troyernorms.csv'
similaritypath =  'data/lexical_data/similaritymatrix.csv'
frequencypath =  'data/lexical_data/frequencies.csv'
phonpath = 'data/lexical_data/phonmatrix.csv'

# Global Variables
models = ['static','dynamic','pstatic','pdynamic','all']
switch_methods = ['simdrop','multimodal','troyer','delta','all']

#Methods
def retrieve_data(path):
    """
    1. Verify that data path exists

    """
    if os.path.exists(path) == False:
        ex_str = "Provided path to data \"{path}\" does not exist. Please specify a proper path".format(path=path)
        raise Exception(ex_str)
    data = prepareData(path)
    return data

def get_lexical_data():
    norms = pd.read_csv(normspath, encoding="unicode-escape")
    similarity_matrix = np.loadtxt(similaritypath,delimiter=' ')
    frequency_list = np.array(pd.read_csv(frequencypath,header=None,encoding="unicode-escape")[1])
    phon_matrix = np.loadtxt(phonpath,delimiter=',')
    labels = pd.read_csv(frequencypath,header=None)[0].values.tolist()
    return norms, similarity_matrix, phon_matrix, frequency_list,labels

def calculate_model(model, history_vars, switch_names, switch_vecs):
    """
    1. Check if specified model is valid
    2. Return a set of model functions to pass
    """
    model_name = []
    model_results = []
    if model not in models:
        ex_str = "Specified model is invalid. Model must be one of the following: {models}".format(models=models)
        raise Exception(ex_str)
    if model == models[0] or model == models[4]:
        r1 = np.random.rand()
        r2 = np.random.rand()

        v = minimize(forage.model_static, [r1,r2], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1])).x
        beta_df = float(v[0]) # Optimized weight for frequency cue
        beta_ds = float(v[1]) # Optimized weight for similarity cue
        
        nll, nll_vec = forage.model_static_report([beta_df, beta_ds], history_vars[2], history_vars[3], history_vars[0], history_vars[1])
        model_name.append('forage_static')
        model_results.append((beta_df, beta_ds, nll, nll_vec))
    if model == models[1] or model == models[4]:
        for i, switch_vec in enumerate(switch_vecs):
            r1 = np.random.rand()
            r2 = np.random.rand()

            v = minimize(forage.model_dynamic, [r1,r2], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1], switch_vec)).x
            beta_df = float(v[0]) # Optimized weight for frequency cue
            beta_ds = float(v[1]) # Optimized weight for similarity cue
            
            nll, nll_vec = forage.model_dynamic_report([beta_df, beta_ds], history_vars[2], history_vars[3], history_vars[0], history_vars[1],switch_vec)
            model_name.append('forage_dynamic_' + switch_names[i])
            model_results.append((beta_df, beta_ds, nll, nll_vec))
    if model == models[2] or model == models[4]:
        r1 = np.random.rand()
        r2 = np.random.rand()
        r3 = np.random.rand()
        v = minimize(forage.model_static_phon, [r1,r2,r3], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1], history_vars[4],history_vars[5])).x
        beta_df = float(v[0]) # Optimized weight for frequency cue
        beta_ds = float(v[1]) # Optimized weight for similarity cue
        beta_dp = float(v[2]) # Optimized weight for phonological cue

        nll, nll_vec = forage.model_static_phon_report([beta_df, beta_ds, beta_dp], history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5])
        model_name.append('forage_phonologicalstatic')
        model_results.append((beta_df, beta_ds, beta_dp, nll, nll_vec))
    if model == models[3] or model == models[4]:
        for i, switch_vec in enumerate(switch_vecs):
            # Global Dynamic Phonological Model
            r1 = np.random.rand()
            r2 = np.random.rand()
            r3 = np.random.rand()
            v = minimize(forage.model_dynamic_phon, [r1,r2,r3], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5], switch_vec,'global')).x
            beta_df = float(v[0]) # Optimized weight for frequency cue
            beta_ds = float(v[1]) # Optimized weight for similarity cue
            beta_dp = float(v[2]) # Optimized weight for phonological cue
            
            nll, nll_vec = forage.model_dynamic_phon_report([beta_df, beta_ds,beta_dp], history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5],switch_vec,'global')
            model_name.append('forage_phonologicaldynamicglobal_' + switch_names[i])
            model_results.append((beta_df, beta_ds, beta_dp, nll, nll_vec))
    
            # Local Dynamic Phonological Model
            r1 = np.random.rand()
            r2 = np.random.rand()
            r3 = np.random.rand()
            v = minimize(forage.model_dynamic_phon, [r1,r2,r3], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5], switch_vec,'local')).x
            beta_df = float(v[0]) # Optimized weight for frequency cue
            beta_ds = float(v[1]) # Optimized weight for similarity cue
            beta_dp = float(v[2]) # Optimized weight for phonological cue
            
            nll, nll_vec = forage.model_dynamic_phon_report([beta_df, beta_ds,beta_dp], history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5],switch_vec,'local')
            model_name.append('forage_phonologicaldynamiclocal_' + switch_names[i])
            model_results.append((beta_df, beta_ds, beta_dp, nll, nll_vec))
    
            # Switch Dynamic Phonological Model
            r1 = np.random.rand()
            r2 = np.random.rand()
            r3 = np.random.rand()
            v = minimize(forage.model_dynamic_phon, [r1,r2,r3], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5], switch_vec,'switch')).x
            beta_df = float(v[0]) # Optimized weight for frequency cue
            beta_ds = float(v[1]) # Optimized weight for similarity cue
            beta_dp = float(v[2]) # Optimized weight for phonological cue
            
            nll, nll_vec = forage.model_dynamic_phon_report([beta_df, beta_ds,beta_dp], history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5],switch_vec,'switch')
            model_name.append('forage_phonologicaldynamicswitch_' + switch_names[i])

            model_results.append((beta_df, beta_ds, beta_dp, nll, nll_vec))
    
    # Unoptimized Model
    model_name.append('forage_random_baseline')
    nll_baseline, nll_baseline_vec = forage.model_static_report(beta = [0,0], freql = history_vars[2], freqh = history_vars[3], siml = history_vars[0], simh = history_vars[1])
    model_results.append((0, 0, nll_baseline, nll_baseline_vec))
    return model_name, model_results

def calculate_switch(switch, fluency_list, semantic_similarity, phon_similarity, norms, alpha = np.arange(0, 1.1, 0.1), rise = np.arange(0, 1.25, 0.25), fall = np.arange(0, 1.25, 0.25)):
    '''
    1. Check if specified switch model is valid
    2. Return set of switches, including parameter value, if required
    '''
    switch_names = []
    switch_vecs = []

    if switch not in switch_methods:
        ex_str = "Specified switch method is invalid. Switch method must be one of the following: {switch}".format(switch=switch_methods)
        raise Exception(ex_str)

    if switch == switch_methods[0] or switch == switch_methods[4]:
        switch_names.append(switch_methods[0])
        switch_vecs.append(switch_simdrop(fluency_list, semantic_similarity))

    if switch == switch_methods[1] or switch == switch_methods[4]:
        for i, a in enumerate(alpha):
            switch_names.append('multimodal_alpha={alpha}'.format(alpha=a))
            switch_vecs.append(switch_multimodal(fluency_list, semantic_similarity, phon_similarity, a))

    if switch == switch_methods[2] or switch == switch_methods[4]:
        switch_names.append(switch_methods[2])
        switch_vecs.append(switch_troyer(fluency_list,norms))

    if switch == switch_methods[3] or switch == switch_methods[4]:
        for i, r in enumerate(rise):
            for j, f in enumerate(fall):
                switch_names.append("delta_rise={rise}_fall={fall}".format(rise=r,fall=f))
                switch_vecs.append(switch_delta(fluency_list, semantic_similarity, r, f))

    return switch_names, switch_vecs

# def synthesize_results(outputs):
#     """
#     Output file(s):
#     - All model result(s)
#     - Switch Vector Result(s)
#     - Item-Wise Negative Log Likelihood
#     """
#     model_results = []
#     switch_results = []
#     nll_results = []
#     for output in outputs:
#         subj = output[0]
#         fl_list = output[1]
#         model_names = output[2]
#         results = output[3]
#         switch_methods = output[4]
#         switch_vectors = output[5]
#         #Create Model Output Results DataFrame
#         for i, model in enumerate(model_names):
#             model_dict = dict()
#             model_dict['Subject'] = subj
#             model_dict['Model'] = model
#             model_dict['Beta_Frequency'] = results[i][0]
#             model_dict['Beta_Semantic'] = results[i][1]
#             # print(results[i])
#             # sys.exit()
#             if len(results[i]) == 4:
#                 model_dict['Beta_Phonological'] = None
#                 model_dict['Negative_Log_Likelihood_Optimized'] = results[i][2]
#             if len(results[i]) == 5:
#                 model_dict['Beta_Phonological'] = results[i][2]
#                 model_dict['Negative_Log_Likelihood_Optimized'] = results[i][3]
#             model_results.append(model_dict)
        
#         #Create  Switch Results DataFrame
#         switch_df = []
#         for j, switch in enumerate(switch_vectors):
#             df = pd.DataFrame()
#             df['Subject'] = len(switch) * [subj]
#             df['Fluency_Item'] = fl_list
#             df['Switch_Value'] = switch
#             df['Switch_Method'] = switch_methods[j]
#             switch_df.append(df)
    
#         switch_df = pd.concat(switch_df, ignore_index=True)
#         switch_results.append(switch_df)

#         # Create Negative Log Likelihood DataFrame with Item Wise NLL 
#         nll_df = pd.DataFrame()
#         nll_df['Subject'] = len(fl_list) * [subj]
#         nll_df['Fluency_Item'] = fl_list
#         for k, result in enumerate(results):
#             if len(result) == 4:
#                 nll_df['NLL_{model}'.format(model=model_names[k])] = result[3]
#             if len(result) == 5:
#                 nll_df['NLL_{model}'.format(model=model_names[k])] = result[4]
#         # Add freq, semantic sim, and phon sim values to itemwise nll data
#         nll_df['Semantic_Similarity'] = output[6][0]
#         nll_df['Frequency_Value'] = output[6][2]
#         nll_df['Phonological_Similarity'] = output[6][4]
#         nll_results.append(nll_df)

#     model_results = pd.DataFrame(model_results)
#     switch_results = pd.concat(switch_results, ignore_index=True)
#     nll_results = pd.concat(nll_results,ignore_index=True)
  
#     return model_results, switch_results, nll_results

# def output_results(results,dname,dpath='output',sep=','):
#     if os.path.exists(dpath) == False:
#         os.mkdir(dpath)
#     results[0].to_csv(os.path.join(dpath,dname + '_modelresults.csv'), index=False, sep=sep)        
#     results[1].to_csv(os.path.join(dpath,dname + '_switchresults.csv'), index=False, sep=sep)
#     results[2].to_csv(os.path.join(dpath,dname + '_individualitemfits.csv'), index=False, sep=sep)


def run_model(data, model_type, switch_type):
    # Get Lexical Data needed for executing methods
    norms, similarity_matrix, phon_matrix, frequency_list, labels = get_lexical_data()
    forager_results = []
    # Run through each fluency list in dataset
    for i, (subj, fl_list) in enumerate(tqdm(data)):
        print("\nRunning Model for Subject {subj}".format(subj=subj))
        import time
        start_time = time.time()
        # Get History Variables 
        history_vars = create_history_variables(fl_list, labels, similarity_matrix, frequency_list, phon_matrix)
        
        # Calculate Switch Vector(s)
        switch_names, switch_vecs = calculate_switch(switch_type, fl_list, history_vars[0],   history_vars[4], norms)

        #Execute Individual Model(s) and get result(s)
        model_names, model_results = calculate_model(model_type,history_vars, switch_names, switch_vecs)

        #Create Model Output Results DataFrame
        for i, model in enumerate(model_names):
            model_dict = dict()
            model_dict['Subject'] = subj
            model_dict['Model'] = model
            model_dict['Beta_Frequency'] = model_results[i][0]
            model_dict['Beta_Semantic'] = model_results[i][1]
            # print(results[i])
            # sys.exit()
            if len(model_results[i]) == 4:
                model_dict['Beta_Phonological'] = None
                model_dict['Negative_Log_Likelihood_Optimized'] = model_results[i][2]
            if len(model_results[i]) == 5:
                model_dict['Beta_Phonological'] = model_results[i][2]
                model_dict['Negative_Log_Likelihood_Optimized'] = model_results[i][3]
            forager_results.append(model_dict)
    forager_results = pd.DataFrame(forager_results)
        # outputs.append([subj, fl_list, model_names, model_results, switch_names, switch_vecs,history_vars])
        # print("Results: {names} , {res}".format(names = model_names, res=model_results))
    # print("--- Ran for %s seconds ---" % (time.time() - start_time))
    return forager_results
    # model_results, switch_results, nll_results = synthesize_results(outputs)
    # output_results([model_results,switch_results,nll_results],dname)

def run_lexical(data):
    # Get Lexical Data needed for executing methods
    norms, similarity_matrix, phon_matrix, frequency_list, labels = get_lexical_data()
    lexical_results = []
    for i, (subj, fl_list) in enumerate(tqdm(data)):
        history_vars = create_history_variables(fl_list, labels, similarity_matrix, frequency_list, phon_matrix)
        lexical_df = pd.DataFrame()
        lexical_df['Subject'] = len(fl_list) * [subj]
        lexical_df['Fluency_Item'] = fl_list
        lexical_df['Semantic_Similarity'] = history_vars[0]
        lexical_df['Frequency_Value'] = history_vars[2]
        lexical_df['Phonological_Similarity'] = history_vars[4]
        lexical_results.append(lexical_df)
    lexical_results = pd.concat(lexical_results,ignore_index=True)
    return lexical_results

def run_switches(data,switch_type):
    norms, similarity_matrix, phon_matrix, frequency_list, labels = get_lexical_data()
    switch_results = []
    for i, (subj, fl_list) in enumerate(tqdm(data)):
        history_vars = create_history_variables(fl_list, labels, similarity_matrix, frequency_list, phon_matrix)
        switch_names, switch_vecs = calculate_switch(switch_type, fl_list, history_vars[0], history_vars[4], norms)
    
        switch_df = []
        for j, switch in enumerate(switch_vecs):
            df = pd.DataFrame()
            df['Subject'] = len(switch) * [subj]
            df['Fluency_Item'] = fl_list
            df['Switch_Value'] = switch
            df['Switch_Method'] = switch_names[j]
            switch_df.append(df)
    
        switch_df = pd.concat(switch_df, ignore_index=True)
        switch_results.append(switch_df)
    switch_results = pd.concat(switch_results, ignore_index=True)
    return switch_results

parser = argparse.ArgumentParser(description='Execute Semantic Foraging Code.')
parser.add_argument('--data', type=str,  help='specifies path to fluency lists')
parser.add_argument('--pipeline',type=str, help='specifies which part of pipeline (lexical, switches, model) to execute')
parser.add_argument('--model', type=str, help='specifies foraging model to use')
parser.add_argument('--switch', type=str, help='specifies switch model to use')

args = parser.parse_args()

if os.path.exists('output') == False:
    os.mkdir('output')

if args.data == None:
    parser.error("Please specify a data file for which you would like to run the forager pipeline for")

if args.pipeline == None:
    parser.error("Please specify which part of the forager pipeline you would like to execute for your data (e.g. \'lexical\', \'switches\',\'model\')")

args.data = os.path.join(os.getcwd(),args.data)
oname = 'output/' + args.data.split('/')[-1].split('.')[0] + '_forager_results.zip'


if args.pipeline == 'evaluate_data':
    data, replacement_df, processed_df = retrieve_data(args.data)
    with zipfile.ZipFile(oname, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Save the first DataFrame as a CSV file inside the zip
        with zipf.open('evaluation_results.csv', 'w') as csvf:
            replacement_df.to_csv(csvf, index=False)

        # Save the second DataFrame as a CSV file inside the zip
        with zipf.open('processed_data.csv', 'w') as csvf:
            processed_df.to_csv(csvf, index=False)

        print(f"File 'evaluation_results.csv' detailing the changes made to the dataset has been saved in '{oname}'")
        print(f"File 'processed_data.csv' containing the processed dataset used in the forager pipeline saved in '{oname}'")

elif args.pipeline == 'lexical':
    dname = 'lexical_results.csv'
    # Retrieve the Data for Getting Lexical Info
    data, replacement_df, processed_df = retrieve_data(args.data)
    # Run subroutine for getting strictly the similarity & frequency values 
    lexical_results = run_lexical(data)
    with zipfile.ZipFile(oname, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Save the first DataFrame as a CSV file inside the zip
        with zipf.open('evaluation_results.csv', 'w') as csvf:
            replacement_df.to_csv(csvf, index=False)

        # Save the second DataFrame as a CSV file inside the zip
        with zipf.open('processed_data.csv', 'w') as csvf:
            processed_df.to_csv(csvf, index=False)

        with zipf.open(dname,'w') as csvf:
            lexical_results.to_csv(csvf, index=False) 

        print(f"File 'evaluation_results.csv' detailing the changes made to the dataset has been saved in '{oname}'")
        print(f"File 'processed_data.csv' containing the processed dataset used in the forager pipeline saved in '{oname}'")
        print(f"File 'lexical_results.csv' containing similarity and frequency values of fluency list data saved in '{oname}'")

        
elif args.pipeline == 'switches':
    dname = 'switch_results.csv'
    # Check if switches, then there is a switch method specified
    if args.switch == None:
        parser.error(f"Please specify a switch method (e.g. {switch_methods})")
    if args.switch not in switch_methods:
        parser.error(f"Please specify a proper switch method (e.g. {switch_methods})")
    # Run subroutine for getting strictly switch outputs 
    data, replacement_df, processed_df = retrieve_data(args.data) 
    switch_results = run_switches(data,args.switch)
    with zipfile.ZipFile(oname, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Save the first DataFrame as a CSV file inside the zip
        with zipf.open('evaluation_results.csv', 'w') as csvf:
            replacement_df.to_csv(csvf, index=False)

        # Save the second DataFrame as a CSV file inside the zip
        with zipf.open('processed_data.csv', 'w') as csvf:
            processed_df.to_csv(csvf, index=False)

        with zipf.open(dname,'w') as csvf:
            switch_results.to_csv(csvf, index=False) 

        print(f"File 'evaluation_results.csv' detailing the changes made to the dataset has been saved in '{oname}'")
        print(f"File 'processed_data.csv' containing the processed dataset used in the forager pipeline saved in '{oname}'")
        print(f"File 'switch_results.csv' containing designated switch methods and switch values of fluency list data saved in '{oname}'")

elif args.pipeline == 'model':
    switch_name = 'switch_results.csv'
    lexical_name = 'lexical_results.csv'
    models_name = 'model_results.csv'
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
    data, replacement_df, processed_df = retrieve_data(args.data)
    print("Retrieving Lexical Data ...")
    lexical_results = run_lexical(data)
    print("Obtaining Switch Designations ...")
    switch_results = run_switches(data,args.switch)
    print("Running Forager Models...")
    forager_results = run_model(data, args.model, args.switch)

    with zipfile.ZipFile(oname, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Save the first DataFrame as a CSV file inside the zip
        with zipf.open('evaluation_results.csv', 'w') as csvf:
            replacement_df.to_csv(csvf, index=False)

        # Save the second DataFrame as a CSV file inside the zip
        with zipf.open('processed_data.csv', 'w') as csvf:
            processed_df.to_csv(csvf, index=False)

        with zipf.open(lexical_name,'w') as csvf:
            lexical_results.to_csv(csvf, index=False) 

        with zipf.open(switch_name,'w') as csvf:
            switch_results.to_csv(csvf, index=False) 

        with zipf.open(models_name,'w') as csvf:
            forager_results.to_csv(csvf, index=False) 

        print(f"File 'evaluation_results.csv' detailing the changes made to the dataset has been saved in '{oname}'")
        print(f"File 'processed_data.csv' containing the processed dataset used in the forager pipeline saved in '{oname}'")
        print(f"File 'lexical_results.csv' containing similarity and frequency values of fluency list data saved in '{oname}'")
        print(f"File 'switch_results.csv' containing designated switch methods and switch values of fluency list data saved in '{oname}'")
        print(f"File 'model_results.csv' containing model level NLL results of provided fluency data saved in '{oname}'")

else:
    parser.error("Please specify a proper pipeline option (e.g. \'evaluate_data\', \'lexical\', \'switches\',\'model\')")




# Sample execution with single model and all switches: 
# python run_foraging.py --data data/fluency_lists/psyrev_data.txt --model dynamic --switch all

# Sample execution with all models and single switch: 
# python run_foraging.py --data data/fluency_lists/psyrev_data.txt --model all --switch simdrop

# Running all models and switches
# python run_foraging.py --data data/fluency_lists/psyrev_data.txt --model all --switch all