# forager

Semantic Foraging methods for estimating semantic memory search on semantic fluency data

## Installation:

In order to install the package as is , there are two options

### Recommended Installation 
To install the broader package, including command-line interfacable programs along with metadata, we encourage you to use the following installation instructions:

    1. Clone this github repository to a home directory of your choice
        ```
        git clone https://github.com/larryzhang95/forager.git
        ```

    2. In current home directory, execute:
        ```
        pip install forager -r forager/requirements.txt
        ```

    3. To utilize the package, you can now change your current working directory to the forager package, and execute the run_foraging.py file to utilize the command line interface  
        ```
        cd forager
        python run_foraging.py --data <datapath> --model <modelname> --switch <switchmethod> 
        ```

### Alternative Installation
In order to install the package without auxilliary files, you can also install with the following command

    pip install git+https://github.com/larryzhang95/forager

## Installation Requirements
Requires Python 3.8 

Requirements:
- nltk>=3.6
- numpy>=1.20
- pandas>=1.3
- pytest>=6.2
- scipy>=1.6
- requests>=2.25
- urllib>=1.26
- tqdm>=4.59
## Usage: 
 
In order to utilize the package, there are a few key parameters that must be satisfied
   
    1. data : The --data flag requires you to specify the path to the fluency list file that you would like to execute foraging methods on

    2. model: The --model flag requires you to pass one of the following arguments, to run corresponding model(s) you would like to execute.
        a. static
        b. dynamic
        c. pstatic
        d. pdynamic
        e. all

    3. switch: The --switch flag requires you to pass one of the following arguments to utilize corresponding switch method(s) in the model selected
        a. troyer
        b. simdrop
        c. multimodal
        d. delta
        e. all

Below are sample executions to execute the code, on example data we provide with our package:

    a.  Sample execution with single model and all switches:
        ```
        python run_foraging.py --data data/fluency_lists/data-psyrev.txt --model dynamic --switch all
        ```

    b. Sample execution with all models and single switch:
        ```
        python run_foraging.py --data data/fluency_lists/data-psyrev.txt --model all --switch simdrop
        ```

    c.	Running all models and switches
        ```
        python run_foraging.py --data data/fluency_lists/data-psyrev.txt --model all --switch all
        ```

## Functionality

WHAT TO DO: Give description of the method(s) -- should generally reflect what the comments read or should read as. 

### Semantic Foraging Models
- Static Model 
    - TODO: Larry
- Dynamic Model
    - TODO: Larry
- Phonological Static Model
    - the phonological static model (```pstatic```) is an extension of the static model, where all transitions are based on a combined product of semantic similarity, word frequency, and phonological similarity.
- Phonological Dynamic Model
    - the phonological dynamic model has 3 versions, indexed by the ```phoncue``` parameter. The "local" model uses frequency, semantic, and phonological similarity during within-cluster transitions and frequency during between-cluster transitions. The "global" model uses frequency, semantic, and phonological similarity during within-cluster transitions, and frequency and phonological similarity during between-cluster transitions. Finally, the "switch" model uses only semantic similarity and frequency during within-cluster transitions and phonological similarity and frequency for between-cluster transitions.

### Switch Methods
- Norms-based (Troyer Norms)
    - TODO: Larry
- Similarity Drop
    - TODO: Larry
- Delta Similarity
    - TODO: Molly 
- Multimodal Similarity Drop
    - TODO: Molly

### Cues (Semantic, Phonological, and Frequency Matrix) Generation
Semantic Similarity Matrix Generation
- The semantic similarity matrix is generated using an underlying semantic representational model ("embeddings"). The package currently uses the word2vec model and computes pairwise cosine similarity for all items in the space (indexed by the size of embeddings).

Phonological Matrix Generation
- The phonological similarity matrix computes the pairwise normalized edit distance between the phonemic transcriptions of all items in the space (indexed by a list (```labels```). Phonemic transcriptions are obtained via CMUdict, which uses Arpabet phonemic transcriptions.

Frequency Data Generation
- TODO: Molly

History Variabile Creation:
- History variables is a utility function that keeps track of lexical metrics (frequency, semantic, and phonological similarity) within a given fluency list. Specifically, the function uses underlying semantic and phonological similarity matrices as well as word frequency, and returns the similarites between consecutive items within a specific fluency list.

### Util Functions
Prepare Data Function
- TODO: Molly 


## Development Notes

