# forager

Semantic Foraging methods for estimating semantic memory search on semantic fluency data

## Installation:

In order to install the package as is , there are two options

### Recommended Installation 
To install the broader package, including command-line interfacable programs along with metadata, we encourage you to use the following installation instructions:

    1. Clone this github repository to a home directory of your choice
        ```
        git clone https://github.com/thelexiconlab/forager.git
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

    pip install git+https://github.com/thelexiconlab/forager

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

    c. Running all models and switches
        ```
        python run_foraging.py --data data/fluency_lists/data-psyrev.txt --model all --switch all
        ```

## Functionality

### Semantic Foraging Models
- Static Model 
    - the original static model (```static```) executes foraging where all transitions are based on the same set of cues over the entire retrieval interval, effectively ignoring the patchy structure of the memory retrieval environment. All transitions are based on a combined product of semantic similarity and word frequency. The static foraging model was introduced in Hills TT, Jones MN, Todd PM (2012).
- Dynamic Model
    - the original dynamic model (```dynamic```) executes foraging by employing a clustering and switching mechanism that exploits the patchy structure of memory. The dynamic model utilizes word frequency and semantic similarity during within-clustering transitions, and word frequency during between-cluster transitions. Cluster and switching behavior is captured via the ```switchvals``` parameter, which can be calculated by the provided switch methods in the package. The static foraging model was introduced in Hills TT, Jones MN, Todd PM (2012).
- Phonological Static Model
    - the phonological static model (```pstatic```) is an extension of the static model, where all transitions are based on a combined product of semantic similarity, word frequency, and phonological similarity. The phonological static model was introduced in Kumar AA, Lundin NB, & Jones MN (2022)
- Phonological Dynamic Model
    - the phonological dynamic (```pdynamic```) model has 3 versions, indexed by the ```phoncue``` parameter. The "local" model uses frequency, semantic, and phonological similarity during within-cluster transitions and frequency during between-cluster transitions. The "global" model uses frequency, semantic, and phonological similarity during within-cluster transitions, and frequency and phonological similarity during between-cluster transitions. Finally, the "switch" model uses only semantic similarity and frequency during within-cluster transitions and phonological similarity and frequency for between-cluster transitions. By default, if using run_foraging.py, if ```pdynamic``` is passed to --model flag, it will execute all three versions of the model. The phonological dynamic model was introduced in Kumar AA, Lundin NB, & Jones MN (2022)

### Switch Methods
- Norms-based (Troyer Norms)
    - the troyer norms switching method (```troyer```) adapts the categorization norms proposed by Troyer, AK, Moscovitch, M, & Winocur, G (1997), subsequently extended by our lab for analysis. Switches are predicted when moving from one subcategory of the Troyer categorization norms to another.  
- Similarity Drop
    - the similarity drop switching method (```simdrop```) is based on a switch heuristic used in Hills TT, Jones MN, Todd PM (2012) to mimic optimal foraging behavior. A switch is predicted within a series of items A,B,C,D after B if S(A,B) > S(B,C) and S(B,C) < S(C,D).
- Delta Similarity
    - the delta similarity switching method (```delta```) is a switch method proposed by Nancy Lundin in her dissertation to bypass the limits of the similarity drop switching method by allowing for consecutive switches and accounting for small dips in similarity that similarity drop may deem as a switch. This is done through the inclusion of z-scoring semantic similarity across all transitions in a list, and the inclusion of rise and fall threshold parameters to control clustering and switching via thresholding on z-score similarity values.
- Multimodal Similarity Drop
    - the multimodal similarity drop switching method (```multimodal```) is a switch method developed to include phonological similarity into the switch heuristic proposed by Hills TT, Jones MN, Todd PM (2012). It includes an alpha parameter which dictates the weighting of semantic versus phonological similarity in switching from cluster to cluster.

### Cues (Semantic, Phonological, and Frequency Matrix) Generation
Semantic Similarity Matrix Generation
- The semantic similarity matrix is generated using an underlying semantic representational model ("embeddings"). The package currently uses the word2vec model and computes pairwise cosine similarity for all items in the space (indexed by the size of embeddings).

Phonological Matrix Generation
- The phonological similarity matrix computes the pairwise normalized edit distance between the phonemic transcriptions of all items in the space (indexed by a list (```labels```). Phonemic transcriptions are obtained via CMUdict, which uses Arpabet phonemic transcriptions.

Frequency Data Generation
- A table of item frequencies is generated by obtaining raw counts for each item in the embedding labels from the Google Books Ngram Dataset via the PhraseFinder API. The raw counts are log transformed, and these log counts are the metrics used later by the models.

History Variabile Creation:
- History variables is a utility function that keeps track of lexical metrics (frequency, semantic, and phonological similarity) within a given fluency list. Specifically, the function uses underlying semantic and phonological similarity matrices as well as word frequency, and returns the similarites between consecutive items within a specific fluency list.

### Util Functions
Prepare Data Function
- The data preparation function cleans and reformats the fluency list data provided by the user. It takes in a path to data in the form of a file in which each line contains a participant ID and one response item. The function also takes an optional delimiter argument that specifies the character(s) separating the ID from the item. If none is specified, the default is a tab delimiter ('\t'). The function checks for any items outside of the vocabulary set used in the lexical metrics (OOV items). The number of OOV items is printed. The user is given the option to apply the default policy to all OOV items or review each one and decide to replace it or truncate that participant's fluency list. Items replaced by the default policy filled by the closest match found in the vocabulary. In the "review" option, users are presented with the top three closest matches and may choose one as a replacement. This should be done if the OOV item clearly corresponds to some vocabulary item that will preserve the intended semantic content, such as in the case of typos, alternate spellings, or pluralizations. When a fluency list is truncated, the OOV item and any responses after it with the same participant ID are removed. This should be done if the OOV item is not in the VFT category or there is no clear replacement that would preserve the response's semantics. After the process is complete, a count of the number of replacements and truncations applied is printed. The fluency data is then reformatted into a list of tuples, each containing the participant ID and the corresponding fluency list. 

## Development Notes

## References
    Hills, T. T, Jones, M. N, & Todd, P. M (2012). Optimal foraging in semantic memory. Psychological Review, 119(2), 431–440.
    Kumar, A. A, Lundin, N. B, & Jones, M. N (2022). Mouse-mole-vole: The inconspicuous benefit of phonology during retrieval from semantic memory. Proceedings of the Annual Meeting of the Cognitive Science Society. 
    Troyer A. K, Moscovitch M., Winocur G. (1997). Clustering and switching as two components of verbal fluency: evidence from younger and older healthy adults. Neuropsychology. Jan;11(1):138-46. 
