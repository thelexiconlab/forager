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

### Semantic Foraging
- Static Model 
    - TODO: Larry
- Dynamic Model
    - TODO: Larry
- Phonological Static Model
    - TODO: Abhilasha
- Phonological Dynamic Model
    - TODO: Abhilasha 

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
Similarity Matrix Generation
- TODO: Abhilasha

Phonological Matrix Generation
- TODO: Abhilasha

Frequency Data Generation
- TODO: Molly

History Variabile Creation:
- TODO: Abhilasha 

### Util Functions
Prepare Data Function
- TODO: Molly 


## Installation
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

## Development Notes

