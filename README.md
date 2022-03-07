# semforage

Semantic Foraging methods for estimating semantic memory search on semantic fluency data

## Methods and Usage

### Semantic Foraging
- Static Model 
- Dynamic Model
### Switching Methods
- Norms-based (Troyer Norms, Subject Determined Norms)
- Similarity Drop
- Delta Similarity
- Enhanced Similarity Drop
- Semantic Scent (TBD)

### Similarity Matrix, Phonology Matrix, and Frequency Data Generation 
Similarity Matrix Generation
    - Psychological Review Word2Vec Matrix
    - Constructing Dynamic Similarity Matrices

Phonological Matrix Generation
    - TODO:

Frequency Data Generation
    - Google News Corpus Frequency Data
    - Subtlex Frequency Data? 

## Installation

Requirements:
- numpy
- scipy

Requires Python 3.8 

- TODO: Add versions for numpy, scipy
- TODO: Add finalized instructions for general installation on any machine

## Development Notes

### Version Control and Management
- https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request

The simpler approach here is learning how to create an arbitrary pull request.

The idea is as follows:

1) Create a new fork/branch : https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository
    a) This will allow you to develop independently from the main branch, which can always be reverted to if something goes wrong on your individual branch

2) Submit a Pull Request
    a) Pull Requests can either be done from command line or from the web browser. I recommend reading the documentation linked above regarding creating a pull request. From the command line, it can be as easy as ` gh pr create `.
    b) You can perform the same action via your web browser by going to the repository, choose the branch your code was committed to (do not push), and open a pull request.

Either method of performing pull request is fairly simple.

When there is an update in the codebase (main branch) that you would like to bring to your local, you can do so by the command line command:
``` git pull origin main ```

### Unit Testing
- See: https://realpython.com/python-testing/

The idea behind unit testing is to compose simple tests to evaluate common conditions and edge cases which may come up in the usage of a particular method. This is helpful for both decoding, and producing verifiable behavior. 

In the case of the Semantic Foraging project, often there is use of existing data structures, such as similarity matrix or frequency list, in many of the methods. It may not be necessary to utilize them in testing methods. You can think of simple abstractions to the applications of methods, with very reliable outputs to evaluate your method. 

For the purpose of running unit testing, pytest is recommended. It is simple, and easy to use, and gives detailed results. There are ways to "upgrade" testing capabilities, but these should be all you need.

Every time you develop a new method, you should think of the

### Local editable package

Utilize `pip install -e PROJECT_PACKAGE_NAME` command to install it as an
"editable" package that does not require reinstallation after changes. Other external libraries can be installed here as well.

One thing to note here, is that you would perform pip install in the directory that this git repository is located in, not within the git repository itself. 

For example, my git repository is located in `../larry/semforage/ `. I execute `pip install -e semforage` in `../larry/`, which will allow you to pip install your local editable repository. 

Every time you update your local codebase, the semforage package will have every new functionality added enabled.
