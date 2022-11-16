# Data

Below are descriptions of files which exist in our forager package. Primarily, the fluency list and lexical data are used in the pipeline

### Fluency Lists
The fluency list folder contains existing fluency list data that can be utilized with our pipeline. We provide two initial datasets, including the *"psych review"* dataset, which is the original dataset used in analysis in Hills TT, Jones MN, Todd PM (2012). The second dataset, is a clinical dataset from a Dartmouth Study. 

Fluency Lists should be formatted as follows:
- There are no headers in the data files
- There are two columns. The first column is the Subject ID. The second column is a fluency list item
- Each row represents a singular fluency list item from a fluency list
- Columns are delimited by a tabspace ('\t')

You are welcome to place your own fluency data in this folder, as long as they follow the specified format.

### Lexical Data
The lexical data folder contains multiple forms of lexical data. 

These include the following:
- frequencies.csv : This contains frequency data for our large set of animals. These are derived from Google Books ngrams dataset.
- phonmatrix.csv : This contains an N by N sized matrix containing the phonological similarity between items in the set of animals we have in our data. 
- semantic_embeddings.csv: This contains a set of the original word2vec embeddings for the animals items, by which the similarity matrix is calculated.
- similaritymatrix.csv:  This contains an N by N sized matrix containing semantic similarity between items in the set of animals we have in our data. 

You can provide your own lexical data too, which we provide functions for. However, it should follow similar format to the outline data. 

### Misc 
The misc folder includes miscellaneous data needed for any testing, methods, etc. For the most part, this folder is not used. 

### Norms
The norms folder includes any categorization-based norms data, such as the Troyer Norms (which we include by default). This is where you can place any categorization data that you would like to perform analysis with.
