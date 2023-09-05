# Package Contents

### switch.py 
This contains switch functions that can be utilized to provide switch vectors to the foraging models. If new switch methods are created, they should be added here, ideally with the same format 

Currently this includes:
- Similarty Drop (switch_simdrop)
- Troyer Norms (switch_troyer)
- Multimodal Similarity drop (switch_multimodal)
- Delta Similarity (switch_delta)

### foraging.py
This contains foraging functions that an be used to perform analysis on fluency lists. If new foraging models are proposed, they should be added here, ideally with the same functional format. 

Currently this includes:
- Static Foraging Model (model_static)
- Dynamic Foraging Model (model_dynamic)
- Phonological Static Foraging Model (model_pstatic)
- Phonological Dynamic Foraging Model (model_pdynamic)

### cues.py
This contains all functions that are used to get information about different cues used in analysis (semantic, phonological, frequency). If new functions are created to develop cues, they should be added here.

Currently this includes:
- Getting history variables for running foraging models (create_history_variables)
- Getting Labels and Frequency Data (get_labels_and frequency)
- Creating Semantic Matrix from Embeddings (create_semantic_matrix)
- Phonological Matrix Functions (phonology_funcs)

### frequency.py
This contains functions pertaining to pinging the API for Google Books ngrams, and getting frequency values.

### utils.py
This contains all helper functions needed in our work. It currently contains the prepareData function which helps with making fluency data amenable to our package/analysis pipeline. 
