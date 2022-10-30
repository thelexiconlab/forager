import numpy as np


class forage:
    ''' 
    Class Description: 
        Forage class model to execute static and dynamic models of Semantic Foraging, including based models 
        proposed in Hills TT, Jones MN, Todd PM(2012) and  Hills TT, Todd PM, Jones MN (2015). Includes 
        phonological extensions proposed by (Insert Abhilasha's paper)

    Current Supported Methods: 
        model_static: The static model follows the implementation mentioned in "Modeling the search process" 
            section of  Hills TT, Jones MN, Todd PM(2012). As explained, the static model uses the same set of
            cues over the entire retrieval interval, effectively ignoring patchy structure of the environment.
        
        model_dynamic: The dynamic model follows the implementation mentioned in "Modeling the search process" 
            section of  Hills TT, Jones MN, Todd PM(2012). As explained, the dynamic model exploits the patchy 
            structure of memory, switching from patch to patch by changing contents of the memory probe where
            local-to-global transitions occur

        model_static_phon: The phonological static model is an extension of the static model, where the phonological
            similarity cue is included in the memory probe

        model_dynamic_phon: The phonological dynamic model is an extension of the dynamic model, where the phonological 
            similarity cue is included in the memory probe, in either solely local, global, or in both local and global
            transitions
    '''

    def model_static(beta, freql, freqh, siml, simh):
        '''
        Static Foraging Model following proposed approach in Hills, T. T., Jones, M. N., & Todd, P. M. (2012).
            Optimal Foraging in Semantic Memory.

             
            Args: 
                beta (tuple, size: 2): saliency parameter(s) encoding (beta_frequency, beta_semantic).
                freql (list, size: L): frequency list containing frequency value of corresponding items.
                freqh (list, size: L): frequency history list of containing frequency value list up 
                    to current point.
                siml (list, size: L): similarity list containing similarity value of corresponding items
                simh (list, size: L): similarity history list of containing similarity value list up 
                    to current point.

            Returns: 
                ct (np.float): negative log-likelihood to be minimized in parameter fit 
        '''
        ct = 0

        for k in range(0, len(freql)):
            if k == 0:
                # P of item based on frequency alone (freq of this item / freq of all items)
                numrat = pow(freql[k],beta[0])
                denrat = sum(pow(freqh[k],beta[0]))
            
            else:    
                # if not first item then its probability is based on its similarity to prev item AND frequency
                # P of item based on frequency and similarity
                numrat = pow(freql[k],beta[0]) * pow(siml[k],beta[1])
                denrat = sum(pow(freqh[k],beta[0]) * pow(simh[k],beta[1]))
                
            ct += - np.log(numrat/denrat)

        return ct
        
    def model_dynamic(beta, freql, freqh, siml, simh, switchvals):
        ''' 
        Dynamic Foraging Model based on Hills, T. T., Jones, M. N., & Todd, P. M. (2012).
            Optimal Foraging in Semantic Memory.

            Description: This model computes the likelihood of each given item in the fluency list based 
            on two cues: semantic similarity (local)  and frequency (global). The likelihood is computed
            based on the product of semantic similarity and frequency until a switch is detected, at which 
            point the likelihood is computed based on frequency (with the exception of the first item, whose 
            likelihood is computed based on frequency).

            Args: 
                beta (tuple, size: 2): saliency parameter(s) encoding (beta_frequency, beta_semantic).
                freql (list, size: L): frequency list obtained via create_history_variables
                freqh (list, size: L arrays of size N): frequency history list obtained via create_history_variables
                siml (list, size: L ): semantic similarity list obtained via create_history_variables
                simh (list, size: L arrays of size N ): similarity history list obtained via create_history_variables
                switchvals (list, size: L ): list of switch values for each item in the fluency list.
            Returns: 
                ct (np.float): negative log-likelihood to be minimized in parameter fit 
        '''
        ct = 0

        for k in range(0, len(freql)):
            if k == 0:
                # P of item based on frequency alone (freq of this item / freq of all items)
                numrat = pow(freql[k],beta[0])
                denrat = sum(pow(freqh[k],beta[0]))
            
            elif switchvals[k]==1: ## "dip" based on sim-drop
                # If similarity dips, P of item is based on frequency 
                numrat = pow(freql[k],beta[0]) 
                denrat = sum(pow(freqh[k],beta[0]))

            else:    
                # if not first item then its probability is based on its similarity to prev item AND frequency
                # P of item based on frequency and similarity
                numrat = pow(freql[k],beta[0]) * pow(siml[k],beta[1])
                denrat = sum(pow(freqh[k],beta[0]) * pow(simh[k],beta[1]))
                
            ct += - np.log(numrat/denrat)

        return ct

    def model_static_phon(beta, freql, freqh, siml, simh, phonl, phonh):
        '''
            Description: 
                This model is an adapted version of static foraging model proposed by Hills, T. T., Jones, M. N., & Todd, P. M. (2012)
                that incorporates phonological similarity. This model computes the likelihood of each given item 
                in the fluency list based on three cues: semantic similarity, phonological similarity, and frequency.             
            Args: 
                beta (tuple, size: 3): saliency parameter(s) encoding (beta_frequency, beta_semantic, beta_phon).
                freql (list, size: L): frequency list containing frequency value of corresponding items.
                freqh (list, size: L arrays of size N): frequency history list of containing frequency value list up to current point.
                siml (list, size: L): semantic similarity list obtained via create_history_variables
                simh (list, size: L arrays of size N): similarity history list obtained via create_history_variables
                phonl (list, size: L): phonological similarity list obtained via create_history_variables
                phonh (list, size: ): phonological cue history list obtained via create_history_variables
            Returns: 
                ct (np.float): negative log-likelihood to be minimized in parameter fit 
        '''
        ct = 0

        for k in range(0, len(freql)):
            if k == 0:
                # P of item based on frequency alone (freq of this item / freq of all items)
                numrat = pow(freql[k],beta[0])
                denrat = sum(pow(freqh[k],beta[0]))            
            else:    
                numrat = pow(freql[k],beta[0]) * pow(phonl[k],beta[2]) * pow(siml[k],beta[1])
                denrat = sum(pow(freqh[k],beta[0]) * pow(phonh[k],beta[2])* pow(simh[k],beta[1]))

            ct += - np.log(numrat/denrat)

        return ct

    def model_dynamic_phon(beta, freql, freqh, siml, simh, phonl, phonh, switchvals, phoncue):
        '''
            Description: 
                This model is an adapted version of dynamic foraging model proposed by Hills, T. T., Jones, M. N., & Todd, P. M. (2012)
                that incorporates phonological similarity in three different ways. This model computes the likelihood of each given item 
                in the fluency list based  on three cues: semantic similarity (local), phonological similarity (local/global/switch), and frequency (global). 
                Depending on the value of phoncue, the likelihood is computed based on a combination of semantic similarity,
                phonological similarity, and frequency. 
                - If phoncue is "local", then the likelihood is computed based on semantic similarity, phonological similarity, and frequency 
                    until a switch is detected at which point the likelihood is computed based on only frequency
                - If phoncue is "global", then the likelihood is computed based on all three cues during both non-switch transitions, and based on frequency
                    and phonological similarity for switch transitions.
                - If phoncue is "switch", then the likelihood is computed based on phonological similarity and frequency during switch transitions
                    and only based on semantic similarity and frequency during non-switch transitions
            
            Args: 
                beta (tuple, size: 3): saliency parameter(s) encoding (beta_frequency, beta_semantic, beta_phon).
                freql (list, size: L): frequency list containing frequency value of corresponding items.
                freqh (list, size: L arrays of size N): frequency history list of containing frequency value list up to current point.
                siml (list, size: L): semantic similarity list obtained via create_history_variables
                simh (list, size: L arrays of size N): similarity history list obtained via create_history_variables
                switchvals (list, size: L): list of switch values at given item in fluency list
                phonl (list, size: L): phonological similarity list obtained via create_history_variables
                phonh (list, size: ): phonological cue history list obtained via create_history_variables
                phoncue (str): Determines how to use phonological cue: "global", "local", or "switch"
            Returns: 
                ct (np.float): negative log-likelihood to be minimized in parameter fit 
            Raises:
                Exception: if phoncue is not one of the three options ("global", "local", or "switch")
        '''
        if phoncue not in ["global","local","switch"]:
            raise Exception("To use dynamic phonological cue, you must pass a valid parameter value from possible list of values: ['global','local','switch']")

        ct = 0

        for k in range(0, len(freql)):
            if k == 0:
                # P of item based on frequency alone (freq of this item / freq of all items)
                numrat = pow(freql[k],beta[0])
                denrat = sum(pow(freqh[k],beta[0]))
            
            elif switchvals[k]==1: # a switch has been detected
                if phoncue in ["global","switch"]:
                    numrat = pow(freql[k],beta[0]) * pow(phonl[k],beta[2]) 
                    denrat = sum(pow(freqh[k],beta[0]) * pow(phonh[k],beta[2]) )
                else:
                    numrat = pow(freql[k],beta[0]) 
                    denrat = sum(pow(freqh[k],beta[0]))

            else:    
                if phoncue in ["local","global"]:
                    numrat = pow(freql[k],beta[0])*pow(phonl[k],beta[2])*pow(siml[k],beta[1])
                    denrat = sum(pow(freqh[k],beta[0])*pow(phonh[k],beta[2])*pow(simh[k],beta[1]))
                else:
                    numrat = pow(freql[k],beta[0]) * pow(siml[k],beta[1])
                    denrat = sum(pow(freqh[k],beta[0]) * pow(simh[k],beta[1]))
                
            ct += - np.log(numrat/denrat)

        return ct



    def model_static_report(beta, freql, freqh, siml, simh):
        '''
        Static Foraging Model following proposed approach in Hills, T. T., Jones, M. N., & Todd, P. M. (2012).
            Optimal Foraging in Semantic Memory.

             
            Args: 
                beta (tuple, size: 2): saliency parameter(s) encoding (beta_frequency, beta_semantic).
                freql (list, size: L): frequency list containing frequency value of corresponding items.
                freqh (list, size: L): frequency history list of containing frequency value list up 
                    to current point.
                siml (list, size: L): similarity list containing frequency value of corresponding items
                simh (list, size: L): similarity history list of containing similarity value list up 
                    to current point.

            Returns: 
                ct (np.float): negative log-likelihood to be minimized in parameter fit 
        '''
        ct = 0
        nll_vec = []
        for k in range(0, len(freql)):
            if k == 0:
                # P of item based on frequency alone (freq of this item / freq of all items)
                numrat = pow(freql[k],beta[0])
                denrat = sum(pow(freqh[k],beta[0]))
            
            else:    
                # if not first item then its probability is based on its similarity to prev item AND frequency
                # P of item based on frequency and similarity
                numrat = pow(freql[k],beta[0]) * pow(siml[k],beta[1])
                denrat = sum(pow(freqh[k],beta[0]) * pow(simh[k],beta[1]))
                
            ct += - np.log(numrat/denrat)
            nll_vec.append(- np.log(numrat/denrat))

        return ct, nll_vec
        
    def model_dynamic_report(beta, freql, freqh, siml, simh, switchvals):
        ''' 
        Dynamic Foraging Model based on Hills, T. T., Jones, M. N., & Todd, P. M. (2012).
            Optimal Foraging in Semantic Memory.

            Description: This model computes the likelihood of each given item in the fluency list based 
            on two cues: semantic similarity (local)  and frequency (global). The likelihood is computed
            based on the product of semantic similarity and frequency until a switch is detected, at which 
            point the likelihood is computed based on frequency (with the exception of the first item, whose 
            likelihood is computed based on frequency).

            Args: 
                beta (tuple, size: 2): saliency parameter(s) encoding (beta_frequency, beta_semantic).
                freql (list, size: L): frequency list obtained via create_history_variables
                freqh (list, size: L arrays of size N): frequency history list obtained via create_history_variables
                siml (list, size: L ): semantic similarity list obtained via create_history_variables
                simh (list, size: L arrays of size N ): similarity history list obtained via create_history_variables
                switchvals (list, size: L ): list of switch values for each item in the fluency list.
            Returns: 
                ct (np.float): negative log-likelihood to be minimized in parameter fit 
        '''
        ct = 0
        nll_vec = []
        for k in range(0, len(freql)):
            if k == 0:
                # P of item based on frequency alone (freq of this item / freq of all items)
                numrat = pow(freql[k],beta[0])
                denrat = sum(pow(freqh[k],beta[0]))
            
            elif switchvals[k]==1: ## "dip" based on sim-drop
                # If similarity dips, P of item is based on frequency 
                numrat = pow(freql[k],beta[0]) 
                denrat = sum(pow(freqh[k],beta[0]))

            else:    
                # if not first item then its probability is based on its similarity to prev item AND frequency
                # P of item based on frequency and similarity
                numrat = pow(freql[k],beta[0]) * pow(siml[k],beta[1])
                denrat = sum(pow(freqh[k],beta[0]) * pow(simh[k],beta[1]))
                
            ct += - np.log(numrat/denrat)
            nll_vec.append(- np.log(numrat/denrat))

        return ct, nll_vec

    def model_static_phon_report(beta, freql, freqh, siml, simh, phonl, phonh):
        '''
            Description: 
                This model is an adapted version of static foraging model proposed by Hills, T. T., Jones, M. N., & Todd, P. M. (2012)
                that incorporates phonological similarity. This model computes the likelihood of each given item 
                in the fluency list based on three cues: semantic similarity, phonological similarity, and frequency.             
            Args: 
                beta (tuple, size: 3): saliency parameter(s) encoding (beta_frequency, beta_semantic, beta_phon).
                freql (list, size: L): frequency list containing frequency value of corresponding items.
                freqh (list, size: L arrays of size N): frequency history list of containing frequency value list up to current point.
                siml (list, size: L): semantic similarity list obtained via create_history_variables
                simh (list, size: L arrays of size N): similarity history list obtained via create_history_variables
                phonl (list, size: L): phonological similarity list obtained via create_history_variables
                phonh (list, size: ): phonological cue history list obtained via create_history_variables
            Returns: 
                ct (np.float): negative log-likelihood to be minimized in parameter fit 
        '''
        ct = 0
        nll_vec = []
        for k in range(0, len(freql)):
            if k == 0:
                # P of item based on frequency alone (freq of this item / freq of all items)
                numrat = pow(freql[k],beta[0])
                denrat = sum(pow(freqh[k],beta[0]))            
            else:    
                numrat = pow(freql[k],beta[0]) * pow(phonl[k],beta[2]) * pow(siml[k],beta[1])
                denrat = sum(pow(freqh[k],beta[0]) * pow(phonh[k],beta[2])* pow(simh[k],beta[1]))

            ct += - np.log(numrat/denrat)
            nll_vec.append(- np.log(numrat/denrat))

        return ct, nll_vec

    def model_dynamic_phon_report(beta, freql, freqh, siml, simh, phonl, phonh, switchvals, phoncue):
        '''
            Description: 
                This model is an adapted version of dynamic foraging model proposed by Hills, T. T., Jones, M. N., & Todd, P. M. (2012)
                that incorporates phonological similarity in three different ways. This model computes the likelihood of each given item 
                in the fluency list based  on three cues: semantic similarity (local), phonological similarity (local/global/switch), and frequency (global). 
                Depending on the value of phoncue, the likelihood is computed based on a combination of semantic similarity,
                phonological similarity, and frequency. 
                - If phoncue is "local", then the likelihood is computed based on semantic similarity, phonological similarity, and frequency 
                    until a switch is detected at which point the likelihood is computed based on only frequency
                - If phoncue is "global", then the likelihood is computed based on all three cues during both switch and non-switch transitions
                - If phoncue is "switch", then the likelihood is computed based on phonological similarity and frequency during switch transitions
                    and only based on semantic similarity and frequency during non-switch transitions
            
            Args: 
                beta (tuple, size: 3): saliency parameter(s) encoding (beta_frequency, beta_semantic, beta_phon).
                freql (list, size: L): frequency list containing frequency value of corresponding items.
                freqh (list, size: L arrays of size N): frequency history list of containing frequency value list up to current point.
                siml (list, size: L): semantic similarity list obtained via create_history_variables
                simh (list, size: L arrays of size N): similarity history list obtained via create_history_variables
                switchvals (list, size: L): list of switch values at given item in fluency list
                phonl (list, size: L): phonological similarity list obtained via create_history_variables
                phonh (list, size: ): phonological cue history list obtained via create_history_variables
                phoncue (str): Determines how to use phonological cue: "global", "local", or "switch"
            Returns: 
                ct (np.float): negative log-likelihood to be minimized in parameter fit 
            Raises:
                Exception: if phoncue is not one of the three options ("global", "local", or "switch")
        '''
        if phoncue not in ["global","local","switch"]:
            raise Exception("To use dynamic phonological cue, you must pass a valid parameter value from possible list of values: ['global','local','switch']")

        ct = 0
        nll_vec = []

        for k in range(0, len(freql)):
            if k == 0:
                # P of item based on frequency alone (freq of this item / freq of all items)
                numrat = pow(freql[k],beta[0])
                denrat = sum(pow(freqh[k],beta[0]))
            
            elif switchvals[k]==1: # a switch has been detected
                if phoncue in ["global","switch"]:
                    numrat = pow(freql[k],beta[0]) * pow(phonl[k],beta[2]) 
                    denrat = sum(pow(freqh[k],beta[0]) * pow(phonh[k],beta[2]) )
                else:
                    numrat = pow(freql[k],beta[0]) 
                    denrat = sum(pow(freqh[k],beta[0]))

            else:    
                if phoncue in ["local","global"]:
                    numrat = pow(freql[k],beta[0])*pow(phonl[k],beta[2])*pow(siml[k],beta[1])
                    denrat = sum(pow(freqh[k],beta[0])*pow(phonh[k],beta[2])*pow(simh[k],beta[1]))
                else:
                    numrat = pow(freql[k],beta[0]) * pow(siml[k],beta[1])
                    denrat = sum(pow(freqh[k],beta[0]) * pow(simh[k],beta[1]))
                
            ct += - np.log(numrat/denrat)
            nll_vec.append(- np.log(numrat/denrat))

        return ct, nll_vec
