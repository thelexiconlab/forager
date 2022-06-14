import numpy as np
from scipy.optimize import fmin

from semforage import switch

class forage:
    """ 
    Class Description: 
        Forage class model to execute static and dynamic models of Semantic Foraging
    """

    def model_static(beta, freql, freqh, siml, simh):
        '''
        Static Foraging Model following proposed approach in Hills, T. T., Jones, M. N., & Todd, P. M. (2012).
            Optimal Foraging in Semantic Memory.

            Description: 
             
            Args: 
                beta (tuple, size: 2): saliency parameter(s) encoding (beta_local, beta_global).
                freql (list, size: ): frequency list containing frequency value of corresponding items.
                freqh (list, size: ): frequency history list of containing frequency value list up 
                    to current point.
                siml (list, size: ): similarity list containing frequency value of corresponding items
                simh (list, size: ): similarity history list of containing similarity value list up 
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
        Dynamic Foraging Model following proposed approach in Hills, T. T., Jones, M. N., & Todd, P. M. (2012).
            Optimal Foraging in Semantic Memory.

            Description: 

            Args: 
                beta (tuple, size: 2): saliency parameter(s) encoding (beta_local, beta_global).
                freql (list, size: ): frequency list containing frequency value of corresponding items.
                freqh (list, size: ): frequency history list of containing frequency value list up 
                    to current point.
                siml (list, size: ): similarity list containing frequency value of corresponding items
                simh (list, size: ): similarity history list of containing similarity value list up 
                    to current point.
                switchvals (list, size: ): list of switch values at given item in fluency list

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
                # If similarity dips, P of item is based on a combination of frequency and phonemic similarity
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
        Static Foraging Model following proposed approach in Hills, T. T., Jones, M. N., & Todd, P. M. (2012).
            Optimal Foraging in Semantic Memory.

            Description: 
             
            Args: 
                beta (tuple, size: 2): saliency parameter(s) encoding (beta_local, beta_global).
                freql (list, size: N): frequency list containing frequency value of corresponding items.
                freqh (list, size: N): frequency history list of containing frequency value list up 
                    to current point.
                siml (list, size: N): similarity list containing frequency value of corresponding items
                simh (list, size: N): similarity history list of containing similarity value list up 
                    to current point.
                phonl (list, size: N): phonological cue list containing frequency value of corresponding items.
                    Defaults to none, as it is optional, but used in multimodal cue.
                phonh (list, size: N): phonological cue history list of containing similarity value list up 
                    to current point. Defaults to none, as it is optional but used in multimodal cue.
                phoncue(int): Determines if phonology is used globally, locally, or both
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
                numrat = pow(freql[k],beta[0]) * pow(phonl[k],beta[2]) * pow(siml[k],beta[1])
                denrat = sum(pow(freqh[k],beta[0]) * pow(phonh[k],beta[2])* pow(simh[k],beta[1]))
                
            ct += - np.log(numrat/denrat)
        return ct

    def model_dynamic_phon(beta, freql, freqh, siml, simh, phonl, phonh, switchvals, phoncue):
        '''
        Dynamic Foraging Model following proposed approach in Hills, T. T., Jones, M. N., & Todd, P. M. (2012).
            Optimal Foraging in Semantic Memory.

            Description: 

            Args: 
                beta (tuple, size: 2): saliency parameter(s) encoding (beta_local, beta_global).
                freql (list, size: ): frequency list containing frequency value of corresponding items.
                freqh (list, size: ): frequency history list of containing frequency value list up 
                    to current point.
                siml (list, size: ): similarity list containing frequency value of corresponding items
                simh (list, size: ): similarity history list of containing similarity value list up 
                    to current point.
                switchvals (list, size: ): list of switch values at given item in fluency list
                phonl (list, size: ): phonological cue list containing frequency value of corresponding items.
                    Defaults to none, as it is optional, but used in multimodal cue.
                phonh (list, size: ): phonological cue history list of containing similarity value list up 
                    to current point. Defaults to none, as it is optional but used in multimodal cue.
                phoncue (int): Determines whether to use phonological cue globally (2), locally(1), or both(0).

            Returns: 
                ct (np.float): negative log-likelihood to be minimized in parameter fit 
        '''
        if phoncue not in [0,1,2]:
            raise Exception("To use dynamic phonological cue, you must pass a valid parameter value from possible list of values: [0,1,2]")

        ct = 0

        for k in range(0, len(freql)):
            if k == 0:
                # P of item based on frequency alone (freq of this item / freq of all items)
                numrat = pow(freql[k],beta[0])
                denrat = sum(pow(freqh[k],beta[0]))
            
            elif switchvals[k]==1: ## "dip" based on sim-drop
                # If similarity dips, P of item is based on a combination of frequency and phonemic similarity
                if phoncue != 1:
                    numrat = pow(freql[k],beta[0]) * pow(phonl[k],beta[2]) 
                    denrat = sum(pow(freqh[k],beta[0]) * pow(phonh[k],beta[2]) )
                else:
                    numrat = pow(freql[k],beta[0]) 
                    denrat = sum(pow(freqh[k],beta[0]))

            else:    
                # if not first item then its probability is based on its similarity to prev item AND frequency
                # P of item based on frequency and similarity
                if phoncue != 2:
                    numrat = pow(freql[k],beta[0])*pow(phonl[k],beta[2])*pow(siml[k],beta[1])
                    denrat = sum(pow(freqh[k],beta[0])*pow(phonh[k],beta[2])*pow(simh[k],beta[1]))
                else:
                    numrat = pow(freql[k],beta[0]) * pow(siml[k],beta[1])
                    denrat = sum(pow(freqh[k],beta[0]) * pow(simh[k],beta[1]))
                
            ct += - np.log(numrat/denrat)
        return ct

# TODO: verify this function
def optimize_model(func, switchvals, histvars, randvars=[np.random.rand(),np.random.rand(),np.random.rand()]):
    '''
    
    Args:
        func - passes one of the static or dynamic foraging functions
        switchvals - vector of switch values
        histvars - history variables
        randvars - random variables, 
    Returns: 
    '''
    r1,r2,r3 = randvars
    freql, freqh, siml, simh, phonl, phonh = histvars
    return fmin(func, [r1,r2,r3], args=(freql, freqh, siml, simh, phonl, phonh, switchvals), ftol = 0.001, full_output=True, disp=False)