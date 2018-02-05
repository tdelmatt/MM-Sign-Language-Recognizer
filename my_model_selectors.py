import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        #raise NotImplementedError
        
        #this list stores tuples of format (score, model)
        #modelscores = []
        
        X = self.X
        #print(X)
        lengths = self.lengths
        #for all models in self.
        
        for n in range(self.min_n_components, (self.max_n_components + 1)):
            #okay so how do I train the model, do I need to train it?
            try:
                #warnings.filterwarnings("ignore", category=DeprecationWarning)
                model = GaussianHMM(n, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(X, lengths)
            
                score =  -2 * model.score(X, lengths) + math.log(len(X)) * (n*n + 2 * n * len(X[0]) - 1)
            except ValueError as e:
                #print(e)
                pass
            
            try:
                score
            except:
                #print("score does not exist")
                score = float('-inf')
            
            if n == self.min_n_components or score < maxscore[0]:
                maxscore = (score, model)
            #create model with n components.  
            #obtain model score
            
        return maxscore[1]
            
            #maxmodel = model with highest score
        
        #return maxmodel
        
        #p is the number of data points, N is the number of parameters
        #           log L                          log(data points)                    parameters
        


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        X = self.X
        #print(X)
        lengths = self.lengths
        #for all models in self.
        
        for n in range(self.min_n_components, (self.max_n_components + 1)):
            #okay so how do I train the model, do I need to train it?
            try:
                #warnings.filterwarnings("ignore", category=DeprecationWarning)
                model = GaussianHMM(n, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(X, lengths)
                
            
            
                wordsnoti = [v for k, v in self.hwords.items() if k != self.this_word]
                
                sumscorenoti = sum(model.score(hword[0], hword[1]) for hword in wordsnoti)
                
                #          log(P(X(i))             1/(M-1)SUM(log(P(X(all but i))    (M-1) = len(wordsnoti) because we have already removed 1 word from the list
                score = model.score(X, lengths) - (1/(len(wordsnoti))) * sumscorenoti
                
            except ValueError as e:
                #print(e)
                pass
                
            try:
                score
            except:
                #print("score does not exist")
                score = float('-inf')
            
            
            if n == self.min_n_components or score > maxscore[0]:
                maxscore = (score, model)
            #create model with n components.  
            #obtain model score
            
        return maxscore[1]


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        #raise NotImplementedError
        #X = self.X
        #print(X)
        #lengths = self.lengths
        n_splits = min(len(self.lengths), 3)
        
        if n_splits <= 1:
            return self.base_model(3)
            
        split_method = KFold(n_splits)
        splits = split_method.split(self.sequences)
        
        
        for n in range(self.min_n_components, (self.max_n_components + 1)):
            
            #I guess here we could iterate through all of the cross validation
            #data training on the training set, and then scoring on the test set
                        
            sumlog = 0

            failedmodeltrains = 0
            #count = 0
            for cv_trainidx, cv_testidx in split_method.split(self.sequences):
                #count += 1
                
                #traindata = get train data
                trainsequences = [self.sequences[ind] for ind in cv_trainidx]
                traindata, trainlengths = combine_sequences(range(len(trainsequences)), trainsequences)
                
                #testdata = get test data
                testsequences = [self.sequences[ind] for ind in cv_testidx]
                testdata, testlengths = combine_sequences(range(len(testsequences)), testsequences)
                
                #X, L = combinesequences(traindata)
                
                #xtest ltest = combinesequences(testdata)
                
                try:
                    #model = train model
                    model = GaussianHMM(n, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(traindata, trainlengths)
                    
                    #assuming selector CV does not return there will be two things that need to be done
                    #tempscore should be assigned to model score
                    #if tempscore does not exist, one less model was trained, so we should
                    #create a variable untrained models and subtract it from n_splits
                   
                    
                    #sum
                    tempscore = model.score(testdata, testlengths)
                    #sumlog += model.score(testdata, testlengths)
                    
                except ValueError as e:
                    #print(e)
                    pass
                
                try:
                    tempscore
                except:
                    print("tempscore not initialized properly")
                    tempscore = 0
                    failedmodeltrains += 1
            
                sumlog += tempscore
            #print("does count == n_splits")
            #print(count == n_splits)
            score = sumlog/(n_splits - failedmodeltrains)
            
            if n == self.min_n_components or score > maxscore[0]:
                
                try:
                #warnings.filterwarnings("ignore", category=DeprecationWarning)
                    model = GaussianHMM(n, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                except ValueError:
                    pass
                
                maxscore = (score, model)
            
            
      
            
        return maxscore[1]
        
        
        
