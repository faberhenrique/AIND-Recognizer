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
                print("model created for {} with {} states".format(
                    self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(
                    self.this_word, num_states))
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
        # warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        # BIC = −2 log L + p log N
        # Loop para variar o numero de estados da Hmm
        BestScore = float('Inf')
        for nStates in range(self.min_n_components, self.max_n_components+1):
            try:
                model = GaussianHMM(n_components=nStates, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                l = model.score(self.X, self.lengths)
            except Exception as e: 
                print(e)
                l=float('Inf')
                pass

            # P =  n*n + 2*n*d-1 (n= n_componenets and d=n_features) --> katie_tiwari
            p = nStates*nStates+2*nStates*model.n_features-1
            bic = -2 * l+p*nStates
            if(bic < BestScore):
                BestScore = bic
                BestModel = model
        return BestModel


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        #     DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
        BestScore = -float('Inf')
        M = len(self.hwords)
        BestModel=None
        for nStates in range(self.min_n_components, self.max_n_components+1):
            try:
                model = GaussianHMM(n_components=nStates, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                l = model.score(self.X, self.lengths)
                   # Reiniciando a soma das probablidades
                sump = 0
                '''
                SUM(log(P(X(all but i)) essa parte necessita da soma de todas as probabilidades do modelo. 
                Esse valor é constante pois de baseia no didcionario, contudo, como o nº de componentes varia deve ser 
                calculado a cada interaçaão do for de states. 
                Dentro da variavel hwords esta armazenado o dicionario. 
                '''
                for word,(wordX, wordLengths) in self.hwords.items():
                    if word == self.this_word:
                        pword = model.score(wordX, wordLengths)
                        sump += pword
            except Exception as e: 
                print(e,self.this_word)
                l = -float('Inf')
                pword=0
                pass

            term2 = (1/(M-1)) * (pword)
            dic = l - term2
            
            if(dic > BestScore):
                BestScore = dic
                BestModel = model
        return BestModel 


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # Saving instance as notebook
        meanScore = 0
        maxScore = -float('Inf')
        aux = 0
        # Loop para variar o numero de estados da Hmm
        for nStates in range(self.min_n_components, self.max_n_components+1):
            if len(self.sequences) > 1:
                split_method = KFold(min(len(self.sequences), 3))

                # Cross Validation para cada sequencia recuperando the fold indices e testes
                for trainFold, testFold in split_method.split(self.sequences):
                    # Adicionando Seq --> Similar ao alg genetico
                    nTest, lengthTest = combine_sequences(
                        testFold, self.sequences)
                    nTrain, lengthTrain = combine_sequences(
                        trainFold, self.sequences)
                    # Calculando Model
                    #model = self.base_model(nStates).fit(nTrain, lengthTrain)
                    try:
                        model = GaussianHMM(n_components=nStates, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(nTrain, lengthTrain)
                        meanScore = meanScore+model.score(nTest, lengthTest)
                    except:
                        pass
                    aux += 1
                if maxScore < (meanScore/aux):
                    maxScore = meanScore/aux
                    meanScore = 0
                    aux = 0
                    bestModel = model
                else:
                    meanScore = 0
                    aux = 0
            else:
                    # Loop para variar o numero de estados da Hmm
                for nStates in range(self.min_n_components, self.max_n_components+1):
                    try:
                        model= GaussianHMM(n_components=nStates, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                        meanScore = meanScore+model.score(self.X, self.lengths)
                    except:
                        pass
                    aux += 1
                    if maxScore < (meanScore/aux):
                        maxScore = meanScore/aux
                        meanScore = 0
                        aux = 0
                        bestModel = model
                    else:
                        meanScore = 0
                        aux = 0
       # print('\nbest model {}, best model number of components {}'.format(
        #    bestModel, bestModel.n_components))

        return bestModel
