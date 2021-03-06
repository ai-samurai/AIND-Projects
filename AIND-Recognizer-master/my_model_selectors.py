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
        return None
        #raise NotImplementedError

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

    def score(self, num_states):
        model = self.base_model(num_states)
        logL = model.score(self.X, self.lengths)
        logN = np.log(len(self.X))
        # Ref - https://ai-nd.slack.com/files/ylu/F4S90AJFR/number_of_parameters_in_bic.txt
        p = (num_states**2) + 2*(len(self.X[0])*(num_states))-1
        return (-2*(logL) + p*logN), model

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        # Ref - https://discussions.udacity.com/t/hmmlearn-valueerror-rows-of-transmat--must-sum-to-1-0/229995/5
        try:
            best_score = float("inf")
            best_model = None
            for num_states in range(self.min_n_components, self.max_n_components+1):
                score, model = self.score(num_states)
                if score < best_score:
                    best_score = score
                    best_model = model
            return best_model
        except:
            return self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    #def dic_score(self, num_states, mean):
    #    model = self.base_model(num_states)
    #    return model.score(self.X, self.lengths) - mean


    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            best_score = float("-inf")
            best_model = None
            for num_states in range(self.min_n_components, self.max_n_components):
                mean_score_array = []
                model = self.base_model(num_states)
                for word, (X, lengths) in self.hwords.items():
                    if word != self.this_word:
                        mean_score_array.append(model.score(X, lengths))
                mean=np.mean(mean_score_array)
                score = model.score(self.X, self.lengths) - mean
                if score > best_score:
                    best_score = score
                    best_model = model
            return best_model
        except:
            return self.base_model(self.n_constant)

        # TODO implement model selection based on DIC scores
        #raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def score(self, num_states):

        scores = []
        split_method = KFold(n_splits=2)

        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
            X_train, l_train = combine_sequences(cv_train_idx, self.sequences)
            model=GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(X_train, l_train)
            (X_test,L_test) = combine_sequences(cv_test_idx, self.sequences)
            scores.append(model.score(X_test,L_test))
        return np.mean(scores)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_score = float("-inf")
        best_states = 3

        try:
            for num_states in range(self.min_n_components, self.max_n_components+1):
                score = self.score(num_states)
                if score > best_score:
                    best_score = score
                    best_states = num_states
            return self.base_model(best_states)
        except:
            return self.base_model(self.n_constant)
