import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # TODO implement the recognizer

    for (X, lengths) in test_set.get_all_Xlengths().values():
        word_probabilities ={}
        best_score = float('-inf')
        best_guess = None

        for train_word, model in models.items():
            try:
                score = model.score(X, lengths)
                word_probabilities[train_word] = score
                if score > best_score:
                    best_score, best_guess = score, train_word
            except:
                word_probabilities[train_word] = float('-inf')
        guesses.append(best_guess)
        probabilities.append(word_probabilities)

    return probabilities, guesses


