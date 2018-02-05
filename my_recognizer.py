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
    
    #instantiate probability dictionary list and guesses list
    probabilities = []
    guesses = []
    
    testdata = test_set.get_all_Xlengths()

    #for number of items in testdata
    for i in range(0, test_set.num_items):
        
        #get data from test dictionary
        #data should be a tuple of X, lengths
        data = testdata[i] #however you access a dictionary value
        
        #instantiate probability dictionary
        pdict = {}
        
        for word, model in models.items():
            try:
                #dictionary add word, model.score(data.x, data.lengths)
                pdict[word] = model.score(data[0], data[1])
                
            except ValueError:
                pass
        
        #get maximum probability from dictionary and add associated word to guesses
        guesses.append(max(pdict.items(), key=lambda x:x[1])[0])
        probabilities.append(pdict)
        
    return (probabilities, guesses)
        
        
        
                