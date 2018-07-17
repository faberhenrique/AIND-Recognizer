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
    '''
    Passo a passo do algoritimo:
    1) Recuperar toas as sequencas
    2) Para cada palavra na sequencia => Testar em cada modelo existente
    3) Armazenar palavaras testadas e valor do melhor modelo
    '''
    probabilities = []
    guesses = []
    #pegando todas as palavras de testes
    allwords = test_set.get_all_Xlengths()
    for idW in allwords:
        wTestadas={}
        listaWords, sizeListaW = allwords[idW]
        for word,model in models.items():
            #Try cathc para descarte
            try:
                score = model.score(listaWords,sizeListaW)
                #Armazenando probabilidade no objeto
                #key(word) - > value(prob)
                wTestadas[word] = score
            except:
                #ocorreu um erro palavra testada deve ser descartada
                wTestadas[word]=float('-Inf')
                continue
        probabilities.append(wTestadas)
        guesses.append( max(wTestadas.keys(), key=(lambda k: wTestadas[k])))
    # TODO implement the recognizer
    return probabilities, guesses
