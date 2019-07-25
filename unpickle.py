import pickle


def unpickle():
    '''

    Unpickles the resnet18_features.pkl file to a dictionary and returns the dictionary.

    :return: pFile: a dictionary with is obtained by unpickling the resnet19_features.pkl file
    '''
    with open('resnet18_features.pkl', mode='rb') as file:
        pFile = pickle.load(file)
    return pFile
