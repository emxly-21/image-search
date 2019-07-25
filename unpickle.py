import pickle


def unpickle():

    """

    Unpickles the resnet18_features.pkl file and saves it into a dictionary called pfile

    Returns pfile, a dictionary containing the content of the resnet18_features.pkl
    -------

    """
    with open('resnet18_features.pkl', mode='rb') as file:
        pfile = pickle.load(file)
    return pfile
