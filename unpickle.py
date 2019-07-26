import pickle


def unpickle():

    """

    Unpickles the resnet18_features.pkl file and saves it into a dictionary called pfile

    Returns pfile, a dictionary where pfile[key = image_id] returns the (1,512) image features
    -------

    """
    with open('resnet18_features.pkl', mode='rb') as file:
        pfile = pickle.load(file)
    return pfile
