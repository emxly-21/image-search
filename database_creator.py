def makeDictionary(imgFeat, semFeat):

    """

    Creates a database that maps the image features ( shape (1,512) ) to the semantic features ( shape (1,50) )

    Parameters
    ----------
    imgFeat: The image descriptors
    semFeat: The semantic features

    Returns
    -------
    myDict: A dictionary containing imgFeat as its keys and semFeat as its values

    """

    myDict = dict(zip(imgFeat, semFeat))
    return myDict
