def makeDictionary(imgFeat, semFeat):
    '''

    Creates a database that maps the image features ( shape (1,512) ) to the semantic features ( shape (1,50) )

    :param imgFeat: The image descriptors
    :param semFeat: The semantic features
    :return: None
    '''
    myDict = dict(zip(imgFeat, semFeat))
    return myDict
