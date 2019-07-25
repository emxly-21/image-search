def sim(v1, v2):
    '''

    Calculates the cosine similarity between two vectors

    Parameters
    ----------
    v1 : vector of shape (1,M)
    v2 : vector of shape (1,M)

    Returns
    -------
    mygrad.Tensor, shape=(N, 1)
        The model outputs.
    '''
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    return v1_norm @ v2_norm