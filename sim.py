import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def norm(tens):
    """

    A function that takes a tensor 'tens' and returns its magnitude (similar to np.linalg.norm)

    Parameters
    ----------
    tens: A mygrad tensor.

    Returns
    -------
    The magnitude of tens
    """
    pass


def sim(v1, v2):
    '''

    Calculates the cosine similarity between two vectors

    Parameters
    ----------
    v1 : vector of shape (M,)
    v2 : vector of shape (M,)

    Returns
    -------
    mygrad.Tensor, shape=(N, 1)
        The model outputs.
    '''
    # v1_norm = v1 / np.linalg.norm(v1)
    # v2_norm = v2 / np.linalg.norm(v2)
    # return np.dot(v1_norm, v2_norm)
    return cosine_similarity(v1, v2)