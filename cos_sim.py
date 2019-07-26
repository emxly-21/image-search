import mygrad as mg

def cos_sim(v1, v2):
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
    v1_sumsq = mg.sum(v1**2)
    v2_sumsq = mg.sum(v2**2)
    v1_mag = mg.sqrt(v1_sumsq)
    v2_mag = mg.sqrt(v2_sumsq)
    v1_norm = v1 / v1_mag
    v2_norm = v2 / v2_mag
    return mg.matmul(v1_norm, v2_norm)