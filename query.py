from collections import Counter

def query(database, query, k):
    """
    create function to query database and return top k images

    Parameters
    -----------
    database: np.array of images where each row corresponds to a different image's
                    semantic features
    query: string that describes the image the user is looking for

    Returns
    --------
    np.array of top k images that semantically correspond to the query
    """

    db = database
    se_query = se_text(query)
    sim = Counter()
    for se_img in db:
        sim[se_img] = sim(se_img, se_query)
    return np.array(sim.most_common(k))
