import numpy as np
def word_query(database, query, k):
    """
    create function to query database and return top k images

    Parameters
    -----------
    database: np.array of images where each row corresponds to a different image's
                    semantic features
    query: string that describes the image the user is looking for
    k: number of top images to return

    Returns
    --------
    np.array of top k images that semantically correspond to the query
    """

    db = database
    se_query = se_text(query)
    similarities = []
    for se_img in db:
        similarities = similarities.append(sim(se_img, se_query))
    similarities = sorted(similarities)
    return np.array(similarities[-k::])


def image_query(database, query, k, semantic):
    """

    :param database: np.array of images where each row corresponds to a
                        different image's semantic features
    :param query: np.array(1,512) image of features
    :param k: number of images to return
    :param semantic: boolean of if the images should be similar by image features or semantics
    :return: np.array of top k images that correspond to the image query
    """

    if semantic:
        database = se_image(database)
        query = se_image(query)
    similarities = []
    for img in database:
        similarities = similarities.append(sim(img, query))
    similarities = sorted(similarities)
    return np.array(similarities[-k::])