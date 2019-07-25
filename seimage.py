import numpy as np


def se_image(images, w, b):

    """

    Parameters
    ----------
    images: A (1,512) array containing the images to be embedded.
    w: The weights of the model
    b: The biases of the model

    Returns
    -------
    A (1,50) ndarray containing the embedded image
    """

    return np.dot(images, w) + b
