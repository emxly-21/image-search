import numpy as np

def se_image(images, dense):

    """

    Parameters
    ----------
    images: A (1,512) ndarray containing the images to be embedded.
    dense: A dense layer that embeds the image.

    Returns
    -------
    A (1,50) ndarray containing the embedded image
    """

    return dense(images)
