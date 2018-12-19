import numpy as np
from skimage.transform import resize

def preprocess_input(x, size=None, BGRTranspose=True):
    """input standardizing function
    Args:
        x: numpy.ndarray with shape (H, W, C)
        size: tuple (H_new, W_new), resized input shape
    Return:
        x: numpy.ndarray
    """
    if size:
        x = resize(x, size) * 255

    if BGRTranspose:
        x = x[..., ::-1]

    return x