import numpy as np



def flip_horizontal(image):
    """
    Flip an image horizontally.

    Parameters:
    ---
        image: ndarray
            Input image as a NumPy array.

    Returns:
    ---
        image: ndarray
            Horizontally flipped image.

    """

    flipped_image = np.flip(image, axis=0)
    flipped_image = (flipped_image*255).astype(np.uint8)

    return flipped_image
