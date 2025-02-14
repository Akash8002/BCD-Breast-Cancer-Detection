from skimage import io
from skimage.exposure import is_low_contrast
from utils.create_directory import create_directory
import numpy as np



def save_image(image, name, dir):
    """Save the input image with a given name to the specified directory.

    Parameters:
    ---
        image: ndarray
            Input image as a NumPy array.
        name: string
            Desired name for the saved image.
        dir: string
            Directory path where the image will be saved.

    Returns:
    ---
        None

    """

    image = (image*255).astype(np.uint8)

    if is_low_contrast(image):
        dir = r'error-images'
        print(f'Skipping Low Contrast Image --> {name}\n\n\n')

    create_directory(dir)

    io.imsave(f"{dir}/{name}", image, check_contrast=False)
