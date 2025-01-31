from skimage.transform import rotate



def rotate_image(image, angle):
    """Rotate the input image by a specified angle.

    Parameters:
    ---
        image: ndarray
            Input image as a NumPy array.
        angle: float
            Rotation angle in degrees.

    Returns:
    ---
        image: ndarray
            Rotated image.

    """

    rotated_image = rotate(image, angle)

    return rotated_image
