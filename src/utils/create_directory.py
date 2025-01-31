import os



def create_directory(complete_path):
    """Create a directory at the specified path if it does not exist.

    Parameters:
    ---
        complete_path: string
            The complete path of the directory to be created.

    Returns:
    ---
        None

    """

    if not os.path.exists(complete_path):

        os.makedirs(complete_path)
