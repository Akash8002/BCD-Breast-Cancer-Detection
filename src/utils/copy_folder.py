import shutil



def copy_folder(source_folder, destination_folder):

    shutil.copytree(source_folder, destination_folder)
