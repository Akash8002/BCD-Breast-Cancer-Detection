from utils.image_processing.flip_image import flip_horizontal
from utils.image_processing.rotate_image import rotate_image
from utils.save_image import save_image
import os



def ro(image, image_name, output_dir):

    rotate_angles = [0, 180]

    file_name, extension = os.path.splitext(image_name)

    for angle in rotate_angles:
        rotated_image = rotate_image(image, angle)
        new_name = f'{file_name}_{angle}{extension}'
        save_image(rotated_image, new_name, output_dir)



def ro_fh(image, image_name, output_dir):

    rotate_angles = [0, 90, 180, 270]

    file_name, extension = os.path.splitext(image_name)

    hr_flipped_image = flip_horizontal(image)
    new_name = f'{file_name}_hf{extension}'
    save_image(hr_flipped_image, new_name, output_dir)

    for angle in rotate_angles:
        rotated_image = rotate_image(image, angle)
        new_name = f'{file_name}_{angle}{extension}'
        save_image(rotated_image, new_name, output_dir)
