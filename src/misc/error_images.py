from skimage import io
from skimage.exposure import is_low_contrast
from utils.create_directory import create_directory
from utils.save_image import save_image
import pandas as pd



def check_image_size(labels_csv, save_dir):

    save_dir = f'{save_dir}/not50x50'

    create_directory(save_dir)

    labels_csv_df = pd.read_csv(labels_csv)

    for index in range(0, len(labels_csv_df)):

        image_dir = labels_csv_df['dir'][index]
        image_name = labels_csv_df['image'][index]

        image = io.imread(f"{image_dir}/{image_name}")

        if (image.shape != (50, 50, 3)):
            save_image(image, image_name, save_dir)



def check_low_contrast_image(labels_csv, save_dir):

    save_dir = f'{save_dir}/low-contrast'

    create_directory(save_dir)

    labels_csv_df = pd.read_csv(labels_csv)

    for index in range(0, len(labels_csv_df)):

        image_dir = labels_csv_df['dir'][index]
        image_name = labels_csv_df['image'][index]

        image = io.imread(f"{image_dir}/{image_name}")

        if (is_low_contrast(image)):
            save_image(image, image_name, save_dir)



if __name__ == '__main__':

    labels_csv = r'src/labels/labels.csv'

    save_dir = r'error-images'

    check_image_size(labels_csv, save_dir)

    check_low_contrast_image(labels_csv, save_dir)
