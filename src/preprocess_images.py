from concurrent.futures import ThreadPoolExecutor
from skimage import io
from skimage.exposure import is_low_contrast
from utils.preprocessing_pipelines import ro, ro_fh
from utils.save_image import save_image
import pandas as pd
import time



def check_shape(image):
    if (image.shape == (50, 50, 3)):
        return True
    else:
        return False



def preprocess_images(labels_csv, start_index, pipelines, output_dir):

    print("\n\n")

    labels_csv_df = pd.read_csv(labels_csv)

    error_images = []

    total_time = 0
    items_remaining = (len(labels_csv_df)-start_index)
    print(f"\033[A\033[AProcessed: 0        Remaining: {items_remaining}        Ellapsed Time: 0        Estimated Remaining Time: --:--:--        \n")

    with ThreadPoolExecutor() as executor:

        for index in range(start_index, len(labels_csv_df)):

            start_time = time.time()

            image_dir = labels_csv_df['dir'][index]
            image_name = labels_csv_df['image'][index]
            idc = labels_csv_df['idc'][index]

            try:
                image = io.imread(f"{image_dir}/{image_name}")

                if (check_shape(image) and (not is_low_contrast(image))):
                    executor.submit(pipelines[idc], image, image_name, output_dir)

                else:
                    error_images.append(image_name)
                    print(f'Skipping Image --> {image_name}\n\n\n')
                    save_image(image, image_name, r'error-images')

            except:
                error_images.append(image_name)

            total_time += (time.time() - start_time)
            items_remaining -= 1
            print(f"\033[A\033[AProcessed: {(index+1)-start_index}        Remaining: {items_remaining}        Ellapsed Time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}        Estimated Remaining Time: {time.strftime('%H:%M:%S', time.gmtime((items_remaining*(total_time/((index+1)-start_index)))))}        \n")

        print(f'{error_images}\nTotal: {len(error_images)}')



if __name__ == '__main__':

    # Required Paths
    labels_csv = r'labels/labels.csv'
    output_dir = r'dataset/processed-dataset'

    # Start Index To PreProcess Images From
    # Useful In Case When Need To Resume Preprocessing From Certain Index
    start_index = 0

    # idc: pipeline
    pipelines = {
        0: ro,
        1: ro_fh
    }

    preprocess_images(labels_csv, start_index, pipelines, output_dir)
