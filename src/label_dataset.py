from sklearn.model_selection import train_test_split
from utils.create_directory import create_directory
import os
import pandas as pd



def label_dataset(labels_dir, dataset_dir, labels_filename, tvt_split):

    create_directory(labels_dir)

    labels_df_rows = []

    for (dirpath, _, filenames) in os.walk(dataset_dir, topdown=True):

        for filename in filenames:

            idc = 0 if "class0" in filename else 1

            labels_df_rows.append({"dir":dirpath, "image":filename, "idc":idc})

    labels_df = pd.DataFrame(data=labels_df_rows)

    if tvt_split[0]:

        create_directory(f'{labels_dir}/{labels_filename}')

        train_size, validation_size, test_size = tvt_split[1]

        train_df, test_df = train_test_split(labels_df, train_size=train_size, shuffle=True)
        val_df, test_df = train_test_split(test_df, test_size=(test_size / (validation_size + test_size)), shuffle=True)

        labels_df.to_csv(f'{labels_dir}/{labels_filename}/{labels_filename}.csv', index=False)
        train_df.to_csv(f'{labels_dir}/{labels_filename}/{labels_filename}-train.csv', index=False)
        val_df.to_csv(f'{labels_dir}/{labels_filename}/{labels_filename}-val.csv', index=False)
        test_df.to_csv(f'{labels_dir}/{labels_filename}/{labels_filename}-test.csv', index=False)

    else:
        labels_df.to_csv(f'{labels_dir}/{labels_filename}.csv', index=False)



if __name__ == "__main__":

    labels_dir = r'labels'

    dataset_dir = r'dataset/kaggle/breast-histopathology-images/IDC_regular_ps50_idx5'
    labels_filename = 'labels'
    # Weather to split the labels into train, validation and test part
    # (Boolean, (train_split, validation_split, test_split))
    tvt_split = (False, (0.7, 0.15, 0.15))

    # dataset_dir = r'dataset/processed-dataset'
    # labels_filename = 'labels-v2'
    # tvt_split = (True, (0.7, 0.15, 0.15))

    label_dataset(labels_dir, dataset_dir, labels_filename, tvt_split)
