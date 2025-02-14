### The Following Guide Assumes-
1. Python is installed on your system.
2. TensorFlow CPU or GPU environment is set up.
3. You have your Kaggle API Key JSON.
4. You are in the root folder of the project when executing commands on the terminal.

<br>

### 1. Installing Packages
Install the required Python packages by executing the following command in your terminal:

```python
pip install -r src/requirements.txt
```

<br>

### 2. Downloading Dataset
[download_dataset.py](https://github.com/syntaxticsugr/breast-cancer-detection/blob/main/src/download_dataset.py) script facilitates the download and extraction of the [Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) dataset from Kaggle.

Usage:

1. Open [download_dataset.py](https://github.com/syntaxticsugr/breast-cancer-detection/blob/main/src/download_dataset.py) in your preferred Code Editor and provide `kaggle_username` and `kaggle_key` from `kaggle.json` file.

2. Execute the following command in your terminal:

    ```python
    python src/download_dataset.py
    ```

    *Depending on your internet speed and system specifications, the process may take some time.

<br>

### 3. Labelling Orignal Dataset
[label_dataset.py](https://github.com/syntaxticsugr/breast-cancer-detection/blob/main/src/label_dataset.py) script is utilized twice in this project: first for labelling the orignal dataset and then the second time for labelling the preprocessed ataset.

Usage: (Orignal Dataset):

1. Open [label_dataset.py](https://github.com/syntaxticsugr/breast-cancer-detection/blob/main/src/label_dataset.py) in your preferred Code Editor and make sure:

    - Line No. 53, 54 and 55 are commented.

    - Line No. 47, 48 and 51 are un-commented.

2. Execute the following command in your terminal:

    ```python
    python src/label_dataset.py
    ```

    *Depending on your system specifications, the process may take some time.

<br>

### 3.5. EDA
[eda.py](https://github.com/syntaxticsugr/breast-cancer-detection/blob/main/src/eda.py) script facilitates visualizing some random IDC positive and IDC Ngative images from the dataset and plotting the dataset distribution across the two classes.

Usage: (Before Preprocessing)

1. Open [eda.py](https://github.com/syntaxticsugr/breast-cancer-detection/blob/main/src/eda.py) in your preferred Code Editor and make sure: `csv_path = r'labels/labels.csv'`

2. Execute the following command in your terminal:
    ```python
    python src/eda.py
    ```

<br>

### 4. Preprocessing Dataset
[preprocess_images.py](https://github.com/syntaxticsugr/breast-cancer-detection/blob/main/src/preprocess_images.py) script facilitates the preprocessing of IDC images.

Images are processed and saved to a new directory following specific criteria outlined below:
1. Images not of shape (50, 50, 3) or identified as low contrast are discarded.
2. IDC Negative images are rotated 0 and 180 degrees.
3. IDC Positive images are rotated 0, 90, 180, 270 degrees and Flipped Horizontally.

Usage:

Execute the following command in your terminal:

```python
python src/preprocess_images.py
```

*Depending on your system specifications, the process may take some time.

<br>

### 5. Labelling Processed Dataset
[label_dataset.py](https://github.com/syntaxticsugr/breast-cancer-detection/blob/main/src/label_dataset.py) script is utilized twice in this project: first for labelling the orignal dataset and then the second time for labelling the preprocessed ataset.

Usage: (Processed Dataset):

1. Open [label_dataset.py](https://github.com/syntaxticsugr/breast-cancer-detection/blob/main/src/label_dataset.py) in your preferred Code Editor and make sure:

    - Line No. 47, 48 and 51 are commented.

    - Line No. 53, 54 and 55 are un-commented.

2. Execute the following command in your terminal:

    ```python
    python src/label_dataset.py
    ```

    *Depending on your system specifications, the process may take some time.

<br>

### 5.5. EDA
[eda.py](https://github.com/syntaxticsugr/breast-cancer-detection/blob/main/src/eda.py) script facilitates visualizing some random IDC positive and IDC Ngative images from the dataset and plotting the dataset distribution across the two classes.

Usage: (After Preprocessing)

1. Open [eda.py](https://github.com/syntaxticsugr/breast-cancer-detection/blob/main/src/eda.py) in your preferred Code Editor and make sure: `csv_path = r'labels/labels-v2/labels-v2.csv`

2. Execute the following command in your terminal:
    ```python
    python src/eda.py
    ```

<br>

### 6. Model Training
[cnn.py](https://github.com/syntaxticsugr/breast-cancer-detection/blob/main/src/cnn.py) script is responsible for training the CNN Model.
<br>
The model architecture is defined in [bcd_models.py](https://github.com/syntaxticsugr/breast-cancer-detection/blob/main/src/utils/bcd_models.py) and data loading code is implemented in [prepare_dataset.py](https://github.com/syntaxticsugr/breast-cancer-detection/blob/main/src/utils/prepare_dataset.py).

Usage:

Execute the following command in your terminal:
```python
python src/cnn.py
```
*Depending on your system specifications, the process may take some time.

<br>

### 6.1. Accuracy And Loss Graphs
[accuracy_loss_graphs.py](https://github.com/syntaxticsugr/breast-cancer-detection/blob/main/src/accuracy_loss_graphs.py) script facilitates plotting the Training and Validation Accuracy and Loss graphs for monitoring the performance of the CNN model during training.

Usage:

Execute the following command in your terminal:
```python
python src/accuracy_loss_graphs.py
```
*Run the script after desired intervals of epochs to monitor the performance of the CNN model.

<br>
