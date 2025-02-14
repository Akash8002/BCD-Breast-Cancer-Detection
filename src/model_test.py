from keras.saving import load_model
from utils.prepare_dataset import prepare_dataset
import numpy as np
import pandas as pd



def test_model(model_save_dir, model_name, batch_size, test_csv):

    test_df = pd.read_csv(test_csv)

    test_ds = prepare_dataset(test_df, batch_size)

    model = load_model(f'{model_save_dir}/{model_name}/{model_name}.keras')

    predictions = model.predict(test_ds)

    pred_rows = []

    for prediction in predictions:

        pred_rows.append(np.argmax(prediction))

    pred_df = pd.DataFrame(pred_rows, columns=['pred'])

    pred_df[['image', 'idc']] = test_df[['image', 'idc']]

    pred_df = pred_df[['image', 'idc', 'pred']]

    pred_df.to_csv(f'{model_save_dir}/{model_name}/{model_name}-pred.csv', index=False)



if __name__ == '__main__':

    model_save_dir = r'saved-models'
    model_name = "bcd-final"

    test_csv = r'labels/labels-v2/labels-v2-test.csv'

    batch_size = 32

    test_model(model_save_dir, model_name, batch_size, test_csv)
