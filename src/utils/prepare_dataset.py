import tensorflow as tf



def read_image(image_path, dr):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    return(image, dr)

def normalize(image, dr):
    return image/255, dr

def configure_for_performance(dataset, batch_size):
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset



def prepare_dataset(df, batch_size):
    df['full_path'] = df['dir'] + '/' + df['image']
    dataset = tf.data.Dataset.from_tensor_slices((df['full_path'], df['idc']))
    dataset = dataset.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = configure_for_performance(dataset, batch_size)
    return dataset
