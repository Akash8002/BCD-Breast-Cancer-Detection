from keras import Sequential
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.src.optimizers.adam import Adam
from keras.utils import plot_model



def get_model(learning_rate, model_save_dir, model_name):

    model = simp(learning_rate)

    plot_model(model,
               to_file = f'{model_save_dir}/{model_name}/{model_name}.png',
               show_shapes = True,
               show_dtype= True,
               show_layer_names = True,
               show_layer_activations= True,
               show_trainable= True,
               expand_nested = False)

    return model



def simp(learning_rate):

    model = Sequential(
        [
            Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(50, 50, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            Dropout(0.3),
            Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            Flatten(),
            Dense(128, activation='relu', kernel_initializer='he_uniform'),
            BatchNormalization(),
            Dense(64, activation='relu', kernel_initializer='he_uniform'),
            BatchNormalization(),
            Dense(64, activation='relu', kernel_initializer='he_uniform'),
            Dropout(0.3),
            Dense(24, activation='relu', kernel_initializer='he_uniform'),
            Dense(2, activation='softmax'),
        ]
    )

    model.compile(optimizer=Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
