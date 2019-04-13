import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential, load_model

from config import Config
from modules import globals

# Classifier settings
# ====================================================#
row = 23
col = 13


class Classifier:
    # init empty arrays, as defaults
    training_data = np.empty([0, 1, row, col])  # XS Example array to be trained
    training_labels = np.empty([0, Config.NUM_CLASSES])  # YS Label array

    def __init__(self):
        # Change to True if wake word is trained in the loaded model
        globals.HAS_BEEN_TRAINED = False
        self.load_BG_examples()
        self.graph = tf.get_default_graph()

        if not Config.LOAD_MODEL:
            self.model = Sequential()
            self.model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(1, row, col), data_format='channels_first'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Dropout(rate=0.2))
            self.model.add(Flatten())
            self.model.add(Dense(units=Config.DENSE_UNITS, activation='relu'))
            self.model.add(Dropout(rate=0.5))
            self.model.add(Dense(units=Config.NUM_CLASSES, activation='sigmoid'))
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        else:
            self.model = load_model('data/previous_model.h5')  # load the last model saved to continue.
            self.model._make_predict_function()
            self.graph = tf.get_default_graph()
            print("just loaded model")

    # Load or reset training examples for class 0 (background sounds)
    def load_BG_examples(self):
        if Config.LOAD_MODEL:
            self.training_data = np.load('data/background_sound_examples.npy')
            self.training_labels = np.load('data/background_sound_labels.npy')
        else:
            self.training_data = np.empty([0, 1, row, col])  # XS Example array to be trained
            self.training_labels = np.empty([0, Config.NUM_CLASSES])  # YS Label array

        globals.BG_EXAMPLES = len(self.training_data)
        globals.TR_EXAMPLES = 0
        print("- loaded example shape")

    # add examples to training dataset
    def add_example(self, sample: np.ndarray, label: int):
        encoded_y = keras.utils.np_utils.to_categorical(label, num_classes=Config.NUM_CLASSES)  # make one-hot
        encoded_y = np.reshape(encoded_y, (1, 2))
        self.training_labels = np.append(self.training_labels, encoded_y, axis=0)
        sample = sample.reshape(sample.shape[0], 1, row, col)
        self.training_data = np.append(self.training_data, sample, axis=0)
        print('add example for label %d' % label)

        globals.HAS_BEEN_TRAINED = False

    # Train the model on recorded examples
    def train_model(self):
        if globals.TR_EXAMPLES > 0:
            weight_ratio = np.ceil(globals.BG_EXAMPLES / globals.TR_EXAMPLES)
        else:
            weight_ratio = 1
        print(weight_ratio)

        self.model.fit(self.training_data,
                       self.training_labels,
                       epochs=Config.EPOCHS,
                       batch_size=Config.BATCH_SIZE,
                       class_weight={0: 1, 1: weight_ratio}
                       )
        print("model trained")
        globals.HAS_BEEN_TRAINED = True

        if Config.LOAD_MODEL:
            self.model.save('data/previous_model.h5')  # when trained save as the previous model
            print("saved model")

        # When true the background data set can be updated
        if Config.UPDATE_BG_DATA:
            print(self.training_data.shape)
            self.model.save("data/neutral_model.h5")
            self.model.save("data/previous_model.h5")
            np.save('data/background_sound_examples.npy', self.training_data)
            np.save('data/background_sound_labels.npy', self.training_labels)

    # Predict incoming frames
    def predict(self, sample: np.ndarray):
        print(sample.shape)
        sample = np.expand_dims(sample, axis=0)
        with self.graph.as_default():
            prediction = self.model.predict(sample)
        return np.argmax(prediction)

    # Reset the current model to the neutral with no wake-word trained yet.
    def reset_model(self):
        self.load_BG_examples()
        K.clear_session()
        self.model = load_model('data/neutral_model.h5')
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()
        print("LOADED MODEL")
        globals.HAS_BEEN_TRAINED = False
