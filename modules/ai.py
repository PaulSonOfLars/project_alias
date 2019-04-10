import keras
import numpy as np
from keras import backend as K
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.models import load_model

from config import Config
from modules import globals

# Classifier settings
# ====================================================#

row = 23
col = 13


# Load or reset training examples for class 0 (background sounds)
def load_BG_examples():
    global TRAINING_DATA, TRAINING_LABELS
    if Config.LOAD_MODEL:
        TRAINING_DATA = np.load('data/background_sound_examples.npy')
        TRAINING_LABELS = np.load('data/background_sound_labels.npy')
    else:
        TRAINING_DATA = np.empty([0, 1, row, col])  # XS Example array to be trained
        TRAINING_LABELS = np.empty([0, Config.NUM_CLASSES])  # YS Label array
    globals.BG_EXAMPLES = len(TRAINING_DATA)
    globals.TR_EXAMPLES = 0
    print("- loaded example shape")


# Initializing the kreas model
def create_model():
    global model
    load_BG_examples()

    if not Config.LOAD_MODEL:
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(1, row, col), data_format='channels_first'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.2))
        model.add(Flatten())
        model.add(Dense(units=Config.DENSE_UNITS, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=Config.NUM_CLASSES, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        globals.HAS_BEEN_TRAINED = False

    else:
        model = load_model('data/previous_model.h5')  # load the last model saved to continue.
        print("just loaded model")
        globals.HAS_BEEN_TRAINED = False  # Change to True if wake word is trained in the loaded model


# add examples to training dataset
def add_example(sample, label):
    global TRAINING_DATA, TRAINING_LABELS
    encoded_y = keras.utils.np_utils.to_categorical(label, num_classes=Config.NUM_CLASSES)  # make one-hot
    encoded_y = np.reshape(encoded_y, (1, 2))
    TRAINING_LABELS = np.append(TRAINING_LABELS, encoded_y, axis=0)
    sample = sample.reshape(sample.shape[0], 1, row, col)
    TRAINING_DATA = np.append(TRAINING_DATA, sample, axis=0)
    print('add example for label %d' % label)

    if globals.HAS_BEEN_TRAINED:
        globals.HAS_BEEN_TRAINED = False


# Train the model on recorded examples
def train_model():
    if globals.TR_EXAMPLES > 0:
        weight_ratio = np.ceil(globals.BG_EXAMPLES / globals.TR_EXAMPLES)
    else:
        weight_ratio = 1
    print(weight_ratio)

    model.fit(TRAINING_DATA,
              TRAINING_LABELS,
              epochs=Config.EPOCHS,
              batch_size=Config.BATCH_SIZE,
              class_weight={0: 1, 1: weight_ratio}
              )
    print("model trained")
    globals.HAS_BEEN_TRAINED = True

    if Config.LOAD_MODEL:
        model.save('data/previous_model.h5')  # when trained save as the previous model
        print("saved model")

    # When true the background data set can be updated
    if globals.UPDATE_BG_DATA:
        print(TRAINING_DATA.shape)
        model.save("data/neutral_model.h5")
        model.save("data/previous_model.h5")
        np.save('data/background_sound_examples.npy', TRAINING_DATA)
        np.save('data/background_sound_labels.npy', TRAINING_LABELS)


# Predict incoming frames
def predict(sample):
    print(sample.shape)
    sample = np.expand_dims(sample, axis=0)
    prediction = model.predict(sample)
    return np.argmax(prediction)


# Reset the current model to the neutral with no wake-word trained yet.
def reset_model():
    global model
    del model
    load_BG_examples()
    K.clear_session()
    model = load_model('data/neutral_model.h5')
    model._make_predict_function()
    print("LOADED MODEL")
    globals.HAS_BEEN_TRAINED = False
