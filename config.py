class Config:
    # Assistant selection
    ASSISTANT = "GoogleHome"  # GoogleHome or Alexa
    VOLUME = "83"  # speaker volume

    # Classifier stuff
    NUM_CLASSES = 2
    LEARNING_RATE = 0.0001
    EPOCHS = 10
    BATCH_SIZE = 8
    DENSE_UNITS = 128
    LOAD_MODEL = True
