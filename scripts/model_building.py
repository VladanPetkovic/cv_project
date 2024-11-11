from tensorflow.keras import models, layers
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from tensorflow.python.keras import utils
# from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint

# Model requirements for image
ageList = ['(0-4)', '(5-14)', '(15-24)', '(25-34)', '(35-49)', '(50-69)', '(70-100)']
age_bins = [0, 4, 14, 24, 34, 49, 69, 100]  # Age bin edges


# ------------ Model for Age detection ------------ #
def create_cnn_model(input_shape=(200, 200, 1), num_classes=len(ageList)):  # change for resized image size
    model = models.Sequential()
    # Input layer with 32 filters, followed by an AveragePooling2D layer
    model.add(layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.AveragePooling2D((2, 2)))
    # Three Conv2D layers with filters increasing by a factor of 2 for every successive Conv2D layer
    model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Conv2D(128, kernel_size=3, activation='relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Conv2D(256, kernel_size=3, activation='relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    # A GlobalAveragePooling2D layer before going into Dense layers below
    # GlobalAveragePooling2D layer limits outputs to number of filters in last Conv2D layer above (256)
    model.add(layers.GlobalAveragePooling2D())
    # model.add(layers.Flatten())    #should not be needed
    model.add(layers.Dense(132, activation='relu'))  # Reduces layers to 132 before final output
    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# ----- Image preprocessing --------#

# -------Age Prediction---------#