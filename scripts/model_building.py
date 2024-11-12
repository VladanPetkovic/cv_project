from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# from sklearn.model_selection import train_test_split
# from tensorflow.python.keras import utils
# from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint


# ------------ Model for Age detection ------------ #
# We have cropped/resized all images to 100x100x1 (grey-scale)
def create_cnn_model(num_classes, input_shape=(100, 100, 1)):
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


# ------------ Training ------------#
# TODO: load images from prepared and augmented folder
def train_cnn_model(model, df_final, epochs=20, learning_rate=0.001, batch_size=32, validation_split=0.2):
    # 1. (Random) initialization of weights and biases --> done by the model

    img_height, img_width = 100, 100  # Set image dimensions for the model input

    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=validation_split)

    train_generator = datagen.flow_from_dataframe(
        dataframe=df_final,
        x_col="FilePath",
        y_col="age-bin",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    validation_generator = datagen.flow_from_dataframe(
        dataframe=df_final,
        x_col="FilePath",
        y_col="age-bin",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Mini-batch gradient descent
        for x_batch, y_batch in train_generator:
            with tf.GradientTape() as tape:
                predictions = model(x_batch, training=True)
                loss = tf.keras.losses.categorical_crossentropy(y_batch, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss, train_accuracy = model.evaluate(train_generator, verbose=0)
        val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
        history['loss'].append(train_loss)
        history['accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(
            f"loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")

        # 4. Repeat until stopping criterion is met (number of epochs)
    return history


def plot_training_history(history):
    # training & validation accuracy
    plt.figure(figsize=(12, 5))

    # accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
