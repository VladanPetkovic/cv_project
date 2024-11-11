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

    # Split data into training and validation sets
    images = []
    labels = df_final['age-bin']
    num_samples = len(images)
    val_size = int(num_samples * validation_split)
    x_train, x_val = images[val_size:], images[:val_size]
    y_train, y_val = labels[val_size:], labels[:val_size]

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Shuffle training data at the start of each epoch
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]

        # Mini-batch gradient descent
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # 2. Forward propagation to determine predictions
            with tf.GradientTape() as tape:
                predictions = model(x_batch, training=True)

                # Calculate loss for the current batch
                loss = tf.keras.losses.categorical_crossentropy(y_batch, predictions)

            # 3. Backpropagation/Gradient descent
            # Compute gradients with respect to loss
            gradients = tape.gradient(loss, model.trainable_variables)
            # Update weights and biases based on gradients
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
        val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
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
