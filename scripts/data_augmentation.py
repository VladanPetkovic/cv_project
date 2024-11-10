import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def augment_images(df_subset, target_count, save_dir, label_col, unique_id_col='Unique-Identifier'):
    # Initialize ImageDataGenerator with augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    # Define directories to search for images
    directories = ['data/part1_prepared', 'data/part2_prepared', 'data/part3_prepared']

    # Calculate how many additional images are needed
    images_needed = target_count - len(df_subset)
    if images_needed <= 0:
        print("No additional images needed.")
        return

    # Determine number of augmented images per original image
    augment_per_image = images_needed // len(df_subset) + 1

    for index, row in tqdm(df_subset.iterrows(), total=df_subset.shape[0], desc="Augmenting images"):
        unique_id = row[unique_id_col]
        filename = unique_id  # Assumes unique_id includes the file name with extension

        # Search for the image in the specified directories
        img_path = None
        for directory in directories:
            potential_path = os.path.join(directory, filename)
            if os.path.exists(potential_path):
                img_path = potential_path
                break

        if not img_path:
            print(f"Image '{filename}' not found in the specified directories.")
            continue

        # Load and preprocess the image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image '{img_path}'.")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0)

        # Generate and save augmented images
        try:
            prefix = f"aug_{row[label_col]}_{index}"
            for i, augmented_image in enumerate(datagen.flow(
                    image,
                    batch_size=1,
                    save_to_dir=save_dir,
                    save_prefix=prefix,
                    save_format='jpg'
            )):
                if i >= augment_per_image:
                    break
        except Exception as e:
            print(f"Error augmenting image '{img_path}': {e}")



def augment_data(df, label_column, augmented_data_dir):
    # Create directory to save augmented images
    os.makedirs(augmented_data_dir, exist_ok=True)

    # Determine target count
    target_count = df[label_column].value_counts().max()

    # Augment data for each label
    for label in df[label_column].unique():
        df_subset = df[df[label_column] == label]
        save_dir = os.path.join(augmented_data_dir, str(label))
        os.makedirs(save_dir, exist_ok=True)
        augment_images(df_subset, target_count, save_dir, label_column)


def get_augmented_data(augmented_data_dir, label_column):
    augmented_data = []
    for label in os.listdir(augmented_data_dir):
        label_dir = os.path.join(augmented_data_dir, label)
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            augmented_data.append({
                'file_path': img_path,
                label_column: label
            })
    return pd.DataFrame(augmented_data)
