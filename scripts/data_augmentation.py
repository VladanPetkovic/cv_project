import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

AUGMENTED_DATA_DIR = 'data/augmented_images'


def augment_images(df_subset, target_count, save_dir, label_col):
    created_images = 0

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
    if target_count < len(df_subset):
        print("No additional images needed.")
        return 0

    # number of augmented images per original image
    images_needed = min(1000, target_count - len(df_subset))  # create 1000 images MAX for one age-bin
    augment_per_image = images_needed // len(df_subset) + 1

    for index, row in tqdm(df_subset.iterrows(), total=df_subset.shape[0], desc="Augmenting images"):
        if images_needed <= created_images:
            continue

        unique_id = row['Unique-Identifier']
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
                created_images += 1
                if i >= augment_per_image:
                    break
        except Exception as e:
            print(f"Error augmenting image '{img_path}': {e}")

    return created_images


def augment_data(df, label_column):
    augmented_images = 0
    # Create directory to save augmented images
    os.makedirs(AUGMENTED_DATA_DIR, exist_ok=True)

    # target_count = max-age-bin
    target_count = df[label_column].value_counts().max()

    # create data for every age-bin
    for label in df[label_column].unique():
        df_subset = df[df[label_column] == label]
        save_dir = os.path.join(AUGMENTED_DATA_DIR, str(label))
        os.makedirs(save_dir, exist_ok=True)
        augmented_images += augment_images(df_subset, target_count, save_dir, label_column)

    print(f"Augmented {augmented_images} images.")


def get_augmented_data(label_column):
    augmented_data = []
    for label in os.listdir(AUGMENTED_DATA_DIR):
        label_dir = os.path.join(AUGMENTED_DATA_DIR, label)
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            augmented_data.append({
                'file_path': img_path,
                label_column: label
            })
    return pd.DataFrame(augmented_data)


def update_dataframe(previous_df):
    df_augmented = get_augmented_data('age_bin')
    df_combined = pd.concat([previous_df, df_augmented], ignore_index=True)
    return df_combined.sample(frac=1).reset_index(drop=True)
