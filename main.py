# ------------- Imports ---------------------------------------------------------------------------
from scripts.data_exploration import *
from scripts.image_helper_functions import *
from scripts.data_augmentation import *
from scripts.model_building import *

# ------------- Data exploration before image-preparation -----------------------------------------
file_endings = print_file_endings()

folder_names = ['data/part1', 'data/part2', 'data/part3']
df = get_dataframe(folder_names)
create_csv(df, "unprepared_images.csv")
show_all_plots(df, "Before Image-preprocessing")
print_statistics(df)

# ------------- Image preparation -----------------------------------------------------------------
# TODO: uncomment this to crop all images, if not already done --> it will take some time
# crop_all_images_multi_threaded()


# ------------- Data exploration after image preparation ------------------------------------------
folder_names = ['data/part1_prepared', 'data/part2_prepared', 'data/part3_prepared']
df_prepared = get_dataframe(folder_names)
create_csv(df_prepared, "prepared_images.csv")
show_all_plots(df_prepared, "After Image-preprocessing")
print_statistics(df_prepared)

# ------------- Data augmentation -----------------------------------------------------------------
ageList = ['(0-4)', '(5-14)', '(15-24)', '(25-34)', '(35-49)', '(50-69)', '(70-117)']
age_bins = [0, 4, 14, 24, 34, 49, 69, 117]  # Age bin edges

df_prepared['age_bin'] = pd.cut(df_prepared['Age'], bins=age_bins, labels=ageList, right=False)

print("Columns in df_prepared:", df_prepared.columns.tolist())

# Perform data augmentation to balance the dataset
augmented_data_dir = 'data/augmented_images'
augment_data(df_prepared, 'age_bin', augmented_data_dir)

# Update the DataFrame with augmented data
df_augmented = get_augmented_data(augmented_data_dir, 'age_bin')
df_combined = pd.concat([df_prepared, df_augmented], ignore_index=True)
df_combined = df_combined.sample(frac=1).reset_index(drop=True)

# ------------- Model building --------------------------------------------------------------------
