import os
import pandas as pd
import re
import datetime
from collections import defaultdict


def extract_data_from_file(file_name):
    # [age]_[gender]_[race]_[date&time].jpg
    pattern = r"(\d+)_(\d)_(\d)_(\d+)\.jpg"
    match = re.match(pattern, file_name)

    if match:
        unique_id = file_name
        age = int(match.group(1))  # age
        gender = int(match.group(2))  # gender (0 = male, 1 = female)
        race = int(match.group(3))  # race (0 = White, 1 = Black, 2 = Asian, 3 = Indian, 4 = Others)
        date_time = match.group(4)  # date-time
        return unique_id, age, gender, race, date_time
    return None


def convert_to_meaningful_data(data):
    unique_id, age, gender, race, datetime_str = data

    # convert gender
    gender_str = "male" if gender == 0 else "female"

    # convert race
    race_dict = {
        0: "White",
        1: "Black",
        2: "Asian",
        3: "Indian",
        4: "Others"
    }
    race_str = race_dict.get(race)

    # convert date-time to a more readable format
    try:
        formatted_datetime = datetime.datetime.strptime(datetime_str[:14], "%Y%m%d%H%M%S")
        formatted_datetime_str = formatted_datetime.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        formatted_datetime_str = "Invalid date"

    return unique_id, age, gender_str, race_str, formatted_datetime_str


def get_dataframe():
    folders = ['data/part1', 'data/part2', 'data/part3']
    all_data = []

    # collect all data
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith(".jpg"):
                data = extract_data_from_file(filename)
                if data is not None:
                    all_data.append(convert_to_meaningful_data(data))

    return pd.DataFrame(all_data, columns=["Unique-Identifier", "Age", "Gender", "Race", "DateTime"])


def create_csv(dataframe):
    dataframe.to_csv("data/images.csv", index=False)


def print_file_endings():
    folders = ['data/part1', 'data/part2', 'data/part3']
    format_counts = defaultdict(int)

    # checking all file-endings
    for folder in folders:
        for filename in os.listdir(folder):
            file_extension = filename.split('.')[-1].lower()
            format_counts[file_extension] += 1

    for ext, count in format_counts.items():
        print(f"Number of {ext.upper()} files: {count}")
