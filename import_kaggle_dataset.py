"""
Import the Enron spam dataset.

Run from inside the folder:
    python import_kaggle_dataset.py

"""

import sys
from pathlib import Path

import pandas as pd


# Paths and constants

DATASET_URL = "https://github.com/MWiechmann/enron_spam_data/raw/master/enron_spam_data.zip"
DATASET_PATH = Path("database/mail_dataset.csv")
CATEGORY_LABELS = ["work", "personal", "promotion"]


# Import pipeline

def main():
    source = sys.argv[1] if len(sys.argv) > 1 else DATASET_URL

    # Keep the manually labeled category rows, then replace the larger spam/ham set.
    old_data = pd.read_csv(DATASET_PATH, keep_default_na=False)
    category_rows = old_data[old_data["category"].isin(CATEGORY_LABELS)]

    kaggle = pd.read_csv(source, keep_default_na=False)

    # Normalize Kaggle columns to the app's text / label / category format.
    new_data = pd.DataFrame()
    new_data["text"] = kaggle["Subject"].astype(str) + " " + kaggle["Message"].astype(str)
    new_data["label"] = kaggle["Spam/Ham"].str.lower()
    new_data["category"] = ""

    new_data = new_data[new_data["label"].isin(["ham", "spam"])]
    new_data = new_data[["text", "label", "category"]]

    final_data = pd.concat([new_data, category_rows], ignore_index=True)
    final_data = final_data.drop_duplicates()
    final_data.to_csv(DATASET_PATH, index=False)

    print("Dataset updated successfully.")
    print(f"Total rows: {len(final_data)}")
    print(f"Category rows kept: {len(category_rows)}")
    print("Now run: python train.py")
    print("Then run: python evaluate.py")


# Entry point

if __name__ == "__main__":
    main()
