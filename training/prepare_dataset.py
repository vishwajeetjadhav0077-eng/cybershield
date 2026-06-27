import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os

print("=" * 60)
print("CYBERSHIELD AI - DATASET PREPARATION")
print("=" * 60)

INPUT_FILE = "datasets/processed/merged_dataset.csv"
OUTPUT_FOLDER = "datasets/processed"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("\nLoading merged dataset...")
df = pd.read_csv(INPUT_FILE)

print(f"Original Size : {len(df)}")

# -----------------------------
# Remove Missing Values
# -----------------------------
df = df.dropna(subset=["text", "label", "category"])

# -----------------------------
# Remove Duplicates
# -----------------------------
df = df.drop_duplicates(subset=["text"])

# -----------------------------
# Normalize Categories
# -----------------------------
category_map = {
    "not_cyberbullying": "safe",
    "safe": "safe",
    "other_cyberbullying": "harassment",
    "offensive_language": "offensive",
    "severe_toxic": "toxic"
}

df["category"] = df["category"].replace(category_map)

# -----------------------------
# Clean Text
# -----------------------------
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\\S+", "", text)

    text = re.sub(r"www\\S+", "", text)

    text = re.sub(r"@\\w+", "", text)

    text = re.sub(r"#", "", text)

    text = re.sub(r"[^a-zA-Z0-9\\s]", " ", text)

    text = re.sub(r"\\s+", " ", text)

    return text.strip()

print("\nCleaning text...")

df["text"] = df["text"].apply(clean_text)

df = df[df["text"].str.len() > 3]

print("After Cleaning :", len(df))

# -----------------------------
# Balance Binary Classes
# -----------------------------
print("\nBalancing dataset...")

safe = df[df.label == 0]
bully = df[df.label == 1]

target = min(len(safe), len(bully))

safe = resample(
    safe,
    replace=False,
    n_samples=target,
    random_state=42
)

bully = resample(
    bully,
    replace=False,
    n_samples=target,
    random_state=42
)

balanced = pd.concat([safe, bully])

balanced = balanced.sample(frac=1, random_state=42)

print("\nBalanced Distribution")

print(balanced["label"].value_counts())

# -----------------------------
# Split
# -----------------------------
train_df, test_df = train_test_split(
    balanced,
    test_size=0.15,
    stratify=balanced["label"],
    random_state=42
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.1765,
    stratify=train_df["label"],
    random_state=42
)

print("\nTrain :", len(train_df))
print("Validation :", len(val_df))
print("Test :", len(test_df))

train_df.to_csv(
    f"{OUTPUT_FOLDER}/train.csv",
    index=False
)

val_df.to_csv(
    f"{OUTPUT_FOLDER}/validation.csv",
    index=False
)

test_df.to_csv(
    f"{OUTPUT_FOLDER}/test.csv",
    index=False
)

print("\nSaved Successfully!")

print("\nGenerated Files")

print("train.csv")

print("validation.csv")

print("test.csv")

print("=" * 60)