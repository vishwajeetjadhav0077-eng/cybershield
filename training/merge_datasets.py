import pandas as pd
import os

print("=" * 60)
print("CYBERSHIELD AI - DATASET MERGER")
print("=" * 60)

RAW_PATH = "datasets/raw"
OUTPUT_PATH = "datasets/processed"

# Create processed folder if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# -----------------------------
# DATASET 1 : Cyberbullying Tweets
# -----------------------------
print("\nLoading Cyberbullying Tweets Dataset...")

twitter = pd.read_csv(f"{RAW_PATH}/cyberbullying_tweets.csv")

twitter = twitter.rename(columns={
    "tweet_text": "text",
    "cyberbullying_type": "category"
})

twitter["label"] = twitter["category"].apply(
    lambda x: 0 if x == "not_cyberbullying" else 1
)

twitter["source"] = "twitter"

twitter = twitter[["text", "label", "category", "source"]]

print("Twitter Dataset :", len(twitter))


# -----------------------------
# DATASET 2 : Jigsaw Toxic
# -----------------------------
print("\nLoading Jigsaw Dataset...")

jigsaw = pd.read_csv(f"{RAW_PATH}/train.csv")

jigsaw["label"] = (
    (
        jigsaw["toxic"] +
        jigsaw["severe_toxic"] +
        jigsaw["obscene"] +
        jigsaw["threat"] +
        jigsaw["insult"] +
        jigsaw["identity_hate"]
    ) > 0
).astype(int)

def get_category(row):

    if row["identity_hate"] == 1:
        return "identity_hate"

    if row["threat"] == 1:
        return "threat"

    if row["insult"] == 1:
        return "insult"

    if row["obscene"] == 1:
        return "obscene"

    if row["severe_toxic"] == 1:
        return "severe_toxic"

    if row["toxic"] == 1:
        return "toxic"

    return "safe"

jigsaw["category"] = jigsaw.apply(get_category, axis=1)

jigsaw = jigsaw.rename(columns={
    "comment_text": "text"
})

jigsaw["source"] = "jigsaw"

jigsaw = jigsaw[["text", "label", "category", "source"]]

print("Jigsaw Dataset :", len(jigsaw))


# -----------------------------
# DATASET 3 : Hate Speech
# -----------------------------
print("\nLoading Hate Speech Dataset...")

hate = pd.read_csv(f"{RAW_PATH}/labeled_data.csv")

hate = hate.rename(columns={
    "tweet": "text"
})

def convert_class(x):

    if x == 2:
        return 0

    return 1

hate["label"] = hate["class"].apply(convert_class)

def category(x):

    if x == 0:
        return "hate_speech"

    elif x == 1:
        return "offensive_language"

    else:
        return "safe"

hate["category"] = hate["class"].apply(category)

hate["source"] = "hate_speech"

hate = hate[["text", "label", "category", "source"]]

print("Hate Dataset :", len(hate))


# -----------------------------
# MERGE
# -----------------------------
print("\nMerging datasets...")

merged = pd.concat(
    [twitter, jigsaw, hate],
    ignore_index=True
)

print("Total Before Cleaning :", len(merged))

merged.drop_duplicates(subset=["text"], inplace=True)

merged.dropna(inplace=True)

merged["text"] = merged["text"].astype(str)

merged = merged[merged["text"].str.len() > 3]

merged = merged.sample(frac=1, random_state=42)

print("Total After Cleaning :", len(merged))

print("\nLabel Distribution")

print(merged["label"].value_counts())

print("\nCategory Distribution")

print(merged["category"].value_counts())

output_file = f"{OUTPUT_PATH}/merged_dataset.csv"

merged.to_csv(output_file, index=False)

print("\nDataset Saved Successfully!")

print(output_file)

print("=" * 60)