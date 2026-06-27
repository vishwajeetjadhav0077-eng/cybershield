import os
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

# =====================================================
# CONFIGURATION
# =====================================================

MODEL_NAME = "distilbert-base-uncased"

TRAIN_FILE = "datasets/processed/train.csv"
VALID_FILE = "datasets/processed/validation.csv"
TEST_FILE = "datasets/processed/test.csv"

OUTPUT_DIR = "model/distilbert_model"
CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =====================================================
# RANDOM SEED
# =====================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("CYBERSHIELD AI")
print("=" * 60)
print("Device :", DEVICE)

# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
valid_df = pd.read_csv(VALID_FILE)
test_df = pd.read_csv(TEST_FILE)

train_df = train_df[["text", "label"]]
valid_df = valid_df[["text", "label"]]
test_df = test_df[["text", "label"]]

print("Train :", len(train_df))
print("Validation :", len(valid_df))
print("Test :", len(test_df))

# =====================================================
# CREATE HF DATASETS
# =====================================================

train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)
test_dataset = Dataset.from_pandas(test_df)

# =====================================================
# TOKENIZER
# =====================================================

print("\nLoading tokenizer...")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )

train_dataset = train_dataset.map(tokenize, batched=True)
valid_dataset = valid_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

columns = [
    "input_ids",
    "attention_mask",
    "label"
]

train_dataset.set_format(
    type="torch",
    columns=columns
)

valid_dataset.set_format(
    type="torch",
    columns=columns
)

test_dataset.set_format(
    type="torch",
    columns=columns
)

data_collator = DataCollatorWithPadding(tokenizer)

# =====================================================
# MODEL
# =====================================================

print("\nLoading DistilBERT...")

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.to(DEVICE)

# =====================================================
# METRICS
# =====================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary"
    )

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# =====================================================
# TRAINING ARGUMENTS
# =====================================================

training_args = TrainingArguments(

    output_dir=CHECKPOINT_DIR,

    overwrite_output_dir=True,

    evaluation_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",

    logging_steps=50,

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=EPOCHS,

    weight_decay=0.01,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    fp16=torch.cuda.is_available(),

    report_to="none"
)
# =====================================================
# TRAINER
# =====================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=valid_dataset,

    tokenizer=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ]

)

# =====================================================
# TRAIN
# =====================================================

print("\n" + "=" * 60)
print("STARTING DISTILBERT TRAINING")
print("=" * 60)

trainer.train()

print("\nTraining Finished Successfully.")

# =====================================================
# VALIDATION
# =====================================================

print("\nRunning Validation...")

validation_results = trainer.evaluate()

print("\nValidation Results")

for key, value in validation_results.items():

    print(f"{key}: {value}")

# =====================================================
# TESTING
# =====================================================

print("\nRunning Test Set...")

test_results = trainer.predict(test_dataset)

predictions = np.argmax(
    test_results.predictions,
    axis=1
)

labels = test_results.label_ids

accuracy = accuracy_score(
    labels,
    predictions
)

precision, recall, f1, _ = precision_recall_fscore_support(

    labels,

    predictions,

    average="binary"

)

print("\n" + "=" * 60)
print("FINAL TEST METRICS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

from sklearn.metrics import classification_report

print("\nClassification Report\n")

print(

    classification_report(

        labels,

        predictions,

        target_names=[

            "Safe",

            "Cyberbullying"

        ]

    )

)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    labels,
    predictions
)

print("\nConfusion Matrix\n")

print(cm)

# =====================================================
# SAVE MODEL
# =====================================================

print("\nSaving Model...")

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print("\nModel Saved Successfully!")

print("Location:")

print(OUTPUT_DIR)
# =====================================================
# SAMPLE PREDICTIONS
# =====================================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

samples = [
    "Hello everyone",
    "My name is Om",
    "You are stupid",
    "Nobody likes you",
    "Content is informative",
    "I hate you",
    "Go kill yourself",
    "Have a wonderful day"
]

model.eval()

for text in samples:

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probs = torch.softmax(
            outputs.logits,
            dim=1
        )

        prediction = torch.argmax(
            probs,
            dim=1
        ).item()

        confidence = probs[0][prediction].item() * 100

    label = "Cyberbullying" if prediction == 1 else "Safe"

    print("-" * 60)
    print(f"Text       : {text}")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.2f}%")

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nSaved files:")

print(f"{OUTPUT_DIR}/config.json")
print(f"{OUTPUT_DIR}/model.safetensors")
print(f"{OUTPUT_DIR}/tokenizer.json")
print(f"{OUTPUT_DIR}/tokenizer_config.json")
print(f"{OUTPUT_DIR}/special_tokens_map.json")

print("\nModel is now ready to be used in the Flask API.")

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print("\nStarting DistilBERT Pipeline...\n")