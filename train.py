"""
================================================================
CYBERSHIELD — DistilBERT TRAINING SCRIPT
================================================================
Replaces the old SVM + TF-IDF pipeline with DistilBERT fine-tuning.

What this does:
  - Fine-tunes distilbert-base-uncased on the cyberbullying dataset
  - Trains a BINARY classifier  (bullying / not bullying)
  - Trains a MULTI-CLASS classifier  (which type of bullying)
  - Saves both as HuggingFace models in ./models/
  - Expected accuracy: ~92-95%  (vs ~87% for linear SVM)

Run:
    pip install -r requirements.txt
    python train.py

Requirements: ~4 GB RAM, ~30 min on CPU  /  ~5 min on GPU
================================================================
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
DATASET_PATH  = "cyberbullying_tweets.csv"
MODELS_DIR    = "models"
BINARY_DIR    = os.path.join(MODELS_DIR, "binary")
CATEGORY_DIR  = os.path.join(MODELS_DIR, "category")

MAX_LEN       = 128
BATCH_SIZE    = 16
EPOCHS_BIN    = 3       # binary model epochs
EPOCHS_CAT    = 4       # category model epochs
LEARNING_RATE = 2e-5
BULLY_THRESHOLD = 0.65  # min probability to flag as bullying

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

os.makedirs(BINARY_DIR,   exist_ok=True)
os.makedirs(CATEGORY_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# TEXT PREPROCESSING  (same as api/text_utils.py)
# ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ──────────────────────────────────────────────────────────────
# PYTORCH DATASET
# ──────────────────────────────────────────────────────────────
class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ──────────────────────────────────────────────────────────────
# TRAIN HELPER
# ──────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, epochs, save_dir, label_names=None):
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    model.to(DEVICE)
    best_val_acc = 0.0

    for epoch in range(epochs):
        # ── Training ──
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids      = batch["input_ids"].to(DEVICE),
                attention_mask = batch["attention_mask"].to(DEVICE),
                labels         = batch["label"].to(DEVICE),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()

        avg_loss = total_loss / len(train_loader)

        # ── Validation ──
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    input_ids      = batch["input_ids"].to(DEVICE),
                    attention_mask = batch["attention_mask"].to(DEVICE),
                )
                logits = out.logits
                preds_all.extend(torch.argmax(logits, dim=1).cpu().numpy())
                labels_all.extend(batch["label"].numpy())

        val_acc = accuracy_score(labels_all, preds_all)
        print(f"  Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}  val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(save_dir)
            print(f"    ✓ Saved best model  (acc={val_acc:.4f})")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # ── Final evaluation on val set with best weights ──
    model = DistilBertForSequenceClassification.from_pretrained(save_dir).to(DEVICE)
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in val_loader:
            out = model(
                input_ids      = batch["input_ids"].to(DEVICE),
                attention_mask = batch["attention_mask"].to(DEVICE),
            )
            preds_all.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            labels_all.extend(batch["label"].numpy())

    print("\nClassification Report:")
    print(classification_report(labels_all, preds_all, target_names=label_names))

    # Confusion matrix
    cm = confusion_matrix(labels_all, preds_all)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {os.path.basename(save_dir)}")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_dir}/confusion_matrix.png")

    return model

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── 1. Load & clean dataset ──
    print("\n" + "="*60)
    print("STEP 1 — Loading dataset")
    print("="*60)
    df = pd.read_csv(DATASET_PATH)
    df = df.rename(columns={"tweet_text": "Content", "cyberbullying_type": "Category"})
    df = df[["Content", "Category"]].dropna()
    df["Content"] = df["Content"].apply(clean_text)
    df = df[df["Content"].str.len() > 0]
    print(f"Loaded {len(df)} rows after cleaning")
    print(df["Category"].value_counts())

    # ── 2. Binary labels ──
    df["BinaryLabel"] = df["Category"].apply(lambda x: 0 if x == "not_cyberbullying" else 1)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    tokenizer.save_pretrained(os.path.join(MODELS_DIR, "tokenizer"))

    # ── 3. BINARY MODEL ──
    print("\n" + "="*60)
    print("STEP 2 — Training BINARY model (bullying / not bullying)")
    print("="*60)

    X_train, X_test, y_train, y_test = train_test_split(
        df["Content"].tolist(), df["BinaryLabel"].tolist(),
        test_size=0.2, random_state=42, stratify=df["BinaryLabel"],
    )

    train_ds = CommentDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_ds   = CommentDataset(X_test,  y_test,  tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    bin_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    train_model(
        bin_model, train_loader, val_loader,
        epochs=EPOCHS_BIN,
        save_dir=BINARY_DIR,
        label_names=["Not Bullying", "Bullying"],
    )

    # ── 4. Save threshold config ──
    config = {"bully_threshold": BULLY_THRESHOLD, "max_len": MAX_LEN}
    with open(os.path.join(MODELS_DIR, "config.pkl"), "wb") as f:
        pickle.dump(config, f)

    # ── 5. CATEGORY MODEL ──
    print("\n" + "="*60)
    print("STEP 3 — Training CATEGORY model (type of bullying)")
    print("="*60)

    bully_df = df[df["BinaryLabel"] == 1].copy()
    le = LabelEncoder()
    bully_df["CategoryEncoded"] = le.fit_transform(bully_df["Category"])
    print("Categories:", list(le.classes_))

    # Save label encoder
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        bully_df["Content"].tolist(), bully_df["CategoryEncoded"].tolist(),
        test_size=0.2, random_state=42, stratify=bully_df["CategoryEncoded"],
    )

    cat_train_ds = CommentDataset(Xc_train, yc_train, tokenizer, MAX_LEN)
    cat_val_ds   = CommentDataset(Xc_test,  yc_test,  tokenizer, MAX_LEN)
    cat_train_loader = DataLoader(cat_train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    cat_val_loader   = DataLoader(cat_val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    cat_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(le.classes_)
    )
    train_model(
        cat_model, cat_train_loader, cat_val_loader,
        epochs=EPOCHS_CAT,
        save_dir=CATEGORY_DIR,
        label_names=list(le.classes_),
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"  Binary model   → {BINARY_DIR}")
    print(f"  Category model → {CATEGORY_DIR}")
    print(f"  Tokenizer      → {MODELS_DIR}/tokenizer")
    print(f"  Label encoder  → {MODELS_DIR}/label_encoder.pkl")
    print("="*60)
