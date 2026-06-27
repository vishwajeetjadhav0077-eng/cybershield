import os
import torch
from api.database import save_prediction
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "distilbert_model"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Loading DistilBERT Model...")

tokenizer = DistilBertTokenizerFast.from_pretrained(
    MODEL_PATH
)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)
model.eval()

print("Model Loaded Successfully!")

# ============================================================
# PREDICT
# ============================================================

def predict_text(text):

    inputs = tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        padding=True,

        max_length=128

    )

    inputs = {

        key: value.to(DEVICE)

        for key, value in inputs.items()

    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(

            outputs.logits,

            dim=1

        )

        prediction = torch.argmax(

            probabilities,

            dim=1

        ).item()

        confidence = (

            probabilities[0][prediction].item()

            * 100

        )

    if prediction == 1:

        result = 1

        message = "Cyberbullying Detected"

    else:

        result = 0

        message = "Safe Comment"

    severity = "none"

    if result == 1:

        if confidence >= 90:

            severity = "high"

        elif confidence >= 75:

            severity = "medium"

        else:

            severity = "low"

    prediction = {
        "result": result,
        "message": message,
        "confidence": round(confidence, 2),
        "severity": severity,
        "category": "cyberbullying" if result == 1 else "safe"
    }

    save_prediction(text, prediction)

    return prediction


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    while True:

        text = input("\nEnter Text : ")

        if text.lower() == "exit":

            break

        print(predict_text(text))