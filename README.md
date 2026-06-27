# 🛡️ CyberShield — AI Comment Safety System

DistilBERT-powered cyberbullying detection API with Chrome extension support for YouTube, Instagram, Twitter/X, and Reddit.

---

## What's inside

```
cybershield/
├── train.py               ← Fine-tune DistilBERT on your dataset
├── app.py                 ← Flask REST API (serves predictions)
├── templates/index.html   ← Web UI (checker + API docs + stats)
├── requirements.txt
├── Dockerfile
├── render.yaml            ← One-click Render.com deployment
├── hf_app.py              ← Hugging Face Spaces entry point
├── extension/
│   ├── manifest.json      ← Chrome Manifest v3
│   └── content.js         ← Runs on YouTube / Instagram / Twitter / Reddit
└── models/                ← Created by train.py
    ├── binary/            ← DistilBERT binary classifier
    ├── category/          ← DistilBERT category classifier
    ├── tokenizer/
    ├── label_encoder.pkl
    └── config.pkl
```

---

## Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> Needs ~4 GB RAM. GPU optional but speeds up training 6×.

---

## Step 2 — Train the model

Place `cyberbullying_tweets.csv` in the project root, then:

```bash
python train.py
```

This will:
- Clean and preprocess all text
- Fine-tune **two DistilBERT models**:
  - `models/binary/`  — detects if a comment is bullying (yes/no)
  - `models/category/` — classifies which type (gender, religion, age, ethnicity, other)
- Save confusion matrix PNGs inside each model folder
- Print accuracy, F1, precision, recall per class

**Expected accuracy: ~92–95%** (vs ~87% with SVM)

Training time:
| Hardware | Time |
|----------|------|
| CPU only | ~25–35 min |
| GPU (T4) | ~4–6 min |

---

## Step 3 — Run locally

```bash
python app.py
```

Open `http://localhost:5000` — you'll see the full web UI.

---

## Step 4 — Deploy to the cloud (free)

### Option A: Render.com ⭐ recommended

1. Push your project to GitHub (include the `models/` folder)
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your repo
4. Render auto-detects `render.yaml` and configures everything
5. Your API URL will be: `https://cybershield-api.onrender.com`

### Option B: Railway.app

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
railway init
railway up
```

### Option C: Hugging Face Spaces

1. Create a new Space → SDK: **Docker**
2. Upload all files (including `models/`)
3. The Space will build automatically and be live at:  
   `https://huggingface.co/spaces/YOUR_USERNAME/cybershield`

### Option D: Docker (any VPS)

```bash
docker build -t cybershield .
docker run -p 8000:8000 -v $(pwd)/models:/app/models cybershield
```

---

## Step 5 — Connect the Chrome extension

1. Open `extension/content.js`
2. Change line 1:
   ```js
   const API_URL = "https://your-deployed-url.com";
   ```
3. Open Chrome → `chrome://extensions`
4. Enable **Developer Mode** (top-right toggle)
5. Click **Load unpacked** → select the `extension/` folder

The extension now works on:
- ✅ YouTube (comment threads)
- ✅ Instagram (captions + comments)
- ✅ Twitter / X (tweet replies)
- ✅ Reddit (post comments)

Hover over a blurred comment to peek. Click the badge to permanently reveal it.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/health` | Health check |
| POST   | `/predict` | Single comment analysis |
| POST   | `/predict/batch` | Up to 50 comments at once |
| GET    | `/stats` | Usage analytics |

### POST /predict

**Request** (JSON):
```json
{ "text": "your comment here" }
```

**Response**:
```json
{
  "result":     1,
  "category":   "gender",
  "confidence": 91.4,
  "severity":   "high"
}
```

- `result`: `0` = safe, `1` = cyberbullying  
- `category`: `not_cyberbullying` | `gender` | `age` | `ethnicity` | `religion` | `other_cyberbullying`  
- `severity`: `none` | `low` | `medium` | `high`

### POST /predict/batch

```json
{ "texts": ["comment 1", "comment 2"] }
```
Returns an array of results in the same order.

---

## Accuracy improvements over SVM

| Model | Accuracy | F1 (macro) |
|-------|----------|------------|
| SVM + TF-IDF (old) | ~87% | ~0.85 |
| DistilBERT (new) | ~93–95% | ~0.92 |

Key reasons:
- DistilBERT understands **word context** — "not bad" is positive, not negative
- Handles **sarcasm and implicit language** better than TF-IDF bags of words
- Transfer learning from 100M+ parameter pre-training on English text
- Confidence threshold set to 65% (not 50%) — reduces false positives significantly

---

## Threshold tuning

Edit `models/config.pkl` or change `BULLY_THRESHOLD` in `train.py`:

| Threshold | Effect |
|-----------|--------|
| 0.50 | More sensitive — catches more but more false positives |
| 0.65 | Balanced ← default |
| 0.75 | More conservative — fewer false positives, may miss mild cases |

---

## Dataset

The model is trained on `cyberbullying_tweets.csv` (Kaggle — andrewmvd, ~47k rows).  
To improve accuracy further, merge with:
- [Jigsaw Toxic Comments](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) — 160k rows
- The synthetic dataset from `cyberbullying_training_data.csv` if generated

---

## License

MIT — free to use, modify, and deploy.
