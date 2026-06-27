from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from api.predictor import predict_text
from api.database import get_statistics
app = Flask(__name__)
CORS(app)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        text = request.form.get("text", "").strip()

        if text:

            prediction = predict_text(text)

            result = prediction

    return render_template(
        "index.html",
        result=result
    )


# ============================================================
# API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Chrome Extension
    if request.form:

        text = request.form.get("text", "").strip()

    else:

        data = request.get_json(silent=True)

        text = data.get("text", "").strip() if data else ""

    if text == "":

        return jsonify({
            "error": "No text provided"
        }), 400

    prediction = predict_text(text)

    return jsonify({

    "result": prediction["result"],

    "message": prediction["message"],

    "confidence": prediction["confidence"],

    "severity": prediction["severity"],

    "category": prediction["category"]

})


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "running",

        "model": "DistilBERT",

        "device": "GPU"

    })
# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    return render_template("dashboard.html")


@app.route("/api/stats")
def api_stats():

    return jsonify(get_statistics())

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CYBERSHIELD AI SERVER")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )