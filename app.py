# =============================================================================
# app.py — Flask Web Server for Interactive Model Testing
# =============================================================================
"""
A lightweight web testing server that allows you to input raw GitHub issue titles
and bodies, select between Dataset 1 (15 classes) or Dataset 2 (3 classes) models,
and receive real-time classification predictions, confidences, and probability distributions.

Usage:
    python app.py
"""

import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import Predictor
from src.utils import get_logger

logger = get_logger("demo_server")
app = Flask(__name__)

# Initialize Predictors for both datasets at startup
logger.info("Initializing Dataset 1 predictor...")
try:
    predictor_d1 = Predictor(
        model_path=PROJECT_ROOT / "models" / "dataset_1" / "best_model.joblib",
        vectorizer_path=PROJECT_ROOT / "models" / "dataset_1" / "tfidf_vectorizer.joblib",
        encoder_path=PROJECT_ROOT / "models" / "dataset_1" / "label_encoder.joblib"
    )
    logger.info("Dataset 1 predictor loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Dataset 1 predictor: {e}")
    predictor_d1 = None

logger.info("Initializing Dataset 2 predictor...")
try:
    predictor_d2 = Predictor(
        model_path=PROJECT_ROOT / "models" / "dataset_2" / "best_model.joblib",
        vectorizer_path=PROJECT_ROOT / "models" / "dataset_2" / "tfidf_vectorizer.joblib",
        encoder_path=PROJECT_ROOT / "models" / "dataset_2" / "label_encoder.joblib"
    )
    logger.info("Dataset 2 predictor loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Dataset 2 predictor: {e}")
    predictor_d2 = None


@app.route("/")
def home():
    """Render the main testing dashboard interface."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept text data and a selected dataset target, and return ML predictions.
    """
    data = request.get_json() or {}
    title = data.get("title", "").strip()
    body = data.get("body", "").strip()
    dataset = data.get("dataset", "dataset_1")

    if not title:
        return jsonify({"error": "An issue title is required."}), 400

    # Choose the correct predictor
    if dataset == "dataset_1":
        predictor = predictor_d1
    elif dataset == "dataset_2":
        predictor = predictor_d2
    else:
        return jsonify({"error": f"Invalid dataset selection: {dataset}"}), 400

    if predictor is None:
        return jsonify({
            "error": f"Predictor for {dataset} was not loaded. Verify that the trained model joblib files exist."
        }), 500

    try:
        # Combine title and body with space separator (exactly how it was preprocessed)
        combined_text = f"{title} {body}".strip()
        result = predictor.predict_text(combined_text)

        # Structure response
        response = {
            "dataset": dataset,
            "predicted_class": result["predicted_class"],
            "confidence": result.get("confidence", "N/A"),
            "probabilities": result.get("class_probabilities", {})
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed internally: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
