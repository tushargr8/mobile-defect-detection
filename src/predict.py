# =============================================================================
# predict.py — Inference with Saved Models
# =============================================================================
"""
Load a saved model and make predictions on new GitHub issue text.

Provides:
    - Single-text prediction
    - Batch prediction from a DataFrame
    - CLI interface for quick testing

Usage:
    from src.predict import Predictor

    predictor = Predictor()
    result = predictor.predict_text("App crashes when opening settings menu")
    print(result)
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any

from src.preprocessing import TextPreprocessor
from src.feature_engineering import load_vectorizer, load_label_encoder
from src.utils import get_logger, PATHS, CONFIG

logger = get_logger(__name__)


class Predictor:
    """
    End-to-end predictor for GitHub issue classification.

    Loads the saved model, vectorizer, and label encoder, then provides
    methods for single-text and batch prediction.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to saved model. Defaults to best_model.joblib in models/.
    vectorizer_path : str or Path, optional
        Path to saved TF-IDF vectorizer.
    encoder_path : str or Path, optional
        Path to saved label encoder.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = None,
        vectorizer_path: Union[str, Path] = None,
        encoder_path: Union[str, Path] = None,
    ):
        models_dir = PATHS["models_dir"]

        # Load model
        model_path = model_path or models_dir / "best_model.joblib"
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from: {model_path}")

        # Load vectorizer
        self.vectorizer = load_vectorizer(vectorizer_path)
        vec_type = "TF-IDF" if hasattr(self.vectorizer, "idf_") else "Count"
        logger.info(f"Loaded {vec_type} vectorizer")

        # Load label encoder
        self.label_encoder = load_label_encoder(encoder_path)
        logger.info(f"Label classes: {list(self.label_encoder.classes_)}")

        # Text preprocessor
        self.preprocessor = TextPreprocessor()

    def predict_text(self, text: str) -> Dict[str, Any]:
        """
        Predict the class of a single text input.

        Parameters
        ----------
        text : str
            Raw issue text (title + body combined).

        Returns
        -------
        dict
            Prediction result with class, confidence, and all probabilities.
        """
        # Create a dummy DataFrame row
        row = pd.Series({"title": text, "body": ""})
        combined = self.preprocessor._combine_text(row)
        cleaned = self.preprocessor._clean_text(combined)
        cleaned = self.preprocessor._tokenize_and_normalize(cleaned)

        # Vectorize
        X = self.vectorizer.transform([cleaned])

        # Predict
        y_pred = self.model.predict(X)[0]
        predicted_class = self.label_encoder.inverse_transform([y_pred])[0]

        result = {
            "input_text": text[:200],
            "cleaned_text": cleaned[:200],
            "predicted_class": predicted_class,
            "predicted_index": int(y_pred),
        }

        # Add probabilities if available
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)[0]
            result["confidence"] = round(float(probabilities.max()), 4)
            result["class_probabilities"] = {
                self.label_encoder.inverse_transform([i])[0]: round(float(p), 4)
                for i, p in enumerate(probabilities)
            }
        else:
            result["confidence"] = "N/A (model does not support probabilities)"

        return result

    def predict_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Predict classes for a batch of texts.

        Parameters
        ----------
        texts : list of str
            List of raw issue texts.

        Returns
        -------
        pd.DataFrame
            DataFrame with predictions and confidences.
        """
        logger.info(f"Batch prediction on {len(texts)} texts...")

        cleaned_texts = []
        for text in texts:
            row = pd.Series({"title": text, "body": ""})
            combined = self.preprocessor._combine_text(row)
            cleaned = self.preprocessor._clean_text(combined)
            cleaned = self.preprocessor._tokenize_and_normalize(cleaned)
            cleaned_texts.append(cleaned)

        X = self.vectorizer.transform(cleaned_texts)
        y_pred = self.model.predict(X)
        predicted_classes = self.label_encoder.inverse_transform(y_pred)

        results_df = pd.DataFrame({
            "text": [t[:100] for t in texts],
            "predicted_class": predicted_classes,
        })

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)
            results_df["confidence"] = probabilities.max(axis=1).round(4)

        return results_df


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import sys

    predictor = Predictor()

    if len(sys.argv) > 1:
        # Predict from command-line argument
        text = " ".join(sys.argv[1:])
        result = predictor.predict_text(text)
        print("\n--- Prediction Result ---")
        for k, v in result.items():
            if k != "class_probabilities":
                print(f"  {k}: {v}")
        if "class_probabilities" in result:
            print("  Class Probabilities:")
            for cls, prob in sorted(result["class_probabilities"].items(),
                                     key=lambda x: -x[1]):
                print(f"    {cls:<25s} {prob:.4f}")
    else:
        # Interactive mode
        print("Enter issue text (or 'quit' to exit):")
        while True:
            text = input("\n> ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            result = predictor.predict_text(text)
            print(f"\n  Predicted: {result['predicted_class']}")
            print(f"  Confidence: {result.get('confidence', 'N/A')}")
