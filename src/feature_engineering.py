# =============================================================================
# feature_engineering.py — TF-IDF and Feature Extraction Pipeline
# =============================================================================
"""
Builds the feature matrix from preprocessed text using TF-IDF vectorization.

Designed for extensibility — future versions can add:
    - Word2Vec / FastText embeddings
    - Sentence Transformers (BERT-based)
    - Statistical text features (length, readability, etc.)

Usage:
    from src.feature_engineering import build_features, load_vectorizer
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any

from src.utils import get_logger, PATHS, CONFIG, print_section

logger = get_logger(__name__)

# =============================================================================
# 1. Feature Extraction (TF-IDF & CountVectorizer)
# =============================================================================

def build_text_features(
    texts: pd.Series,
    config: dict = None,
    vectorizer = None,
    fit: bool = True,
):
    """
    Transform text data into feature vectors using TF-IDF or CountVectorizer.
    """
    config = config or CONFIG
    strategy = config.get("feature_strategy", "tfidf") # tfidf or count

    if vectorizer is None:
        if strategy == "tfidf":
            logger.info("Using TF-IDF Vectorizer")
            vectorizer = TfidfVectorizer(
                max_features=config.get("tfidf_max_features", 10000),
                ngram_range=config.get("tfidf_ngram_range", (1, 2)),
                min_df=config.get("tfidf_min_df", 3),
                max_df=config.get("tfidf_max_df", 0.95),
                sublinear_tf=True,
                strip_accents="unicode",
                analyzer="word",
                token_pattern=r"(?u)\b\w\w+\b",
            )
        elif strategy == "count":
            logger.info("Using Count Vectorizer")
            vectorizer = CountVectorizer(
                max_features=config.get("tfidf_max_features", 10000),
                ngram_range=config.get("tfidf_ngram_range", (1, 2)),
                min_df=config.get("tfidf_min_df", 3),
                max_df=config.get("tfidf_max_df", 0.95),
                strip_accents="unicode",
                analyzer="word",
                token_pattern=r"(?u)\b\w\w+\b",
            )
        else:
            raise ValueError(f"Unknown feature strategy: {strategy}")

    if fit:
        logger.info(f"Fitting {strategy} vectorizer on training text...")
        X = vectorizer.fit_transform(texts)
    else:
        logger.info(f"Transforming text with pre-fitted {strategy} vectorizer...")
        X = vectorizer.transform(texts)

    X = X.astype(np.float64)
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return X, vectorizer


def get_top_tfidf_features(vectorizer, n: int = 30) -> pd.DataFrame:
    """
    Extract the top N features by average TF-IDF score across all documents.
    """
    if not hasattr(vectorizer, 'idf_'):
        return pd.DataFrame()
        
    feature_names = vectorizer.get_feature_names_out()
    idf_scores = vectorizer.idf_

    top_indices = np.argsort(idf_scores)[:n]  # lowest IDF = most common
    bottom_indices = np.argsort(idf_scores)[-n:]  # highest IDF = rarest

    top_features = pd.DataFrame({
        "feature": feature_names[top_indices],
        "idf_score": idf_scores[top_indices],
        "type": "most_common",
    })

    rare_features = pd.DataFrame({
        "feature": feature_names[bottom_indices],
        "idf_score": idf_scores[bottom_indices],
        "type": "most_rare",
    })

    return pd.concat([top_features, rare_features], ignore_index=True)


# =============================================================================
# 2. Label Encoding
# =============================================================================

def encode_labels(
    y: pd.Series,
    encoder: LabelEncoder = None,
    fit: bool = True,
) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode string labels to integer indices.

    Parameters
    ----------
    y : pd.Series
        Target label series.
    encoder : LabelEncoder, optional
        Pre-fitted encoder for transform-only mode.
    fit : bool
        Whether to fit the encoder.

    Returns
    -------
    Tuple[np.ndarray, LabelEncoder]
        - Encoded label array
        - Fitted LabelEncoder
    """
    if encoder is None:
        encoder = LabelEncoder()

    if fit:
        y_encoded = encoder.fit_transform(y)
    else:
        y_encoded = encoder.transform(y)

    logger.info(f"Label classes: {list(encoder.classes_)}")
    logger.info(f"Number of classes: {len(encoder.classes_)}")

    return y_encoded, encoder


# =============================================================================
# 3. Train / Validation / Test Split
# =============================================================================

def split_data(
    X, y: np.ndarray, config: dict = None,
) -> Dict[str, Any]:
    """
    Split data into train / validation / test sets with stratification.

    Parameters
    ----------
    X : sparse matrix or array
        Feature matrix.
    y : np.ndarray
        Encoded label array.
    config : dict, optional
        Configuration dictionary.

    Returns
    -------
    dict
        Dictionary with keys: X_train, X_val, X_test, y_train, y_val, y_test,
        and their respective indices.
    """
    config = config or CONFIG
    seed = config["random_seed"]
    test_size = config["test_size"]
    val_size = config["val_size"]

    print_section("DATA SPLITTING")

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )

    # Second split: train vs val (from the remaining data)
    # Adjust val_size relative to the remaining data
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, random_state=seed, stratify=y_temp,
    )

    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }

    logger.info(f"Train set:      {X_train.shape[0]:>6,} samples ({X_train.shape[0] / len(y) * 100:.1f}%)")
    logger.info(f"Validation set: {X_val.shape[0]:>6,} samples ({X_val.shape[0] / len(y) * 100:.1f}%)")
    logger.info(f"Test set:       {X_test.shape[0]:>6,} samples ({X_test.shape[0] / len(y) * 100:.1f}%)")

    return splits


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

def select_best_feature_strategy(df: pd.DataFrame, y_encoded: np.ndarray, config: dict = None) -> str:
    """
    Evaluates TF-IDF vs CountVectorizer using a baseline Logistic Regression
    and returns the name of the best performing strategy.
    """
    config = config or CONFIG
    print_section("FEATURE STRATEGY SELECTION")
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config["random_seed"])
    model = LogisticRegression(max_iter=1000, random_state=config["random_seed"])
    
    scores = {}
    for strategy in ["tfidf", "count"]:
        config_copy = config.copy()
        config_copy["feature_strategy"] = strategy
        X, _ = build_text_features(df["cleaned_text"], config_copy)
        
        try:
            score = cross_val_score(model, X, y_encoded, cv=cv, scoring=config["scoring_metric"], n_jobs=-1).mean()
        except Exception as e:
            logger.warning(f"Strategy {strategy} failed during CV: {e}")
            score = 0
            
        scores[strategy] = score
        logger.info(f"Strategy '{strategy}' CV F1-Score: {score:.4f}")
        
    best_strategy = max(scores, key=scores.get)
    logger.info(f"Selected best feature strategy: {best_strategy}")
    return best_strategy


# =============================================================================
# 4. Full Feature Engineering Pipeline
# =============================================================================

def build_features(
    df: pd.DataFrame,
    config: dict = None,
) -> Tuple[Dict[str, Any], Any, LabelEncoder]:
    """
    Execute the full feature engineering pipeline.

    Steps:
        1. Extract TF-IDF features from cleaned text.
        2. Encode target labels.
        3. Split into train / val / test sets.
        4. Save vectorizer and encoder.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame with 'cleaned_text' and 'target' columns.
    config : dict, optional
        Configuration dictionary.

    Returns
    -------
    Tuple[dict, TfidfVectorizer, LabelEncoder]
        - Data splits dictionary
        - Fitted TF-IDF vectorizer
        - Fitted label encoder
    """
    config = config or CONFIG
    print_section("FEATURE ENGINEERING PIPELINE")

    # Encode labels first (needed for stratified splitting)
    y_encoded, label_encoder = encode_labels(df["target"])

    # Determine best feature strategy
    best_strategy = select_best_feature_strategy(df, y_encoded, config)
    config["feature_strategy"] = best_strategy

    # Build Text features
    X_text, vectorizer = build_text_features(df["cleaned_text"], config)

    # Split data
    splits = split_data(X_text, y_encoded, config)

    # Save artifacts
    models_dir = PATHS["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, models_dir / "tfidf_vectorizer.joblib")
    joblib.dump(label_encoder, models_dir / "label_encoder.joblib")
    logger.info(f"Saved vectorizer and label encoder to {models_dir}")

    # Log top features
    top_features = get_top_tfidf_features(vectorizer)
    if not top_features.empty:
        logger.info(f"Top common TF-IDF features: "
                    f"{list(top_features[top_features['type'] == 'most_common']['feature'][:10])}")

    return splits, vectorizer, label_encoder


def load_vectorizer(path=None) -> TfidfVectorizer:
    """Load a saved TF-IDF vectorizer."""
    path = path or PATHS["models_dir"] / "tfidf_vectorizer.joblib"
    return joblib.load(path)


def load_label_encoder(path=None) -> LabelEncoder:
    """Load a saved label encoder."""
    path = path or PATHS["models_dir"] / "label_encoder.joblib"
    return joblib.load(path)
