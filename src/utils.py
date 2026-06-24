# =============================================================================
# utils.py — Logging, Configuration, and Helper Functions
# =============================================================================
"""
Provides centralized configuration, logging setup, and reusable utilities
for the entire research pipeline.

Usage:
    from src.utils import get_logger, PATHS, CONFIG
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime


# =============================================================================
# 1. Project Paths & Dataset Configuration
# =============================================================================

# Dataset Selection
ACTIVE_DATASET = "dataset_2"

# Resolve the project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure directories exist
PATHS = {
    "project_root": PROJECT_ROOT,
    "data_dir": PROJECT_ROOT / "data",
    "raw_data_dir": PROJECT_ROOT / "data" / ACTIVE_DATASET / "raw",
    "processed_data_dir": PROJECT_ROOT / "data" / ACTIVE_DATASET / "processed",
    "models_dir": PROJECT_ROOT / "models" / ACTIVE_DATASET,
    "reports_dir": PROJECT_ROOT / "reports" / ACTIVE_DATASET,
    "figures_dir": PROJECT_ROOT / "reports" / ACTIVE_DATASET / "figures",
    "pipeline_logs_dir": PROJECT_ROOT / "reports" / ACTIVE_DATASET / "pipeline_logs",
    "notebooks_dir": PROJECT_ROOT / "notebooks",
}

# =============================================================================
# 2. Pipeline Configuration
# =============================================================================

CONFIG = {
    # --- Data ---
    "random_seed": 42,
    "test_size": 0.15,          # 15% held-out test set
    "val_size": 0.15,           # 15% validation (from remaining 85%)

    # --- Text Preprocessing ---
    "text_columns": ["title", "body"],      # columns to combine for NLP
    "lowercase": True,
    "remove_urls": True,
    "remove_punctuation": True,
    "remove_stopwords": True,
    "use_lemmatization": True,              # False → use stemming instead
    "min_token_length": 2,

    # --- Feature Engineering ---
    "tfidf_max_features": 10000,
    "tfidf_ngram_range": (1, 2),            # unigrams + bigrams
    "tfidf_min_df": 3,
    "tfidf_max_df": 0.95,

    # --- Target Engineering ---
    "min_class_samples": 30 if ACTIVE_DATASET == "dataset_2" else 100, 
    "max_classes": 15,
    "target_column": "inner category" if ACTIVE_DATASET == "dataset_2" else "labels",
    "target_mode": "direct" if ACTIVE_DATASET == "dataset_2" else "parse_labels",

    # --- Training ---
    "cv_folds": 5,
    "n_jobs": -1,
    "scoring_metric": "f1_weighted",
    "hyperparameter_tuning_n_iter": 30,     # for RandomizedSearchCV
}


# =============================================================================
# 3. Logging
# =============================================================================

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a configured logger with both console and file output.

    Parameters
    ----------
    name : str
        Logger name (typically __name__ of the calling module).
    level : int
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    log_dir = PATHS["pipeline_logs_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


# =============================================================================
# 4. Helper Functions
# =============================================================================

def ensure_dirs():
    """Create all project directories if they don't exist."""
    for key, path in PATHS.items():
        if key.endswith("_dir"):
            path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, filepath: Path):
    """Save a dictionary as a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Path) -> dict:
    """Load a JSON file into a dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def timestamp() -> str:
    """Return a filesystem-safe timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_section(title: str, width: int = 70):
    """Print a formatted section header."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")
