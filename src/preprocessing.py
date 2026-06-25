# =============================================================================
# preprocessing.py — Data Loading, Cleaning, and Text Preprocessing
# =============================================================================
"""
Handles all data ingestion, inspection, cleaning, and NLP preprocessing steps.

Pipeline stages:
    1. Load raw JSON data into a Pandas DataFrame.
    2. Inspect and report dataset statistics (missing values, duplicates, etc.).
    3. Parse and engineer the classification target from issue labels.
    4. Clean and preprocess text fields (title + body).

Usage:
    from src.preprocessing import load_data, inspect_data, preprocess_pipeline
"""

import re
import ast
import json
import warnings
from collections import Counter
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import nltk
for resource, path in [
    ('stopwords', 'corpora/stopwords'),
    ('wordnet', 'corpora/wordnet'),
    ('omw-1.4', 'corpora/omw-1.4'),
    ('punkt', 'tokenizers/punkt')
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

from src.utils import get_logger, PATHS, CONFIG, print_section

warnings.filterwarnings("ignore")

logger = get_logger(__name__)


# =============================================================================
# 1. Data Loading
# =============================================================================

def load_data(filepath=None) -> pd.DataFrame:
    """
    Load the raw dataset file into a DataFrame, supporting JSON, CSV, and XLSX.
    Automatically fetches missing textual features if URLs are provided.
    """
    if filepath is None:
        raw_dir = PATHS["raw_data_dir"]
        files = list(raw_dir.glob("*"))
        valid_files = [f for f in files if f.suffix in ['.json', '.csv', '.xlsx'] and not f.name.startswith('~')]
        if not valid_files:
            raise FileNotFoundError(f"No valid data files found in {raw_dir}")
        filepath = valid_files[0]

    logger.info(f"Loading data from: {filepath}")

    if filepath.suffix == '.json':
        with open(filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        df = pd.DataFrame(raw_data)
    elif filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
    elif filepath.suffix == '.xlsx':
        xl = pd.ExcelFile(filepath)
        dfs = []
        for sheet in xl.sheet_names:
            sheet_df = xl.parse(sheet)
            sheet_df.columns = [str(c).strip().lower() for c in sheet_df.columns]
            dfs.append(sheet_df)
        df = pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    # Fix known typos in raw data right away
    if "inner category" in df.columns:
        df["inner category"] = df["inner category"].replace({"[A] Model conversion": "[A] Model Conversion"})

    # Data collection step (fetch missing text)
    if "title" not in df.columns or "body" not in df.columns:
        if "url" in df.columns:
            logger.info("Dataset lacks 'title' and/or 'body'. Attempting to collect text using data collector...")
            from src.data_collector import DataCollector
            cache_path = PATHS["raw_data_dir"] / "scraped_text_cache.json"
            collector = DataCollector(cache_path)
            
            titles, bodies = [], []
            for url in df["url"]:
                title, body = collector.fetch_github_issue(url)
                titles.append(title)
                bodies.append(body)
                
            df["title"] = titles
            df["body"] = bodies
        else:
            logger.warning("Dataset lacks textual columns and 'url' column. Feature extraction may fail.")

    logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")

    return df


# =============================================================================
# 2. Data Inspection & EDA Statistics
# =============================================================================

def inspect_data(df: pd.DataFrame) -> dict:
    """
    Perform comprehensive data inspection and return a statistics dictionary.

    Generates:
        - Shape, dtypes, memory usage
        - Missing value counts per column
        - Duplicate record counts
        - Basic text length statistics
        - Label distribution summary

    Parameters
    ----------
    df : pd.DataFrame
        The raw (or partially processed) DataFrame.

    Returns
    -------
    dict
        Dictionary of inspection statistics.
    """
    print_section("DATA INSPECTION REPORT")
    stats = {}

    # --- Shape & Memory ---
    stats["n_rows"], stats["n_cols"] = df.shape
    stats["memory_mb"] = round(df.memory_usage(deep=True).sum() / 1e6, 2)
    logger.info(f"Shape: {stats['n_rows']:,} rows × {stats['n_cols']} columns")
    logger.info(f"Memory usage: {stats['memory_mb']} MB")

    # --- Column Types ---
    stats["dtypes"] = df.dtypes.astype(str).to_dict()
    print("\nColumn Data Types:")
    for col, dtype in stats["dtypes"].items():
        print(f"  {col:<20s} {dtype}")

    # --- Missing Values ---
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    stats["missing_values"] = missing[missing > 0].to_dict()
    stats["missing_pct"] = missing_pct[missing_pct > 0].to_dict()
    print("\nMissing Values:")
    if stats["missing_values"]:
        for col, count in stats["missing_values"].items():
            print(f"  {col:<20s} {count:>6,} ({stats['missing_pct'][col]:.2f}%)")
    else:
        print("  No missing values detected.")

    # --- Duplicate Records ---
    # Only check hashable columns (exclude lists/dicts which can't be hashed)
    hashable_cols = []
    for col in df.columns:
        try:
            df[col].apply(hash)
            hashable_cols.append(col)
        except (TypeError, ValueError):
            # Column contains unhashable types like list or dict
            pass
    if hashable_cols:
        n_duplicates = df[hashable_cols].duplicated().sum()
    else:
        n_duplicates = 0
    stats["n_duplicates"] = int(n_duplicates)
    logger.info(f"Duplicate rows (hashable columns): {n_duplicates:,}")

    # --- Check for duplicate IDs ---
    if "_id" in df.columns:
        n_dup_ids = df["_id"].duplicated().sum()
        stats["n_duplicate_ids"] = int(n_dup_ids)
        logger.info(f"Duplicate _id values: {n_dup_ids:,}")

    # --- Text Length Statistics ---
    print("\nText Length Statistics (character counts):")
    for col in ["title", "body"]:
        if col in df.columns:
            lengths = df[col].fillna("").str.len()
            col_stats = {
                "mean": round(lengths.mean(), 1),
                "median": round(lengths.median(), 1),
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "std": round(lengths.std(), 1),
            }
            stats[f"{col}_length_stats"] = col_stats
            print(f"  {col}: mean={col_stats['mean']}, "
                  f"median={col_stats['median']}, "
                  f"min={col_stats['min']}, max={col_stats['max']}")

    # --- Empty text checks ---
    for col in ["title", "body"]:
        if col in df.columns:
            empty_count = (df[col].fillna("").str.strip() == "").sum()
            stats[f"{col}_empty_count"] = int(empty_count)
            logger.info(f"Empty '{col}' fields: {empty_count:,}")

    return stats


# =============================================================================
# 3. Label Parsing & Target Engineering
# =============================================================================

def _parse_labels(label_field) -> list:
    """
    Safely parse the 'labels' field from JSON string to a list of label names.

    Handles:
        - String representations of lists of dicts
        - Already-parsed lists
        - None / NaN values
        - numpy arrays
    """
    # Handle None, NaN, and empty values
    if label_field is None:
        return []

    # Handle scalar NaN (but not lists/arrays which fail pd.isna)
    try:
        if not isinstance(label_field, (list, dict)) and pd.isna(label_field):
            return []
    except (ValueError, TypeError):
        pass

    if isinstance(label_field, list):
        labels_list = label_field
    elif isinstance(label_field, str):
        if not label_field.strip() or label_field.strip() == "[]":
            return []
        try:
            labels_list = ast.literal_eval(label_field)
        except (ValueError, SyntaxError):
            try:
                labels_list = json.loads(label_field)
            except json.JSONDecodeError:
                return []
    else:
        return []

    if not isinstance(labels_list, list):
        return []

    names = []
    for item in labels_list:
        if isinstance(item, dict) and "name" in item:
            names.append(item["name"].lower().strip())
        elif isinstance(item, str):
            names.append(item.lower().strip())
    return names


def engineer_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Automatically determine and engineer the classification target based on config.
    """
    print_section("TARGET ENGINEERING")

    min_samples = CONFIG["min_class_samples"]
    max_classes = CONFIG["max_classes"]
    target_mode = CONFIG.get("target_mode", "parse_labels")
    target_column = CONFIG.get("target_column", "labels")

    if target_mode == "parse_labels":
        # Parse all labels (Dataset 1 behavior)
        logger.info(f"Parsing label fields from '{target_column}'...")
        df["parsed_labels"] = df[target_column].apply(_parse_labels)

        # Count all label occurrences
        all_labels = []
        for labels in df["parsed_labels"]:
            all_labels.extend(labels)
        label_counts = Counter(all_labels)

        logger.info(f"Total unique labels found: {len(label_counts)}")

        # Define workflow/meta labels to exclude (not useful for defect classification)
        exclude_labels = {
            "bug", "stale", "pr exists", "approved", "ready for review",
            "ready for merge", "in progress", "tested", "resolved",
            "needs info", "needs info/discussion", "confirmed",
            "needs triage", "triaged", "merged", "closed", "reopened",
            "won't fix", "wontfix", "needs reproduction",
        }

        # Extract primary (most meaningful) secondary label
        def _extract_target(label_list):
            meaningful = [l for l in label_list if l not in exclude_labels]
            if not meaningful:
                return "bug_only"
            meaningful.sort(key=lambda x: label_counts.get(x, 0), reverse=True)
            return meaningful[0]

        df["target"] = df["parsed_labels"].apply(_extract_target)
    elif target_mode == "direct":
        # Direct extraction from specified column (Dataset 2 behavior)
        logger.info(f"Directly using target column '{target_column}'...")
        df["target"] = df[target_column].astype(str).str.strip()
        
        # Specific fix for known typos in Dataset 2
        df["target"] = df["target"].replace({"[A] Model conversion": "[A] Model Conversion"})
        
        df["target"] = df["target"].replace({"nan": "unknown", "": "unknown", "None": "unknown"})
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    # Filter to classes with sufficient samples
    target_counts = df["target"].value_counts()
    valid_classes = target_counts[target_counts >= min_samples].index.tolist()

    # Cap at max_classes (keep most frequent)
    if len(valid_classes) > max_classes:
        valid_classes = valid_classes[:max_classes]

    logger.info(f"Classes with >= {min_samples} samples: {len(valid_classes)}")
    logger.info(f"Keeping top {len(valid_classes)} classes")

    # Filter DataFrame
    df_filtered = df[df["target"].isin(valid_classes)].copy()
    df_filtered["target"] = df_filtered["target"].astype("category")

    # Compute distribution stats
    target_dist = df_filtered["target"].value_counts().to_dict()
    total = len(df_filtered)
    target_pct = {k: round(v / total * 100, 2) for k, v in target_dist.items()}

    target_stats = {
        "n_classes": len(valid_classes),
        "class_names": valid_classes,
        "class_distribution": target_dist,
        "class_percentages": target_pct,
        "total_samples": total,
        "dropped_samples": len(df) - total,
        "imbalance_ratio": round(
            max(target_dist.values()) / min(target_dist.values()), 2
        ),
    }

    print("\nTarget Distribution:")
    for cls, count in sorted(target_dist.items(), key=lambda x: -x[1]):
        print(f"  {cls:<25s} {count:>6,}  ({target_pct[cls]:>5.2f}%)")
    print(f"\n  Total: {total:,} samples across {len(valid_classes)} classes")
    print(f"  Imbalance ratio: {target_stats['imbalance_ratio']}:1")

    return df_filtered, target_stats


# =============================================================================
# 4. Text Preprocessing
# =============================================================================

class TextPreprocessor:
    """
    NLP text preprocessing pipeline for GitHub issue text.

    Steps:
        1. Combine title + body into a single text field
        2. Lowercase
        3. Remove URLs, HTML tags, code blocks
        4. Remove punctuation and special characters
        5. Tokenize
        6. Remove stop words
        7. Lemmatize or stem tokens
        8. Reconstruct cleaned text

    Parameters
    ----------
    config : dict
        Configuration dictionary (from utils.CONFIG).
    """

    def __init__(self, config: dict = None):
        self.config = config or CONFIG
        self._stop_words = set(stopwords.words("english"))

        # Add domain-specific stop words
        self._stop_words.update([
            "http", "https", "www", "github", "com", "android",
            "app", "issue", "bug", "error", "would", "could",
            "also", "using", "used", "use", "one", "get", "got",
            "like", "see", "try", "need", "want", "make",
        ])

        if self.config.get("use_lemmatization", True):
            self._lemmatizer = WordNetLemmatizer()
            self._normalize = self._lemmatizer.lemmatize
        else:
            self._stemmer = PorterStemmer()
            self._normalize = self._stemmer.stem

        # Compiled regex patterns for speed
        self._url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self._html_pattern = re.compile(r"<[^>]+>")
        self._code_block_pattern = re.compile(r"```[\s\S]*?```")
        self._inline_code_pattern = re.compile(r"`[^`]+`")
        self._punctuation_pattern = re.compile(r"[^\w\s]")
        self._numbers_pattern = re.compile(r"\b\d+\b")
        self._whitespace_pattern = re.compile(r"\s+")

    def _combine_text(self, row: pd.Series) -> str:
        """Combine title and body into a single string."""
        parts = []
        for col in self.config.get("text_columns", ["title", "body"]):
            val = row.get(col, "")
            if pd.notna(val) and str(val).strip():
                parts.append(str(val).strip())
        return " ".join(parts)

    def _clean_text(self, text: str) -> str:
        """Apply all text cleaning transformations."""
        if not text or not isinstance(text, str):
            return ""

        # Remove code blocks and inline code
        text = self._code_block_pattern.sub(" ", text)
        text = self._inline_code_pattern.sub(" ", text)

        # Remove HTML tags
        text = self._html_pattern.sub(" ", text)

        # Remove URLs
        if self.config.get("remove_urls", True):
            text = self._url_pattern.sub(" ", text)

        # Lowercase
        if self.config.get("lowercase", True):
            text = text.lower()

        # Remove punctuation
        if self.config.get("remove_punctuation", True):
            text = self._punctuation_pattern.sub(" ", text)

        # Remove standalone numbers
        text = self._numbers_pattern.sub(" ", text)

        # Normalize whitespace
        text = self._whitespace_pattern.sub(" ", text).strip()

        return text

    def _tokenize_and_normalize(self, text: str) -> str:
        """Tokenize, remove stop words, and lemmatize/stem."""
        if not text:
            return ""

        # Simple whitespace tokenization (faster than NLTK for clean text)
        tokens = text.split()

        min_len = self.config.get("min_token_length", 2)

        # Filter and normalize
        processed = []
        for token in tokens:
            if len(token) < min_len:
                continue
            if self.config.get("remove_stopwords", True) and token in self._stop_words:
                continue
            processed.append(self._normalize(token))

        return " ".join(processed)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full text preprocessing pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with text columns (title, body).

        Returns
        -------
        pd.DataFrame
            DataFrame with a new 'cleaned_text' column.
        """
        print_section("TEXT PREPROCESSING")
        logger.info("Combining text fields (title + body)...")

        df = df.copy()
        df["raw_text"] = df.apply(self._combine_text, axis=1)

        logger.info("Cleaning text (URLs, HTML, code, punctuation)...")
        df["cleaned_text"] = df["raw_text"].apply(self._clean_text)

        logger.info("Tokenizing and normalizing (stopwords, lemmatization)...")
        df["cleaned_text"] = df["cleaned_text"].apply(self._tokenize_and_normalize)

        # Remove empty text rows
        empty_mask = df["cleaned_text"].str.strip() == ""
        n_empty = empty_mask.sum()
        if n_empty > 0:
            logger.warning(f"Removing {n_empty} rows with empty text after cleaning")
            df = df[~empty_mask].copy()

        # Log statistics
        text_lengths = df["cleaned_text"].str.split().str.len()
        logger.info(f"Token count stats after preprocessing:")
        logger.info(f"  Mean: {text_lengths.mean():.1f}, "
                    f"Median: {text_lengths.median():.1f}, "
                    f"Min: {text_lengths.min()}, Max: {text_lengths.max()}")

        return df


# =============================================================================
# 5. Full Preprocessing Pipeline
# =============================================================================

def preprocess_pipeline(filepath=None) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Execute the complete preprocessing pipeline end-to-end.

    Steps:
        1. Load raw data
        2. Inspect dataset
        3. Engineer classification target
        4. Preprocess text

    Parameters
    ----------
    filepath : str or Path, optional
        Path to the raw JSON file.

    Returns
    -------
    Tuple[pd.DataFrame, dict, dict]
        - Cleaned DataFrame ready for feature engineering
        - Data inspection statistics
        - Target engineering statistics
    """
    # Step 1: Load
    df = load_data(filepath)

    # Step 2: Inspect
    inspection_stats = inspect_data(df)

    # Step 3: Engineer target
    df, target_stats = engineer_target(df)

    # Step 4: Preprocess text
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess(df)

    logger.info(f"Final dataset shape: {df.shape}")

    return df, inspection_stats, target_stats


if __name__ == "__main__":
    # Run as standalone script for testing
    df, insp_stats, tgt_stats = preprocess_pipeline()
    print(f"\nFinal DataFrame shape: {df.shape}")
    print(f"Target classes: {tgt_stats['class_names']}")
