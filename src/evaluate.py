# =============================================================================
# evaluate.py — Model Evaluation, Visualization, and Reporting
# =============================================================================
"""
Comprehensive evaluation of trained models with metrics, visualizations,
and comparison tables suitable for a research paper.

Metrics:
    - Accuracy, Precision, Recall, F1-score (per-class + weighted)
    - Confusion Matrix
    - ROC-AUC (One-vs-Rest for multi-class)
    - Classification Report

Visualizations:
    - Class distribution bar chart
    - Confusion matrix heatmaps
    - Model comparison charts
    - TF-IDF feature importance
    - ROC curves (per class)

Usage:
    from src.evaluate import evaluate_model, evaluate_all, generate_visualizations
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from typing import Dict, Any, List, Optional, Tuple
from src.utils import get_logger, PATHS, CONFIG, print_section

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

# --- Plot Style Configuration ---
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
FIGSIZE_STANDARD = (10, 6)
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (8, 8)
DPI = 150


# =============================================================================
# 1. Single Model Evaluation
# =============================================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str,
    class_names: list,
) -> dict:
    """
    Evaluate a single model on the test set.

    Parameters
    ----------
    model : estimator
        Trained sklearn-compatible model.
    X_test : sparse matrix or array
        Test feature matrix.
    y_test : np.ndarray
        True test labels.
    model_name : str
        Name of the model (for logging/reporting).
    class_names : list
        List of human-readable class names.

    Returns
    -------
    dict
        Dictionary containing all evaluation metrics.
    """
    logger.info(f"Evaluating: {model_name}")

    y_pred = model.predict(X_test)

    metrics = {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision_weighted": round(
            precision_score(y_test, y_pred, average="weighted", zero_division=0), 4
        ),
        "Recall_weighted": round(
            recall_score(y_test, y_pred, average="weighted", zero_division=0), 4
        ),
        "F1_weighted": round(
            f1_score(y_test, y_pred, average="weighted", zero_division=0), 4
        ),
        "F1_macro": round(
            f1_score(y_test, y_pred, average="macro", zero_division=0), 4
        ),
    }

    # ROC-AUC (requires probability estimates)
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
            metrics["ROC_AUC_weighted"] = round(
                roc_auc_score(
                    y_test_bin, y_proba, average="weighted", multi_class="ovr",
                ),
                4,
            )
        else:
            metrics["ROC_AUC_weighted"] = "N/A"
    except Exception as e:
        metrics["ROC_AUC_weighted"] = f"Error: {str(e)[:50]}"

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred)

    # Full classification report
    metrics["classification_report"] = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0,
    )

    # Per-class metrics
    metrics["per_class_f1"] = f1_score(
        y_test, y_pred, average=None, zero_division=0,
    )

    logger.info(f"  Accuracy: {metrics['Accuracy']}")
    logger.info(f"  F1 (weighted): {metrics['F1_weighted']}")
    logger.info(f"  ROC-AUC: {metrics.get('ROC_AUC_weighted', 'N/A')}")

    return metrics


# =============================================================================
# 2. Multi-Model Evaluation
# =============================================================================

def evaluate_all(
    models: Dict[str, Any],
    X_test,
    y_test,
    class_names: list,
) -> Tuple:
    """
    Evaluate multiple trained models and compile a comparison table.

    Parameters
    ----------
    models : dict
        Dictionary of model_name → fitted estimator.
    X_test : sparse matrix or array
        Test feature matrix.
    y_test : np.ndarray
        True test labels.
    class_names : list
        Class names for display.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        - Comparison DataFrame
        - Full metrics dictionary per model
    """
    print_section("EVALUATING ALL MODELS ON TEST SET")

    all_metrics = {}
    comparison_rows = []

    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name, class_names)
        all_metrics[name] = metrics

        comparison_rows.append({
            "Model": name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision_weighted"],
            "Recall": metrics["Recall_weighted"],
            "F1 (weighted)": metrics["F1_weighted"],
            "F1 (macro)": metrics["F1_macro"],
            "ROC-AUC": metrics["ROC_AUC_weighted"],
        })

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        "F1 (weighted)", ascending=False,
    ).reset_index(drop=True)

    print("\n" + "=" * 100)
    print("  FINAL MODEL COMPARISON — Test Set Metrics")
    print("=" * 100)
    print(comparison_df.to_string(index=False))

    # Save comparison to CSV
    reports_dir = PATHS["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(reports_dir / "model_comparison.csv", index=False)
    logger.info(f"Comparison table saved to {reports_dir / 'model_comparison.csv'}")

    return comparison_df, all_metrics


# =============================================================================
# 3. Visualization Functions
# =============================================================================

def plot_class_distribution(
    y: pd.Series,
    class_names: list = None,
    title: str = "Class Distribution",
    filename: str = "class_distribution.png",
):
    """Plot a bar chart of class distribution."""
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    if class_names is not None:
        counts = pd.Series(y).value_counts().sort_index()
        labels = [class_names[i] if i < len(class_names) else str(i) for i in counts.index]
    else:
        counts = y.value_counts()
        labels = counts.index.tolist()

    colors = sns.color_palette("viridis", len(counts))
    bars = ax.bar(range(len(counts)), counts.values, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    # Add count labels on bars
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            f"{count:,}", ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    plt.tight_layout()
    save_path = PATHS["figures_dir"] / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: list,
    model_name: str,
    filename: str = None,
    normalize: bool = True,
    cm: np.ndarray = None,
):
    """Plot a confusion matrix heatmap."""
    if filename is None:
        filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"

    if cm is None:
        cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
        title_suffix = "(Normalized)"
    else:
        cm_display = cm
        fmt = "d"
        title_suffix = "(Counts)"

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.8),
                                     max(6, len(class_names) * 0.7)))

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name} {title_suffix}",
                 fontsize=12, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    save_path = PATHS["figures_dir"] / filename
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    filename: str = "model_comparison.png",
):
    """Plot a grouped bar chart comparing all models across metrics."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 (weighted)", "F1 (macro)"]
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]

    plot_df = comparison_df[["Model"] + available_metrics].copy()
    plot_df = plot_df.set_index("Model")

    x = np.arange(len(plot_df))
    width = 0.15
    colors = sns.color_palette("Set2", len(available_metrics))

    for i, metric in enumerate(available_metrics):
        offset = (i - len(available_metrics) / 2 + 0.5) * width
        bars = ax.bar(x + offset, plot_df[metric], width, label=metric,
                      color=colors[i], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — All Metrics", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=30, ha="right", fontsize=9)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    plt.tight_layout()
    save_path = PATHS["figures_dir"] / filename
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_f1_comparison(
    comparison_df: pd.DataFrame,
    filename: str = "f1_comparison.png",
):
    """Plot a horizontal bar chart of F1 scores for all models."""
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    df_sorted = comparison_df.sort_values("F1 (weighted)", ascending=True)
    colors = sns.color_palette("RdYlGn", len(df_sorted))

    bars = ax.barh(
        df_sorted["Model"], df_sorted["F1 (weighted)"],
        color=colors, edgecolor="white", linewidth=0.5,
    )

    for bar, score in zip(bars, df_sorted["F1 (weighted)"]):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}", va="center", fontsize=9, fontweight="bold",
        )

    ax.set_xlabel("F1 Score (Weighted)", fontsize=11)
    ax.set_title("Model Comparison — F1 Score (Weighted)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    save_path = PATHS["figures_dir"] / filename
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_tfidf_importance(
    vectorizer,
    model=None,
    class_names: list = None,
    top_n: int = 20,
    filename: str = "tfidf_importance.png",
):
    """
    Plot top TF-IDF features by IDF score.

    If a linear model is provided, shows feature importance by coefficient magnitude.
    """
    feature_names = vectorizer.get_feature_names_out()
    idf_scores = vectorizer.idf_

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # --- Most common features (lowest IDF) ---
    top_indices = np.argsort(idf_scores)[:top_n]
    ax = axes[0]
    ax.barh(
        [feature_names[i] for i in top_indices],
        idf_scores[top_indices],
        color=sns.color_palette("Blues_r", top_n),
        edgecolor="white",
    )
    ax.set_xlabel("IDF Score", fontsize=10)
    ax.set_title(f"Top {top_n} Most Common Features", fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    # --- Rarest features (highest IDF) ---
    bottom_indices = np.argsort(idf_scores)[-top_n:]
    ax = axes[1]
    ax.barh(
        [feature_names[i] for i in bottom_indices],
        idf_scores[bottom_indices],
        color=sns.color_palette("Reds", top_n),
        edgecolor="white",
    )
    ax.set_xlabel("IDF Score", fontsize=10)
    ax.set_title(f"Top {top_n} Rarest Features", fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    plt.suptitle("TF-IDF Feature Analysis", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = PATHS["figures_dir"] / filename
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_label_distribution_pie(
    target_stats: dict,
    filename: str = "label_distribution_pie.png",
):
    """Plot a pie chart of label distribution."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    dist = target_stats["class_distribution"]
    labels = list(dist.keys())
    sizes = list(dist.values())
    colors = sns.color_palette("Set3", len(labels))

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%\n({int(pct / 100 * sum(sizes)):,})",
        colors=colors,
        startangle=90,
        pctdistance=0.8,
        textprops={"fontsize": 8},
    )

    for autotext in autotexts:
        autotext.set_fontsize(7)

    ax.set_title("Target Label Distribution", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()

    save_path = PATHS["figures_dir"] / filename
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_text_length_distribution(
    df: pd.DataFrame,
    filename: str = "text_length_distribution.png",
):
    """Plot distribution of text lengths before and after cleaning."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Raw text length
    if "raw_text" in df.columns:
        lengths = df["raw_text"].str.len()
        axes[0].hist(lengths, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
        axes[0].set_xlabel("Character Count", fontsize=10)
        axes[0].set_ylabel("Frequency", fontsize=10)
        axes[0].set_title("Raw Text Length Distribution", fontsize=11, fontweight="bold")
        axes[0].axvline(lengths.median(), color="red", linestyle="--",
                        label=f"Median: {lengths.median():.0f}")
        axes[0].legend()

    # Cleaned text token count
    if "cleaned_text" in df.columns:
        token_counts = df["cleaned_text"].str.split().str.len()
        axes[1].hist(token_counts, bins=50, color="#2ecc71", edgecolor="white", alpha=0.8)
        axes[1].set_xlabel("Token Count", fontsize=10)
        axes[1].set_ylabel("Frequency", fontsize=10)
        axes[1].set_title("Cleaned Text Token Distribution", fontsize=11, fontweight="bold")
        axes[1].axvline(token_counts.median(), color="red", linestyle="--",
                        label=f"Median: {token_counts.median():.0f}")
        axes[1].legend()

    plt.suptitle("Text Length Analysis", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = PATHS["figures_dir"] / filename
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_per_class_f1(
    all_metrics: dict,
    class_names: list,
    filename: str = "per_class_f1.png",
):
    """Plot per-class F1 scores for each model as a grouped bar chart."""
    fig, ax = plt.subplots(figsize=(max(12, len(class_names) * 1.2), 7))

    models = list(all_metrics.keys())
    x = np.arange(len(class_names))
    width = 0.8 / len(models)
    colors = sns.color_palette("tab10", len(models))

    for i, model_name in enumerate(models):
        f1_scores = all_metrics[model_name].get("per_class_f1", [])
        if len(f1_scores) > 0:
            offset = (i - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, f1_scores, width, label=model_name,
                   color=colors[i], edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Per-Class F1 Scores by Model", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    save_path = PATHS["figures_dir"] / filename
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


# =============================================================================
# 4. Generate All Visualizations
# =============================================================================

def generate_visualizations(
    df: pd.DataFrame,
    target_stats: dict,
    comparison_df: pd.DataFrame,
    all_metrics: dict,
    vectorizer=None,
    class_names: list = None,
):
    """
    Generate all visualization artifacts.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame.
    target_stats : dict
        Target engineering statistics.
    comparison_df : pd.DataFrame
        Model comparison DataFrame.
    all_metrics : dict
        Per-model evaluation metrics.
    vectorizer : TfidfVectorizer, optional
        Fitted TF-IDF vectorizer.
    class_names : list, optional
        Human-readable class names.
    """
    print_section("GENERATING VISUALIZATIONS")

    figures_dir = PATHS["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Class distribution
    if "target" in df.columns:
        plot_class_distribution(df["target"], title="Target Class Distribution")

    # 2. Label distribution pie chart
    if target_stats:
        plot_label_distribution_pie(target_stats)

    # 3. Text length distributions
    plot_text_length_distribution(df)

    # 4. TF-IDF feature importance
    if vectorizer is not None:
        plot_tfidf_importance(vectorizer)

    # 5. Model comparison charts
    if comparison_df is not None and len(comparison_df) > 0:
        plot_model_comparison(comparison_df)
        plot_f1_comparison(comparison_df)

    # 6. Confusion matrices for all evaluated models
    if all_metrics:
        for model_name, metrics in all_metrics.items():
            if "confusion_matrix" in metrics and class_names:
                # Re-create predictions from confusion matrix
                cm = metrics["confusion_matrix"]
                plot_confusion_matrix(
                    y_true=None, y_pred=None,
                    class_names=class_names,
                    model_name=model_name,
                    cm=cm,
                )

    # 7. Per-class F1 scores
    if all_metrics and class_names:
        plot_per_class_f1(all_metrics, class_names)

    logger.info(f"All visualizations saved to: {figures_dir}")


def generate_confusion_matrices(
    models: Dict[str, Any],
    X_test,
    y_test,
    class_names: list,
):
    """Generate confusion matrix plots for each model."""
    for name, model in models.items():
        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, class_names, name)
