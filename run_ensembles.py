# =============================================================================
# run_ensembles.py -- Phase 2: Ensemble Learning Pipeline
# =============================================================================
"""
Ensemble Learning extension for the Software Defect Classification pipeline.

This script:
    1. Reloads saved TF-IDF features, label encoder, and preprocessed data.
    2. Reconstructs the EXACT same train/test split used by baseline models.
    3. Trains Hard Voting, Soft Voting, and Stacking ensemble classifiers.
    4. Evaluates ensembles on the held-out test set.
    5. Compares ensemble results against previously saved baseline results.
    6. Generates comparison visualizations and a markdown summary report.
    7. Saves all ensemble models and updates best_model.joblib if improved.

Usage:
    python run_ensembles.py

Note:
    - Does NOT modify baseline results or retrain baseline models.
    - Reuses existing preprocessing, features, and train/test split exactly.
    - Stacking Classifier trains 5 base learners x 5 CV folds internally,
      so it may take 30-45 minutes on large datasets.
"""

import sys
import time
import warnings
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.sparse import vstack

from src.utils import (
    get_logger, PATHS, CONFIG, ensure_dirs, save_json, print_section, timestamp,
)
from src.feature_engineering import split_data
from src.train import (
    get_model_registry, get_ensemble_registry, train_final_model,
)
from src.evaluate import (
    evaluate_model, evaluate_all, generate_confusion_matrices,
    plot_model_comparison, plot_f1_comparison, plot_per_class_f1,
)

warnings.filterwarnings("ignore")
logger = get_logger("ensemble_pipeline")


# =============================================================================
# Ensemble-specific comparison charts
# =============================================================================

def plot_metric_comparison(
    comparison_df: pd.DataFrame,
    metric: str,
    title: str,
    filename: str,
):
    """
    Plot a horizontal bar chart comparing a single metric across all models.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Full model comparison table.
    metric : str
        Column name of the metric to plot.
    title : str
        Plot title.
    filename : str
        Output filename.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 6))

    df_sorted = comparison_df.sort_values(metric, ascending=True)

    # Color ensemble models differently
    colors = []
    for model_name in df_sorted["Model"]:
        if "(Ensemble)" in model_name:
            colors.append("#e74c3c")  # Red for ensembles
        else:
            colors.append("#3498db")  # Blue for baselines
    
    bars = ax.barh(
        df_sorted["Model"], df_sorted[metric],
        color=colors, edgecolor="white", linewidth=0.5,
    )

    for bar, score in zip(bars, df_sorted[metric]):
        ax.text(
            bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}", va="center", fontsize=9, fontweight="bold",
        )

    ax.set_xlabel(metric, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Ensemble"),
        Patch(facecolor="#3498db", label="Baseline"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    save_path = PATHS["figures_dir"] / filename
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def generate_ensemble_report(
    combined_df: pd.DataFrame,
    ensemble_metrics: dict,
    baseline_best_f1: float,
    baseline_best_name: str,
    ensemble_registry: dict,
    class_names: list,
    run_id: str,
    elapsed_minutes: float,
    vectorizer = None,
    total_samples: int = 0,
):
    """
    Generate the ensemble_results.md markdown report.
    """
    report_path = PATHS["reports_dir"] / "ensemble_results.md"

    # Determine the overall best model
    overall_best = combined_df.iloc[0]
    overall_best_name = overall_best["Model"]
    overall_best_f1 = overall_best["F1 (weighted)"]

    # Did ensembles improve?
    ensemble_rows = combined_df[combined_df["Model"].str.contains("Ensemble")]
    best_ensemble = ensemble_rows.iloc[0] if len(ensemble_rows) > 0 else None

    improved = best_ensemble is not None and best_ensemble["F1 (weighted)"] > baseline_best_f1

    vec_type = "TF-IDF" if hasattr(vectorizer, "idf_") else "Count"
    n_features = len(vectorizer.vocabulary_) if vectorizer else 0
    dataset_name = "Mobile Application GitHub Issues" if total_samples > 1000 else "Functional Bugs Annotated Dataset"

    lines = []
    lines.append("# Phase 2: Ensemble Learning Results")
    lines.append("")
    lines.append(f"**Run ID:** {run_id}  ")
    lines.append(f"**Total Training Time:** {elapsed_minutes:.1f} minutes  ")
    lines.append(f"**Dataset:** {dataset_name} ({total_samples:,} samples, {len(class_names)} classes)  ")
    lines.append(f"**Features:** {vec_type} ({n_features:,} features)  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # --- Ensemble Architecture ---
    lines.append("## Ensemble Architecture")
    lines.append("")
    for ens_name, ens_info in ensemble_registry.items():
        lines.append(f"### {ens_name}")
        lines.append("")
        lines.append(f"**Description:** {ens_info['description']}")
        lines.append("")
        lines.append(f"**Base Learners:** {', '.join(ens_info['base_learners'])}")
        lines.append("")
        if ens_info.get("meta_learner"):
            lines.append(f"**Meta Learner:** {ens_info['meta_learner']}")
            lines.append("")
        lines.append("---")
        lines.append("")

    # --- Full Comparison Table ---
    lines.append("## Performance Comparison (All Models)")
    lines.append("")
    lines.append("| Model | Accuracy | Precision | Recall | F1 (weighted) | F1 (macro) | ROC-AUC |")
    lines.append("|-------|----------|-----------|--------|---------------|------------|---------|")
    for _, row in combined_df.iterrows():
        marker = " **" if row["Model"] == overall_best_name else ""
        marker_end = "**" if marker else ""
        lines.append(
            f"| {marker}{row['Model']}{marker_end} "
            f"| {row['Accuracy']:.4f} "
            f"| {row['Precision']:.4f} "
            f"| {row['Recall']:.4f} "
            f"| {row['F1 (weighted)']:.4f} "
            f"| {row['F1 (macro)']:.4f} "
            f"| {row['ROC-AUC']} |"
        )
    lines.append("")

    # --- Classification Reports ---
    lines.append("## Ensemble Classification Reports")
    lines.append("")
    for ens_name, metrics in ensemble_metrics.items():
        lines.append(f"### {ens_name}")
        lines.append("")
        lines.append("```")
        lines.append(metrics["classification_report"])
        lines.append("```")
        lines.append("")

    # --- Conclusion ---
    lines.append("## Conclusion")
    lines.append("")
    lines.append(f"**Previous Best Baseline:** {baseline_best_name} (F1 = {baseline_best_f1:.4f})  ")
    if best_ensemble is not None:
        lines.append(f"**Best Ensemble:** {best_ensemble['Model']} "
                      f"(F1 = {best_ensemble['F1 (weighted)']:.4f})  ")
    lines.append(f"**Overall Best Model:** {overall_best_name} "
                  f"(F1 = {overall_best_f1:.4f})  ")
    lines.append("")
    if improved:
        delta = best_ensemble["F1 (weighted)"] - baseline_best_f1
        lines.append(
            f"Ensemble learning **improved** over the best baseline by "
            f"+{delta:.4f} F1 points. The best ensemble model has been "
            f"saved as `models/best_model.joblib`."
        )
    else:
        lines.append(
            "Ensemble learning did **not** improve over the best baseline model. "
            "This is consistent with published findings that for highly imbalanced, "
            "sparse text classification tasks, well-regularized gradient boosting "
            "models (XGBoost/LightGBM) are already near-optimal and difficult to "
            "beat with simple ensemble stacking. The baseline best model remains "
            "saved as `models/best_model.joblib`."
        )
    lines.append("")

    report_content = "\n".join(lines)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"Ensemble report saved to: {report_path}")
    return report_path


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    """Execute the Phase 2 Ensemble Learning pipeline."""

    pipeline_start = time.time()
    run_id = timestamp()

    print("=" * 70)
    print("  PHASE 2: ENSEMBLE LEARNING PIPELINE")
    print("  Extension for Software Defect Classification")
    print("=" * 70)

    ensure_dirs()

    # =========================================================================
    # STEP 1: Reload Saved Artifacts
    # =========================================================================
    print_section("STEP 1: RELOADING SAVED ARTIFACTS")

    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    df = pd.read_csv(PATHS["processed_data_dir"] / "preprocessed_issues.csv")
    logger.info(f"Loaded {len(df):,} preprocessed samples")

    # Load saved vectorizer and encoder
    vectorizer = joblib.load(PATHS["models_dir"] / "tfidf_vectorizer.joblib")
    label_encoder = joblib.load(PATHS["models_dir"] / "label_encoder.joblib")
    class_names = list(label_encoder.classes_)
    logger.info(f"Classes ({len(class_names)}): {class_names}")

    # Rebuild features from saved vectorizer
    vec_type = "TF-IDF" if hasattr(vectorizer, "idf_") else "Count"
    logger.info(f"Rebuilding {vec_type} features from saved vectorizer...")
    X_tfidf = vectorizer.transform(df["cleaned_text"].fillna("")).astype(np.float64)
    y_encoded = label_encoder.transform(df["target"])
    logger.info(f"Feature matrix: {X_tfidf.shape}")

    # Reconstruct the EXACT same split (same seed = same split)
    splits = split_data(X_tfidf, y_encoded)
    X_train = splits["X_train"]
    X_val = splits["X_val"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_val = splits["y_val"]
    y_test = splits["y_test"]

    # Combine train + val for final training (same as baseline evaluation)
    X_train_full = vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    logger.info(f"Training set: {X_train_full.shape[0]:,} samples")
    logger.info(f"Test set: {X_test.shape[0]:,} samples")

    # Load previous baseline results (do NOT overwrite)
    baseline_df = pd.read_csv(PATHS["reports_dir"] / "model_comparison.csv")
    print("\nPrevious Baseline Results (preserved):")
    print(baseline_df.to_string(index=False))

    baseline_best_name = baseline_df.iloc[0]["Model"]
    baseline_best_f1 = baseline_df.iloc[0]["F1 (weighted)"]
    logger.info(f"Previous best: {baseline_best_name} (F1={baseline_best_f1:.4f})")

    # =========================================================================
    # STEP 2: Train Ensemble Models
    # =========================================================================
    print_section("STEP 2: TRAINING ENSEMBLE MODELS")

    ensemble_registry = get_ensemble_registry()
    ensemble_models = {}

    for name, entry in ensemble_registry.items():
        model = entry["model"]
        logger.info(f"\nTraining: {name}...")
        logger.info(f"  Base learners: {entry['base_learners']}")
        if entry.get("meta_learner"):
            logger.info(f"  Meta learner: {entry['meta_learner']}")

        if name == "Stacking":
            logger.info("  NOTE: Stacking trains base models across 5 internal "
                        "CV folds. This may take 30-45 minutes.")

        t_start = time.time()
        try:
            model.fit(X_train_full, y_train_full)
            elapsed = time.time() - t_start
            logger.info(f"  {name}: Training completed in {elapsed:.1f}s")
            ensemble_models[f"{name} (Ensemble)"] = model

            # Save each ensemble model
            model_filename = f"ensemble_{name.lower().replace(' ', '_')}.joblib"
            save_path = PATHS["models_dir"] / model_filename
            joblib.dump(model, save_path)
            logger.info(f"  Saved: {save_path}")

        except Exception as e:
            elapsed = time.time() - t_start
            logger.error(f"  {name} FAILED after {elapsed:.1f}s: {e}")

    if not ensemble_models:
        logger.error("No ensemble models trained successfully. Exiting.")
        return

    # =========================================================================
    # STEP 3: Evaluate Ensembles on Test Set
    # =========================================================================
    print_section("STEP 3: EVALUATING ENSEMBLES ON TEST SET")

    ensemble_comparison_df, ensemble_metrics = evaluate_all(
        ensemble_models, X_test, y_test, class_names,
    )

    # Print classification reports
    print_section("ENSEMBLE CLASSIFICATION REPORTS")
    for name in ensemble_comparison_df["Model"].tolist():
        if name in ensemble_metrics:
            print(f"\n{'-' * 60}")
            print(f"  {name}")
            print(f"{'-' * 60}")
            print(ensemble_metrics[name]["classification_report"])

    # =========================================================================
    # STEP 4: Combined Comparison (Baselines + Ensembles)
    # =========================================================================
    print_section("STEP 4: COMBINED COMPARISON (BASELINES + ENSEMBLES)")

    combined_df = pd.concat(
        [baseline_df, ensemble_comparison_df], ignore_index=True,
    ).sort_values("F1 (weighted)", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 100)
    print("  COMPLETE MODEL COMPARISON -- Baselines + Ensembles")
    print("=" * 100)
    print(combined_df.to_string(index=False))

    # Save combined comparison (as a NEW file, not overwriting baseline)
    combined_df.to_csv(
        PATHS["reports_dir"] / "combined_model_comparison.csv", index=False,
    )
    logger.info("Combined comparison saved to combined_model_comparison.csv")

    # =========================================================================
    # STEP 5: Determine Overall Best Model
    # =========================================================================
    print_section("STEP 5: BEST MODEL DETERMINATION")

    overall_best_name = combined_df.iloc[0]["Model"]
    overall_best_f1 = combined_df.iloc[0]["F1 (weighted)"]

    logger.info(f"Overall best model: {overall_best_name} (F1={overall_best_f1:.4f})")
    logger.info(f"Previous best baseline: {baseline_best_name} (F1={baseline_best_f1:.4f})")

    # Update best_model.joblib if an ensemble beats the baseline
    if "(Ensemble)" in overall_best_name and overall_best_f1 > baseline_best_f1:
        best_model = ensemble_models[overall_best_name]
        logger.info(f"Ensemble improved! Saving {overall_best_name} as best_model.joblib")
        train_final_model(best_model, X_train_full, y_train_full, "best_model")

        # Update metadata
        model_metadata = {
            "model_name": overall_best_name,
            "f1_weighted": float(overall_best_f1),
            "accuracy": float(combined_df.iloc[0]["Accuracy"]),
            "n_classes": len(class_names),
            "class_names": class_names,
            "n_features": X_tfidf.shape[1],
            "n_training_samples": int(X_train_full.shape[0]),
            "pipeline_run_id": run_id,
            "phase": "ensemble",
            "config": {k: str(v) for k, v in CONFIG.items()},
        }
        save_json(model_metadata, PATHS["models_dir"] / "model_metadata.json")
    else:
        logger.info("Baseline model remains the best. best_model.joblib unchanged.")

    # =========================================================================
    # STEP 6: Generate Visualizations
    # =========================================================================
    print_section("STEP 6: GENERATING ENSEMBLE VISUALIZATIONS")

    # Individual metric comparison charts (ensemble-specific filenames)
    plot_metric_comparison(
        combined_df, "Accuracy",
        "Accuracy Comparison -- Baselines vs Ensembles",
        "ensemble_accuracy_comparison.png",
    )
    plot_metric_comparison(
        combined_df, "Precision",
        "Precision Comparison -- Baselines vs Ensembles",
        "ensemble_precision_comparison.png",
    )
    plot_metric_comparison(
        combined_df, "Recall",
        "Recall Comparison -- Baselines vs Ensembles",
        "ensemble_recall_comparison.png",
    )
    plot_metric_comparison(
        combined_df, "F1 (weighted)",
        "F1 Score Comparison -- Baselines vs Ensembles",
        "ensemble_f1_comparison.png",
    )

    # Grouped bar chart (all metrics)
    plot_model_comparison(combined_df, filename="ensemble_model_comparison.png")

    # Per-class F1 for ensembles
    plot_per_class_f1(ensemble_metrics, class_names, filename="ensemble_per_class_f1.png")

    # Confusion matrices for ensemble models
    generate_confusion_matrices(ensemble_models, X_test, y_test, class_names)

    # =========================================================================
    # STEP 7: Generate Markdown Report
    # =========================================================================
    print_section("STEP 7: GENERATING ENSEMBLE REPORT")

    pipeline_elapsed = (time.time() - pipeline_start) / 60

    report_path = generate_ensemble_report(
        combined_df=combined_df,
        ensemble_metrics=ensemble_metrics,
        baseline_best_f1=baseline_best_f1,
        baseline_best_name=baseline_best_name,
        ensemble_registry=ensemble_registry,
        class_names=class_names,
        run_id=run_id,
        elapsed_minutes=pipeline_elapsed,
        vectorizer=vectorizer,
        total_samples=len(df),
    )

    # =========================================================================
    # Summary
    # =========================================================================
    pipeline_elapsed = (time.time() - pipeline_start) / 60

    print("\n" + "=" * 70)
    print("  PHASE 2 COMPLETE: ENSEMBLE LEARNING")
    print("=" * 70)
    print(f"\n  Run ID:                    {run_id}")
    print(f"  Total time:                {pipeline_elapsed:.1f} minutes")
    print(f"  Ensembles trained:         {len(ensemble_models)}")
    print(f"  Previous best (baseline):  {baseline_best_name} (F1={baseline_best_f1:.4f})")
    print(f"  Overall best model:        {overall_best_name} (F1={overall_best_f1:.4f})")
    print(f"\n  Saved artifacts:")
    print(f"    Ensemble models:   {PATHS['models_dir']}")
    print(f"    Combined report:   {PATHS['reports_dir'] / 'ensemble_results.md'}")
    print(f"    Comparison CSV:    {PATHS['reports_dir'] / 'combined_model_comparison.csv'}")
    print(f"    Visualizations:    {PATHS['figures_dir']}")
    print("=" * 70)

    return combined_df, ensemble_metrics


if __name__ == "__main__":
    main()
