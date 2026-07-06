# =============================================================================
# main.py — Complete Research Pipeline Orchestrator
# =============================================================================
"""
Software Defect Classification Pipeline
========================================

Orchestrates the entire ML pipeline for classifying mobile application
defects from GitHub issue data.

Reference Paper:
    "From Bugs to Benchmarks: A Comprehensive Survey of Software Defect Datasets"
    https://dl.acm.org/doi/pdf/10.1145/3797033

Pipeline Stages:
    1. Data Loading & Inspection
    2. Exploratory Data Analysis (EDA)
    3. Target Engineering (automatic label selection)
    4. Text Preprocessing (NLP pipeline)
    5. Feature Engineering (TF-IDF)
    6. Baseline Model Training (7 classifiers)
    7. Model Evaluation & Comparison
    8. Hyperparameter Tuning (top-3 models)
    9. Final Model Selection & Saving
    10. Visualization Generation

Usage:
    python main.py
"""

import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd

# Add project root to path
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    get_logger, PATHS, CONFIG, ensure_dirs, save_json, print_section, timestamp,
)
from src.preprocessing import load_data, inspect_data, engineer_target, TextPreprocessor
from src.feature_engineering import (
    encode_labels, split_data, get_top_tfidf_features, build_features, build_text_features
)
from src.train import train_all_models, tune_best_models, train_final_model, get_model_registry, get_ensemble_registry
from src.evaluate import (
    evaluate_model, evaluate_all, generate_confusion_matrices,
    plot_class_distribution, plot_label_distribution_pie,
    plot_text_length_distribution, plot_tfidf_importance,
    plot_model_comparison, plot_f1_comparison, plot_per_class_f1,
)

from scipy.sparse import vstack

warnings.filterwarnings("ignore")
logger = get_logger("pipeline")


def main():
    """Execute the full research pipeline."""

    pipeline_start = time.time()

    print("=" * 70)
    print("  SOFTWARE DEFECT CLASSIFICATION — RESEARCH PIPELINE")
    print("  Reference: 'From Bugs to Benchmarks' (ACM 2025)")
    print("  Dataset: Mobile Application GitHub Issues")
    print("=" * 70)

    # --- Setup ---
    ensure_dirs()
    run_id = timestamp()
    logger.info(f"Pipeline run ID: {run_id}")
    logger.info(f"Configuration: {CONFIG}")

    # =========================================================================
    # STAGE 1: Data Loading & Inspection
    # =========================================================================
    print_section("STAGE 1: DATA LOADING & INSPECTION")

    df = load_data()
    inspection_stats = inspect_data(df)

    # Save inspection report
    save_json(inspection_stats, PATHS["reports_dir"] / "inspection_stats.json")
    logger.info("Inspection stats saved")

    # =========================================================================
    # STAGE 2: Target Engineering
    # =========================================================================
    print_section("STAGE 2: TARGET ENGINEERING")

    df, target_stats = engineer_target(df)
    save_json(target_stats, PATHS["reports_dir"] / "target_stats.json")

    class_names = list(target_stats["class_distribution"].keys())
    logger.info(f"Classification target: {len(class_names)} classes")
    logger.info(f"Classes: {class_names}")

    # =========================================================================
    # STAGE 3: Text Preprocessing
    # =========================================================================
    print_section("STAGE 3: TEXT PREPROCESSING")

    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess(df)

    # Save preprocessed data to CSV (for notebook analysis)
    preprocessed_path = PATHS["processed_data_dir"] / "preprocessed_issues.csv"
    save_cols = ["title", "body", "target", "cleaned_text"]
    if "_id" in df.columns:
        save_cols.insert(0, "_id")
    elif "url" in df.columns:
        save_cols.insert(0, "url")
        
    df[save_cols].to_csv(
        preprocessed_path, index=False,
    )
    logger.info(f"Preprocessed data saved to {preprocessed_path}")

    # =========================================================================
    # STAGE 4: EDA Visualizations (Pre-modeling)
    # =========================================================================
    print_section("STAGE 4: EXPLORATORY DATA ANALYSIS")

    plot_class_distribution(
        df["target"],
        title="Target Class Distribution (After Filtering)",
    )
    plot_label_distribution_pie(target_stats)
    plot_text_length_distribution(df)
    logger.info("EDA visualizations generated")

    # =========================================================================
    # STAGE 5: Feature Engineering
    # =========================================================================
    print_section("STAGE 5: FEATURE ENGINEERING")

    splits, vectorizer, label_encoder = build_features(df, CONFIG)

    X_train = splits["X_train"]
    X_val = splits["X_val"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_val = splits["y_val"]
    y_test = splits["y_test"]

    # Save features for analysis
    top_features = get_top_tfidf_features(vectorizer, n=30)
    if not top_features.empty:
        top_features.to_csv(PATHS["reports_dir"] / "top_features.csv", index=False)
        plot_tfidf_importance(vectorizer)

    logger.info("Feature engineering complete.")

    # =========================================================================
    # STAGE 6: Baseline Model Training
    # =========================================================================
    print_section("STAGE 6: BASELINE MODEL TRAINING")

    # Use training + validation for cross-validation
    X_train_full = vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    results_df = train_all_models(X_train_full, y_train_full)
    results_df.to_csv(PATHS["reports_dir"] / "baseline_cv_results.csv", index=False)

    # =========================================================================
    # STAGE 7: Hyperparameter Tuning
    # =========================================================================
    print_section("STAGE 7: HYPERPARAMETER TUNING")

    tuned_models = tune_best_models(X_train_full, y_train_full, results_df, top_k=3)

    # Log tuning results
    tuning_summary = {}
    for name, info in tuned_models.items():
        tuning_summary[name] = {
            "best_params": info["best_params"],
            "best_score": info["best_score"],
            "tuning_time_s": info["tuning_time_s"],
        }
    save_json(tuning_summary, PATHS["reports_dir"] / "tuning_results.json")

    # =========================================================================
    # STAGE 8: Final Evaluation on Test Set
    # =========================================================================
    print_section("STAGE 8: FINAL EVALUATION ON TEST SET")

    # Prepare models for evaluation
    eval_models = {}
    for name, info in tuned_models.items():
        eval_models[f"{name} (Tuned)"] = info["best_estimator"]

    # Also train and evaluate baseline versions of untuned models
    registry = get_model_registry()
    for name in results_df[results_df["Status"] == "OK"]["Model"].tolist():
        if name in registry:
            model = registry[name]["model"]
            model.fit(X_train_full, y_train_full)
            eval_models[f"{name} (Baseline)"] = model

    # Add Ensemble Learning (from strongest baseline models)
    print_section("STAGE 7.5: ENSEMBLE LEARNING")
    ensemble_registry = get_ensemble_registry(tuned_models)
    for name, entry in ensemble_registry.items():
        ensemble_model = entry["model"]
        logger.info(f"Training ensemble: {name}")
        ensemble_model.fit(X_train_full, y_train_full)
        eval_models[f"{name} (Ensemble)"] = ensemble_model

    # Evaluate all models
    comparison_df, all_metrics = evaluate_all(
        eval_models, X_test, y_test, class_names,
    )

    # Print detailed classification reports for top models
    print_section("DETAILED CLASSIFICATION REPORTS")
    for name in list(comparison_df["Model"].head(3)):
        if name in all_metrics:
            print(f"\n{'-' * 60}")
            print(f"  {name}")
            print(f"{'-' * 60}")
            print(all_metrics[name]["classification_report"])

    # =========================================================================
    # STAGE 9: Save Best Model
    # =========================================================================
    print_section("STAGE 9: SAVE BEST MODEL")

    best_model_name = comparison_df.iloc[0]["Model"]
    best_model = eval_models[best_model_name]
    best_score = comparison_df.iloc[0]["F1 (weighted)"]

    logger.info(f"Best model: {best_model_name} (F1={best_score:.4f})")

    # Save best model
    train_final_model(best_model, X_train_full, y_train_full, "best_model")

    # Save model metadata
    model_metadata = {
        "model_name": best_model_name,
        "f1_weighted": float(best_score),
        "accuracy": float(comparison_df.iloc[0]["Accuracy"]),
        "n_classes": len(class_names),
        "class_names": class_names,
        "n_features": X_train_full.shape[1],
        "n_training_samples": X_train_full.shape[0],
        "pipeline_run_id": run_id,
        "config": {k: str(v) for k, v in CONFIG.items()},
    }
    save_json(model_metadata, PATHS["models_dir"] / "model_metadata.json")

    # =========================================================================
    # STAGE 10: Generate All Visualizations
    # =========================================================================
    print_section("STAGE 10: GENERATE FINAL VISUALIZATIONS")

    plot_model_comparison(comparison_df)
    plot_f1_comparison(comparison_df)
    plot_per_class_f1(all_metrics, class_names)
    generate_confusion_matrices(eval_models, X_test, y_test, class_names)

    # =========================================================================
    # Summary
    # =========================================================================
    pipeline_elapsed = time.time() - pipeline_start

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Run ID:              {run_id}")
    print(f"  Total time:          {pipeline_elapsed / 60:.1f} minutes")
    print(f"  Dataset size:        {len(df):,} samples")
    print(f"  Number of classes:   {len(class_names)}")
    print(f"  Text features:       {X_train_full.shape[1]:,}")
    print(f"  Models evaluated:    {len(eval_models)}")
    print(f"  Best model:          {best_model_name}")
    print(f"  Best F1 (weighted):  {best_score:.4f}")
    print(f"\n  Artifacts saved to:")
    print(f"    Models:  {PATHS['models_dir']}")
    print(f"    Reports: {PATHS['reports_dir']}")
    print(f"    Figures: {PATHS['figures_dir']}")
    print("=" * 70)

    return comparison_df, all_metrics, best_model


if __name__ == "__main__":
    main()
