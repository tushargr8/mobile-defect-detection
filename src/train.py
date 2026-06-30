# =============================================================================
# train.py — Model Training with Multiple Classifiers
# =============================================================================
"""
Trains multiple machine learning classifiers and performs hyperparameter tuning.

Classifiers:
    1. Logistic Regression
    2. Multinomial Naive Bayes
    3. Decision Tree
    4. Random Forest
    5. Support Vector Machine (SVM)
    6. XGBoost
    7. LightGBM

Extensibility hooks:
    - Voting Classifier
    - Stacking Ensemble
    - Bagging / Boosting ensembles
    - Custom model registration

Usage:
    from src.train import train_all_models, tune_best_models
"""

import time
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    cross_val_score,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import make_scorer, f1_score

from typing import Dict, Any, Tuple, List, Optional

from src.utils import get_logger, PATHS, CONFIG, print_section

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


# =============================================================================
# 1. Model Registry
# =============================================================================

def get_model_registry() -> Dict[str, Any]:
    """
    Return a dictionary of model name → (model instance, hyperparameter grid).

    Each entry contains:
        - "model": instantiated sklearn-compatible estimator
        - "params": hyperparameter search space for RandomizedSearchCV

    Returns
    -------
    dict
        Model registry with names, instances, and parameter grids.
    """
    seed = CONFIG["random_seed"]

    registry = {
        "Logistic Regression": {
            "model": LogisticRegression(
                max_iter=1000, random_state=seed, solver="saga", n_jobs=-1,
            ),
            "params": {
                "C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                "penalty": ["l1", "l2"],
            },
        },

        "Naive Bayes": {
            "model": MultinomialNB(),
            "params": {
                "alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            },
        },

        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=seed),
            "params": {
                "max_depth": [5, 10, 20, 30, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5, 10],
                "criterion": ["gini", "entropy"],
            },
        },

        "Random Forest": {
            "model": RandomForestClassifier(
                random_state=seed, n_jobs=-1, n_estimators=200,
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5],
                "max_features": ["sqrt", "log2"],
            },
        },

        "SVM": {
            "model": CalibratedClassifierCV(
                estimator=LinearSVC(
                    max_iter=2000, random_state=seed, dual="auto",
                ),
                cv=3,
            ),
            "params": {
                "estimator__C": [0.01, 0.1, 0.5, 1.0, 5.0],
                "estimator__loss": ["hinge", "squared_hinge"],
            },
        },
    }

    # --- XGBoost ---
    try:
        from xgboost import XGBClassifier
        registry["XGBoost"] = {
            "model": XGBClassifier(
                random_state=seed,
                n_jobs=-1,
                eval_metric="mlogloss",
                use_label_encoder=False,
                verbosity=0,
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7, 10],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "min_child_weight": [1, 3, 5],
            },
        }
    except ImportError:
        logger.warning("XGBoost not available — skipping")

    # --- LightGBM ---
    try:
        from lightgbm import LGBMClassifier
        registry["LightGBM"] = {
            "model": LGBMClassifier(
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7, 10, -1],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "num_leaves": [15, 31, 63, 127],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "min_child_samples": [5, 10, 20],
            },
        }
    except ImportError:
        logger.warning("LightGBM not available — skipping")

    return registry


# =============================================================================
# 2. Baseline Training (Cross-Validated)
# =============================================================================

def train_all_models(
    X_train,
    y_train,
    config: dict = None,
    model_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Train all registered models with cross-validation and return a comparison.

    Parameters
    ----------
    X_train : sparse matrix
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    config : dict, optional
        Pipeline configuration.
    model_names : list, optional
        Subset of model names to train. If None, trains all.

    Returns
    -------
    pd.DataFrame
        DataFrame with model names, CV scores (mean/std), and training times.
    """
    config = config or CONFIG
    print_section("TRAINING ALL BASELINE MODELS")

    registry = get_model_registry()

    if model_names:
        registry = {k: v for k, v in registry.items() if k in model_names}

    cv = StratifiedKFold(
        n_splits=config["cv_folds"],
        shuffle=True,
        random_state=config["random_seed"],
    )

    results = []

    for name, entry in registry.items():
        model = entry["model"]
        logger.info(f"Training: {name}...")

        t_start = time.time()
        try:
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv,
                scoring=config["scoring_metric"],
                n_jobs=config["n_jobs"],
            )
            elapsed = time.time() - t_start

            result = {
                "Model": name,
                "CV_Mean_F1": round(np.mean(scores), 4),
                "CV_Std_F1": round(np.std(scores), 4),
                "CV_Min_F1": round(np.min(scores), 4),
                "CV_Max_F1": round(np.max(scores), 4),
                "Training_Time_s": round(elapsed, 2),
                "Status": "OK",
            }
            logger.info(f"  {name}: F1={result['CV_Mean_F1']:.4f} "
                        f"(±{result['CV_Std_F1']:.4f}) in {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t_start
            result = {
                "Model": name,
                "CV_Mean_F1": 0.0,
                "CV_Std_F1": 0.0,
                "CV_Min_F1": 0.0,
                "CV_Max_F1": 0.0,
                "Training_Time_s": round(elapsed, 2),
                "Status": f"FAILED: {str(e)[:100]}",
            }
            logger.error(f"  {name} FAILED: {e}")

        results.append(result)

    results_df = pd.DataFrame(results).sort_values(
        "CV_Mean_F1", ascending=False,
    ).reset_index(drop=True)

    print("\n" + "=" * 90)
    print("  MODEL COMPARISON — Cross-Validated F1 (Weighted)")
    print("=" * 90)
    print(results_df.to_string(index=False))

    return results_df


# =============================================================================
# 3. Hyperparameter Tuning
# =============================================================================

def tune_best_models(
    X_train,
    y_train,
    results_df: pd.DataFrame,
    top_k: int = 3,
    config: dict = None,
) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning on the top-K performing models.

    Uses RandomizedSearchCV for efficiency with large parameter spaces.

    Parameters
    ----------
    X_train : sparse matrix
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    results_df : pd.DataFrame
        Model comparison DataFrame from train_all_models.
    top_k : int
        Number of top models to tune.
    config : dict, optional
        Pipeline configuration.

    Returns
    -------
    dict
        Dictionary mapping model names to their best estimators and scores.
    """
    config = config or CONFIG
    print_section(f"HYPERPARAMETER TUNING — Top {top_k} Models")

    registry = get_model_registry()

    # Select top-K models
    top_models = results_df[results_df["Status"] == "OK"].head(top_k)["Model"].tolist()
    logger.info(f"Tuning models: {top_models}")

    cv = StratifiedKFold(
        n_splits=config["cv_folds"],
        shuffle=True,
        random_state=config["random_seed"],
    )

    tuned_models = {}

    for name in top_models:
        if name not in registry:
            continue

        entry = registry[name]
        model = entry["model"]
        param_grid = entry["params"]

        logger.info(f"\nTuning: {name}")
        logger.info(f"  Parameter space: {list(param_grid.keys())}")

        t_start = time.time()
        try:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=min(config["hyperparameter_tuning_n_iter"],
                           _count_combinations(param_grid)),
                cv=cv,
                scoring=config["scoring_metric"],
                n_jobs=config["n_jobs"],
                random_state=config["random_seed"],
                verbose=0,
                refit=True,
            )
            search.fit(X_train, y_train)
            elapsed = time.time() - t_start

            tuned_models[name] = {
                "best_estimator": search.best_estimator_,
                "best_params": search.best_params_,
                "best_score": round(search.best_score_, 4),
                "tuning_time_s": round(elapsed, 2),
            }

            logger.info(f"  Best F1: {search.best_score_:.4f}")
            logger.info(f"  Best params: {search.best_params_}")
            logger.info(f"  Tuning time: {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"  Tuning {name} FAILED: {e}")

    return tuned_models


def _count_combinations(param_grid: dict) -> int:
    """Count the total number of parameter combinations."""
    n = 1
    for values in param_grid.values():
        n *= len(values)
    return n


# =============================================================================
# 4. Final Model Training & Saving
# =============================================================================

def train_final_model(
    model,
    X_train,
    y_train,
    model_name: str = "best_model",
) -> Any:
    """
    Train the final model on the full training set and save it.

    Parameters
    ----------
    model : estimator
        Sklearn-compatible model (possibly from tuning).
    X_train : sparse matrix
        Full training feature matrix.
    y_train : np.ndarray
        Training labels.
    model_name : str
        Name for the saved model file.

    Returns
    -------
    estimator
        Fitted model.
    """
    print_section("TRAINING FINAL MODEL")

    logger.info(f"Training {model_name} on {X_train.shape[0]:,} samples...")
    model.fit(X_train, y_train)

    # Save model
    save_path = PATHS["models_dir"] / f"{model_name}.joblib"
    joblib.dump(model, save_path)
    logger.info(f"Model saved to: {save_path}")

    return model


def get_ensemble_registry(tuned_models: dict = None, seed: int = None) -> Dict[str, Any]:
    """
    Return ensemble model definitions for Phase 2 experimentation,
    using either the provided tuned models or reloading them from saved tuning results/defaults.
    """
    from sklearn.ensemble import (
        VotingClassifier,
        StackingClassifier,
    )
    from sklearn.linear_model import LogisticRegression

    seed = seed or CONFIG["random_seed"]

    # --- Base Estimators ---
    estimators = []
    
    if tuned_models is not None:
        for name, info in tuned_models.items():
            estimators.append((name.lower().replace(" ", "_"), info["best_estimator"]))
    else:
        # Try to reload from saved tuning results
        tuning_path = PATHS["reports_dir"] / "tuning_results.json"
        model_registry = get_model_registry()
        
        if tuning_path.exists():
            try:
                import json
                with open(tuning_path, "r", encoding="utf-8") as f:
                    tuning_results = json.load(f)
                
                logger.info(f"Recreating tuned estimators from {tuning_path} for ensembles...")
                for name, info in tuning_results.items():
                    if name in model_registry:
                        model = model_registry[name]["model"]
                        if "best_params" in info:
                            model.set_params(**info["best_params"])
                        estimators.append((name.lower().replace(" ", "_"), model))
            except Exception as e:
                logger.warning(f"Failed to load tuning results: {e}. Falling back to default baselines.")
                
        # If still empty, fall back to top baseline models
        if not estimators:
            logger.info("Using default baseline estimators for ensembles...")
            for name in ["XGBoost", "LightGBM", "Random Forest", "Logistic Regression", "SVM"]:
                if name in model_registry:
                    estimators.append((name.lower().replace(" ", "_"), model_registry[name]["model"]))
        
    if not estimators:
        logger.warning("No base estimators available for ensemble registry.")
        return {}

    registry = {}

    # -----------------------------------------------------------------
    # Ensemble 1: Hard Voting Classifier
    # -----------------------------------------------------------------
    registry["Hard Voting"] = {
        "model": VotingClassifier(
            estimators=estimators,
            voting="hard",
            n_jobs=-1,
        ),
        "base_learners": [name for name, _ in estimators],
        "meta_learner": None,
        "description": (
            "Majority-vote ensemble. Each base classifier casts one vote; "
            "the class with the most votes wins."
        ),
    }

    # -----------------------------------------------------------------
    # Ensemble 2: Soft Voting Classifier
    # -----------------------------------------------------------------
    # All our base learners support predict_proba (SVM is wrapped in
    # CalibratedClassifierCV, which adds probability calibration).
    registry["Soft Voting"] = {
        "model": VotingClassifier(
            estimators=estimators,
            voting="soft",
            n_jobs=-1,
        ),
        "base_learners": [name for name, _ in estimators],
        "meta_learner": None,
        "description": (
            "Probability-averaged ensemble. Predicted class probabilities "
            "from each base learner are averaged; the class with the "
            "highest mean probability is selected."
        ),
    }

    # -----------------------------------------------------------------
    # Ensemble 3: Stacking Classifier
    # -----------------------------------------------------------------
    meta_learner = LogisticRegression(
        max_iter=1000, random_state=seed, solver="saga", n_jobs=-1,
    )
    registry["Stacking"] = {
        "model": StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,
            stack_method="predict_proba",
            n_jobs=-1,
            passthrough=False,
        ),
        "base_learners": [name for name, _ in estimators],
        "meta_learner": "Logistic Regression",
        "description": (
            "Two-layer stacking ensemble. Layer 1: base classifiers generate "
            "out-of-fold probability predictions via 5-fold CV. Layer 2: a "
            "Logistic Regression meta-learner is trained on these stacked "
            "probabilities to produce final predictions."
        ),
    }

    logger.info(f"Ensemble registry initialized with {len(registry)} models: "
                f"{list(registry.keys())}")

    return registry

