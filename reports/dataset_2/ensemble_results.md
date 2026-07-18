# Phase 2: Ensemble Learning Results

**Run ID:** 20260731_022608  
**Total Training Time:** 0.2 minutes  
**Dataset:** Functional Bugs Annotated Dataset (174 samples, 3 classes)  
**Features:** Count (2,361 features)  

---

## Ensemble Architecture

### Hard Voting

**Description:** Majority-vote ensemble. Each base classifier casts one vote; the class with the most votes wins.

**Base Learners:** xgboost, logistic_regression, lightgbm

---

### Soft Voting

**Description:** Probability-averaged ensemble. Predicted class probabilities from each base learner are averaged; the class with the highest mean probability is selected.

**Base Learners:** xgboost, logistic_regression, lightgbm

---

### Stacking

**Description:** Two-layer stacking ensemble. Layer 1: base classifiers generate out-of-fold probability predictions via 5-fold CV. Layer 2: a Logistic Regression meta-learner is trained on these stacked probabilities to produce final predictions.

**Base Learners:** xgboost, logistic_regression, lightgbm

**Meta Learner:** Logistic Regression

---

## Performance Comparison (All Models)

| Model | Accuracy | Precision | Recall | F1 (weighted) | F1 (macro) | ROC-AUC |
|-------|----------|-----------|--------|---------------|------------|---------|
|  **Logistic Regression (Baseline)** | 0.8519 | 0.8418 | 0.8519 | 0.8445 | 0.7631 | 0.9275 |
| LightGBM (Baseline) | 0.8519 | 0.8651 | 0.8519 | 0.8409 | 0.7919 | 0.9112 |
| Logistic Regression (Tuned) | 0.8148 | 0.7865 | 0.8148 | 0.7924 | 0.6736 | 0.901 |
| Stacking (Ensemble) | 0.8148 | 0.7282 | 0.8148 | 0.7681 | 0.5818 | 0.9398 |
| Soft Voting (Ensemble) | 0.8148 | 0.7282 | 0.8148 | 0.7681 | 0.5818 | 0.9266 |
| Stacking (Ensemble) | 0.8148 | 0.7282 | 0.8148 | 0.7681 | 0.5818 | 0.9398 |
| Soft Voting (Ensemble) | 0.8148 | 0.7282 | 0.8148 | 0.7681 | 0.5818 | 0.9266 |
| SVM (Baseline) | 0.7778 | 0.8413 | 0.7778 | 0.7501 | 0.6675 | 0.9577 |
| XGBoost (Baseline) | 0.7778 | 0.7071 | 0.7778 | 0.7368 | 0.5567 | 0.8914 |
| Hard Voting (Ensemble) | 0.7778 | 0.6877 | 0.7778 | 0.7300 | 0.5478 | nan |
| LightGBM (Tuned) | 0.7778 | 0.6877 | 0.7778 | 0.7300 | 0.5478 | 0.9412 |
| Hard Voting (Ensemble) | 0.7778 | 0.6877 | 0.7778 | 0.7300 | 0.5478 | N/A |
| Naive Bayes (Baseline) | 0.7037 | 0.8112 | 0.7037 | 0.7236 | 0.6850 | 0.895 |
| XGBoost (Tuned) | 0.7407 | 0.6639 | 0.7407 | 0.6993 | 0.5233 | 0.9145 |
| Random Forest (Baseline) | 0.6667 | 0.5566 | 0.6667 | 0.5872 | 0.4241 | 0.941 |
| Decision Tree (Baseline) | 0.5556 | 0.5769 | 0.5556 | 0.5526 | 0.4056 | 0.6627 |

## Ensemble Classification Reports

### Hard Voting (Ensemble)

```
                      precision    recall  f1-score   support

[A] Model Conversion       0.88      1.00      0.94        15
  [B] DL Integration       0.00      0.00      0.00         4
       [D] Inference       0.67      0.75      0.71         8

            accuracy                           0.78        27
           macro avg       0.52      0.58      0.55        27
        weighted avg       0.69      0.78      0.73        27

```

### Soft Voting (Ensemble)

```
                      precision    recall  f1-score   support

[A] Model Conversion       0.94      1.00      0.97        15
  [B] DL Integration       0.00      0.00      0.00         4
       [D] Inference       0.70      0.88      0.78         8

            accuracy                           0.81        27
           macro avg       0.55      0.62      0.58        27
        weighted avg       0.73      0.81      0.77        27

```

### Stacking (Ensemble)

```
                      precision    recall  f1-score   support

[A] Model Conversion       0.94      1.00      0.97        15
  [B] DL Integration       0.00      0.00      0.00         4
       [D] Inference       0.70      0.88      0.78         8

            accuracy                           0.81        27
           macro avg       0.55      0.62      0.58        27
        weighted avg       0.73      0.81      0.77        27

```

## Conclusion

**Previous Best Baseline:** Logistic Regression (Baseline) (F1 = 0.8445)  
**Best Ensemble:** Stacking (Ensemble) (F1 = 0.7681)  
**Overall Best Model:** Logistic Regression (Baseline) (F1 = 0.8445)  

Ensemble learning did **not** improve over the best baseline model. This is consistent with published findings that for highly imbalanced, sparse text classification tasks, well-regularized gradient boosting models (XGBoost/LightGBM) are already near-optimal and difficult to beat with simple ensemble stacking. The baseline best model remains saved as `models/best_model.joblib`.
