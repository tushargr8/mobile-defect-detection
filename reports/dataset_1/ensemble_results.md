# Phase 2: Ensemble Learning Results

**Run ID:** 20260716_032232  
**Total Training Time:** 562.5 minutes  
**Dataset:** Mobile Application GitHub Issues (65,760 samples, 15 classes)  
**Features:** TF-IDF (10,000 unigram + bigram features)  

---

## Ensemble Architecture

### Hard Voting

**Description:** Majority-vote ensemble. Each base classifier casts one vote; the class with the most votes wins.

**Base Learners:** lr, rf, svm, xgb, lgbm

---

### Soft Voting

**Description:** Probability-averaged ensemble. Predicted class probabilities from each base learner are averaged; the class with the highest mean probability is selected.

**Base Learners:** lr, rf, svm, xgb, lgbm

---

### Stacking

**Description:** Two-layer stacking ensemble. Layer 1: base classifiers generate out-of-fold probability predictions via 5-fold CV. Layer 2: a Logistic Regression meta-learner is trained on these stacked probabilities to produce final predictions.

**Base Learners:** lr, rf, svm, xgb, lgbm

**Meta Learner:** Logistic Regression

---

## Performance Comparison (All Models)

| Model | Accuracy | Precision | Recall | F1 (weighted) | F1 (macro) | ROC-AUC |
|-------|----------|-----------|--------|---------------|------------|---------|
|  **Stacking (Ensemble)** | 0.8706 | 0.8192 | 0.8706 | 0.8247 | 0.2470 | 0.732 |
| XGBoost (Baseline) | 0.8683 | 0.8186 | 0.8683 | 0.8182 | 0.2134 | 0.7169 |
| LightGBM (Baseline) | 0.8660 | 0.8120 | 0.8660 | 0.8167 | 0.2172 | 0.7126 |
| LightGBM (Tuned) | 0.8656 | 0.8148 | 0.8656 | 0.8156 | 0.2128 | 0.7084 |
| Random Forest (Baseline) | 0.8651 | 0.8049 | 0.8651 | 0.8150 | 0.1821 | 0.6926 |
| Soft Voting (Ensemble) | 0.8683 | 0.8241 | 0.8683 | 0.8146 | 0.1917 | 0.7358 |
| Hard Voting (Ensemble) | 0.8673 | 0.8254 | 0.8673 | 0.8124 | 0.1739 | N/A |
| Logistic Regression (Baseline) | 0.8653 | 0.8051 | 0.8653 | 0.8097 | 0.1567 | 0.7208 |
| SVM (Baseline) | 0.8660 | 0.8145 | 0.8660 | 0.8091 | 0.1658 | 0.72 |
| Naive Bayes (Baseline) | 0.8552 | 0.7677 | 0.8552 | 0.7964 | 0.0815 | 0.5824 |
| Decision Tree (Baseline) | 0.7990 | 0.7801 | 0.7990 | 0.7892 | 0.2090 | 0.5834 |

## Ensemble Classification Reports

### Hard Voting (Ensemble)

```
                      precision    recall  f1-score   support

             android       0.75      0.07      0.13       125
            bug_only       0.87      1.00      0.93      8472
           duplicate       0.00      0.00      0.00        63
         enhancement       0.80      0.01      0.02       336
    good first issue       0.00      0.00      0.00        71
         help wanted       0.33      0.00      0.01       270
       high priority       1.00      0.09      0.17        95
             invalid       0.00      0.00      0.00        37
                 ios       0.45      0.10      0.16        50
minor (quick review)       0.89      0.18      0.30        45
       navigation-ui       0.67      0.17      0.27        36
                  p1       1.00      0.02      0.03        57
                  p2       0.80      0.46      0.58       111
      priority: high       0.00      0.00      0.00        40
            question       0.00      0.00      0.00        56

            accuracy                           0.87      9864
           macro avg       0.50      0.14      0.17      9864
        weighted avg       0.83      0.87      0.81      9864

```

### Soft Voting (Ensemble)

```
                      precision    recall  f1-score   support

             android       0.75      0.07      0.13       125
            bug_only       0.87      1.00      0.93      8472
           duplicate       0.00      0.00      0.00        63
         enhancement       0.75      0.02      0.03       336
    good first issue       0.00      0.00      0.00        71
         help wanted       0.33      0.00      0.01       270
       high priority       1.00      0.09      0.17        95
             invalid       0.00      0.00      0.00        37
                 ios       0.56      0.20      0.29        50
minor (quick review)       0.90      0.20      0.33        45
       navigation-ui       0.67      0.17      0.27        36
                  p1       0.75      0.05      0.10        57
                  p2       0.85      0.48      0.61       111
      priority: high       0.00      0.00      0.00        40
            question       0.00      0.00      0.00        56

            accuracy                           0.87      9864
           macro avg       0.50      0.15      0.19      9864
        weighted avg       0.82      0.87      0.81      9864

```

### Stacking (Ensemble)

```
                      precision    recall  f1-score   support

             android       0.69      0.18      0.28       125
            bug_only       0.88      0.99      0.93      8472
           duplicate       0.00      0.00      0.00        63
         enhancement       0.61      0.07      0.12       336
    good first issue       0.00      0.00      0.00        71
         help wanted       0.36      0.01      0.03       270
       high priority       0.59      0.17      0.26        95
             invalid       0.00      0.00      0.00        37
                 ios       0.57      0.34      0.42        50
minor (quick review)       0.83      0.33      0.48        45
       navigation-ui       0.63      0.33      0.44        36
                  p1       0.38      0.05      0.09        57
                  p2       0.77      0.50      0.61       111
      priority: high       0.25      0.03      0.05        40
            question       0.00      0.00      0.00        56

            accuracy                           0.87      9864
           macro avg       0.44      0.20      0.25      9864
        weighted avg       0.82      0.87      0.82      9864

```

## Conclusion

**Previous Best Baseline:** XGBoost (Baseline) (F1 = 0.8182)  
**Best Ensemble:** Stacking (Ensemble) (F1 = 0.8247)  
**Overall Best Model:** Stacking (Ensemble) (F1 = 0.8247)  

Ensemble learning **improved** over the best baseline by +0.0065 F1 points. The best ensemble model has been saved as `models/best_model.joblib`.
