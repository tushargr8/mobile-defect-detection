# Cross-Dataset Comparison Report

## 1. Configurations
| Metric | Dataset 1 | Dataset 2 |
| :--- | :--- | :--- |
| Target Column | labels | inner category |
| Feature Strategy | tfidf | tfidf |
| Samples | 55896 | 147 |
| Classes | 15 | 3 |
| Features Count | 10000 | 2361 |

## 2. Performance Outcomes
| Metric | Dataset 1 | Dataset 2 |
| :--- | :--- | :--- |
| Best Model | Stacking (Ensemble) | Logistic Regression (Baseline) |
| F1-Score (Weighted) | 0.8247 | 0.8445 |
| Accuracy | 0.8706 | 0.8519 |

## 3. Analysis
Dataset 2 was integrated using the extended pipeline. The pipeline automatically deduced the best target configuration based on EDA metrics and independently scaled down the tuning overhead. 
Because Dataset 2 is relatively small, CountVectorizer vs TF-IDF evaluations were conducted, and ensemble learning was strictly applied to only the top-performing baseline models, thus adhering to robust machine learning principles.
