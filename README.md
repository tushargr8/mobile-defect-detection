# Software Defect Classification Using Machine Learning

## Objective

Classify mobile application defect reports (GitHub issues) into categories using NLP and Machine Learning. The project implements a complete ML pipeline — from data preprocessing through model evaluation and ensemble learning.

**Reference Paper:** *From Bugs to Benchmarks: A Comprehensive Survey of Software Defect Datasets* ([ACM Digital Library](https://dl.acm.org/doi/pdf/10.1145/3797033))

## Datasets

| Attribute | Dataset 1 (Issues) | Dataset 2 (Functional Bugs) |
|---|---|---|
| Source | GitHub Issues (JSON) | Annotated Dataset (XLSX) |
| Records | 82,455 (filtered to 65,760) | 295 (filtered to 173) |
| Target Variable | `labels` (parsed) | `inner category` (direct) |
| Classes | 15 | 3 |
| Feature Strategy | TF-IDF | CountVectorizer |

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Select the active dataset

Edit `src/utils.py` and set:

```python
ACTIVE_DATASET = "dataset_1"   # or "dataset_2"
```

### 2. Run the baseline pipeline

```bash
python main.py
```

### 3. Run ensemble learning (after baselines are complete)

```bash
python run_ensembles.py
```

### 4. Compare datasets (after both datasets are trained)

```bash
python src/compare_datasets.py
```

## Folder Structure

```
project/
├── data/
│   ├── dataset_1/          # Raw + processed data
│   └── dataset_2/
├── models/
│   ├── dataset_1/          # Trained models, vectorizer, encoder
│   └── dataset_2/
├── reports/
│   ├── dataset_1/          # Metrics, figures, logs
│   ├── dataset_2/
│   └── dataset_comparison_report.md
├── src/                    # Core pipeline modules
│   ├── preprocessing.py    # Data loading, cleaning, text preprocessing
│   ├── feature_engineering.py  # TF-IDF / CountVectorizer features
│   ├── train.py            # Model training + hyperparameter tuning
│   ├── evaluate.py         # Evaluation metrics + visualizations
│   ├── predict.py          # Inference with saved models
│   ├── data_collector.py   # Fetch missing text from GitHub URLs
│   ├── compare_datasets.py # Cross-dataset comparison
│   └── utils.py            # Config, logging, paths
├── main.py                 # Full pipeline orchestrator
├── run_ensembles.py        # Ensemble learning pipeline
├── resume_pipeline.py      # Resume pipeline (skip tuning)
├── requirements.txt
└── README.md
```

## Results Summary

### Dataset 1 — Best Models

| Model | Accuracy | F1 (Weighted) |
|---|---|---|
| Stacking (Ensemble) | 0.8706 | 0.8247 |
| XGBoost (Baseline) | 0.8683 | 0.8182 |
| LightGBM (Baseline) | 0.8660 | 0.8167 |

### Dataset 2 — Best Models

| Model | Accuracy | F1 (Weighted) |
|---|---|---|
| Logistic Regression | 0.8519 | 0.8445 |
| Random Forest (Tuned) | 0.8148 | 0.8108 |
| XGBoost (Baseline) | 0.8148 | 0.8022 |

## Models Used

- Logistic Regression
- Multinomial Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- LightGBM
- Ensemble Methods (Hard Voting, Soft Voting, Stacking)

---

*Developed for an internship project on Mobile Application Defect Classification.*
