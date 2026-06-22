# Data Directory

This folder is intended to hold all raw and processed dataset files for the mobile application defect classification project.

## Structure

```
data/
    {ACTIVE_DATASET}/
        raw/
        processed/
```

- **raw/**: Contains the original dataset files (e.g., `.json`, `.csv`, `.xlsx`).
- **processed/**: Contains the output from the text preprocessing pipeline (e.g., `preprocessed_issues.csv`).

## Note on Datasets

The actual dataset files (e.g., `issues.json` or `preprocessed_issues.csv`) are intentionally excluded from version control to prevent exposing large, proprietary, or unauthorized data. 

To add a new dataset:
1. Create a new folder under `data/` (e.g., `dataset_2/raw/`).
2. Place your raw dataset files inside this `raw/` directory.
3. Update `ACTIVE_DATASET = "dataset_2"` in `src/utils.py`.
4. Run the pipeline.
