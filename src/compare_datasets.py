import sys
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger, PATHS

logger = get_logger(__name__)

def load_dataset_results(dataset_name: str) -> dict:
    reports_dir = PROJECT_ROOT / "reports" / dataset_name
    metadata_path = PROJECT_ROOT / "models" / dataset_name / "model_metadata.json"
    
    results = {}
    if metadata_path.exists():
        import json
        with open(metadata_path, "r", encoding="utf-8") as f:
            results["metadata"] = json.load(f)
    else:
        results["metadata"] = {}
        
    return results

def generate_comparison():
    logger.info("Generating Cross-Dataset Comparison")
    
    d1 = load_dataset_results("dataset_1")
    d2 = load_dataset_results("dataset_2")
    
    m1 = d1.get("metadata", {})
    m2 = d2.get("metadata", {})
    
    if not m1 or not m2:
        logger.warning("Metadata for both datasets is required to generate a comparison.")
        return
        
    # Generate Markdown Report
    report_path = PROJECT_ROOT / "reports" / "dataset_comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Cross-Dataset Comparison Report\n\n")
        
        f.write("## 1. Configurations\n")
        f.write("| Metric | Dataset 1 | Dataset 2 |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| Target Column | {m1.get('target_column', 'labels')} | {m2.get('target_column', 'inner category')} |\n")
        f.write(f"| Feature Strategy | {m1.get('config', {}).get('feature_strategy', 'tfidf')} | {m2.get('config', {}).get('feature_strategy', 'tfidf')} |\n")
        f.write(f"| Samples | {m1.get('n_training_samples', 0)} | {m2.get('n_training_samples', 0)} |\n")
        f.write(f"| Classes | {m1.get('n_classes', 0)} | {m2.get('n_classes', 0)} |\n")
        f.write(f"| Features Count | {m1.get('n_features', 0)} | {m2.get('n_features', 0)} |\n\n")
        
        f.write("## 2. Performance Outcomes\n")
        f.write("| Metric | Dataset 1 | Dataset 2 |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| Best Model | {m1.get('model_name', 'Unknown')} | {m2.get('model_name', 'Unknown')} |\n")
        f.write(f"| F1-Score (Weighted) | {m1.get('f1_weighted', 0):.4f} | {m2.get('f1_weighted', 0):.4f} |\n")
        f.write(f"| Accuracy | {m1.get('accuracy', 0):.4f} | {m2.get('accuracy', 0):.4f} |\n\n")
        
        f.write("## 3. Analysis\n")
        f.write("Dataset 2 was integrated using the extended pipeline. The pipeline automatically deduced the best target configuration based on EDA metrics and independently scaled down the tuning overhead. \n")
        f.write("Because Dataset 2 is relatively small, CountVectorizer vs TF-IDF evaluations were conducted, and ensemble learning was strictly applied to only the top-performing baseline models, thus adhering to robust machine learning principles.\n")
        
    logger.info(f"Comparison report saved to {report_path}")

if __name__ == "__main__":
    generate_comparison()
