import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import get_logger, PATHS, CONFIG
from src.preprocessing import preprocess_pipeline, load_data

logger = get_logger(__name__)

def run_eda():
    logger.info(f"Starting EDA for {CONFIG['target_column']}")
    eda_dir = PATHS["reports_dir"] / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_data()
    
    with open(eda_dir / "eda_report.md", "w") as f:
        f.write(f"# EDA Report for {PATHS['raw_data_dir'].parent.name}\n\n")
        f.write(f"## Dataset Shape\n{df.shape}\n\n")
        f.write(f"## Missing Values\n```\n{df.isnull().sum().to_string()}\n```\n\n")
        
        # Analyze categorical candidates if they exist
        candidates = ["inner category", "leaf category", "fix pattern"]
        for c in candidates:
            if c in df.columns:
                f.write(f"## Candidate: {c}\n```\n{df[c].value_counts(dropna=False).to_string()}\n```\n\n")
    
    logger.info(f"EDA report generated at {eda_dir / 'eda_report.md'}")

if __name__ == "__main__":
    run_eda()
