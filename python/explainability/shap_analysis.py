import os
import sys
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import BASE_DIR, PAM50_CLASS_MAPPING
from data.loader import load_dataset

FIGURES_DIR = BASE_DIR / "outputs" / "figures"
RANDOM_STATE = 42

def run_shap_analysis():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("Loading data for SHAP analysis...")
    df_meth, df_anno, df_pam50, df_atac = load_dataset()
    
    y = df_pam50['subtype'].map(PAM50_CLASS_MAPPING).values
    X = df_meth
    
    print("Training RandomForest baseline model for interpretation...")
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X, y)
    
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    
    out_path = FIGURES_DIR / "shap_baseline_summary.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved SHAP summary plot to {out_path}")

if __name__ == "__main__":
    run_shap_analysis()
