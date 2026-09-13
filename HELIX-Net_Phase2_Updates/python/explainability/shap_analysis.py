import os
import sys
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import PAM50_CLASS_MAPPING
from data.loader import load_dataset

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'figures'))
RANDOM_STATE = 42

def run_shap_analysis():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("Loading data for SHAP analysis...")
    df_meth, df_anno, df_pam50, df_atac = load_dataset()
    
    y = df_pam50['subtype'].map(PAM50_CLASS_MAPPING).values
    X = df_meth
    
    print("Training full RandomForest baseline model for interpretation...")
    # Train on all data for a global summary plot
    # EXPLICIT DEVIATION NOTE: Phase 3 uses XGBoost as the primary baseline, 
    # but newer XGBoost versions pass base_score as a string array for multiclass problems,
    # which currently crashes shap.TreeExplainer. We use RandomForestClassifier here
    # purely as a pragmatic workaround to generate the SHAP plot without crashing. 
    # Do not assume these feature importances perfectly mirror the XGBoost baseline.
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X, y)
    
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # SHAP on PyG GNN models requires specialized explainers (e.g. GNNExplainer via PyG or Captum),
    # which is skipped here in favor of analyzing the flat XGBoost baseline to identify raw driver features.
    
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    # shap_values for multiclass is a list of arrays (one per class). 
    # summary_plot handles it by showing stacked bars for feature importance across classes.
    shap.summary_plot(shap_values, X, show=False)
    
    out_path = os.path.join(FIGURES_DIR, "shap_baseline_summary.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved SHAP summary plot to {out_path}")
    
    # TODO: When the real TCGA dataset replaces the synthetic one, this script should be updated 
    # to check whether the top-ranked CpG sites based on SHAP importance map to known breast 
    # cancer oncogenes or tumor suppressors (e.g. ESR1, ERBB2, PIK3CA, TP53, GATA3, FOXA1).
    # Currently, on synthetic data, the CpG probe IDs are arbitrary and carry no genomic meaning.

if __name__ == "__main__":
    run_shap_analysis()
