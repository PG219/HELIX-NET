import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.train import run_training
from config import BASE_DIR

RESULTS_DIR = BASE_DIR / "outputs" / "tables"

def run_ablation():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("="*60)
    print("ABLATION STUDY: Full Model vs Ablated Model")
    print("="*60)
    
    # 1. Run Full Model
    print("\n[1/2] Training Full Model (use_fusion=True)...")
    df_full = run_training(epochs=5, use_fusion=True)
    
    # 2. Run Ablated Model
    print("\n[2/2] Training Ablated Model (use_fusion=False)...")
    df_ablated = run_training(epochs=5, use_fusion=False)
    
    # 3. Merge and Compute Deltas
    df_merged = pd.merge(
        df_full.drop(columns=["Model"]), 
        df_ablated.drop(columns=["Model"]), 
        on="Fold", 
        suffixes=('_Full', '_Ablated')
    )
    
    df_merged['Accuracy_Delta'] = df_merged['Accuracy_Full'] - df_merged['Accuracy_Ablated']
    df_merged['Macro_F1_Delta'] = df_merged['Macro_F1_Full'] - df_merged['Macro_F1_Ablated']
    df_merged['AUROC_Delta'] = df_merged['AUROC_Full'] - df_merged['AUROC_Ablated']
    
    csv_path = RESULTS_DIR / "ablation_comparison.csv"
    df_merged.to_csv(csv_path, index=False)
    
    print("\n" + "="*80)
    print("ABLATION COMPARISON RESULTS")
    print("="*80)
    
    print_cols = [
        "Fold", 
        "Accuracy_Full", "Accuracy_Ablated", "Accuracy_Delta",
        "Macro_F1_Full", "Macro_F1_Ablated", "Macro_F1_Delta",
        "AUROC_Full", "AUROC_Ablated", "AUROC_Delta"
    ]
    print(df_merged[print_cols].to_string(index=False))
    print(f"\nSaved comparison to {csv_path}")

if __name__ == "__main__":
    run_ablation()
