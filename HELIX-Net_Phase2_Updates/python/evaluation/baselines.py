import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Add python dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_SOURCE, CURRENT_DATA_DIR, PAM50_CLASS_MAPPING

CV_SPLITS_PATH = CURRENT_DATA_DIR / "cv_splits.json"
CV_N_SPLITS = 5
RANDOM_STATE = 42

from data.loader import load_dataset

def load_data():
    df_meth, df_anno, df_pam50, df_atac = load_dataset()
    
    # Map to integer labels via shared config mapping
    y = df_pam50['subtype'].map(PAM50_CLASS_MAPPING).values
    X = df_meth.values
    barcodes = df_meth.index.tolist()
    
    return X, y, barcodes

def generate_and_save_splits(X, y, barcodes, CV_SPLITS_PATH=CV_SPLITS_PATH, CV_N_SPLITS=5, RANDOM_STATE=42):
    """
    Generate StratifiedKFold splits and save barcode assignments to disk.

    IMPORTANT: this assumes singleton/tiny classes (e.g. a class with only 1 member) have
    ALREADY been filtered out upstream via data.loader.filter_singleton_classes -- applied
    consistently to X, y, and barcodes together, not just to labels. If that filtering
    didn't happen, min_class_count could still be 1, and this will correctly refuse to
    proceed (raising ValueError) rather than silently falling back to unstratified KFold,
    which is what produced NaN AUROC and wildly unbalanced folds previously. Don't add a
    KFold fallback branch here -- fix the input data instead, at the loader level.

    n_splits is capped to the smallest class's sample count so StratifiedKFold never
    hard-errors on a small real cohort (e.g. n=42 after HER2 exclusion, Normal=3 becomes
    the new minority, requiring n_splits<=3 rather than the default 5).
    """
    _, class_counts = np.unique(y, return_counts=True)
    min_class_count = class_counts.min()
    n_splits = min(CV_N_SPLITS, min_class_count)

    if n_splits < CV_N_SPLITS:
        print(f"NOTE: smallest class has only {min_class_count} samples -- reducing n_splits "
              f"from {CV_N_SPLITS} to {n_splits} to keep StratifiedKFold valid.")
    if n_splits < 2:
        raise ValueError(
            f"Smallest class has only {min_class_count} sample(s) -- cannot do any meaningful "
            "cross-validation split. This should have been caught by filter_singleton_classes() "
            "at the data loading stage -- check that it ran and was applied to this y before "
            "calling generate_and_save_splits."
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    splits = {}
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        train_barcodes = [barcodes[i] for i in train_idx]
        test_barcodes = [barcodes[i] for i in test_idx]
        splits[f"fold_{fold_idx}"] = {
            "train": train_barcodes,
            "test": test_barcodes
        }

    os.makedirs(os.path.dirname(CV_SPLITS_PATH), exist_ok=True)
    with open(CV_SPLITS_PATH, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"Saved {n_splits}-fold CV splits to {CV_SPLITS_PATH}")
    return splits


def evaluate_model(model, X, y, train_idx, test_idx, num_classes=len(PAM50_CLASS_MAPPING)):
    """
    num_classes: pass the TOTAL number of classes in the label mapping (e.g. 4 after HER2
    exclusion), not just however many appear in this particular fold's y_test. Passing this
    explicitly to roc_auc_score's `labels` param (matching what train.py already does
    correctly) prevents spurious crashes/NaNs when a fold's test set happens to be missing
    a class that IS in the overall label set -- the previous version of this function didn't
    pass `labels` at all, which is part of why AUROC came back NaN so often.
    """
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    try:
        auroc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro',
                               labels=list(range(num_classes)))
    except ValueError as e:
        print(f"AUROC error (likely a class truly absent from both this fold's train AND "
              f"model's predict_proba columns): {e}")
        auroc = np.nan

    return acc, f1, auroc

def run_baselines():
    X, y, barcodes = load_data()
    splits = generate_and_save_splits(X, y, barcodes)
    
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            objective="multi:softprob", 
            num_class=len(PAM50_CLASS_MAPPING), 
            eval_metric="mlogloss", 
            random_state=RANDOM_STATE
        )
    }
    
    results = []
    
    for fold_name, fold_data in splits.items():
        # Map barcodes back to indices for array slicing
        train_idx = [barcodes.index(b) for b in fold_data['train']]
        test_idx = [barcodes.index(b) for b in fold_data['test']]
        
        for model_name, model in models.items():
            acc, f1, auroc = evaluate_model(model, X, y, train_idx, test_idx)
            results.append({
                "Model": model_name,
                "Fold": fold_name,
                "Accuracy": acc,
                "Macro_F1": f1,
                "AUROC": auroc
            })
            
    df_results = pd.DataFrame(results)
    
    # Calculate averages
    avg_results = df_results.groupby("Model")[["Accuracy", "Macro_F1", "AUROC"]].mean().reset_index()
    avg_results["Fold"] = "Average"
    
    final_df = pd.concat([df_results, avg_results], ignore_index=True)
    
    print("\n" + "="*50)
    print(f"Baseline Evaluation Results ({DATA_SOURCE.capitalize()} Data)")
    print("="*50)
    print(final_df.to_string(index=False))
    
    return final_df

if __name__ == "__main__":
    run_baselines()
