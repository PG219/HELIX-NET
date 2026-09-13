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
from config import CURRENT_DATA_DIR, PAM50_CLASS_MAPPING
from data.loader import load_dataset

CV_SPLITS_PATH = CURRENT_DATA_DIR / "cv_splits.json"
CV_N_SPLITS = 5
RANDOM_STATE = 42

def load_data():
    df_meth, df_anno, df_pam50, df_atac = load_dataset()
    
    # Map to integer labels via shared config mapping
    y = df_pam50['subtype'].map(PAM50_CLASS_MAPPING).values
    X = df_meth.values
    barcodes = df_meth.index.tolist()
    
    return X, y, barcodes

from sklearn.model_selection import StratifiedKFold, KFold

def generate_and_save_splits(X, y, barcodes):
    """
    Generate CV splits and save barcode assignments to disk.
    Uses StratifiedKFold when class counts allow, or falls back to KFold
    when minority classes have < 2 samples (e.g. singletons like HER2 in small cohorts).
    """
    _, class_counts = np.unique(y, return_counts=True)
    min_class_count = class_counts.min()

    if min_class_count >= 2:
        n_splits = min(CV_N_SPLITS, min_class_count)
        if n_splits < CV_N_SPLITS:
            print(f"NOTE: Smallest class has {min_class_count} samples -- using {n_splits}-fold StratifiedKFold.")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        split_generator = splitter.split(X, y)
    else:
        n_splits = CV_N_SPLITS
        print(f"NOTE: Minority class has {min_class_count} sample(s) (< 2) -- using {n_splits}-fold KFold for cross-validation.")
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        split_generator = splitter.split(X)

    splits = {}
    for fold_idx, (train_idx, test_idx) in enumerate(split_generator):
        train_barcodes = [barcodes[i] for i in train_idx]
        test_barcodes = [barcodes[i] for i in test_idx]
        splits[f"fold_{fold_idx}"] = {
            "train": train_barcodes,
            "test": test_barcodes
        }
        
    CURRENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CV_SPLITS_PATH, 'w') as f:
        json.dump(splits, f, indent=2)
        
    print(f"Saved {n_splits}-fold CV splits to {CV_SPLITS_PATH}")
    return splits

def evaluate_model(model, X, y, train_idx, test_idx, num_classes=len(PAM50_CLASS_MAPPING)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    train_classes = np.unique(y_train)
    is_xgb = isinstance(model, XGBClassifier)
    
    if is_xgb:
        local_mapping = {c: i for i, c in enumerate(train_classes)}
        y_train_mapped = np.array([local_mapping[val] for val in y_train])
        model_fold = XGBClassifier(
            objective="multi:softprob",
            num_class=len(train_classes),
            eval_metric="mlogloss",
            random_state=RANDOM_STATE
        )
        model_fold.fit(X_train, y_train_mapped)
        y_prob_local = model_fold.predict_proba(X_test)
        
        y_prob = np.zeros((len(X_test), num_classes), dtype=np.float32)
        for local_idx, orig_class in enumerate(train_classes):
            y_prob[:, orig_class] = y_prob_local[:, local_idx]
        y_pred = np.argmax(y_prob, axis=1)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = np.zeros((len(X_test), num_classes), dtype=np.float32)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            for idx, c in enumerate(model.classes_):
                y_prob[:, c] = probs[:, idx]
        else:
            for i, p in enumerate(y_pred):
                y_prob[i, p] = 1.0
        
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    try:
        auroc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro', labels=list(range(num_classes)))
    except (ValueError, IndexError):
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
    avg_results = df_results.groupby("Model")[["Accuracy", "Macro_F1", "AUROC"]].mean().reset_index()
    avg_results["Fold"] = "Average"
    
    final_df = pd.concat([df_results, avg_results], ignore_index=True)
    
    print("\n" + "="*50)
    print("Baseline Evaluation Results")
    print("="*50)
    print(final_df.to_string(index=False))
    
    return final_df

if __name__ == "__main__":
    run_baselines()
