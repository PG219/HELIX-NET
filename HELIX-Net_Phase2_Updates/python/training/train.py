import os
import sys
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CURRENT_DATA_DIR, LEARNING_RATE, PAM50_CLASS_MAPPING
from graphs.build_graph import build_patient_graphs
from models.helix_net import HelixNet
from models.classification_head import ClassificationHead

CV_SPLITS_PATH = CURRENT_DATA_DIR / "cv_splits.json"

def _ensure_cv_splits_exist():
    """
    train.py depends on cv_splits.json, but only baselines.py generates it.
    Running train.py fresh (without having run baselines.py first) used to fail with a bare
    FileNotFoundError and no explanation. This generates the splits on demand instead, using
    the exact same logic/seed baselines.py uses, so results stay comparable across the two.
    """
    if CV_SPLITS_PATH.exists():
        return
    print(f"{CV_SPLITS_PATH} not found -- generating CV splits now via baselines.py "
          f"(this is expected on a fresh run; baselines.py doesn't need to be run manually first).")
    from evaluation.baselines import load_data, generate_and_save_splits
    X, y, barcodes = load_data()
    generate_and_save_splits(X, y, barcodes)

def run_training(epochs: int = 5, batch_size: int = 16, use_fusion: bool = True):
    _ensure_cv_splits_exist()

    print("Loading graphs...")
    graphs = build_patient_graphs()
    
    # Map barcode -> graph for easy fold assignment
    graph_map = {g.patient_barcode: g for g in graphs}
    
    with open(CV_SPLITS_PATH, 'r') as f:
        splits = json.load(f)
        
    results = []
    model_name = "HelixNet_Full" if use_fusion else "HelixNet_Ablated"
    
    print("\nStarting Cross-Validation Training (Synthetic Data)")
    print(f"NOTE: Training {model_name} for a fixed number of {epochs} epochs just to check mechanics and loss convergence.")
    
    for fold_name, fold_data in splits.items():
        print(f"\n--- {fold_name} ---")
        
        train_graphs = [graph_map[b] for b in fold_data['train'] if b in graph_map]
        test_graphs = [graph_map[b] for b in fold_data['test'] if b in graph_map]
        
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=len(test_graphs), shuffle=False)
        
        # Initialize model components (unfrozen)
        backbone = HelixNet(use_fusion=use_fusion)
        head = ClassificationHead()
        
        criterion = nn.CrossEntropyLoss()
        # Optimize both modules
        optimizer = Adam(list(backbone.parameters()) + list(head.parameters()), lr=LEARNING_RATE)
        
        first_epoch_loss = None
        last_epoch_loss = None
        
        for epoch in range(epochs):
            backbone.train()
            head.train()
            epoch_loss = 0.0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Forward
                graph_emb = backbone(batch)
                logits = head(graph_emb)
                
                # Assert no NaNs
                assert not torch.isnan(graph_emb).any(), f"NaN in embeddings at epoch {epoch+1}"
                assert not torch.isnan(logits).any(), f"NaN in logits at epoch {epoch+1}"
                
                # Loss
                loss = criterion(logits, batch.y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch.num_graphs
                
            epoch_loss /= len(train_graphs)
            
            # Epoch-level logging
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}", flush=True)
            
            if epoch == 0:
                first_epoch_loss = epoch_loss
            if epoch == epochs - 1:
                last_epoch_loss = epoch_loss
                
        print(f"Loss Start (Ep 1): {first_epoch_loss:.4f} -> Loss End (Ep {epochs}): {last_epoch_loss:.4f}")
        
        # Evaluation
        backbone.eval()
        head.eval()
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                graph_emb = backbone(batch)
                logits = head(graph_emb)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_probs.append(probs.numpy())
                all_preds.append(preds.numpy())
                all_labels.append(batch.y.numpy())
                
        y_test = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        y_prob = np.concatenate(all_probs, axis=0)
        
        # Check for majority class collapse
        unique, counts = np.unique(y_pred, return_counts=True)
        pred_dist = dict(zip(unique, counts))
        print(f"  Test set predictions distribution: {pred_dist}", flush=True)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        try:
            auroc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro', labels=list(range(len(PAM50_CLASS_MAPPING))))
        except ValueError:
            auroc = np.nan
            
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
    print(f"{model_name} Evaluation Results (Synthetic Data)")
    print("="*50)
    print(final_df.to_string(index=False))
    
    return final_df

if __name__ == "__main__":
    run_training(epochs=5, use_fusion=True)
