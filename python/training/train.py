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
from config import DATA_SOURCE, CURRENT_DATA_DIR, LEARNING_RATE, PAM50_CLASS_MAPPING, USE_ATAC
from graphs.build_graph import build_patient_graphs
from models.helix_net import HelixNet
from models.classification_head import ClassificationHead

CV_SPLITS_PATH = CURRENT_DATA_DIR / "cv_splits.json"

def _ensure_cv_splits_exist():
    """
    train.py depends on cv_splits.json. If missing, auto-generate on demand.
    """
    if CV_SPLITS_PATH.exists():
        return
    print(f"{CV_SPLITS_PATH} not found -- generating CV splits now via baselines.py...")
    from evaluation.baselines import load_data, generate_and_save_splits
    X, y, barcodes = load_data()
    generate_and_save_splits(X, y, barcodes)

def run_training(epochs: int = 40, batch_size: int = 16, use_fusion: bool = True):
    _ensure_cv_splits_exist()

    print("Loading graphs...")
    graphs = build_patient_graphs(use_atac=USE_ATAC)

    # Compute class weights from the ACTUAL training label distribution to counteract
    # majority-class collapse. Previously (5 epochs, no weighting) HelixNet predicted only
    # the majority class (LumA) in every single fold on real data -- a known failure mode
    # for GNNs on small, imbalanced cohorts, not a mysterious bug. This doesn't guarantee
    # good performance at n=42, but removes the single most avoidable cause of it.
    all_labels = np.array([g.y.item() for g in graphs])
    class_counts = np.bincount(all_labels, minlength=len(PAM50_CLASS_MAPPING))
    class_counts = np.where(class_counts == 0, 1, class_counts)  # avoid div-by-zero for absent classes
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Class weights (inverse-frequency, from {len(graphs)} patients): "
          f"{dict(zip(PAM50_CLASS_MAPPING.keys(), class_weights.round(2)))}")
    
    # Map barcode -> graph for easy fold assignment
    graph_map = {g.patient_barcode: g for g in graphs}
    
    with open(CV_SPLITS_PATH, 'r') as f:
        splits = json.load(f)
        
    results = []
    model_name = "HelixNet_Full" if use_fusion else "HelixNet_Ablated"
    
    print(f"\nStarting Cross-Validation Training ({DATA_SOURCE.capitalize()} Data)")
    print(f"Training {model_name} for {epochs} epochs.")
    
    for fold_name, fold_data in splits.items():
        print(f"\n--- {fold_name} ---")
        
        train_graphs = [graph_map[b] for b in fold_data['train'] if b in graph_map]
        test_graphs = [graph_map[b] for b in fold_data['test'] if b in graph_map]
        
        if len(train_graphs) == 0 or len(test_graphs) == 0:
            print(f"Skipping {fold_name}: no matching patient graphs found.")
            continue
            
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=len(test_graphs), shuffle=False)
        
        # Initialize model components (unfrozen)
        num_modalities = 2 if USE_ATAC else 1
        backbone = HelixNet(use_fusion=use_fusion, num_modalities=num_modalities)
        head = ClassificationHead()
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
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
                
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch.y.cpu().numpy())
                
        y_test = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        y_prob = np.concatenate(all_probs, axis=0)
        
        unique, counts = np.unique(y_pred, return_counts=True)
        pred_dist = dict(zip(unique, counts))
        print(f"  Test set predictions distribution: {pred_dist}", flush=True)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        try:
            auroc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro', labels=list(range(len(PAM50_CLASS_MAPPING))))
        except (ValueError, IndexError):
            auroc = np.nan
            
        results.append({
            "Model": model_name,
            "Fold": fold_name,
            "Accuracy": acc,
            "Macro_F1": f1,
            "AUROC": auroc
        })
        
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        avg_results = df_results.groupby("Model")[["Accuracy", "Macro_F1", "AUROC"]].mean().reset_index()
        avg_results["Fold"] = "Average"
        final_df = pd.concat([df_results, avg_results], ignore_index=True)
    else:
        final_df = df_results
    
    print("\n" + "="*50)
    print(f"{model_name} Evaluation Results ({DATA_SOURCE.capitalize()} Data)")
    print("="*50)
    print(final_df.to_string(index=False))
    
    return final_df

if __name__ == "__main__":
    run_training(epochs=40, use_fusion=True)
