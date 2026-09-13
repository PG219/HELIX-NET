import os
import sys
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

# Add python dir to path if not exists
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import PROXIMITY_THRESHOLD_KB, PAM50_CLASS_MAPPING, USE_ATAC
from data.loader import load_dataset

def build_edges(df_anno: pd.DataFrame, proximity_kb: int = PROXIMITY_THRESHOLD_KB) -> torch.Tensor:
    """
    Builds the edge_index for the graph. Two CpGs are connected if they are on the 
    same chromosome and their distance is <= proximity_kb * 1000.
    Returns edge_index of shape [2, num_edges].
    """
    threshold_bp = proximity_kb * 1000
    edge_list = []
    
    # Create a mapping from probe_id to node_index (0 to N-1)
    probe_to_idx = {probe: i for i, probe in enumerate(df_anno.index)}
    
    for chrom, group in df_anno.groupby("chromosome"):
        indices = group.index
        positions = group["position"].values
        node_idxs = [probe_to_idx[p] for p in indices]
        
        # O(N^2) within chromosome is fine for 5000 total nodes
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if abs(positions[i] - positions[j]) <= threshold_bp:
                    u = node_idxs[i]
                    v = node_idxs[j]
                    # Add undirected edges
                    edge_list.append((u, v))
                    edge_list.append((v, u))
                    
    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    return edge_index

def build_patient_graphs(use_atac: bool = USE_ATAC) -> list[Data]:
    """
    Builds one torch_geometric.data.Data object per patient.

    use_atac: if False, node features are methylation-only ([Nodes, 1]) instead of
    methylation+ATAC ([Nodes, 2]). Must match what CrossModalFusion/HelixNet was
    constructed with (they also read USE_ATAC from config by default) or the fusion
    layer's input dimension won't match the data's feature dimension.
    """
    df_meth, df_anno, df_pam50, df_atac = load_dataset()

    if use_atac and df_atac is None:
        raise ValueError(
            "build_patient_graphs(use_atac=True) but the loaded dataset has no ATAC data "
            "(df_atac is None). Either set config.USE_ATAC=False / pass use_atac=False, "
            "or check that DATA_SOURCE's loader actually returns ATAC data."
        )

    print("Building global graph edge topology...")
    edge_index = build_edges(df_anno, PROXIMITY_THRESHOLD_KB)
    
    graphs = []
    
    # Loop over patients
    for patient_barcode in df_meth.index:
        # Extract methylation for this patient
        # [Nodes=5000] -> single feature per node
        meth_features = df_meth.loc[patient_barcode].values

        if use_atac:
            # Extract ATAC accessibility for this patient (transposed relation handled by joining correctly via index)
            # atac_accessibility rows are probes, cols are patients
            # [Nodes=5000] -> single feature per node
            atac_features = df_atac[patient_barcode].values

            # Concatenate features
            # [Nodes=5000, Features=2] -> methylation + atac per probe
            x = np.stack([meth_features, atac_features], axis=1)
        else:
            # Methylation-only: [Nodes=5000, Features=1]
            x = meth_features.reshape(-1, 1)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        
        # Encode PAM50 label
        subtype_str = df_pam50.loc[patient_barcode, "subtype"]
        y_label = PAM50_CLASS_MAPPING[subtype_str]
        # [1] -> single class integer
        y_tensor = torch.tensor([y_label], dtype=torch.long)
        
        data = Data(
            x=x_tensor, 
            edge_index=edge_index.clone(), 
            y=y_tensor,
            patient_barcode=patient_barcode # Store for reference
        )
        graphs.append(data)
        
    return graphs

if __name__ == "__main__":
    print("Starting Phase 1 self-test...")
    graphs = build_patient_graphs()
    
    print(f"\n--- Phase 1 Self-Test Report ---")
    print(f"Total patient graphs built: {len(graphs)}")
    
    # Summary of first graph structure
    sample_data = graphs[0]
    print(f"\nSample Graph Data Object Structure:")
    print(f"  Field names: {sample_data.keys()}")
    print(f"  x (Node Features) shape: {list(sample_data.x.shape)} [Nodes, Features]")
    print(f"  x dtype: {sample_data.x.dtype}")
    print(f"  edge_index shape: {list(sample_data.edge_index.shape)} [2, Edges]")
    print(f"  edge_index dtype: {sample_data.edge_index.dtype}")
    print(f"  y (Label) shape: {list(sample_data.y.shape)} [1]")
    print(f"  y dtype: {sample_data.y.dtype}")
    
    # Verification checks
    orphan_count = 0
    for i, data in enumerate(graphs):
        num_edges = data.edge_index.size(1) if data.edge_index.numel() > 0 else 0
        
        if num_edges == 0:
            orphan_count += 1
            print(f"WARNING: Graph for patient {data.patient_barcode} has 0 edges (orphan graph).")
            
        if num_edges > 0:
            # Check valid COO format (2, num_edges)
            assert data.edge_index.size(0) == 2, f"edge_index must have shape [2, num_edges], got {data.edge_index.shape}"
            # No out of range indices
            assert data.edge_index.max() < sample_data.x.size(0), f"edge_index has out of range max index {data.edge_index.max()}"
            assert data.edge_index.min() >= 0, f"edge_index has out of range min index {data.edge_index.min()}"
            
    print(f"\nCompleted verification. Orphan graphs found: {orphan_count}")
