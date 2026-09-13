import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import sys

# Add python dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import EMBEDDING_DIM, NUM_GNN_LAYERS, USE_ATAC
from attention.cross_modal_fusion import CrossModalFusion
from graphs.build_graph import build_patient_graphs

class HelixNet(nn.Module):
    def __init__(self, use_fusion: bool = True, num_modalities: int = None):
        super().__init__()

        # num_modalities defaults to config.USE_ATAC (2 if ATAC available, 1 if methylation-only).
        # Explicit override supported for scripts that construct graphs directly (e.g. tests).
        if num_modalities is None:
            num_modalities = 2 if USE_ATAC else 1

        # Cross modal fusion
        self.fusion = CrossModalFusion(embed_dim=EMBEDDING_DIM, num_heads=4, use_fusion=use_fusion,
                                        num_modalities=num_modalities)

        # GNN Backbone
        self.gnn_layers = nn.ModuleList()
        for i in range(NUM_GNN_LAYERS):
            # Using GATv2Conv to propagate node embeddings
            # We set concat=False to maintain EMBEDDING_DIM output size
            self.gnn_layers.append(
                GATv2Conv(in_channels=EMBEDDING_DIM, out_channels=EMBEDDING_DIM, heads=4, concat=False)
            )

    def forward(self, data: Data) -> torch.Tensor:
        """
        data: PyG Data object containing x, edge_index, and batch
        Returns: graph-level embeddings of shape [batch_size, EMBEDDING_DIM]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Ensure batch attribute exists if we are running single graph without DataLoader
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 1. Fusion Layer
        # x is [N_total, num_modalities]
        # [N_total, num_modalities] -> [N_total, EMBEDDING_DIM]
        x = self.fusion(x)

        # 2. GNN Layers
        for conv in self.gnn_layers:
            # [N_total, EMBEDDING_DIM] -> [N_total, EMBEDDING_DIM]
            x = conv(x, edge_index)
            x = torch.relu(x)

        # 3. Global Pooling
        # [N_total, EMBEDDING_DIM] -> [batch_size, EMBEDDING_DIM]
        graph_embeds = global_mean_pool(x, batch)

        return graph_embeds

    def get_last_attention_weights(self):
        """
        Returns the most recent cross-modal attention weights captured during forward(),
        or None if the fusion layer is in single-modality/no-fusion mode (nothing to attend
        across) or forward() hasn't been called yet. For interpretability (proposal 5.2.6):
        map these back to which modality/genomic position drove a prediction.
        """
        return getattr(self.fusion, "last_attn_weights", None)

    def freeze_backbone(self, freeze: bool = True):
        """
        Freezes or unfreezes the GNN backbone layers.
        Leaves CrossModalFusion and any subsequent heads trainable.
        """
        for param in self.gnn_layers.parameters():
            param.requires_grad = not freeze


if __name__ == "__main__":
    print("Loading graphs for Phase 2 self-test...")
    graphs = build_patient_graphs()
    
    # Create DataLoader with a batch size
    batch_size = 16
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    
    # Init model
    model = HelixNet(use_fusion=True)
    
    # Get a batch
    batch_data = next(iter(loader))
    
    print("\n--- Phase 2 Self-Test Report ---")
    print(f"Batch shape check:")
    print(f"  Input node features: {list(batch_data.x.shape)} (Total nodes across {batch_size} graphs)")
    
    # Forward pass
    out = model(batch_data)
    
    print(f"  Forward pass successful!")
    print(f"  Output graph embedding shape: {list(out.shape)} (Expected: [{batch_size}, {EMBEDDING_DIM}])")
    assert out.size(0) == batch_size, f"Output batch dimension mismatch! Got {out.size(0)} expected {batch_size}"
    assert out.size(1) == EMBEDDING_DIM, "Output embedding dimension mismatch!"
    
    # Test Freezing
    print("\nTesting Freeze/Unfreeze Mechanism:")
    # Check baseline (should be True)
    gnn_grads = [p.requires_grad for p in model.gnn_layers.parameters()]
    fusion_grads = [p.requires_grad for p in model.fusion.parameters()]
    print(f"  Initial - GNN layers require_grad: {all(gnn_grads)}")
    print(f"  Initial - Fusion layers require_grad: {all(fusion_grads)}")
    
    # Freeze
    model.freeze_backbone(True)
    gnn_grads = [p.requires_grad for p in model.gnn_layers.parameters()]
    fusion_grads = [p.requires_grad for p in model.fusion.parameters()]
    print(f"  After freeze - GNN layers require_grad (should be False): {any(gnn_grads)}")
    print(f"  After freeze - Fusion layers require_grad (should be True): {all(fusion_grads)}")
    
    # Unfreeze
    model.freeze_backbone(False)
    gnn_grads = [p.requires_grad for p in model.gnn_layers.parameters()]
    fusion_grads = [p.requires_grad for p in model.fusion.parameters()]
    print(f"  After unfreeze - GNN layers require_grad (should be True): {all(gnn_grads)}")
    print(f"  After unfreeze - Fusion layers require_grad (should be True): {all(fusion_grads)}")
    
    print("\nPhase 2 Self-Test Complete.")
