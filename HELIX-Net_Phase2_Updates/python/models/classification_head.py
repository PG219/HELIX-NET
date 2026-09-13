import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import EMBEDDING_DIM, PAM50_CLASS_MAPPING

class ClassificationHead(nn.Module):
    def __init__(self, in_features: int = EMBEDDING_DIM, num_classes: int = len(PAM50_CLASS_MAPPING)):
        super().__init__()
        
        hidden_dim = 32
        
        self.net = nn.Sequential(
            # [batch_size, in_features] -> [batch_size, hidden_dim]
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            # Dropout for regularization due to small sample size (74 samples)
            nn.Dropout(p=0.3),
            # [batch_size, hidden_dim] -> [batch_size, num_classes]
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: graph-level embeddings of shape [batch_size, in_features]
        Returns: logits of shape [batch_size, num_classes]
        """
        # x is [batch_size, EMBEDDING_DIM]
        # net(x) is [batch_size, 5]
        return self.net(x)
