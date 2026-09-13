import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_SOURCE, USE_ATAC

def load_dataset():
    """
    Unified entry point for loading the HelixNet dataset.
    Branches on config.DATA_SOURCE to load either synthetic or real data.
    Both paths guarantee the exact same schema, index, and dtype structures.
    """
    if DATA_SOURCE == "synthetic":
        from data.synthetic import generate_synthetic_data
        return generate_synthetic_data()
    elif DATA_SOURCE == "real":
        from data.real_loader import load_real_data
        return load_real_data(use_atac=USE_ATAC)
    else:
        raise ValueError(f"Unknown DATA_SOURCE in config.py: {DATA_SOURCE}")
