import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_SOURCE

def load_dataset():
    """
    Unified entry point for loading the HelixNet dataset.
    Branches on config.DATA_SOURCE to load either synthetic or real data.
    Both paths guarantee the exact same schema, index, and dtype structures.
    """
    if DATA_SOURCE == "synthetic":
        from data.synthetic import generate_synthetic_data
        # This regenerates and saves synthetic data each time, 
        # or we could load from CSV, but synthetic.py naturally returns the DFs.
        # However, to be perfectly uniform, we'll return what generate_synthetic_data returns
        return generate_synthetic_data()
    elif DATA_SOURCE == "real":
        from data.real_loader import load_real_data
        return load_real_data()
    else:
        raise ValueError(f"Unknown DATA_SOURCE in config.py: {DATA_SOURCE}")
