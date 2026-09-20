from pathlib import Path

# Data assumptions and paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SYNTHETIC_DATA_DIR = BASE_DIR / "data" / "processed" / "synthetic"
REAL_DATA_DIR = BASE_DIR / "data" / "processed" / "real"
MODEL_READY_DATA_DIR = BASE_DIR / "data" / "model_ready"

# "synthetic" or "real"
DATA_SOURCE = "real"
CURRENT_DATA_DIR = REAL_DATA_DIR if DATA_SOURCE == "real" else SYNTHETIC_DATA_DIR

# Whether ATAC-seq is available as a second modality.
# Set to True now that locus_join.py has generated real atac_accessibility.csv
USE_ATAC = True

# Synthetic Data Constraints
# NOTE: These are SYNTHETIC generation constants only. The real matched cohort is 43
# patients (not 74) -- 74 is the ATAC-seq-only cohort size before methylation matching.
# See README.md from the data engineering handoff for the full breakdown.
# Do not use NUM_PATIENTS/NUM_CPG_SITES to validate real data shapes; real_loader.py
# infers shapes directly from the CSVs instead.
NUM_PATIENTS = 74
NUM_CPG_SITES = 5000

# Model Hyperparameters
EMBEDDING_DIM = 64
NUM_GNN_LAYERS = 2
LEARNING_RATE = 1e-3

# Graph construction parameters
PROXIMITY_THRESHOLD_KB = 50

# Class mapping for classification head
# Note: HER2 (n=1) is filtered out upstream by filter_singleton_classes()
PAM50_CLASS_MAPPING = {
    'LumA': 0,
    'LumB': 1,
    'Basal': 2,
    'Normal': 3
}
