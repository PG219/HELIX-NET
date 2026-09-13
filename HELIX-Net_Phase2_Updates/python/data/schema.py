import pandas as pd
from dataclasses import dataclass
import re
import numpy as np

@dataclass
class DataSchema:
    """
    Schema constants and definitions.
    """
    num_patients: int = 74
    num_cpg_sites: int = 5000

def validate_patient_barcodes(index: pd.Index):
    """Ensure index contains valid TCGA patient barcodes (e.g., TCGA-XX-XXXX)."""
    assert index.is_unique, "Patient barcodes must be unique."
    pattern = re.compile(r"^TCGA-[A-Za-z0-9]{2}-[a-zA-Z0-9]{4}$")
    for barcode in index:
        if not pattern.match(str(barcode)):
            raise ValueError(f"Invalid patient barcode format: {barcode}")

def validate_cpg_probe_ids(columns: pd.Index):
    """Ensure columns are valid CpG probe IDs (e.g., cg########)."""
    assert columns.is_unique, "CpG probe IDs must be unique."
    pattern = re.compile(r"^cg\d{8}$")
    for probe in columns:
        if not pattern.match(str(probe)):
            raise ValueError(f"Invalid CpG probe ID format: {probe}")

def validate_methylation_data(df: pd.DataFrame):
    """Validate the handoff methylation matrix."""
    validate_patient_barcodes(df.index)
    validate_cpg_probe_ids(df.columns)
    
    if df.isna().any().any():
        row, col = np.where(df.isna())
        raise ValueError(f"Methylation matrix contains missing values. First failure at patient {df.index[row[0]]}, probe {df.columns[col[0]]}.")
    
    if not ((df.dtypes == 'float64').all() or (df.dtypes == 'float32').all()):
        bad_cols = df.dtypes[~df.dtypes.isin(['float64', 'float32'])].index.tolist()
        raise ValueError(f"Methylation values must be float. First failure column: {bad_cols[0]}")
        
    if not ((df >= 0.0).all().all() and (df <= 1.0).all().all()):
        bad_mask = (df < 0.0) | (df > 1.0)
        row, col = np.where(bad_mask)
        val = df.iat[row[0], col[0]]
        raise ValueError(f"Methylation beta-values must be in [0, 1]. Failure at patient {df.index[row[0]]}, probe {df.columns[col[0]]} = {val}")
        
    print("Methylation data schema validation passed.")

def validate_cpg_annotation(df: pd.DataFrame, expected_probes: pd.Index):
    """Validate CpG annotation table: probe_id -> {chromosome, position}."""
    if not df.index.equals(expected_probes):
        if set(df.index) == set(expected_probes):
            raise ValueError("CpG annotation probe IDs match expected probes but order differs. Reindex before validation.")
        raise ValueError("CpG annotation index must match methylation probe IDs.")
    if 'chromosome' not in df.columns or 'position' not in df.columns:
        raise ValueError(f"CpG annotation missing 'chromosome' or 'position' columns. Found: {list(df.columns)}")
    if not pd.api.types.is_numeric_dtype(df['position']):
        raise ValueError(f"'position' must be numeric, found dtype {df['position'].dtype}.")
    print("CpG annotation schema validation passed.")

def validate_pam50_labels(df: pd.DataFrame, expected_barcodes: pd.Index):
    """Validate PAM50 label table."""
    if not df.index.equals(expected_barcodes):
        if set(df.index) == set(expected_barcodes):
            raise ValueError("PAM50 labels match expected barcodes but order differs. Reindex before validation.")
        raise ValueError("PAM50 labels index must exactly match methylation patient barcodes.")
    if 'subtype' not in df.columns:
        raise ValueError("PAM50 labels missing 'subtype' column.")
    
    valid_subtypes = {'LumA', 'LumB', 'HER2', 'Basal', 'Normal'}
    invalid_mask = ~df['subtype'].isin(valid_subtypes)
    if invalid_mask.any():
        bad_idx = df[invalid_mask].index[0]
        bad_val = df.loc[bad_idx, 'subtype']
        raise ValueError(f"Invalid PAM50 subtype found at patient {bad_idx}: {bad_val}")
    print("PAM50 labels schema validation passed.")

def validate_atac_accessibility(df: pd.DataFrame, expected_probes: pd.Index, expected_barcodes: pd.Index):
    """Validate ATAC-seq accessibility table."""
    if not df.index.equals(expected_probes):
        raise ValueError("ATAC-seq index must match probe IDs.")
    if not df.columns.equals(expected_barcodes):
        raise ValueError("ATAC-seq columns must match patient barcodes.")
        
    if df.isna().any().any():
        row, col = np.where(df.isna())
        raise ValueError(f"ATAC-seq table contains missing values. First failure at probe {df.index[row[0]]}, patient {df.columns[col[0]]}.")
        
    if not ((df.dtypes == 'float64').all() or (df.dtypes == 'float32').all()):
        bad_cols = df.dtypes[~df.dtypes.isin(['float64', 'float32'])].index.tolist()
        raise ValueError(f"ATAC-seq values must be float. First failure column: {bad_cols[0]}")
    print("ATAC-seq accessibility schema validation passed.")
