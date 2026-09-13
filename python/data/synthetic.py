import pandas as pd
import numpy as np
import sys
import os

# Ensure python directory is in path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import SYNTHETIC_DATA_DIR, NUM_PATIENTS, NUM_CPG_SITES
from data.schema import (
    validate_methylation_data,
    validate_cpg_annotation,
    validate_pam50_labels,
    validate_atac_accessibility
)

def generate_synthetic_data(seed: int = 42):
    np.random.seed(seed)
    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Patient Barcodes (TCGA-XX-XXXX)
    patients = [f"TCGA-AB-{str(i).zfill(4)}" for i in range(1, NUM_PATIENTS + 1)]
    patient_index = pd.Index(patients, name="patient_barcode")
    
    # 2. CpG Probe IDs (cg########)
    probes = [f"cg{str(i).zfill(8)}" for i in range(1, NUM_CPG_SITES + 1)]
    probe_index = pd.Index(probes, name="probe_id")
    
    # 3. Methylation Data (beta-values in [0, 1])
    meth_values = np.random.beta(a=2, b=5, size=(NUM_PATIENTS, NUM_CPG_SITES)).astype('float32')
    df_meth = pd.DataFrame(meth_values, index=patient_index, columns=probe_index)
    df_meth.to_csv(SYNTHETIC_DATA_DIR / "methylation.csv")
    
    # 4. CpG Annotation (chromosome, position)
    chromosomes = [f"chr{c}" for c in range(1, 23)]
    cpg_chr = np.random.choice(chromosomes, size=NUM_CPG_SITES)
    cpg_pos = np.random.randint(1, 150_000_000, size=NUM_CPG_SITES).astype('int64')
    
    df_anno = pd.DataFrame({"chromosome": cpg_chr, "position": cpg_pos}, index=probe_index)
    df_anno.to_csv(SYNTHETIC_DATA_DIR / "cpg_annotation.csv")
    
    # 5. PAM50 Labels
    subtypes = ['LumA', 'LumB', 'HER2', 'Basal', 'Normal']
    patient_subtypes = np.random.choice(subtypes, size=NUM_PATIENTS, p=[0.4, 0.2, 0.15, 0.15, 0.1])
    df_pam50 = pd.DataFrame({"subtype": patient_subtypes}, index=patient_index)
    df_pam50.to_csv(SYNTHETIC_DATA_DIR / "pam50_labels.csv")
    
    # 6. ATAC-seq accessibility
    # Correlated with methylation: ATAC = 1 - meth + noise
    atac_noise = np.random.normal(0, 0.1, size=(NUM_CPG_SITES, NUM_PATIENTS))
    atac_values = (1.0 - meth_values.T) + atac_noise
    atac_values = np.clip(atac_values, 0, 1).astype('float32')
    df_atac = pd.DataFrame(atac_values, index=probe_index, columns=patient_index)
    df_atac.to_csv(SYNTHETIC_DATA_DIR / "atac_accessibility.csv")
    
    # Validation
    print("Validating generated synthetic data...")
    validate_methylation_data(df_meth)
    validate_cpg_annotation(df_anno, probe_index)
    validate_pam50_labels(df_pam50, patient_index)
    validate_atac_accessibility(df_atac, probe_index, patient_index)
    print(f"Synthetic data generation complete. Saved to {SYNTHETIC_DATA_DIR}")
    print(f"Shapes:")
    print(f"  Methylation: {df_meth.shape}")
    print(f"  CpG Annotation: {df_anno.shape}")
    print(f"  PAM50 Labels: {df_pam50.shape}")
    print(f"  ATAC-seq: {df_atac.shape}")
    
    return df_meth, df_anno, df_pam50, df_atac

if __name__ == "__main__":
    generate_synthetic_data()
