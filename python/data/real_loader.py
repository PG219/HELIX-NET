import os
import sys
import pandas as pd
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import BASE_DIR, REAL_DATA_DIR, MODEL_READY_DATA_DIR, USE_ATAC
from data.schema import (
    validate_methylation_data,
    validate_cpg_annotation,
    validate_pam50_labels,
    validate_atac_accessibility
)

def _find_file(filename_candidates, search_dirs):
    """Searches for candidate filenames across multiple directory locations."""
    for d in search_dirs:
        for fn in filename_candidates:
            p = Path(d) / fn
            if p.exists():
                return p
    return None

def load_real_data(data_dir=None, use_atac: bool = USE_ATAC):
    """
    Reads the real handoff CSVs, casts dtypes to ensure precision,
    and validates them against schema rules.

    use_atac: if False, ATAC-seq accessibility is not required or loaded, and the
    returned df_atac will be None. This supports the methylation-only bootstrap path
    used while the real ATAC-peak-to-CpG-probe mapping is unresolved.
    """
    search_dirs = []
    if data_dir is not None:
        search_dirs.append(Path(data_dir))
    search_dirs.extend([
        REAL_DATA_DIR,
        MODEL_READY_DATA_DIR,
        BASE_DIR
    ])

    meth_path = _find_file(["methylation.csv", "M1_top5000_CpGs.csv"], search_dirs)
    cpg_path = _find_file(["cpg_annotation.csv"], search_dirs)
    pam50_path = _find_file(["pam50_labels.csv"], search_dirs)
    atac_path = _find_file(["atac_accessibility.csv"], search_dirs) if use_atac else None

    missing = []
    if not meth_path:
        missing.append("methylation.csv / M1_top5000_CpGs.csv (produced by scripts/07_export_for_python.R)")
    if not cpg_path:
        missing.append("cpg_annotation.csv (produced by cpg_annotation.R)")
    if not pam50_path:
        missing.append("pam50_labels.csv (produced by scripts/08_fetch_pam50_labels.R)")
    if use_atac and not atac_path:
        missing.append("atac_accessibility.csv (multi-modal ATAC feature matrix)")

    if missing:
        missing_list = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"Missing required real data file(s):\n  - {missing_list}\n"
            f"Search directories checked:\n  - " + "\n  - ".join(str(d) for d in search_dirs)
        )

    # Load methylation data
    print(f"Loading methylation data from: {meth_path}")
    df_meth = pd.read_csv(meth_path, index_col=0)
    df_meth.index.name = "patient_barcode"
    if df_meth.dtypes.iloc[0] not in ['float32', 'float64']:
        df_meth = df_meth.astype('float32')

    # Load CpG annotation
    print(f"Loading CpG annotation from: {cpg_path}")
    df_cpg = pd.read_csv(cpg_path, index_col=0)
    df_cpg.index.name = 'probe_id'  # Restore index name in case pd.read_csv dropped it
    df_cpg['position'] = df_cpg['position'].astype('int64')

    # Align CpG annotation index to methylation columns if needed
    if set(df_cpg.index) == set(df_meth.columns) and not df_cpg.index.equals(df_meth.columns):
        df_cpg = df_cpg.reindex(df_meth.columns)

    # Load PAM50 labels
    print(f"Loading PAM50 labels from: {pam50_path}")
    df_pam50 = pd.read_csv(pam50_path, index_col=0)
    df_pam50.index.name = "patient_barcode"

    # Align PAM50 labels index to methylation rows if needed
    if set(df_pam50.index) == set(df_meth.index) and not df_pam50.index.equals(df_meth.index):
        df_pam50 = df_pam50.reindex(df_meth.index)

    # Load ATAC-seq accessibility, if applicable
    df_atac = None
    if use_atac and atac_path:
        print(f"Loading ATAC data from: {atac_path}")
        df_atac = pd.read_csv(atac_path, index_col=0)
        if df_atac.dtypes.iloc[0] not in ['float32', 'float64']:
            df_atac = df_atac.astype('float32')

    # Run Validators
    validate_methylation_data(df_meth)
    validate_cpg_annotation(df_cpg, expected_probes=df_meth.columns)
    validate_pam50_labels(df_pam50, expected_barcodes=df_meth.index)
    if use_atac:
        validate_atac_accessibility(df_atac, expected_probes=df_meth.columns, expected_barcodes=df_meth.index)
    else:
        print("USE_ATAC=False: skipping ATAC-seq load/validation (methylation-only mode).")

    return df_meth, df_cpg, df_pam50, df_atac

if __name__ == "__main__":
    from config import SYNTHETIC_DATA_DIR
    print("Self-testing real_loader.py using synthetic CSVs...")
    df_m, df_c, df_p, df_a = load_real_data(SYNTHETIC_DATA_DIR, use_atac=True)
    print(f"Loaded successfully. Shapes: Methylation {df_m.shape}, CpG {df_c.shape}, Labels {df_p.shape}, "
          f"ATAC {df_a.shape if df_a is not None else None}")

    print("\nSelf-testing methylation-only path (use_atac=False)...")
    df_m2, df_c2, df_p2, df_a2 = load_real_data(SYNTHETIC_DATA_DIR, use_atac=False)
    print(f"Loaded successfully. Shapes: Methylation {df_m2.shape}, CpG {df_c2.shape}, Labels {df_p2.shape}, "
          f"ATAC {df_a2}")
