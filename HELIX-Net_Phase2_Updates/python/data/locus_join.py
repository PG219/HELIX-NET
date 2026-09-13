"""
locus_join.py — Resolves the ATAC-seq peak / methylation CpG probe feature-space mismatch
by mapping each CpG probe to its nearest ATAC peak (by genomic distance), within a
configurable threshold. Produces atac_accessibility.csv in the exact schema real_loader.py
expects (index=probe_id matching methylation columns, columns=patient barcodes),
so USE_ATAC=True in config.py can be enabled seamlessly.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add python dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import BASE_DIR, REAL_DATA_DIR, MODEL_READY_DATA_DIR

def _find_candidate_file(candidates):
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None

def build_atac_accessibility(
    cpg_annotation_path: str = None,
    peak_coordinates_path: str = None,
    patient_peak_matrix_path: str = None,
    output_path: str = None,
    mapping_audit_path: str = None,
    max_distance_bp: int = 2000,
    peak_coords_sep: str = "\t",
):
    """
    Maps each CpG probe in cpg_annotation to the nearest ATAC peak within max_distance_bp.
    Zero-fills probes with no ATAC peak within threshold.
    """
    # 1. Resolve CpG Annotation
    if cpg_annotation_path is None:
        cpg_annotation_path = _find_candidate_file([
            MODEL_READY_DATA_DIR / "cpg_annotation.csv",
            REAL_DATA_DIR / "cpg_annotation.csv",
            BASE_DIR / "cpg_annotation.csv"
        ])
    if not cpg_annotation_path or not Path(cpg_annotation_path).exists():
        raise FileNotFoundError(f"Cannot find cpg_annotation.csv at {cpg_annotation_path}")

    print(f"Loading CpG annotations from: {cpg_annotation_path}")
    cpg_anno = pd.read_csv(cpg_annotation_path, index_col=0)
    cpg_anno.index.name = "probe_id"

    # 2. Resolve Peak Coordinates
    if peak_coordinates_path is None:
        peak_coordinates_path = _find_candidate_file([
            BASE_DIR / "data" / "atac" / "raw" / "BRCA_peakCalls.txt",
            BASE_DIR / "peak_coordinates.csv",
            MODEL_READY_DATA_DIR / "peak_coordinates.csv"
        ])
    if not peak_coordinates_path or not Path(peak_coordinates_path).exists():
        raise FileNotFoundError(f"Cannot find peak coordinates file at {peak_coordinates_path}")

    print(f"Loading ATAC peak coordinates from: {peak_coordinates_path}")
    if str(peak_coordinates_path).endswith(".csv") and peak_coords_sep == "\t":
        peak_coords_sep = ","
    peaks = pd.read_csv(peak_coordinates_path, sep=peak_coords_sep)
    
    required_peak_cols = {"seqnames", "start", "end", "name"}
    missing_cols = required_peak_cols - set(peaks.columns)
    if missing_cols:
        raise ValueError(f"peak_coordinates file missing expected columns: {missing_cols}. Found: {list(peaks.columns)}")
    peaks["midpoint"] = (peaks["start"] + peaks["end"]) // 2

    # 3. Resolve Patient Peak Matrix
    if patient_peak_matrix_path is None:
        patient_peak_matrix_path = _find_candidate_file([
            MODEL_READY_DATA_DIR / "patient_peak_matrix_matched.csv",
            MODEL_READY_DATA_DIR / "patient_peak_matrix.csv",
            BASE_DIR / "data" / "atac" / "processed" / "patient_peak_matrix.csv",
            BASE_DIR / "patient_peak_matrix.csv"
        ])
    if not patient_peak_matrix_path or not Path(patient_peak_matrix_path).exists():
        raise FileNotFoundError(f"Cannot find patient peak matrix CSV at {patient_peak_matrix_path}")

    print(f"Loading patient peak matrix from: {patient_peak_matrix_path}")
    patient_matrix = pd.read_csv(patient_peak_matrix_path, index_col=0)

    # 4. Resolve Matched Patient Barcodes (align with methylation matrix)
    meth_candidates = [
        MODEL_READY_DATA_DIR / "M1_top5000_CpGs.csv",
        REAL_DATA_DIR / "methylation.csv",
        BASE_DIR / "methylation.csv"
    ]
    meth_path = _find_candidate_file(meth_candidates)
    if meth_path:
        df_meth_ref = pd.read_csv(meth_path, index_col=0)
        target_patient_barcodes = df_meth_ref.index
        matched_barcodes = [b for b in target_patient_barcodes if b in patient_matrix.index]
        patient_matrix = patient_matrix.loc[matched_barcodes]
        print(f"Aligned patient matrix to {len(matched_barcodes)} matched cohort barcodes.")
    else:
        target_patient_barcodes = patient_matrix.index

    # 5. Spatial Join: Nearest ATAC Peak within max_distance_bp
    peaks_by_chrom = {chrom: g.sort_values("midpoint") for chrom, g in peaks.groupby("seqnames")}

    mapping_rows = []
    matched_count = 0

    print(f"Performing nearest-peak spatial join (threshold: {max_distance_bp} bp)...")
    for probe_id, row in cpg_anno.iterrows():
        chrom, pos = row["chromosome"], row["position"]

        if chrom not in peaks_by_chrom:
            mapping_rows.append([probe_id, None, np.nan, False])
            continue

        chrom_peaks = peaks_by_chrom[chrom]
        midpoints = chrom_peaks["midpoint"].values
        idx = np.searchsorted(midpoints, pos)
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(midpoints):
            candidates.append(idx)

        best_dist = None
        best_peak = None
        for c in candidates:
            dist = abs(midpoints[c] - pos)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_peak = chrom_peaks.iloc[c]["name"]

        within_threshold = best_dist is not None and best_dist <= max_distance_bp
        if within_threshold:
            matched_count += 1
        mapping_rows.append([probe_id, best_peak if within_threshold else None,
                              best_dist, within_threshold])

    mapping_df = pd.DataFrame(mapping_rows, columns=["probe_id", "matched_peak", "distance_bp", "within_threshold"])
    mapping_df.set_index("probe_id", inplace=True)

    coverage_pct = 100 * matched_count / len(cpg_anno)
    print(f"Spatial join completed: Matched {matched_count}/{len(cpg_anno)} probes ({coverage_pct:.1f}%) within {max_distance_bp}bp.")

    # 6. Construct ATAC Accessibility Table [Probes x Patients]
    patient_barcodes = target_patient_barcodes
    atac_accessibility = pd.DataFrame(0.0, index=cpg_anno.index, columns=patient_barcodes, dtype="float32")

    matched_sub_df = mapping_df[mapping_df["within_threshold"]]
    for probe_id, m_row in matched_sub_df.iterrows():
        peak_name = m_row["matched_peak"]
        if peak_name in patient_matrix.columns:
            atac_accessibility.loc[probe_id, :] = patient_matrix.loc[patient_barcodes, peak_name].values.astype("float32")

    atac_accessibility.index.name = "probe_id"

    # 7. Save Outputs to Target Locations
    out_targets = [
        REAL_DATA_DIR / "atac_accessibility.csv",
        MODEL_READY_DATA_DIR / "atac_accessibility.csv",
        BASE_DIR / "atac_accessibility.csv"
    ]
    if output_path:
        out_targets.insert(0, Path(output_path))

    for target in out_targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        atac_accessibility.to_csv(target)
        print(f"Saved ATAC accessibility matrix to: {target}")

    audit_targets = [
        BASE_DIR / "outputs" / "tables" / "locus_join_mapping.csv",
        BASE_DIR / "locus_join_mapping.csv"
    ]
    if mapping_audit_path:
        audit_targets.insert(0, Path(mapping_audit_path))

    for target in audit_targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        mapping_df.to_csv(target)
        print(f"Saved locus join audit trail to: {target}")

    print(f"\nSummary:")
    print(f"  Matrix Shape: {atac_accessibility.shape[0]} probes x {atac_accessibility.shape[1]} patients")
    print(f"  Real ATAC Signals: {matched_count} probes | Zero-filled: {len(cpg_anno) - matched_count} probes")

    return atac_accessibility, mapping_df

if __name__ == "__main__":
    build_atac_accessibility(max_distance_bp=2000)
