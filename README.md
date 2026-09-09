# HELIX-NET: Multi-Omics Pipeline & Data Handoff

> **Multi-Omics Data Integration, QC, and Feature Preprocessing Pipeline for TCGA-BRCA**  
> *Milestone 1 (M1) Deliverable: Data Analytics & Feature Handoff*

---

## ⚠️ Critical Cohort Correction (Read Before Building Models)

> [!IMPORTANT]
> The project charter references an ATAC-seq cohort of $N \approx 74$. While that figure is correct for **ATAC-seq alone**, the **methylation-matched cohort** (from which the primary M1 CpG matrix is generated) is **$N = 43$**, not 74.
>
> - **30** of the 74 ATAC-seq patients have no Illumina 450K DNA methylation data in TCGA.
> - **1** patient had methylation only from matched normal tissue (excluded to prevent mixing tumor ATAC-seq with normal-tissue methylation).
> - **Any graph construction (PyTorch Geometric) or baseline model (XGBoost/SHAP) built on the M1 feature matrix must assume $N = 43$ patients.**

---

## 🧬 Cohort Breakdown & Summary

- **74 TCGA-BRCA patients** with ATAC-seq chromatin accessibility data (*Corces et al. 2018, Science*).
- **43 TCGA-BRCA patients** with matched Illumina 450K DNA methylation array data (primary tumor only).
- **43 patients** is the effective bi-modal fine-tuning cohort size.
- Cross-modality validation shows strong biological consistency ($r = 0.807$ between ATAC-seq PC1 and Methylation PC1).

```
   74 Patients (ATAC-seq)
        │
        ├── 30 Patients (No 450K methylation record) ───► Excluded from M1 joint matrix
        ├──  1 Patient  (Normal tissue only) ───────────► Excluded from tumor cohort
        │
        ▼
   43 Patients (Matched Bimodal Cohort: ATAC-seq + 450K Tumor Methylation)
```

---

## 📦 Key Deliverables & File Registry

### Primary M1 Deliverable (Start Here)

| File | Contents | Shape / Size | Target Consumer |
| :--- | :--- | :--- | :--- |
| `data/model_ready/M1_top5000_CpGs.csv` | **43 patients × 5,000 highest-variance CpG sites.** Rows = 12-character TCGA barcodes, columns = CpG probe IDs. Filtered for $<10\%$ missingness *before* variance ranking. | $43 \times 5,000$<br>(~2 MB) | **Person A** (GNN/PyG)<br>**Person C** (Baselines/SHAP) |

### Supporting Feature Sets & Metadata

| File | Contents | Size | Purpose |
| :--- | :--- | :--- | :--- |
| `data/atac/processed/atac_peak_feature_matrix.rds` | $74 \text{ patients} \times 215,978 \text{ peaks}$ (Normalized insertion signal). | ~130 MB | Full ATAC feature set (intersect on barcode to obtain the 43 matched patients). |
| `data/atac/processed/brca_patient_barcodes.txt` | 74 matched TCGA patient barcodes (one per line). | <1 KB | Cohort barcode indexing. |
| `data/atac/processed/bw_barcode_map.rds` | File $\rightarrow$ UUID $\rightarrow$ barcode trace map. | ~1 MB | Audit trail to raw BigWig files. |
| `data/meth/processed/brca_beta_matrix.rds` | $\approx 224,000 \text{ probes} \times 43 \text{ patients}$ methylation beta values. | ~80 MB | Full unfiltered methylation matrix. |
| `data/atac/raw/*_peakCalls.txt` | Consensus peak coordinates across **all 23 TCGA cancer types**. | ~5 MB each | Ready for pan-cancer pretraining. |
| `MANIFEST.txt` | GDC download manifest metadata. | ~1.2 KB | Genomic Data Commons data provenance. |
| `HELIX-Net_Data_Report_FINAL.docx` | Comprehensive report: cohort creation, QC, PCA plots, cross-modality metrics. | ~1 MB | Full methodology and analysis documentation. |

> [!NOTE]
> Raw BigWig files (`data/atac/raw/oak/...`, ~14 GB) are excluded from this repository. The processed `.rds` and `.csv` files contain all extracted signals. Raw links are documented in the data report if reprocessing is ever required.

---

## 💻 Data Loading Quickstart

### Python (PyTorch Geometric & Scikit-Learn Workflow)

```python
import pandas as pd

# Load primary M1 deliverable
cpg_matrix = pd.read_csv("data/model_ready/M1_top5000_CpGs.csv", index_col=0)
print(f"Loaded CpG Matrix Shape: {cpg_matrix.shape}")  # Expect (43, 5000)
print(f"Sample Barcodes: {cpg_matrix.index[:3].tolist()}")
```

### R (Full Feature Set & Cross-Modal Intersect)

```R
# Load full feature matrices
atac_features <- readRDS("data/atac/processed/atac_peak_feature_matrix.rds") # 74 x 215,978
meth_beta     <- readRDS("data/meth/processed/brca_beta_matrix.rds")        # ~224,000 x 43

# Intersect to the 43-patient bimodal cohort
matched_barcodes <- intersect(rownames(atac_features), colnames(meth_beta))
cat("Matched cohort count:", length(matched_barcodes), "\n") # 43

atac_matched <- atac_features[matched_barcodes, ]  # 43 x 215,978
meth_matched <- t(meth_beta[, matched_barcodes])   # 43 x 224,000
```

---

## 🛠️ Key Pipeline Decisions (Pre-Built Specifications)

1. **Replicate Aggregation**:
   - Technical replicates (65 patients had 2 replicates, 8 had 1, 1 had 3) were aggregated by **averaging per-base genomic coverage**, rather than arbitrary subsampling.
2. **Shared BRCA Consensus Peak Set**:
   - Standardized across $215,978$ consensus peaks (GRCh38) rather than patient-specific peak calls, ensuring aligned feature dimensions across the cohort.
3. **Missingness Thresholding vs. Naive `na.omit()`**:
   - DNA methylation arrays frequently contain sporadic probe dropouts. A naive `na.omit()` across all probes would have dropped nearly all $\approx 224,000$ probes due to the probability of single NAs across 43 samples.
   - **Implemented approach**: Probes with $\ge 10\%$ missingness were removed first, followed by residual NA cleaning and top-5,000 variance selection.
4. **Barcode Standardization**:
   - All sample IDs are truncated to standard 12-character patient-level barcodes (e.g. `TCGA-A7-A13F`) to ensure consistent join keys across multi-omics assays.

---

## 🔬 Dataset Curation Status & Open Items

### 1. Histone ChIP-seq (Cell-Line Pretraining)
- **MCF-7: ✅ Complete.**
  - Processed using ENCODE GRCh38 canonical default analysis:
    - **H3K4me3**: Fold-change signal bigWig (`ENCFF008ZOP`) + Peaks (`ENCFF145CCI`)
    - **H3K27ac**: Fold-change signal bigWig (`ENCFF353CZO`) + Peaks (`ENCFF340KSH`)
  - Files stored in `data/chip/raw/`.
- **MDA-MB-231: ❌ Verified Unavailable (Confirmed Finding).**
  - Not included in ENCODE's standard histone panel.
  - Candidate `GSM2058903` is MCF10A (incorrect cell line; lacks supplementary files).
  - Candidate `GSM2572593` is MDA-MB-231 H3K27ac **HiChIP** chromatin interaction data (loop pairs, not ChIP-seq signal) aligned to **hg19**, making both data type and genome assembly incompatible.
  - **Recommendation**:
    - *(Option A)* Proceed with **MCF-7 only** for the histone pretraining stage.
    - *(Option B)* Perform targeted GEO searches for standard (non-HiChIP) GRCh38 ChIP-seq if a second cell line is strictly required.

### 2. Pan-Cancer ATAC-seq Pretraining
- Peak coordinate definitions for **all 23 TCGA cancer types** are staged in `data/atac/raw/`.
- Raw bigWig files for non-BRCA cancer types can be retrieved via the [GDC ATAC-seq AWG Open Manifest](https://gdc.cancer.gov/files/public/file/ATACseq-AWG_Open_GDC-Manifest.txt).

### 3. High-Dimensionality & Feature Reduction
- $215,978$ ATAC-seq peaks relative to $N=43 \text{ to } 74$ patients represents high feature sparsity. Downstream graph construction should apply variance thresholding or functional promoter/enhancer filtering.

---

## 📂 Repository File Tree

```
helix-net/
├── scripts/
│   ├── 00_setup.R                    # Environment setup, package imports & directories
│   ├── 01_atac_build_barcode_map.R   # ATAC-seq patient sample barcode mapping
│   ├── 02_atac_merge_replicates.R    # Replicate aggregation and signal averaging
│   ├── 03_atac_peak_quantification.R # Consensus peak quantification
│   ├── 04_meth_download_matched.R    # TCGAbiolinks download of matched methylation data
│   ├── 05_join_qc_check.R            # Multi-omics alignment, cohort QC & PCA plots
│   ├── 06_chip_process.R             # ChIP-seq processing script
│   └── 07_export_for_python.R        # Feature selection & ML matrix export
├── outputs/
│   ├── figures/                      # Generated QC & PCA visualizations
│   │   ├── atac_pca.png
│   │   ├── meth_pca.png
│   │   ├── atac_signal_distribution.png
│   │   └── cohort_funnel.png
│   └── tables/                       # Summary statistics & patient mapping tables
├── graph.R                           # Standalone visualization script for PCA & histograms
├── MANIFEST.txt                      # GDC data manifest metadata
├── helix-net.Rproj                   # RStudio project configuration
├── .gitignore                        # Git exclusion rules
└── README.md                         # Complete project documentation & M1 handoff
```

---

## 📄 License
This project is licensed under the MIT License.
