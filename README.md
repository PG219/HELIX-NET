# HELIX-NET

> Multi-Omics Data Integration and Preprocessing Pipeline for TCGA-BRCA

HELIX-NET is a computational pipeline designed to process, harmonize, quality-control, and integrate multi-omics datasets (ATAC-seq chromatin accessibility, DNA methylation array, and ChIP-seq) from the TCGA Breast Invasive Carcinoma (TCGA-BRCA) cohort into downstream machine learning-ready feature matrices.

---

## 🧬 Overview & Pipeline Architecture

The workflow automates the multi-step curation and cross-modal patient alignment:

1. **ATAC-seq Processing**:
   - Extraction of patient-barcode mapping across replicate samples.
   - Merging biological and technical replicates.
   - Signal quantification across consensus accessibility peak sets.
2. **DNA Methylation Processing**:
   - Querying and downloading matched Illumina HumanMethylation450/EPIC level-3 beta values via `TCGAbiolinks`.
   - Probe missingness filtering (<10% threshold) and top variance feature selection.
3. **Cross-Omics Harmonization & QC**:
   - Identifying matched patient subsets across modalities.
   - Dimensionality reduction (PCA) and signal distribution checks.
4. **Machine Learning Feature Export**:
   - Formatting transposed feature matrices ($N \times P$) for Python/PyTorch model training.

---

## 📁 Repository Structure

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
│   └── tables/                       # Summary statistics & patient mapping tables
├── graph.R                           # Standalone visualization script for PCA & histograms
├── MANIFEST.txt                      # GDC data manifest metadata
├── helix-net.Rproj                   # RStudio project configuration
├── .gitignore                        # Git exclusion rules
└── README.md                         # Project documentation
```

---

## ⚙️ Prerequisites & Setup

### Requirements
- **R** >= 4.2 (tested on R 4.6.x)
- **Bioconductor Packages**:
  - `TCGAbiolinks`
  - `rtracklayer`
  - `GEOquery`
- **CRAN Packages**:
  - `dplyr`, `stringr`, `ggplot2`

### Installation
Run `scripts/00_setup.R` to initialize directory paths and install any missing Bioconductor / CRAN dependencies:

```R
source("scripts/00_setup.R")
```

---

## 🚀 Execution Workflow

Execute the pipeline in sequential order:

```R
# Step 1: Build ATAC-seq Barcode Map
source("scripts/01_atac_build_barcode_map.R")

# Step 2: Merge ATAC Replicates
source("scripts/02_atac_merge_replicates.R")

# Step 3: Quantify Peaks
source("scripts/03_atac_peak_quantification.R")

# Step 4: Download Matched TCGA Methylation Data
source("scripts/04_meth_download_matched.R")

# Step 5: Cohort Alignment & QC Analysis
source("scripts/05_join_qc_check.R")

# Step 6: ChIP-seq Processing (if applicable)
source("scripts/06_chip_process.R")

# Step 7: Export Model-Ready Matrix for Python
source("scripts/07_export_for_python.R")
```

---

## 📊 Output Artifacts

- **Figures** (`outputs/figures/`):
  - `atac_pca.png`: PCA of chromatin accessibility across patient cohort.
  - `meth_pca.png`: PCA of top 5,000 variable methylation probes.
  - `atac_signal_distribution.png`: Normalized peak signal distributions.
  - `cohort_funnel.png`: Patient overlap breakdown between modalities.
- **Model-Ready Data** (`data/model_ready/`):
  - `M1_top5000_CpGs.csv`: Filtered $N \times 5000$ patient methylation matrix.

---

## 📄 License
This project is open-source under the MIT License.
