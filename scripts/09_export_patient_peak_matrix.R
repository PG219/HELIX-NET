# 09_export_patient_peak_matrix.R — Export ATAC peak matrix to CSV for locus join

source("scripts/00_setup.R")

feature_matrix_path <- file.path(paths$atac_processed, "atac_peak_feature_matrix.rds")
if (!file.exists(feature_matrix_path)) {
  stop("Cannot find atac_peak_feature_matrix.rds")
}

cat("Loading ATAC peak feature matrix...\n")
feature_matrix <- readRDS(feature_matrix_path)
cat("Loaded feature matrix:", nrow(feature_matrix), "patients x", ncol(feature_matrix), "peaks\n")

# Load 43 matched barcodes from M1 matrix if available to also allow matched export
m1_path <- "data/model_ready/M1_top5000_CpGs.csv"
if (file.exists(m1_path)) {
  m1_matrix <- read.csv(m1_path, row.names = 1, check.names = FALSE)
  matched_barcodes <- intersect(rownames(feature_matrix), rownames(m1_matrix))
  cat("Found", length(matched_barcodes), "matched barcodes with methylation\n")
  matched_matrix <- feature_matrix[matched_barcodes, ]
  
  write.csv(matched_matrix, "data/model_ready/patient_peak_matrix_matched.csv", row.names = TRUE)
  cat("Saved data/model_ready/patient_peak_matrix_matched.csv\n")
}

# Export full matrix to CSV
out_targets <- c(
  "patient_peak_matrix.csv",
  "data/atac/processed/patient_peak_matrix.csv",
  "data/model_ready/patient_peak_matrix.csv"
)

for (target in out_targets) {
  dir.create(dirname(target), recursive = TRUE, showWarnings = FALSE)
  write.csv(feature_matrix, target, row.names = TRUE)
  cat("Saved:", target, "\n")
}
