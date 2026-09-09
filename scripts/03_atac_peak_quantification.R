source("scripts/00_setup.R")

# 1. Load the consensus BRCA peak set (shared reference across all patients)
peaks <- read.delim(file.path(paths$atac_raw, "BRCA_peakCalls.txt"), stringsAsFactors = FALSE)
cat("Loaded", nrow(peaks), "consensus BRCA peaks\n")

# Convert to GRanges for interval operations
peak_gr <- GRanges(
  seqnames = peaks$seqnames,
  ranges = IRanges(start = peaks$start, end = peaks$end),
  peak_name = peaks$name
)

# 2. Load list of merged per-patient coverage tracks
merged_dir <- file.path(paths$atac_processed, "merged_tracks")
patient_files <- list.files(merged_dir, pattern = "\\.rds$", full.names = TRUE)
patient_ids <- gsub("\\.rds$", "", basename(patient_files))
cat("Quantifying", length(patient_files), "patients over", length(peak_gr), "peaks\n")

# 3. For each patient, extract mean coverage within each peak interval
quantify_patient <- function(rds_path, peak_gr) {
  cov <- readRDS(rds_path)
  # binnedAverage-style extraction: mean signal per peak interval
  score_vec <- tryCatch({
    v <- Views(cov, as(peak_gr, "IntegerRangesList")[names(cov)])
    unlist(viewMeans(v), use.names = FALSE)
  }, error = function(e) {
    cat("  ERROR:", conditionMessage(e), "\n")
    rep(NA_real_, length(peak_gr))
  })
  score_vec
}

# 4. Build the feature matrix: rows = patients, columns = peaks
feature_matrix <- matrix(NA_real_, nrow = length(patient_files), ncol = length(peak_gr))
rownames(feature_matrix) <- patient_ids
colnames(feature_matrix) <- peak_gr$peak_name

for (i in seq_along(patient_files)) {
  cat(sprintf("[%d/%d] %s\n", i, length(patient_files), patient_ids[i]))
  feature_matrix[i, ] <- quantify_patient(patient_files[i], peak_gr)
}

cat("\nFeature matrix dims:", nrow(feature_matrix), "x", ncol(feature_matrix), "\n")
cat("NAs in matrix:", sum(is.na(feature_matrix)), "\n")

saveRDS(feature_matrix, file.path(paths$atac_processed, "atac_peak_feature_matrix.rds"))