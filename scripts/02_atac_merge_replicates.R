# 02_atac_merge_replicates.R
source("scripts/00_setup.R")

bw_df <- readRDS(file.path(paths$atac_processed, "bw_barcode_map.rds"))

merge_patient_tracks <- function(patient_id, bw_df) {
  files <- bw_df$bw_path[bw_df$patient_barcode == patient_id]
  tracks <- lapply(files, import, format = "BigWig")
  covs <- lapply(tracks, function(t) coverage(t, weight = "score"))
  
  if (length(covs) == 1) return(covs[[1]])
  
  chrom_ok  <- all(sapply(covs[-1], function(c) identical(names(c), names(covs[[1]]))))
  length_ok <- all(sapply(covs[-1], function(c) all(lengths(c) == lengths(covs[[1]]))))
  
  if (!chrom_ok || !length_ok) {
    warning(paste("Skipping", patient_id, "- mismatched chrom names/lengths across replicates"))
    return(NULL)
  }
  
  Reduce(`+`, covs) / length(covs)
}

# Output directory for per-patient merged tracks
out_dir <- file.path(paths$atac_processed, "merged_tracks")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

all_patients <- unique(bw_df$patient_barcode)
cat("Processing", length(all_patients), "patients\n")

failed_patients <- c()

for (i in seq_along(all_patients)) {
  pid <- all_patients[i]
  out_file <- file.path(out_dir, paste0(pid, ".rds"))
  
  if (file.exists(out_file)) {
    cat(sprintf("[%d/%d] %s - already done, skipping\n", i, length(all_patients), pid))
    next
  }
  
  cat(sprintf("[%d/%d] %s - processing...\n", i, length(all_patients), pid))
  
  result <- tryCatch(
    merge_patient_tracks(pid, bw_df),
    error = function(e) {
      cat("  ERROR:", conditionMessage(e), "\n")
      NULL
    }
  )
  
  if (is.null(result)) {
    failed_patients <- c(failed_patients, pid)
    cat("  FAILED -", pid, "\n")
  } else {
    saveRDS(result, out_file)
    cat("  saved\n")
  }
  
  # Force garbage collection to free RAM after each patient
  rm(result)
  gc(verbose = FALSE)
}

cat("\nDone.", length(all_patients) - length(failed_patients), "succeeded,",
    length(failed_patients), "failed\n")

if (length(failed_patients) > 0) {
  writeLines(failed_patients, file.path(paths$atac_processed, "failed_patients.txt"))
  cat("Failed patient list saved to failed_patients.txt\n")
}