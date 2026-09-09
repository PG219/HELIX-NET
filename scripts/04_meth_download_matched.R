source("scripts/00_setup.R")

# 1. Load the ATAC-matched patient barcode list from Step 1
atac_barcodes <- readLines(file.path(paths$atac_processed, "brca_patient_barcodes.txt"))
cat("Querying methylation for", length(atac_barcodes), "ATAC-matched patients\n")

# 2. Query GDC for 450K methylation, restricted to these barcodes
query_meth <- GDCquery(
  project = "TCGA-BRCA",
  data.category = "DNA Methylation",
  platform = "Illumina Human Methylation 450",
  data.type = "Methylation Beta Value",
  sample.type = "Primary Tumor",
  barcode = atac_barcodes
)

# 3. Check what actually matched BEFORE downloading anything
results_table <- getResults(query_meth)
matched_barcodes <- unique(substr(results_table$cases, 1, 12))
cat(length(matched_barcodes), "of", length(atac_barcodes),
    "ATAC patients have matched 450K methylation data\n")

missing_barcodes <- setdiff(atac_barcodes, matched_barcodes)
if (length(missing_barcodes) > 0) {
  cat(length(missing_barcodes), "ATAC patients have NO methylation match:\n")
  print(missing_barcodes)
  writeLines(missing_barcodes, file.path(paths$meth_processed, "atac_patients_missing_methylation.txt"))
}

# 1. Clean up again
unlink(file.path(paths$meth_raw, "*"), recursive = TRUE)
dir.create(paths$meth_raw, showWarnings = FALSE)

# 2. Build a download list file for aria2c (url + destination pairs)
results_table <- getResults(query_meth)

download_list <- file.path(paths$meth_raw, "download_list.txt")
con <- file(download_list, "w")
for (i in seq_len(nrow(results_table))) {
  url <- paste0("https://api.gdc.cancer.gov/data/", results_table$id[i])
  writeLines(url, con)
  writeLines(paste0("  dir=", paths$meth_raw), con)
  writeLines(paste0("  out=", results_table$file_name[i]), con)
}
close(con)

cat("Wrote", nrow(results_table), "download entries to", download_list, "\n")