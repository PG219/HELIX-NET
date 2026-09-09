source("scripts/00_setup.R")

bw_files <- list.files(file.path(paths$atac_raw, "oak"),
                       pattern = "\\.bw$", recursive = TRUE, full.names = TRUE)
cat("Found", length(bw_files), "bigwig files\n")

lookup <- read.delim(file.path(paths$atac_raw, "sample_id_lookup.txt"),
                     stringsAsFactors = FALSE)

lookup_unique <- lookup %>%
  distinct(stanfordUUID, Case_ID, Case_UUID)
cat(sum(duplicated(lookup_unique$stanfordUUID)), "duplicate UUIDs remain\n")

extract_uuid <- function(fname) {
  base <- basename(fname)
  parts <- str_match(base, "^[A-Z]+_([A-F0-9]{8}_[A-F0-9]{4}_[A-F0-9]{4}_[A-F0-9]{4}_[A-F0-9]{12})_")
  gsub("_", "-", parts[,2])
}

bw_df <- data.frame(
  bw_path = bw_files,
  stanfordUUID = sapply(bw_files, extract_uuid),
  stringsAsFactors = FALSE
)

bw_df <- bw_df %>%
  left_join(lookup_unique, by = "stanfordUUID", relationship = "many-to-one")

n_unmatched <- sum(is.na(bw_df$Case_ID))
cat(n_unmatched, "files failed to match a Case_ID\n")

bw_df$patient_barcode <- substr(bw_df$Case_ID, 1, 12)

replicate_counts <- bw_df %>% count(patient_barcode)
print(table(replicate_counts$n))

unique_patients <- unique(bw_df$patient_barcode)
cat(length(unique_patients), "unique BRCA patients with ATAC-seq\n")

saveRDS(bw_df, file.path(paths$atac_processed, "bw_barcode_map.rds"))
writeLines(unique_patients, file.path(paths$atac_processed, "brca_patient_barcodes.txt"))
