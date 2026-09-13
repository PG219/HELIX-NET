lib_path <- "C:/Users/Pranay Gupta/Documents/R/win-library/4.6"
dir.create(lib_path, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib_path, .libPaths()))

if (!requireNamespace("IlluminaHumanMethylation450kanno.ilmn12.hg19", quietly = TRUE)) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", lib = lib_path)
  BiocManager::install("IlluminaHumanMethylation450kanno.ilmn12.hg19", lib = lib_path)
}
if (!requireNamespace("minfi", quietly = TRUE)) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", lib = lib_path)
  BiocManager::install("minfi", lib = lib_path)
}

library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
library(minfi)

# Locate M1 CSV
candidate_paths <- c(
  "data/model_ready/M1_top5000_CpGs.csv",
  "M1_top5000_CpGs.csv",
  "../data/model_ready/M1_top5000_CpGs.csv"
)
m1_path <- candidate_paths[file.exists(candidate_paths)][1]
if (is.na(m1_path)) {
  stop("Could not find M1_top5000_CpGs.csv in data/model_ready/ or working directory.")
}
cat("Loading M1 matrix from:", m1_path, "\n")

# Load your M1 CSV to get the exact probe list used
m1_matrix <- read.csv(m1_path, row.names = 1, check.names = FALSE)
probe_ids <- colnames(m1_matrix)
cat(length(probe_ids), "probes to annotate\n")

# Pull the full 450K manifest and subset to just your 5,000 probes
full_annotation <- getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)
matched <- full_annotation[probe_ids, c("chr", "pos")]

failed_count <- sum(is.na(matched$chr))
cat(failed_count, "probes failed to match the manifest (should be 0)\n")

# Build in the exact schema real_loader.py expects: index=probe_id, columns=chromosome,position
cpg_annotation <- data.frame(
  chromosome = matched$chr,
  position = as.integer(matched$pos),
  row.names = probe_ids
)

# Output to root, data/model_ready, and data/processed/real
out_targets <- c(
  "cpg_annotation.csv",
  "data/model_ready/cpg_annotation.csv",
  "data/processed/real/cpg_annotation.csv"
)

for (target in out_targets) {
  dir.create(dirname(target), recursive = TRUE, showWarnings = FALSE)
  write.csv(cpg_annotation, target, row.names = TRUE)
  cat("Saved:", target, "(", nrow(cpg_annotation), "probes)\n")
}