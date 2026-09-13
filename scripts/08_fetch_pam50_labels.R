# 08_fetch_pam50_labels.R — Extract PAM50 subtype labels for the 43 matched cohort

source("scripts/00_setup.R")

# 1. Load patient barcodes from the M1 matrix
candidate_paths <- c(
  "data/model_ready/M1_top5000_CpGs.csv",
  "M1_top5000_CpGs.csv"
)
m1_path <- candidate_paths[file.exists(candidate_paths)][1]
if (is.na(m1_path)) {
  stop("Could not find M1_top5000_CpGs.csv in data/model_ready/ or working directory.")
}

m1_matrix <- read.csv(m1_path, row.names = 1, check.names = FALSE)
barcodes <- rownames(m1_matrix)
cat("Found", length(barcodes), "matched patient barcodes in M1 matrix\n")

# 2. Query TCGA BRCA molecular subtypes via TCGAbiolinks
cat("Querying TCGA-BRCA PanCancer subtype data...\n")
subtypes_df <- TCGAbiolinks::TCGAquery_subtype(tumor = "BRCA")

# Normalize barcode formats to 12-char (e.g. TCGA-A7-A13F)
subtypes_df$patient_barcode <- substr(subtypes_df$patient, 1, 12)

# Match barcodes
matched_subtypes <- subtypes_df[match(barcodes, subtypes_df$patient_barcode), ]

# Subtype column name in TCGAquery_subtype BRCA is 'PAM50' or 'Subtype_mRNA'
pam50_raw <- if ("PAM50" %in% colnames(matched_subtypes)) {
  matched_subtypes$PAM50
} else if ("Subtype_mRNA" %in% colnames(matched_subtypes)) {
  matched_subtypes$Subtype_mRNA
} else {
  matched_subtypes[[grep("PAM50|Subtype|mRNA", colnames(matched_subtypes), ignore.case = TRUE)[1]]]
}

# Standardize subtype labels: LumA, LumB, HER2, Basal, Normal
pam50_standardized <- as.character(pam50_raw)
pam50_standardized[pam50_standardized %in% c("Her2", "HER2", "Her2-enriched")] <- "HER2"
pam50_standardized[pam50_standardized %in% c("Basal", "Basal-like")] <- "Basal"
pam50_standardized[pam50_standardized %in% c("LumA", "Luminal A")] <- "LumA"
pam50_standardized[pam50_standardized %in% c("LumB", "Luminal B")] <- "LumB"
pam50_standardized[pam50_standardized %in% c("Normal", "Normal-like")] <- "Normal"

# Fallback for any unmatched/NA labels (if any patient has missing subtype in PanCancer table)
if (any(is.na(pam50_standardized))) {
  cat("Warning:", sum(is.na(pam50_standardized)), "patients had NA subtype in PanCancer table. Inspecting...\n")
  # Fill missing with most prevalent subtype or report
}

df_pam50 <- data.frame(
  subtype = pam50_standardized,
  row.names = barcodes
)

# Output destinations
out_targets <- c(
  "pam50_labels.csv",
  "data/model_ready/pam50_labels.csv",
  "data/processed/real/pam50_labels.csv"
)

for (target in out_targets) {
  dir.create(dirname(target), recursive = TRUE, showWarnings = FALSE)
  write.csv(df_pam50, target, row.names = TRUE)
  cat("Saved PAM50 labels to:", target, "\n")
}

cat("PAM50 Class Distribution for the matched cohort:\n")
print(table(df_pam50$subtype, useNA = "ifany"))
