source("scripts/00_setup.R")
results_table <- getResults(query_meth)   # re-run if not still in your session

downloaded <- list.files(paths$meth_raw, pattern = "sesame.*\\.txt$", full.names = TRUE)

# Map filename -> patient barcode using the query results you already have
file_to_barcode <- setNames(substr(results_table$cases, 1, 12), results_table$file_name)

# Read first file to establish probe order
first <- read.delim(downloaded[1], header = FALSE, col.names = c("probe", "beta"))
probes <- first$probe

beta_matrix <- matrix(NA_real_, nrow = length(probes), ncol = length(downloaded))
rownames(beta_matrix) <- probes
colnames(beta_matrix) <- file_to_barcode[basename(downloaded)]

for (i in seq_along(downloaded)) {
  df <- read.delim(downloaded[i], header = FALSE, col.names = c("probe", "beta"))
  if (!identical(df$probe, probes)) df <- df[match(probes, df$probe), ]  # align if order differs
  beta_matrix[, i] <- df$beta
  cat(i, "/", length(downloaded), "\r")
}

cat("\nBeta matrix dims:", nrow(beta_matrix), "probes x", ncol(beta_matrix), "patients\n")
cat("NAs:", round(100 * mean(is.na(beta_matrix)), 1), "%\n")

complete_probes <- rowSums(is.na(beta_matrix)) == 0
beta_complete <- beta_matrix[complete_probes, ]
cat(nrow(beta_complete), "complete probes retained\n")

probe_var <- apply(beta_complete, 1, var)
top_probes <- order(probe_var, decreasing = TRUE)[1:5000]
beta_top <- beta_complete[top_probes, ]

pca_meth <- prcomp(t(beta_top), scale. = TRUE)
var_meth <- round(100 * summary(pca_meth)$importance[2, 1:2], 1)

png("outputs/figures/meth_pca.png", width = 800, height = 600)
plot(pca_meth$x[,1], pca_meth$x[,2],
     xlab = paste0("PC1 (", var_meth[1], "% variance)"),
     ylab = paste0("PC2 (", var_meth[2], "% variance)"),
     main = "PCA of DNA Methylation (Top 5,000 Variable Probes) — 43 Matched Patients",
     pch = 19, col = "darkgreen")
dev.off()
cat("Saved meth_pca.png\n")

saveRDS(beta_matrix, file.path(paths$meth_processed, "brca_beta_matrix.rds"))

atac_features <- readRDS(file.path(paths$atac_processed, "atac_peak_feature_matrix.rds"))
matched_barcodes <- colnames(beta_top)   # your 43 methylation patients

atac_matched <- atac_features[rownames(atac_features) %in% matched_barcodes, ]
atac_matched <- atac_matched[matched_barcodes, ]   # ensure same patient order

pca_atac_matched <- prcomp(atac_matched, scale. = TRUE)

png("outputs/figures/cross_modality_comparison.png", width = 800, height = 600)
plot(pca_atac_matched$x[,1], pca_meth$x[,1],
     xlab = "ATAC-seq PC1", ylab = "Methylation PC1",
     main = "Cross-Modality Structure Comparison (43 Matched Patients)",
     pch = 19, col = "purple")
abline(lm(pca_meth$x[,1] ~ pca_atac_matched$x[,1]), col = "gray", lty = 2)
dev.off()

cat("Correlation:", round(cor(pca_atac_matched$x[,1], pca_meth$x[,1]), 3), "\n")

funnel_data <- c(
  "ATAC-seq\n(Corces et al.)" = 74,
  "+ Matched\nMethylation" = 43
)

png("outputs/figures/cohort_funnel.png", width = 700, height = 500)
bp <- barplot(funnel_data, col = c("steelblue", "darkgreen"),
              main = "Patient Cohort Construction — TCGA-BRCA",
              ylab = "Number of Patients", ylim = c(0, 80))
text(bp, funnel_data + 3, labels = funnel_data, font = 2)
dev.off()

replicate_counts <- c("1 replicate" = 8, "2 replicates" = 65, "3 replicates" = 1)

png("outputs/figures/replicate_distribution.png", width = 700, height = 500)
bp <- barplot(replicate_counts, col = "coral",
              main = "ATAC-seq Technical Replicates per Patient",
              ylab = "Number of Patients")
text(bp, replicate_counts + 2, labels = replicate_counts, font = 2)
dev.off()

library(stats)
source("scripts/00_setup.R")
feature_matrix <- readRDS(file.path(paths$atac_processed, "atac_peak_feature_matrix.rds"))
subset_patients <- feature_matrix[1:15, ]  # first 15 patients, keep it readable
cor_matrix <- cor(t(subset_patients))

png("outputs/figures/atac_patient_correlation_heatmap.png", width = 700, height = 700)
heatmap(cor_matrix, symm = TRUE, main = "ATAC-seq Patient-Patient Correlation\n(15-Patient Subset)",
        col = heat.colors(20))
dev.off()