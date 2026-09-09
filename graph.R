# In the NEW session — this is a separate R process, won't conflict
source("scripts/00_setup.R")
feature_matrix <- readRDS(file.path(paths$atac_processed, "atac_peak_feature_matrix.rds"))

png("outputs/figures/atac_signal_distribution.png", width = 800, height = 600)
hist(feature_matrix[1, ], breaks = 50,
     main = "ATAC-seq Peak Signal Distribution (Representative Patient)",
     xlab = "Normalized Insertion Signal", ylab = "Number of Peaks",
     col = "steelblue", border = "white")
dev.off()

pca <- prcomp(feature_matrix, scale. = TRUE)
var_explained <- round(100 * summary(pca)$importance[2, 1:2], 1)

png("outputs/figures/atac_pca.png", width = 800, height = 600)
plot(pca$x[,1], pca$x[,2],
     xlab = paste0("PC1 (", var_explained[1], "% variance)"),
     ylab = paste0("PC2 (", var_explained[2], "% variance)"),
     main = "PCA of ATAC-seq Chromatin Accessibility — 74 TCGA-BRCA Patients",
     pch = 19, col = "darkred")
dev.off()

cat("Saved. Variance explained by PC1+PC2:", sum(var_explained), "%\n")


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