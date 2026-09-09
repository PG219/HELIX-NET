source("scripts/00_setup.R")

meth_beta <- readRDS(file.path(paths$meth_processed, "brca_beta_matrix.rds"))
cat("Loaded:", nrow(meth_beta), "probes x", ncol(meth_beta), "patients\n")  # confirm 43, not 74

# Filter probes by missingness threshold, NOT blanket na.omit()
missing_frac <- rowMeans(is.na(meth_beta))
keep_probes <- missing_frac < 0.10   # keep probes with <10% missing across the 43 patients
meth_filtered <- meth_beta[keep_probes, ]
cat(nrow(meth_filtered), "probes retained after missingness filter\n")

# Drop any remaining scattered NAs only after the bulk filter (should be a small number left)
meth_complete <- na.omit(meth_filtered)
cat(nrow(meth_complete), "fully complete probes remain\n")

# Top 5,000 by variance
probe_var <- apply(meth_complete, 1, var)
top5000 <- meth_complete[order(probe_var, decreasing = TRUE)[1:5000], ]

# Transpose: rows = patients, columns = features, per the operational schema
final_matrix <- t(top5000)
cat("Final matrix:", nrow(final_matrix), "patients x", ncol(final_matrix), "CpGs\n")  # should be 43 x 5000

write.csv(final_matrix, "data/model_ready/M1_top5000_CpGs.csv", row.names = TRUE)