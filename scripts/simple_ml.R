pkgs <- c("caret", "randomForest", "glmnet", "pROC", "pheatmap", "ggplot2", "kernlab")
for (p in pkgs) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)

library(caret)
library(randomForest)
library(glmnet)
library(pROC)
library(pheatmap)
library(ggplot2)

set.seed(42)
dir.create("outputs/figures_r", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/tables_r", recursive = TRUE, showWarnings = FALSE)

X <- read.csv("C:/Users/Pranay Gupta/Desktop/helix-net/data/model_ready/M1_top5000_CpGs.csv",
              row.names = 1, check.names = FALSE)
y <- read.csv("C:/Users/Pranay Gupta/Desktop/helix-net/data/model_ready/pam50_labels.csv",
              row.names = 1)

common <- intersect(rownames(X), rownames(y))
X <- X[common, ]
y <- factor(y[common, "subtype"])

dim(X)
table(y)

keep <- y %in% names(which(table(y) >= 3))
X <- X[keep, ]
y <- droplevels(y[keep])

p1 <- ggplot(data.frame(y), aes(y, fill = y)) + geom_bar() +
  labs(title = "Samples per subtype", x = NULL, y = "Count") +
  theme_minimal() + theme(legend.position = "none")
p1

X_pca <- X[, apply(X, 2, var) > 0]
pc <- prcomp(X_pca, center = TRUE, scale. = FALSE)
ve <- round(100 * summary(pc)$importance[2, 1:2], 1)
dfp <- data.frame(PC1 = pc$x[, 1], PC2 = pc$x[, 2], Subtype = y)

p2 <- ggplot(dfp, aes(PC1, PC2, colour = Subtype)) +
  geom_point(size = 3, alpha = 0.85) +
  labs(title = "PCA of methylation (5,000 CpGs)",
       x = paste0("PC1 (", ve[1], "%)"),
       y = paste0("PC2 (", ve[2], "%)")) +
  theme_minimal()
p2
ctrl <- trainControl(method = "repeatedcv", number = 3, repeats = 3, classProbs = TRUE,
                     savePredictions = "final")

colnames(X) <- make.names(colnames(X))
levels(y) <- make.names(levels(y))
p_feat <- ncol(X)

m_rf <- train(x = X, y = y, method = "rf", trControl = ctrl, ntree = 500,
              tuneGrid = data.frame(mtry = round(sqrt(p_feat))))

m_en <- train(x = X, y = y, method = "glmnet", trControl = ctrl,
              preProcess = c("center", "scale"),
              tuneGrid = expand.grid(alpha = 0.5, lambda = 0.13))

m_sv <- train(x = X, y = y, method = "svmLinear", trControl = ctrl,
              preProcess = c("center", "scale"),
              tuneGrid = data.frame(C =  0.1))

models <- list(RandomForest = m_rf, ElasticNet = m_en, LinearSVM = m_sv)

res <- resamples(models)
summ <- summary(res)
summ

acc_df <- do.call(rbind, lapply(names(models), function(n)
  data.frame(Model = n, Accuracy = res$values[[paste0(n, "~Accuracy")]])))

p4 <- ggplot(acc_df, aes(Model, Accuracy, fill = Model)) +
  geom_boxplot(alpha = 0.6) +
  geom_jitter(width = 0.1, size = 2) +
  geom_hline(yintercept = max(table(y)) / length(y), linetype = "dashed") +
  labs(title = "Cross-validated accuracy (3-fold x 3 repeats)",
       subtitle = "Dashed line = always guess the biggest class",
       y = "Accuracy", x = NULL) +
  theme_minimal() + theme(legend.position = "none")
p4

best_name <- names(which.max(sapply(models, function(m) max(m$results$Accuracy))))
best <- models[[best_name]]

pred <- best$pred
cm <- confusionMatrix(pred$pred, pred$obs)
cm

print(cm$byClass[, c("Sensitivity", "Specificity", "Balanced Accuracy")])
write.csv(round(cm$byClass, 3), "outputs/tables_r/per_class_metrics.csv")

max_acc_df <- data.frame(
  Model = c("RandomForest", "ElasticNet", "LinearSVM"),
  Accuracy = c(max(m_rf$results$Accuracy), max(m_en$results$Accuracy), max(m_sv$results$Accuracy))
)

p5 <- ggplot(max_acc_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_col() +
  geom_text(aes(label = round(Accuracy, 3)), vjust = -0.5, fontface = "bold") +
  labs(title = "Highest Accuracy Comparison", y = "Max Accuracy") +
  theme_minimal() + theme(legend.position = "none")
p5
pred <- best$pred
classes <- levels(pred$obs)

roc_list <- lapply(classes, function(cl) roc(as.numeric(pred$obs == cl), pred[[cl]], quiet = TRUE))
names(roc_list) <- classes

auc_vals <- sapply(roc_list, auc)
auc_vals

multiclass.roc(pred$obs, pred[, classes])

roc_df <- do.call(rbind, lapply(classes, function(cl) {
  r <- roc_list[[cl]]
  data.frame(FPR = 1 - r$specificities, TPR = r$sensitivities,
             Class = paste0(cl, " (AUC = ", round(auc(r), 2), ")"))
}))

p6 <- ggplot(roc_df, aes(FPR, TPR, colour = Class)) +
  geom_line(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  labs(title = paste("One-vs-rest ROC:", best_name),
       x = "False positive rate", y = "True positive rate") +
  theme_minimal()
p6



cmt <- as.data.frame(cm$table)
cmt$Prop <- cmt$Freq / ave(cmt$Freq, cmt$Reference, FUN = sum)

p7 <- ggplot(cmt, aes(Reference, Prediction, fill = Prop)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = paste0(round(100 * Prop), "%")), size = 4) +
  scale_fill_gradient(low = "white", high = "steelblue", limits = c(0, 1)) +
  labs(title = paste("Confusion matrix:", best_name), x = "True subtype", y = "Predicted subtype", fill = "Fraction") +
  theme_minimal()
p7


imp <- varImp(m_rf)$importance
imp$score <- rowMeans(imp[, sapply(imp, is.numeric), drop = FALSE])
imp$CpG <- rownames(imp)
imp <- imp[order(-imp$score), ]
top20 <- imp[1:20, ]
top20$CpG <- factor(top20$CpG, levels = rev(top20$CpG))

p10 <- ggplot(top20, aes(score, CpG)) +
  geom_col(fill = "darkorange") +
  labs(title = "Top 20 CpGs (Random Forest importance)", x = "Importance", y = NULL) +
  theme_minimal()
p10
ggsave("outputs/figures_r/10_rf_importance.png", p10, width = 6.5, height = 5.5, dpi = 200)

write.csv(imp[1:100, c("CpG", "score")], "outputs/tables_r/top100_cpgs_rf.csv", row.names = FALSE)


top_cpgs <- imp$CpG[1:40]
annot <- data.frame(Subtype = y, row.names = rownames(X))

prob_df <- do.call(rbind, lapply(levels(pred$obs), function(cl)
  data.frame(True = pred$obs, Class = cl, Prob = pred[[cl]])))


bc <- as.data.frame(cm$byClass[, c("Sensitivity", "Specificity", "Balanced Accuracy")], check.names = FALSE)
bc$Class <- sub("Class: ", "", rownames(bc))
bc_long <- do.call(rbind, lapply(c("Sensitivity", "Specificity", "Balanced Accuracy"), function(m)
  data.frame(Class = bc$Class, Metric = m, Value = bc[[m]])))

p14 <- ggplot(bc_long, aes(Class, Value, fill = Metric)) +
  geom_col(position = "dodge") +
  geom_text(aes(label = round(Value, 2)), position = position_dodge(width = 0.9), vjust = -0.3, size = 3) +
  ylim(0, 1.05) +
  labs(title = paste("Per-class performance:", best_name), y = NULL, x = NULL) +
  theme_minimal()
p14


png("outputs/figures_r/15_resample_dotplot.png", width = 1400, height = 800, res = 200)
print(dotplot(res))

p1 <- ggplot(patient_summary, aes(x = Mean_Meth, fill = Subtype)) +
  geom_histogram(bins = 20, alpha = 0.7, position = "identity", color = "white") +
  scale_fill_manual(values = subtype_colors) +
  labs(
    title = "DNA Methylation Mean Beta Distribution by PAM50 Subtype",
    subtitle = "Cohorted by TCGA-BRCA Molecular Classification (n = 43 matched patients)",
    x = "Mean DNA Methylation Beta Value (Top 5,000 CpGs)",
    y = "Number of Patients",
    fill = "PAM50 Subtype"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    panel.grid.minor = element_blank(),
    legend.position = "right"
  )


patient_summary <- data.frame(Mean_Meth = rowMeans(X), Subtype = y)

p_mean <- ggplot(patient_summary, aes(Mean_Meth, fill = Subtype)) +
  geom_histogram(bins = 20, alpha = 0.7, position = "identity", color = "white") +
  labs(title = "Mean methylation by PAM50 subtype",
       x = "Mean beta value (top 5,000 CpGs)", y = "Number of patients") +
  theme_minimal()
p_mean


