# 00_setup.R — run this first, every session

# Use a writeable personal library instead of Program Files
lib_path <- "C:/Users/Pranay Gupta/Documents/R/win-library/4.6"
dir.create(lib_path, recursive = TRUE, showWarnings = FALSE)
.libPaths(lib_path)

options(timeout = 300)

required_packages <- c("TCGAbiolinks", "rtracklayer", "dplyr", "stringr", "GEOquery")
installed <- rownames(installed.packages())
missing <- setdiff(required_packages, installed)
if (length(missing) > 0) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
  BiocManager::install(missing, lib = lib_path)
}

library(TCGAbiolinks)
library(rtracklayer)
library(dplyr)
library(stringr)

paths <- list(
  atac_raw       = "data/atac/raw",
  atac_processed = "data/atac/processed",
  meth_raw       = "data/meth/raw",
  meth_processed = "data/meth/processed",
  chip_raw       = "data/chip/raw",
  chip_processed = "data/chip/processed"
)