# Phase 2 Update — What Changed and Why

## Files changed (see accompanying `python/` folder)

1. **`config.py`** — added `USE_ATAC` flag (auto-False for real data until the peak↔probe
   mapping is resolved), corrected the misleading `NUM_PATIENTS=74` comment (real matched
   cohort is 43, not 74 — see README_HANDOFF.md).

2. **`data/real_loader.py`** — ATAC is now optional (`use_atac` param, defaults from config).
   Missing-file errors now list ALL missing files at once with guidance on who owns producing
   each one, instead of failing on the first one found with no context.

3. **`attention/cross_modal_fusion.py`** — added single-modality mode (`num_modalities=1`)
   for methylation-only operation. Also now exposes `self.last_attn_weights` after forward()
   — this closes the "attention-weight extraction not wired up" gap from the earlier code
   review (proposal §5.2.6, interpretability).

4. **`models/helix_net.py`** — passes `num_modalities` through from config automatically;
   added `get_last_attention_weights()` accessor.

5. **`graphs/build_graph.py`** — `build_patient_graphs(use_atac=...)` now builds `[N, 1]`
   node features (methylation only) or `[N, 2]` (methylation + ATAC) depending on the flag,
   instead of hard-requiring ATAC unconditionally.

6. **`training/train.py`** — fixes the review-flagged bug: previously crashed with a bare
   `FileNotFoundError` if `baselines.py` hadn't been run first (no `cv_splits.json` yet).
   Now auto-generates splits on demand with a clear log message.

7. **`evaluation/baselines.py`** — `generate_and_save_splits` now dynamically caps
   `n_splits` to the smallest class's sample count. This matters because at n=43 (real data),
   a minority PAM50 class could have as few as 3-4 patients, and `StratifiedKFold` **hard-errors**
   (not just degrades) if any class has fewer members than `n_splits=5`. Verified this exact
   failure mode with a simulated 43-sample, 3-minority-class test — confirmed it would have
   crashed without the fix, now degrades gracefully to fewer folds with a clear log line.

**All changes tested against the existing synthetic data fixtures — zero regression** on the
default (ATAC+methylation, n=74) path. New methylation-only path and attention-weight
extraction independently verified working via forward-pass tests.

## Still not resolved — needs a team decision, not just code

**ATAC peaks (215,978 genomic intervals) and real CpG probes (5,000 `cg` IDs) are not the
same feature space.** Synthetic data faked ATAC at identical positions to methylation; real
data can't do that. Three ways to resolve (unchanged from earlier discussion):
- (a) Methylation-only for now — **this is what USE_ATAC=False currently does.** Recommended
  starting point.
- (b) Map each ATAC peak to its nearest CpG probe by genomic coordinate (spatial join).
- (c) Separate node sets per modality in one graph, with cross-modal edges.

(a) is wired up and working today. (b)/(c) are real engineering work for later — worth a
team conversation before committing to one, since it affects what the model architecture
even means for the paper.

---

## For Pranay: R snippet to generate `cpg_annotation.csv` for the real 5,000 probes

`build_graph.py` needs chromosome + position per CpG to build the 50kb-proximity edges.
The M1 CSV only has probe IDs, not positions. You already have the Bioconductor environment
set up (used for TCGAbiolinks/sesame earlier) — this uses the same kind of package, just for
annotation lookup rather than data download:

```r
# Run this once, using the same R environment as your data pipeline

if (!requireNamespace("IlluminaHumanMethylation450kanno.ilmn12.hg19", quietly = TRUE)) {
  BiocManager::install("IlluminaHumanMethylation450kanno.ilmn12.hg19")
}
library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
library(minfi)

# Load your M1 CSV to get the exact probe list used
m1_matrix <- read.csv("M1_top5000_CpGs.csv", row.names = 1, check.names = FALSE)
probe_ids <- colnames(m1_matrix)
cat(length(probe_ids), "probes to annotate\n")

# Pull the full 450K manifest and subset to just your 5,000 probes
full_annotation <- getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)
matched <- full_annotation[probe_ids, c("chr", "pos")]

cat(sum(is.na(matched$chr)), "probes failed to match the manifest (should be 0)\n")

# Build in the exact schema real_loader.py expects: index=probe_id, columns=chromosome,position
cpg_annotation <- data.frame(
  chromosome = matched$chr,
  position = as.integer(matched$pos),
  row.names = probe_ids
)

write.csv(cpg_annotation, "cpg_annotation.csv", row.names = TRUE)
cat("Saved cpg_annotation.csv:", nrow(cpg_annotation), "probes\n")
```

Double-check the `sum(is.na(matched$chr))` line prints 0 before sending this over — if any
probes fail to match (shouldn't happen for standard 450K IDs, but worth the 2-second check),
those rows need investigating before this file is usable.

## For Person C: real PAM50 labels still needed

`pam50_labels.csv` (patient barcode → subtype) doesn't exist yet for the real 43-patient
cohort. This blocks `build_patient_graphs()`, `baselines.py`, and `train.py` from running
on real data at all — worth checking on this now rather than after the annotation file lands,
since both are needed simultaneously before anything real can run end to end.
