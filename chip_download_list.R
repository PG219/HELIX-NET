source("scripts/00_setup.R")
dir.create(paths$chip_raw, showWarnings = FALSE, recursive = TRUE)

# 1. Skip downloading the JSONs entirely! You already have the exact accessions.
files_to_get <- data.frame(
  accession = c("ENCFF008ZOP", "ENCFF145CCI", "ENCFF353CZO", "ENCFF340KSH"),
  filename  = c("MCF7_H3K4me3_signal.bigWig", "MCF7_H3K4me3_peaks.bed.gz", 
                "MCF7_H3K27ac_signal.bigWig", "MCF7_H3K27ac_peaks.bed.gz")
)

# 2. Vectorize the URL string creation (no slow loops)
urls <- sprintf(
  "https://www.encodeproject.org/files/%s/@@download/%s%s", 
  files_to_get$accession, 
  files_to_get$accession, 
  ifelse(grepl("bigWig", files_to_get$filename), ".bigWig", ".bed.gz")
)

# 3. Interleave the URLs, dirs, and output names into one text vector instantly
aria_lines <- c(rbind(
  urls,
  paste0("  dir=", paths$chip_raw),
  paste0("  out=", files_to_get$filename)
))

# Write the entire file in a single pass
download_list <- file.path(paths$chip_raw, "chip_download_list.txt")
writeLines(aria_lines, download_list)

# 4. Execute with maxed-out connection parameters (-x 16 -s 16 -j 4)
cmd <- sprintf('aria2c -c -x 16 -s 16 -j 4 --retry-wait=5 --max-tries=20 -i "%s"', download_list)
system(cmd)