
library(pheatmap)
library(readr)
library(RColorBrewer)
library(gridExtra)
library(Cairo)

setwd("/Users/ashworth/Library/CloudStorage/OneDrive-NorwichBioscienceInstitutes/Jonathan_PhD_project/Rebonto/Eloise/scratch_outputs/BEST_mock/")

# Define filenames and titles
unthresholded <- c("confusion_matrix_RandomForest_1u.csv",
                   "confusion_matrix_XGBoost_1u.csv",
                   "confusion_matrix_GradientBoosting_1u.csv")

thresholded <- c("confusion_matrix_RandomForest_1.csv",
                 "confusion_matrix_XGBoost_1.csv",
                 "confusion_matrix_GradientBoosting_1.csv")

titles <- c("Random Forest", "XGBoost", "Gradient Boosting")

# Function to load matrices
load_matrix <- function(file_list) {
  lapply(file_list, function(f){
    cm <- read.csv(f, row.names = 1)
    as.matrix(cm)
  })
}

cm_unthresh <- load_matrix(unthresholded)
cm_thresh <- load_matrix(thresholded)

# Remove "K_arvensis" if present
remove_species <- "K_arvensis"
trim_species <- function(mat_list){
  lapply(mat_list, function(mat){
    if(remove_species %in% rownames(mat)){
      mat <- mat[rownames(mat) != remove_species, colnames(mat) != remove_species]
    }
    mat
  })
}

cm_unthresh_trimmed <- trim_species(cm_unthresh)
cm_thresh_trimmed <- trim_species(cm_thresh)

# Convert matrices to percentages, rounded
to_percent <- function(mat_list){
  lapply(mat_list, function(mat){
    round(sweep(mat, 1, rowSums(mat), FUN = "/") * 100, 0)
  })
}

perc_unthresh <- to_percent(cm_unthresh_trimmed)
perc_thresh <- to_percent(cm_thresh_trimmed)

# Use consistent color scaling across all matrices
max_perc <- max(sapply(c(perc_unthresh, perc_thresh), max))
color_palette <- colorRampPalette(brewer.pal(3, "Blues"))(100)

# Plotting function
plot_cm <- function(perc, title){
  pheatmap(
    perc,
    color = color_palette,
    breaks = seq(0, max_perc, length.out = 101),
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    display_numbers = matrix(sprintf("%.0f%%", perc), nrow=nrow(perc)),
    fontsize_number = 10,
    number_color = "black",
    fontsize = 10,
    angle_col = 45,
    legend = TRUE,
    main = title,
    silent = TRUE
  )$gtable
}

# Generate plots (unthresholded top row, thresholded bottom row)
plots <- c(
  mapply(plot_cm, perc_unthresh, paste(titles, "(Unthresholded)"), SIMPLIFY=FALSE),
  mapply(plot_cm, perc_thresh, paste(titles, "(Thresholded)"), SIMPLIFY=FALSE)
)

# Save directly to high-quality 300 DPI PNG
CairoPNG("confusion_matrices_threshold_comparison.png", width=16, height=10, units="in", dpi=300)
grid.arrange(grobs = plots, ncol = 3)
dev.off()
