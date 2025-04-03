library(pheatmap)
library(readr)  # or any other method to read CSV

setwd("/Users/ashworth/Library/CloudStorage/OneDrive-NorwichBioscienceInstitutes/Jonathan_PhD_project/Rebonto/Eloise/scratch_outputs/mocks/non-calibrated_non-thresholded/")

# 1. Read CSV with row.names = 1 so that the first column becomes row names
cm <- read.csv("confusion_matrix_RandomForest_1.csv")
rownames(cm) <- cm[[1]]
cm <- cm[,-1]

# Convert to a numeric matrix:
cm_mat <- as.matrix(cm)


# 1) Compute row sums
row_sums <- rowSums(cm_mat)

# 2) Divide each cell by its row's total, then multiply by 100
#    This means each row will sum to 100%
cm_perc <- sweep(cm_mat, 1, row_sums, FUN = "/") * 100

library(RColorBrewer)
  
pheatmap(
  cm_perc,
  color = colorRampPalette(brewer.pal(3, "Blues"))(100),
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  # Show raw counts in each cell
  display_numbers = cm_mat,  
  # Format the text as integers (no decimal places)
  number_format = "%.0f",
  main = "GB Iteration 1 \n(colour: %, text: raw counts)",
  angle_col = 45,
  fontsize = 14,
)

# Remove the row and column for "K_arvensis"
cm_mat2 <- cm_mat[rownames(cm_mat) != "K_arvensis",
                  colnames(cm_mat) != "K_arvensis"]

# Recompute row sums
row_sums2 <- rowSums(cm_mat2)

# Convert to percentages (row-wise)
cm_perc2 <- sweep(cm_mat2, 1, row_sums2, FUN = "/") * 100

# Plot using pheatmap, displaying raw counts but colouring by percentage
pheatmap(
  cm_perc2,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  display_numbers = cm_mat2,
  number_format = "%.0f",
  main = "Random Forest",
  angle_col = 45,
  fontsize = 14
)


