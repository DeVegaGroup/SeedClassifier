library(tidyverse)
library(readr)
library(stringr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(viridis)
library(kableExtra)


setwd('/Users/ashworth/Library/CloudStorage/OneDrive-NorwichBioscienceInstitutes/Jonathan_PhD_project/Rebonto/Eloise/scratch_outputs/BEST_evaluate/')

report_df<- read_csv("best_evaluation.csv")

report_filtered <- report_df %>%
  filter(!(species == "K_arvensis") &
           !(species == "accuracy") &
           !(species =="macro avg") &
           !(species == "weighted avg")) 
head(report_filtered)

# Convert necessary columns to numeric
report_filtered$`f1-score` <- as.numeric(report_filtered$`f1-score`)

# Reshape the data to long format
report_long <- report_filtered %>%
  gather(key = "metric", value = "value", precision, recall, `f1-score`)

# Filter for the 'f1-score' metric
f1_data <- report_long %>%
  filter((metric == "f1-score") & 
          !(model == "knn_dist") &
          !(model == "knn_unif")
)
# Create the violin plot
ggplot(f1_data, aes(x = model, y = value)) +
  geom_violin(trim = FALSE, fill = "skyblue", color = "black") +
  stat_summary(fun = "median", geom = "point", shape = 23, size = 2, fill = "red") +
  labs(title = "Distribution of F1-Scores Across Models",
       x = "Model",
       y = "F1-Score") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Plot, reordering models by their mean F1-score:
ggplot(f1_data, aes(x = reorder(model, value, FUN = mean), y = value, fill = platform)) +
  geom_violin(trim = FALSE, color = "grey10", width = 0.4, alpha = 0.8) +
  stat_summary(fun = "mean", geom = "point", shape = 23, size = 1, fill = "black") +
  ylim(0.25, 1.0) +
  labs(
    title = "Evaluation: top performing models across platforms",
    subtitle = "F1_macro score performed on evaluation set",
    x = "Model", 
    y = "F1 Macro Score",
    caption = "Red points represent mean values. Blue bars indicate 95% CI."
  ) +
  theme_minimal() +
  theme(
    text = element_text(family = "serif"),
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10),
    legend.position = "none",
    panel.grid.major = element_line(size = 0.2, color = "grey80"),
    panel.grid.minor = element_blank()
  )


ggplot(f1_data, aes(x = reorder(model, value, FUN = mean), y = value, fill = platform, group = interaction(model, platform))) +
  geom_violin(trim = FALSE, color = "grey10", width = 0.4, alpha = 0.8, position = position_dodge(width = 0.5)) +
  stat_summary(fun = "mean", geom = "point", shape = 23, size = 1.5, fill = "black", position = position_dodge(width = 0.5)) +
  ylim(0.25, 1.0) +
  labs(
    title = "Evaluation: top performing models across platforms",
    subtitle = "F1_macro score performed on evaluation set",
    x = "Model",
    y = "F1 Macro Score",
    caption = "Black points represent mean values for each model-platform combination."
  ) +
  theme_minimal() +
  theme(
    text = element_text(family = "serif"),
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10),
    legend.position = "none",
    panel.grid.major = element_line(size = 0.2, color = "grey80"),
    panel.grid.minor = element_blank()
  )

# Print the table using kableExtra for nicer formatting
model_summary %>%
  kbl(digits = 3, caption = "Summary of Mean and SD for F1-Score by Model") %>%
  kable_styling(full_width = FALSE, bootstrap_options = c("striped", "hover"))

summary_table <- report_filtered %>%
  group_by(platform, `full name`, model) %>%
  summarize(
    mean_precision = mean(precision, na.rm = TRUE),
    sd_precision   = sd(precision, na.rm = TRUE),
    mean_recall    = mean(recall, na.rm = TRUE),
    sd_recall      = sd(recall, na.rm = TRUE),
    mean_f1        = mean(`f1-score`, na.rm = TRUE),
    sd_f1          = sd(`f1-score`, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(platform, `full name`, model)

# Print with kableExtra
summary_table %>%
  kbl(
    caption = "Mean and SD of Precision, Recall, and F1-Score grouped by Platform, Full Name, and Model",
    digits = 3
  ) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover"),
    full_width = FALSE
  )
print(summary_table)

write_csv(summary_table,"evaluate_summary.csv")

write
