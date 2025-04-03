library(tidyverse)
library(readr)
library(stringr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(viridis)
library(kableExtra)
library(RColorBrewer)

setwd('/Users/ashworth/Library/CloudStorage/OneDrive-NorwichBioscienceInstitutes/Jonathan_PhD_project/Rebonto/Eloise/scratch_outputs/mocks_revisited/')

report_df<- read_csv("best_mock2.csv")

head(report_df)

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
  filter(metric == "f1-score")

# Calculate summary statistics (mean, standard deviation, 95% confidence intervals)
stats <- f1_data %>%
  group_by(model) %>%
  summarise(
    mean = mean(value),
    sd = sd(value),
    #    ci_lower = mean - qt(0.975, df = n()-1) * sd(Score) / sqrt(n()),
    #   ci_upper = mean + qt(0.975, df = n()-1) * sd(Score) / sqrt(n())
    ci_upper = mean + sd(value),
    ci_lower = mean - sd(value)
  )


# Plot, reordering models by their mean F1-score:
# Refined plot with customizations for better publication appearance
ggplot(f1_data, aes(x = reorder(model, value, FUN = mean), y = value, fill = platform)) +
  geom_violin(trim = FALSE, color = "grey10",(aes(fill = platform)), width = 1, alpha = 0.8) +
  #  geom_boxplot(width = 0.1, fill = "grey98",alpha = 1) + # Subtle colors and borders for violins
  scale_fill_viridis_d(option = "plasma") +  # Color palette for violin plots
  geom_jitter(width = 0.05, alpha = 0.8, size = 1, shape = 16, aes(color = species)) +  # Smaller jitter points
  geom_point(data = stats, aes(x = model, y = mean),         # Plot means
             color = "black", size = 4, shape = 18, inherit.aes = FALSE) +  # Larger mean points
  #  geom_errorbar(data = stats, aes(x = Model, ymin = ci_lower, ymax = ci_upper),  # Confidence intervals
  #                width = 0.2, color = "black", size = 0.8, inherit.aes = FALSE) +   # Thicker error bars
  ylim(0.0, 1.0) +  # Set y-axis limits
  labs(title = "Handling unseen classes: Top performing models in handling *unseen*",
       subtitle = "F1_macro evaluation including 50% unseen data",  # Move details to a subtitle
       x = "Model", 
       y = "F1 Macro Score",
       caption = "Red points represent mean values. Blue bars indicate 95% CI.") +  # Add a caption for clarity
  theme_minimal() +
  theme(text = element_text(family = "serif"),  # Use serif font for a professional look
        plot.title = element_text(size = 14, face = "bold"),  # Bold title
        plot.subtitle = element_text(size = 12),  # Subtitle with smaller font
        axis.title = element_text(size = 12, face = "bold"),  # Bold axis titles
        axis.text = element_text(size = 10),  # Adjust axis text size
        legend.position = "none",  # Remove legend if not needed
        panel.grid.major = element_line(size = 0.2, color = "grey80"),  # Lighter gridlines
        panel.grid.minor = element_blank())  # Remove minor gridlines

# Compute mean and SD for each model
model_summary <- f1_data %>%
  group_by(model) %>%
  summarize(
    Mean_F1 = mean(value, na.rm = TRUE),
    SD_F1   = sd(value, na.rm = TRUE)
  ) %>%
  ungroup() %>%
  arrange(desc(Mean_F1))  # Sort by Mean F1 if you want highest to lowest

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


# 1. Reorder 'species' by descending mean performance:
species_order <- f1_data %>%
  group_by(species) %>%
  summarise(mean_perf = mean(value, na.rm = TRUE)) %>%
  arrange(desc(mean_perf)) %>%
  pull(species)

# If 'unseen' is not already in this data but appears in your data,
# you can manually append it to the vector so it's placed at the end:
# species_order <- c(species_order, "unseen")

# Convert species to an ordered factor
f1_data$species <- factor(f1_data$species, levels = species_order)

ggplot(f1_data, aes(x = reorder(model, value, FUN = mean), 
                    y = value)) +
  geom_violin(
    trim = FALSE,
    colour = "grey10",
    width = 1,
    alpha = 0.8
  ) +
  # Hide fill legend for 'platform'
  scale_fill_viridis_d(option = "plasma", guide = FALSE) +
  
  # 2. Colour points by species in the factor order we just created.
  #    If 'unseen' was automatically mapped to white,
  #    using a palette with enough contrast will fix that.
  #    Here we demonstrate viridis with reversed direction so the
  #    highest performers get the darkest colour, etc.
  geom_jitter(
    width = 0.05,
    alpha = 0.8,
    size = 1,
    shape = 16,
    aes(colour = species)
  ) +
  scale_colour_viridis_d(
    option = "plasma",     # Or "viridis", "magma", etc.
    direction = -1,        # Reverses the palette so top species get the darkest colour
    name = "Species"       # Legend title
  ) +
  guides(
    # Increase point size in the legend to make the colours more visible
    colour = guide_legend(
      override.aes = list(size = 3)
    )) +
  
  # Means (not mapped to colour)
  geom_point(
    data = stats,
    aes(x = model, y = mean),
    colour = "black",
    size = 4,
    shape = 18,
    inherit.aes = FALSE
  ) +
  
  labs(
    title = "Handling unseen classes: Top performing models in handling *unseen*",
    subtitle = "F1_macro evaluation including 50% unseen data",
    x = "Model",
    y = "F1 Macro Score",
    caption = "Red points represent mean values. Blue bars indicate 95% CI."
  ) +
  theme_minimal() +
  theme(
    text = element_text(family = "Times"),
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10),
    legend.position = "bottom",
    panel.grid.major = element_line(size = 0.2, colour = "grey80"),
    panel.grid.minor = element_blank()
  )

#########

# Suppose you have these species plus "unseen":
species_order <- f1_data %>%
  group_by(species) %>%
  summarise(mean_perf = mean(value, na.rm = TRUE)) %>%
  arrange(desc(mean_perf)) %>%
  pull(species)

# Ensure "unseen" is included in the levels. 
# If "unseen" is already in 'species_order', no need for this step:
if(!"unseen" %in% species_order) {
  species_order <- c(species_order, "unseen")
}

# Convert species to an ordered factor with "unseen" in the vector
f1_data$species <- factor(f1_data$species, levels = species_order)

# Create a palette for your 9 or so species. For demonstration,
# here we use a palette from RColorBrewer plus one slot for "unseen".
# In reality, you might pick colours that match the order in species_order.
my_colours <- c(
  "P_rhoeas"      = "#9E3D22",  # example brown
  "A_millefolium" = "#327EBA",  # example blue
  "L_vulgare"     = "#45A645",  # example green
  "unseen"        = "red",      # bright red so it really stands out
  "G_verum"       = "#A54FA5",  # example purple
  "R_acetosa"     = "#FF9F00",  # orange
  "M_moschata"    = "#DFAE2B",  # goldenrod
  "D_carota"      = "#7D58A6",  # violet
  "C_nigra"       = "#444444",  # grey or blackish
  "P_vulgaris"    = "#999999"   # light grey
)

ggplot(f1_data, aes(x = reorder(model, value, FUN = mean),
                    y = value)) +
  geom_violin(trim = FALSE,
              colour = "grey10",
              width = 1,
              alpha = 0.8) +
  scale_fill_viridis_d(option = "plasma", guide = FALSE) +
  
  # Jittered points, but now using the manual scale for species colour
  geom_jitter(aes(colour = species),
              width = 0.05,
              alpha = 0.8,
              size = 1.5,
              shape = 16) +
  scale_colour_manual(values = my_colours, name = "Species") +
  
  # Means (not mapped to colour)
  geom_point(data = stats,
             aes(x = model, y = mean),
             colour = "black",
             size = 4,
             shape = 18,
             inherit.aes = FALSE) +
  
  ylim(-0.1, 1.0) +
  labs(
    title = "Handling unseen classes: Top performing models in handling *unseen*",
    subtitle = "F1_macro evaluation including 50% unseen data",
    x = "Model",
    y = "F1 Macro Score",
    caption = "Red points represent mean values. Blue bars indicate 95% CI."
  ) +
  # Make sure the legend is visible; also enlarge the legend points
  guides(colour = guide_legend(override.aes = list(size = 4))) +
  theme_minimal() +
  theme(
    text = element_text(family = "Times"),
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10),
    legend.position = "bottom",
    panel.grid.major = element_line(size = 0.2, colour = "grey80"),
    panel.grid.minor = element_blank()
  )

