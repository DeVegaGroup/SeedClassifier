library(tidyverse)
library(readr)
library(stringr)
library(dplyr)
library(RColorBrewer)


c200<- read_csv("~/Documents/R/comparing_providers/n200_classification_report.csv")
c400<- read_csv("~/Documents/R/comparing_providers/n400_classification_report.csv")
c600<- read_csv("~/Documents/R/comparing_providers/n600_classification_report.csv")
c800<- read_csv("~/Documents/R/comparing_providers/n800_classification_report.csv")
c1000<- read_csv("~/Documents/R/comparing_providers/n1000_classification_report.csv")
c1200<- read_csv("~/Documents/R/comparing_providers/n1200_classification_report.csv")
c1400<- read_csv("~/Documents/R/comparing_providers/n1400_classification_report.csv")
c1600<- read_csv("~/Documents/R/comparing_providers/n1600_classification_report.csv")
c1800<- read_csv("~/Documents/R/comparing_providers/n1800_classification_report.csv")
c2000<- read_csv("~/Documents/R/comparing_providers/n2000_classification_report.csv")

c200['Seeds']<- '200' 
c400['Seeds']<- '400' 
c600['Seeds']<- '600' 
c800['Seeds']<- '800' 
c1000['Seeds']<- '1000' 
c1200['Seeds']<- '1200'
c1400['Seeds']<- '1400'
c1600['Seeds']<- '1600'
c1800['Seeds']<- '1800'
c2000['Seeds']<- '2000'


class_df<- rbind(c200,c400,c600,c800,c1000,c1200,c1400,c1600,c1800,c2000)

names(class_df)[names(class_df) == "...1"] <- "Species"
names(class_df)[names(class_df) == "Model"] <- "Suppliers"
names(class_df)[names(class_df) == "f1-score"] <- "F1"

rownames(class_df)[rownames(Species) == "A_millefolium"] = "Am"
rownames(class_df)[rownames(class_df) == "C_nigra"] = "Cn"
rownames(class_df)[rownames(class_df) == "D_carota"] = "Dc"
rownames(class_df)[rownames(class_df) == "G_verum"] = "Gv"
rownames(class_df)[rownames(class_df) == "L_vulgare"] = "Lv"
rownames(class_df)[rownames(class_df) == "M_moschata"] = "Mm"
rownames(class_df)[rownames(class_df) == "P_rhoeas"] = "Pr"
rownames(class_df)[rownames(class_df) == "P_vulgaris"] = "Pv"
rownames(class_df)[rownames(class_df) == "R_acetosa"] = "Ra"
rownames(class_df)[rownames(class_df) == "A_millefolium"] = "Am"


class_df <- class_df %>%
  filter(!(Species == "K_arvensis") &
        !(Species == "accuracy") &
        !(Species == "weighted avg") &
        !(Species == "macro avg"))
        

# Convert Seeds to a factor and specify the order of the levels
class_df$Seeds <- factor(class_df$Seeds, levels = c("200", "400", "600", "800", "1000","1200", "1400", "1600", "1800", "2000"))

## this vector might be useful for other plots/analyses
level_order <- c('A_millefolium', 'C_nigra', 'D_carota', 'G_verum', 'L_vulgare', 'M_moschata', 'P_rhoeas', 'P_vulgaris','R_acetosa', 'accuracy', 'macro avg', 'weighted avg') 

class_df$Species <- factor(class_df$Species, levels = level_order)

# Define your custom color palette for Models
model_colors <- c("1" = "blue", "2" = "green", "3" = "yellow", "4" = "orange", "5" = "red")


# Aggregate the data to get mean recall per model and seed number
agg_class <- class_df %>%
  group_by(Suppliers, Seeds, Species) %>%
  summarize(
    Mean_f1 = mean(F1),
    SD = sd(F1),
    .groups = 'drop'
  )

agg_class <- agg_class %>%
  mutate(Species_new = case_when(
    Species == "A_millefolium" ~ "Achillea millefolium",
    Species == "C_nigra" ~ "Centaurea nigra",
    Species == "D_carota" ~ "Daucus carota",
    Species == "G_verum" ~ "Galium verum",
    Species == "L_vulgare" ~ "Leucanthemum vulgare",
    Species == "M_moschata" ~ "Malva moschata",
    Species == "P_rhoeas" ~ "Papaver rhoeas",
    Species == "P_vulgaris" ~ "Prunella vulgaris",
    Species == "R_acetosa" ~ "Rumex acetosa"
  ))

ggplot(agg_class, aes(x = Seeds, y = Mean_f1, group = Suppliers, color = as.factor(Suppliers))) +
  geom_line(size = 1.2) +  # Thicker lines
  geom_point(position = position_jitter(width = 0.1, height = 0), size = 2) +  # Adjust point size
  geom_errorbar(aes(ymin = Mean_f1 - SD, ymax = Mean_f1 + SD), width = 0.3, size = 0.8, alpha = 0.6) +  # Lighter error bars
  scale_color_brewer(palette = "Set1") +  # Use a subtler color palette
  labs(title = "Model f1-scores by Seed Number",
       x = "Training set size (rows per species)",
       y = "Mean F1-score",
       color = "Suppliers") +
  facet_wrap(~ Species_new, scales = "free_y") +  # Italicize species names
  coord_cartesian(ylim = c(0.7, 1)) +  # Y-axis limits
  theme_minimal() +
  theme(text = element_text(family = "sans"),  # Change font to serif
        axis.text.x = element_text(angle = 90, hjust = 1, size = 10),  
        axis.text.y = element_text(size = 10),  # Rotate x-axis labels
        axis.title = element_text(size = 11, face = "bold"),  # Bold axis titles
        strip.text = element_text(size = 11, face = "italic"),  # Larger facet titles
        panel.grid.major = element_line(size = 0.2, color = "grey80"),  # Lighten gridlines
        panel.grid.minor = element_blank())  # Remove minor gridlines


library(Cairo)

# Save the plot directly to PNG at 300 DPI
ggsave("F1_provenance.png", 
       width = 12, 
       height = 8, 
       dpi = 300, 
       type = "cairo-png")





#1 mean precision

# Correctly factor the Species column



# Aggregate the data to get mean precision per model and seed number
agg_class <- class_df %>%
  group_by(Suppliers, Seeds, Species) %>%
  summarize(
    Mean_precision = mean(precision),
    SD = sd(precision),
    .groups = 'drop'
  )

# Define your custom color palette for Models
model_colors <- c("1" = "red", "2" = "orange", "3" = "yellow", "4" = "green", "5" = "blue")

# Plotting the data with faceting by Species
ggplot(agg_class, aes(x = Seeds, y = Mean_precision, group = Suppliers, color = as.factor(Suppliers))) +
  geom_line() +
  geom_point(position = position_jitter(width = 0.1, height = 0), size = 3) +
  geom_errorbar(aes(ymin = Mean_precision - SD, ymax = Mean_precision + SD), width = 0.1, size = 0.5, alpha = 0.6) +
  scale_color_manual(values = model_colors) +
  labs(title = "Average Model Precision by Number of Seeds",
       x = "Number of Seeds",
       y = "Average Precision",
       color = "Suppliers") +
  facet_wrap(~ Species, scales = "fixed") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        strip.text = element_text(size = 10)) +
  scale_y_continuous(limits = c(0.7, 1.0), breaks = seq(0.7, 1.0, by = 0.05))  

### recall ###

# Aggregate the data to get mean recall per model and seed number
agg_class <- class_df %>%
  group_by(Suppliers, Seeds, Species) %>%
  summarize(
    Mean_recall = mean(recall),
    SD = sd(recall),
    .groups = 'drop'
  )

# Define your custom color palette for Models
model_colors <- c("1" = "red", "2" = "orange", "3" = "yellow", "4" = "green", "5" = "blue")

# Plotting the data with faceting by Species
ggplot(agg_class, aes(x = Seeds, y = Mean_recall, group = Suppliers, color = as.factor(Suppliers))) +
  geom_line() +
  geom_point(position = position_jitter(width = 0.1, height = 0), size = 3) +
  geom_errorbar(aes(ymin = Mean_recall - SD, ymax = Mean_recall + SD), width = 0.1, size = 0.5, alpha = 0.6) +
  scale_color_manual(values = model_colors) +
  labs(title = "Average Model recall by Number of Seeds",
       x = "Number of Seeds",
       y = "Average recall",
       color = "Suppliers") +
  facet_wrap(~ Species, scales = "fixed") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        strip.text = element_text(size = 10)) +
  scale_y_continuous(limits = c(0.65, 1.0), breaks = seq(0.65, 1.0, by = 0.05))  



### f1-score ###

# Aggregate the data to get mean recall per model and seed number
agg_class <- class_df %>%
  group_by(Suppliers, Seeds, Species) %>%
  summarize(
    Mean_f1 = mean(F1),
    SD = sd(F1),
    .groups = 'drop'
  )

# Define your custom color palette for Models
model_colors <- c("1" = "red", "2" = "orange", "3" = "yellow", "4" = "green", "5" = "blue")

# Plotting the data with faceting by Species
ggplot(agg_class, aes(x = Seeds, y = Mean_f1, group = Suppliers, color = as.factor(Suppliers))) +
  geom_line() +
  geom_point(position = position_jitter(width = 0.1, height = 0), size = 3) +
  geom_errorbar(aes(ymin = Mean_f1 - SD, ymax = Mean_f1 + SD), width = 0.1, size = 0.5, alpha = 0.6) +
  scale_color_manual(values = model_colors) +
  labs(title = "Average Model f1-score by Number of Seeds",
       x = "Number of Seeds",
       y = "Average f1-score",
       color = "Suppliers") +
  facet_wrap(~ Species, scales = "free_y") +  # Facet by Species
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),  # Rotate x-axis labels for clarity
        strip.text = element_text(size = 10))  # Adjust facet title size if necessary


