#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:40:47 2025

@author: ashworth
"""

import os
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.metrics import f1_score

# Original species_to_int mapping
species_to_int = {
    'A_millefolium': 0, 'C_nigra': 1, 'D_carota': 2, 'G_verum': 3, 
    'K_arvensis': 4, 'L_vulgare': 5, 'M_moschata': 6, 'P_rhoeas': 7, 
    'P_vulgaris': 8, 'R_acetosa': 9
}

# Updated mapping without 'K_arvensis'
species_to_int = {k: v for k, v in species_to_int.items() if k != 'K_arvensis'}

df = pd.read_csv("/Users/ashworth/Library/CloudStorage/OneDrive-NorwichBioscienceInstitutes/Jonathan_PhD_project/Rebonto/Seed_Images + CSV/full_f.csv")

# Filter out 'K_arvensis'
df = df[df['species'] != 'K_arvensis']

# Re-map species names to integers
y = df["species"].values
Y = np.array([species_to_int[species] for species in y])

# Drop 'species' and 'supplier' columns from X
X = df.drop(labels=["species", "supplier"], axis=1)

# Reverse mapping for labels
int_to_species = {v: k for k, v in species_to_int.items()}
unique_species = sorted(species_to_int, key=species_to_int.get)
num_classes = len(unique_species)

Y_species = [int_to_species[label] for label in Y]
colors = plt.cm.get_cmap('hsv', num_classes)
species_to_color = {species: colors(i) for i, species in enumerate(unique_species)}
Y_colors = np.array([species_to_color[int_to_species[y]] for y in Y])

# Standardizing the features
X_scaled = StandardScaler().fit_transform(X) 

##############################################################################
#                              LDA Fitting                                   #
##############################################################################
lda = LDA(n_components=3)
X_lda = lda.fit_transform(X_scaled, Y)

##############################################################################
#                    1. Explained Variance & Loadings                        #
##############################################################################
explained_variance_ratio = lda.explained_variance_ratio_
cumulative_variance = np.sum(explained_variance_ratio)

print("=== Variance Explained by LDA Components ===")
for i, variance in enumerate(explained_variance_ratio, start=1):
    print(f"  Linear Discriminant {i}: {variance * 100:.2f}%")

print(f"\nTotal variance explained (LD1 to LD3): {cumulative_variance * 100:.2f}%\n")

# LDA coefficients
lda_coefficients = lda.scalings_

# Compute the importance of each feature by summing absolute coefficients
feature_importance = np.sum(np.abs(lda_coefficients), axis=1)
top_feature_indices = np.argsort(feature_importance)[-10:]
top_feature_names = [X.columns[i] for i in top_feature_indices]

print("=== Top 10 Most Important Features (by LDA Loadings) ===")
for idx, fname in zip(top_feature_indices, top_feature_names):
    print(f"  {fname} (index {idx})")
print()

##############################################################################
#                    2. Classification Accuracy & Confusion Matrix           #
##############################################################################
# Predict on the same dataset
predictions = lda.predict(X_scaled)

# Compute classification accuracy
accuracy = accuracy_score(Y, predictions)
print("=== Classification Accuracy (on entire dataset) ===")
print(f"  Accuracy: {accuracy * 100:.2f}%\n")

# Confusion matrix
cm = confusion_matrix(Y, predictions)

print("=== Confusion Matrix ===")
print(cm, "\n")

# Classification report (precision, recall, F1-score per class)
print("=== Classification Report ===")
print(classification_report(Y, predictions, target_names=unique_species))


# Predictions:
predictions = lda.predict(X_scaled)

# F1-macro score
f1_macro = f1_score(Y, predictions, average='macro')

print("F1 Score (Macro):", f1_macro)

##############################################################################
#                    3. Cross-Validation for Robustness           #
##############################################################################
# 5-fold cross-validation
cv_scores = cross_val_score(lda, X_scaled, Y, cv=5)
print("=== 5-Fold Cross-Validation Scores ===")
print("  Scores:", cv_scores)
print(f"  Mean CV Accuracy: {cv_scores.mean() * 100:.2f}%")
print(f"  Std. Dev.: {cv_scores.std() * 100:.2f}%\n")

##############################################################################
#                            4. Plotting the LDA                             #
##############################################################################
species_colors = ListedColormap(sns.color_palette("hsv", len(species_to_int)).as_hex())

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Map the species names to indices to create a color array
color_indices = [species_to_int[species] for species in Y_species]
scatter = ax.scatter(
    X_lda[:, 0],
    X_lda[:, 1],
    X_lda[:, 2],
    c=color_indices,
    cmap=species_colors,
    edgecolor='k',
    s=40,
    alpha=0.8
)

ax.set_xlabel('Linear Discriminant 1')
ax.set_ylabel('Linear Discriminant 2')
ax.set_zlabel('Linear Discriminant 3')
ax.set_title('3D LDA Projection')

plt.show()
