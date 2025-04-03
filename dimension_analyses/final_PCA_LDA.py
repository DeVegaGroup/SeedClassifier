#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:09:33 2024

@author: ashworth
"""
import os
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

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

# Create color mapping for species
colors = plt.cm.get_cmap('hsv', num_classes)
species_to_color = {species: colors(i) for i, species in enumerate(unique_species)}

# Standardizing the features
X_scaled = StandardScaler().fit_transform(X)

##############################################################################
#                           1. PCA Fitting                                   
##############################################################################
pca = PCA(n_components=3)  # We choose 3 components for 3D visualization
X_pca = pca.fit_transform(X_scaled)

##############################################################################
#                           2. Explained Variance                            
##############################################################################
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Print variance explained by each component and cumulative variance
print("=== PCA Explained Variance ===")
for i, var in enumerate(explained_variance, start=1):
    print(f"  PC{i}: {var * 100:.2f}%")
print(f"Total variance explained (PC1 to PC3): {cumulative_variance[-1] * 100:.2f}%\n")

# Optionally, create a "scree plot" to show variance explained by each component
plt.figure(figsize=(6,4))
plt.plot(range(1, 4), explained_variance, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.show()

##############################################################################
#                           3. Feature Loadings                              
##############################################################################
# "components_" has shape (n_components, n_features). Each row is a PC's loadings.
# The loadings show how each original feature contributes to that PC.

# Let's identify the top 5 contributing features for each PC (for illustration).
loadings = pca.components_  # shape: [3, n_features]
feature_names = X.columns.tolist()

for pc_idx in range(loadings.shape[0]):
    # Sort features by absolute contribution to this PC
    sorted_idx = np.argsort(np.abs(loadings[pc_idx]))[::-1]
    top_features_idx = sorted_idx[:5]  # top 5
    print(f"=== Top 5 Features Contributing to PC{pc_idx + 1} ===")
    for f_idx in top_features_idx:
        print(f"  {feature_names[f_idx]}: loading={loadings[pc_idx, f_idx]:.4f}")
    print()

##############################################################################
#                           4. 3D PCA Plot                                   
##############################################################################
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Map each sample's species to a color
Y_colors = [species_to_color[int_to_species[label]] for label in Y]

# Scatter plot in 3D
scatter = ax.scatter(X_pca[:, 0],
                     X_pca[:, 1],
                     X_pca[:, 2],
                     c=Y_colors,
                     edgecolor='k',
                     s=40,
                     alpha=0.8)

# Custom legend
legend_elements = [
    plt.Line2D([0], [0],
               marker='o', color='w',
               label=species,
               markerfacecolor=species_to_color[species],
               markersize=10)
    for species in unique_species
]
ax.legend(handles=legend_elements, title="Species")

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D PCA Projection')

plt.show()

