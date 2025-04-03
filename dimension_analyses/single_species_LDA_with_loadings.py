#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:55:44 2024

@author: ashworth
"""

import os
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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

# Specify the species to analyze
target_species = 'R_acetosa'

# Filter the dataframe for the specified species
df_species = df[df['species'] == target_species]

# Map suppliers to integers
supplier_to_int = {supplier: idx for idx, supplier in enumerate(df_species['supplier'].unique())}
Y = df_species['supplier'].map(supplier_to_int).values

# Drop 'species' and 'supplier' columns from X
X = df_species.drop(labels=["species", "supplier"], axis=1)

# Reverse mapping for labels
int_to_supplier = {v: k for k, v in supplier_to_int.items()}
unique_suppliers = sorted(supplier_to_int, key=supplier_to_int.get)
num_classes = len(unique_suppliers)

colors = plt.cm.get_cmap('hsv', num_classes)
supplier_to_color = {supplier: colors(i) for i, supplier in enumerate(unique_suppliers)}
Y_colors = np.array([supplier_to_color[int_to_supplier[y]] for y in Y])

# Standardizing the features
X_scaled = StandardScaler().fit_transform(X)  # Assuming X is your feature matrix

lda = LDA(n_components=3)  # As the number of classes is 10, n_components can be at most 9
X_lda = lda.fit_transform(X_scaled, Y)  # Assuming Y is your target variable

# Get the coefficients of the linear discriminants
lda_coefficients = lda.scalings_

# Compute the importance of each feature
feature_importance = np.sum(np.abs(lda_coefficients), axis=1)

# Get indices of the top 10 features
top_feature_indices = np.argsort(feature_importance)[-5:]

# Get the corresponding feature names
top_feature_names = [X.columns[i] for i in top_feature_indices]

# Create a color map with a unique color for each supplier
supplier_colors = ListedColormap(sns.color_palette("hsv", len(supplier_to_int)).as_hex())

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Map the supplier names to indices to create a color array
colors = [supplier_to_int[supplier] for supplier in df_species['supplier']]
scatter = ax.scatter(X_lda[:, 0], X_lda[:, 1], X_lda[:, 2], c=colors, cmap=supplier_colors, edgecolor='k', s=40)

# Normalize the vectors for better visualization
scale_factor = 0.3  # Increase this factor for better visibility
for i, feature_name in zip(top_feature_indices, top_feature_names):
    vector = lda_coefficients[i] * scale_factor
    print("Plotting vector:", vector, "for feature:", feature_name)
    ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color='r', linewidth=2.5)  # Increased linewidth
    # Add labels to the end of each scaled vector with a larger offset
    ax.text(vector[0] * 1.1, vector[1] * 1.1, vector[2] * 1.1, feature_name, color='blue', fontsize=10, weight='bold')  # Changed color and fontsize

ax.set_xlabel('Linear Discriminant 1')
ax.set_ylabel('Linear Discriminant 2')
ax.set_zlabel('Linear Discriminant 3')
ax.set_title('3D LDA Projection')

# Creating a legend with species names
# Custom legend
# Creating a legend with supplier names
# Custom legend
#legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=supplier,
#                              markerfacecolor=color, markersize=10) for supplier, color in supplier_to_color.items()]
#ax.legend(handles=legend_elements, title="Suppliers")

plt.show()
