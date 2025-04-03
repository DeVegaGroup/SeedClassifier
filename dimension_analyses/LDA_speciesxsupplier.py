#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:37:18 2024

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

# Create a color map with a unique color for each supplier
supplier_colors = ListedColormap(sns.color_palette("hsv", len(supplier_to_int)).as_hex())

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Map the supplier names to indices to create a color array
colors = [supplier_to_int[supplier] for supplier in df_species['supplier']]
scatter = ax.scatter(X_lda[:, 0], X_lda[:, 1], X_lda[:, 2], c=colors, cmap=supplier_colors, edgecolor='k', s=40)

ax.set_xlabel('Linear Discriminant 1')
ax.set_ylabel('Linear Discriminant 2')
ax.set_zlabel('Linear Discriminant 3')
ax.set_title('3D LDA Projection for {}'.format(target_species))

# Creating a legend with supplier names
# Custom legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=supplier,
                              markerfacecolor=color, markersize=10) for supplier, color in supplier_to_color.items()]
ax.legend(handles=legend_elements, title="Suppliers")

plt.show()
