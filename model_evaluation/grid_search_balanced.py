#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:07:56 2024

@author: ashworth
"""

import joblib
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from imblearn.pipeline import Pipeline
import pickle


# Load in seed data
seed_data = pd.read_csv('train.csv')

# Create a label encoder object
label_encoder = preprocessing.LabelEncoder()

# Label encoder to convert string into integer
seed_data['species'] = label_encoder.fit_transform(seed_data['species'])

# Set X to all features and y to target
target = 'species'

X = seed_data.drop([target, 'supplier'], axis=1)
y = seed_data[target]

# Define parameter grids for each classifier (excluding KNN and MLP)
param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'class_weight': [None, 'balanced']
    },
    'AdaBoost': {
        'n_estimators': [50, 100],
        'learning_rate': [0.5, 1.0],
    },
    'GradientBoosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
    },
    'HistGradientBoosting': {
        'max_iter': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [None, 10],
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
    }
}

# Initialize classifiers (excluding KNN and MLP)
classifiers = {
    'RandomForest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'HistGradientBoosting': HistGradientBoostingClassifier(),
    'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss')
}

best_models = {}

# Use Stratified K-Fold for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in classifiers.items():
    print(f"Processing {name}...")
    param_grid = param_grids[name]
    # For classifiers that don't support 'class_weight', use resampling in a pipeline
    if name in ['AdaBoost','GradientBoosting','HistGradientBoosting','XGBoost']:
        # Create a pipeline with resampling
        pipeline = Pipeline([
            ('resample', RandomOverSampler(random_state=42)),
            ('classifier', clf)
        ])
        # Adjust param_grid keys to match pipeline steps
        param_grid = {f'classifier__{key}': value for key, value in param_grid.items()}
        estimator = pipeline
    else:
        estimator = clf
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        verbose=3,
        scoring='f1_macro',
        refit=True
    )
    grid_search.fit(X, y)
    best_models[name] = grid_search.best_estimator_
    print(f"Best Parameters for {name}: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score for {name}: {grid_search.best_score_}\n")

# Train KNN and MLP with default parameters

# KNN with default parameters and scaling
print("Processing KNN with default parameters...")
pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('resample', RandomOverSampler(random_state=42)),
    ('classifier', KNeighborsClassifier())
])
pipeline_knn.fit(X, y)
best_models['KNN'] = pipeline_knn
print("KNN model trained with default parameters.")

# MLP with default parameters and scaling
print("Processing MLP with default parameters...")
pipeline_mlp = Pipeline([
    ('scaler', StandardScaler()),
    ('resample', RandomOverSampler(random_state=42)),
    ('classifier', MLPClassifier(max_iter=500, random_state=42))
])
pipeline_mlp.fit(X, y)
best_models['MLP'] = pipeline_mlp
print("MLP model trained with default parameters.")

# Save each best model separately
for name, model in best_models.items():
    filename = f'best_model_{name}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved {name} model to {filename}")
