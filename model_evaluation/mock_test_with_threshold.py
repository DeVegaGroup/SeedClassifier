#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Check for iteration number argument
if len(sys.argv) != 2:
    print("Usage: python calibrate_and_test.py <iteration_number>")
    sys.exit(1)

iteration = int(sys.argv[1])

# Create output directory for this iteration
output_dir = f'outputs/iteration_{iteration}'
os.makedirs(output_dir, exist_ok=True)

# Your label mapping
label_mapping = {
    'A_millefolium': 0,
    'C_nigra': 1,
    'D_carota': 2,
    'G_verum': 3,
    'K_arvensis': 4,
    'L_vulgare': 5,
    'M_moschata': 6,
    'P_rhoeas': 7,
    'P_vulgaris': 8,
    'R_acetosa': 9,
}

label_mapping_rev = {v: k for k, v in label_mapping.items()}

def map_labels(label):
    if label in label_mapping:
        return label_mapping[label]
    else:
        return 'unknown'

# Load your test data
test_data = pd.read_csv(f'../datasets/mock_test_dataset_{iteration}.csv')
X_test = test_data.drop(['species', 'supplier'], axis=1)
y_test_species = test_data['species']

# Map test labels
y_test_mapped = y_test_species.apply(map_labels)

# Prepare true labels
true_species = []
for label in y_test_mapped:
    if label == 'unknown':
        true_species.append('unknown')
    else:
        species_name = label_mapping_rev.get(label, 'unknown')
        true_species.append(species_name)
# Ensure all labels are strings
true_species = [str(label) for label in true_species]

# Load your saved models
model_filenames = {
    'RandomForest': 'best_model_RandomForest.pkl',
    'AdaBoost': 'best_model_AdaBoost.pkl',
    'GradientBoosting': 'best_model_GradientBoosting.pkl',
    'HistGradientBoosting': 'best_model_HistGradientBoosting.pkl',
    'XGBoost': 'best_model_XGBoost.pkl',
    'MLP': 'best_model_MLP.pkl',
    'KNN': 'best_model_KNN.pkl'
}
threshold_dict = {
    'AdaBoost': 0.15,
    'GradientBoosting': 0.95,
    'KNN': 0.9,
    'HistGradientBoosting': 0.95,
    'MLP': 0.95,
    'RandomForest': 0.8,
    'XGBoost': 0.95
}

models = []
for filename in model_filenames.values():
    with open(filename, 'rb') as file:
        model = pickle.load(file)
        models.append(model)

# Load your calibration data
calibration = pd.read_csv(f'../datasets/calibration_dataset_{iteration}.csv')
X_calibration = calibration.drop(['species', 'supplier'], axis=1)
y_calibration_species = calibration['species']
y_calibration_mapped = y_calibration_species.apply(map_labels)

# Ensure all models support predict_proba
for i, model in enumerate(models):
    if not hasattr(model, 'predict_proba'):
        calibrated_model = CalibratedClassifierCV(model, cv='prefit', method='sigmoid')
        calibrated_model.fit(X_calibration, y_calibration_mapped)
        models[i] = calibrated_model

# Process each model individually
for model_name, model in zip(model_filenames.keys(), models):
    print(f"\nProcessing model: {model_name}")

    # Get the threshold for the current model
    threshold = threshold_dict.get(model_name)

    # Check if the threshold exists
    if threshold is None:
        print(f"Threshold for model '{model_name}' not found in threshold_dict.")
        continue  # Skip to the next model or handle as needed

    # Get predicted probabilities
    probas = model.predict_proba(X_test)

    # Apply thresholding
    max_probas = np.max(probas, axis=1)
    predicted_class_indices = np.argmax(probas, axis=1)
    classes = model.classes_

    # Map predicted class indices to species names
    predicted_labels = []
    for proba, cls_index in zip(max_probas, predicted_class_indices):
        if proba < threshold:
            predicted_labels.append('unknown')
        else:
            cls_label = classes[cls_index]
            species_name = label_mapping_rev.get(cls_label, 'unknown')
            predicted_labels.append(species_name)


    # Ensure all labels are strings
    predicted_labels = [str(label) for label in predicted_labels]

    # Clean predicted labels to only include valid labels
    valid_labels = set(true_species)
    predicted_labels = [label if label in valid_labels else 'unknown' for label in predicted_labels]

    # Evaluate and export classification report to CSV
    print(f"Classification Report for {model_name}:")
    report = classification_report(true_species, predicted_labels, zero_division=0, output_dict=True)
    print(report)

    # Convert classification report to DataFrame
    report_df = pd.DataFrame(report).transpose()

    # Export to CSV
    report_filename = os.path.join(output_dir, f'classification_report_{model_name}_{iteration}.csv')
    report_df.to_csv(report_filename, index=True)

    # Compute confusion matrix
    labels = sorted(valid_labels)
    cm = confusion_matrix(true_species, predicted_labels, labels=labels)

    # Convert confusion matrix to DataFrame
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Export confusion matrix to CSV
    cm_filename = os.path.join(output_dir, f'confusion_matrix_{model_name}_{iteration}.csv')
    cm_df.to_csv(cm_filename)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name} (Iteration {iteration})')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'confusion_matrix_{model_name}_{iteration}.png')
    plt.savefig(plot_filename)
    plt.close()