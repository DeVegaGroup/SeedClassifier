#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.calibration import CalibratedClassifierCV

# **Retain this part**: Argument parsing
if len(sys.argv) != 2:
    print("Usage: python calibrate_and_test.py <iteration_number>")
    sys.exit(1)

iteration = int(sys.argv[1])

# **Retain this part**: Output directory creation
output_dir = f'outputs/iteration_{iteration}'
os.makedirs(output_dir, exist_ok=True)

# **Retain this part**: Label mapping
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
    return label_mapping.get(label, 'unknown')

# **Retain this part**: Load and preprocess test data
test_data = pd.read_csv(f'../datasets/mock_test_dataset_{iteration}.csv')
X_test = test_data.drop(['species', 'supplier'], axis=1)
y_test_species = test_data['species']

# Map test labels
y_test_mapped = y_test_species.apply(map_labels)

# Prepare true labels as strings
true_species = [
    label_mapping_rev.get(label, 'unknown') if label != 'unknown' else 'unknown'
    for label in y_test_mapped
]

# **Retain this part**: Load saved models
model_filenames = {
    'RandomForest': 'best_model_RandomForest.pkl',
    'AdaBoost': 'best_model_AdaBoost.pkl',
    'GradientBoosting': 'best_model_GradientBoosting.pkl',
    'HistGradientBoosting': 'best_model_HistGradientBoosting.pkl',
    'XGBoost': 'best_model_XGBoost.pkl',
    'MLP': 'best_model_MLP.pkl',
    'KNN': 'best_model_KNN.pkl'
}

models = []
for filename in model_filenames.values():
    with open(filename, 'rb') as file:
        model = pickle.load(file)
        models.append(model)

# **Retain this part**: Load and preprocess calibration data
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

# **New Part**: Define a range of thresholds
thresholds = np.arange(0.0, 1.05, 0.05)

# Process each model individually
for model_name, model in zip(model_filenames.keys(), models):
    print(f"\nProcessing model: {model_name}")

    # Get predicted probabilities
    probas = model.predict_proba(X_test)
    classes = model.classes_

    # Initialize lists to store metrics
    precision_scores = []
    recall_scores = []
    f1_scores = []
    coverage_scores = []

    for threshold in thresholds:
        # Apply thresholding
        max_probas = np.max(probas, axis=1)
        predicted_class_indices = np.argmax(probas, axis=1)

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

        # Clean predicted labels
        predicted_labels = [label if label in label_mapping else 'unknown' for label in predicted_labels]

        # Compute metrics including 'unknown' predictions
        precision = precision_score(
            true_species,
            predicted_labels,
            labels=list(label_mapping.keys()),
            average='macro',
            zero_division=0
        )
        recall = recall_score(
            true_species,
            predicted_labels,
            labels=list(label_mapping.keys()),
            average='macro',
            zero_division=0
        )
        f1 = f1_score(
            true_species,
            predicted_labels,
            labels=list(label_mapping.keys()),
            average='macro',
            zero_division=0
        )

        # Calculate coverage
        coverage = sum(1 for label in predicted_labels if label != 'unknown') / len(predicted_labels)

        # Store the metrics
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        coverage_scores.append(coverage)

    # Plotting the metrics vs thresholds
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision_scores, label='Precision', marker='o')
    plt.plot(thresholds, recall_scores, label='Recall', marker='s')
    plt.plot(thresholds, f1_scores, label='F1 Score', marker='^')
    plt.plot(thresholds, coverage_scores, label='Coverage', marker='x')
    plt.title(f'Precision, Recall, F1 Score, Coverage vs Thresholds for {model_name}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(
        output_dir, f'precision_recall_f1_coverage_{model_name}_{iteration}.png'
    )
    plt.savefig(plot_filename)
    plt.close()

    # Optional: Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Threshold': thresholds,
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1 Score': f1_scores,
        'Coverage': coverage_scores
    })
    metrics_filename = os.path.join(
        output_dir, f'metrics_{model_name}_{iteration}.csv'
    )
    metrics_df.to_csv(metrics_filename, index=False)

# Process ensemble average predictions
print("\nProcessing ensemble average predictions")

def get_average_probabilities(models, X):
    probas = [model.predict_proba(X) for model in models]
    avg_probas = np.mean(probas, axis=0)
    return avg_probas

avg_probas = get_average_probabilities(models, X_test)
classes = models[0].classes_

# Initialize lists to store metrics
precision_scores = []
recall_scores = []
f1_scores = []
coverage_scores = []

for threshold in thresholds:
    # Apply thresholding
    max_probas = np.max(avg_probas, axis=1)
    predicted_class_indices = np.argmax(avg_probas, axis=1)

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

    # Clean predicted labels
    predicted_labels = [label if label in label_mapping else 'unknown' for label in predicted_labels]

    # Compute metrics including 'unknown' predictions
    precision = precision_score(
        true_species,
        predicted_labels,
        labels=list(label_mapping.keys()),
        average='macro',
        zero_division=0
    )
    recall = recall_score(
        true_species,
        predicted_labels,
        labels=list(label_mapping.keys()),
        average='macro',
        zero_division=0
    )
    f1 = f1_score(
        true_species,
        predicted_labels,
        labels=list(label_mapping.keys()),
        average='macro',
        zero_division=0
    )

    # Calculate coverage
    coverage = sum(1 for label in predicted_labels if label != 'unknown') / len(predicted_labels)

    # Store the metrics
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    coverage_scores.append(coverage)

# Plotting the metrics vs thresholds for Ensemble
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision_scores, label='Precision', marker='o')
plt.plot(thresholds, recall_scores, label='Recall', marker='s')
plt.plot(thresholds, f1_scores, label='F1 Score', marker='^')
plt.plot(thresholds, coverage_scores, label='Coverage', marker='x')
plt.title('Precision, Recall, F1 Score, Coverage vs Thresholds for Ensemble')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plot_filename = os.path.join(
    output_dir, f'precision_recall_f1_coverage_Ensemble_{iteration}.png'
)
plt.savefig(plot_filename)
plt.close()

# Optional: Save ensemble metrics to CSV
metrics_df = pd.DataFrame({
    'Threshold': thresholds,
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1 Score': f1_scores,
    'Coverage': coverage_scores
})
metrics_filename = os.path.join(
    output_dir, f'metrics_Ensemble_{iteration}.csv'
)
metrics_df.to_csv(metrics_filename, index=False)