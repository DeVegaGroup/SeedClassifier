#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:04:11 2024

@author: ashworth
"""
from autogluon.tabular import TabularPredictor
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load predictor
predictor = TabularPredictor.load("AutogluonModels/ag-20241031_125239")

# Load test data
test_data = pd.read_csv("eval.csv")
X_test = test_data.drop(columns=["species"])
y_test = test_data["species"]

# Get the list of all models, including ensemble
model_names = predictor.get_model_names()

# Iterate over each model to generate predictions, classification reports, and confusion matrices
for model_name in model_names:
    # Get predictions by model
    predicted_labels = predictor.predict(X_test, model=model_name)

    # Generate classification report
    report = classification_report(y_test, predicted_labels, zero_division=0)
    print(f"\nClassification Report for model {model_name}:")
    print(report)

    # Convert classification report to DataFrame
    report_df = pd.DataFrame(classification_report(y_test, predicted_labels, zero_division=0, output_dict=True)).transpose()

    # Save report to .csv
    report_df.to_csv(f"classification_report_by_species_{model_name}.csv", index=True)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, predicted_labels)

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    classes = np.unique(y_test)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Annotate the confusion matrix with the actual values
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment="center", 
                     color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

    # Save the confusion matrix as .png
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model_name}.png")
    plt.close()
