#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:04:11 2024

@author: ashworth
"""
from autogluon.tabular import TabularPredictor
from sklearn.metrics import classification_report
import pandas as pd

# Load the predictor
predictor = TabularPredictor.load("AutogluonModels/ag-20241031_125239")

# Load your test data
test_data = pd.read_csv("eval.csv")
X_test = test_data.drop(columns=["species"])
y_test = test_data["species"]

# Get the list of all models, including the ensemble
model_names = predictor.get_model_names()

# Iterate over each model to generate predictions and classification reports
for model_name in model_names:
    # Get predictions from the specific model
    predicted_labels = predictor.predict(X_test, model=model_name)
    
    # Generate a classification report
    report = classification_report(y_test, predicted_labels, zero_division=0)
    print(f"\nClassification Report for model {model_name}:")
    print(report)
    
    # Convert the classification report to a DataFrame
    report_df = pd.DataFrame(classification_report(y_test, predicted_labels, zero_division=0, output_dict=True)).transpose()
    
    # Save the report to a CSV file
    report_df.to_csv(f"classification_report_by_species_{model_name}.csv", index=True)
