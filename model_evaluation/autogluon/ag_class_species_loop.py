#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17

@author: Eloise Barrett
"""

from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import metrics

# load predictor
predictor = TabularPredictor.load("AutogluonModels/ag-20241031_154442")


#  Clean up species names for unclassified
def new_data_clean_names(filename):
    trained_species = ['A_millefolium', 'C_nigra', 'D_carota', 'G_verum', 'K_arvensis', 'L_vulgare','M_moschata', 'P_rhoeas', 'P_vulgaris', 'R_acetosa']
    new_seed_data = pd.read_csv(f'datasets/{filename}')
    new_seed_data.loc[~new_seed_data['species'].isin(trained_species), 'species'] = 'unclassified'
    target = 'species'
    X = new_seed_data.drop([target, 'supplier'], axis=1)
    y = new_seed_data[target]
    return new_seed_data, X, y

# Rename your dictionary to avoid shadowing built-in names
threshold_dict = {
    'KNeighborsUnif_BAG_L1': 0.85,
    'KNeighborsDist_BAG_L1': 0.97,
    'NeuralNetFastAI_BAG_L1': 0.97,
    'LightGBMXT_BAG_L1': 0.97,
    'LightGBM_BAG_L1': 0.99,
    'RandomForestGini_BAG_L1': 0.65,
    'RandomForestEntr_BAG_L1': 0.65,
    'CatBoost_BAG_L1': 0.95,
    'ExtraTreesGini_BAG_L1': 0.65,
    'ExtraTreesEntr_BAG_L1': 0.65,
    'XGBoost_BAG_L1': 0.95,
    'NeuralNetTorch_BAG_L1': 0.97,
    'LightGBMLarge_BAG_L1': 0.98,
    'WeightedEnsemble_L2': 0.70
}

def autogluon_csv_table(filenames, dictionary):
    cumulative_confusion_matrix = None
    cumulative_report_df = []
    cumulative_df = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'threshold'])
    
    for filename in filenames:
        print(f"Processing dataset: {filename}")
        mock_data, X, y = new_data_clean_names(filename)
        sampled_seed_data = mock_data  # Assuming you want to use the whole dataset
        test_data = TabularDataset(sampled_seed_data)
        label = 'species'

        available_models = predictor.model_names()
        print("Available models in predictor:", available_models)
        print("Models in threshold_dict:", list(dictionary.keys()))

        for model in available_models:
            if model not in dictionary:
                print(f"Threshold for model '{model}' not found in the dictionary. Skipping.")
                continue

            print(f"Processing model: {model}")
            j = dictionary[model]
            print(f"Threshold for {model}: {j}")

            try:
                y_pred_raw = predictor.predict(test_data.drop(columns=[label]), model=model)
                new_data_probs = predictor.predict_proba(test_data, model=model)

                prediction_df = pd.DataFrame(sampled_seed_data)
                prediction_df.loc[:, 'prediction'] = y_pred_raw

                for a in range(len(new_data_probs)):
                    if new_data_probs.iloc[a].max() < j:
                        prediction_df.loc[a, 'prediction'] = "unclassified"

                y_true = test_data['species']
                y_pred = prediction_df['prediction']

                # Generate confusion matrix for this dataset
                labels = sorted(y_true.unique())
                cm = confusion_matrix(y_true, y_pred, labels=labels)

                # Accumulate confusion matrices
                if cumulative_confusion_matrix is None:
                    cumulative_confusion_matrix = cm
                else:
                    cumulative_confusion_matrix += cm

                # Generate classification report
                report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report_dict).transpose()
                report_df['model'] = model
                report_df['dataset'] = filename
                cumulative_report_df.append(report_df)

                # Calculate overall metrics
                accuracy = metrics.accuracy_score(y_true, y_pred)
                precision = metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = metrics.recall_score(y_true, y_pred, average='weighted', zero_division=0)

                cumulative_df.loc[len(cumulative_df)] = [model, accuracy, precision, recall, j]

            except Exception as e:
                print(f"An error occurred while processing model '{model}' on dataset '{filename}': {e}")

    # After processing all datasets
    # Save the cumulative confusion matrix
    if cumulative_confusion_matrix is not None:
        plt.figure(figsize=(10, 8))
        labels = sorted(y_true.unique())
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cumulative_confusion_matrix, display_labels=labels)
        disp.plot(xticks_rotation='vertical')
        plt.tight_layout()
        plt.savefig("output/autogluon_matrices/cumulative_confusion_matrix.png")
        plt.close()

    # Save cumulative classification report
    if cumulative_report_df:
        all_reports_df = pd.concat(cumulative_report_df)
        all_reports_df.to_csv('output/classification_reports/cumulative_classification_report.csv')

    # Save cumulative metrics
    cumulative_df.to_csv('autogluon_cumulative_metrics.csv', index=False)

    return cumulative_df


        # print(new_data_probs)

# Prepare the list of filenames
filenames = [f"mock_test_dataset_{i}.csv" for i in range(1, 50)]  # 1 to 49

# Ensure directories exist
import os
os.makedirs('output/classification_reports/', exist_ok=True)
os.makedirs('output/autogluon_matrices/', exist_ok=True)

# Call the function with the list of filenames
cumulative_df = autogluon_csv_table(filenames, dictionary=threshold_dict)
