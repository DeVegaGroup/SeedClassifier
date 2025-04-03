#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17

@author: Eloise Barrett
"""

from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import metrics

# load predictor
predictor = TabularPredictor.load("AutogluonModels/ag-20241031_154442")


#  Clean up species names for unclassified
def new_data_clean_names(new_data):
    trained_species = ['A_millefolium', 'C_nigra', 'D_carota', 'G_verum', 'K_arvensis', 'L_vulgare','M_moschata', 'P_rhoeas', 'P_vulgaris', 'R_acetosa']
    new_seed_data = pd.read_csv(f'datasets/{new_data}.csv')
    print(new_seed_data['species'].value_counts())
    print(new_seed_data['species'].unique())
    new_seed_data.loc[~new_seed_data['species'].isin(trained_species), 'species'] = 'unclassified'
    # new_seed_data['species'] = new_seed_data['species'].map(full_species_dict)
    print(new_seed_data['species'].unique())
    #new_seed_data['species'] = new_seed_data['actual_species'].map(full_species_dict)

    target = 'species'
    #X = new_seed_data.drop([target, 'actual_species'], axis = 1)
    X = new_seed_data.drop([target, 'supplier'], axis = 1)
    y = new_seed_data[target]

    return new_seed_data, X, y

mock_data, X, y = new_data_clean_names("mock_test_dataset_1")
print("hello", predictor.model_names())

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

def autogluon_csv_table(mock_data, X, y, rows, dictionary):
    df = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'threshold'])

    sampled_seed_data = mock_data.sample(n=rows, ignore_index=True)
    test_data = TabularDataset(sampled_seed_data)
    label = 'species'

    available_models = predictor.model_names()
    print("Available models in predictor:", available_models)
    print("Models in threshold_dict:", list(dictionary.keys()))

    for model in available_models:
        if model not in dictionary:
            print(f"Threshold for model '{model}' not found in the dictionary. Skipping.")
            continue

        df_model = pd.DataFrame(columns=['threshold', 'accuracy', 'precision', 'recall'])
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
                    prediction_df.loc[[a], ['prediction']] = "unclassified"

            y_true = test_data['species']
            y_pred = prediction_df['prediction']

            # Generate classification report
            classes = sorted(y_true.unique())
            report = classification_report(y_true, y_pred, labels=classes, zero_division=0)
            print(report)

            # Save the report as a text file
            with open(f"{model}_classification_report.txt", 'w') as f:
                f.write(report)

            # Convert the report to a DataFrame and save as CSV
            report_dict = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report_dict).transpose()
            report_df.to_csv(f"{model}_classification_report.csv")

            # Calculate overall metrics
            accuracy = metrics.accuracy_score(y_true, y_pred)
            precision = metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = metrics.recall_score(y_true, y_pred, average='weighted', zero_division=0)

            # Generate and save confusion matrix
            metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, xticks_rotation='vertical')
            plt.tight_layout()
            plt.savefig(f"{model}.png")
            plt.close()

            df_model.loc[len(df_model)] = [j, accuracy, precision, recall]

            max_index = df_model['accuracy'].idxmax()
            max_value_row = df_model.loc[max_index]

            df.loc[len(df.index)] = [model, max_value_row['accuracy'], max_value_row['precision'], max_value_row['recall'], max_value_row['threshold']]

        except Exception as e:
            print(f"An error occurred while processing model '{model}': {e}")

    print(df)
    df.to_csv('autogluon_repeat_list_unclassified.csv', index=False)

    return df

        # print(new_data_probs)

autogluon_csv_table(mock_data, X, y, rows=1000, dictionary = threshold_dict)

trained_species = ['A_millefolium', 'C_nigra', 'D_carota', 'G_verum', 'K_arvensis', 'L_vulgare','M_moschata', 'P_rhoeas', 'P_vulgaris', 'R_acetosa']
unclassified_data = mock_data.loc[~mock_data["species"].isin(trained_species)]
print(unclassified_data)

# target = 'species'

X_unclassified = unclassified_data.drop(['species'], axis = 1)
y_unclassified = unclassified_data['species']
