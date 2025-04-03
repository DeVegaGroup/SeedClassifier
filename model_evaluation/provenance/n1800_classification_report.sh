#!/usr/bin/env python

import os
import seaborn as sns
import pandas as pd
import numpy as np
import xgboost as xgb
from collections import Counter
from xgboost import cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def run_and_save_models(df, iteration):
    df_sorted = df.sort_values('species')

    def get_model(df, num_suppliers, total_rows=1800): #total_rows is the default number of rows taken for the function.
        model_rows = [] # empty list to store sampled rows
        model_suppliers = {} # a dictionary to keep track of which suppliers are selected for each species
        for species in df['species'].unique(): # list iterates over each unique species
            species_df = df[df['species'] == species] # filters df to include only rows corresponding to current species
            unique_suppliers = species_df['supplier'].unique() # extracts unique suppliers for the current species
            selected_suppliers = np.random.choice(unique_suppliers, size=min(num_suppliers, len(unique_suppliers)), replace=False) # randomly selects suppliers for the species, being the smaller of num_suppliers (given in model training for loop) and total number of unique suppliers
            rows_per_supplier = total_rows // len(selected_suppliers) # calculates how many rows each selected supplier should contribute to meet the total_rows target.
            remaining_rows = total_rows % len(selected_suppliers) # calculates how many leftover rows

            for supplier in selected_suppliers: # for loop iterates over each supplier in selected suppliers
                supplier_df = species_df[species_df['supplier'] == supplier] # for the current species, filters rows for the current supplier 
                num_rows_to_sample = rows_per_supplier + (1 if remaining_rows > 0 else 0) # each supplier gets an equal share and the remaining are are dustributed one by one until none left
                if remaining_rows > 0:
                    remaining_rows -= 1 # this decrement reflects that one of the "remaining_rows" has been distributed
                if len(supplier_df) > num_rows_to_sample:
                    sampled_supplier_df = supplier_df.sample(n=num_rows_to_sample) # if therer are enough rows to meet the sampling target, sample them.
                else:
                    sampled_supplier_df = supplier_df # if there aren't enough rows just take the whole lot
                model_rows.append(sampled_supplier_df) # append to empty list created above
            model_suppliers[species] = selected_suppliers.tolist() # records the supplier chosen for each species, outputted below 
        return pd.concat(model_rows, ignore_index=True), model_suppliers

    species_to_int = {'A_millefolium': 0, 'C_nigra': 1, 'D_carota': 2, 'G_verum': 3, 'K_arvensis': 4,   # this is a dictionary that maps species names to integer labels
                      'L_vulgare': 5, 'M_moschata': 6, 'P_rhoeas': 7, 'P_vulgaris': 8, 'R_acetosa': 9}
    supplier_info = {} # an empty dictionary to store info on suppliers used for each model

    for i in range(1, 6): #iterates 5 times with the i taking the values 1 to 5 iteratively
        model, suppliers = get_model(df_sorted, i) # runs the get_model function 5 times, "model" being the sampled dataframe and 'suppliers' being a list of suppliers included 
        supplier_info[f'model_{i}'] = suppliers # Stores the supplier information for each model configuration in the supplier_info dictionary.
        y = model["species"].values # extracts the species column
        Y = np.array([species_to_int[species] for species in y]) # uses the species_to_int mapping to convert y into integers
        X = model.drop(labels=["species", "supplier"], axis=1) # Removes the species and supplier columns from the model DataFrame, leaving only the features to be used for training
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, stratify=y) # Splits the features and labels into training and testing sets, with 20% of the data reserved for testing. The stratify=y parameter ensures that the train and test sets have the same proportion of class labels as the original dataset.
        m = xgb.XGBClassifier(n_estimators=350, eta=0.09, min_child_weight=3, max_depth=6, n_jobs=10) # Initializes an XGBoost classifier with specific hyperparameters.
        evalset = [(X_train, y_train), (X_test, y_test)]
        m.fit(X_train, y_train, eval_metric='mlogloss', eval_set=evalset) # Trains the model using the training data while also using the test set as an evaluation set to monitor performance (e.g., minimizing the 'mlogloss' metric).
        model_path = f"/ei/.project-scratch/4/4e0cae75-e749-4340-a118-93a3d5555375/seed_classifier/trained_models/n1800/m{i}_{iteration}.json"
        m.save_model(model_path)
        print(f"Model {i} of iteration {iteration} saved to {model_path}")

# Load and split data
df = pd.read_csv("/ei/.project-scratch/4/4e0cae75-e749-4340-a118-93a3d5555375/seed_classifier/full_f.csv") # load training data
# Split initial test data
initial_test_data, remaining_data = train_test_split(df, test_size=0.2, stratify=df['species']) # all training is carried out using remaining_data

# Loop to run the function 100 times
for iteration in range(1, 100):
    run_and_save_models(remaining_data, iteration)
    
# this is the end of the training 
# below is testing and evaluation only
    
from sklearn.metrics import accuracy_score, confusion_matrix

def prepare_test_data(df):
    species_to_int = {'A_millefolium': 0, 'C_nigra': 1, 'D_carota': 2, 'G_verum': 3, 'K_arvensis': 4, 
                      'L_vulgare': 5, 'M_moschata': 6, 'P_rhoeas': 7, 'P_vulgaris': 8, 'R_acetosa': 9}
    df['species'] = df['species'].map(species_to_int)
    X_test = df.drop(labels=["species", "supplier"], axis=1)
    y_test = df['species'].values
    return X_test, y_test

X_test, y_test = prepare_test_data(initial_test_data) # here we finally can the test data

def evaluate_models(num_iterations, species_to_int, report_filename='n1800_classification_report.csv'):
    # Initialize an empty DataFrame to store all classification reports
    all_reports = pd.DataFrame()

    species_int_to_name = {v: k for k, v in species_to_int.items()}
    total_species_accuracies = {species: [] for species in species_to_int}

    for iteration in range(1, num_iterations + 1):
        for i in range(1, 6):
            model_path = f"/ei/.project-scratch/4/4e0cae75-e749-4340-a118-93a3d5555375/seed_classifier/trained_models/n1800/m{i}_{iteration}.json"
            m = xgb.XGBClassifier()
            m.load_model(model_path)
            y_pred = m.predict(X_test)
            overall_accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            # Generate classification report
            report = classification_report(y_test, y_pred, target_names=list(species_int_to_name.values()), output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            # Add columns to identify the model and iteration
            df_report['Model'] = i
            df_report['Iteration'] = iteration

            # Append the new report to the all_reports DataFrame
            all_reports = pd.concat([all_reports, df_report], axis=0)

            # Calculate per-species accuracy
            for idx, class_id in enumerate(np.unique(y_test)):
                true_positive = cm[idx, idx]
                total_actual = np.sum(cm[idx, :])
                species_accuracy = true_positive / total_actual if total_actual > 0 else 0
                species_name = species_int_to_name[class_id]
                total_species_accuracies[species_name].append(species_accuracy)

    # Save the combined DataFrame to a CSV file at the end
    all_reports.to_csv(report_filename)
    print(f"Combined classification report saved to {report_filename}")

    return total_species_accuracies

# Evaluate all models for 100 iterations
species_to_int = {'A_millefolium': 0, 'C_nigra': 1, 'D_carota': 2, 'G_verum': 3, 'K_arvensis': 4, 
                  'L_vulgare': 5, 'M_moschata': 6, 'P_rhoeas': 7, 'P_vulgaris': 8, 'R_acetosa': 9}

# Call your model evaluation function to get model_species_accuracies
num_iterations = 99  # This should match the number you use in the evaluation function
model_species_accuracies = evaluate_models(num_iterations, species_to_int)  # This function needs to be defined as before


