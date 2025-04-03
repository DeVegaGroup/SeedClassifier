#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os

# Step 1: Load known and unknown species data
known_data = pd.read_csv('eval.csv')
unknown_data = pd.read_csv('unknown_spp.csv')

# Create a directory to store the datasets
os.makedirs('datasets', exist_ok=True)

# Number of iterations
num_iterations = 49

# Loop over iterations
for i in range(1, num_iterations + 1):
    random_state_cal = i       # Random seed for calibration dataset
    random_state_test = i + 49 # Random seed for mock test dataset

    # Step 2: Sample for calibration dataset
    calibration_known = known_data.sample(n=1000, replace=False, random_state=random_state_cal)
    calibration_unknown = unknown_data.sample(n=1000, replace=False, random_state=random_state_cal)

    calibration_dataset = pd.concat([calibration_known, calibration_unknown], ignore_index=True)
    calibration_dataset = calibration_dataset.sample(frac=1, random_state=random_state_cal).reset_index(drop=True)

    # Step 3: Sample for mock_test dataset
    mock_test_known = known_data.sample(n=1000, replace=False, random_state=random_state_test)
    mock_test_unknown = unknown_data.sample(n=1000, replace=False, random_state=random_state_test)

    mock_test_dataset = pd.concat([mock_test_known, mock_test_unknown], ignore_index=True)
    mock_test_dataset = mock_test_dataset.sample(frac=1, random_state=random_state_test).reset_index(drop=True)

    # Step 4: Save datasets to files
    calibration_filename = f'datasets/calibration_dataset_{i}.csv'
    mock_test_filename = f'datasets/mock_test_dataset_{i}.csv'

    calibration_dataset.to_csv(calibration_filename, index=False)
    mock_test_dataset.to_csv(mock_test_filename, index=False)

    print(f"Iteration {i}: Datasets created and saved successfully.")