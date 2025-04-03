#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Eloise Barrett, JDV2024
"""

from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from sklearn.model_selection import train_test_split
import data_clean
from sklearn.preprocessing import StandardScaler
import numpy as np
#from imblearn.over_sampling import RandomOverSampler

seed_data = pd.read_csv('train.csv')

# Set X to all features and y to target
target = 'species'

X = seed_data.drop([target, 'supplier'], axis=1)
y = seed_data[target]

#oversample
#ros = RandomOverSampler(random_state=42)
#X, y = ros.fit_resample(X, y)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

scaler = StandardScaler()

# Create full dataframes to put into autogloun
train_data = X_train.join(y_train)
test_data = X_test.join(y_test)

# Convert into correct form
train_data = TabularDataset(train_data)
test_data = TabularDataset(test_data)

# Fit the models

predictor = TabularPredictor(label =target,sample_weight='balance_weight', eval_metric="f1_macro")
predictor.fit(train_data, presets='best_quality')

y_pred = predictor.predict(test_data.drop(columns=[target]))

# # Evaluate predictions of the test data
predictor.evaluate(test_data, silent=True)
print(predictor.model_best)
# # Create csv file with the models and their corresponding accuracy scores
model_scores=predictor.leaderboard(test_data)
model_scores.to_csv('autogluon_balanced_JA.csv', index=False)

features_important=predictor.feature_importance(test_data)
features_important.to_csv("autogluon_balanced_JA_features_importance.csv", index=False)
