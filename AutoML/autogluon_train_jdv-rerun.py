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

# Clean data up
seed_data, X, y = data_clean.clean_data()

# Split into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

scaler = StandardScaler()

print(X_train)

# Create full dataframes to put into autogloun 
train_data = X_train.join(y_train)
test_data = X_test.join(y_test)

# Convert into correct form
train_data = TabularDataset(train_data)
test_data = TabularDataset(test_data)

print(train_data)

label = 'species'

# Fit the models

predictor = TabularPredictor(label ='species', eval_metric = 'f1_macro', verbosity=3)

print('start fitting...')
predictor.fit(train_data, verbosity = 3, presets='best_quality', time_limit=7200000)

#Output summary of information about models produced during fit().  store them in folder: predictor.path.
print('#############fit_summary#########')
sumfit=predictor.fit_summary(verbosity=3, show_plot=False)
print(sumfit)
#pd.DataFrame.from_dict(sumfit).to_csv("autogluon_evaluate.csv", index=False)
print('#############end_fit_summary#########')




y_pred = predictor.predict(test_data.drop(columns=[label]))
# # Evaluate predictions of the test data
print('#############evaluate#########')
print('evaluate predictions on test data')
eval=predictor.evaluate(test_data, display=True, auxiliary_metrics = True, detailed_report=True)
print(eval)
#pd.DataFrame.from_dict(eval).to_csv("autogluon_evaluate.csv", index=False)

print('#############leaderboard#########')
model_scores=predictor.leaderboard(test_data, extra_info=True, extra_metrics=['f1_macro','precision_macro','recall_macro'], display=True)
model_scores.to_csv('autogluon_BestQ_models.csv', index=False, mode='a') #append


#pd.DataFrame.from_dict(predictor.info).to_csv("autogluon_predictor_info.csv", index=False)




features_important=predictor.feature_importance(test_data)
features_important.to_csv("autogluon_BestQ_features_importance.csv", index=False, mode='a') #append

# Experimenting trying an FT Transformer
#predictor = TabularPredictor(label = label).fit(train_data, hyperparameters={'FT_TRANSFORMER':{}, 'FASTTEXT': {}, 'TABPFN': {}, 'VW' : {}})
# predictor = TabularPredictor(label = label).fit(train_data)
#y_pred = predictor.predict(test_data.drop(columns=[label]))

#predictor.evaluate(test_data, silent=True)

#print(predictor.leaderboard(test_data))

#print(predictor.model_best)
