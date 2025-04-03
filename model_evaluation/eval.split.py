#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9

@author: ashworth
"""
import pandas as pd
import numpy as np
import os

training_data = pd.read_csv('full_f.csv')
evaluation = training_data.sample(frac = 0.2)

# Creating dataframe with
# rest of the 50% values
training = training_data.drop(evaluation.index)

training.to_csv('train.csv', index=False)
evaluation.to_csv('eval.csv', index=False)