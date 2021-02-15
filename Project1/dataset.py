#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:10:32 2021

@author: apolloseeds
"""

import numpy as np
import pandas as pd

filename = '../Data/dataset.csv'
df = pd.read_csv(filename)

raw_data = df.to_numpy() 

cols = range(0, 13) 
X = raw_data[:, cols]

attributeNames = np.asarray(df.columns[cols])

N, M = X.shape

# Check for missing values
print(np.unique(df.isnull()))