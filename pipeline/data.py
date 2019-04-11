#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:31:21 2019

@author: tianqiluke
"""

import numpy as np
import pandas as pd
import os
import gc

train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')
fake_ids = np.load('../input/fakeids/synthetic_samples_indexes.npy')
ids = np.arange(test.shape[0])
real_ids = list(set(ids) - set(fake_ids))
real_test = test.iloc[real_ids]
fake_test = test.iloc[fake_ids]
real_test_id = real_test.ID_code

features = [c for c in train.columns if c not in ['target', 'ID_code']]

# generate var_plus
df = pd.concat([train,real_test], axis = 0)
for feat in tqdm_notebook(features):
    df[feat+'_var'] = df.groupby([feat])[feat].transform('var')
for feat in tqdm_notebook(features):
    df[feat+'plus_'] = df[feat] + df[feat+'_var']
drop_features = [c for c in df.columns if '_var' in c]
df.drop(drop_features, axis=1, inplace=True)

# generate count
for feat in tqdm_notebook(features):
    df['{}_count'.format(feat)] = df.groupby([feat])[feat].transform('count')
    df['{}_prod'.format(feat)] = df[feat] * df['{}_count'.format(feat)]
    df['{}_div'.format(feat)] = df[feat] / df['{}_count'.format(feat)]

drop_features = [c for c in df.columns if 'count' in c]
df.drop(drop_features, axis=1, inplace=True)

train = df.iloc[:train.shape[0]]
real_test = df.iloc[train.shape[0]:]

train.to_csv('./train.csv', index=False)
real_test.to_csv('./realtest.csv', index=False)