#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 01:17:51 2019

@author: tianqiluke
"""

import numpy as np
import pandas as pd
import os
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import gc
import time

import xgboost as xgb

# Load dataset
train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')
fake_ids = np.load('../input/fakeids/synthetic_samples_indexes.npy')

ids = np.arange(test.shape[0])
real_ids = list(set(ids) - set(fake_ids))
real_test = test.iloc[real_ids]
fake_test = test.iloc[fake_ids]
real_test_id = real_test.ID_code
features = [c for c in train.columns if c not in ['target', 'ID_code']]
df = pd.concat([train,real_test], axis = 0)

# FE
for feat in features:
    df[feat+'_var'] = df.groupby([feat])[feat].transform('var')
    df[feat+'plus_'] = df[feat] + df[feat+'_var']
    df[feat+"minus"] = df[feat] - df[feat+'_var']
    
drop_features = [c for c in df.columns if '_var' in c]
df.drop(drop_features, axis=1, inplace=True)


# set up model mat
train = df.iloc[:train.shape[0]]
real_test = df.iloc[train.shape[0]:]

features = [c for c in train.columns if c not in ['ID_code', 'target']]
target = train['target'] 

del df
gc.collect()

# shuffle func
def disarrange(a, axis=-1):
    """
    Shuffle `a` in-place along the given axis.

    Apply numpy.random.shuffle to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])
    return

def augment_fast2(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        disarrange(x1,axis=0)
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        disarrange(x1,axis=0)
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y

# modeling
param = {
    'bagging_freq': 5,  'bagging_fraction': 0.4,  'boost_from_average':'false',   
    'boost': 'gbdt',    'feature_fraction': 0.04, 'learning_rate': 0.01,
    'max_depth': -1,    'metric':'auc',             'min_data_in_leaf': 80,
    'num_leaves': 10,  'num_threads': 8,            
    'tree_learner': 'serial',   'objective': 'binary',       'verbosity': 1
}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4242)
oof = np.zeros(len(train))
predictions = np.zeros(len(real_test))
val_aucs = []
feature_importance_df = pd.DataFrame()

#shuffle version
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("Fold {}".format(fold_))
    X_train, y_train = train.iloc[trn_idx][features], train.iloc[trn_idx]['target']
    X_valid, y_valid = train.iloc[val_idx][features], train.iloc[val_idx]['target']
    N = 2 #shuffle times for each fold
    p_valid,yp = 0,0
    imp = np.zeros(len(features))
    for i in range(N):
        print("shuffle {}".format(i))
        X_t, y_t = augment_fast2(X_train.values, y_train.values)
        X_t = pd.DataFrame(X_t)
        X_t = X_t.add_prefix('var_')
    
        trn_data = lgb.Dataset(X_t, label=y_t)
        val_data = lgb.Dataset(X_valid, label=y_valid)
        lgb_clf = lgb.train(param,
                        trn_data,
                        100000,
                        valid_sets = [trn_data, val_data],
                        early_stopping_rounds=3000,
                        verbose_eval = 5000
                       )
        p_valid += lgb_clf.predict(X_valid, num_iteration=lgb_clf.best_iteration)/N
        yp += lgb_clf.predict(real_test[features], num_iteration=lgb_clf.best_iteration)/N
        imp += lgb_clf.feature_importance()/N
    
    oof[val_idx] = p_valid
    val_aucs.append(roc_auc_score(target[val_idx] , oof[val_idx] ))
    predictions += yp / folds.n_splits
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = imp
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    mean_auc = np.mean(val_aucs)
    std_auc = np.std(val_aucs)
    all_auc = roc_auc_score(target, oof)
    print("Mean auc: %.9f, std: %.9f. All auc: %.9f." % (mean_auc, std_auc, all_auc))
    
    #save the real test
    subreal = pd.DataFrame({"ID_code": real_test_id.values})
    subreal['target']=predictions
    sub = pd.DataFrame({"ID_code": test.ID_code.values})
    finalsub = sub.set_index('ID_code').join(subreal.set_index('ID_code')).reset_index()
    finalsub.fillna(0,inplace=True)
    finalsub.to_csv('lgb_sub{}.csv'.format(all_auc), index=False)
    
    # save the oof pred for train
    oof_pred = pd.DataFrame()
    oof_pred['oof'] = oof
    oof_pred.to_csv('lgb_oof{}.csv'.format(all_auc), index=False)
    
    