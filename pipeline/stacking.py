#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:36:52 2019

@author: tianqiluke
"""

import os
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression

np.sort(os.listdir('./subs'))
pred = list(np.sort([c for c in (os.listdir('./subs')) if 'sub' in c]))
oof = list(np.sort([c for c in (os.listdir('./subs')) if 'oof' in c]))
target = pd.read_csv('./input/train.csv')['target']
val = pd.DataFrame()
for j,i in enumerate(oof):
    val['m{}'.format(j)] = pd.read_csv(os.path.join('./subs',i))['oof']
    print(i,'\t auc : ', roc_auc_score(target,val['m{}'.format(j)]))
cv = cross_val_score(LinearRegression(), val, target, cv = 5, scoring = 'roc_auc')
print(cv.mean(), cv.std())

from sklearn.model_selection import StratifiedKFold

models = []
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(val))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(val.values, target.values)):

    X_tr = val.iloc[trn_idx]
    y_tr = target.iloc[trn_idx]
    
    X_val = val.iloc[val_idx]
    y_val = target.iloc[val_idx]
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    models.append(model)
    oof[val_idx] = model.predict(X_val)
    print("Fold {}".format(fold_),'roc_auc_score : ', roc_auc_score(y_val,oof[val_idx]))
    
score = roc_auc_score(target,oof)
print('roc_auc_score : ', score)

test = pd.read_csv('./subs/'+pred[0])
test.rename(columns={'target':'m0'},inplace = True)
for j,i in enumerate(pred[1:]):
    df = pd.read_csv(os.path.join('./subs',i))
    df.rename(columns={'target':'m{}'.format(j+1)},inplace = True)
    test = test.merge(df, on='ID_code')

# any file is ok, just to get ids 
sub = pd.read_csv('./subs/sub_cat_0.9180511510872795.csv')

preds = 0.
for model in models:
    preds+=model.predict(test[[c for c in test.columns if c!='ID_code']])/5.

sub['target'] = preds
sub.to_csv('linear_stacking_{}.csv;'.format(score), index = False)
