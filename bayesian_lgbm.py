import os
import pickle
import logging
import warnings
import pandas as pd
import lightgbm as lgb

from skopt import gp_minimize
from skopt.space import Real, Integer,Categorical
from skopt.utils import use_named_args


from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from utils import *

print(os.listdir('../input/'))
warnings.filterwarnings('ignore')


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('bayesian_search.log', 'a'))
print = logger.info

nr_fold = 5
random_state = 42

train = pd.read_csv('../input/train_input_2yaM34J.csv', parse_dates=['Date'])
y = pd.read_csv('../input/train_output_2kCtjpF.csv')['Score']

train = get_dates(train)

train = train.drop(['ID','Date'],axis=1)

space  = [
          Integer(3, 200, name='max_depth'),
          Integer(2, 2056, name='num_leaves'),
          Integer(3, 200, name='min_child_samples'),
          Real(0.2, 0.90, name='subsample'),
          Real(0.2, 0.90, name='colsample_bytree'),
          Real(0.001, 0.2, name ='learning_rate'),
          Real(0.0001, 100, name = 'reg_alpha'),
          Real(0.0001, 100, name = 'reg_lambda'),
          Integer(2, 1000, name='min_child_weight'),
          Real(0.001, 1, name='min_split_gain'),
          Categorical(['gbdt','dart','goss'], name = 'boosting_type')
         ]

def objective(values):
    params = {
            'device': 'cpu', 
            'objective': 'multiclass', 
            'num_class': 5, 
            'boosting_type': values[10], 
            'n_jobs': -1, 
            'n_estimators': 5000, 
            'metric': 'multi_logloss', 
            'max_depth': values[0], 
            'num_leaves': values[1], 
            'min_child_samples': values[2], 
            'subsample': values[3],
            'colsample_bytree': values[4], 
            'learning_rate':values[5], 
            'reg_alpha': values[6], 
            'reg_lambda': values[7],
            'min_child_weight': values[8], 
            'min_split_gain': values[9],
            'verbose':-1,
            'is_unbalance': 'true'}

    print('\nNext set of params.....')
    print(params)
    oof_preds = np.zeros((len(train), np.unique(y).shape[0]))
    folds = StratifiedKFold(n_splits=nr_fold, 
                                shuffle=True, 
                                random_state=random_state)

    oof_preds = np.zeros((len(train), np.unique(y).shape[0]))

    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):

        X_tr = lgb.Dataset(train.iloc[trn_], y.iloc[trn_])
        X_va = lgb.Dataset(train.iloc[val_], y.iloc[val_])

        model = lgb.train(params, X_tr, valid_sets=X_va, num_boost_round=1000, verbose_eval=None, early_stopping_rounds=100)

        oof_preds[val_, :] = model.predict(train.iloc[val_])
        print('no {}-fold acc: {}'.format(fold_ + 1, 
              accuracy_score(y.iloc[val_], np.argmax(oof_preds[val_,:],axis=1))))

    score =  accuracy_score(y, np.argmax(oof_preds,axis=1))
    print('Score: {:.5f}'.format(score))
    return  -score

res_gp = gp_minimize(objective, space, n_calls = 60 , random_state = random_state, n_random_starts = 15)

print(res_gp)

with open('res_bayesian.pkl', 'wb') as f :
    pickle.dump(res_gp,f)