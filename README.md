# Kaggle-Competition-Santander
Top 0.5% rankings (44/9038) code sharing for Kaggle competition: Santander Customer https://www.kaggle.com/c/santander-customer-transaction-prediction

* 922VarNoShuffle.py: Apply var transformation and shuffling (augmentation)
* 923Var_Shuffle.py: Apply var transformation only
* Stacking.ipynb: model stacking (final model) using linear stacking
* bayesian_lgbm.py: hyper-parameter tuning for lightgbm
* pipline: separate pipeline script 

Final pipeline: Raw data -> data.py ->  lightgbm.py -> Stacking.py

Original data set can be downloaded from the competition page
