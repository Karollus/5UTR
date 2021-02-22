## Imports
# base python
from importlib import reload
import re
import itertools
import random
import os
from pathlib import Path
random.seed(1337)
import pickle
from decimal import Decimal
import collections
# numpy and similar
import numpy as np
np.random.seed(1337)
import pandas as pd
pd.options.mode.chained_assignment = None 
import scipy.stats as stats
# modelling and utility code
import model
import utils_data
import utils
# sklearn
from sklearn import ensemble, preprocessing, pipeline, model_selection, metrics
from joblib import dump, load
# hyperopt
import hyperopt
from hyperopt import tpe, hp, fmin

import sys

out_encoding_fn = utils_data.DataFrameExtractor("rl", method="direct")

with open(Path("../Data/data_dict.pkl"), 'rb') as handle:
    data_dict = pickle.load(handle)
    

out_encoding_fn = utils_data.DataFrameExtractor("rl", method="direct")
# extract kmers
kmer_extractor = utils_data.FramedKmerExtractor(seq_col="utr", new_col="kmer_count", k=4, jump=False, divide_counts=False)

mpra_data = data_dict["mpra"]
train_data_50 = mpra_data[(mpra_data.set == "train") & (mpra_data.library == "egfp_unmod_1")]
generator_50 = utils_data.DataSequence(train_data_50, encoding_functions=[], precomputations=[kmer_extractor],
                                    output_encoding_fn=out_encoding_fn, shuffle=True, batch_size=len(train_data_50))
X_train, y_train = [(X[0],y) for X,y in generator_50][0]
val_data_50 = mpra_data[(mpra_data.set == "val") & (mpra_data.library == "egfp_unmod_1")]
val_50 = utils_data.DataSequence(val_data_50, encoding_functions=[], precomputations=[kmer_extractor],
                                    output_encoding_fn=out_encoding_fn, shuffle=True, batch_size=len(val_data_50))
X_val, y_val = [(X[0],y) for X,y in val_50][0]
    
    
def objective(params):
    if params['max_depth']:
        params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    print(params['n_estimators'])
    params["n_jobs"] = -1
    params["verbose"] = 3
    rf = ensemble.RandomForestRegressor(**params)
    print(rf)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_val)
    score = -(stats.pearsonr(pred, y_val)[0] ** 2)
    return score

search_space = {
    'max_depth': hp.quniform('max_depth', 8, 32, 1),
    'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
    'n_estimators': hp.choice('number_of_trees', [16, 32, 64, 100, 200])
}

best = fmin(
    fn=objective, # Objective Function to optimize
    space=search_space, # Hyperparameter's Search Space
    algo=tpe.suggest, # Optimization algorithm
    max_evals=500 # Number of optimization attempts
)

print(best)

