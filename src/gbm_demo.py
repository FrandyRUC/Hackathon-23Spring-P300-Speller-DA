"""
    @authors Zhida Li
    @email zhidal@sfu.ca
    @date Oct. 18, 2020
    @version: 1.0.0
    @description:
                LightGBM: final train test
    @copyright Copyright (c) Oct. 18, 2020
        All Rights Reserved
    @date edited: June 12, 2021
"""

# print(__doc__)

# Import Python libraries
import os
import sys
import time
import warnings

# import pandas as pd
import numpy as np
import scipy.io
from scipy.stats import zscore

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# import lightgbm as lbg
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Import customized libraries
# sys.path.append('./')
from bls.processing.replaceNan import replaceNan
from bls.processing.feature_select_cnl import feature_select_cnl
from bls.processing.metrics_cnl import confusion_matrix_cnl

from sklearn.model_selection import train_test_split

np.seterr(divide='ignore', invalid='ignore')
# warnings.filterwarnings('ignore')


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


# GBDT parameters: list format. Order: 'all', 16, 8 or 'all', 64, 32
num_estimators = [150, 150, 150]  # need to be updated
learn_rate = [0.02, 0.02, 0.02]  # need to be updated

dataset = 'p300'  # need to be updated
algo_gbm = 'lightgbm'  # xgboost, lightgbm, catboost
# dataset_exp_test = ['powerOutage_pk_ripe', 'powerOutage_pk_routeviews',
#                     'ransomware_westRock_ripe', 'ransomware_westRock_routeviews']

# Combine the parameters
total_exp_features_bgp = ['all', 6, 4]
elnList = zip(num_estimators, learn_rate, total_exp_features_bgp)

print('------ Start ------')
result_save = []
head = ['Dataset and number of selected features', 'Accuracy', 'F-Score',
        'Precision', 'Sensitivity', 'TP', 'FN', 'FP', 'TN', 'Training time']
for estNum, lr, n in elnList:
    # Load the datasets
    mats = []
    for i in range(1, 6):
        mats.append(scipy.io.loadmat(f'./p300_datasets/S{i}.mat'))

    SAMPLING_FREQUENCY = 250
    PRE_SAMPLES = int(0.1 * SAMPLING_FREQUENCY)  # -0.1
    POST_SAMPLES = int(0.7 * SAMPLING_FREQUENCY)  # 0.6

    p_samples = {}
    n_samples = {}

    for s in range(len(mats)):
        p_samples[s] = []
        n_samples[s] = []
        for i in range(len(mats[s]['trig'])):
            if mats[s]['trig'][i] == 0:
                pass  # ignore this
            elif mats[s]['trig'][i] == 1:
                p_samples[s].append((i - PRE_SAMPLES, i + POST_SAMPLES))
            elif mats[s]['trig'][i] == -1:
                n_samples[s].append((i - PRE_SAMPLES, i + POST_SAMPLES))

    # let's consider one subject for now, and then we can concat all together after if we want
    # consider subject 0 for now
    subject_index = 0
    x = mats[subject_index]['y']

    y = np.zeros(len(mats[subject_index]['trig']))
    for start_i, end_i in p_samples[subject_index]:
        y[start_i:end_i] = 1

    # Data partition
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, shuffle=False)
    # print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
    print('Number of 1s in train and test:', np.sum(train_y == 1), np.sum(test_y == 1))

    # Normalize training data
    train_x = zscore(train_x, axis=0, ddof=1)  # For each feature, mean = 0 and std = 1
    replaceNan(train_x)  # Replace "nan" with 0
    # train_y = train_dataset[:, -1]

    # Change training labels
    inds1 = np.where(train_y == 0)
    train_y[inds1] = 2

    # Normalize test data
    test_x = zscore(test_x, axis=0, ddof=1)  # For each feature, mean = 0 and std = 1
    replaceNan(test_x)  # Replace "nan" with 0
    # test_y = test_dataset[:, -1]

    # Change test labels
    inds1 = np.where(test_y == 0)
    test_y[inds1] = 2

    # ### feature selection - begin
    if n == 'all':
        pass
    else:
        features = feature_select_cnl(train_x, train_y, n)
        train_x = train_x[:, features]
        test_x = test_x[:, features]
    # ### feature selection - end

    X = train_x
    Y = train_y

    Xx = test_x
    Yy = test_y

    # Training
    time_start = time.time()
    if algo_gbm == 'lightgbm':
        # gbm = LGBMClassifier(n_estimators=estNum, learning_rate=lr)
        # gbm = LGBMClassifier(n_estimators=estNum, learning_rate=lr, max_depth=-1, subsample=1, num_leaves=31,
        #                      boosting_type='gbdt', reg_lambda=0)  #
        gbm = LGBMClassifier(n_estimators=estNum, learning_rate=lr, max_depth=10, subsample=1, num_leaves=20,
                             boosting_type='gbdt', reg_lambda=0)  #
    elif algo_gbm == 'xgboost':
        gbm = XGBClassifier(n_estimators=estNum, learning_rate=lr, max_depth=6, subsample=1,
                            booster='gbtree')  #
    elif algo_gbm == 'catboost':
        gbm = CatBoostClassifier(n_estimators=estNum, learning_rate=lr, max_depth=6, subsample=1, num_leaves=31,
                                 boosting_type='Plain')  # or Ordered, Plain type
    else:
        print('Re-enter the algorithm')
        exit()

    gbm.fit(X, Y.ravel())
    time_end = time.time()
    trainingTime = time_end - time_start

    # Testing
    predicted = gbm.predict(Xx)
    # Test Results
    accuracy = accuracy_score(Yy, predicted)
    fscore = f1_score(Yy, predicted)
    precision = precision_score(Yy, predicted)
    sensitivity = recall_score(Yy, predicted)
    tp, fn, fp, tn = confusion_matrix_cnl(Yy, predicted)

    result_append = ['{}_{}_{}'.format(algo_gbm, dataset, n),
                     '{:.6f}'.format(accuracy * 100), '{:.6f}'.format(fscore * 100),
                     '{:.6f}'.format(precision * 100), '{:.6f}'.format(sensitivity * 100),
                     '{}'.format(tp), '{}'.format(fn), '{}'.format(fp), '{}'.format(tn), '{}'.format(trainingTime)]

    result_save.append(result_append)

result_save = np.asarray(result_save)
head = np.array(head)
head = head.reshape((1, len(head)))
result_save = np.concatenate((head, result_save), axis=0)
# Save final results for each dataset
np.savetxt('%s_results_%s.csv' % (algo_gbm, dataset),
           result_save, fmt='%s',
           delimiter=',')
print('------ Results have been saved to %s_results_%s.csv ------' % (algo_gbm, dataset))

print('------ Completed ------')
