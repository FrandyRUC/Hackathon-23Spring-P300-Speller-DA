"""
    @authors Zhida Li, Ana Laura Gonzalez Rios
    @email zhidal@sfu.ca
    @date Feb. 19, 2022
    @version: 1.1.0
    @description:
                This module contains implementation VFBLS and VCFBLS with datasets.

    @copyright Copyright (c) Feb. 19, 2022
        All Rights Reserved

    This Python code (versions 3.6 and newer)
"""

# ==============================================
# VFBLS and VCFBLS cross-validation code
# ==============================================
# Last modified: Apr. 30, 2022

# Import the built-in libraries
import os
import sys
import time
import random

# Import external libraries
import numpy

numpy.set_printoptions(threshold=sys.maxsize)
import numpy as np
import scipy.io
import scipy.stats
from scipy.stats import zscore
from scipy import signal
import matplotlib.pyplot as plt

# Import customized libraries
# sys.path.append('../processing')
from bls.processing.replaceNan import replaceNan
from bls.model.vfbls_train_fscore import vfbls_train_fscore
from bls.model.vcfbls_train_fscore import vcfbls_train_fscore
from bls.processing.feature_select_cnl import feature_select_cnl

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold
# np.set_printoptions(threshold=np.inf)

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


blockPrint()

if os.path.exists("tempAcc_p300"):
    pass
else:
    os.mkdir('tempAcc_p300')

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
train_dataset0 = np.concatenate((train_x, train_y.reshape((-1, 1))), axis=1)
print('train_dataset0 shape:', train_dataset0.shape)

seed = 1;
num_class = 2;

data_kfold = 3
dataSplit = TimeSeriesSplit(n_splits=data_kfold)  # k-fold cross validation
# dataSplit = KFold(n_splits=data_kfold)  # k-fold cross validation

# bls parameters
C = 2 ** -15;
s = 0.8;

epochs = 1;
algo = 'VCFBLS'  # 'VFBLS' or 'VCFBLS'

list1 = [20, 40, 100]
list2 = [10, 20]
list3 = [50, 100, 200]

list11 = [20, 40]
list12 = [10]

list21 = [10, 20]
list22 = [5]

add_nfeature1 = 6
add_nfeature2 = 4

n1n2n3List = []
for N2_bls_fsm2 in list22:
    for N1_bls_fsm2 in list21:
        for N2_bls_fsm1 in list12:
            for N1_bls_fsm1 in list11:
                for N3 in list3:
                    for N2 in list2:
                        for N1 in list1:
                            list0 = [N1, N2, N3, N1_bls_fsm1, N2_bls_fsm1, N1_bls_fsm2, N2_bls_fsm2]
                            n1n2n3List.append(list0)

# n1n2n3List = [[10,10,10],[20,10,10],[30,10,10]]

index_n1n2n3 = 1
for N1_bls_fsm, N2_bls_fsm, N3_bls_fsm, N1_bls_fsm1, N2_bls_fsm1, N1_bls_fsm2, N2_bls_fsm2 in n1n2n3List:
    index = 1
    acc_val_all = np.array([])
    acc_val_all_00 = np.array([])
    acc_val_all_save = np.array([])
    # Extract train and validate data below
    for train_index, val_index in dataSplit.split(train_dataset0):
        print("--------------------------------------------Validation: %d" % index)
        train = train_dataset0[train_index, :]
        validate = train_dataset0[val_index, :]
        train_dataset = train
        test_dataset = validate

        train_x = train_dataset[:, 0:train_dataset.shape[1] - 1];
        train_x = zscore(train_x, axis=0, ddof=1);
        replaceNan(train_x);

        train_y = train_dataset[:, train_dataset.shape[1] - 1: train_dataset.shape[1]];

        # Change label 0 to 2
        inds1 = np.where(train_y == 0);
        train_y[inds1] = 2;

        test_x = test_dataset[:, 0:test_dataset.shape[1] - 1];
        test_x = zscore(test_x, axis=0, ddof=1);
        replaceNan(test_x);

        test_y = test_dataset[:, test_dataset.shape[1] - 1: test_dataset.shape[1]];

        inds1 = np.where(test_y == 0);
        test_y[inds1] = 2;

        # train_y = one_hot_m(train_y, num_class);
        # test_y = one_hot_m(test_y, num_class);

        train_err = np.zeros((1, epochs));
        test_err = np.zeros((1, epochs));
        train_time = np.zeros((1, epochs));
        test_time = np.zeros((1, epochs));

        acc_val_all_0 = np.array([])
        print("================== BLS ===========================\n\n");

        bls_test_acc = 0;
        bls_test_f_score = 0;
        bls_train_time = 0;
        bls_test_time = 0;

        # print ("test accuracy and F-Score above...\n\n\n");
        # sys.exit();

        print("================== RBF BLS ===========================\n\n");

        rbfbls_test_acc = 0;
        rbfbls_test_f_score = 0;
        rbfbls_train_time = 0;
        rbfbls_test_time = 0;

        # print ("test accuracy and F-Score above...\n\n\n");

        if algo == 'VFBLS':
            print("================== VFBLS ===========================\n\n");
            np.random.seed(seed);

            for j in range(0, epochs):
                TrainingAccuracy, TestingAccuracy, Training_time, Testing_time, f_score, _ = \
                    vfbls_train_fscore(train_x, train_y, test_x,
                                       test_y, s, C, N1_bls_fsm, N2_bls_fsm, N3_bls_fsm, N1_bls_fsm1, N2_bls_fsm1,
                                       N1_bls_fsm2, N2_bls_fsm2, add_nfeature1, add_nfeature2);
        elif algo == 'VCFBLS':
            print("================== VCFBLS ===========================\n\n");
            np.random.seed(seed);

            for j in range(0, epochs):
                TrainingAccuracy, TestingAccuracy, Training_time, Testing_time, f_score, _ = \
                    vcfbls_train_fscore(train_x, train_y, test_x,
                                       test_y, s, C, N1_bls_fsm, N2_bls_fsm, N3_bls_fsm, N1_bls_fsm1, N2_bls_fsm1,
                                       N1_bls_fsm2, N2_bls_fsm2, add_nfeature1, add_nfeature2);
        else:
            print("Please re-enter the algorithm")
            exit()

            train_err[0, j] = TrainingAccuracy * 100;
            test_err[0, j] = TestingAccuracy * 100;
            train_time[0, j] = Training_time;
            test_time[0, j] = Testing_time;

        blsfsm_test_acc = TestingAccuracy;
        blsfsm_test_f_score = f_score;
        blsfsm_train_time = Training_time;
        blsfsm_test_time = Testing_time;

        # print ("test accuracy and F-Score above...\n\n\n");
        '''
        print ("BLS Test Acc: ", bls_test_acc*100, " fscore: ", bls_test_f_score*100, "Training time: ", bls_train_time);
        print ("RBF-BLS Test Acc: ", rbfbls_test_acc*100, " fscore: ", rbfbls_test_f_score*100, "Training time: ", rbfbls_train_time);
        print ("CFBLS Test Acc: ", cfbls_test_acc*100, " fscore: ", cfbls_test_f_score*100, "Training time: ", cfbls_train_time);
        print ("CEBLS Test Acc: ", cebls_test_acc*100, " fscore: ", cebls_test_f_score*100, "Training time: ", cebls_train_time);
        print ("CFEBLS Test Acc: ", cfebls_test_acc*100, " fscore: ", cfebls_test_f_score*100, "Training time: ", cfebls_train_time);
        print("End of the execution");
        '''
        acc_val_all_0 = np.append([bls_test_acc, rbfbls_test_acc], blsfsm_test_acc)
        if index == 1:
            acc_val_all = acc_val_all_0.reshape(-1, 1)
            # acc_tr_all = acc_tr_all_0.reshape(-1,1)
        else:
            acc_val_all_00 = acc_val_all_0.reshape(-1, 1)
            acc_val_all = np.append(acc_val_all, acc_val_all_00, axis=1)
            # acc_tr_all_00 = acc_tr_all_0.reshape(-1,1)
            # acc_tr_all = np.append(acc_tr_all, acc_tr_all_00,axis=1)

        index += 1

    # row: bls, column accuracies of val1 to val10, average
    # print(acc_val_all)
    acc_val_all_mean = np.mean(acc_val_all, axis=1)
    acc_val_all_save = np.append(acc_val_all, acc_val_all_mean.reshape(-1, 1), axis=1)

    # np.savetxt('validation_acc.csv', acc_val_all_save, delimiter=',',fmt='%.4f')
    np.savetxt('./tempAcc_p300/train_acc' + str(N1_bls_fsm) + str(N2_bls_fsm) + str(N3_bls_fsm)
               + str(N1_bls_fsm1) + str(N2_bls_fsm1)
               + str(N1_bls_fsm2) + str(N2_bls_fsm2) + '.csv', acc_val_all_save, delimiter=',', fmt='%.4f')
    index_n1n2n3 += 1
