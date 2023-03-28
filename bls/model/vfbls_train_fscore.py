"""
    @authors Zhida Li, Ana Laura Gonzalez Rios
    @email zhidal@sfu.ca
    @date Mar. 20, 2020
    @version: 1.1.0
    @description:
                This module contains the vfbls function for training/testing.

    @copyright Copyright (c) Mar. 20, 2020
        All Rights Reserved

    This Python code (versions 3.6 and newer)
"""

# ==============================================
# VFBLS module
# ==============================================
# Last modified: Apr. 30, 2022

# Import the Python libraries
import os
import sys
import time
import random
import math

# Import external libraries
import numpy as np
from scipy.stats import zscore
from scipy.linalg import orth
from numpy.linalg import inv

# sys.path.append("..")
from bls.processing.result import result
from bls.processing.sparse_bls import sparse_bls

from sklearn import preprocessing
from bls.processing.mapminmax import mapminmax

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

sys.path.append('../processing')
from bls.processing.feature_select_cnl import feature_select_cnl
from bls.processing.one_hot_m import one_hot_m


# import torch
# torch.cuda.is_available()


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


# rectified linear function
def relu(data):
    return np.maximum(data, 0)


# RBF function
def kerf(matrix):
    return np.exp(np.multiply(-1 * matrix, matrix) / 2.0) / np.sqrt(2 * math.pi);


def vfbls_train_fscore(train_x, train_y, test_x, test_y,
                       s, C, N1, N2, N3,
                       N1_bls_fsm1, N2_bls_fsm1, N1_bls_fsm2, N2_bls_fsm2,
                       add_nfeature1, add_nfeature2):
    """
    Function that creates the VFBLS model
    :param train_x: entire training data
    :param train_y: entire training labels
    :param test_x: entire test data
    :param test_y: entire test labels
    :param s: the shrinkage parameter for enhancement nodes
    :param C: the parameter for sparse regularization
    :param N1: the number of mapped feature nodes (1st set)
    :param N2: the groups of mapped features (1st set)
    :param N3: the number of enhancement nodes
    :param N1_bls_fsm1: the number of mapped feature nodes (2nd set)
    :param N2_bls_fsm1: the groups of mapped features (2nd set)
    :param N1_bls_fsm2: the number of mapped feature nodes (3rd set)
    :param N2_bls_fsm2: the groups of mapped features (3rd set)
    :param add_nfeature1: no. of top relevant features in the 2nd set
    :param add_nfeature2: # no. of top relevant features in the 3rd set
    :return:
    """

    # blockPrint()
    time_start = time.time()
    num_add_vf = 2
    ### feature selection - begin
    # add_nfeature1 = larger
    features = feature_select_cnl(train_x, train_y, -1)
    feature1 = features[0:add_nfeature1]
    train_x1 = train_x[:, feature1]
    test_x1 = test_x[:, feature1]

    # add_nfeature2 = smaller
    # features = feature_select_cnl(train_x, train_y, add_nfeature2)
    feature2 = features[0:add_nfeature2]
    train_x2 = train_x[:, feature2]
    test_x2 = test_x[:, feature2]
    ### feature selection - end
    train_x_vf = [1, train_x1, train_x2]
    test_x_vf = [1, test_x1, test_x2]

    N1_vfbls = [1, N1_bls_fsm1, N1_bls_fsm2]
    N2_vfbls = [1, N2_bls_fsm1, N2_bls_fsm2]

    # Training - begin
    # torch.cuda.synchronize()
    # time_start=time.time()
    beta11 = [];

    train_x = zscore(train_x.transpose(), axis=0, ddof=1).transpose();
    # print(train_x.shape)
    H1 = np.concatenate((train_x, 0.1 * np.ones((train_x.shape[0], 1))), axis=1);

    y = np.zeros((train_x.shape[0], N2 * N1));

    max_list_set = [];
    min_list_set = [];

    ### Generation of mapped features
    for i in range(0, N2):
        # we = 2.0 * np.random.rand(train_x.shape[1] + 1, N1) - 1.0;
        we = 2.0 * np.random.rand(N1, train_x.shape[1] + 1).transpose() - 1.0;

        # we = np.loadtxt("we.csv", delimiter = ",");
        # np.savetxt("we.csv", we, delimiter=",");

        A1 = np.dot(H1, we);
        [A1, max_list, min_list] = mapminmax(A1);
        del we;

        beta1 = sparse_bls(A1, H1, 1e-3, 50).transpose();
        beta11.append(beta1);
        T1 = np.dot(H1, beta1);

        # print("Feature nodes in window ", i, ": Max Val of Output ", T1.max(), " Min Val ", T1.min());

        [T1, max_list, min_list] = mapminmax(T1.transpose(), 0, 1);
        T1 = T1.transpose();

        max_list_set.append(max_list);
        min_list_set.append(min_list);

        y[:, N1 * i: N1 * (i + 1)] = T1;

    del H1;
    del T1;

    ### 2nd set
    beta11_ivf = [1, 1, 1]
    max_list_set_ivf = [1, 1, 1]
    min_list_set_ivf = [1, 1, 1]
    for ivf in range(1, num_add_vf + 1):
        beta11_fsm = [];

        train_x_fsm = zscore(train_x_vf[ivf].transpose(), axis=0, ddof=1).transpose();
        H1_fsm = np.concatenate((train_x_fsm, 0.1 * np.ones((train_x_fsm.shape[0], 1))), axis=1);

        y_fsm = np.zeros((train_x_fsm.shape[0], N2_vfbls[ivf] * N1_vfbls[ivf]));

        max_list_set_fsm = [];
        min_list_set_fsm = [];

        ### Generation of mapped features
        for i in range(0, N2_vfbls[ivf]):
            # we = 2.0 * np.random.rand(train_x.shape[1] + 1, N1) - 1.0;
            we_fsm = 2.0 * np.random.rand(N1_vfbls[ivf], train_x_fsm.shape[1] + 1).transpose() - 1.0;

            # we = np.loadtxt("we.csv", delimiter = ",");
            # np.savetxt("we.csv", we, delimiter=",");

            A1_fsm = np.dot(H1_fsm, we_fsm);
            [A1_fsm, max_list_fsm, min_list_fsm] = mapminmax(A1_fsm);
            del we_fsm;

            beta1_fsm = sparse_bls(A1_fsm, H1_fsm, 1e-3, 50).transpose();
            beta11_fsm.append(beta1_fsm);
            T1_fsm = np.dot(H1_fsm, beta1_fsm);

            # print("Feature nodes in window ", i, ": Max Val of Output ", T1_fsm.max(), " Min Val ", T1_fsm.min());

            [T1_fsm, max_list_fsm, min_list_fsm] = mapminmax(T1_fsm.transpose(), 0, 1);
            T1_fsm = T1_fsm.transpose();

            max_list_set_fsm.append(max_list_fsm);
            min_list_set_fsm.append(min_list_fsm);

            y_fsm[:, N1_vfbls[ivf] * i: N1_vfbls[ivf] * (i + 1)] = T1_fsm;

        del H1_fsm;
        del T1_fsm;
        beta11_ivf[ivf] = beta11_fsm
        max_list_set_ivf[ivf] = max_list_set_fsm
        min_list_set_ivf[ivf] = min_list_set_fsm
        y = np.concatenate((y, y_fsm), axis=1);
    y = np.concatenate((train_x, y), axis=1);

    ### 3rd set
    train_y = one_hot_m(train_y, 2);
    test_y = one_hot_m(test_y, 2);

    ### Generation of enhancement nodes
    H2 = np.concatenate((y, 0.1 * np.ones((y.shape[0], 1))), axis=1);

    if (N1 * N2 + N2_vfbls[1] * N1_vfbls[1] + N2_vfbls[2] * N1_vfbls[2] + train_x.shape[1]) >= N3:
        # wh = orth(2 * np.random.rand(N2 * N1 + 1, N3) - 1);
        wh = orth(2 * np.random.rand(N3, (
                N1 * N2 + N2_vfbls[1] * N1_vfbls[1] + N2_vfbls[2] * N1_vfbls[2] + train_x.shape[
            1]) + 1).transpose() - 1);

    else:
        # wh = orth(2 * np.random.rand(N2 * N1 + 1, N3).transpose() - 1).transpose();
        wh = orth(2 * np.random.rand(N3, (
                N1 * N2 + N2_vfbls[1] * N1_vfbls[1] + N2_vfbls[2] * N1_vfbls[2] + train_x.shape[
            1]) + 1) - 1).transpose();

    # wh = np.loadtxt("wh.csv", delimiter = ",");
    # np.savetxt("wh.csv", wh, delimiter=",");

    T2 = np.dot(H2, wh);
    l2 = T2.max();
    l2 = s * 1.0 / l2;

    # print("Enhancement nodes: Max Val of Output ", l2, " Min Val ", T2.min());

    T2 = np.tanh(T2 * l2);
    T3 = np.concatenate((y, T2), axis=1);

    del H2;
    del T2;
    # print((np.dot(T3.transpose(), T3) + np.identity(T3.transpose().shape[0]) * C).shape)
    # Moore-Penrose pseudoinverse (function pinv)
    beta = np.dot(inv(np.dot(T3.transpose(), T3) + np.identity(T3.transpose().shape[0]) * C),
                  np.dot(T3.transpose(), train_y));

    xx = np.dot(T3, beta);
    del T3;

    # torch.cuda.synchronize()
    time_end = time.time()
    Training_time = time_end - time_start

    # Training - end

    print("Training has been finished!");
    print("The Total Training Time is : ", Training_time, " seconds");

    ### Training Accuracy
    yy = result(xx);
    train_yy = result(train_y);

    cnt = 0;
    for i in range(0, len(yy)):
        if yy[i] == train_yy[i]:
            cnt = cnt + 1;

    TrainingAccuracy = cnt * 1.0 / train_yy.shape[0];

    print("Training Accuracy is : ", TrainingAccuracy * 100, " %");

    ### Testing Process
    # Testing - begin
    time_start = time.time()
    test_x = zscore((test_x).transpose(), axis=0, ddof=1).transpose();

    HH1 = np.concatenate((test_x, 0.1 * np.ones((test_x.shape[0], 1))), axis=1);
    yy1 = np.zeros((test_x.shape[0], N2 * N1));

    ### Generation of mapped features
    for i in range(0, N2):
        beta1 = beta11[i];

        TT1 = np.dot((HH1), (beta1));

        max_list = max_list_set[i];
        min_list = min_list_set[i];

        [TT1, max_list, min_list] = mapminmax(TT1.transpose(), 0, 1, max_list, min_list);
        TT1 = TT1.transpose();

        del beta1;
        del max_list;
        del min_list;

        yy1[:, N1 * i: N1 * (i + 1)] = TT1;

    del TT1;
    del HH1;

    ### 2nd set
    for ivf in range(1, num_add_vf + 1):
        test_x_fsm = zscore((test_x_vf[ivf]).transpose(), axis=0, ddof=1).transpose();

        HH1_fsm = np.concatenate((test_x_fsm, 0.1 * np.ones((test_x_fsm.shape[0], 1))), axis=1);
        yy1_fsm = np.zeros((test_x_fsm.shape[0], N2_vfbls[ivf] * N1_vfbls[ivf]));

        beta11_fsm = beta11_ivf[ivf]
        max_list_set_fsm = max_list_set_ivf[ivf]
        min_list_set_fsm = min_list_set_ivf[ivf]
        ### Generation of mapped features
        for i in range(0, N2_vfbls[ivf]):
            beta1_fsm = beta11_fsm[i];

            TT1_fsm = np.dot((HH1_fsm), (beta1_fsm));

            max_list_fsm = max_list_set_fsm[i];
            min_list_fsm = min_list_set_fsm[i];

            [TT1_fsm, max_list_fsm, min_list_fsm] = mapminmax(TT1_fsm.transpose(), 0, 1, max_list_fsm, min_list_fsm);
            TT1_fsm = TT1_fsm.transpose();

            del beta1_fsm;
            del max_list_fsm;
            del min_list_fsm;

            yy1_fsm[:, N1_vfbls[ivf] * i: N1_vfbls[ivf] * (i + 1)] = TT1_fsm;

        del TT1_fsm;
        del HH1_fsm;

        yy1 = np.concatenate((yy1, yy1_fsm), axis=1);
    yy1 = np.concatenate((test_x, yy1), axis=1);

    ### 3rd set
    ### Generation of enhancement nodes
    HH2 = np.concatenate((yy1, 0.1 * np.ones((yy1.shape[0], 1))), axis=1);
    TT2 = np.tanh(np.dot(HH2, wh) * l2);

    TT3 = np.concatenate((yy1, TT2), axis=1);

    del HH2;
    del wh;
    del TT2;

    x = np.dot(TT3, beta);

    time_end = time.time()
    Testing_time = time_end - time_start

    # Testing - end

    print("Testing has been finished!");
    print("The Total Testing Time is : ", Testing_time, " seconds");

    ### Testing accuracy
    y = result(x);
    test_yy = result(test_y);

    cnt = 0;
    for i in range(0, len(y)):
        if y[i] == test_yy[i]:
            cnt = cnt + 1;

    TestingAccuracy = cnt * 1.0 / test_yy.shape[0];

    label = test_yy;
    predicted = y;

    TestingAccuracy = accuracy_score(label, predicted)
    f_score = f1_score(label, predicted)

    del TT3;

    print("Testing Accuracy is : ", TestingAccuracy * 100, " %");
    print("Testing F-Score is : ", f_score * 100, " %");

    return TrainingAccuracy, TestingAccuracy, Training_time, Testing_time, f_score, predicted
