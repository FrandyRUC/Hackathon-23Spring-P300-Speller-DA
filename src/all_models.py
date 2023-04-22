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
# import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

# Import customized libraries
# sys.path.append('../processing')
from bls.processing.replaceNan import replaceNan
from bls.model.vfbls_train_fscore import vfbls_train_fscore
from bls.model.vcfbls_train_fscore import vcfbls_train_fscore
from bls.processing.feature_select_cnl import feature_select_cnl
from bls.processing.metrics_cnl import confusion_matrix_cnl

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#from keras.models import Sequential
#from keras.layers import Dense, LSTM
from keras.optimizers.optimizer_v2.adam import Adam

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)
from keras import regularizers

from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.layers import Dropout
from keras.regularizers import l2

def data_process(nn):
    mats = []

    for i in range(1, 6):
        mats.append(scipy.io.loadmat(f'../p300_datasets/S{i}.mat'))

    SAMPLING_FREQUENCY = 250
    PRE_SAMPLES = int(0.1 * SAMPLING_FREQUENCY)  # -0.1
    POST_SAMPLES = int(1.0 * SAMPLING_FREQUENCY)  # 0.7

    p_samples = {}
    n_samples = {}
    # print('fs',mats[0]['fs'])
    # print('trig', len(mats[0]['trig']))
    # print('y', mats[0]['y'])
    files_lst = range(len(mats))
    for s in files_lst:
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

    # for i in range(1,6):
    #     print(len(p_samples[subject_index]),len(n_samples[subject_index]))

    # for subject_index in files_lst:
    subject_index = 2
    # x = mats[subject_index]['y']
    num_cols = len(p_samples[subject_index]) + len(n_samples[subject_index])
    y = []
    X = []
    for subject_index  in files_lst:
        for i in range(len(p_samples[subject_index])):
            sample = mats[subject_index]['y'][p_samples[subject_index][i][0]:p_samples[subject_index][i][1], :]
            X.append(sample)
            y.append(1)

        for i in range(len(n_samples[subject_index])):
            sample = mats[subject_index]['y'][n_samples[subject_index][i][0]:n_samples[subject_index][i][1], :]
            X.append(sample)
            y.append(0)

    print(len(X))
    print(len(X[0][0]))
    X_np = np.array(X)
    y_np = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=4)
    if nn:
        y_train = to_categorical(y_train, num_classes=2)
        y_test = to_categorical(y_test, num_classes=2)
    return X_train, y_train, X_test, y_test



def model_cnn(X_train, y_train,X_test, y_test):
    import tensorflow as tf
    from tensorflow.python.keras import layers
    from keras.callbacks import CSVLogger, ModelCheckpoint

    # 设置输入维度
    input_shape = (X_train.shape[1],X_train.shape[2])
    print("input_shape:",input_shape)
    # input_shape = (275, 8)
    # 275= (0.1+1)*250
    # 定义模型
    model = tf.keras.Sequential([
        # 卷积层1
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),

        # 卷积层2
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),

        # 卷积层3
        layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),

        # 全连接层
        layers.Flatten(),
        layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.1)),
        layers.Dropout(0.7),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.1)),
        layers.Dropout(0.7),
        layers.Dense(2, activation='softmax')
    ])


    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    logger = CSVLogger('logs.csv', append=True)
    his = model.fit(X_train, y_train, epochs=30, batch_size=32,
                    validation_data=(X_test, y_test), callbacks=[logger])

    # check the model performance on test data
    scores = model.evaluate(X_test, y_test)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    y_pred = model.predict(X_test)
    auc_score = roc_auc_score(y_test, y_pred)
    print("CNN -AUC Score: ", auc_score)
    # # 打印模型结构
    # model.summary()
    return model

def model_dnn(X_train, y_train, X_test, y_test):
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    model = Sequential()
    model.add(Dense(512, input_shape=(X_train.shape[1],), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(2, activation='softmax'))
    # model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
    auc_score = roc_auc_score(y_test, y_pred)
    print("DNN - AUC Score: ", auc_score)
    print('DNN Test Loss:', loss)
    print('DNN Test Accuracy:', accuracy)


def model_lstm(X_train, y_train, X_test, y_test):
    model = Sequential()

    # 添加LSTM层和Dense层
    model.add(LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=l2(0.05)))
    model.add(Dropout(0.2))  # 添加 Dropout 层
    model.add(Dense(units=2, activation='softmax', kernel_regularizer=l2(0.05)))

    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 训练模型
    model.fit(X_train, y_train, epochs=10, batch_size=32,  validation_data=(X_test, y_test))

    # 评估模型
    loss, accuracy = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
    auc_score = roc_auc_score(y_test, y_pred)
    print("LSTM -AUC Score: ", auc_score)
    print('LSTM -Test Loss:', loss)
    print('LSTM -Test Accuracy:', accuracy)

def model_xgb(X_train, y_train, X_test, y_test):


    import xgboost as xgb
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=800,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.5,
        reg_alpha=5, # L1正则化项系数
        reg_lambda=100, # L2正则化项系数
        random_state=42
    )
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, eval_metric=['error', 'auc'], verbose=True)
    # Testing
    y_pred = model.predict(X_test)
    # auc_score = roc_auc_score(y_test, y_pred)
    # print("XGB -AUC Score: ", auc_score)
    # accuracy = accuracy_score(y_test, y_pred)
    # print("XGB Accuracy:", accuracy)

    # 计算准确率和AUC
    accuracy = 1 - model.evals_result()['validation_1']['error'][model.best_iteration]
    auc = model.evals_result()['validation_1']['auc'][model.best_iteration]

    print("XGB Accuracy:", accuracy)
    print("XGB AUC:", auc)


if __name__ == "__main__":

    # X_train, y_train, X_test, y_test = data_process(nn=True)
    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    # model_cnn(X_train, y_train, X_test, y_test)
    # model_dnn(X_train, y_train, X_test, y_test)
    # model_lstm(X_train, y_train, X_test, y_test)
    X_train, y_train, X_test, y_test = data_process(nn=False)
    model_xgb(X_train, y_train, X_test, y_test)



