"""
     @description:
     All the dataset feature are added into feature_vectors
    @date edited: April 16, 2023 
"""
#Import the built-in libraries th
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
#from keras.optimizers.optimizer_v2.adam import Adam # for MAC
from keras.optimizers import  Adam #for linux

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================== may edit --begin
SAMPLING_FREQUENCY = 250
PRE_SAMPLES = int(0.1 * SAMPLING_FREQUENCY)  # -0.1
POST_SAMPLES = int(1.0 * SAMPLING_FREQUENCY)  # 0.7
# Load the datasets
mats = []
for i in range(1, 6):
    mats.append(scipy.io.loadmat(f'../p300_datasets/S{i}.mat'))

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
sum_cols=0
sum_rows=0
for s in range(len(p_samples)):
    x = mats[s]['y']
    # Transpose the array using zip()
    transposed_s = np.array(list(map(list, zip(*x))))
    num_cols = len(p_samples[s]) + len(n_samples[s])
    sum_cols=sum_rows+num_cols
    num_rows = len(transposed_s) * (PRE_SAMPLES + POST_SAMPLES)
    sum_rows=sum_rows+num_rows
print(f"num_cols: {sum_cols}, num_rows: {sum_rows} ")

feature_vectors = np.zeros((sum_rows, sum_cols))
y = np.zeros((sum_cols))
# Function to convert array to feature vector
def convert_to_feature_vector(arr):
    rows = len(arr)
    cols = len(arr[0])
    new_arr = np.zeros((rows * cols, 1))
    index = 0
    for j in range(cols):
        for i in range(rows):
            new_arr[index, 0] = arr[i][j]
            index += 1
    return new_arr
for s in range(len(mats)):
    p_samples = []
    n_samples = []

    for i in range(len(mats[s]['trig'])):
        if mats[s]['trig'][i] == 0:
            pass  # ignore this
        elif mats[s]['trig'][i] == 1:
            p_samples.append((i - PRE_SAMPLES, i + POST_SAMPLES))
        elif mats[s]['trig'][i] == -1:
            n_samples.append((i - PRE_SAMPLES, i + POST_SAMPLES))
    # Concatenate feature vectors and labels for the current subject
    for i in range(len(p_samples)):
        s_data = mats[s]['y']
        transposed_s = np.array(list(map(list, zip(*s_data))))
        matSlice = convert_to_feature_vector(transposed_s[:, p_samples[i][0]:p_samples[i][1]])
        sLengh = len(matSlice)
        startL = s * sLengh
        endL = startL + sLengh
        # Pad matSlice with zeros along dimension 1 to match the size of the larger array
        max_size = feature_vectors.shape[1]
        if matSlice.shape[1] < max_size:
            matSlice = np.pad(matSlice, ((0, 0), (0, max_size - matSlice.shape[1])), mode='constant')
        feature_vectors[startL:endL, i] = matSlice[:, 0]
        y[startL+i] = 1

    for i in range(len(n_samples)):
        s_data = mats[s]['y']
        transposed_s = np.array(list(map(list, zip(*s_data))))
        matSlice = convert_to_feature_vector(transposed_s[:, n_samples[i][0]:n_samples[i][1]])
        sLengh = len(matSlice)
        startL = s * sLengh
        endL = startL + sLengh
        # Pad matSlice with zeros along dimension 1 to match the size of the larger array
        max_size = feature_vectors.shape[1]
        if matSlice.shape[1] < max_size:
            matSlice = np.pad(matSlice, ((0, 0), (0, max_size - matSlice.shape[1])), mode='constant')
        feature_vectors[startL:endL, len(p_samples[s]) + i] = matSlice[:, 0]
        y[startL+len(p_samples[s]) + i] = 0


# Convert feature_vectors and y to NumPy arrays
feature_vectors = np.array(feature_vectors)
y = np.array(y)

#
trans_feature = np.array(list(map(list, zip(*feature_vectors))))
print(len(trans_feature))
# Data partition
X_train, X_test, y_train, y_test = train_test_split(trans_feature, y, test_size=0.2, random_state=4)

scaler = StandardScaler()
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print('Number of train and test:', len(y_train), len(y_test))

# # Scaling the data scaled the columns of data to the range of [0, 1].
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


classifiers = {
    'SVM': SVC(kernel='linear', C=1),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42),
}


for name, clf in classifiers.items():
    start_time = time.time()
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    elapsed_time = time.time() - start_time
    auc_score = roc_auc_score(y_test, y_pred)
    # print(y_train, y_pred)
    print(f"{name} - Accuracy: {accuracy:.4f}, - AUC: {auc_score:.4f} Time: {elapsed_time:.2f} seconds")

#
# LSTM Model
X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

model = Sequential()
model.add(LSTM(32, input_shape=(X_train_scaled.shape[1], 1)))
model.add(Dense(1, activation='sigmoid'))



# model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.1) , metrics=['mean_squared_error'])

# model.compile(loss='binary_crossentropy', metrics=['accuracy'])

start_time = time.time()
model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=0)

elapsed_time = time.time() - start_time


# score, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)



y_pred = model.predict(X_test_scaled)
y_pred_classes = tf.argmax(y_pred, axis=1) # 将预测结果转换为类别标签
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_classes, y_test), tf.float32)) # 计算准确率
print("LSTM -Accuracy: ", accuracy)
# print(y_test,y_pred)

# 计算 AUC
auc_score = roc_auc_score(y_test, y_pred)
print("LSTM -AUC Score: ", auc_score)
# print(f"LSTM - Accuracy: {accuracy}, Time: {elapsed_time:.2f} seconds")
# print(f"LSTM - Accuracy: {accuracy:.4f}, Time: {elapsed_time:.2f} seconds")

   #print('------ Results have been saved to %s_results_%s.csv ------')
#
print('------ Completed ------')