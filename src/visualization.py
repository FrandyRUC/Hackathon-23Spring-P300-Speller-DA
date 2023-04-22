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


def data_view(X_train, y_train):
    import matplotlib.pyplot as plt
    import numpy as np

    # 生成模拟信号数据

    x = np.linspace(0,X_train.shape[1],X_train.shape[1])

    sample_num = 12
    arr = X_train[sample_num]
    transposed_arr = np.transpose(arr)
    y1 = transposed_arr[0]


    # 生成数字信号数据
    x2 = np.array([1, 2, 3, 4, 5])
    y2 = np.array([1, -1, 1, -1, 1])

    # 创建图表和第一个坐标轴
    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, color='red')
    ax1.set_ylabel('模拟信号', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    # 创建第二个坐标轴
    ax2 = ax1.twinx()
    ax2.plot(x2, y2, color='blue')
    ax2.set_ylabel('数字信号', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # 调整坐标轴和标题
    ax1.set_ylim(-25, 25)
    ax2.set_ylim(-1.5, 1.5)
    ax1.set_xlabel('时间')
    ax1.set_title('信号可视化')

    # 显示图表
    plt.show()

def data_view2(X_train):
    import matplotlib.pyplot as plt
    import numpy as np

    sample_num = 10
    arr = X_train[sample_num]
    data = np.transpose(arr)

    # 创建一个图表
    fig, ax = plt.subplots()

    # 为每个波形图绘制线条
    for i in range(8):
        ax.plot(data[i], label=f'wave{i + 1}')

    # 添加图例
    ax.legend()

    # 设置图表标题和横纵坐标标签
    ax.set_title('8 waves')
    ax.set_xlabel('time')
    ax.set_ylabel('y')
    ax.set_ylim(-40, 40)
    # 显示图表
    plt.show()

def data_view3():
    import matplotlib.pyplot as plt
    import numpy as np
    mats=[]
    for i in range(1, 6):
        mats.append(scipy.io.loadmat(f'../p300_datasets/S{i}.mat'))

    print('fs',mats[0]['fs'])
    print('trig', len(mats[0]['trig']))
    print('y', mats[0]['y'])
    data = np.transpose(mats[0]['y'])

    start = 5000
    end = 6000
    y1=data[1][start:end]
    x1 = np.linspace(0, len(y1), len(y1))
    y2 = mats[0]['trig'][start:end]
    x2 = np.linspace(0, len(y2), len(y2))

    # 创建图表和第一个坐标轴
    fig, ax1 = plt.subplots()
    ax1.plot(x1, y1, color='#FF5733')  # 橙色
    ax1.set_ylabel('brainwave', color='#FF5733')
    ax1.tick_params(axis='y', labelcolor='#FF5733')

    # 创建第二个坐标轴
    ax2 = ax1.twinx()
    ax2.plot(x2, y2, color='#3498DB')  # 蓝色
    ax2.set_ylabel('trig', color='#3498DB')
    ax2.tick_params(axis='y', labelcolor='#3498DB')

    # 调整坐标轴和标题
    ax1.set_ylim(-80, 80)
    ax2.set_ylim(-1.1, 1.1)
    ax1.set_xlabel('time')
    ax1.set_title('title')

    # 显示图表
    plt.show()


if __name__ == "__main__":
    # data_view(X_train)
    # X_train, y_train, X_test, y_test = data_process(nn=False)
    # data_view(X_train, y_train)
    # print(y_train[12])
    data_view3()