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

# from sklearn.metrics import accuracy_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score

# Import customized libraries
# sys.path.append('../processing')
# from bls.processing.replaceNan import replaceNan
# from bls.model.vfbls_train_fscore import vfbls_train_fscore
# from bls.model.vcfbls_train_fscore import vcfbls_train_fscore
# from bls.processing.feature_select_cnl import feature_select_cnl
# from bls.processing.metrics_cnl import confusion_matrix_cnl

# from sklearn.model_selection import train_test_split


# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


# blockPrint()

# ============================================== may edit --begin
# Load the datasets



mats = []
for i in range(1, 2):
    mats.append(scipy.io.loadmat(f'./p300_datasets/S{i}.mat'))

print(len(mats[0]['trig']))


SAMPLING_FREQUENCY = 250
PRE_SAMPLES = int(0.1 * SAMPLING_FREQUENCY)  # -0.1
POST_SAMPLES = int(0.7 * SAMPLING_FREQUENCY)  # 0.6

p_samples = {}
n_samples = {}

# for s in range(len(mats)):
#     p_samples[s] = []
#     n_samples[s] = []
#     for i in range(len(mats[s]['trig'])):
#         if mats[s]['trig'][i] == 0:
#             pass  # ignore this
#         elif mats[s]['trig'][i] == 1:
#             p_samples[s].append((i - PRE_SAMPLES, i + POST_SAMPLES))
#         elif mats[s]['trig'][i] == -1:
#             n_samples[s].append((i - PRE_SAMPLES, i + POST_SAMPLES))
#





print(mats[0].keys())

print(len(mats[0]))
print(len(mats[0]['y']))

print(mats[0]['__header__'])
print(mats[0]['__version__'])
print(mats[0]['__globals__'])
print(mats[0]['fs'])





