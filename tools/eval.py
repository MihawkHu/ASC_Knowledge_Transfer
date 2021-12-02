import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import h5py
import scipy.io
import pandas as pd

import keras
import tensorflow as tf

import sys
sys.path.append("../utils/")
from utils import *
from funcs import *


feat_path = '../features/logmel128_scaled_d_dd/'


device = 'b'
model_path = 'exp/model.hdf5'
csv_file = 'evaluation_setup/fold1_evaluate_' + device + '.csv'


num_freq_bin = 128
num_classes = 10

best_model = keras.models.load_model(model_path)


accs = []
data_val, y_val = load_data_2020(feat_path, csv_file, num_freq_bin, 'logmel')

preds = best_model.predict(data_val)
y_pred_val = np.argmax(preds,axis=1)

acc = np.sum(y_pred_val==y_val) / data_val.shape[0]
accs.append(acc)

np.set_printoptions(precision=4)
print(np.array(accs))







