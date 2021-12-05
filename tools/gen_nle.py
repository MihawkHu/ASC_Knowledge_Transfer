# For method Neural Label Emebdding (NLE) https://arxiv.org/abs/2004.13480
# Generate the label embedding of data
# Usage: python gen_nle.py --model <resnet or fcnn> --source_model <source model path> --nle_path <output nle path>

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Lambda
import tensorflow
from keras.optimizers import Adam 
from scipy.special import softmax

import sys
sys.path.append("./utils/")
import argparse
from utils import *

#from models.resnet import model_resnet
#from models.fcnn_att import model_fcnn

parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--model', type=str, default='resnet')
parser.add_argument('--feat_path', type=str, default='../ASC_Adaptation/features/logmel128_scaled_d_dd/')
parser.add_argument('--train_csv', type=str, default='./tools/evaluation_setup/fold1_train_a.csv')
parser.add_argument('--source_model', type=str, default="../ASC_Adaptation/exp_2020_resnet_baseline_source//model-62-0.7909.hdf5")
parser.add_argument('--nle_path', type=str, default='tools/nle.txt')
parser.add_argument('--epochs', type=int, default=10)
opt = parser.parse_args()
print(opt)


source_model = opt.source_model
train_csv = opt.train_csv
feat_path = opt.feat_path
epochs = opt.epochs

num_audio_channels = 1
num_freq_bin = 128
num_classes = 10
batch_size = 32
sample_num = len(open(train_csv, 'r').readlines()) - 1



def skld(target, inputs):
    loss1 = K.sum(inputs * K.log(inputs / target), axis=-1)
    loss2 = K.sum(target * K.log(target / inputs), axis=-1)
    return (loss1 + loss2) / 2


model = Sequential()
model.add(Dense(num_classes, activation=None, use_bias=False, input_shape=(num_classes,)))
model.add(Activation(keras.layers.activations.softmax))
model.build((1, num_classes))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss=skld, optimizer=adam)
model.summary()

data_train, y_train = load_data_2020(feat_path, train_csv, num_freq_bin, 'logmel')
y_train = keras.utils.to_categorical(y_train, num_classes)

source_model = keras.models.load_model(source_model)
source_model._make_predict_function()
soft_outputs = source_model.predict(data_train)
print(soft_outputs.shape, y_train.shape)

hist = model.fit(y_train, soft_outputs, epochs=epochs, batch_size=batch_size, validation_split=0.0, verbose=1, shuffle=True)

w = model.get_weights()
np.savetxt(opt.nle_path, w[0])
