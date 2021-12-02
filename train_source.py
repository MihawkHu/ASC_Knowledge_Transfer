# train the source seed model
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import keras
import tensorflow
from keras.optimizers import SGD

import argparse
import random
import sys
from utils import *
from models import *

parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--seed', type=int, default=1122, help='random seed')
parser.add_argument('--feat_path', type=str, default='../ASC_Adaptation/features/logmel128_scaled_d_dd/', help='input feature path')
parser.add_argument('--train_csv', type=str, default='tools/evaluation_setup/fold1_train_a.csv', help='training data csv file')
parser.add_argument('--val_csv', type=str, default='tools/evaluation_setup/fold1_evaluate_a.csv', help='evaluation data csv file')
parser.add_argument('--model', type=str, default='resnet', help='target model, should be one of [resnet, fcnn]')
parser.add_argument('--experiments', type=str, default='exp/', help='output experimental files saving path')

parser.add_argument('--num_audio_channels', type=int, default=1, help='input audio channel number')
parser.add_argument('--num_freq_bin', type=int, default=128, help='number of frequncy bins')
parser.add_argument('--num_classes', type=int, default=10, help='number of target classes')
parser.add_argument('--max_lr', type=float, default=0.1, help='maximum learning rate')
parser.add_argument('--mixup_alpha', type=float, default=0.4, help='parameter setting for mixup augmentation')
parser.add_argument('--crop_length', type=int, default=400, help='random cropping input data while training')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--num_epochs', type=int, default=126, help='training epochs')
opt = parser.parse_args()

os.environ['PYTHONHASHSEED']=str(opt.seed)
tensorflow.random.set_random_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)


feat_path = opt.feat_path
train_csv = opt.train_csv
val_csv = opt.val_csv
experiments = opt.experiments

if not os.path.exists(experiments):
    os.makedirs(experiments)

num_audio_channels = opt.num_audio_channels
num_freq_bin = opt.num_freq_bin
num_classes = opt.num_classes
max_lr = opt.max_lr
mixup_alpha = opt.mixup_alpha
crop_length = opt.crop_length
batch_size = opt.batch_size
num_epochs = opt.num_epochs
sample_num = len(open(train_csv, 'r').readlines()) - 1


data_val, y_val = load_data_2020(feat_path, val_csv, num_freq_bin, 'logmel')
y_val = keras.utils.to_categorical(y_val, num_classes)

if opt.model == 'resnet':
    model = model_resnet(num_classes, input_shape=[num_freq_bin,None,3*num_audio_channels], 
                        num_filters=24, wd=0)
elif opt.model == 'fcnn':
    model = model_fcnn(num_classes, input_shape=[num_freq_bin,None,3*num_audio_channels], 
                        num_filters=[48, 96, 192], wd=0)
    

model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
              optimizer =SGD(lr=max_lr,decay=0, momentum=0.9, nesterov=False))
model.summary()

lr_scheduler = LR_WarmRestart(nbatch=np.ceil(sample_num/batch_size), Tmult=2,
                              initial_lr=max_lr, min_lr=max_lr*1e-4,
                              epochs_restart = [3.0, 7.0, 15.0, 31.0, 63.0,127.0,255.0,511.0]) 

callbacks = [lr_scheduler]

train_data_generator = Generator_splitted(feat_path, train_csv, num_freq_bin, 
                              batch_size=batch_size,
                              alpha=mixup_alpha,
                              crop_length=crop_length, splitted_num=4)()

history = model.fit_generator(train_data_generator,
                              validation_data=(data_val, y_val),
                              epochs=num_epochs, 
                              verbose=1, 
                              workers=4,
                              max_queue_size = 100,
                              callbacks=callbacks,
                              steps_per_epoch=np.ceil(sample_num/batch_size)
                              ) 

model.save(experiments + "/model-epoch" + str(num_epochs) + "-seed" + str(seed)+ ".hdf5")
