import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import keras
import tensorflow
from keras.optimizers import SGD
from scipy.special import softmax

import argparse
import random
from kt_losses import *
from utils.utils import *
from utils.funcs import *
from utils.DCASE_training_functions import *
from models import *


parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--device', type=str, default='b', 
                        help='target device, should be one of [b, c, s1, s2, s3, s4, s5, s6])')
parser.add_argument('--seed', type=int, default=1122, help='random seed')
parser.add_argument('--feat_path', type=str, default='../ASC_Adaptation/features/logmel128_scaled_d_dd/', 
                        help='input feature path')
parser.add_argument('--model', type=str, default='resnet', help='target model, should be one of [resnet, fcnn]')
parser.add_argument('--source_model', type=str, default="../ASC_Adaptation/exp_2020_resnet_baseline_source//model-62-0.7909.hdf5", 
                        help='pretrained source model path')
parser.add_argument('--experiments', type=str, default='exp/', help='output experimental files saving path')

parser.add_argument('--num_audio_channels', type=int, default=1, help='input audio channel number')
parser.add_argument('--num_freq_bin', type=int, default=128, help='number of frequncy bins')
parser.add_argument('--num_classes', type=int, default=10, help='number of target classes')
parser.add_argument('--max_lr', type=float, default=0.1, help='maximum learning rate')
parser.add_argument('--mixup_alpha', type=float, default=0.4, help='parameter setting for mixup augmentation')

parser.add_argument('--trans_way', type=str, default='tsl', help='knowledge transfer way, \
                        should be one of [onehot, tsl, nle, fitnets, at, ab, vid, fsp, cofd, sp, cckd, pkt, nst, rkd, vbkt]')
parser.add_argument('--lmd', type=float, default=0.0, help='loss weight of the basic CE loss')
parser.add_argument('--soft_ratio', type=float, default=1.0, help='loss weights of the TSL loss, zero to disable TSL')
parser.add_argument('--temperature', type=float, default=1.0, help='temperature parameter for TSL')
parser.add_argument('--alpha', type=float, default=1.0, help='loss weight of the knowledge transfer method')
parser.add_argument('--beta', type=float, default=2.0, help='loss weight for the triangle loss term of RKD method')
parser.add_argument('--nle_path', type=str, default="tools/nle.txt", help='label embedding path for NLE method')
parser.add_argument('--cckd_emb_size', type=int, default=128, help='embdding size for CCKD method')
parser.add_argument('--sigma', type=float, default=0.2, help='std value for vbkt method')
parser.add_argument('--latent_layer', type=str, default='activation_33', help='the focused hidden layer name, for only using single layer')
parser.add_argument('--latent_layers', nargs='+', default=['activation_9', 'activation_10', 'activation_21', 'activation_22', 'activation_33'], 
                        help='the focused group of hedden layer names, for using multiple layers')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--num_epochs', type=int, default=62, help='training epochs')
parser.add_argument('--num_epochs_pretrain', type=int, default=30, help='pretraining epochs, only for AB and FSP methods')
opt = parser.parse_args()
print(opt)


os.environ['PYTHONHASHSEED']=str(opt.seed)
tensorflow.random.set_random_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)


train_csv = 'toos/evaluation_setup/fold1_train_' + opt.device + '.csv'
train_paired_csv = 'tools/evaluation_setup/fold1_train_' + opt.device + '_paired_a.csv'
val_csv = 'tools/evaluation_setup/fold1_evaluate_' + opt.device + '.csv'

feat_path = opt.feat_path
source_model = opt.source_model
experiments = opt.experiments

if not os.path.exists(experiments):
    os.makedirs(experiments)

num_audio_channels = opt.num_audio_channels
num_freq_bin = opt.num_freq_bin
num_classes = opt.num_classes
max_lr = opt.max_lr
mixup_alpha = opt.mixup_alpha
sample_num = len(open(train_csv, 'r').readlines()) - 1

trans_way = opt.trans_way
batch_size = opt.batch_size
num_epochs = opt.num_epochs
num_epochs_pretrain = opt.num_epochs_pretrain
lmd = opt.lmd
tem = opt.temperature
soft_ratio = opt.soft_ratio
alpha = opt.alpha
beta = opt.beta
nle_path = opt.nle_path
cckd_emb_size = opt.cckd_emb_size
sigma = opt.sigma
latent_layer = opt.latent_layer
latent_layers = opt.latent_layers


data_val, y_val = load_data_2020(feat_path, val_csv, num_freq_bin, 'logmel')
y_val = keras.utils.to_categorical(y_val, num_classes)

if trans_way == 'nle':
    nle_mat = np.loadtxt(nle_path)
    nle_mat = softmax(nle_mat / tem, axis=1)



def div_tem(x):
    x = x / tem
    return x

# modify model to add temperature parameter and kt output
if trans_way != 'vbkt':
    ori_model = keras.models.load_model(source_model)
elif trans_way == 'vbkt':
    if opt.model == 'resnet':
        ori_model = model_resnet_vbkt(num_classes, input_shape=[num_freq_bin,None,3*num_audio_channels], 
                                    num_filters=24, latent_layer=latent_layer, wd=0)
    elif opt.model == 'fcnn':
        ori_model = model_fcnn_vbkt(num_classes, input_shape=[num_freq_bin,None,3*num_audio_channels], 
                                    num_filters=[48, 96, 192], latent_layer=latent_layer, wd=0)
    ref_model = keras.models.load_model(source_model)
    for layer in ref_model.layers:
        ori_model.get_layer(layer.name).set_weights(layer.get_weights())

ori_model.layers.pop()
x = keras.layers.Lambda(div_tem, name='lambda_T')(ori_model.layers[-1].output)
o_ce = keras.layers.Activation(keras.activations.softmax, name='output_ce')(ori_model.layers[-1].output)
o_ts = keras.layers.Activation(keras.activations.softmax, name='output_ts')(x)
if trans_way == 'tsl':
    model_outputs = [o_ce, o_ts]
elif trans_way == 'nle':
    o_kt = keras.layers.Activation(keras.activations.softmax, name='output_kt')(x)
    model_outputs = [o_ce, o_ts, o_kt]
elif trans_way in ['fitnets', 'sp', 'cckd', 'pkt', 'nst', 'rkd']:
    o_kt = ori_model.get_layer(latent_layer).output
    if trans_way == 'cckd':
        o_kt = keras.layers.Dense(cckd_emb_size, name='output_cckd')(o_kt)
    model_outputs = [o_ce, o_ts, o_kt]
elif trans_way in ['at', 'ab', 'vid', 'cofd']:
    model_outputs = [o_ce, o_ts]
    for i in range(len(latent_layers)):
        model_outputs.append(ori_model.get_layer(latent_layers[i]).output)
elif trans_way == 'fsp':
    model_outputs = [o_ce, o_ts]
    if opt.model == 'resnet':
        # our resnet model is two-path, we need to saparately handle two paths
        for i in range(len(latent_layers) - 1):
            s_pos = i
            e_pos = i + 2
            if i == len(latent_layers) - 2:
                e_pos = i + 1
            o_temp1 = ori_model.get_layer(latent_layers[s_pos]).output
            o_temp2 = ori_model.get_layer(latent_layers[e_pos]).output
            model_outputs.append(keras.layers.Lambda(gram_fsp_resnet, name=('lambda_fsp'+str(i)))([o_temp1, o_temp2]))
    elif opt.model == 'fcnn':
        for i in range(len(latent_layers) - 1):
            o_temp1 = ori_model.get_layer(latent_layers[i]).output
            o_temp2 = ori_model.get_layer(latent_layers[i+1]).output
            model_outputs.append(keras.layers.Lambda(gram_fsp_fcnn, name=('lambda_fsp'+str(i)))([o_temp1, o_temp2]))
elif trans_way == 'vbkt':
    o_mu = ori_model.get_layer(latent_layer).output
    o_log_sigma = ori_model.get_layer('input_2').output
    o_kt = keras.layers.Concatenate(axis=1, name='concatenate_mu_logsigma_'+latent_layer)([o_mu, o_log_sigma])
    model_outputs = [o_ce, o_ts, o_kt]
model = keras.Model(inputs=ori_model.inputs, outputs=model_outputs)


# modify the source model as well
teacher_ori_model = keras.models.load_model(source_model)
teacher_ori_model.layers.pop()
x = keras.layers.Lambda(div_tem, name='lambda_T')(teacher_ori_model.layers[-1].output)
o_ce = keras.layers.Activation(keras.activations.softmax, name='output_ce')(teacher_ori_model.layers[-1].output)
o_ts = keras.layers.Activation(keras.activations.softmax, name='output_ts')(x)
if trans_way == 'tsl':
    teacher_model_outputs = [o_ce, o_ts]
elif trans_way == 'nle':
    o_kt = keras.layers.Activation(keras.activations.softmax, name='output_kt')(x)
    teacher_model_outputs = [o_ce, o_ts, o_kt]
elif trans_way in ['fitnets', 'sp', 'cckd', 'pkt', 'nst', 'rkd', 'vbkt']:
    o_kt = teacher_ori_model.get_layer(latent_layer).output
    if trans_way == 'cckd':
        o_kt = keras.layers.Dense(cckd_emb_size, name='output_cckd')(o_kt)
    teacher_model_outputs = [o_ce, o_ts, o_kt]
elif trans_way in ['at', 'ab', 'vid', 'cofd']:
    teacher_model_outputs = [o_ce, o_ts]
    for i in range(len(latent_layers)):
        teacher_model_outputs.append(teacher_ori_model.get_layer(latent_layers[i]).output)
elif trans_way == 'fsp':
    teacher_model_outputs = [o_ce, o_ts]
    if opt.model == 'resnet':
        for i in range(len(latent_layers) - 1):
            s_pos = i
            e_pos = i + 2
            if i == len(latent_layers) - 2:
                e_pos = i + 1
            o_temp1 = teacher_ori_model.get_layer(latent_layers[s_pos]).output
            o_temp2 = teacher_ori_model.get_layer(latent_layers[e_pos]).output
            teacher_model_outputs.append(keras.layers.Lambda(gram_fsp_resnet, name=('lambda_fsp'+str(i)))([o_temp1, o_temp2]))
    elif opt.model == 'fcnn':
        for i in range(len(latent_layers) - 1):
            o_temp1 = teacher_ori_model.get_layer(latent_layers[i]).output
            o_temp2 = teacher_ori_model.get_layer(latent_layers[i+1]).output
            teacher_model_outputs.append(keras.layers.Lambda(gram_fsp_fcnn, name=('lambda_fsp'+str(i)))([o_temp1, o_temp2]))
teacher_model = keras.Model(inputs=teacher_ori_model.inputs, outputs=teacher_model_outputs)
teacher_model._make_predict_function()

y_val_soft = teacher_model.predict(data_val)[1]
if trans_way == 'nle':
    y_val_kt = y_val.dot(nle_mat)
elif trans_way in ['fitnets', 'sp', 'cckd', 'pkt', 'nst', 'rkd']:
    y_val_kt = teacher_model.predict(data_val)[2]
elif trans_way in ['at', 'ab', 'vid', 'fsp', 'cofd']:
    y_val_kt = teacher_model.predict(data_val)[2:]
elif trans_way == 'vbkt':
    y_val_mu = teacher_model.predict(data_val)[2]
    y_val_sigma = np.ones(y_val_mu.shape) * sigma
    y_val_logsigma = np.log(y_val_sigma)
    y_val_mu_logsigma = np.concatenate([y_val_mu, y_val_logsigma], axis=1)



# initilization stage, do pre-training with only kt loss
# only for ab and fsp as suggested in their papers
if trans_way in ['ab', 'fsp']:
    model_loss_pretrain = {'output_ce': 'categorical_crossentropy', 'output_ts': 'categorical_crossentropy'}
    loss_weights_pretrain = {'output_ce': 0.0, 'output_ts': 0.0}
    if trans_way == 'ab':
        for i in range(len(latent_layers)):
            model_loss_pretrain[latent_layers[i]] = ab_loss
            loss_weights_pretrain[latent_layers[i]] = 1.0 / len(latent_layers)
    elif trans_way == 'fsp':
        for i in range(len(latent_layers) - 1):
            model_loss_pretrain['lambda_fsp'+str(i)] = 'mean_squared_error'
            loss_weights_pretrain['lambda_fsp'+str(i)] = 1.0 / len(latent_layers)

    model.compile(loss=model_loss_pretrain, loss_weights=loss_weights_pretrain, metrics={'output_ce': ['accuracy']},
              optimizer=SGD(lr=max_lr,decay=0, momentum=0.9, nesterov=False))
    model.summary()

    lr_scheduler_pretrain = LR_WarmRestart(nbatch=np.ceil(sample_num/batch_size), Tmult=2,
                              initial_lr=max_lr, min_lr=max_lr*1e-4,
                              epochs_restart = [3.0, 7.0, 15.0, 31.0, 63.0,127.0,255.0,511.0])
    callbacks_pretrain = [lr_scheduler_pretrain]

    train_data_generator_pretrain =  Generator_kt_multilayer_splitted(feat_path, train_csv, train_paired_csv, teacher_model, num_freq_bin, 
                                  batch_size=batch_size, alpha=mixup_alpha, splitted_num=4)()

    validation_data = (data_val, [y_val, y_val_soft] + y_val_kt)
    history_pretrain = model.fit_generator(train_data_generator_pretrain, 
                              validation_data=validation_data,
                              epochs=num_epochs_pretrain, 
                              verbose=1, 
                              workers=4,
                              max_queue_size=100,
                              callbacks=callbacks_pretrain,
                              steps_per_epoch=np.ceil(sample_num/batch_size)
                              ) 

# training stage
if trans_way == 'tsl':
    kt_loss = 'categorical_crossentropy'
elif trans_way == 'nle':
    kt_loss = 'categorical_crossentropy'
elif trans_way == 'fitnets':
    kt_loss = 'mean_squared_error'
elif trans_way == 'at':
    kt_loss = at_loss
elif trans_way == 'ab':
    kt_loss = ab_loss
elif trans_way == 'vid':
    kt_loss = vid_loss
elif trans_way == 'fsp':
    kt_loss = 'mean_squared_error'
elif trans_way == 'cofd':
    kt_loss = cofd_loss
elif trans_way == 'sp':
    kt_loss = sp_loss
elif trans_way == 'cckd':
    kt_loss = cckd_loss
elif trans_way == 'pkt':
    kt_loss = pkt_loss
elif trans_way == 'nst':
    kt_loss = nst_loss
elif trans_way == 'rkd':
    def rkd_loss(target, inputs):
        loss = alpha * biloss(inputs, target) + beta * triloss(inputs, target)
        return loss
    kt_loss = rkd_loss
elif trans_way == 'vbkt':
    kt_loss = vbkt_loss


if trans_way == 'tsl':
    model_loss = {'output_ce': 'categorical_crossentropy', 'output_ts': kt_loss}
    loss_weights = {'output_ce': lmd, 'output_ts': soft_ratio}
elif trans_way == 'nle':
    model_loss = {'output_ce': 'categorical_crossentropy', 'output_ts': 'categorical_crossentropy', 'output_kt': kt_loss}
    loss_weights = {'output_ce': lmd, 'output_ts': soft_ratio, 'output_kt':alpha}
elif trans_way in ['fitnets', 'sp', 'pkt', 'nst', 'rkd']:
    model_loss = {'output_ce': 'categorical_crossentropy', 'output_ts': 'categorical_crossentropy', latent_layer: kt_loss}
    loss_weights = {'output_ce': lmd, 'output_ts': soft_ratio, latent_layer: alpha}
elif trans_way == 'cckd':
    model_loss = {'output_ce': 'categorical_crossentropy', 'output_ts': 'categorical_crossentropy', 'output_cckd': kt_loss}
    loss_weights = {'output_ce': lmd, 'output_ts': soft_ratio, 'output_cckd': alpha}
elif trans_way in ['at', 'ab', 'vid', 'cofd']:
    model_loss = {'output_ce': 'categorical_crossentropy', 'output_ts': 'categorical_crossentropy'}
    loss_weights = {'output_ce': lmd, 'output_ts': soft_ratio}
    for i in range(len(latent_layers)):
        model_loss[latent_layers[i]] = kt_loss
        loss_weights[latent_layers[i]] = alpha / len(latent_layers)
elif trans_way == 'fsp':
    model_loss = {'output_ce': 'categorical_crossentropy', 'output_ts': 'categorical_crossentropy'}
    loss_weights = {'output_ce': lmd, 'output_ts': soft_ratio}
    for i in range(len(latent_layers) - 1):
        model_loss['lambda_fsp'+str(i)] = kt_loss
        loss_weights['lambda_fsp'+str(i)] = 1.0 / len(latent_layers)
elif trans_way == 'vbkt':
    model_loss = {'output_ce': 'categorical_crossentropy', 'output_ts': 'categorical_crossentropy', 'concatenate_mu_logsigma_'+latent_layer: vbkt_loss}
    loss_weights = {'output_ce': lmd, 'output_ts': soft_ratio, 'concatenate_mu_logsigma_'+latent_layer: alpha}


model.compile(loss=model_loss, loss_weights=loss_weights, metrics={'output_ce': ['accuracy']},  
              optimizer=SGD(lr=max_lr,decay=0, momentum=0.9, nesterov=False),)
model.summary()

lr_scheduler = LR_WarmRestart(nbatch=np.ceil(sample_num/batch_size), Tmult=2,
                              initial_lr=max_lr, min_lr=max_lr*1e-4,
                              epochs_restart = [3.0, 7.0, 15.0, 31.0, 63.0,127.0,255.0,511.0]) 

callbacks = [lr_scheduler]

if trans_way == 'tsl':
    train_data_generator = Generator_tslearning_splitted(feat_path, train_csv, train_paired_csv, teacher_model, num_freq_bin,
                              batch_size=batch_size, alpha=mixup_alpha, splitted_num=4)()
elif trans_way == 'nle':
    train_data_generator = Generator_nle_splitted(feat_path, train_csv, train_paired_csv, teacher_model, nle_mat, num_freq_bin,
                              batch_size=batch_size, alpha=mixup_alpha, splitted_num=4)()
elif trans_way in ['fitnets', 'sp', 'cckd', 'pkt', 'nst', 'rkd']:
    train_data_generator = Generator_kt_singlelayer_splitted(feat_path, train_csv, train_paired_csv, teacher_model, num_freq_bin, 
                              batch_size=batch_size, alpha=mixup_alpha, splitted_num=4)()
elif trans_way in ['at', 'ab', 'vid', 'fsp', 'cofd']:
    train_data_generator = Generator_kt_multilayer_splitted(feat_path, train_csv, train_paired_csv, teacher_model, num_freq_bin,
                              batch_size=batch_size, alpha=mixup_alpha, splitted_num=4)()
elif trans_way == 'vbkt':
    train_data_generator = Generator_vbkt_splitted(feat_path, train_csv, train_paired_csv, teacher_model, sigma, num_freq_bin,
                              batch_size=batch_size, alpha=mixup_alpha, splitted_num=4)()


if trans_way == 'tsl':
    validation_data = (data_val, [y_val, y_val_soft])
elif trans_way in ['fitnets', 'nle', 'sp', 'cckd', 'pkt', 'nst', 'rkd']:
    validation_data = (data_val, [y_val, y_val_soft, y_val_kt])
elif trans_way in ['at', 'ab', 'vid', 'fsp', 'cofd']:
    validation_data = (data_val, [y_val, y_val_soft] + y_val_kt)
elif trans_way == 'vbkt':
    validation_data = ([data_val, y_val_logsigma], [y_val, y_val_soft, y_val_mu_logsigma])


history = model.fit_generator(train_data_generator,
                              validation_data=validation_data,
                              epochs=num_epochs, 
                              verbose=1, 
                              workers=4,
                              max_queue_size=100,
                              callbacks=callbacks,
                              steps_per_epoch=np.ceil(sample_num/batch_size)
                              ) 
if trans_way != 'vbkt':
    model.save(experiments + "/model-epoch" + str(num_epochs) + "-seed" + str(seed) + ".hdf5")
if trans_way == 'vbkt':
    # save model parameter into model without sampling layer, for testing in no-sampling case
    # due to we import extra input parameters by lambda layer, keras can not directly load the model
    model_nosampling = keras.models.load_model(source_model)
    for layer in model_nosampling.layers:
        if layer.name != 'activation_11' and layer.name != 'activation_34':
            weights = model.get_layer(layer.name).get_weights()
            model_nosampling.get_layer(layer.name).set_weights(weights)
    model_nosampling.save(experiments + "/model-epoch" + str(num_epochs) + "-seed" + str(seed)+ ".hdf5")




