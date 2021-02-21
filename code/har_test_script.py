# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 10:18:45 2020

@author: Joshua
"""

import har_v2_f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy import signal
import scipy.stats
import tsfresh
import sklearn
from collections import defaultdict, Counter
import pywt
from sklearn.neural_network import MLPClassifier

import tensorflow as tf
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
import random as rn

from tensorflow.python.eager import context

context._context = None
context._create_context()

np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

raw_gyro_data = har_v2_f.read_raw_gyro_data("../datasets/uci_har/v2/RawData/")
raw_acc_data = har_v2_f.read_raw_acc_data("../datasets/uci_har/v2/RawData/")


gyro_data_denoised = har_v2_f.denoise_har_raw_data(raw_gyro_data)
acc_data_denoised = har_v2_f.denoise_har_raw_data(raw_acc_data)


# full dataset
har_data, har_labels, har_pers_label = har_v2_f.transform_har_data(acc_data_denoised, gyro_data_denoised)

# split by har_pers_label: the first 41 to train, last 10 to val, next 10 to test
# hard coded so one experiment can't be in train/val/test set at once
# should be the same regardless of the preprocessing done as we split by experiment
train_group = 0
val_group = 0

for i in range(len(har_pers_label)):
    if har_pers_label[i] < 42:
        train_group = i
    elif har_pers_label[i] < 52:
        val_group = i

har_train_data = har_data[:train_group+1, :, :]
har_train_labels = har_labels[:train_group+1]

har_val_data = har_data[(train_group+1):(val_group+1), :, :]
har_val_labels = har_labels[(train_group+1):(val_group+1)]

har_test_data = har_data[(val_group+1):, :, :]
har_test_labels = har_labels[(val_group+1):]

#-----------------------------------------------------------------------------
# cwt transform only for 2D CNNs
x_train, y_train, x_val, y_val, x_test, y_test = har_v2_f.cwt_transform(har_train_data,
                                                               har_train_labels,
                                                               har_val_data,
                                                               har_val_labels,
                                                               har_test_data,
                                                               har_test_labels,
                                                               waveletname = "morl",
                                                               train_size = har_train_data.shape[0],
                                                               val_size = har_val_data.shape[0],
                                                               test_size = har_test_data.shape[0])
# further image squeezing, if necessary, comes here



# model building
img_x = x_train.shape[1]
img_y = x_train.shape[2]
img_z = x_train.shape[3]
input_shape = (img_x, img_y, img_z)

num_classes = 12
batch_size = 32
epochs = 10

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


data_path = "./uci_har_cnn_models/processed_data/" + "db4_0.3_128morl_128" + ".npz"
saved_data_file = np.load(data_path)
x_train = saved_data_file["x_train"]
y_train = saved_data_file["y_train"]
x_val = saved_data_file["x_val"]
y_val = saved_data_file["y_val"]
x_test = saved_data_file["x_test"]
y_test = saved_data_file["y_test"]

np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())


trained_model, cnn_hist = har_v2_f.compile_and_fit(model, x_train, y_train, x_val, y_val, x_test, y_test, batch_size, epochs, model_name="db4_morl_128_128")


# load best model and then evaluate
best_model = tf.keras.models.load_model("./uci_har_models/db4_morl_128_128.h5")

train_score = best_model.evaluate(x_train, y_train, verbose=0)
print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
test_score = best_model.evaluate(x_val, y_val, verbose=0)
print('Val loss: {}, Val accuracy: {}'.format(test_score[0], test_score[1]))
test_score = best_model.evaluate(x_test, y_test, verbose=0)
print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))

har_v2_f.plot_hist(cnn_hist)


#------------------------------------------------------------------------------
# 1D cnn and rnn transform with subband starts here
# dwt subband function here
x_train, y_train, x_val, y_val, x_test, y_test = har_v2_f.subband_dwt_transform(har_train_data,
                                                                       har_train_labels,
                                                                       har_val_data,
                                                                       har_val_labels,
                                                                       har_test_data,
                                                                       har_test_labels,
                                                                       subband_wavelet = "db4")

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#------------------------------------------------------------------------------
# feature extraction
x_train, y_train, x_val, y_val, x_test, y_test = har_v2_f.har_feature_extration_with_subband(har_train_data,
                                                                       har_train_labels,
                                                                       har_val_data,
                                                                       har_val_labels,
                                                                       har_test_data,
                                                                       har_test_labels,
                                                                       subband_wavelet = "none")

trained_model = har_v2_f.compile_and_fit_feat(x_train, y_train, x_val,
                                             y_val, x_test, y_test,
                                             model_type="gbc",
                                             model_name="test",
                                             data_name="test")

trained_model.score(x_test, y_test)


clf = sklearn.svm.SVC(decision_function_shape= "ovr", break_ties = True)
clf.fit(x_train, y_train)
clf.score(x_test, y_test)

pos_set = tsfresh.feature_extraction.ComprehensiveFCParameters()


test_list = [1, 2, 3, 4, 5,6,7,8,9,10,11,12]
test_list = [i -1 for i in test_list]
test_list = keras.utils.to_categorical(test_list, 12)

#------------------------------------------------------------------------------
# trial if original personal label based split can be done with split by experiment
# split by person, train <= 20, < val <= 25, < test
def read_and_split_paper_features(path):
    feat_x_train =  np.loadtxt(path + "Train/X_train.txt")
    feat_y_train = np.loadtxt(path + "Train/y_train.txt")
    feat_sub_id_train = np.loadtxt(path + "Train/subject_id_train.txt")
    
    feat_x_test =  np.loadtxt(path+"Test/X_test.txt")
    feat_y_test = np.loadtxt(path+"Test/y_test.txt")
    feat_sub_id_test = np.loadtxt(path+"Test/subject_id_test.txt")
    
    train_all = np.column_stack((feat_sub_id_train,feat_y_train))
    train_all = np.concatenate((train_all, feat_x_train), axis = 1)

    test_all = np.column_stack((feat_sub_id_test,feat_y_test))
    test_all = np.concatenate((test_all, feat_x_test), axis = 1)
    
    # all data in one array and sorted by personal number
    har_all = np.concatenate((train_all,test_all),axis = 0)
    har_all = har_all[har_all[:,0].argsort()]
    
    train_group = 0
    val_group = 0
    for i in range(len(har_all)):
        if har_all[i,0] < 21:
            train_group = i
        elif har_all[i,0] < 26:
            val_group = i
    
    # split based on personal number
    train = har_all[:train_group+1, :] 
    val = har_all[(train_group+1):(val_group+1), :]
    test = har_all[(val_group+1):, :]
    
    # first two columns are id and label, rest goes into x_*
    x_train = train[:,2:]
    y_train = train[:,1]
    
    x_val = val[:,2:]
    y_val = val[:,1]
    
    x_test = test[:,2:]
    y_test = test[:,1]
    
    # alles in ein array packen, sortieren nach subject nummer, danach dann split basierend auf der sub. nummer

    return x_train, y_train, x_val, y_val, x_test, y_test

x_train, y_train, x_val, y_val, x_test, y_test = read_and_split_paper_features("../datasets/uci_har/v2/")