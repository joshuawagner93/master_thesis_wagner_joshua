# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 11:40:44 2020

@author: Joshua
trials with the v2 raw version of uci har data
"""

import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy import signal
from scipy import ndimage
import os.path
import cv2
import tsfresh
import sklearn
from joblib import dump, load
import scipy.stats
from collections import defaultdict, Counter

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# import keras
import keras
from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import GRU, LSTM, Bidirectional
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import random as rn
import tensorflow as tf
from tensorflow.python.eager import context




def read_raw_gyro_data(path):
    """
    input:
        path : path to the directory with the raw har v2 data
    
    returns:
        the raw gyro data from the har v2 dataset as list of dicts with each dict containing one experiment
    """
    gyro_raw_data = []
    for i in range(1,62):
        if i < 10:
            in_path = path + "gyro_exp0" + str(i) + "_*"
        else:
            in_path = path + "gyro_exp" + str(i) + "_*"
        new_data = np.loadtxt(glob.glob(in_path)[0])
        gyro_raw_data.append(new_data)
    
    return gyro_raw_data

def read_raw_acc_data(path):
    """
    equivalent to read_raw_gyro_data() for acc data 
    """
    acc_raw_data = []
    for i in range(1,62):
        if i < 10:
            in_path = path + "acc_exp0" + str(i) + "_*"
        else:
            in_path = path + "acc_exp" + str(i) + "_*"
        new_data = np.loadtxt(glob.glob(in_path)[0])
        acc_raw_data.append(new_data)
    
    return acc_raw_data


# DWT noise filtering step, performed before the splitting of the dataset
# as we split by experiment and thus don't have any overlap between train/val/test

def dwt_lowpassfilter(signal_in, thresh = 0.4, wavelet="db4"):
    thresh = thresh*np.nanmax(signal_in)
    coeff = pywt.wavedec(signal_in, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per")
    return reconstructed_signal

def denoise_har_raw_data(data, thresh = 0.3, wavelet="db4"):
    output = []
    if wavelet != "median":
        even_check = 0
        for i in range(len(data)):
            temp_data = data[i]
            if (temp_data.shape[0] % 2) > 0: # neccessary because dwt somehow only works on even number of obs
                temp_data = np.append(temp_data, [temp_data[-1,:]], axis = 0)
                even_check = 1
            else:
                even_check = 0
            output_arr = np.ones(shape=temp_data.shape)
            for j in range(temp_data.shape[1]):
                output_arr[:, j] = dwt_lowpassfilter(temp_data[:, j], thresh = thresh, wavelet = wavelet)
            if even_check == 1:
                output_arr = output_arr[:-1,:]
            output.append(output_arr)
    else:
        for i in range(len(data)):
            temp_data = data[i]
            output_arr = np.ones(shape=temp_data.shape)
            for j in range(temp_data.shape[1]):
                output_arr[:,j] = ndimage.median_filter(temp_data[:,j], size = 3) # third order median filter as in the original paper: https://upcommons.upc.edu/bitstream/handle/2117/79600/online_har.pdf
            output.append(output_arr)
    return output



def transform_har_data(acc_data, gyro_data, window_size = 128, step_size = 64):
    """
    transformation of denoised har data into windows and 
    (no conversions to images, these can be done from the resulting windowed ts)
    
    output:
        output : 3-dim array with number of windows as first dim, window_size as second dim and number of features as third dim (here 9)
    """
    
    labels_raw_data = np.loadtxt("../datasets/uci_har/v2/RawData/labels.txt", dtype = int)
    
    total_acc_x = np.zeros([1,window_size])
    total_acc_y = np.zeros([1,window_size])
    total_acc_z = np.zeros([1,window_size])
    gyro_acc_x = np.zeros([1,window_size])
    gyro_acc_y = np.zeros([1,window_size])
    gyro_acc_z = np.zeros([1,window_size])
    body_acc_x = np.zeros([1,window_size])
    body_acc_y = np.zeros([1,window_size])
    body_acc_z = np.zeros([1,window_size])
    y_labels = []
    person_label = []
    
    # butterworth filter with 0.3 Hz cutoff as described in the readme of v2 of the uci har data
    # the readme falsly claims a low-pass filter here, the paper for the dataset
    # uses a high-pass which is also used here, see:
    # https://upcommons.upc.edu/bitstream/handle/2117/79600/online_har.pdf
    butter_filter = signal.butter(10, 0.3, 'highpass', fs = 50, output='sos') # 0.3 Hz cutoff for gravitational acc, fs=50 is the sampling rate of the sensors at 50Hz
    
    # needs a rewrite to work with the split data format
    # 
    for i in range(labels_raw_data.shape[0]):
        temp_gyro_data = gyro_data[(labels_raw_data[i,0]-1)][labels_raw_data[i,3]:labels_raw_data[i,4],]
        temp_acc_data = acc_data[(labels_raw_data[i,0]-1)][labels_raw_data[i,3]:labels_raw_data[i,4],]
        
        times_outseries = (len(temp_gyro_data)-window_size)//step_size + 1 # to adjust for range() counting
        if times_outseries > 0 and len(temp_gyro_data) > 128: # to check if our window can even fit into the timeseries, gyro and acc have the same length
            for j in range(times_outseries): # for each posture block from labels_raw_data, go through and select the gyro and acc data and window it j times
                gyro_acc_x = np.concatenate((gyro_acc_x, np.array([temp_gyro_data[(j*step_size):(window_size + j * step_size), 0]])), axis = 0)
                gyro_acc_y = np.concatenate((gyro_acc_y, np.array([temp_gyro_data[(j*step_size):(window_size + j * step_size), 1]])), axis = 0)
                gyro_acc_z = np.concatenate((gyro_acc_z, np.array([temp_gyro_data[(j*step_size):(window_size + j * step_size), 2]])), axis = 0)
                y_labels.append(labels_raw_data[i,2])
                person_label.append(labels_raw_data[i,0])
                # acc data
                total_acc_x = np.concatenate((total_acc_x, np.array([temp_acc_data[(j*step_size):(window_size + j * step_size), 0]])), axis = 0)
                total_acc_y = np.concatenate((total_acc_y, np.array([temp_acc_data[(j*step_size):(window_size + j * step_size), 1]])), axis = 0)
                total_acc_z = np.concatenate((total_acc_z, np.array([temp_acc_data[(j*step_size):(window_size + j * step_size), 2]])), axis = 0)
                
                filtered_acc_x = signal.sosfilt(butter_filter, temp_acc_data[(j*step_size):(window_size + j * step_size), 0])
                filtered_acc_y = signal.sosfilt(butter_filter, temp_acc_data[(j*step_size):(window_size + j * step_size), 1])
                filtered_acc_z = signal.sosfilt(butter_filter, temp_acc_data[(j*step_size):(window_size + j * step_size), 2])
                
                body_acc_x = np.concatenate((body_acc_x, np.array([filtered_acc_x])), axis = 0)
                body_acc_y = np.concatenate((body_acc_y, np.array([filtered_acc_y])), axis = 0)
                body_acc_z = np.concatenate((body_acc_z, np.array([filtered_acc_z])), axis = 0)
                
    # the readme from the dataset is misleading as it uses total as description
    # reading the coresponding paper reveales that the body acc is substracted
    # from the total to extract the gravitational component and this is then used, see:
    # https://upcommons.upc.edu/bitstream/handle/2117/79600/online_har.pdf
    grav_acc_x = np.subtract(total_acc_x, body_acc_x)
    grav_acc_y = np.subtract(total_acc_y, body_acc_y)
    grav_acc_z = np.subtract(total_acc_z, body_acc_z)
    
    output = body_acc_x.copy()
    output = output[..., np.newaxis]
    
    output = np.concatenate((output,
                             body_acc_y[...,np.newaxis],
                             body_acc_z[...,np.newaxis],
                             gyro_acc_x[...,np.newaxis],
                             gyro_acc_y[...,np.newaxis],
                             gyro_acc_z[...,np.newaxis],
                             grav_acc_x[...,np.newaxis],
                             grav_acc_y[...,np.newaxis],
                             grav_acc_z[...,np.newaxis]), axis = 2)
    
    output = np.delete(output, 0, 0)
    
    return output, y_labels, person_label
    

# preprocessing with cwt which transforms ts 1d data into 2d data, 
def cwt_transform(x_train, y_train, x_val, y_val, x_test, y_test, waveletname = "morl", max_scale = 128, train_size = 5000, val_size = 500, test_size = 500):
    scales = range(1, (max_scale+1))
    waveletname = waveletname
    train_size = train_size
    test_size= test_size
    
    train_data_cwt = np.ndarray(shape=(train_size, max_scale, x_train.shape[1], x_train.shape[2]))
    for ii in range(0, train_size):
        if ii % 1000 == 0:
            print(ii)
        for jj in range(0, 9):
            signal = x_train[ii, :, jj]
            coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
            train_data_cwt[ii, :, :, jj] = coeff
            
    val_data_cwt = np.ndarray(shape=(val_size, max_scale, x_val.shape[1],  x_val.shape[2]))
    for ii in range(0, val_size):
        if ii % 500 == 0:
            print(ii)
        for jj in range(0, 9):
            signal = x_val[ii, :, jj]
            coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
            val_data_cwt[ii, :, :, jj] = coeff
            
    
    test_data_cwt = np.ndarray(shape=(test_size, max_scale, x_test.shape[1],  x_test.shape[2]))
    for ii in range(0, test_size):
        if ii % 500 == 0:
            print(ii)
        for jj in range(0, 9):
            signal = x_test[ii, :, jj]
            coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
            test_data_cwt[ii, :, :, jj] = coeff
    
    uci_har_labels_train = list(map(lambda x: int(x) - 1, y_train))
    uci_har_labels_val = list(map(lambda x: int(x) - 1, y_val))
    uci_har_labels_test = list(map(lambda x: int(x) - 1, y_test))
    
    x_train = train_data_cwt
    y_train = list(uci_har_labels_train[:train_size])
    x_val = val_data_cwt
    y_val = list(uci_har_labels_val[:val_size])
    x_test = test_data_cwt
    y_test = list(uci_har_labels_test[:test_size])
    
    return x_train, y_train, x_val, y_val, x_test, y_test



# model_name in the schema of: noise-preprocessing, wavelet-name, scale, img_y_size
def compile_and_fit(model, X_train, y_train, X_val, y_val, X_test, y_test, batch_size, n_epochs, model_name):
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['categorical_accuracy'])
    
    print("compiled")
    path = "./uci_har_cwt_cnn_models/" + model_name + '.h5'
    logging_path = "./uci_har_cwt_cnn_models/logs/" + model_name + ".log"
    # define callback for model saving
    checkpoint_callback = ModelCheckpoint(filepath= path, monitor='val_categorical_accuracy', save_best_only=True)
    csv_logger_callback = keras.callbacks.CSVLogger(filename=logging_path)

    history  = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              verbose=1,
              validation_data=(X_val, y_val),
              callbacks = [checkpoint_callback, csv_logger_callback],
              shuffle = True)
    # add csv saving for x/y_train, x/y_val and x/y_test (build new folder for the data with the modelname)
    data_path = "./uci_har_cwt_cnn_models/processed_data/" + model_name
    np.savez(data_path, x_train = X_train, y_train = y_train, x_val = X_val, y_val = y_val, x_test = X_test, y_test = y_test)


    return model, history
    

def plot_hist(history):
    plt.title('Accuracy of training and test sets')
    
    plt.subplot(1,2,1)
    plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='Val Accuracy')
    plt.ylabel('categorical_accuracy')
    plt.xlabel('No. epoch')
    
    plt.subplot(1,2,2)
    plt.xlim(0,14)
    plt.ylim(0,1)
    plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='Val Accuracy')
    plt.ylabel('categorical_accuracy')
    plt.xlabel('No. epoch')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()
    

def resize_image(image, target):
   return cv2.resize(image, dsize=(target[0], target[1]), interpolation=cv2.INTER_AREA)



def cnn_variations(acc_data_path, gyro_data_path, denoise_thresh, denoise_dwt_wavelet, window_size, cwt_wavelet, cwt_scale):
    
    # randomness reduction and allowing reproduciblity, though the reduction to 1 thread could introduce time constraints
    # as found in https://github.com/keras-team/keras/issues/2280#issuecomment-492826113
    context._context = None
    context._create_context()

    np.random.seed(42)
    rn.seed(42)
    tf.random.set_seed(42)
    #tf.config.threading.set_intra_op_parallelism_threads(1)
    #tf.config.threading.set_inter_op_parallelism_threads(1)
    
    # name of the files etc is now in the format: dwt-wavelet_dwt-threshold_window-size_cwt-wavelet_cwt-scale
    model_path_name = denoise_dwt_wavelet + "_" + str(denoise_thresh) + "_" + str(window_size) + cwt_wavelet + "_" + str(cwt_scale)
    num_classes = 12
    # check if that specific combination already exists, if true reads the already preprocessed data
    # allows multiple models to be trained on the same preprocessed data
    data_path = "./uci_har_cwt_cnn_models/processed_data/" + model_path_name + ".npz"
    if os.path.exists(data_path):
        print("preprocessed data already exists, reading from file")
        saved_data_file = np.load(data_path)
        x_train = saved_data_file["x_train"]
        y_train = saved_data_file["y_train"]
        x_val = saved_data_file["x_val"]
        y_val = saved_data_file["y_val"]
        x_test = saved_data_file["x_test"]
        y_test = saved_data_file["y_test"]
    
    else:
        print("no preprocessed data found, begin preprocessing...")
        raw_acc_data = read_raw_acc_data(acc_data_path)
        raw_gyro_data = read_raw_gyro_data(gyro_data_path)
        
        print("denoising with " + denoise_dwt_wavelet + " and a thresh. of: " + str(denoise_thresh))
        if denoise_dwt_wavelet != "noisy":
            gyro_data_denoised = denoise_har_raw_data(raw_gyro_data, thresh=denoise_thresh, wavelet=denoise_dwt_wavelet)
            acc_data_denoised = denoise_har_raw_data(raw_acc_data, thresh=denoise_thresh, wavelet=denoise_dwt_wavelet)
        else:
            gyro_data_denoised = raw_gyro_data
            acc_data_denoised = raw_acc_data
        print("windowing data into windows of size: " + str(window_size) + " with 50% overlap")
        
        har_data, har_labels, har_pers_label = transform_har_data(acc_data_denoised, gyro_data_denoised)
        
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
        
        print("cwt transform in progress...")
        x_train, y_train, x_val, y_val, x_test, y_test = cwt_transform(har_train_data,
                                                                   har_train_labels,
                                                                   har_val_data,
                                                                   har_val_labels,
                                                                   har_test_data,
                                                                   har_test_labels,
                                                                   waveletname = cwt_wavelet,
                                                                   max_scale=cwt_scale,
                                                                   train_size = har_train_data.shape[0],
                                                                   val_size = har_val_data.shape[0],
                                                                   test_size = har_test_data.shape[0])

        x_train = x_train.astype('float32')
        x_val = x_val.astype('float32')
        x_test = x_test.astype('float32')

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_val = keras.utils.to_categorical(y_val, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)


    if x_train.shape[1] != x_train.shape[2]: # unequal x- and y-dim
            print("unequally sized input: " + str(x_train.shape))
            print("starting shrinking of image along the x_axis")
            target_shape = (cwt_scale, cwt_scale)
            
            image_x_train = [resize_image(image=i, target=target_shape) for i in x_train]
            x_train = np.stack(image_x_train, axis=0)
            
            image_x_val = [resize_image(image=i, target=target_shape) for i in x_val]
            x_val = np.stack(image_x_val, axis=0)
            
            image_x_test = [resize_image(image=i, target=target_shape) for i in x_test]
            x_test = np.stack(image_x_test, axis=0)
    
    
    
    
    # until here is data preprocessing, check if the file exists and if possible read the data
    # model building
    img_x = x_train.shape[1]
    img_y = x_train.shape[2]
    img_z = x_train.shape[3]
    input_shape = (img_x, img_y, img_z)
    
    batch_size = 32
    epochs = 15
    
    # seeds so every new model is somewhat reproducible
    # see https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
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

    trained_model, cnn_hist = compile_and_fit(model, x_train, y_train, x_val, y_val, x_test, y_test, batch_size, epochs, model_name=model_path_name)
    
    plot_hist(cnn_hist)
    
    # load best model and then evaluate
    loading_model_path = "./uci_har_cwt_cnn_models/" + model_path_name + ".h5"
    best_model = keras.models.load_model(loading_model_path)
    
    train_score = best_model.evaluate(x_train, y_train, verbose=0)
    print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
    val_score = best_model.evaluate(x_val, y_val, verbose=0)
    print('Val loss: {}, Val accuracy: {}'.format(val_score[0], val_score[1]))
    test_score = best_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))
    
    result_dataframe = pd.DataFrame(np.array([train_score, val_score, test_score]), index = ["train", "val", "test"],columns = ["loss", "acc"])
    result_path = "./uci_har_cwt_cnn_models/results/" + model_path_name + ".csv"
    result_dataframe.to_csv(result_path)
    
    print("saved results for the best version of model: " + model_path_name)
    keras.backend.clear_session()

def compile_and_fit_ts(model, X_train, y_train, X_val, y_val, X_test, y_test, batch_size, n_epochs, model_name, data_name):
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['categorical_accuracy'])
    
    print("compiled")
    path = "./uci_har_ts_models/" + model_name + '.h5'
    logging_path = "./uci_har_ts_models/logs/" + model_name + ".log"
    # define callback for model saving
    checkpoint_callback = ModelCheckpoint(filepath= path, monitor='val_categorical_accuracy', save_best_only=True)
    csv_logger_callback = keras.callbacks.CSVLogger(filename=logging_path)

    history  = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              verbose=1,
              validation_data=(X_val, y_val),
              callbacks = [checkpoint_callback, csv_logger_callback],
              shuffle = True)
    # add csv saving for x/y_train, x/y_val and x/y_test (build new folder for the data with the modelname)
    data_path = "./uci_har_ts_models/processed_data/" + data_name
    np.savez(data_path, x_train = X_train, y_train = y_train, x_val = X_val, y_val = y_val, x_test = X_test, y_test = y_test)


    return model, history

def subband_dwt_transform(x_train, y_train, x_val, y_val, x_test, y_test, subband_wavelet):
    if subband_wavelet != "none":
        print("dwt subband splitting in progress...")
        train_data_dwt = np.ndarray(shape=(x_train.shape[0], x_train.shape[1], x_train.shape[2]))
        for ii in range(x_train.shape[0]):
            for jj in range(x_train.shape[2]):
                signal = x_train[ii, :, jj]
                coeff = pywt.wavedec(signal, subband_wavelet, mode="per" )
                train_data_dwt[ii, :, jj] = np.transpose(np.concatenate(coeff, axis = 0)[...,np.newaxis])

                
        val_data_dwt = np.ndarray(shape=(x_val.shape[0], x_val.shape[1], x_val.shape[2]))
        for ii in range(x_val.shape[0]):
            for jj in range(x_val.shape[2]):
                signal = x_val[ii, :, jj]
                coeff = pywt.wavedec(signal, subband_wavelet, mode="per" )
                val_data_dwt[ii, :, jj] = np.transpose(np.concatenate(coeff, axis = 0)[...,np.newaxis])
                
        
        test_data_dwt = np.ndarray(shape=(x_test.shape[0], x_test.shape[1],  x_test.shape[2]))
        for ii in range(x_test.shape[0]):
            for jj in range(x_test.shape[2]):
                signal = x_test[ii, :, jj]
                coeff = pywt.wavedec(signal, subband_wavelet, mode="per" )
                test_data_dwt[ii, :, jj] = np.transpose(np.concatenate(coeff, axis = 0)[...,np.newaxis])
    else:
        print("no subband analysis chosen, proceeding with only denoised data")
        train_data_dwt = x_train
        val_data_dwt = x_val
        test_data_dwt = x_test
        
    # check data transform here
    uci_har_labels_train = list(map(lambda x: int(x) - 1, y_train))
    uci_har_labels_val = list(map(lambda x: int(x) - 1, y_val))
    uci_har_labels_test = list(map(lambda x: int(x) - 1, y_test))
    
    x_train = train_data_dwt
    y_train = list(uci_har_labels_train)
    x_val = val_data_dwt
    y_val = list(uci_har_labels_val)
    x_test = test_data_dwt
    y_test = list(uci_har_labels_test)
    return x_train, y_train, x_val, y_val, x_test, y_test
        
        

        
        
    

def pure_ts_variations(acc_data_path, gyro_data_path, denoise_thresh, denoise_dwt_wavelet, window_size, subband_dwt_wavelet, model_type):
    
    # randomness reduction and allowing reproduciblity, though the reduction to 1 thread could introduce time constraints
    # as found in https://github.com/keras-team/keras/issues/2280#issuecomment-492826113

    np.random.seed(42)
    rn.seed(42)
    tf.random.set_seed(42)
    #tf.config.threading.set_intra_op_parallelism_threads(1)
    #tf.config.threading.set_inter_op_parallelism_threads(1)
    
    # name of the files etc is now in the format: dwt-wavelet_dwt-threshold_window-size_cwt-wavelet_cwt-scale
    model_path_name = denoise_dwt_wavelet + "_" + str(denoise_thresh) + "_" + str(window_size) + subband_dwt_wavelet + "_" + model_type
    data_path_name = denoise_dwt_wavelet + "_" + str(denoise_thresh) + "_" + str(window_size) + subband_dwt_wavelet
    num_classes = 12
    # check if that specific combination already exists, if true reads the already preprocessed data
    # allows multiple models to be trained on the same preprocessed data
    data_path = "./uci_har_ts_models/processed_data/" + data_path_name + ".npz"
    if os.path.exists(data_path):
        print("preprocessed data already exists, reading from file")
        saved_data_file = np.load(data_path)
        x_train = saved_data_file["x_train"]
        y_train = saved_data_file["y_train"]
        x_val = saved_data_file["x_val"]
        y_val = saved_data_file["y_val"]
        x_test = saved_data_file["x_test"]
        y_test = saved_data_file["y_test"]
    
    else:
        print("no preprocessed data found, begin preprocessing...")
        raw_acc_data = read_raw_acc_data(acc_data_path)
        raw_gyro_data = read_raw_gyro_data(gyro_data_path)
        
        print("denoising with " + denoise_dwt_wavelet + " and a thresh. of: " + str(denoise_thresh))
        if denoise_dwt_wavelet != "noisy":
            gyro_data_denoised = denoise_har_raw_data(raw_gyro_data, thresh=denoise_thresh, wavelet=denoise_dwt_wavelet)
            acc_data_denoised = denoise_har_raw_data(raw_acc_data, thresh=denoise_thresh, wavelet=denoise_dwt_wavelet)
        else:
            gyro_data_denoised = raw_gyro_data
            acc_data_denoised = raw_acc_data
        print("windowing data into windows of size: " + str(window_size) + " with 50% overlap")
        
        har_data, har_labels, har_pers_label = transform_har_data(acc_data_denoised, gyro_data_denoised)
        
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
        
        # dwt subband function here
        x_train, y_train, x_val, y_val, x_test, y_test = subband_dwt_transform(har_train_data,
                                                                               har_train_labels,
                                                                               har_val_data,
                                                                               har_val_labels,
                                                                               har_test_data,
                                                                               har_test_labels,
                                                                               subband_wavelet = subband_dwt_wavelet)

        x_train = x_train.astype('float32')
        x_val = x_val.astype('float32')
        x_test = x_test.astype('float32')

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_val = keras.utils.to_categorical(y_val, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

    
    
    
    
    # until here is data preprocessing
    # model building
    input_shape = (x_train.shape[1], x_train.shape[2])
    batch_size = 32
    epochs = 15

    # seeds so every new model is somewhat reproducible
    # see https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    if model_type == "rnn":
        print("rnn chosen")
        # seeds in elif so both are initialised with the same seed
        np.random.seed(42)
        rn.seed(42)
        tf.random.set_seed(42)

        model = Sequential()
        model.add(GRU(64, input_shape=input_shape))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        print(model.summary())
    elif model_type == "cnn":
        print("cnn chosen")
        # seeds in elif so both are initialised with the same seed
        np.random.seed(42)
        rn.seed(42)
        tf.random.set_seed(42)

        model = Sequential()
        model.add(Conv1D(32, kernel_size=5,
                         activation='relu',
                         input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(Conv1D(64, 5, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        print(model.summary())
    elif model_type == "birnn":
        print("birnn chosen")
        # seeds in elif so both are initialised with the same seed
        np.random.seed(42)
        rn.seed(42)
        tf.random.set_seed(42)

        model = Sequential()
        model.add(Bidirectional(GRU(64), input_shape=input_shape))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        print(model.summary())
    elif model_type == "nn":
        print("nn chosen")
        # seeds in elif so both are initialised with the same seed
        np.random.seed(42)
        rn.seed(42)
        tf.random.set_seed(42)

        model = Sequential()
        model.add(Dense(500, activation='relu', input_shape=input_shape))
        model.add(Dense(1000, activation='relu'))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        print(model.summary())
    elif model_type == "lstmcnn":
        print("lstmcnn dual model chosen")
        # seeds in elif so both are initialised with the same seed
        np.random.seed(42)
        rn.seed(42)
        tf.random.set_seed(42)

        input_all = Input(shape = input_shape)

        # cnn part
        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding="same")(input_all)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.ReLU()(conv1)

        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding="same")(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.ReLU()(conv2)

        conv3 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.ReLU()(conv3)

        gap = keras.layers.GlobalAveragePooling1D()(conv3)

        # rnn part
        rnn_layer = LSTM(64, activation = "relu")(input_all)
        dropout_layer = keras.layers.Dropout(rate=0.8)(rnn_layer)

        # comination and final dense
        combin = keras.layers.concatenate([gap, dropout_layer])

        out_layer = Dense(num_classes, activation='softmax')(combin)
        model = keras.Model(inputs=input_all,
                            outputs=out_layer)
        print(model.summary())
    else:
        print("only cnn, rnn, birnn, lstmcnn or nn are available as model type")

    trained_model, hist = compile_and_fit_ts(model, x_train, y_train, x_val,
                                             y_val, x_test, y_test, batch_size,
                                             epochs, model_name=model_path_name,
                                             data_name=data_path_name)

    plot_hist(hist)

    # load best model and then evaluate
    loading_model_path = "./uci_har_ts_models/" + model_path_name + ".h5"
    best_model = keras.models.load_model(loading_model_path)

    train_score = best_model.evaluate(x_train, y_train, verbose=0)
    print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
    val_score = best_model.evaluate(x_val, y_val, verbose=0)
    print('Val loss: {}, Val accuracy: {}'.format(val_score[0], val_score[1]))
    test_score = best_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))

    result_dataframe = pd.DataFrame(np.array([train_score, val_score, test_score]), index = ["train", "val", "test"],columns = ["loss", "acc"])
    result_path = "./uci_har_ts_models/results/" + model_path_name + ".csv"
    result_dataframe.to_csv(result_path)

    print("saved results for the best version of model: " + model_path_name)
    keras.backend.clear_session()

def compile_and_fit_feat(X_train, y_train, X_val, y_val, X_test, y_test, model_type, model_name, data_name):
    # adding val data to training data as no val data is used during the fitting of the ml models
    # add csv saving for x/y_train, x/y_val and x/y_test (build new folder for the data with the modelname)
    data_path = "./uci_har_feat_models/processed_data/" + data_name
    np.savez(data_path, x_train = X_train, y_train = y_train, x_val = X_val, y_val = y_val, x_test = X_test, y_test = y_test)
    
    # train and val are appended because the sklearn model have no use for validation data except for cross-val.
    X_train = np.append(X_train, X_val, axis = 0)
    Y_train = [*y_train, *y_val]

    # nn is close to 
    if model_type == "nn":
        model = MLPClassifier(hidden_layer_sizes=[500,1000,12],
                                                     random_state = 42,
                                                     batch_size = 32,
                                                     max_iter = 15,
                                                     n_iter_no_change = 5)
    elif model_type == "svc":
        model = SVC(decision_function_shape= "ovr", break_ties = True, random_state=42)
    elif model_type == "rfc":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "dtc":
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == "knc":
        model = KNeighborsClassifier(metric="chebyshev",random_state=42)
    elif model_type == "gbc":
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_type == "logreg":
        model = LogisticRegression(max_iter = 5000, random_state=42)
    else:
        print("choose from nn, svc, rfc, dtc, knc or gbc")
    model.fit(X_train, Y_train)
    print("model is fitted")
    path = "./uci_har_feat_models/" + model_name + '.joblib'

    dump(model, path)
    return model

def feature_extraction_sub1(input_signal, settings):
    """
    
    """
    temp_dataframe_in = pd.DataFrame(columns= ["b_x", "b_y", "b_z", "gy_x", "gy_y", "gy_z", "g_x", "g_y", "g_z","id", "t"])
    for i in range(input_signal.shape[0]):
        temp_data = pd.DataFrame(input_signal[i,:,:], columns = ["b_x", "b_y", "b_z", "gy_x", "gy_y", "gy_z", "g_x", "g_y", "g_z"])
        temp_data["id"] = i
        temp_data["t"] = temp_data.index
        temp_dataframe_in = temp_dataframe_in.append(temp_data)
    temp_data_feat = tsfresh.feature_extraction.extract_features(temp_dataframe_in, column_id = "id", column_sort = "t", default_fc_parameters=settings)
    temp_data_feat_comp = np.nan_to_num(temp_data_feat,posinf=0, neginf=0)
    return temp_data_feat_comp

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy

def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]

def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]

def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics

def feature_extraction_wave(input_signal, wavelet_name):
    list_coeff = pywt.wavedec(input_signal, wavelet_name)
    features = []
    for coeff in list_coeff:
        features += get_features(coeff)
    return features

def feature_extraction_wave_array(input_array, wavelet_name):
    output_list = []
    for i in range(input_array.shape[0]):
        temp_list = []
        for j in range(input_array.shape[2]):
            temp_list += feature_extraction_wave(input_array[i,:,j], wavelet_name)
        output_list.append(temp_list)
    output_array = np.array(output_list)
    output_array = np.nan_to_num(output_array,posinf=0, neginf=0)
    return output_array

def har_feature_extration_with_subband(x_train, y_train, x_val, y_val, x_test, y_test, subband_wavelet):
    print("extracting features")
    if subband_wavelet == "none":
        settings_paper = {"mean": None,
                      "standard_deviation" : None,
                      "median" : None,
                      "maximum" : None,
                      "minimum" : None,
                      "abs_energy" : None,
                      "variance" : None,
                      "mean_abs_change" : None,
                      "ar_coefficient": [{'coeff': 0, 'k': 10}, {'coeff': 1, 'k': 10}, {'coeff': 2, 'k': 10}, {'coeff': 3, 'k': 10}, {'coeff': 4, 'k': 10}, {'coeff': 5, 'k': 10}, {'coeff': 6, 'k': 10}, {'coeff': 7, 'k': 10}, {'coeff': 8, 'k': 10}, {'coeff': 9, 'k': 10}, {'coeff': 10, 'k': 10}],
                      "fft_aggregated" : [{'aggtype': 'skew'},{'aggtype': 'kurtosis'}],
                      "number_peaks" : [{"n" : 5}],
                      "sample_entropy" : None,
                      "quantile" : [{"q" : 0.25}, {"q" : 0.75}]}
        
        x_train = feature_extraction_sub1(x_train, settings_paper)
        x_val = feature_extraction_sub1(x_val, settings_paper)
        x_test = feature_extraction_sub1(x_test, settings_paper)
    else:
        x_train = feature_extraction_wave_array(x_train, subband_wavelet)
        x_val = feature_extraction_wave_array(x_val, subband_wavelet)
        x_test = feature_extraction_wave_array(x_test, subband_wavelet)
        
        

    return x_train, y_train, x_val, y_val, x_test, y_test

def compile_and_fit_feat_nn(model, X_train, y_train, X_val, y_val, X_test, y_test, batch_size, n_epochs, model_name, data_name):
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['categorical_accuracy'])
    
    print("compiled")
    path = "./uci_har_feat_models/" + model_name + '.h5'
    logging_path = "./uci_har_feat_models/logs/" + model_name + ".log"
    # define callback for model saving
    checkpoint_callback = ModelCheckpoint(filepath= path, monitor='val_categorical_accuracy', save_best_only=True)
    csv_logger_callback = keras.callbacks.CSVLogger(filename=logging_path)

    history  = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              verbose=1,
              validation_data=(X_val, y_val),
              callbacks = [checkpoint_callback, csv_logger_callback],
              shuffle = True)
    return model, history

def feat_variations(acc_data_path, gyro_data_path, denoise_thresh, denoise_dwt_wavelet, window_size, subband_dwt_wavelet, model_type):
    
    # randomness reduction and allowing reproduciblity, though the reduction to 1 thread introduces time constraints
    # as found in https://github.com/keras-team/keras/issues/2280#issuecomment-492826113

    np.random.seed(42)
    rn.seed(42)
    #tf.config.threading.set_intra_op_parallelism_threads(1)
    #tf.config.threading.set_inter_op_parallelism_threads(1)
    
    # name of the files etc is now in the format: dwt-wavelet_dwt-threshold_window-size_cwt-wavelet_cwt-scale
    model_path_name = denoise_dwt_wavelet + "_" + str(denoise_thresh) + "_" + str(window_size) + subband_dwt_wavelet + "_" + model_type
    data_path_name = denoise_dwt_wavelet + "_" + str(denoise_thresh) + "_" + str(window_size) + subband_dwt_wavelet
    # check if that specific combination already exists, if true reads the already preprocessed data
    # allows multiple models to be trained on the same preprocessed data
    data_path = "./uci_har_feat_models/processed_data/" + data_path_name + ".npz"
    if os.path.exists(data_path):
        print("preprocessed data already exists, reading from file")
        saved_data_file = np.load(data_path)
        x_train = saved_data_file["x_train"]
        y_train = saved_data_file["y_train"]
        x_val = saved_data_file["x_val"]
        y_val = saved_data_file["y_val"]
        x_test = saved_data_file["x_test"]
        y_test = saved_data_file["y_test"]
    
    else:
        print("no preprocessed data found, begin preprocessing...")
        raw_acc_data = read_raw_acc_data(acc_data_path)
        raw_gyro_data = read_raw_gyro_data(gyro_data_path)
        
        print("denoising with " + denoise_dwt_wavelet + " and a thresh. of: " + str(denoise_thresh))
        if denoise_dwt_wavelet != "noisy":
            gyro_data_denoised = denoise_har_raw_data(raw_gyro_data, thresh=denoise_thresh, wavelet=denoise_dwt_wavelet)
            acc_data_denoised = denoise_har_raw_data(raw_acc_data, thresh=denoise_thresh, wavelet=denoise_dwt_wavelet)
        else:
            gyro_data_denoised = raw_gyro_data
            acc_data_denoised = raw_acc_data
        print("windowing data into windows of size: " + str(window_size) + " with 50% overlap")
        
        har_data, har_labels, har_pers_label = transform_har_data(acc_data_denoised, gyro_data_denoised)
        
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


        # feature extraction
        x_train, y_train, x_val, y_val, x_test, y_test = har_feature_extration_with_subband(har_train_data,
                                                                               har_train_labels,
                                                                               har_val_data,
                                                                               har_val_labels,
                                                                               har_test_data,
                                                                               har_test_labels,
                                                                               subband_wavelet=subband_dwt_wavelet)

    # until here is data preprocessing
    if model_type != "kerasnn":
        trained_model = compile_and_fit_feat(x_train, y_train, x_val,
                                                 y_val, x_test, y_test,
                                                 model_type=model_type,
                                                 model_name=model_path_name,
                                                 data_name=data_path_name)
        best_model = trained_model
        train_score = best_model.score(x_train, y_train)
        print('Train accuracy: {}'.format(train_score))
        test_score = best_model.score(x_test, y_test)
        print('Test accuracy: {}'.format(test_score))
    
        result_dataframe = pd.DataFrame(np.array([train_score, test_score]), index = ["train", "test"],columns = ["acc"])
        result_path = "./uci_har_feat_models/results/" + model_path_name + ".csv"
        result_dataframe.to_csv(result_path)
        print("saved results for model: " + model_path_name)
    elif model_type == "kerasnn":
        # keras nn model as sklearn nn doesn't allow saving the best model
        
        num_classes = 12
        input_shape = (x_train.shape[1],)
        batch_size = 32
        epochs = 15
        y_train = [i-1 for i in y_train]    
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_val = [i-1 for i in y_val]
        y_val = keras.utils.to_categorical(y_val, num_classes)
        y_test = [i-1 for i in y_test]
        y_test = keras.utils.to_categorical(y_test, num_classes)

        print("keras nn chosen")
        # seeds in elif so all are initialised with the same seed
        np.random.seed(42)
        rn.seed(42)
        tf.random.set_seed(42)

        model = Sequential()
        model.add(Dense(500, activation='relu', input_shape=input_shape))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        print(model.summary())
        trained_model, hist = compile_and_fit_feat_nn(model, x_train, y_train, x_val,
                                             y_val, x_test, y_test, batch_size,
                                             epochs, model_name=model_path_name,
                                             data_name=data_path_name)
        plot_hist(hist)
        loading_model_path = "./uci_har_feat_models/" + model_path_name + ".h5"
        best_model =  keras.models.load_model(loading_model_path)
        
        train_score = best_model.evaluate(x_train, y_train, verbose=0)
        print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
        val_score = best_model.evaluate(x_val, y_val, verbose=0)
        print('Val loss: {}, Val accuracy: {}'.format(val_score[0], val_score[1]))
        test_score = best_model.evaluate(x_test, y_test, verbose=0)
        print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))
    
        result_dataframe = pd.DataFrame(np.array([train_score, val_score, test_score]), index = ["train", "val", "test"],columns = ["loss", "acc"])
        result_path = "./uci_har_feat_models/results/" + model_path_name + ".csv"
        result_dataframe.to_csv(result_path)
    
        print("saved results for the best version of model: " + model_path_name)
        keras.backend.clear_session()
    
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

def compile_and_fit_org_feat_nn(model, X_train, y_train, X_val, y_val, X_test, y_test, batch_size, n_epochs, model_name, data_name):
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['categorical_accuracy'])
    
    print("compiled")
    path = "./uci_har_original_feat_models/" + model_name + '.h5'
    logging_path = "./uci_har_original_feat_models/logs/" + model_name + ".log"
    # define callback for model saving
    checkpoint_callback = ModelCheckpoint(filepath= path, monitor='val_categorical_accuracy', save_best_only=True)
    csv_logger_callback = keras.callbacks.CSVLogger(filename=logging_path)

    history  = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              verbose=1,
              validation_data=(X_val, y_val),
              callbacks = [checkpoint_callback, csv_logger_callback],
              shuffle = True)
    return model, history

def compile_and_fit_org_feat(X_train, y_train, X_val, y_val, X_test, y_test, model_type, model_name, data_name):
    # adding val data to training data as no val data is used during the fitting of the ml models
    # add csv saving for x/y_train, x/y_val and x/y_test (build new folder for the data with the modelname)
    data_path = "./uci_har_original_feat_models/processed_data/" + data_name
    np.savez(data_path, x_train = X_train, y_train = y_train, x_val = X_val, y_val = y_val, x_test = X_test, y_test = y_test)
    
    # train and val are appended because the sklearn model have no use for validation data except for cross-val.
    X_train = np.append(X_train, X_val, axis = 0)
    Y_train = [*y_train, *y_val]

    # nn is close to 
    if model_type == "nn":
        model = MLPClassifier(hidden_layer_sizes=[500,1000,12],
                                                     random_state = 42,
                                                     batch_size = 32,
                                                     max_iter = 15,
                                                     n_iter_no_change = 5)
    elif model_type == "svc":
        model = SVC(decision_function_shape= "ovr", break_ties = True, random_state=42)
    elif model_type == "rfc":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "dtc":
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == "knc":
        model = KNeighborsClassifier(metric="chebyshev",random_state=42)
    elif model_type == "gbc":
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_type == "logreg":
        model = LogisticRegression(max_iter = 5000, random_state=42)
    else:
        print("choose from nn, svc, rfc, dtc, knc or gbc")
    model.fit(X_train, Y_train)
    print("model is fitted")
    path = "./uci_har_original_feat_models/" + model_name + '.joblib'

    dump(model, path)
    return model

def feature_orginal_data_variations(model_type):
    # todo:
    # - orginal daten einlesen 
    # - split by experiment
    # - split the same as the other variations with custom preprocessing
    # - modelle auf den "neuen" daten laufen lassen
    
    np.random.seed(42)
    rn.seed(42)
    #tf.config.threading.set_intra_op_parallelism_threads(1)
    #tf.config.threading.set_inter_op_parallelism_threads(1)
    
    # name of the files etc is now in the format: dwt-wavelet_dwt-threshold_window-size_cwt-wavelet_cwt-scale
    model_path_name = "original_paper_preprocessing_" + model_type
    data_path_name = "original_paper_preprocessing"
    # check if that specific combination already exists, if true reads the already preprocessed data
    # allows multiple models to be trained on the same preprocessed data
    data_path = "./uci_har_original_feat_models/processed_data/" + data_path_name + ".npz"
    if os.path.exists(data_path):
        print("preprocessed data already exists, reading from file")
        saved_data_file = np.load(data_path)
        x_train = saved_data_file["x_train"]
        y_train = saved_data_file["y_train"]
        x_val = saved_data_file["x_val"]
        y_val = saved_data_file["y_val"]
        x_test = saved_data_file["x_test"]
        y_test = saved_data_file["y_test"]
    
    else:
        print("no preprocessed data found, reading features from the original dataset")
        x_train, y_train, x_val, y_val, x_test, y_test = read_and_split_paper_features("../datasets/uci_har/v2/")

    # until here is data preprocessing
    if model_type != "kerasnn":
        trained_model = compile_and_fit_org_feat(x_train, y_train, x_val,
                                                 y_val, x_test, y_test,
                                                 model_type=model_type,
                                                 model_name=model_path_name,
                                                 data_name=data_path_name)
        best_model = trained_model
        train_score = best_model.score(x_train, y_train)
        print('Train accuracy: {}'.format(train_score))
        test_score = best_model.score(x_test, y_test)
        print('Test accuracy: {}'.format(test_score))
    
        result_dataframe = pd.DataFrame(np.array([train_score, test_score]), index = ["train", "test"],columns = ["acc"])
        result_path = "./uci_har_original_feat_models/results/" + model_path_name + ".csv"
        result_dataframe.to_csv(result_path)
        print("saved results for model: " + model_path_name)
    elif model_type == "kerasnn":
        # keras nn model as sklearn nn doesn't allow saving the best model
        
        num_classes = 12
        input_shape = (x_train.shape[1],)
        batch_size = 32
        epochs = 15
        y_train = [i-1 for i in y_train]    
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_val = [i-1 for i in y_val]
        y_val = keras.utils.to_categorical(y_val, num_classes)
        y_test = [i-1 for i in y_test]
        y_test = keras.utils.to_categorical(y_test, num_classes)

        print("keras nn chosen")
        # seeds in elif so all are initialised with the same seed
        np.random.seed(42)
        rn.seed(42)
        tf.random.set_seed(42)

        model = Sequential()
        model.add(Dense(500, activation='relu', input_shape=input_shape))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        print(model.summary())
        trained_model, hist = compile_and_fit_org_feat_nn(model, x_train, y_train, x_val,
                                             y_val, x_test, y_test, batch_size,
                                             epochs, model_name=model_path_name,
                                             data_name=data_path_name)
        plot_hist(hist)
        loading_model_path = "./uci_har_original_feat_models/" + model_path_name + ".h5"
        best_model =  keras.models.load_model(loading_model_path)
        
        train_score = best_model.evaluate(x_train, y_train, verbose=0)
        print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
        val_score = best_model.evaluate(x_val, y_val, verbose=0)
        print('Val loss: {}, Val accuracy: {}'.format(val_score[0], val_score[1]))
        test_score = best_model.evaluate(x_test, y_test, verbose=0)
        print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))
    
        result_dataframe = pd.DataFrame(np.array([train_score, val_score, test_score]), index = ["train", "val", "test"],columns = ["loss", "acc"])
        result_path = "./uci_har_original_feat_models/results/" + model_path_name + ".csv"
        result_dataframe.to_csv(result_path)
    
        print("saved results for the best version of model: " + model_path_name)
        keras.backend.clear_session()