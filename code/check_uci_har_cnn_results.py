# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:07:15 2020

@author: Joshua
"""

import keras
import numpy as np
import har_v2_f
import pywt
import pandas as pd
import os

# load a model and the preprocessed data
def read_and_check_cwt_model(model_path_name):
    np.random.seed(42)
    rn.seed(42)
    tf.random.set_seed(42)
    model_path = "./uci_har_cnn_models/" + model_path_name + ".h5"
    data_path = "./uci_har_cnn_models/processed_data/" + model_path_name + ".npz"
    results_path = "./uci_har_cnn_models/results/" + model_path_name + ".csv"
    test_model = keras.models.load_model(model_path)
    saved_data_file = np.load(data_path)
    x_train = saved_data_file["x_train"]
    y_train = saved_data_file["y_train"]
    x_val = saved_data_file["x_val"]
    y_val = saved_data_file["y_val"]
    x_test = saved_data_file["x_test"]
    y_test = saved_data_file["y_test"]
    
    train_score = test_model.evaluate(x_train, y_train, verbose=0)
    print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
    val_score = test_model.evaluate(x_val, y_val, verbose=0)
    print('Val loss: {}, Val accuracy: {}'.format(val_score[0], val_score[1]))
    test_score = test_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))
    
    # Read results in from the path
    loaded_test = pd.read_csv(results_path, index_col = 0)
    print(loaded_test)





pywt.wavelist(kind = "discrete")
pywt.wavelist(kind = "continuous")

# cnn cwt models
data_path = "../datasets/uci_har/v2/RawData/"
#discrete_wavelet = ["noisy","median","haar","db4","dmey","sym4","coif4"]
discrete_wavelet = ["dmey","sym4","coif4"]
# disc_thresh = [0.2,0.3]
disc_thresh  = 0.3
window_size = 128
#continuous_wavelet = ["morl","mexh","gaus2"]
continuous_wavelet = ["gaus2"]
cont_scale = [64,128] # postprocessing resizes the resulting image to cont_scale x cont_scale if necessary

for j in discrete_wavelet:
    for k in cont_scale:
        for i in continuous_wavelet:
            har_v2_f.cnn_variations(acc_data_path = data_path,
                           gyro_data_path = data_path,
                           denoise_thresh = disc_thresh,
                           denoise_dwt_wavelet = j,
                           window_size = window_size,
                           cwt_wavelet = i,
                           cwt_scale = k)
    


read_and_check_cwt_model("db4_0.3_128mexh_128")



test = pd.read_csv("./uci_har_cnn_models/results/db4_0.2_128mexh_128.csv", index_col = 0)
test.loc["test","acc"]
