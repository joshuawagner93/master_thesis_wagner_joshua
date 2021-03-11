# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:24:08 2021

@author: Joshua
"""

# define path as is done in the har script
# load models,data and get confusion matrices
# for now take a look at the best and worst performing models per prep.
# fucntion which reads a preprocessed data file and the corresponding models

import numpy as np
import keras
import random as rn
import itertools
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from joblib import dump, load

np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)

def plt_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



def plot_cwt_confusion(denoise_dwt_wavelet, denoise_thresh, cwt_wavelet, cwt_scale):
    model_path_name = denoise_dwt_wavelet + "_" + str(denoise_thresh) + "_" + str(128) + cwt_wavelet + "_" + str(cwt_scale)
    model_path = "./uci_har_cwt_cnn_models/" + model_path_name + ".h5"
    data_path = "./uci_har_cwt_cnn_models/processed_data/" + model_path_name + ".npz"
    #results_path = "./uci_har_cwt_cnn_models/results/" + model_path_name + ".csv"
    test_model = keras.models.load_model(model_path)
    saved_data_file = np.load(data_path)
    x_test = saved_data_file["x_test"]
    y_test = saved_data_file["y_test"]
    
    print("read neccessary components, now predicting")
    test_pred = test_model.predict(x_test, verbose=0)
    predictions= np.argmax(test_pred, axis=1)
    true_test = np.argmax(y_test,axis=1) 
    conf_matrix = confusion_matrix(y_true=true_test, y_pred=predictions)
    y_classes = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING",
                 "STANDING", "LAYING", "STAND_TO_SIT", "SIT_TO_STAND", "SIT_TO_LIE",
                 "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND"]
    plt_confusion_matrix(cm=conf_matrix, classes = y_classes)

# best cwt cnn model    
plot_cwt_confusion(denoise_dwt_wavelet="noisy", denoise_thresh=0.2, cwt_wavelet="morl", cwt_scale=128)

# worst cwt cnn model
plot_cwt_confusion(denoise_dwt_wavelet="haar", denoise_thresh=0.3, cwt_wavelet="mexh", cwt_scale=128)
#plt.savefig("../masters_thesis/images/cwt_cnn_worst_conf.pdf")

def plot_ts_confusion(denoise_dwt_wavelet, denoise_thresh, subband_dwt, model_type):
    model_path_name = denoise_dwt_wavelet + "_" + str(denoise_thresh) + "_" + str(128) + subband_dwt + "_" + model_type
    data_path_name = denoise_dwt_wavelet + "_" + str(denoise_thresh) + "_" + str(128) + subband_dwt
    model_path = "./uci_har_ts_models/" + model_path_name + ".h5"
    data_path = "./uci_har_ts_models/processed_data/" + data_path_name + ".npz"
    #results_path = "./uci_har_cwt_cnn_models/results/" + model_path_name + ".csv"
    test_model = keras.models.load_model(model_path)
    saved_data_file = np.load(data_path)
    x_test = saved_data_file["x_test"]
    y_test = saved_data_file["y_test"]
    
    print("read neccessary components, now predicting")
    test_pred = test_model.predict(x_test, verbose=0)
    predictions= np.argmax(test_pred, axis=1)
    true_test = np.argmax(y_test,axis=1) 
    conf_matrix = confusion_matrix(y_true=true_test, y_pred=predictions)
    y_classes = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING",
                 "STANDING", "LAYING", "STAND_TO_SIT", "SIT_TO_STAND", "SIT_TO_LIE",
                 "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND"]
    plt_confusion_matrix(cm=conf_matrix, classes = y_classes)
    
# best ts model
plot_ts_confusion(denoise_dwt_wavelet = "median",denoise_thresh="0.2",subband_dwt="haar",model_type="nn") # comparable to the cwt cnn model

# worst ts model
plot_ts_confusion(denoise_dwt_wavelet = "dmey",denoise_thresh="0.2",subband_dwt="sym4",model_type="lstmcnn") # literally everything is walking ^^

# second to last model, actually better in the stand_to_lie and lie_to_stand than the rest but doesn't differentiate walking well enough
plot_ts_confusion(denoise_dwt_wavelet = "sym4",denoise_thresh="0.3",subband_dwt="sym4",model_type="rnn")

def plot_feat_confusion(denoise_dwt_wavelet, denoise_thresh, subband_dwt, model_type):
    model_path_name = denoise_dwt_wavelet + "_" + str(denoise_thresh) + "_" + str(128) + subband_dwt + "_" + model_type
    data_path_name = denoise_dwt_wavelet + "_" + str(denoise_thresh) + "_" + str(128) + subband_dwt
    if model_type in ["nn","svc","rfc","dtc","knc","gbc","logreg"]:
        model_path = "./uci_har_feat_models/" + model_path_name + ".joblib"
    else:
        model_path = "./uci_har_feat_models/" + model_path_name + ".h5"
    data_path = "./uci_har_feat_models/processed_data/" + data_path_name + ".npz"
    #results_path = "./uci_har_cwt_cnn_models/results/" + model_path_name + ".csv"
    if model_type in ["nn","svc","rfc","dtc","knc","gbc","logreg"]:
        test_model = load(model_path)
        saved_data_file = np.load(data_path)
        x_test = saved_data_file["x_test"]
        y_test = saved_data_file["y_test"]
        print("read neccessary components, now predicting")
        predictions = test_model.predict(x_test)
        true_test = y_test
    else:
        test_model = keras.models.load_model(model_path)
        saved_data_file = np.load(data_path)
        x_test = saved_data_file["x_test"]
        y_test = saved_data_file["y_test"]
        print("read neccessary components, now predicting")
        test_pred = test_model.predict(x_test, verbose=0)
        predictions= np.argmax(test_pred, axis=1)
        true_test = np.argmax(y_test,axis=1) 
    
    
    conf_matrix = confusion_matrix(y_true=true_test, y_pred=predictions)
    y_classes = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING",
                 "STANDING", "LAYING", "STAND_TO_SIT", "SIT_TO_STAND", "SIT_TO_LIE",
                 "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND"]
    plt_confusion_matrix(cm=conf_matrix, classes = y_classes)

# best feat model, actually decent results for the transitional motions, in transition outperforms the cwt cnn
plot_feat_confusion(denoise_dwt_wavelet = "noisy",denoise_thresh="0.2",subband_dwt="sym4",model_type="gbc")

# worst feat model, most stuff gets classified as lie_to_stand
plot_feat_confusion(denoise_dwt_wavelet = "coif4",denoise_thresh="0.3",subband_dwt="none",model_type="logreg")


def plot_org_feat_confusion(model_type):
    model_path_name = "original_paper_preprocessing_" + model_type
    data_path_name = "original_paper_preprocessing"
    if model_type in ["nn","svc","rfc","dtc","knc","gbc","logreg"]:
        model_path = "./uci_har_original_feat_models/" + model_path_name + ".joblib"
    else:
        model_path = "./uci_har_original_feat_models/" + model_path_name + ".h5"
    data_path = "./uci_har_original_feat_models/processed_data/" + data_path_name + ".npz"
    #results_path = "./uci_har_cwt_cnn_models/results/" + model_path_name + ".csv"
    if model_type in ["nn","svc","rfc","dtc","knc","gbc","logreg"]:
        test_model = load(model_path)
        saved_data_file = np.load(data_path)
        x_test = saved_data_file["x_test"]
        y_test = saved_data_file["y_test"]
        print("read neccessary components, now predicting")
        predictions = test_model.predict(x_test)
        true_test = y_test
    else:
        test_model = keras.models.load_model(model_path)
        saved_data_file = np.load(data_path)
        x_test = saved_data_file["x_test"]
        y_test = saved_data_file["y_test"]
        print("read neccessary components, now predicting")
        test_pred = test_model.predict(x_test, verbose=0)
        predictions= np.argmax(test_pred, axis=1)
        true_test = np.argmax(y_test,axis=1) 
    
    
    conf_matrix = confusion_matrix(y_true=true_test, y_pred=predictions)
    y_classes = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING",
                 "STANDING", "LAYING", "STAND_TO_SIT", "SIT_TO_STAND", "SIT_TO_LIE",
                 "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND"]
    plt_confusion_matrix(cm=conf_matrix, classes = y_classes, title = "org. features model")

plot_org_feat_confusion("logreg")
#plt.savefig("../masters_thesis/images/multinom_confusion.pdf")
plot_org_feat_confusion("svc")
plot_org_feat_confusion("gbc")
