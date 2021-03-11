# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 12:16:23 2021

@author: Joshua
"""

import pandas as pd
from itertools import cycle
import pywt
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pmdarima as pm
from pmdarima.metrics import smape
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from joblib import dump, load

import keras
from keras.layers import Dense, Flatten, Input, Reshape
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import GRU, LSTM, Bidirectional
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.metrics import RootMeanSquaredError
from keras.losses import MeanSquaredError

import random as rn
import tensorflow as tf





seas_dict = {'Hourly': {'seasonality': 24, 'input_size': 24,
                       'output_size': 48, 'freq': 'H'},
             'Daily': {'seasonality': 7, 'input_size': 7,
                       'output_size': 14, 'freq': 'D'},
             'Weekly': {'seasonality': 52, 'input_size': 52,
                        'output_size': 13, 'freq': 'W'},
             'Monthly': {'seasonality': 12, 'input_size': 12,
                         'output_size':18, 'freq': 'M'},
             'Quarterly': {'seasonality': 4, 'input_size': 4,
                           'output_size': 8, 'freq': 'Q'},
             'Yearly': {'seasonality': 1, 'input_size': 4,
                        'output_size': 6, 'freq': 'D'}}
             


def m4_parser(dataset_name, directory, num_obs=1000000):
  """
  Transform M4 data into a panel.
  Parameters
  ----------
  dataset_name: str
    Frequency of the data. Example: 'Yearly'.
  directory: str
    Custom directory where data will be saved.
  num_obs: int
    Number of time series to return.
  """
  data_directory = directory + "/m4"
  train_directory = data_directory + "/Train/"
  test_directory = data_directory + "/Test/"
  freq = seas_dict[dataset_name]['freq']

  m4_info = pd.read_csv(data_directory+'/M4-info.csv', usecols=['M4id','category'])
  m4_info = m4_info[m4_info['M4id'].str.startswith(dataset_name[0])].reset_index(drop=True)

  # Train data
  train_path='{}{}-train.csv'.format(train_directory, dataset_name)

  train_df = pd.read_csv(train_path, nrows=num_obs)
  train_df = train_df.rename(columns={'V1':'unique_id'})

  train_df = pd.wide_to_long(train_df, stubnames=["V"], i="unique_id", j="ds").reset_index()
  train_df = train_df.rename(columns={'V':'y'})
  train_df = train_df.dropna()
  train_df['split'] = 'train'
  train_df['ds'] = train_df['ds']-1
  # Get len of series per unique_id
  len_series = train_df.groupby('unique_id').agg({'ds': 'max'}).reset_index()
  len_series.columns = ['unique_id', 'len_serie']

  # Test data
  test_path='{}{}-test.csv'.format(test_directory, dataset_name)

  test_df = pd.read_csv(test_path, nrows=num_obs)
  test_df = test_df.rename(columns={'V1':'unique_id'})

  test_df = pd.wide_to_long(test_df, stubnames=["V"], i="unique_id", j="ds").reset_index()
  test_df = test_df.rename(columns={'V':'y'})
  test_df = test_df.dropna()
  test_df['split'] = 'test'
  test_df = test_df.merge(len_series, on='unique_id')
  test_df['ds'] = test_df['ds'] + test_df['len_serie'] - 1
  test_df = test_df[['unique_id','ds','y','split']]

  df = pd.concat((train_df,test_df))
  df = df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)

  # Create column with dates with freq of dataset
  len_series = df.groupby('unique_id').agg({'ds': 'max'}).reset_index()
  dates = []
  for i in range(len(len_series)):
      len_serie = len_series.iloc[i,1]
      ranges = pd.date_range(start='1970/01/01', periods=len_serie, freq=freq)
      dates += list(ranges)
  df.loc[:,'ds'] = dates

  df = df.merge(m4_info, left_on=['unique_id'], right_on=['M4id'])
  df.drop(columns=['M4id'], inplace=True)
  df = df.rename(columns={'category': 'x'})

  X_train_df = df[df['split']=='train'].filter(items=['unique_id', 'ds', 'x'])
  y_train_df = df[df['split']=='train'].filter(items=['unique_id', 'ds', 'y'])
  X_test_df = df[df['split']=='test'].filter(items=['unique_id', 'ds', 'x'])
  y_test_df = df[df['split']=='test'].filter(items=['unique_id', 'ds', 'y'])

  X_train_df = X_train_df.reset_index(drop=True)
  y_train_df = y_train_df.reset_index(drop=True)
  X_test_df = X_test_df.reset_index(drop=True)
  y_test_df = y_test_df.reset_index(drop=True)

  return X_train_df, y_train_df, X_test_df, y_test_df


def dwt_lowpassfilter(signal_in, thresh=0.2, wavelet="db4", level=2):
    thresh = thresh*np.nanmax(signal_in)
    coeff = pywt.wavedec(signal_in, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per")
    return reconstructed_signal


def fit_denoised_ml_model(X_train, y_train, X_val, y_val, X_test, y_test, model_type, model_name, threshold, level, wavelet):
    # adding val data to training data as no val data is used during the fitting of the ml models
    if level == "inf":
        level = None
    # val. data additional preprocessing if ml model is chosen and val. data is added to train
    if wavelet != "noisy":
        X_val = np.apply_along_axis(dwt_lowpassfilter, axis=1, arr = X_val, thresh=threshold, level=level, wavelet = wavelet)
        y_val = np.apply_along_axis(dwt_lowpassfilter, axis=1, arr = y_val, thresh=threshold, level=level,wavelet = wavelet)
        
    X_train = np.append(X_train, X_val, axis = 0)
    y_train = np.append(y_train, y_val, axis = 0)
    
    if model_type == "svr":
        model = MultiOutputRegressor(SVR(), n_jobs = -1)
    elif model_type == "rfr":
        model = RandomForestRegressor(random_state=42)
    elif model_type == "dtc":
        model = DecisionTreeRegressor(random_state=42)
    else:
        print("how did you get here?")

    model.fit(X_train, y_train)
    print("model is fitted")
    path = "./m4_ts_models/" + model_name + '.joblib'

    dump(model, path)
    return model


def fit_denoised_dl_model(model, X_train, y_train, X_val, y_val, X_test, y_test, batch_size, n_epochs, model_name):
    model.compile(loss="mean_squared_error",
                  optimizer=keras.optimizers.Adam(),
                  metrics=[RootMeanSquaredError()])
    
    print("compiled")
    path = "./m4_ts_models/" + model_name + '.h5'
    logging_path = "./m4_ts_models/logs/" + model_name + ".log"
    # define callback for model saving
    checkpoint_callback = ModelCheckpoint(filepath= path, monitor='val_root_mean_squared_error', save_best_only=True)
    csv_logger_callback = keras.callbacks.CSVLogger(filename=logging_path)

    history  = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              verbose=1,
              validation_data=(X_val, y_val),
              callbacks = [checkpoint_callback, csv_logger_callback],
              shuffle = True)

    return model, history

def plot_hist(history):
    plt.title('root_mean_squared_error of val. and test sets')
    plt.plot(history.history['root_mean_squared_error'], label='Train RMSE')
    plt.plot(history.history['val_root_mean_squared_error'], label='Val RMSE')
    plt.ylabel('root_mean_squared_error')
    plt.xlabel('No. epoch')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

def m4_denoising_variations(model_type, wavelet="db4", threshold=0.2, level=2):
    np.random.seed(42)
    rn.seed(42)
    tf.random.set_seed(42)
    
    time ="Monthly"
    data_path_name = time+ "_" + wavelet + "_" + str(threshold) + "_" + str(level)
    data_path = "./m4_ts_models/processed_data/" + data_path_name + ".npz"
    model_path_name = time + "_" + wavelet + "_" + str(threshold) + "_" + str(level) + "_" + model_type
    
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
        x_train_input, y_train_input, x_train_output, y_train_output = m4_parser(time,"../datasets")
    
        y_input_wide = y_train_input.pivot(index = "unique_id", columns = "ds", values = "y")
        y_input_wide.columns = range(0, 2794)
        
        y_train_input_fixed = pd.DataFrame()
        first_nan = y_input_wide.notna().idxmin(1)
        lowest_numb_of_obs = first_nan.sort_values().head()
        print(lowest_numb_of_obs)
        
        
        
        for i in first_nan.index:    
            y_train_input_fixed = y_train_input_fixed.append(y_input_wide.loc[i,(first_nan[i]-42):(first_nan[i]-1)].reset_index(drop = True)) # min length of an input series is 42, so every series gets cut to length 42
        
        to_drop = y_train_input_fixed[y_train_input_fixed.isnull().any(axis=1)].index[0]
        print(to_drop)
        indices_y_test = cycle([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]) # length of the monthly forecast horizon
        y_train_output["cols"] = [next(indices_y_test) for i in range(len(y_train_output))]
        y_output_wide = y_train_output.pivot(index = "unique_id", columns = "cols", values = "y")
        
        y_output_wide.drop(to_drop,axis = 0, inplace = True)
        y_train_input_fixed.drop(to_drop, axis = 0, inplace = True)
        
        class_labels = x_train_input.pivot_table(index = "unique_id", values = "x", aggfunc=pd.unique)
        class_labels.drop(to_drop,axis = 0, inplace = True)
        
        # splitting before the denoising as validation and test data are not denoised
        x_train, x_test, y_train, y_test = train_test_split(y_train_input_fixed, y_output_wide,
                                                            test_size = 0.1,
                                                            random_state = 42,
                                                            stratify = class_labels)
        # split train into train/val, 0.1 split
        class_labels = class_labels[class_labels.index.isin(x_train.index)]

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                            test_size = 0.1,
                                                            random_state = 42,
                                                            stratify = class_labels)
        
        # denoising
        if wavelet != "noisy":
            if level != "inf":
                x_train = x_train.apply(dwt_lowpassfilter, axis=1,result_type="broadcast", thresh=threshold, wavelet = wavelet, level=level)
                y_train = y_train.apply(dwt_lowpassfilter, axis=1,result_type="broadcast", thresh=threshold, wavelet = wavelet, level=level)
            else:
                x_train = x_train.apply(dwt_lowpassfilter, axis=1,result_type="broadcast", thresh=threshold, wavelet = wavelet, level=None)
                y_train = y_train.apply(dwt_lowpassfilter, axis=1,result_type="broadcast", thresh=threshold, wavelet = wavelet, level=None)
        # save data with noisy val data, denoise in ml model
        x_train = x_train.to_numpy()
        x_val = x_val.to_numpy()
        x_test = x_test.to_numpy()
        
        y_train = y_train.to_numpy()
        y_val = y_val.to_numpy()
        y_test = y_test.to_numpy()
        
        np.savez(data_path, x_train = x_train, y_train = y_train, x_val = x_val, y_val = y_val, x_test = x_test, y_test = y_test)

    # ml models fuse x_train and x_val, therefore denoise x_val also if one of them is chosen
    if model_type in ["svr", "rfr", "dtc"]:
        print("ML model chosen: "+ model_type)
        ml_model = fit_denoised_ml_model(x_train, y_train, x_val, y_val, x_test,y_test,
                                model_type = model_type,
                                model_name = model_path_name,
                                threshold = threshold,
                                level=level,
                                wavelet = wavelet)
        best_model = ml_model
        test_prediction = best_model.predict(x_test)
        val_prediction = best_model.predict(x_val)
        test_rmse = mean_squared_error(y_test, test_prediction, squared=False) # rmse as per https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html?highlight=mean_squared#sklearn.metrics.mean_squared_error
        val_rmse = mean_squared_error(y_val, val_prediction, squared=False)
        smape_list = []
        for i in range(y_test.shape[0]):
            smape_list.append(smape(y_test[i,:], test_prediction[i,:])) # smape from pmdarima is only defined for a single ts, so iteration away
        test_smape = pd.Series(smape_list).mean()
        val_smape_list = []
        for i in range(y_val.shape[0]):
            val_smape_list.append(smape(y_val[i,:], val_prediction[i,:])) # smape from pmdarima is only defined for a single ts, so iteration away
        test_smape = pd.Series(smape_list).mean()
        val_smape = pd.Series(val_smape_list).mean()
        print('Val. RMSE: {}'.format(val_rmse))
        print('Val. sMAPE: {}'.format(val_smape))
        print('Test RMSE: {}'.format(test_rmse))
        print('Test sMAPE: {}'.format(test_smape))
        
        result_dataframe = pd.DataFrame(np.array([[val_rmse, val_smape],[test_rmse, test_smape]]), columns = ["RMSE", "SMAPE"],index = ["val","test"])
        result_path = "./m4_ts_models/results/" + model_path_name + ".csv"
        result_dataframe.to_csv(result_path)
        print("saved results for model: " + model_path_name)
        
    else:
        print("DL model chosen: " + model_type)
        # todo: look up NN and CNN architecture for TS forecasting
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        OUT_STEPS = 18
        # again from the ml benchmarks for the m4 competition
        batch_size = 20
        epochs = 100 # from https://github.com/Mcompetitions/M4-methods/blob/master/ML_benchmarks.py
        if model_type == "rnn":
            model = Sequential()
            # Shape [batch, time, features] => [batch, lstm_units]
            model.add(LSTM(6, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2]))) 
            # number of units is based on the rnn benchmark from https://github.com/Mcompetitions/M4-methods/blob/master/ML_benchmarks.py
            # and layer layout from https://doi.org/10.1016/j.egyr.2019.11.009
            model.add(keras.layers.Dropout(rate=0.5))
            # Shape => [batch, out_steps*features]
            model.add(Dense(OUT_STEPS))
            # Shape => [batch, out_steps, features]
            print(model.summary())
        elif model_type == "cnn":
            model = Sequential()# based on https://doi.org/10.1016/j.egyr.2019.11.009 , adapted for these timeseries and time constraints, epochs 200->20
            model.add(Conv1D(32, kernel_size=5, activation='relu',
                     input_shape=(x_train.shape[1], x_train.shape[2])))
            model.add(Conv1D(32, kernel_size=5, activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(OUT_STEPS))

            print(model.summary())
        elif model_type == "nn":
            # based on model architecture from https://github.com/Mcompetitions/M4-methods/blob/master/ML_benchmarks.py#L213
            # adapted to keras for ease of implementation of multires model
            m_in = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
            m = Dense(6)(m_in)
            m = Flatten()(m)# because somehow the implicit dimensionality reduction in dense layers isn't working properly
            m_out = Dense(OUT_STEPS, name= "first_out")(m)
            model = keras.Model(
                    inputs=m_in,
                    outputs=m_out)
            print(model.summary())
            
        trained_model, hist = fit_denoised_dl_model(model, x_train, y_train, x_val,
                                             y_val, x_test, y_test, batch_size,
                                             epochs, model_name=model_path_name)
        plot_hist(hist)
        
        
        loading_model_path = "./m4_ts_models/" + model_path_name + ".h5"
        best_model = keras.models.load_model(loading_model_path, compile=False)
    
        test_prediction = best_model.predict(x_test, verbose=0)
        val_prediction = best_model.predict(x_val, verbose=0)
        
        test_rmse = mean_squared_error(y_test, test_prediction, squared=False) # rmse as per https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html?highlight=mean_squared#sklearn.metrics.mean_squared_error
        val_rmse = mean_squared_error(y_val, val_prediction, squared=False)
        smape_list = []
        for i in range(y_test.shape[0]):
            smape_list.append(smape(y_test[i,:], test_prediction[i,:])) # smape from pmdarima is only defined for a single ts, so iteration away
        test_smape = pd.Series(smape_list).mean()
        val_smape_list = []
        for i in range(y_val.shape[0]):
            val_smape_list.append(smape(y_val[i,:], val_prediction[i,:])) # smape from pmdarima is only defined for a single ts, so iteration away
        test_smape = pd.Series(smape_list).mean()
        val_smape = pd.Series(val_smape_list).mean()
        print('Val. RMSE: {}'.format(val_rmse))
        print('Val. sMAPE: {}'.format(val_smape))
        print('Test RMSE: {}'.format(test_rmse))
        print('Test sMAPE: {}'.format(test_smape))
    
        result_dataframe = pd.DataFrame(np.array([[val_rmse, val_smape],[test_rmse, test_smape]]), columns = ["RMSE", "SMAPE"],index = ["val","test"])
        result_path = "./m4_ts_models/results/" + model_path_name + ".csv"
        result_dataframe.to_csv(result_path)
        print("saved results for model: " + model_path_name)
        keras.backend.clear_session()


def ceildiv(a, b):
    return int(-(-a // b))


def batch_wave_dec(data_in, wavelet="db4", level=1):
    if level not in [1,2]:
        print("only levels 1 and 2 are supported currently")
        
    if level == 1:
        wave1 = []
        wave2 = []
        for i in range(data_in.shape[0]):
            t_wave1, t_wave2 = pywt.wavedec(data_in[i,:], wavelet=wavelet, level=level)
            wave1.append(t_wave1)
            wave2.append(t_wave2)
        return np.array(wave1), np.array(wave2)
    elif level == 2:
        wave1 = []
        wave2 = []
        wave3 = []
        for i in range(data_in.shape[0]):
            t_wave1, t_wave2, t_wave3 = pywt.wavedec(data_in[i,:], wavelet=wavelet, level=level)
            wave1.append(t_wave1)
            wave2.append(t_wave2)
            wave3.append(t_wave3)
        return np.array(wave1), np.array(wave2), np.array(wave3)


def batch_wave_rec(list_of_arrays,wavelet="db4"):
    output = []
    for i in range(list_of_arrays[0].shape[0]):
        t_in = []
        for j in range(len(list_of_arrays)):
            t_in.append(list_of_arrays[j][i,:])
        t_out = pywt.waverec(t_in, wavelet)
        output.append(t_out)
    return np.array(output)


def fit_single_res_ml_model(X_train, y_train, X_val, y_val, model_type):
    # adding val data to training data as no val data is used during the fitting of the ml models

    X_train = np.append(X_train, X_val, axis = 0)
    y_train = np.append(y_train, y_val, axis = 0)
    
    if model_type == "svr":
        model = MultiOutputRegressor(SVR(), n_jobs = -1)
    elif model_type == "rfr":
        model = RandomForestRegressor(random_state=42)
    elif model_type == "dtc":
        model = DecisionTreeRegressor(random_state=42)
    else:
        print("how did you get here?")

    model.fit(X_train, y_train)
    print("model is fitted")

    return model



def fit_multires_dl_model(model, x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test, batch_size, n_epochs, model_name):
    model.compile(loss="mean_squared_error",
                  optimizer=keras.optimizers.Adam(),
                  metrics=[RootMeanSquaredError()])
    
    print("compiled")
    path = "./m4_multires_models/" + model_name + '.h5'
    logging_path = "./m4_multires_models/logs/" + model_name + ".log"
    # define callback for model saving
    checkpoint_callback = ModelCheckpoint(filepath= path, monitor='val_first_out_root_mean_squared_error', save_best_only=True)# val. rmse of the approx. coef
    csv_logger_callback = keras.callbacks.CSVLogger(filename=logging_path)

    history  = model.fit(x_train_list, y_train_list,
              batch_size=batch_size,
              epochs=n_epochs,
              verbose=1,
              validation_data=(x_val_list, y_val_list),
              callbacks = [checkpoint_callback, csv_logger_callback],
              shuffle = True)

    return model, history


def plot_hist_level1(history):
    plt.title('root_mean_squared_error of val. and test sets')
    plt.plot(history.history['first_out_root_mean_squared_error'], label='Train1 RMSE')
    plt.plot(history.history['val_first_out_root_mean_squared_error'], label='Val1 RMSE')
    plt.plot(history.history['last_out_root_mean_squared_error'], label='Train2 RMSE')
    plt.plot(history.history['val_last_out_root_mean_squared_error'], label='Val2 RMSE')
    plt.ylabel('root_mean_squared_error')
    plt.xlabel('No. epoch')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


def plot_hist_level2(history):
    plt.title('root_mean_squared_error of val. and test sets')
    plt.plot(history.history['first_out_root_mean_squared_error'], label='Train1 RMSE')
    plt.plot(history.history['val_first_out_root_mean_squared_error'], label='Val1 RMSE')
    plt.plot(history.history['second_out_root_mean_squared_error'], label='Train1 RMSE')
    plt.plot(history.history['val_second_out_root_mean_squared_error'], label='Val1 RMSE')
    plt.plot(history.history['last_out_root_mean_squared_error'], label='Train3 RMSE')
    plt.plot(history.history['val_last_out_root_mean_squared_error'], label='Val3 RMSE')
    plt.ylabel('root_mean_squared_error')
    plt.xlabel('No. epoch')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

def m4_multires_variations(model_type, wavelet="db4", level=1):
    np.random.seed(42)
    rn.seed(42)
    tf.random.set_seed(42)
    
    time ="Monthly"
    data_path_name = time+ "_" + wavelet + "_"  + "_" + str(level)
    data_path = "./m4_multires_models/processed_data/" + data_path_name + ".npz"
    model_path_name = time + "_" + wavelet + "_" + str(level) + "_" + model_type
    
    if os.path.exists(data_path):
        if level == 1:
            print("preprocessed data already exists, reading from file")
            saved_data_file = np.load(data_path)
            x_train1 = saved_data_file["x_train1"]
            x_train2 = saved_data_file["x_train2"]
            y_train1 = saved_data_file["y_train1"]
            y_train2 = saved_data_file["y_train2"]
            x_val = saved_data_file["x_val"]
            x_val1 = saved_data_file["x_val1"]
            x_val2 = saved_data_file["x_val2"]
            y_val = saved_data_file["y_val"]
            y_val1 = saved_data_file["y_val1"]
            y_val2 = saved_data_file["y_val2"]
            x_test1 = saved_data_file["x_test1"]
            x_test2 = saved_data_file["x_test2"]
            y_test = saved_data_file["y_test"]
        elif level == 2:
            print("preprocessed data already exists, reading from file")
            saved_data_file = np.load(data_path)
            x_train1 = saved_data_file["x_train1"]
            x_train2 = saved_data_file["x_train2"]
            x_train3 = saved_data_file["x_train3"]
            y_train1 = saved_data_file["y_train1"]
            y_train2 = saved_data_file["y_train2"]
            y_train3 = saved_data_file["y_train3"]
            x_val = saved_data_file["x_val"]
            x_val1 = saved_data_file["x_val1"]
            x_val2 = saved_data_file["x_val2"]
            x_val3 = saved_data_file["x_val3"]
            y_val = saved_data_file["y_val"]
            y_val1 = saved_data_file["y_val1"]
            y_val2 = saved_data_file["y_val2"]
            y_val3 = saved_data_file["y_val3"]
            x_test1 = saved_data_file["x_test1"]
            x_test2 = saved_data_file["x_test2"]
            x_test3 = saved_data_file["x_test3"]
            y_test = saved_data_file["y_test"]
    
    else:
        print("no preprocessed data found, begin preprocessing...")
        x_train_input, y_train_input, x_train_output, y_train_output = m4_parser(time,"../datasets")
    
        y_input_wide = y_train_input.pivot(index = "unique_id", columns = "ds", values = "y")
        y_input_wide.columns = range(0, 2794)
        
        y_train_input_fixed = pd.DataFrame()
        first_nan = y_input_wide.notna().idxmin(1)
        lowest_numb_of_obs = first_nan.sort_values().head()
        print(lowest_numb_of_obs)
        
        
        
        for i in first_nan.index:    
            y_train_input_fixed = y_train_input_fixed.append(y_input_wide.loc[i,(first_nan[i]-42):(first_nan[i]-1)].reset_index(drop = True)) # min length of an input series is 42, so every series gets cut to length 42
        
        to_drop = y_train_input_fixed[y_train_input_fixed.isnull().any(axis=1)].index[0]
        print(to_drop)
        indices_y_test = cycle([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]) # length of the monthly forecast horizon
        y_train_output["cols"] = [next(indices_y_test) for i in range(len(y_train_output))]
        y_output_wide = y_train_output.pivot(index = "unique_id", columns = "cols", values = "y")
        
        y_output_wide.drop(to_drop,axis = 0, inplace = True)
        y_train_input_fixed.drop(to_drop, axis = 0, inplace = True)
        
        class_labels = x_train_input.pivot_table(index = "unique_id", values = "x", aggfunc=pd.unique)
        class_labels.drop(to_drop,axis = 0, inplace = True)
        
        # splitting before the denoising as validation and test data are not denoised
        x_train, x_test, y_train, y_test = train_test_split(y_train_input_fixed, y_output_wide,
                                                            test_size = 0.1,
                                                            random_state = 42,
                                                            stratify = class_labels)
        # split train into train/val, 0.1 split
        class_labels = class_labels[class_labels.index.isin(x_train.index)]

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                            test_size = 0.1,
                                                            random_state = 42,
                                                            stratify = class_labels)
        x_train = x_train.to_numpy()
        x_val = x_val.to_numpy()
        x_test = x_test.to_numpy()
        
        y_train = y_train.to_numpy()
        y_val = y_val.to_numpy()
        y_test = y_test.to_numpy()
        
        if level == 1:
            x_train1, x_train2 = batch_wave_dec(x_train, wavelet=wavelet, level=level)
            x_val1, x_val2 = batch_wave_dec(x_val, wavelet=wavelet, level=level)
            x_test1, x_test2 = batch_wave_dec(x_test, wavelet=wavelet, level=level)
            
            y_train1, y_train2 = batch_wave_dec(y_train, wavelet=wavelet, level=level)
            y_val1, y_val2 = batch_wave_dec(y_val, wavelet=wavelet, level=level)
            # y_test is not needed as we just wave_rec the prediction
            np.savez(data_path,x_train = x_train, x_train1 = x_train1, x_train2 = x_train2,
                     y_train = y_train,y_train1 = y_train1, y_train2 = y_train2,
                     x_val = x_val,x_val1 = x_val1, x_val2 = x_val2,
                     y_val = y_val,y_val1 = y_val1, y_val2 = y_val2,
                     x_test = x_test,x_test1 = x_test1, x_test2 = x_test2,
                     y_test = y_test)
        
        elif level == 2:
            x_train1, x_train2,x_train3 = batch_wave_dec(x_train, wavelet=wavelet, level=level)
            x_val1, x_val2, x_val3 = batch_wave_dec(x_val, wavelet=wavelet, level=level)
            x_test1, x_test2, x_test3 = batch_wave_dec(x_test, wavelet=wavelet, level=level)
            
            y_train1, y_train2, y_train3 = batch_wave_dec(y_train, wavelet=wavelet, level=level)
            y_val1, y_val2, y_val3 = batch_wave_dec(y_val, wavelet=wavelet, level=level)
            # y_test is not needed as we just wave_rec the prediction
            np.savez(data_path,x_train = x_train, x_train1 = x_train1, x_train2 = x_train2, x_train3 = x_train3,
                     y_train = y_train, y_train1 = y_train1, y_train2 = y_train2, y_train3 = y_train3,
                     x_val = x_val,x_val1 = x_val1, x_val2 = x_val2, x_val3 = x_val3,
                     y_val = y_val,y_val1 = y_val1, y_val2 = y_val2, y_val3 = y_val3,
                     x_test = x_test, x_test1 = x_test1, x_test2 = x_test2, x_test3 = x_test3,
                     y_test = y_test)
            
    # end of all preprocessing check
    if level == 1:
        if model_type in ["rfr","dtc","svr"]:
            ml_model_l1 = fit_single_res_ml_model(x_train1, y_train1, x_val1, y_val1,
                                           model_type=model_type)

            best_model1 = ml_model_l1
            test_prediction1 = best_model1.predict(x_test1)
            val_prediction1 = best_model1.predict(x_val1)
            
            ml_model_l2 = fit_single_res_ml_model(x_train2, y_train2, x_val2, y_val2,
                                                       model_type=model_type)
            
            best_model2 = ml_model_l2
            test_prediction2 = best_model2.predict(x_test2)
            val_prediction2 = best_model2.predict(x_val2)
            
            test_prediction_list = [test_prediction1, test_prediction2]
            val_prediction_list = [val_prediction1, val_prediction2]
            
            test_prediction = batch_wave_rec(test_prediction_list, wavelet=wavelet)# predictions fed as list in the same order as the wavedec spits them out
            val_prediction = batch_wave_rec(val_prediction_list, wavelet=wavelet)
            
            test_rmse = mean_squared_error(y_test, test_prediction, squared=False) # rmse as per https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html?highlight=mean_squared#sklearn.metrics.mean_squared_error
            val_rmse = mean_squared_error(y_val, val_prediction, squared=False)
            smape_list = []
            for i in range(y_test.shape[0]):
                smape_list.append(smape(y_test[i,:], test_prediction[i,:])) # smape from pmdarima is only defined for a single ts, so iteration away
            test_smape = pd.Series(smape_list).mean()
            val_smape_list = []
            for i in range(y_val.shape[0]):
                val_smape_list.append(smape(y_val[i,:], val_prediction[i,:])) # smape from pmdarima is only defined for a single ts, so iteration away
            test_smape = pd.Series(smape_list).mean()
            val_smape = pd.Series(val_smape_list).mean()
            print('Val. RMSE: {}'.format(val_rmse))
            print('Val. sMAPE: {}'.format(val_smape))
            print('Test RMSE: {}'.format(test_rmse))
            print('Test sMAPE: {}'.format(test_smape))
            
            result_dataframe = pd.DataFrame(np.array([[val_rmse, val_smape],[test_rmse, test_smape]]), columns = ["RMSE", "SMAPE"],index = ["val","test"])
            result_path = "./m4_multires_models/results/" + model_path_name + ".csv"
            result_dataframe.to_csv(result_path)
            print("saved results for model: " + model_path_name)
            
        elif model_type in ["cnn","rnn","nn"]:
            x_train1 = x_train1.reshape(x_train1.shape[0], x_train1.shape[1], 1)
            x_train2 = x_train2.reshape(x_train2.shape[0], x_train2.shape[1], 1)
            x_val1 = x_val1.reshape(x_val1.shape[0], x_val1.shape[1], 1)
            x_val2 = x_val2.reshape(x_val2.shape[0], x_val2.shape[1], 1)
            x_test1 = x_test1.reshape(x_test1.shape[0], x_test1.shape[1], 1)
            x_test2 = x_test2.reshape(x_test2.shape[0], x_test2.shape[1], 1)
            
            OUT_STEPS1 = y_train1.shape[1]
            OUT_STEPS2 = y_train2.shape[1]
            batch_size = 20
            epochs = 100
            
            a1_in = keras.Input(shape=(x_train1.shape[1], x_train1.shape[2]))
            d1_in = keras.Input(shape=(x_train2.shape[1], x_train2.shape[2]))
            if model_type == "cnn":
                m1 = Conv1D(32, kernel_size=5, activation='relu')(a1_in)
                m1 = Conv1D(32, kernel_size=5, activation='relu')(m1)
                m1 = MaxPooling1D(pool_size=2)(m1)
                m1 = Flatten()(m1)
                m1_out = Dense(OUT_STEPS1, name= "first_out")(m1)
                
                m2 = Conv1D(32, kernel_size=5, activation='relu')(d1_in)
                m2 = Conv1D(32, kernel_size=5, activation='relu')(m2)
                m2 = MaxPooling1D(pool_size=2)(m2)
                m2 = Flatten()(m2)
                m2_out = Dense(OUT_STEPS2, name= "last_out")(m2)
            elif model_type == "rnn":
                m1 = LSTM(6, return_sequences=False) (a1_in)
                m1 = keras.layers.Dropout(rate=0.5)(m1)
                m1_out = Dense(OUT_STEPS1, name= "first_out")(m1)
                
                m2 = LSTM(6, return_sequences=False) (d1_in)
                m2 = keras.layers.Dropout(rate=0.5)(m2)
                m2_out = Dense(OUT_STEPS2, name= "last_out")(m2)
            elif model_type == "nn":
                m1 = Dense(6) (a1_in)
                m1 = Flatten()(m1)
                m1_out = Dense(OUT_STEPS1, name= "first_out")(m1)
                
                m2 = Dense(6) (d1_in)
                m2 = Flatten()(m2)
                m2_out = Dense(OUT_STEPS2, name= "last_out")(m2)

            model = keras.Model(
                    inputs=[a1_in, d1_in],
                    outputs=[m1_out, m2_out])
            print(model.summary())
            
            # list of numpy.arrays as input for the multi-in-out model
            x_train_list = [x_train1, x_train2]
            y_train_list = [y_train1, y_train2]
            x_val_list = [x_val1, x_val2]
            y_val_list = [y_val1, y_val2]
            x_test_list = [x_test1, x_test2]
            
            trained_model, hist = fit_multires_dl_model(model, x_train_list, y_train_list, x_val_list,
                                         y_val_list, x_test_list, y_test, batch_size,
                                         epochs, model_name=model_path_name)
            
            plot_hist_level1(hist)
            loading_model_path = "./m4_multires_models/" + model_path_name + ".h5"
            best_model = keras.models.load_model(loading_model_path, compile=False)
        
            test_predict= best_model.predict(x_test_list, verbose=0)
            test_prediction = batch_wave_rec(test_predict, wavelet=wavelet)
            val_predict = best_model.predict(x_val_list, verbose=0)
            val_prediction = batch_wave_rec(val_predict, wavelet=wavelet)
            
            test_rmse = mean_squared_error(y_test, test_prediction, squared=False) # rmse as per https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html?highlight=mean_squared#sklearn.metrics.mean_squared_error
            val_rmse = mean_squared_error(y_val, val_prediction, squared=False)
            smape_list = []
            for i in range(y_test.shape[0]):
                smape_list.append(smape(y_test[i,:], test_prediction[i,:])) # smape from pmdarima is only defined for a single ts, so iteration away
            test_smape = pd.Series(smape_list).mean()
            val_smape_list = []
            for i in range(y_val.shape[0]):
                val_smape_list.append(smape(y_val[i,:], val_prediction[i,:])) # smape from pmdarima is only defined for a single ts, so iteration away
            test_smape = pd.Series(smape_list).mean()
            val_smape = pd.Series(val_smape_list).mean()
            print('Val. RMSE: {}'.format(val_rmse))
            print('Val. sMAPE: {}'.format(val_smape))
            print('Test RMSE: {}'.format(test_rmse))
            print('Test sMAPE: {}'.format(test_smape))
        
            result_dataframe = pd.DataFrame(np.array([[val_rmse, val_smape],[test_rmse, test_smape]]), columns = ["RMSE", "SMAPE"],index = ["val","test"])
            result_path = "./m4_multires_models/results/" + model_path_name + ".csv"
            result_dataframe.to_csv(result_path)
            print("saved results for model: " + model_path_name)
            keras.backend.clear_session()
        
        
    elif level == 2:
        if model_type in ["rfr","dtc","svr"]:
            # predict approx. coeff of level 2
            ml_model_l1 = fit_single_res_ml_model(x_train1, y_train1, x_val1, y_val1,
                                           model_type=model_type)

            best_model1 = ml_model_l1
            test_prediction1 = best_model1.predict(x_test1)
            val_prediction1 = best_model1.predict(x_val1)
            
            # predict detail coeff. of level 2
            ml_model_l2 = fit_single_res_ml_model(x_train2, y_train2, x_val2, y_val2,
                                                       model_type=model_type)
            
            best_model2 = ml_model_l2
            test_prediction2 = best_model2.predict(x_test2)
            val_prediction2 = best_model2.predict(x_val2)
            
            # predict detail coeff. of level 1
            ml_model_l3 = fit_single_res_ml_model(x_train3, y_train3, x_val3, y_val3,
                                                       model_type=model_type)
            
            best_model3 = ml_model_l3
            test_prediction3 = best_model3.predict(x_test3)
            val_prediction3 = best_model3.predict(x_val3)
            
            # concat. predictions into list and feed it into the waverec
            test_prediction_list = [test_prediction1, test_prediction2, test_prediction3]
            val_prediction_list = [val_prediction1, val_prediction2, val_prediction3]
            
            test_prediction = batch_wave_rec(test_prediction_list, wavelet=wavelet)
            val_prediction = batch_wave_rec(val_prediction_list, wavelet=wavelet)
            
            # calc. rmse and smape on the reconstructed test results
            test_rmse = mean_squared_error(y_test, test_prediction, squared=False) # rmse as per https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html?highlight=mean_squared#sklearn.metrics.mean_squared_error
            val_rmse = mean_squared_error(y_val, val_prediction, squared=False)
            smape_list = []
            for i in range(y_test.shape[0]):
                smape_list.append(smape(y_test[i,:], test_prediction[i,:])) # smape from pmdarima is only defined for a single ts, so iteration away
            test_smape = pd.Series(smape_list).mean()
            val_smape_list = []
            for i in range(y_val.shape[0]):
                val_smape_list.append(smape(y_val[i,:], val_prediction[i,:])) # smape from pmdarima is only defined for a single ts, so iteration away
            test_smape = pd.Series(smape_list).mean()
            val_smape = pd.Series(val_smape_list).mean()
            print('Val. RMSE: {}'.format(val_rmse))
            print('Val. sMAPE: {}'.format(val_smape))
            print('Test RMSE: {}'.format(test_rmse))
            print('Test sMAPE: {}'.format(test_smape))
            
            result_dataframe = pd.DataFrame(np.array([[val_rmse, val_smape],[test_rmse, test_smape]]), columns = ["RMSE", "SMAPE"],index = ["val","test"])
            result_path = "./m4_multires_models/results/" + model_path_name + ".csv"
            result_dataframe.to_csv(result_path)
            print("saved results for model: " + model_path_name)
            
        elif model_type in ["cnn","rnn","nn"]:
            x_train1 = x_train1.reshape(x_train1.shape[0], x_train1.shape[1], 1)
            x_train2 = x_train2.reshape(x_train2.shape[0], x_train2.shape[1], 1)
            x_train3 = x_train3.reshape(x_train3.shape[0], x_train3.shape[1], 1)
            x_val1 = x_val1.reshape(x_val1.shape[0], x_val1.shape[1], 1)
            x_val2 = x_val2.reshape(x_val2.shape[0], x_val2.shape[1], 1)
            x_val3 = x_val3.reshape(x_val3.shape[0], x_val3.shape[1], 1)
            x_test1 = x_test1.reshape(x_test1.shape[0], x_test1.shape[1], 1)
            x_test2 = x_test2.reshape(x_test2.shape[0], x_test2.shape[1], 1)
            x_test3 = x_test3.reshape(x_test3.shape[0], x_test3.shape[1], 1)
            
            OUT_STEPS1 = y_train1.shape[1]
            OUT_STEPS2 = y_train2.shape[1]
            OUT_STEPS3 = y_train3.shape[1]
            batch_size = 20
            epochs = 100
            
            a2_in = keras.Input(shape=(x_train1.shape[1], x_train1.shape[2]))
            d2_in = keras.Input(shape=(x_train2.shape[1], x_train2.shape[2]))
            d1_in = keras.Input(shape=(x_train3.shape[1], x_train3.shape[2]))
            if model_type == "cnn":
                m1 = Conv1D(32, kernel_size=5, activation='relu')(a2_in)
                m1 = Conv1D(32, kernel_size=5, activation='relu')(m1)
                m1 = MaxPooling1D(pool_size=2)(m1)
                m1 = Flatten()(m1)
                m1_out = Dense(OUT_STEPS1, name= "first_out")(m1)
                
                m2 = Conv1D(32, kernel_size=5, activation='relu')(d2_in)
                m2 = Conv1D(32, kernel_size=5, activation='relu')(m2)
                m2 = MaxPooling1D(pool_size=2)(m2)
                m2 = Flatten()(m2)
                m2_out = Dense(OUT_STEPS2, name= "second_out")(m2)
                
                m3 = Conv1D(32, kernel_size=5, activation='relu')(d1_in)
                m3 = Conv1D(32, kernel_size=5, activation='relu')(m3)
                m3 = MaxPooling1D(pool_size=2)(m3)
                m3 = Flatten()(m3)
                m3_out = Dense(OUT_STEPS3, name= "last_out")(m3)
                
            elif model_type == "rnn":
                m1 = LSTM(6, return_sequences=False) (a2_in)
                m1 = keras.layers.Dropout(rate=0.5)(m1)
                m1_out = Dense(OUT_STEPS1, name= "first_out")(m1)
                
                m2 = LSTM(6, return_sequences=False) (d2_in)
                m2 = keras.layers.Dropout(rate=0.5)(m2)
                m2_out = Dense(OUT_STEPS2, name= "second_out")(m2)
                
                m3 = LSTM(6, return_sequences=False) (d1_in)
                m3 = keras.layers.Dropout(rate=0.5)(m3)
                m3_out = Dense(OUT_STEPS3, name= "last_out")(m3)
            elif model_type == "nn":
                m1 = Dense(6) (a2_in)
                m1 = Flatten()(m1)
                m1_out = Dense(OUT_STEPS1, name= "first_out")(m1)
                
                m2 = Dense(6) (d2_in)
                m2 = Flatten()(m2)
                m2_out = Dense(OUT_STEPS2, name= "second_out")(m2)
                
                m3 = Dense(6) (d1_in)
                m3 = Flatten()(m3)
                m3_out = Dense(OUT_STEPS3, name= "last_out")(m3)


            model = keras.Model(
                    inputs=[a2_in, d2_in, d1_in],
                    outputs=[m1_out, m2_out, m3_out])
            print(model.summary())
            
            # list of numpy.arrays as input for the multi-in-out model
            x_train_list = [x_train1, x_train2, x_train3]
            y_train_list = [y_train1, y_train2, y_train3]
            x_val_list = [x_val1, x_val2, x_val3]
            y_val_list = [y_val1, y_val2, y_val3]
            x_test_list = [x_test1, x_test2, x_test3]
            
            trained_model, hist = fit_multires_dl_model(model, x_train_list, y_train_list, x_val_list,
                                         y_val_list, x_test_list, y_test, batch_size,
                                         epochs, model_name=model_path_name)
            
            plot_hist_level2(hist)
            loading_model_path = "./m4_multires_models/" + model_path_name + ".h5"
            best_model = keras.models.load_model(loading_model_path, compile=False)
        
            test_predict= best_model.predict(x_test_list, verbose=0)
            test_prediction = batch_wave_rec(test_predict, wavelet=wavelet)
            val_predict = best_model.predict(x_val_list, verbose=0)
            val_prediction = batch_wave_rec(val_predict, wavelet=wavelet)
            
            test_rmse = mean_squared_error(y_test, test_prediction, squared=False) # rmse as per https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html?highlight=mean_squared#sklearn.metrics.mean_squared_error
            val_rmse = mean_squared_error(y_val, val_prediction, squared=False)
            smape_list = []
            for i in range(y_test.shape[0]):
                smape_list.append(smape(y_test[i,:], test_prediction[i,:])) # smape from pmdarima is only defined for a single ts, so iteration away
            test_smape = pd.Series(smape_list).mean()
            val_smape_list = []
            for i in range(y_val.shape[0]):
                val_smape_list.append(smape(y_val[i,:], val_prediction[i,:])) # smape from pmdarima is only defined for a single ts, so iteration away
            test_smape = pd.Series(smape_list).mean()
            val_smape = pd.Series(val_smape_list).mean()
            print('Val. RMSE: {}'.format(val_rmse))
            print('Val. sMAPE: {}'.format(val_smape))
            print('Test RMSE: {}'.format(test_rmse))
            print('Test sMAPE: {}'.format(test_smape))
        
            result_dataframe = pd.DataFrame(np.array([[val_rmse, val_smape],[test_rmse, test_smape]]), columns = ["RMSE", "SMAPE"],index = ["val","test"])
            result_path = "./m4_multires_models/results/" + model_path_name + ".csv"
            result_dataframe.to_csv(result_path)
            print("saved results for model: " + model_path_name)
            keras.backend.clear_session()
    elif level not in [1,2]:
        print("please try again with level == 1 or 2")
   
            

def m4_baseline_arima(model_type):
    """
    no dumping of models as we compute ~ 4800 as baseline
    """
    time ="Monthly"
    x_train_input, y_train_input, x_train_output, y_train_output = m4_parser(time,"../datasets")

    y_input_wide = y_train_input.pivot(index = "unique_id", columns = "ds", values = "y")
    y_input_wide.columns = range(0, 2794)
    
    y_train_input_fixed = pd.DataFrame()
    first_nan = y_input_wide.notna().idxmin(1)
    lowest_numb_of_obs = first_nan.sort_values().head()
    print(lowest_numb_of_obs)
    
    
    
    for i in first_nan.index:    
        y_train_input_fixed = y_train_input_fixed.append(y_input_wide.loc[i,(first_nan[i]-42):(first_nan[i]-1)].reset_index(drop = True)) # min length of an input series is 42, so every series gets cut to length 42
    
    to_drop = y_train_input_fixed[y_train_input_fixed.isnull().any(axis=1)].index[0]
    print(to_drop)
    indices_y_test = cycle([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]) # length of the monthly forecast horizon
    y_train_output["cols"] = [next(indices_y_test) for i in range(len(y_train_output))]
    y_output_wide = y_train_output.pivot(index = "unique_id", columns = "cols", values = "y")
    
    y_output_wide.drop(to_drop,axis = 0, inplace = True)
    y_train_input_fixed.drop(to_drop, axis = 0, inplace = True)
    
    class_labels = x_train_input.pivot_table(index = "unique_id", values = "x", aggfunc=pd.unique)
    class_labels.drop(to_drop,axis = 0, inplace = True)
    
    # splitting before the denoising as validation and test data are not denoised
    x_train, x_test, y_train, y_test = train_test_split(y_train_input_fixed, y_output_wide,
                                                        test_size = 0.1,
                                                        random_state = 42,
                                                        stratify = class_labels)
    # split train into train/val, 0.1 split
    class_labels = class_labels[class_labels.index.isin(x_train.index)]

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                        test_size = 0.1,
                                                        random_state = 42,
                                                        stratify = class_labels)
    
    
    
    test_rmse = []
    test_smape = []
    raw_forecasts = []
    errors = []
    if model_type == "unscaled":
        data_path = "./m4_baseline/processed_data/unscaled"
        np.savez(data_path, x_train = x_train, y_train = y_train, x_val = x_val, y_val = y_val, x_test = x_test, y_test = y_test)
        for i in range(x_test.shape[0]):
            print(i)
            try:
                test_step_fit = pm.auto_arima(x_test.iloc[i,:], start_p=1, start_q=1, max_p=5, max_q=5, m=12,
                                       start_P=0, seasonal=True, d=1, max_D=12,max_P=3,max_Q=3, # stepwise can't run in parallel
                                       error_action='ignore',  # don't want to know if an order does not work
                                       suppress_warnings=True,  # don't want convergence warnings
                                       stepwise=True)
                step_pred = test_step_fit.predict(n_periods=18)
                temp_rmse = np.sqrt(mean_squared_error(y_test.iloc[i, :], step_pred))
                temp_smape = smape(y_test.iloc[0, :], step_pred)
                raw_forecasts.append(step_pred)
                test_rmse.append(temp_rmse)
                test_smape.append(temp_smape)
            except:
                print("error in row "+ str(i))
                errors.append([str(i)])
                continue
                
        raw_df = pd.DataFrame(raw_forecasts)
        raw_df.to_csv("./m4_baseline/raw_forecasts/unscaled.csv", index=False)
    elif model_type == "scaled":
        print("todo:pipeline maybe with a scaler, the 4800 ts make it hard to pick a normalizer")
    else: print("pick either scaled or unscaled as model_type")
    output_df = pd.DataFrame()
    output_df["RMSE"] = test_rmse
    output_df["sMAPE"] = test_smape
    print("Mean Test RMSE: %.3f" % output_df.RMSE.mean())
    print("Mean Test sMAPE: %.3f" % output_df.sMAPE.mean())
    # RMSE and sMAPE as metrics, (R)MSE as fitting metric for dl/ml models, sMAPE only to have a second metric
    # take the mean of the output_df.csv to get the same metric as the ML/DL models
    output_path = "./m4_baseline/results/" + model_type + ".csv"
    output_df.to_csv(output_path, index = False)
    return errors