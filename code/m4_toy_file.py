# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 10:47:48 2021

@author: Joshua
"""
import m4_f
from itertools import cycle
import pywt
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
import pmdarima as pm
import numpy as np
from sklearn.metrics import mean_squared_error
import time
from pmdarima.metrics import smape

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


x_train_input, y_train_input, x_train_output, y_train_output = m4_f.m4_parser("Monthly","../datasets")

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

x_train, x_test, y_train, y_test = train_test_split(y_train_input_fixed, y_output_wide,
                                                    test_size = 0.1,
                                                    random_state = 42,
                                                    stratify = class_labels)


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


#-----------------------------------------------------------------------------
# wavelet/fft tests
test_series = x_train[0,:]
wave1,wave2,wave3 = pywt.wavedec(test_series, wavelet = "haar", level = 2)

test_series_rfft = np.fft.rfft(test_series)

# first image of a toy series and then the real series of train to show where it can fail
plt.title('Time-series in the time domain and the FFT transform in the frequency domain')
plt.subplot(1,2,1)
plt.plot(test_series, label='Time-series')
plt.ylabel('value')
plt.xlabel('time')

plt.subplot(1,2,2)
plt.plot(np.abs(test_series_rfft), label='FFT of a real valued time-series')
plt.ylabel('power')
plt.xlabel('frequency')
plt.tight_layout()
plt.show()


test_out_series = x_test[0,:]
test_out_wave = pywt.wavedec(test_out_series, wavelet = "db4", level = 1)
rec_test_out_series = pywt.waverec(test_out_wave, "haar")

#------------------------------------------------------------------------------
# multires tests
def ceildiv(a, b):
    return int(-(-a // b))


def batch_wave_dec(data_in, wavelet="db4", level=1):
    if level not in [1,2]:
        print("only levels 1 and 2 are currently supported")
        
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


bwave1, bwave2 = batch_wave_dec(x_test)
test_bwave = [bwave1,bwave2]
rec_x_test = batch_wave_rec(test_bwave)


# -----------------------------------------------------------------------------
# dl multires model tests
x_train1, x_train2 = batch_wave_dec(x_train, wavelet="db4", level=1)
x_val1, x_val2 = batch_wave_dec(x_val, wavelet="db4", level=1)
x_test1, x_test2 = batch_wave_dec(x_test, wavelet="db4", level=1)

y_train1, y_train2 = batch_wave_dec(y_train, wavelet="db4", level=1)
y_val1, y_val2 = batch_wave_dec(y_val, wavelet="db4", level=1)
# y_test is not needed as we just wave_rec the prediction

x_train1 = x_train1.reshape(x_train1.shape[0], x_train1.shape[1], 1)
x_train2 = x_train2.reshape(x_train2.shape[0], x_train2.shape[1], 1)
x_val1 = x_val1.reshape(x_val1.shape[0], x_val1.shape[1], 1)
x_val2 = x_val2.reshape(x_val2.shape[0], x_val2.shape[1], 1)
x_test1 = x_test1.reshape(x_test1.shape[0], x_test1.shape[1], 1)
x_test2 = x_test2.reshape(x_test2.shape[0], x_test2.shape[1], 1)


OUT_STEPS1 = y_train1.shape[1]
OUT_STEPS2 = y_train2.shape[1]

a1_in = keras.Input(shape=(x_train1.shape[1], x_train1.shape[2]))
d1_in = keras.Input(shape=(x_train2.shape[1], x_train2.shape[2]))
model_type = "rnn"
if model_type == "cnn":
    m1 = Conv1D(32, kernel_size=5, activation='relu')(a1_in)
    m1 = Conv1D(32, kernel_size=5, activation='relu')(m1)
    m1 = MaxPooling1D(pool_size=2)(m1)
    m1 = Flatten()(m1)
    m1_out = Dense(OUT_STEPS1, name= "a1_out")(m1)
    
    m2 = Conv1D(32, kernel_size=5, activation='relu')(d1_in)
    m2 = Conv1D(32, kernel_size=5, activation='relu')(m2)
    m2 = MaxPooling1D(pool_size=2)(m2)
    m2 = Flatten()(m2)
    m2_out = Dense(OUT_STEPS2)(m2)
elif model_type == "rnn":
    m1 = LSTM(6, return_sequences=False) (a1_in)
    m1 = keras.layers.Dropout(rate=0.5)(m1)
    m1_out = Dense(OUT_STEPS1, name= "a1_out")(m1)
    
    m2 = LSTM(6, return_sequences=False) (d1_in)
    m2 = keras.layers.Dropout(rate=0.5)(m2)
    m2_out = Dense(OUT_STEPS2, name= "d1_out")(m2)


model = keras.Model(
        inputs=[a1_in, d1_in],
        outputs=[m1_out, m2_out])
print(model.summary())

model.compile(loss="mean_squared_error",
                  optimizer=keras.optimizers.Adam(),
                  metrics=[RootMeanSquaredError()])

history  = model.fit([x_train1,x_train2], [y_train1,y_train2],
              batch_size=20,
              epochs=10,
              verbose=1,
              validation_data=([x_val1,x_val2], [y_val1,y_val2]),
              shuffle = True)

def plot_hist_level1(history):
    plt.title('root_mean_squared_error of val. and test sets')
    plt.plot(history.history['dense_1_root_mean_squared_error'], label='Train1 RMSE')
    plt.plot(history.history['val_dense_1_root_mean_squared_error'], label='Val1 RMSE')
    plt.plot(history.history['dense_2_root_mean_squared_error'], label='Train2 RMSE')
    plt.plot(history.history['val_dense_2_root_mean_squared_error'], label='Val2 RMSE')
    plt.ylabel('root_mean_squared_error')
    plt.xlabel('No. epoch')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

plot_hist_level1(history)

test_predict = model.predict([x_test1,x_test2])
test_prediction = batch_wave_rec(test_predict, wavelet="db4")


val_predict = model.predict([x_val1,x_val2], verbose=0)
val_prediction = batch_wave_rec(val_predict, wavelet="db4")

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



# test for a model with level 2
saved_data_file = np.load("./m4_multires_models/processed_data/Monthly_haar__2.npz")
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

x_train_list = [x_train1, x_train2, x_train3]
y_train_list = [y_train1, y_train2, y_train3]
x_val_list = [x_val1, x_val2, x_val3]
y_val_list = [y_val1, y_val2, y_val3]
x_test_list = [x_test1, x_test2, x_test3]

model_path_name = "Monthly_haar_2_rnn"
loading_model_path = "./m4_multires_models/"+ model_path_name + ".h5"
best_model = keras.models.load_model(loading_model_path, compile=False)

best_model.summary()

test_predict= best_model.predict(x_test_list, verbose=0)
test_prediction = batch_wave_rec(test_predict, wavelet="haar")
val_predict = best_model.predict(x_val_list, verbose=0)
val_prediction = batch_wave_rec(val_predict, wavelet="haar")

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


# -----------------------------------------------------------------------------
# multires. ml models tests
x_train1, x_train2 = batch_wave_dec(x_train, wavelet="db4", level=1)
x_val1, x_val2 = batch_wave_dec(x_val, wavelet="db4", level=1)
x_test1, x_test2 = batch_wave_dec(x_test, wavelet="db4", level=1)

y_train1, y_train2 = batch_wave_dec(y_train, wavelet="db4", level=1)
y_val1, y_val2 = batch_wave_dec(y_val, wavelet="db4", level=1)

model_type = "rfr"
ml_model_l1 = m4_f.fit_single_res_ml_model(x_train1, y_train1, x_val1, y_val1,
                                           model_type=model_type)

best_model1 = ml_model_l1
test_prediction1 = best_model1.predict(x_test1)
val_prediction1 = best_model1.predict(x_val1)

ml_model_l2 = m4_f.fit_single_res_ml_model(x_train2, y_train2, x_val2, y_val2,
                                           model_type=model_type)

best_model2 = ml_model_l2
test_prediction2 = best_model2.predict(x_test2)
val_prediction2 = best_model2.predict(x_val2)

test_prediction_list = [test_prediction1, test_prediction2]
val_prediction_list = [val_prediction1, val_prediction2]

test_prediction = batch_wave_rec(test_prediction_list, wavelet="db4")
val_prediction = batch_wave_rec(val_prediction_list, wavelet="db4")


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
        







# Model architecture:
# Prediction by a Hybrid of Wavelet Transform and
# Long-Short-Term-Memory Neural Network
#  - single layer lstm rnn
#  - training on the decomposed ts (Haar wavelet)

# Combining the Real-Time Wavelet Denoising and
# Long-Short-Term-Memory Neural Network for
# Predicting Stock Indexes
#  - two-layer lstm
#  - denoising with thresholding, so no inverse
#  - sym, db, coif and (r)bio wavelets

# Multiresolution analysis based on wavelet transform for commodity prices
# time series forecasting
#  - arima on data reconstructed from a single level

# Financial Time Series Prediction Based on Deep Learning
#  - wavelet-denoising with thresholding, prediction for noisy data still better

# possible setup:
# no denoising
# denoising with wavelet thresholding (sym, db, coif), no invertability, different wavelets and thresholds
# decomposition, prediction with multiple models on the approximation components, different wavelets and levels
# decomposition, prediction on only one approximation component. different level of the approx. comp. and diff. wavelets

test_signals = x_train.iloc[:50,:]
test_signals = test_signals.apply(m4_f.dwt_lowpassfilter, axis=1,result_type="broadcast", thresh=0.2, wavelet = "db4")
test_signals2 = x_train.iloc[:50,:]

plt.plot(test_signals.iloc[12,:], label='denoised')
plt.plot(x_train.iloc[12,:], label='noisy')
plt.legend(loc="lower left")
plt.show()


#---------------------------------
# pmdarima auto_arima trials

pm_x_test = x_test.iloc[0,:].astype(np.float64)
start_time = time.time()
test_rs_fit = pm.auto_arima(pm_x_test, start_p=1, start_q=1, max_p=2, max_q=2, m=12,
                       start_P=0, seasonal=True, d=1,max_D=12, trace=True,
                       n_jobs=-1,  # We can run this in parallel by controlling this option
                       error_action='ignore',  # don't want to know if an order does not work
                       suppress_warnings=True,  # don't want convergence warnings
                       stepwise=False, random=True, random_state=42)
test_rs_fit.summary()
rs_prediction = test_rs_fit.predict(n_periods=18)
print("Test RMSE: %.3f" % np.sqrt(mean_squared_error(y_test.iloc[0,:], rs_prediction)))
print("--- %s seconds ---" % (time.time() - start_time))

plt.plot(rs_prediction, label='rs_predicted')
plt.plot(y_test.iloc[0,:], label='true_value')
plt.legend(loc="lower left")
plt.show()

start_time = time.time()
test_step_fit = pm.auto_arima(x_test.iloc[0,:], start_p=1, start_q=1, max_p=2, max_q=2, m=12,
                       start_P=0, seasonal=True, d=1,max_D=12, # We can't run this in parallel by controlling this option
                       error_action='ignore',  # don't want to know if an order does not work
                       suppress_warnings=True,  # don't want convergence warnings
                       stepwise=True)
# test_step_fit.summary()
step_pred = test_step_fit.predict(n_periods=18)
print("Test RMSE: %.3f" % np.sqrt(mean_squared_error(y_test.iloc[0,:], step_pred)))
print("Test sMAPE: %.3f" % smape(y_test.iloc[0,:], step_pred))
print("--- %s seconds ---" % (time.time() - start_time))


plt.plot(step_pred, label='rs_predicted')
plt.plot(y_test.iloc[0,:], label='true_value')
plt.legend(loc="lower left")
plt.show()
# Print the error:

test_rmse = []
test_smape = []
raw_forecasts = []
model_type = "unscaled"
if model_type == "unscaled":
    #data_path = "./m4_baseline/processed_data/unscaled"
    #np.savez(data_path, x_train = x_train, y_train = y_train, x_val = x_val, y_val = y_val, x_test = x_test, y_test = y_test)
    for i in range(10):
        print(i)
        test_step_fit = pm.auto_arima(x_test.iloc[i,:], start_p=1, start_q=1, max_p=2, max_q=2, m=12,
                               start_P=0, seasonal=True, d=1, max_D=12, # stepwise can't run in parallel
                               error_action='ignore',  # don't want to know if an order does not work
                               suppress_warnings=True,  # don't want convergence warnings
                               stepwise=True)
        step_pred = test_step_fit.predict(n_periods=18)
        temp_rmse = np.sqrt(mean_squared_error(y_test.iloc[i, :], step_pred))
        temp_smape = smape(y_test.iloc[0, :], step_pred)
        raw_forecasts.append(step_pred)
        test_rmse.append(temp_rmse)
        test_smape.append(temp_smape)
elif model_type == "scaled":
    print("todo:pipeline maybe with a scaler, the 4800 ts make it hard to pick a normalizer")
else: print("pick either scaled or unscaled as model_type")

output_df = pd.DataFrame()
output_df["RMSE"] = test_rmse
output_df["sMAPE"] = test_smape
print("Mean Test RMSE: %.3f" % output_df.RMSE.mean())
print("Mean Test sMAPE: %.3f" % output_df.sMAPE.mean())

test_df = pd.DataFrame(raw_forecasts)

test_pred = np.ones((100,18))
test_y = np.full(shape = (100,18), fill_value = 10)
mean_squared_error(test_y, test_pred, squared=False)
smape_list = []
for i in range(test_y.shape[0]):
    smape_list.append(smape(test_y[i,:], test_pred[i,:]))
pd.Series(smape_list).mean()


#-----------------------------------------------------------------
model_type ="svr"
wavelet="db4"
threshold=0.2

time ="Monthly"
data_path_name = time+ "_" + wavelet + "_" + str(threshold)
data_path = "./m4_ts_models/processed_data/" + data_path_name + ".npz"
model_path_name = time + "_" + wavelet + "_" + str(threshold) + "_" + model_type

print("preprocessed data already exists, reading from file")
saved_data_file = np.load(data_path)
x_train = saved_data_file["x_train"]
y_train = saved_data_file["y_train"]
x_val = saved_data_file["x_val"]
y_val = saved_data_file["y_val"]
x_test = saved_data_file["x_test"]
y_test = saved_data_file["y_test"]

x_val = np.apply_along_axis(m4_f.dwt_lowpassfilter, axis=1, arr = x_val, thresh=threshold, wavelet = wavelet)
y_val = np.apply_along_axis(m4_f.dwt_lowpassfilter, axis=1, arr = y_val, thresh=threshold, wavelet = wavelet)
    
x_train = np.append(x_train, x_val, axis = 0)
y_train = np.append(y_train, y_val, axis = 0)



# ml models fuse x_train and x_val, therefore denoise x_val also if one of them is chosen
if model_type in ["svr", "rfr","gbr"]:
    print("ML model chosen: "+ model_type)
    ml_model = m4_f.fit_ml_model(x_train, y_train, x_val, y_val, x_test,y_test,
                            model_type = model_type,
                            model_name = model_path_name,
                            threshold = threshold,
                            wavelet = wavelet)
    best_model = ml_model
    test_prediction = best_model.predict(x_test)
    test_rmse = mean_squared_error(y_test, test_prediction, squared=False) # rmse as per https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html?highlight=mean_squared#sklearn.metrics.mean_squared_error
    smape_list = []
    for i in range(y_test.shape[0]):
        smape_list.append(smape(y_test[i,:], test_prediction[i,:])) # smape from pmdarima is only defined for a single ts, so iteration away
    test_smape = pd.Series(smape_list).mean()
    print('Test RMSE: {}'.format(test_rmse))
    print('Test sMAPE: {}'.format(test_smape))

    result_dataframe = pd.DataFrame(np.array([test_rmse, test_smape]), index = ["RMSE", "SMAPE"],columns = ["test"])
    result_path = "./uci_har_feat_models/results/" + model_path_name + ".csv"
    result_dataframe.to_csv(result_path)
    print("saved results for model: " + model_path_name)