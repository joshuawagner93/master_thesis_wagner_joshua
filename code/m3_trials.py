# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:33:15 2020

@author: Joshua

M4 trial file for preprocessing and first runs with models
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from itertools import cycle

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


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


x_train_input, y_train_input, x_train_output, y_train_output = m4_parser("Yearly","../datasets")


# try first with test set then train set
# x/y_train contained the input, y_test the output which is to be predicted, now renamed to x/y_train_input/output

# todo:
# paste last 7 inputs before the output, therefore our model always predicts the next 
# import batch generator from tutorial
# split into train/val/test datasets
# run model from tut. on it

y_input_wide = y_train_input.pivot(index = "unique_id", columns = "ds", values = "y")
y_input_wide.columns = range(0, 835)

y_train_input_fixed = pd.DataFrame()
first_nan = y_input_wide.notna().idxmin(1)

for i in first_nan.index:    
    y_train_input_fixed = y_train_input_fixed.append(y_input_wide.loc[i,(first_nan[i]-13):(first_nan[i]-1)].reset_index(drop = True)) # min length of an input series is 13, so every series gets cut to length 13

indices_y_test = cycle([1,2,3,4,5,6])
y_train_output["cols"] = [next(indices_y_test) for i in range(len(y_train_output))]
y_output_wide = y_train_output.pivot(index = "unique_id", columns = "cols", values = "y")

x_data = y_train_input_fixed.values
y_data = y_output_wide.values

X_train, X_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size = 0.1, random_state = 42, shuffle = False)

# adds the last 7 entries from X to Y so that we always predict 6 hours into the future
new_y_train = X_train[:,-7:]
new_y_train = np.append(new_y_train,y_train, axis = 1)

new_y_test = X_test[:,-7:]
new_y_test = np.append(new_y_test,y_test, axis = 1)

X_train = np.delete(X_train, 3547, 0)
X_train_row_mean = X_train.mean(1)
X_train_row_std = X_train.std(1)
for i in range(len(X_train)):
    X_train[i,:] = (X_train[i,:] - [X_train_row_mean[i]]) / (X_train_row_std[i]+0.01)

y_train = np.delete(y_train, 3547, 0)
y_train_row_mean = y_train.mean(1)
y_train_row_std = y_train.std(1)
for i in range(len(y_train)):
    y_train[i,:] = (y_train[i,:] - [y_train_row_mean[i]]) / (y_train_row_std[i]+ 0.01)
    
X_test_row_mean = X_test.mean(1)
X_test_row_std = X_test.std(1)
for i in range(len(X_test)):
    X_test[i,:] = (X_test[i,:] - [X_test_row_mean[i]]) / (X_test_row_std[i]+0.01)
    
y_test_row_mean =y_test.mean(1)
y_test_row_std = y_test.std(1)
for i in range(len(y_test)):
    y_test[i,:] = (y_test[i,:] - [y_test_row_mean[i]]) / (y_test_row_std[i]+0.01)

X_train = X_train[...,np.newaxis]
X_test = X_test[...,np.newaxis]

# y_* with the last 7 entries from x_* for the 1-hour ahead predictions from model
#y_train = new_y_train[...,np.newaxis]
#y_test = new_y_test[...,np.newaxis]

# normal y_* for the single-shot model (multi_lstm_model)
y_train = y_train[...,np.newaxis]
y_test = y_test[...,np.newaxis]

print(X_train.shape)
print(y_train.shape)

X_test.shape

validation_data = (X_test,y_test)

model = Sequential()
model.add(GRU(units= 512,
              return_sequences= True,
              input_shape=(None, 1,)))
from tensorflow.python.keras.initializers import RandomUniform
init = RandomUniform(minval=-0.05, maxval=0.05)


model.add(Dense(num_y_signals, activation='sigmoid'))
model.compile(loss = "mean_squared_error", optimizer = "Adam")
model.summary()

# Checkpoints
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=2, verbose=1)

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)
callbacks = [callback_early_stopping]

model.fit(x=X_train,
          y=y_train,
          epochs=20,
          batch_size=20,
          validation_data=validation_data,
          callbacks=callbacks)


result = model.evaluate(x=X_test,
                        y=new_y_test)

print("loss (test-set):", result)

OUT_STEPS = 6
num_features = 1
multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=2,
                                                    mode='min')

multi_lstm_model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

history = multi_lstm_model.fit(x=X_train, y=y_train, epochs=20,
                      validation_data=validation_data,
                      callbacks=[early_stopping])

result = multi_lstm_model.evaluate(x=X_test,
                        y=y_test)

