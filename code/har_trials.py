# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:39:18 2020

@author: Joshua
"""
import pywt
import os
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import History

def read_signals_ucihar(filename):
    with open(filename, 'r') as fp:
        data = fp.read().splitlines()
        data = map(lambda x: x.rstrip().lstrip().split(), data)
        data = [list(map(float, line)) for line in data]
    return data

def read_labels_ucihar(filename):        
    with open(filename, 'r') as fp:
        activities = fp.read().splitlines()
        activities = list(map(int, activities))
    return activities

def load_ucihar_data(folder):
    train_folder = folder + 'train/Inertial Signals/'
    test_folder = folder + 'test/Inertial Signals/'
    labelfile_train = folder + 'train/y_train.txt'
    labelfile_test = folder + 'test/y_test.txt'
    train_signals, test_signals = [], []
    for input_file in os.listdir(train_folder):
        signal = read_signals_ucihar(train_folder + input_file)
        train_signals.append(signal)
    train_signals = np.transpose(np.array(train_signals), (1, 2, 0))
    for input_file in os.listdir(test_folder):
        signal = read_signals_ucihar(test_folder + input_file)
        test_signals.append(signal)
    test_signals = np.transpose(np.array(test_signals), (1, 2, 0))
    train_labels = read_labels_ucihar(labelfile_train)
    test_labels = read_labels_ucihar(labelfile_test)
    return train_signals, train_labels, test_signals, test_labels

folder_ucihar = '../datasets/uci_har/v1/UCI HAR Dataset/' 
train_signals_ucihar, train_labels_ucihar, test_signals_ucihar, test_labels_ucihar = load_ucihar_data(folder_ucihar)


# further preprocessing
scales = range(1, 128)
waveletname = 'morl'
train_size = 5000
test_size= 500

train_data_cwt = np.ndarray(shape=(train_size, 127, 127, 9))
for ii in range(0, train_size):
    if ii % 1000 == 0:
        print(ii)
    for jj in range(0, 9):
        signal = train_signals_ucihar[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:, :127]
        train_data_cwt[ii, :, :, jj] = coeff_

test_data_cwt = np.ndarray(shape=(test_size, 127, 127, 9))
for ii in range(0, test_size):
    if ii % 100 == 0:
        print(ii)
    for jj in range(0, 9):
        signal = test_signals_ucihar[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:, :127]
        test_data_cwt[ii, :, :, jj] = coeff_

uci_har_labels_train = list(map(lambda x: int(x) - 1, train_labels_ucihar))
uci_har_labels_test = list(map(lambda x: int(x) - 1, test_labels_ucihar))

x_train = train_data_cwt
y_train = list(uci_har_labels_train[:train_size])
x_test = test_data_cwt
y_test = list(uci_har_labels_test[:test_size])


# model building
history = History()

img_x = 127
img_y = 127
img_z = 9
input_shape = (img_x, img_y, img_z)

num_classes = 6
batch_size = 16
num_classes = 7
epochs = 10

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


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

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])

train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))

history.history

def plot_hist(history):
    plt.title('Accuracy of training and test sets')
    
    plt.subplot(1,2,1)
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('No. epoch')
    
    plt.subplot(1,2,2)
    plt.xlim(0,9)
    plt.ylim(0,1)
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('No. epoch')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()
    
plot_hist(history)
