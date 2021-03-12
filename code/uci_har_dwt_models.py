# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:11:59 2020

@author: Joshua
"""
import har_v2_f


# ts dwt models   
data_path = "../datasets/uci_har/v2/RawData/"
#discrete_wavelet = ["noisy","median","haar","db4","dmey","sym4","coif4"]
discrete_wavelet = ["median"]
# disc_thresh = [0.2,0.3]
disc_thresh  = 0.2
window_size = 128
#subband_dwt_wavelet = ["none","haar","db4","dmey","sym4","coif4"]
subband_dwt_wavelet = ["none","haar","db4","dmey","sym4","coif4"]
#model_type = ["cnn","rnn","nn", "birnn","lstmcnn"]
model_type = ["cnn","rnn","nn", "birnn","lstmcnn"]

for j in discrete_wavelet:
    for k in subband_dwt_wavelet:
        for i in model_type:
            har_v2_f.pure_ts_variations(acc_data_path = data_path,
                           gyro_data_path = data_path,
                           denoise_thresh = disc_thresh,
                           denoise_dwt_wavelet = j,
                           window_size = window_size,
                           subband_dwt_wavelet = k,
                           model_type = i)
