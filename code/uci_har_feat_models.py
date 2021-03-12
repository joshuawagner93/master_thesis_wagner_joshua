# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 17:50:27 2020

@author: Joshua
"""

import har_v2_f
import time

# ts dwt models   
data_path = "../datasets/uci_har/v2/RawData/"
#discrete_wavelet = ["noisy","median","haar","db4","dmey","sym4","coif4"]
discrete_wavelet = ["median"]
#disc_thresh = [0.2,0.3]
disc_thresh  = 0.2
window_size = 128
#subband_dwt_wavelet = ["none","haar","db4","dmey","sym4","coif4"]
subband_dwt_wavelet = ["none","haar","db4","dmey","sym4","coif4"]
#model_type = ["nn","svc","rfc","dtc","knc","gbc","kerasnn","logreg"]
model_type = ["nn","svc","rfc","dtc","knc","gbc","kerasnn","logreg"]

start_time = time.time()
for j in discrete_wavelet:
    for k in subband_dwt_wavelet:
        for i in model_type:
            har_v2_f.feat_variations(acc_data_path = data_path,
                           gyro_data_path = data_path,
                           denoise_thresh = disc_thresh,
                           denoise_dwt_wavelet = j,
                           window_size = window_size,
                           subband_dwt_wavelet = k,
                           model_type = i)
print("--- %s seconds ---" % (time.time() - start_time))

# 3 wavelets for noise filtering, 6 for subband and 6 sklearn models (108 versions) take ~8.4 hours

# models with the original features
model_type = ["nn","svc","rfc","dtc","knc","gbc","kerasnn","logreg"]
start_time = time.time()
for i in model_type:
    har_v2_f.feature_orginal_data_variations(model_type = i)
print("--- %s seconds ---" % (time.time() - start_time))