# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:09:51 2021

@author: Joshua
"""

import m4_f
import time

# -----------------------------------------------------------------------------
# baseline arima model
model_type = "unscaled"
start_time = time.time()
error_list = m4_f.m4_baseline_arima(model_type)
print("---Baseline took %.3f seconds ---" % (time.time() - start_time))


# -----------------------------------------------------------------------------
# wavelet denoising variations
# model_type = ["svr", "rfr", "dtc","nn","rnn","cnn"]
model_type = ["svr", "rfr", "dtc", "nn", "rnn", "cnn"]
# wavelet = ["noisy", "db4", "sym4", "coif4", "haar"]
wavelet = ["db4", "sym4", "coif4", "haar"]
level = "inf"
threshold = [0.1,0.2,0.3]


start_time = time.time()
for i in model_type:
    for j in wavelet:
        for k in threshold:
            m4_f.m4_denoising_variations(model_type=i, wavelet=j,
                                             threshold=k,
                                             level=level)
end_time = time.time()
print("---models {} with wavelets {} and thresholds {} took {} seconds ---".format(model_type,wavelet,threshold,end_time - start_time))



# -----------------------------------------------------------------------------
# wavelet multiresolution analysis
# model_type = ["svr", "rfr", "dtc","nn","rnn","cnn"]
model_type = ["svr","rfr", "dtc"]
# wavelet = ["db4", "sym4", "coif4", "haar"]
wavelet = ["db4", "sym4", "coif4", "haar"]
level = [1,2]


start_time = time.time()
for i in model_type:
    for j in wavelet:
        for k in level:
            m4_f.m4_multires_variations(model_type=i, wavelet=j,
                                             level=k)
end_time = time.time()
print("---models {} with wavelets {} and levels {} took {} seconds ---".format(model_type,wavelet,level,end_time - start_time))
