# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:08:30 2021

@author: Joshua
"""
import numpy as np
import os

# before: 365GB
# after: 327GB

har_cwt_path = "./uci_har_cwt_cnn_models/processed_data/"
_, _, har_cwt_filenames = next(os.walk(har_cwt_path))
for i in har_cwt_filenames:
    cwt_data_path = har_cwt_path + i
    print(cwt_data_path)
    saved_data_file = np.load(cwt_data_path)
    x_train = saved_data_file["x_train"]
    y_train = saved_data_file["y_train"]
    x_val = saved_data_file["x_val"]
    y_val = saved_data_file["y_val"]
    x_test = saved_data_file["x_test"]
    y_test = saved_data_file["y_test"]
    np.savez_compressed(cwt_data_path, x_train=x_train, y_train=y_train,
                        x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)
# before compression: 284GB
# after compression: 265GB


har_dwt_path = "./uci_har_ts_models/processed_data/"
_, _, har_dwt_filenames = next(os.walk(har_dwt_path))
for i in har_dwt_filenames:
    dwt_data_path = har_dwt_path + i
    print(dwt_data_path)
    saved_data_file = np.load(dwt_data_path)
    x_train = saved_data_file["x_train"]
    y_train = saved_data_file["y_train"]
    x_val = saved_data_file["x_val"]
    y_val = saved_data_file["y_val"]
    x_test = saved_data_file["x_test"]
    y_test = saved_data_file["y_test"]
    np.savez_compressed(dwt_data_path, x_train=x_train, y_train=y_train,
                        x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)


har_feat_path = "./uci_har_feat_models/processed_data/"
_, _, har_feat_filenames = next(os.walk(har_feat_path))
for i in har_feat_filenames:
    feat_data_path = har_feat_path + i
    print(feat_data_path)
    saved_data_file = np.load(feat_data_path)
    x_train = saved_data_file["x_train"]
    y_train = saved_data_file["y_train"]
    x_val = saved_data_file["x_val"]
    y_val = saved_data_file["y_val"]
    x_test = saved_data_file["x_test"]
    y_test = saved_data_file["y_test"]
    np.savez_compressed(feat_data_path, x_train=x_train, y_train=y_train,
                        x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)

# before compression: 2.64GB
# after compression: 1.76GB

m4_multires_path = "./m4_multires_models/processed_data/"
_, _, m4_multires_filenames = next(os.walk(m4_multires_path))
for i in m4_multires_filenames:
    m4_multires_data_path = m4_multires_path + i
    print(m4_multires_data_path)
    saved_data_file = np.load(m4_multires_data_path)
    x_train = saved_data_file["x_train"]
    y_train = saved_data_file["y_train"]
    x_val = saved_data_file["x_val"]
    y_val = saved_data_file["y_val"]
    x_test = saved_data_file["x_test"]
    y_test = saved_data_file["y_test"]
    np.savez_compressed(m4_multires_data_path, x_train=x_train, y_train=y_train,
                        x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)

# before compression:
# after compression: 83MB

m4_ts_path = "./m4_ts_models/processed_data/"
_, _, m4_ts_filenames = next(os.walk(m4_ts_path))
for i in m4_ts_filenames:
    m4_ts_data_path = m4_ts_path + i
    print(m4_ts_data_path)
    saved_data_file = np.load(m4_ts_data_path)
    x_train = saved_data_file["x_train"]
    y_train = saved_data_file["y_train"]
    x_val = saved_data_file["x_val"]
    y_val = saved_data_file["y_val"]
    x_test = saved_data_file["x_test"]
    y_test = saved_data_file["y_test"]
    np.savez_compressed(m4_ts_data_path, x_train=x_train, y_train=y_train,
                        x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)

# before compression: 417MB
# after compression: 276MB
