# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 09:22:42 2021

@author: Joshua
"""

import os
import pandas as pd
import pywt
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np
os.getcwd()
sns.set_theme(style="whitegrid")

# multiresolution analysis
multires_path = "./m4_multires_models/results/"

_, _, multires_filenames = next(os.walk(multires_path))

test_results = pd.read_csv(multires_path + multires_filenames[0],index_col=0)
test_results.loc["test","RMSE"]

multires_filenames[0].split("_")

multires_list = []
for i in range(len(multires_filenames)):
    temp_results = pd.read_csv(multires_path + multires_filenames[i],index_col=0)
    temp_split = multires_filenames[i].split("_")
    temp_list = [temp_split[1].strip("[").strip("]"), temp_split[2], temp_split[3].split(".csv")[0], temp_results.loc["test","RMSE"],temp_results.loc["test","SMAPE"]]
    multires_list.append(temp_list)
    

multires_df = pd.DataFrame(multires_list,columns=["dwt_multi_wavelet", "dwt_level", "model", "RMSE","SMAPE"])



# uninvertible denoising
denoise_path = "./m4_ts_models/results/"

_, _, denoise_filenames = next(os.walk(denoise_path))


denoise_filenames[0].split("_")

denoise_list = []
for i in range(len(denoise_filenames)):
    temp_results = pd.read_csv(denoise_path + denoise_filenames[i],index_col=0)
    temp_split = denoise_filenames[i].split("_")
    temp_list = [temp_split[1].strip("[").strip("]"), temp_split[2], temp_split[4].split(".csv")[0], temp_results.loc["test","RMSE"],temp_results.loc["test","SMAPE"]]
    denoise_list.append(temp_list)
    

denoise_df = pd.DataFrame(denoise_list,columns=["dwt_multi_wavelet", "dwt_treshold", "model", "RMSE","SMAPE"])


# auto-arima rmse and smape per test series
arima_results = pd.read_csv("./m4_baseline/results/unscaled.csv")
arima_rmse = arima_results.RMSE.mean()
arima_smape = arima_results.sMAPE.mean()


# full forecasting results
multires_df["maj_prep"] = "multiresolution"
denoise_df["maj_prep"] = "denoising"
arima_df = pd.DataFrame([["ARIMA",arima_rmse,arima_smape]],columns=["model", "RMSE", "SMAPE"])
forecasting_full_results = pd.concat([multires_df, denoise_df, arima_df])
forecasting_full_results['model'] = forecasting_full_results['model'].replace(['cnn'],'CNN')
forecasting_full_results['model'] = forecasting_full_results['model'].replace(['nn'],'NN')
forecasting_full_results['model'] = forecasting_full_results['model'].replace(['rnn'],'RNN')
forecasting_full_results['model'] = forecasting_full_results['model'].replace(['dtc'],'DTR')
forecasting_full_results['model'] = forecasting_full_results['model'].replace(['rfr'],'RFR')
forecasting_full_results['model'] = forecasting_full_results['model'].replace(['svr'],'SVR')
forecasting_full_results['dwt_multi_wavelet'] = forecasting_full_results['dwt_multi_wavelet'].replace(['coif4'],'Coiflets-4')
forecasting_full_results['dwt_multi_wavelet'] = forecasting_full_results['dwt_multi_wavelet'].replace(['haar'],'Haar')
forecasting_full_results['dwt_multi_wavelet'] = forecasting_full_results['dwt_multi_wavelet'].replace(['db4'],'Daubechies-4')
forecasting_full_results['dwt_multi_wavelet'] = forecasting_full_results['dwt_multi_wavelet'].replace(['sym4'],'Symlets-4')
forecasting_full_results['dwt_multi_wavelet'] = forecasting_full_results['dwt_multi_wavelet'].replace([np.nan],'noisy')
forecasting_full_results['maj_prep'] = forecasting_full_results['maj_prep'].replace([np.nan],'baseline')
forecasting_full_results.reset_index(inplace=True, drop=True)
#forecasting_full_results.to_csv("./aggregated_results/m4/full_results.csv", index=False)


# results split by maj. prep. and aggregated results
ax = sns.boxplot(y=forecasting_full_results["RMSE"], x = forecasting_full_results["model"])
ax = sns.boxplot(y=forecasting_full_results["SMAPE"], x = forecasting_full_results["model"])

# plots are too small to really let show much
ax = sns.boxplot(y=forecasting_full_results["SMAPE"], x = forecasting_full_results["model"], hue = forecasting_full_results["maj_prep"])
ax.set(xlabel='Model', ylabel='SMAPE')
ax.legend(title="Preprocessing")
plt.tight_layout()

ax = sns.boxplot(y=forecasting_full_results["RMSE"], x = forecasting_full_results["model"], hue = forecasting_full_results["maj_prep"])
ax.set(xlabel='Model', ylabel='RMSE')
ax.legend(title="Preprocessing")
plt.tight_layout()

# grouped results by model
results_without_baseline = forecasting_full_results[forecasting_full_results["model"] != "ARIMA"]
grouped_full_results_models = results_without_baseline.groupby(by=["model"]).median()
grouped_full_results_models["RMSE_q25"] = results_without_baseline.groupby(by=["model"])["RMSE"].quantile(q=0.25)
grouped_full_results_models["SMAPE_q25"] = results_without_baseline.groupby(by=["model"])["SMAPE"].quantile(q=0.25)
grouped_full_results_models["RMSE_q75"] = results_without_baseline.groupby(by=["model"])["RMSE"].quantile(q=0.75)
grouped_full_results_models["SMAPE_q75"] = results_without_baseline.groupby(by=["model"])["SMAPE"].quantile(q=0.75)
grouped_full_results_models["RMSE_iqr"] = grouped_full_results_models["RMSE_q75"] - grouped_full_results_models["RMSE_q25"]
grouped_full_results_models["SMAPE_iqr"] = grouped_full_results_models["SMAPE_q75"] - grouped_full_results_models["SMAPE_q25"]
grouped_full_results_models.rename({"RMSE" : "median RMSE"}, inplace=True, axis=1)
grouped_full_results_models.rename({"SMAPE" : "median SMAPE"}, inplace=True, axis=1)

grouped_full_results_models.sort_values(by=["median RMSE"], inplace=True, ascending=True)
#grouped_full_results_models.to_latex("./aggregated_results/m4/results_grouped_models.tex",index = True, index_names=True, columns=["median RMSE","RMSE_iqr","median SMAPE" ,"SMAPE_iqr"])


noisy_results = forecasting_full_results[forecasting_full_results["dwt_multi_wavelet"] == "noisy"]
noisy_results.sort_values(by=["RMSE"], inplace=True, ascending=True)
#noisy_results.to_latex("./aggregated_results/m4/results_noisy_models.tex", columns=["model", "RMSE", "SMAPE"], index=False)


grouped_results_model_prep = forecasting_full_results.groupby(by=["model","maj_prep"]).median()
grouped_results_model_prep["RMSE_q25"] = forecasting_full_results.groupby(by=["model","maj_prep"])["RMSE"].quantile(q=0.25)
grouped_results_model_prep["SMAPE_q25"] = forecasting_full_results.groupby(by=["model","maj_prep"])["SMAPE"].quantile(q=0.25)
grouped_results_model_prep["RMSE_q75"] = forecasting_full_results.groupby(by=["model","maj_prep"])["RMSE"].quantile(q=0.75)
grouped_results_model_prep["SMAPE_q75"] = forecasting_full_results.groupby(by=["model","maj_prep"])["SMAPE"].quantile(q=0.75)
grouped_results_model_prep["RMSE_iqr"] = grouped_results_model_prep["RMSE_q75"] - grouped_results_model_prep["RMSE_q25"]
grouped_results_model_prep["SMAPE_iqr"] = grouped_results_model_prep["SMAPE_q75"] - grouped_results_model_prep["SMAPE_q25"]
grouped_results_model_prep.rename({"RMSE" : "median RMSE"}, inplace=True, axis=1)
grouped_results_model_prep.rename({"SMAPE" : "median SMAPE"}, inplace=True, axis=1)
grouped_results_model_prep = grouped_results_model_prep[["median RMSE", "median SMAPE", "RMSE_iqr", "SMAPE_iqr"]]
grouped_results_model_prep.sort_values(by=["median RMSE"], inplace=True, ascending=True)
#grouped_results_model_prep.to_latex("./aggregated_results/m4/results_grouped_models_prep.tex",index = True, index_names=True, columns=["median RMSE","RMSE_iqr","median SMAPE" ,"SMAPE_iqr"])
