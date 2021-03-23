# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:42:25 2021

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


# HAR CWT CNN results
cwt_path = "./uci_har_cwt_cnn_models/results/"

_, _, cwt_filenames = next(os.walk(cwt_path))

test_results = pd.read_csv(cwt_path + cwt_filenames[0],index_col=0)
test_results.loc["test","acc"]

cwt_filenames[0].split("_")

out_list = []
for i in range(len(cwt_filenames)):
    temp_results = pd.read_csv(cwt_path + cwt_filenames[i],index_col=0).loc["test","acc"]
    temp_split = cwt_filenames[i].split("_")
    temp_list = [temp_split[0], temp_split[1].strip("[").strip("]"), temp_split[2].rpartition("128")[2], temp_split[3].split(".csv")[0], temp_results]
    out_list.append(temp_list)
    

out_df = pd.DataFrame(out_list,columns=["dwt_denoise_wavelet", "dwt_threshold", "cwt_wavelet", "cwt_scale","test_acc"])
#out_df.to_csv("./aggregated_results/har/cwt_results.csv", index=False)

sns.set_theme(style="whitegrid")
ax = sns.boxplot(y=out_df["test_acc"])
ax = sns.boxplot(x="cwt_wavelet", y="test_acc", hue="cwt_scale",
                 data=out_df, palette="Set3")
ax = sns.boxplot(x="dwt_wavelet", y="test_acc", hue="dwt_threshold",
                 data=out_df, palette="Set3")


# mean and var, not robust to outliers
out_df.test_acc.mean()
out_df.test_acc.var()

# quartiles, as outlier robust measure
out_df.test_acc.quantile([0.25, 0.5, 0.75])
# interquartile distance
out_df.test_acc.quantile(0.75) - out_df.test_acc.quantile(0.25)

# modifications for latex export
#out_df_latex = out_df.set_index(["dwt_denoise_wavelet", "dwt_threshold", "cwt_wavelet", "cwt_scale"])
#out_df_latex.to_latex("./aggregated_results/har/cwt_results.tex",index = True, index_names=True)


out_df.test_acc.max()



# HAR TS results
ts_path = "./uci_har_ts_models/results/"

_, _, ts_filenames = next(os.walk(ts_path))

test_results = pd.read_csv(ts_path + ts_filenames[0],index_col=0)
test_results.loc["test","acc"]

ts_filenames[0].split("_")

ts_out_list = []
for i in range(len(ts_filenames)):
    temp_results = pd.read_csv(ts_path + ts_filenames[i],index_col=0).loc["test","acc"]
    temp_split = ts_filenames[i].split("_")
    temp_list = [temp_split[0], temp_split[1].strip("[").strip("]"), temp_split[2].rpartition("128")[2], temp_split[3].split(".csv")[0], temp_results]
    ts_out_list.append(temp_list)
    

ts_out_df = pd.DataFrame(ts_out_list,columns=["dwt_denoise_wavelet", "dwt_threshold", "dwt_multi_wavelet", "model", "test_acc"])
#ts_out_df.to_csv("./aggregated_results/har/ts_results.csv", index=False)

sns.set_theme(style="whitegrid")
ax = sns.boxplot(y=ts_out_df["test_acc"], x = ts_out_df["model"])
ax = sns.boxplot(x="dwt_denoise_wavelet", y="test_acc", hue="model",
                 data=ts_out_df, palette="Set3")
ax = sns.boxplot(x="dwt_multi_wavelet", y="test_acc", hue="model",
                 data=ts_out_df, palette="Set3")

# mean and var, not robust to outliers
ts_out_df.test_acc.mean()
ts_out_df.test_acc.var()

# quartiles, as outlier robust measure
ts_out_df.test_acc.quantile([0.25, 0.5, 0.75])
# interquartile distance
ts_out_df.test_acc.quantile(0.75) - ts_out_df.test_acc.quantile(0.25)


ts_out_df.test_acc.max()



# HAR feat results
feat_path = "./uci_har_feat_models/results/"

_, _, feat_filenames = next(os.walk(feat_path))

test_results = pd.read_csv(feat_path + feat_filenames[0],index_col=0)
test_results.loc["test","acc"]

feat_filenames[0].split("_")

feat_out_list = []
for i in range(len(feat_filenames)):
    temp_results = pd.read_csv(feat_path + feat_filenames[i],index_col=0).loc["test","acc"]
    temp_split = feat_filenames[i].split("_")
    temp_list = [temp_split[0], temp_split[1].strip("[").strip("]"), temp_split[2].rpartition("128")[2], temp_split[3].split(".csv")[0], temp_results]
    feat_out_list.append(temp_list)
    

feat_out_df = pd.DataFrame(feat_out_list,columns=["dwt_denoise_wavelet", "dwt_threshold", "dwt_multi_wavelet", "model", "test_acc"])
#feat_out_df.to_csv("./aggregated_results/har/feat_results.csv", index=False)

# 
sns.set_theme(style="whitegrid")
ax = sns.boxplot(y=feat_out_df["test_acc"], x = feat_out_df["model"])
ax = sns.boxplot(x="dwt_denoise_wavelet", y="test_acc", hue="model",
                 data=feat_out_df, palette="Set3")
ax = sns.boxplot(x="dwt_denoise_wavelet", y="test_acc", hue="dwt_threshold",
                 data=feat_out_df, palette="Set3")
ax = sns.boxplot(x="dwt_multi_wavelet", y="test_acc", hue="model",
                 data=feat_out_df, palette="Set3")

ax = sns.boxplot(x="dwt_multi_wavelet", y="test_acc", hue="dwt_denoise_wavelet",
                 data=feat_out_df, palette="Set3")


ax = sns.boxplot(y=feat_out_df[feat_out_df["model"] != "sklearn_nn"]["test_acc"], x = feat_out_df[feat_out_df["model"] != "sklearn_nn"]["model"])
ax = sns.boxplot(y=feat_out_df[feat_out_df["model"] != "sklearn_nn"]["test_acc"])

# feature models median and quantiles
feat_out_df.groupby(by=["model"]).median()

# mean and var, not robust to outliers
feat_out_df.test_acc.mean()
feat_out_df.test_acc.var()

# quartiles, as outlier robust measure
feat_out_df.test_acc.quantile([0.25, 0.5, 0.75])
# interquartile distance
feat_out_df.test_acc.quantile(0.75) - feat_out_df.test_acc.quantile(0.25)


feat_out_df.test_acc.max()

feat_out_df["test_acc"].median()
feat_out_df[feat_out_df["model"] != "sklearn_nn"]["test_acc"].median()

feat_out_df.groupby(by=["dwt_multi_wavelet"]).median()
feat_out_df.groupby(by=["model"]).median()


# HAR org. feat results
org_feat_path = "./uci_har_original_feat_models/results/"

_, _, org_feat_filenames = next(os.walk(org_feat_path))

test_results = pd.read_csv(org_feat_path + org_feat_filenames[0],index_col=0)
test_results.loc["test","acc"]

org_feat_filenames[0].split("_")[3]

org_feat_out_list = []
for i in range(len(org_feat_filenames)):
    temp_results = pd.read_csv(org_feat_path + org_feat_filenames[i],index_col=0).loc["test","acc"]
    temp_split = org_feat_filenames[i].split("_")
    temp_list = [temp_split[3].split(".csv")[0], temp_results]
    org_feat_out_list.append(temp_list)
    

org_feat_out_df = pd.DataFrame(org_feat_out_list,columns=["model", "test_acc"])
#org_feat_out_df.to_csv("./aggregated_results/har/org_feat_results.csv", index=False)

# remove the sklearn nn form the results as the outliers are rather heavy
sns.set_theme(style="whitegrid")
ax = sns.boxplot(y=org_feat_out_df["test_acc"], x = org_feat_out_df["model"])
ax = sns.scatterplot(y=org_feat_out_df["test_acc"], x = org_feat_out_df["model"])

# mean and var, not robust to outliers
org_feat_out_df.test_acc.mean()
org_feat_out_df.test_acc.var()

# quartiles, as outlier robust measure
org_feat_out_df.test_acc.quantile([0.25, 0.5, 0.75])
# interquartile distance
org_feat_out_df.test_acc.quantile(0.75) - org_feat_out_df.test_acc.quantile(0.25)


# combination of the different major preprocessing steps
out_df["model"] = "CNN"
out_df["maj_prep"] = "CWT"
ts_out_df["maj_prep"] = "ts_DWT"

feat_out_df["maj_prep"] = "feat_ext"
feat_out_df["model"] = feat_out_df["model"].replace("nn","sklearn_nn")
feat_out_df["model"] = feat_out_df["model"].replace("kerasnn","NN")
org_feat_out_df["maj_prep"] = "org. Feat. Ext."
org_feat_out_df["model"] = org_feat_out_df["model"].replace("nn","sklearn_nn")
org_feat_out_df["model"] = org_feat_out_df["model"].replace("kerasnn","NN")
test_full_results = pd.concat([out_df, ts_out_df, feat_out_df, org_feat_out_df])
test_full_results['model'] = test_full_results['model'].replace(['cnn'],'CNN')
test_full_results['model'] = test_full_results['model'].replace(['nn'],'NN')
test_full_results['model'] = test_full_results['model'].replace(['rnn'],'RNN')
test_full_results['model'] = test_full_results['model'].replace(['birnn'],'Bi-RNN')
test_full_results['model'] = test_full_results['model'].replace(['lstmcnn'],'LSTM-CNN')
test_full_results['model'] = test_full_results['model'].replace(['dtc'],'DTC')
test_full_results['model'] = test_full_results['model'].replace(['knc'],'KNC')
test_full_results['model'] = test_full_results['model'].replace(['logreg'],'multinom. Reg.')
test_full_results['model'] = test_full_results['model'].replace(['rfc'],'RFC')
test_full_results['model'] = test_full_results['model'].replace(['svc'],'SVC')
test_full_results['model'] = test_full_results['model'].replace(['gbc'],'GBC')
test_full_results['model'] = test_full_results['model'].replace(['sklearn_nn'],'Sklearn NN')
test_full_results['maj_prep'] = test_full_results['maj_prep'].replace(['ts_DWT'],'Multiresolution')
test_full_results['maj_prep'] = test_full_results['maj_prep'].replace(['feat_ext'],'Feature Extraction')
test_full_results.reset_index(inplace=True,drop=True)
# replace shortcuts with real wavelet names
test_full_results['dwt_denoise_wavelet'] = test_full_results['dwt_denoise_wavelet'].replace(['haar'],'Haar')
test_full_results['dwt_denoise_wavelet'] = test_full_results['dwt_denoise_wavelet'].replace(['db4'],'Daubechies-4')
test_full_results['dwt_denoise_wavelet'] = test_full_results['dwt_denoise_wavelet'].replace(['dmey'],'D.-Mey.')
test_full_results['dwt_denoise_wavelet'] = test_full_results['dwt_denoise_wavelet'].replace(['sym4'],'Symlets-4')
test_full_results['dwt_denoise_wavelet'] = test_full_results['dwt_denoise_wavelet'].replace(['median'],'Median')
test_full_results['dwt_denoise_wavelet'] = test_full_results['dwt_denoise_wavelet'].replace(['noisy'],'none')
test_full_results['dwt_denoise_wavelet'] = test_full_results['dwt_denoise_wavelet'].replace(['coif4'],'Coiflets-4')
test_full_results['cwt_wavelet'] = test_full_results['cwt_wavelet'].replace(['gaus2'],'Gaussian-2')
test_full_results['cwt_wavelet'] = test_full_results['cwt_wavelet'].replace(['mexh'],'Mexican Hat')
test_full_results['cwt_wavelet'] = test_full_results['cwt_wavelet'].replace(['morl'],'Morlet')
test_full_results['dwt_multi_wavelet'] = test_full_results['dwt_multi_wavelet'].replace(['coif4'],'Coiflets-4')
test_full_results['dwt_multi_wavelet'] = test_full_results['dwt_multi_wavelet'].replace(['haar'],'Haar')
test_full_results['dwt_multi_wavelet'] = test_full_results['dwt_multi_wavelet'].replace(['db4'],'Daubechies-4')
test_full_results['dwt_multi_wavelet'] = test_full_results['dwt_multi_wavelet'].replace(['dmey'],'D.-Mey.')
test_full_results['dwt_multi_wavelet'] = test_full_results['dwt_multi_wavelet'].replace(['sym4'],'Symlets-4')
#test_full_results.to_csv("./aggregated_results/har/full_results.csv", index=False)



# grouped results by model
grouped_full_results_models = test_full_results.groupby(by=["model"]).median()
grouped_full_results_models["q25"] = test_full_results.groupby(by=["model"]).quantile(q=0.25)
grouped_full_results_models["q75"] = test_full_results.groupby(by=["model"]).quantile(q=0.75)
grouped_full_results_models["iqr"] = grouped_full_results_models["q75"] - grouped_full_results_models["q25"]
grouped_full_results_models.rename({"test_acc" : "median"}, inplace=True, axis=1)
grouped_full_results_models.sort_values(by=["median"], inplace=True, ascending=False)
#grouped_full_results_models.to_latex("./aggregated_results/har/results_grouped_models.tex",index = True, index_names=True, columns=["median","iqr"])


# grouped results by major preprocessing step
grouped_full_results_prep = test_full_results.groupby(by=["maj_prep"]).median()
grouped_full_results_prep["q25"] = test_full_results.groupby(by=["maj_prep"]).quantile(q=0.25)
grouped_full_results_prep["q75"] = test_full_results.groupby(by=["maj_prep"]).quantile(q=0.75)
grouped_full_results_prep["iqr"] = grouped_full_results_prep["q75"] - grouped_full_results_prep["q25"]
grouped_full_results_prep.rename({"test_acc" : "median"}, inplace=True, axis=1)
grouped_full_results_prep.sort_values(by=["median"], inplace=True, ascending=False)
#grouped_full_results_prep.to_latex("./aggregated_results/har/results_grouped_prep.tex",index = True, index_names=True, columns=["median","iqr"])


# grouped results by denoising wavlet
grouped_denoise = test_full_results.groupby(by=["dwt_denoise_wavelet"]).median()
grouped_denoise["q25"] = test_full_results.groupby(by=["dwt_denoise_wavelet"]).quantile(q=0.25)
grouped_denoise["q75"] = test_full_results.groupby(by=["dwt_denoise_wavelet"]).quantile(q=0.75)
grouped_denoise["iqr"] = grouped_denoise["q75"] - grouped_denoise["q25"]
grouped_denoise.rename({"test_acc" : "median"}, inplace=True, axis=1)
grouped_denoise.sort_values(by=["median"], inplace=True, ascending=False)
#grouped_denoise.to_latex("./aggregated_results/har/results_grouped_denoise.tex",index = True, index_names=True, columns=["median","iqr"])


sns.set_theme(style="whitegrid")
ax = sns.boxplot(y=test_full_results["test_acc"], x = test_full_results["model"])
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

# test acc vs model, hue as maj_prep would be too clumpy
ax = sns.boxplot(x=test_full_results["test_acc"], y = test_full_results["model"])
ax.set(xlabel='Test Accuracy', ylabel='Model',xlim=(0,1))
plt.tight_layout()
#plt.savefig("../masters_thesis/images/boxplot_per_model.pdf")

ax = sns.boxplot(x=test_full_results["test_acc"], y = test_full_results["maj_prep"])
ax.set(xlabel='Test Accuracy', ylabel='Preprocessing',xlim=(0,1))
#ax.yaxis.set_label_position("right")
#ax.yaxis.tick_right()
plt.tight_layout()
#plt.savefig("../masters_thesis/images/boxplot_per_prep.pdf")


ax = sns.boxplot(x=test_full_results["test_acc"], y = test_full_results["dwt_denoise_wavelet"])
ax.set(xlabel='Test Accuracy', ylabel='Denoising wavelet',xlim=(0,1))
plt.tight_layout()
#plt.savefig("../masters_thesis/images/boxplot_denoising_wavelets.pdf")



# cwt based model
cwt_based_model = test_full_results[test_full_results["maj_prep"] == "CWT"]

ax = sns.boxplot(y=cwt_based_model["test_acc"], x = cwt_based_model["cwt_wavelet"], hue = cwt_based_model["cwt_scale"])
ax.set(xlabel='CWT Wavelet', ylabel='Test Accuracy',ylim=(None,1))
ax.legend(title="Max. CWT Scale")
plt.tight_layout()
#plt.savefig("../masters_thesis/images/cwt_boxplot_per_wavelet.pdf")

ax = sns.boxplot(y=cwt_based_model["test_acc"], x = cwt_based_model["dwt_denoise_wavelet"], hue = cwt_based_model["cwt_wavelet"])
ax.set(xlabel='Denoising Wavelet', ylabel='Test Accuracy',ylim=(None,1))
ax.legend(title="CWT Wavelet")
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.tight_layout()
#plt.savefig("../masters_thesis/images/cwt_boxplot_and_denoise.pdf")





# latex table aggregated by cwt wavelet and scale
cwt_based_model_latex = cwt_based_model.groupby(["cwt_wavelet", "cwt_scale"]).median()
cwt_based_model_latex["q25"] = cwt_based_model.groupby(by=["cwt_wavelet", "cwt_scale"]).quantile(q=0.25)
cwt_based_model_latex["q75"] = cwt_based_model.groupby(by=["cwt_wavelet", "cwt_scale"]).quantile(q=0.75)
cwt_based_model_latex["iqr"] = cwt_based_model_latex["q75"] - cwt_based_model_latex["q25"]
cwt_based_model_latex.rename({"test_acc" : "median"}, inplace=True, axis=1)
cwt_based_model_latex.sort_values(by=["median"], inplace=True, ascending=False)
#cwt_based_model_latex.to_latex("./aggregated_results/har/cwt_cwt_scale_group.tex",index = True, index_names=True, columns=["median","iqr"])

# latex table aggregated by cwt wavelet and denoising wavelet
cwt_based_model_latex = cwt_based_model.groupby(["cwt_wavelet", "dwt_denoise_wavelet"]).median()
cwt_based_model_latex["q25"] = cwt_based_model.groupby(by=["cwt_wavelet", "dwt_denoise_wavelet"]).quantile(q=0.25)
cwt_based_model_latex["q75"] = cwt_based_model.groupby(by=["cwt_wavelet", "dwt_denoise_wavelet"]).quantile(q=0.75)
cwt_based_model_latex["iqr"] = cwt_based_model_latex["q75"] - cwt_based_model_latex["q25"]
cwt_based_model_latex.rename({"test_acc" : "median"}, inplace=True, axis=1)
cwt_based_model_latex.sort_values(by=["median"], inplace=True, ascending=False)
#cwt_based_model_latex.to_latex("./aggregated_results/har/cwt_denoise_cwt_group.tex",index = True, index_names=True, columns=["median","iqr"])



# dwt based models
dwt_based_model = test_full_results[test_full_results["maj_prep"] == "Multiresolution"]

ax = sns.boxplot(x=dwt_based_model["test_acc"], y = dwt_based_model["dwt_multi_wavelet"])
ax.set(ylabel='DWT Wavelet', xlabel='Test Accuracy')
plt.tight_layout()
#plt.savefig("../masters_thesis/images/dwt_boxplot_per_wavelet.pdf")

ax = sns.boxplot(x=dwt_based_model["test_acc"], y = dwt_based_model["model"])
ax.set(ylabel='Model', xlabel='Test Accuracy')
plt.tight_layout()
#plt.savefig("../masters_thesis/images/dwt_boxplot_per_model.pdf")

ax = sns.boxplot(y=dwt_based_model["test_acc"], x = dwt_based_model["dwt_denoise_wavelet"], hue = dwt_based_model["dwt_multi_wavelet"])
ax.set(xlabel='Model', ylabel='Test Accuracy')
plt.tight_layout()

# rnn does not cope well with wavelet denoising though the bidirectional version does
ax = sns.boxplot(y=dwt_based_model["test_acc"], x= dwt_based_model["dwt_denoise_wavelet"], hue = dwt_based_model["model"])
ax.set(xlabel='Model', ylabel='Test Accuracy')
plt.tight_layout()

ax = sns.boxplot(y=dwt_based_model["test_acc"], x= dwt_based_model["dwt_multi_wavelet"], hue = dwt_based_model["model"])
ax.set(xlabel='Model', ylabel='Test Accuracy')
plt.tight_layout()

# latex tables
dwt_based_model_latex = dwt_based_model.groupby(["model"]).median()
dwt_based_model_latex["q25"] = dwt_based_model.groupby(by=["model"]).quantile(q=0.25)
dwt_based_model_latex["q75"] = dwt_based_model.groupby(by=["model"]).quantile(q=0.75)
dwt_based_model_latex["iqr"] = dwt_based_model_latex["q75"] - dwt_based_model_latex["q25"]
dwt_based_model_latex.rename({"test_acc" : "median"}, inplace=True, axis=1)
dwt_based_model_latex.sort_values(by=["median"], inplace=True, ascending=False)
#dwt_based_model_latex.to_latex("./aggregated_results/har/multires_results_model_group.tex",index = True, index_names=True, columns=["median","iqr"])


dwt_based_model_latex = dwt_based_model.groupby(["model", "dwt_multi_wavelet"]).median()
dwt_based_model_latex["q25"] = dwt_based_model.groupby(by=["model", "dwt_multi_wavelet"]).quantile(q=0.25)
dwt_based_model_latex["q75"] = dwt_based_model.groupby(by=["model", "dwt_multi_wavelet"]).quantile(q=0.75)
dwt_based_model_latex["iqr"] = dwt_based_model_latex["q75"] - dwt_based_model_latex["q25"]
dwt_based_model_latex.rename({"test_acc" : "median"}, inplace=True, axis=1)
dwt_based_model_latex.sort_values(by=["median"], inplace=True, ascending=False)
#dwt_based_model_latex.to_latex("./aggregated_results/har/multires_results_grouped.tex",index = True, index_names=True, columns=["median","iqr"])

dwt_based_model_latex = dwt_based_model.groupby(["model", "dwt_denoise_wavelet"]).median()
dwt_based_model_latex["q25"] = dwt_based_model.groupby(by=["model", "dwt_denoise_wavelet"]).quantile(q=0.25)
dwt_based_model_latex["q75"] = dwt_based_model.groupby(by=["model", "dwt_denoise_wavelet"]).quantile(q=0.75)
dwt_based_model_latex["iqr"] = dwt_based_model_latex["q75"] - dwt_based_model_latex["q25"]
dwt_based_model_latex.rename({"test_acc" : "median"}, inplace=True, axis=1)
dwt_based_model_latex.sort_values(by=["median"], inplace=True, ascending=False)
#dwt_based_model_latex.to_latex("./aggregated_results/har/multires_results_model_denoise_group.tex",index = True, index_names=True, columns=["median","iqr"])





# feature based models
feat_based_model = test_full_results[test_full_results["maj_prep"] == "Feature Extraction"]

ax = sns.boxplot(x=feat_based_model["test_acc"], y = feat_based_model["dwt_multi_wavelet"])
ax.set(xlabel='Test Accuracy', ylabel='Wavelet')
plt.tight_layout()
#plt.savefig("../masters_thesis/images/feat_boxplot_per_wavelet.pdf")

ax = sns.boxplot(x=feat_based_model["test_acc"], y = feat_based_model["model"])
ax.set(xlabel='Test Accuracy', ylabel='Model',xlim=(0,1))
plt.tight_layout()
#plt.savefig("../masters_thesis/images/feat_boxplot_per_model.pdf")

# suggests that denoising messes strongly with the non-wavelet features
ax = sns.boxplot(x=feat_based_model["test_acc"], y = feat_based_model["model"], hue = feat_based_model["dwt_multi_wavelet"])
ax.set(xlabel='Model', ylabel='Test Accuracy')
plt.tight_layout()

# fix plot and include
ax = sns.boxplot(x=feat_based_model["test_acc"], y = feat_based_model["dwt_multi_wavelet"], hue = feat_based_model["dwt_denoise_wavelet"])
ax.set(xlabel='Model', ylabel='Test Accuracy')
plt.tight_layout()

feat_based_model_latex = feat_based_model.groupby(["model", "dwt_multi_wavelet"]).median()
feat_based_model_latex["q25"] = feat_based_model.groupby(by=["model", "dwt_multi_wavelet"]).quantile(q=0.25)
feat_based_model_latex["q75"] = feat_based_model.groupby(by=["model", "dwt_multi_wavelet"]).quantile(q=0.75)
feat_based_model_latex["iqr"] = feat_based_model_latex["q75"] - feat_based_model_latex["q25"]
feat_based_model_latex.rename({"test_acc" : "median"}, inplace=True, axis=1)
feat_based_model_latex.sort_values(by=["median"], inplace=True, ascending=False)
#feat_based_model_latex.to_latex("./aggregated_results/har/feat_results_model_multi_group.tex",index = True, index_names=True, columns=["median","iqr"])


feat_based_model_latex = feat_based_model.groupby(["model"]).median()
feat_based_model_latex["q25"] = feat_based_model.groupby(by=["model"]).quantile(q=0.25)
feat_based_model_latex["q75"] = feat_based_model.groupby(by=["model"]).quantile(q=0.75)
feat_based_model_latex["iqr"] = feat_based_model_latex["q75"] - feat_based_model_latex["q25"]
feat_based_model_latex.rename({"test_acc" : "median"}, inplace=True, axis=1)
feat_based_model_latex.sort_values(by=["median"], inplace=True, ascending=False)
#feat_based_model_latex.to_latex("./aggregated_results/har/feat_results_model_group.tex",index = True, index_names=True, columns=["median","iqr"])


feat_based_model_no_sk = feat_based_model[feat_based_model["model"] != "Sklearn NN"]
ax = sns.boxplot(x=feat_based_model_no_sk["test_acc"], y = feat_based_model_no_sk["model"], hue = feat_based_model_no_sk["dwt_multi_wavelet"])
ax.set(xlabel='Model', ylabel='Test Accuracy')
plt.tight_layout()

ax = sns.boxplot(x=feat_based_model_no_sk["test_acc"], y = feat_based_model_no_sk["dwt_multi_wavelet"], hue = feat_based_model_no_sk["dwt_denoise_wavelet"])
ax.set(xlabel='Model', ylabel='Test Accuracy')
plt.tight_layout()

ax = sns.boxplot(x=feat_based_model_no_sk["test_acc"], y = feat_based_model_no_sk["model"], hue = feat_based_model_no_sk["dwt_denoise_wavelet"])
ax.set(xlabel='Model', ylabel='Test Accuracy')
plt.tight_layout()

# comparison with org. features
org_feat_models = test_full_results[test_full_results["maj_prep"] == "org. Feat. Ext."].reset_index(drop=True)
org_feat_models.sort_values(by=["test_acc"], inplace=True, ascending=False)
org_feat_models.to_latex("./aggregated_results/har/org_features_acc.tex", columns=["model","test_acc"])
