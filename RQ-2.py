#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import sys
import time
import os
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as st

directories = os.listdir("Dataset")
algorithm_name = ["LogisticRegression", "TreeRegressor", "Decision_Tree", "Discriminant_Analysis", "SVM"]
dataset_name = ["census", "credit", "bank", "compas"]
sensitive_attr = ["gender", "race", "age"]
searches = ["random", "mutation", "coverage"]

key_num_inp = {}
key_max_AOD = {}
key_max_TPR = {}
search_name = {}
with open("Results/RQ2.csv", 'w') as f:
    f.write("name,num_inputs_random,num_inputs_mutation,num_inputs_coverage,|range_AOD_random|,|range_AOD_mutation|,|range_AOD_coverage|,|range_TPR_random|,|range_TPR_mutation|,|range_TPR_coverage|\n")
    for drs in directories:
        if not drs.startswith("Run"):
            continue
        for filename in os.listdir("Dataset" + "/" + drs):
            if filename.endswith("res.csv"):
                search = ""
                key = ""
                found = False
                for s in searches:
                    if s in filename:
                        search = s
                        for an in algorithm_name:
                            if an in filename:
                                for ds in dataset_name:
                                    if ds in filename:
                                        for sa in sensitive_attr:
                                            if sa in filename:
                                                key = an + "-" + ds + "-" + sa
                                                found = True
                                            if(found):
                                                break
                                    if(found):
                                        break
                            if(found):
                                break
                    if(found):
                        break
                if found:
                    df = pd.read_csv("Dataset" + "/" + drs + "/" + filename)
                    df = df[df["score"] <= 1.0]
                    df = df[df["AOD"] <= 1.0]
                    df = df[df["TPR"] <= 1.0]
                    accuracy = df["score"]
                    AOD = df["AOD"]
                    TPR = df["TPR"]
                    if key not in key_num_inp:
                        key_num_inp[key] = []
                        key_max_AOD[key] = []
                        key_max_TPR[key] = []
                        search_name[key] = []
                    key_num_inp[key].append(df.shape[0])
                    key_max_AOD[key].append(abs(AOD.max() - AOD.min()))
                    key_max_TPR[key].append(abs(TPR.max() - TPR.min()))
                    search_name[key].append(search)

    for key in key_num_inp:
        print(key)
        f.write(key)
        f.write(",")
        num_inputs_data = {}
        key_max_AOD_data = {}
        key_max_TPR_data = {}
        for i in range(len(search_name[key])):
            if search_name[key][i] not in num_inputs_data:
                num_inputs_data[search_name[key][i]] = []
                num_inputs_data[search_name[key][i]].append(key_num_inp[key][i])
                key_max_AOD_data[search_name[key][i]] = []
                key_max_AOD_data[search_name[key][i]].append(key_max_AOD[key][i])
                key_max_TPR_data[search_name[key][i]] = []
                key_max_TPR_data[search_name[key][i]].append(key_max_TPR[key][i])
            else:
                num_inputs_data[search_name[key][i]].append(key_num_inp[key][i])
                key_max_AOD_data[search_name[key][i]].append(key_max_AOD[key][i])
                key_max_TPR_data[search_name[key][i]].append(key_max_TPR[key][i])
        print(num_inputs_data[search])
        for search in searches:
            num_inputs_data_average = np.mean(num_inputs_data[search])
            num_inputs_data_CI = st.t.interval(0.95, len(num_inputs_data[search])-1, loc=num_inputs_data_average, scale=st.sem(num_inputs_data[search]))
            f.write(str(round(num_inputs_data_average)) + " (+/- " + str(round(num_inputs_data_average - num_inputs_data_CI[0])) + ")")
            f.write(",")

        for search in searches:
            key_max_AOD_data_average = np.mean(key_max_AOD_data[search])
            key_max_AOD_data_CI = st.t.interval(0.95, len(key_max_AOD_data[search])-1, loc=key_max_AOD_data_average, scale=st.sem(key_max_AOD_data[search]))
            f.write(str(round(key_max_AOD_data_average * 100,1)) + "\% (+/- " + str(round((key_max_AOD_data_average - key_max_AOD_data_CI[0]) * 100, 1)) + "\%)")
            f.write(",")

        for search in searches:
            key_max_TPR_data_average = np.mean(key_max_TPR_data[search])
            key_max_TPR_data_CI = st.t.interval(0.95, len(key_max_TPR_data[search])-1, loc=key_max_TPR_data_average, scale=st.sem(key_max_TPR_data[search]))
            f.write(str(round(key_max_TPR_data_average * 100,1)) + "\% (+/- " + str(round((key_max_TPR_data_average - key_max_TPR_data_CI[0]) * 100, 1)) + "\%)")
            f.write(",")
        f.write("\n")


X_dict = {}
Y_dict = {}
search_dict = {}

# plots
for drs in directories:
    if not drs.startswith("Run"):
        continue
    for filename in os.listdir("Dataset" + "/" + drs):
        if filename.endswith("res.csv"):
            print(filename)
            search = ""
            key = ""
            found = False
            for s in searches:
                if s in filename:
                    search = s
                    for an in algorithm_name:
                        if an in filename:
                            for ds in dataset_name:
                                if ds in filename:
                                    for sa in sensitive_attr:
                                        if sa in filename:
                                            key = an + "-" + ds + "-" + sa
                                            found = True
                                        if(found):
                                            break
                                if(found):
                                    break
                        if(found):
                            break
                if(found):
                    break
            if found:
                df = pd.read_csv("Dataset" + "/" + drs + "/" + filename)
                AOD = np.array(df["AOD"])
                x = []
                y = []
                max_AOD_over_time = df["AOD"][0]
                for a in range(0,14500,1):
                    try:
                        df1 = df[df["timer"] >= a]
                        df1 = df1[df1["timer"] < a + 1]
                    except TypeError as TE:
                        pass
                    if not df1.empty:
                        max_AOD_a = df1["AOD"].max()
                        if(max_AOD_over_time < max_AOD_a):
                            max_AOD_over_time = max_AOD_a
                    x.append(a+1)
                    y.append(max_AOD_over_time)
                if key not in X_dict:
                    X_dict[key] = []
                    Y_dict[key] = []
                    search_dict[key] = []
                    X_dict[key].append(x)
                    Y_dict[key].append(y)
                    search_dict[key].append(search)
                else:
                    X_dict[key].append(x)
                    Y_dict[key].append(y)
                    search_dict[key].append(search)

for key in X_dict:
    Y_searach_AOD_data = {}
    Y_search_AOD_average = {}
    Y_search_AOD_CI = {}
#    print(search_dict[key])
    for i in range(len(search_dict[key])):
        if search_dict[key][i] not in Y_searach_AOD_data:
            Y_searach_AOD_data[search_dict[key][i]] = []
            Y_searach_AOD_data[search_dict[key][i]].append(Y_dict[key][i])
        else:
            Y_searach_AOD_data[search_dict[key][i]].append(Y_dict[key][i])
    for search in Y_searach_AOD_data:
        data = np.array(Y_searach_AOD_data[search])
        data_mean = np.mean(data, axis=0)
        data_CI = np.array([st.t.interval(0.95, len(data[:,i])-1, loc=np.mean(data[:,i]), scale=st.sem(data[:,i])) for i in range(data.shape[1])])
        Y_search_AOD_average[search] = data_mean
        Y_search_AOD_CI[search] = np.nan_to_num(data_CI)
    plt.figure(dpi=150)
    X = X_dict[key]
    name = search_dict[key]
    for search in searches:
        plt.plot(X[0], Y_search_AOD_average[search], label=search)
        plt.fill_between(X[0],[y[0] for y in Y_search_AOD_CI[search]],[y[1] for y in Y_search_AOD_CI[search]], alpha=0.1)
    plt.xlabel('Time (s)')
    plt.ylabel('Average Odds Difference (AOD)')
    key = key.replace("_","")
    plt.title(key)
    plt.legend(loc = "lower right")
    print(key)
    plt.savefig("Results/" + key + "_AOD_vs_time", dpi=150)
    plt.close()
