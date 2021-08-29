#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sys
import time
import os
import scipy.stats as st
import math


# In[28]:


directories = os.listdir("Dataset")
algorithm_name = ["LogisticRegression", "Decision_Tree"]
dataset_name = ["census", "credit", "compas", "bank"]
sensitive_attr = ["gender", "race", "age"]
searches = ["mutation","SMBO"]
key_num_inp = {}
key_acc = {}
key_AOD_overall = {}
key_TPR_overall = {}
key_time_overall = {}

with open("Results/RQ4-SMBO.csv", 'w') as f:
    f.write("name,num_inputs,time,accuracy,AOD,EOD\n")
    for drs in directories:
        if not drs.startswith("Mitigations"):
            continue
        print(drs)
        for filename in os.listdir("Dataset" + "/" + drs):
            if filename.endswith("res.csv"):
                print(filename)
                search = ""
                key = ""
                found = False
                mitigation = False
                for s in searches:
                    if s in filename:
                        search = s
                        for an in algorithm_name:
                            if an in filename:
                                for ds in dataset_name:
                                    if ds in filename:
                                        for sa in sensitive_attr:
                                            if sa in filename:
                                                if "SMBO" in filename:
                                                    key = an + "-" + ds + "-" + sa + "-SMBO"
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
                    df_time = df["timer"]
                    lst_acc = []
                    AOD = df["AOD"]
                    TPR = df["EOD"]
                    accuracy = df["score"]
                    lst_AOD_overall = AOD.min()
                    lst_TPR_overall = TPR.min()
                    lst_time_overall = df_time.min()
                    lst_acc_overall = accuracy.max()

                    if key not in key_num_inp:
                        key_num_inp[key] = []
                    key_num_inp[key].append(df.shape[0])

                    if key not in key_acc:
                        key_acc[key] = []
                    key_acc[key].append(lst_acc_overall)

                    if key not in key_AOD_overall:
                        key_AOD_overall[key] = []
                    key_AOD_overall[key].append(lst_AOD_overall)

                    if key not in key_TPR_overall:
                        key_TPR_overall[key] = []
                    key_TPR_overall[key].append(lst_TPR_overall)

                    if key not in key_time_overall:
                        key_time_overall[key] = []
                    key_time_overall[key].append(lst_time_overall)
                    
    for drs in directories:
        if not drs.startswith("Run"):
            continue
        for filename in os.listdir("Dataset" + "/" + drs):
            if filename.endswith("res.csv"):
                print(filename)
                search = ""
                key = ""
                key_SMBO = ""
                found = False
                mitigation = False
                for s in searches:
                    if s in filename:
                        search = s
                        for an in algorithm_name:
                            if an in filename:
                                for ds in dataset_name:
                                    if ds in filename:
                                        for sa in sensitive_attr:
                                            if sa in filename:
                                                key = an + "-" + ds + "-" + sa + "-Parfait"
                                                key_SMBO = an + "-" + ds + "-" + sa + "-SMBO"
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
                    # Restrict timeout to be less than one reported by SMBO.
                    time_SMBO = np.array(key_time_overall[key_SMBO])
                    time_SMBO_max = max(time_SMBO)
                    acc_SMBO = np.array(key_acc[key_SMBO])
                    acc_SMBO_min = min(acc_SMBO)
                    df = df[df["timer"] <= time_SMBO_max]
                    lst_acc = []
                    AOD = df["AOD"]
                    df = df[df["AOD"] >= AOD.min() - 0.005]                     
                    TPR = df["TPR"]
                    accuracy = df["score"]
                    print(AOD.min())
                    print(TPR.min())
                    print(accuracy.max())
                    lst_AOD_overall = AOD.min()
                    lst_TPR_overall = TPR.min()
                    lst_time_overall = time_SMBO_max
                    lst_acc_overall = accuracy.max()

                    if key not in key_num_inp:
                        key_num_inp[key] = []
                    key_num_inp[key].append(df.shape[0])

                    if key not in key_acc:
                        key_acc[key] = []
                    key_acc[key].append(lst_acc_overall)

                    if key not in key_AOD_overall:
                        key_AOD_overall[key] = []
                    key_AOD_overall[key].append(lst_AOD_overall)

                    if key not in key_TPR_overall:
                        key_TPR_overall[key] = []
                    key_TPR_overall[key].append(lst_TPR_overall)

                    if key not in key_time_overall:
                        key_time_overall[key] = []
                    key_time_overall[key].append(lst_time_overall)

                
    for key in key_num_inp:
        f.write(key)
        f.write(",")
        num_inp_lst = np.array(key_num_inp[key])
        num_inp_mean = np.nanmean(num_inp_lst)
        num_inp_confidence = st.t.interval(0.95, len(num_inp_lst)-1, loc=np.mean(num_inp_lst), scale=st.sem(num_inp_lst))
        f.write(str(round(num_inp_mean)) + " (+/- " + str(round(num_inp_mean-num_inp_confidence[0])) + ")")
        f.write(",")
        time_overall_lst = np.array(key_time_overall[key])
        time_overall_min = time_overall_lst
        time_overall_min_mean = max(time_overall_min)
        time_overall_min_confidence = st.t.interval(0.95, len(time_overall_min)-1, loc=max(time_overall_min), scale=st.sem(time_overall_min))
        f.write(str(round(time_overall_min_mean, 3)) + "(s)")
        f.write(",")
        acc_overall_lst = np.array(key_acc[key])
        acc_overall_min = acc_overall_lst
        acc_overall_min_mean = np.nanmean(acc_overall_min)
        acc_overall_min_confidence = st.t.interval(0.95, len(acc_overall_min)-1, loc=np.mean(acc_overall_min), scale=st.sem(acc_overall_min))
        f.write(str(round(acc_overall_min_mean * 100, 1)) + "\% (+/- " + str(round((acc_overall_min_mean-acc_overall_min_confidence[0]) * 100, 1)) + "\%)")
        f.write(",")
        AOD_overall_lst = np.array(key_AOD_overall[key])
        AOD_overall_min = AOD_overall_lst
        AOD_overall_min_mean = np.nanmean(AOD_overall_min)
        AOD_overall_min_confidence = st.t.interval(0.95, len(AOD_overall_min)-1, loc=np.mean(AOD_overall_min), scale=st.sem(AOD_overall_min))
        f.write(str(round(AOD_overall_min_mean * 100, 1)) + "\% (+/- " + str(round((AOD_overall_min_mean-AOD_overall_min_confidence[0]) * 100, 1)) + "\%)")
        f.write(",")        
        TPR_overall_lst = np.array(key_TPR_overall[key])
        TPR_overall_min = TPR_overall_lst
        TPR_overall_min_mean = np.nanmean(TPR_overall_min)
        TPR_overall_min_confidence = st.t.interval(0.95, len(TPR_overall_min)-1, loc=np.mean(TPR_overall_min), scale=st.sem(TPR_overall_min))
        f.write(str(round(TPR_overall_min_mean * 100, 1)) + "\% (+/- " + str(round((TPR_overall_min_mean-TPR_overall_min_confidence[0]) * 100, 1)) + "\%)")
        f.write("\n")


# In[31]:


directories = os.listdir("Dataset")
algorithm_name = ["LogisticRegression", "TreeRegressor", "Decision_Tree", "Discriminant_Analysis", "SVM"]
dataset_name = ["census", "credit", "bank", "compas"]
sensitive_attr = ["gender", "race", "age"]
searches = ["mutation"]
key_num_inp = {}
key_acc = {}
key_AOD_overall = {}
key_TPR_overall = {}
key_AOD_top = {}
key_TPR_top = {}

with open("Results/RQ4-Exp-6(m).csv", 'w') as f:
    f.write("name,num_inputs,accuracy_max,min_TPR\n")
    for drs in directories:
        if not (drs.startswith("Mitigations") or drs.startswith("Run")):
            continue
        print(drs)
        for filename in os.listdir("Dataset" + "/" + drs):
            if filename.endswith("res.csv"):
                print(filename)
                search = ""
                key = ""
                found = False
                mitigation = False
                for s in searches:
                    if s in filename:
                        search = s
                        for an in algorithm_name:
                            if an in filename:
                                for ds in dataset_name:
                                    if ds in filename:
                                        for sa in sensitive_attr:
                                            if sa in filename:
                                                if "Mitigation" in filename:
                                                    key = an + "-" + ds + "-" + sa + "-Tool-Mitigation"
                                                    found = True
                                                    mitigation = True
                                                else:
                                                    key = an + "-" + ds + "-" + sa + "-Tool"
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
                    try:
                        df_time = df[df["counter"] == 0]["timer"]
                    except TypeError as TE:
                        try:
                            df_time = df[df["counter"] == 1]["timer"]
                        except  TypeError as TE:
                            pass
                    if mitigation:
                        try:
                            # limit the results to 6 mins
                            df = df[df["timer"] <= 3600]                                
                        except TypeError as TE:
                            pass
                    else:
                        try:
                            df = df[df["timer"] <= 3600]
                        except TypeError as TE:
                            pass
                    # we expect to have more than just the default parameter.
                    if df.shape[0] == 1:
                        continue
                    lst_acc = []
                    AOD = df["AOD"]
                    TPR = df["TPR"]
                    lst_AOD_overall = []
                    lst_AOD_overall.append(AOD.min())
                    lst_AOD_overall.append(AOD.max())
                    lst_TPR_overall = []
                    lst_TPR_overall.append(TPR.min())
                    lst_TPR_overall.append(TPR.max())
                    df1 = df[df["TPR"] <= df["TPR"].min() + 0.0001]
                    accuracy = df1["score"]
                    lst_acc.append(accuracy.min())
                    lst_acc.append(accuracy.max())
                    lst_AOD_top = []
                    lst_AOD_top.append(df1["AOD"].min())
                    lst_AOD_top.append(df1["AOD"].max())
                    lst_TPR_top = []
                    lst_TPR_top.append(df1["TPR"].min())
                    lst_TPR_top.append(df1["TPR"].max())

                    if key not in key_num_inp:
                        key_num_inp[key] = []
                    key_num_inp[key].append([df.shape[0]])

                    if key not in key_acc:
                        key_acc[key] = []
                    key_acc[key].append(lst_acc)

                    if key not in key_AOD_overall:
                        key_AOD_overall[key] = []
                    key_AOD_overall[key].append(lst_AOD_overall)

                    if key not in key_TPR_overall:
                        key_TPR_overall[key] = []
                    key_TPR_overall[key].append(lst_TPR_overall)

                    if key not in key_AOD_top:
                        key_AOD_top[key] = []
                    key_AOD_top[key].append(lst_AOD_top)

                    if key not in key_TPR_top:
                        key_TPR_top[key] = []
                    key_TPR_top[key].append(lst_TPR_top)
    for key in key_num_inp:
        f.write(key)
        f.write(",")
        num_inp_lst = np.array(key_num_inp[key])[:,0]
        num_inp_mean = np.mean(num_inp_lst)
        num_inp_confidence = st.t.interval(0.95, len(num_inp_lst)-1, loc=np.mean(num_inp_lst), scale=st.sem(num_inp_lst))
        f.write(str(round(num_inp_mean)) + " (+/- " + str(round(num_inp_mean-num_inp_confidence[0])) + ")")
        f.write(",")
        acc_overall_lst = np.array(key_acc[key])
        acc_overall_min = acc_overall_lst[:,0]
        acc_overall_max = acc_overall_lst[:,1]
        acc_overall_min_mean = np.mean(acc_overall_min)
        acc_overall_min_confidence = st.t.interval(0.95, len(acc_overall_min)-1, loc=np.mean(acc_overall_min), scale=st.sem(acc_overall_min))
        acc_overall_max_mean = np.mean(acc_overall_max)
        acc_overall_max_confidence = st.t.interval(0.95, len(acc_overall_max)-1, loc=np.mean(acc_overall_max), scale=st.sem(acc_overall_max))
        f.write(str(round(acc_overall_max_mean * 100, 1))  + "\% (+/- " + str(round((acc_overall_max_mean-acc_overall_max_confidence[0]) * 100, 1)) + "\%)")
        f.write(",")
        AOD_overall_lst = np.array(key_AOD_overall[key])
        AOD_overall_min = AOD_overall_lst[:,0]
        AOD_overall_max = AOD_overall_lst[:,1]
        AOD_overall_min_mean = np.mean(AOD_overall_min)
        AOD_overall_min_confidence = st.t.interval(0.95, len(AOD_overall_min)-1, loc=np.mean(AOD_overall_min), scale=st.sem(AOD_overall_min))
        AOD_overall_max_mean = np.mean(AOD_overall_max)
        AOD_overall_max_confidence = st.t.interval(0.95, len(AOD_overall_max)-1, loc=np.mean(AOD_overall_max), scale=st.sem(AOD_overall_max))
        AOD_top_overall_lst = np.array(key_AOD_top[key])
        AOD_top_overall_min = AOD_top_overall_lst[:,0]
        AOD_top_overall_max = AOD_top_overall_lst[:,1]
        AOD_top_overall_min_mean = np.mean(AOD_top_overall_min)
        AOD_top_overall_min_confidence = st.t.interval(0.95, len(AOD_top_overall_min)-1, loc=np.mean(AOD_top_overall_min), scale=st.sem(AOD_top_overall_min))
        AOD_top_overall_max_mean = np.mean(AOD_top_overall_max)
        AOD_top_overall_max_confidence = st.t.interval(0.95, len(AOD_top_overall_max)-1, loc=np.mean(AOD_top_overall_max), scale=st.sem(AOD_top_overall_max))

        TPR_overall_lst = np.array(key_TPR_overall[key])
        TPR_overall_min = TPR_overall_lst[:,0]
        TPR_overall_max = TPR_overall_lst[:,1]
        TPR_overall_min_mean = np.mean(TPR_overall_min)
        TPR_overall_min_confidence = st.t.interval(0.95, len(TPR_overall_min)-1, loc=np.mean(TPR_overall_min), scale=st.sem(TPR_overall_min))
        TPR_overall_max_mean = np.mean(TPR_overall_max)
        TPR_overall_max_confidence = st.t.interval(0.95, len(TPR_overall_max)-1, loc=np.mean(TPR_overall_max), scale=st.sem(TPR_overall_max))

        TPR_top_overall_lst = np.array(key_TPR_top[key])
        TPR_top_overall_min = TPR_top_overall_lst[:,0]
        TPR_top_overall_max = TPR_top_overall_lst[:,1]
        TPR_top_overall_min_mean = np.mean(TPR_top_overall_min)
        TPR_top_overall_min_confidence = st.t.interval(0.95, len(TPR_top_overall_min)-1, loc=np.mean(TPR_top_overall_min), scale=st.sem(TPR_top_overall_min))
        TPR_top_overall_max_mean = np.mean(TPR_top_overall_max)
        TPR_top_overall_max_confidence = st.t.interval(0.95, len(TPR_top_overall_max)-1, loc=np.mean(TPR_top_overall_max), scale=st.sem(TPR_top_overall_max))
        f.write(str(round(TPR_top_overall_min_mean * 100, 1)) + "\% (+/- " + str(round((TPR_top_overall_min_mean-TPR_top_overall_min_confidence[0]) * 100, 1)) + "\%)")
        f.write("\n")


# In[ ]:




