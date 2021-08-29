#!/usr/bin/env python
# coding: utf-8
# make sure that the data for (10) runs are inside ``Dataset'' folder.
import pandas as pd
import numpy as np
import sys
import time
import os
import scipy.stats as st

directories = os.listdir("Dataset")
algorithm_name = ["LogisticRegression", "TreeRegressor", "Decision_Tree", "Discriminant_Analysis", "SVM"]
dataset_name = ["census", "credit", "bank", "compas"]
sensitive_attr = ["gender", "race", "age"]
searches = ["random", "mutation", "coverage"]
key_num_inp = {}
key_acc = {}
key_AOD_overall = {}
key_TPR_overall = {}
key_AOD_top = {}
key_TPR_top = {}

with open("Results/RQ1.csv", 'w') as f:
    f.write("name,num_inputs,accuracy_min,accuracy_max,overall_min_AOD,overall_max_AOD,min_AOD,max_AOD,overall_min_TPR,overall_max_TPR,min_TPR,max_TPR\n")
    for drs in directories:
        if not drs.startswith("Run"):
            continue
        print(drs)
        for filename in os.listdir("Dataset" + "/" + drs):
            if filename.endswith("res.csv"):
                print(filename)
                search = ""
                key = ""
                found = False
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
                df = pd.read_csv("Dataset" + "/" + drs + "/" + filename)
                try:
                    df = df[df["timer"] <= 14500]
                except TypeError as TE:
                    pass
                accuracy = df["score"]
                lst_acc = []
                lst_acc.append(accuracy.min())
                lst_acc.append(accuracy.max())
                AOD = df["AOD"]
                TPR = df["TPR"]
                lst_AOD_overall = []
                lst_AOD_overall.append(AOD.min())
                lst_AOD_overall.append(AOD.max())
                lst_TPR_overall = []
                lst_TPR_overall.append(TPR.min())
                lst_TPR_overall.append(TPR.max())
                df1 = df[df["score"] >= accuracy.max() - 0.01]
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
        f.write(str(round(acc_overall_min_mean * 100, 1)) + "\% (+/- " + str(round((acc_overall_min_mean-acc_overall_min_confidence[0]) * 100, 1)) + "\%)")
        f.write(",")
        f.write(str(round(acc_overall_max_mean * 100, 1))  + "\% (+/- " + str(round((acc_overall_max_mean-acc_overall_max_confidence[0]) * 100, 1)) + "\%)")
        f.write(",")
        AOD_overall_lst = np.array(key_AOD_overall[key])
        AOD_overall_min = AOD_overall_lst[:,0]
        AOD_overall_max = AOD_overall_lst[:,1]
        AOD_overall_min_mean = np.mean(AOD_overall_min)
        AOD_overall_min_confidence = st.t.interval(0.95, len(AOD_overall_min)-1, loc=np.mean(AOD_overall_min), scale=st.sem(AOD_overall_min))
        AOD_overall_max_mean = np.mean(AOD_overall_max)
        AOD_overall_max_confidence = st.t.interval(0.95, len(AOD_overall_max)-1, loc=np.mean(AOD_overall_max), scale=st.sem(AOD_overall_max))
        f.write(str(round(AOD_overall_min_mean * 100, 1)) + "\% (+/- " + str(round((AOD_overall_min_mean-AOD_overall_min_confidence[0]) * 100, 1)) + "\%)")
        f.write(",")
        f.write(str(round(AOD_overall_max_mean * 100, 1)) + "\% (+/- " + str(round((AOD_overall_max_mean-AOD_overall_max_confidence[0]) * 100, 1)) + "\%)")
        f.write(",")

        AOD_top_overall_lst = np.array(key_AOD_top[key])
        AOD_top_overall_min = AOD_top_overall_lst[:,0]
        AOD_top_overall_max = AOD_top_overall_lst[:,1]
        AOD_top_overall_min_mean = np.mean(AOD_top_overall_min)
        AOD_top_overall_min_confidence = st.t.interval(0.95, len(AOD_top_overall_min)-1, loc=np.mean(AOD_top_overall_min), scale=st.sem(AOD_top_overall_min))
        AOD_top_overall_max_mean = np.mean(AOD_top_overall_max)
        AOD_top_overall_max_confidence = st.t.interval(0.95, len(AOD_top_overall_max)-1, loc=np.mean(AOD_top_overall_max), scale=st.sem(AOD_top_overall_max))
        f.write(str(round(AOD_top_overall_min_mean * 100, 1)) + "\% (+/- " + str(round((AOD_top_overall_min_mean-AOD_top_overall_min_confidence[0]) * 100, 1)) + "\%)")
        f.write(",")
        f.write(str(round(AOD_top_overall_max_mean * 100, 1)) + "\% (+/- " + str(round((AOD_top_overall_max_mean-AOD_top_overall_max_confidence[0]) * 100, 1)) + "\%)")
        f.write(",")

        TPR_overall_lst = np.array(key_TPR_overall[key])
        TPR_overall_min = TPR_overall_lst[:,0]
        TPR_overall_max = TPR_overall_lst[:,1]
        TPR_overall_min_mean = np.mean(TPR_overall_min)
        TPR_overall_min_confidence = st.t.interval(0.95, len(TPR_overall_min)-1, loc=np.mean(TPR_overall_min), scale=st.sem(TPR_overall_min))
        TPR_overall_max_mean = np.mean(TPR_overall_max)
        TPR_overall_max_confidence = st.t.interval(0.95, len(TPR_overall_max)-1, loc=np.mean(TPR_overall_max), scale=st.sem(TPR_overall_max))
        f.write(str(round(TPR_overall_min_mean * 100, 1)) + "\% (+/- " + str(round((TPR_overall_min_mean-TPR_overall_min_confidence[0]) * 100, 1)) + "\%)")
        f.write(",")
        f.write(str(round(TPR_overall_max_mean * 100, 1)) + "\% (+/- " + str(round((TPR_overall_max_mean-TPR_overall_max_confidence[0]) * 100, 1)) + "\%)")
        f.write(",")

        TPR_top_overall_lst = np.array(key_TPR_top[key])
        TPR_top_overall_min = TPR_top_overall_lst[:,0]
        TPR_top_overall_max = TPR_top_overall_lst[:,1]
        TPR_top_overall_min_mean = np.mean(TPR_top_overall_min)
        TPR_top_overall_min_confidence = st.t.interval(0.95, len(TPR_top_overall_min)-1, loc=np.mean(TPR_top_overall_min), scale=st.sem(TPR_top_overall_min))
        TPR_top_overall_max_mean = np.mean(TPR_top_overall_max)
        TPR_top_overall_max_confidence = st.t.interval(0.95, len(TPR_top_overall_max)-1, loc=np.mean(TPR_top_overall_max), scale=st.sem(TPR_top_overall_max))
        f.write(str(round(TPR_top_overall_min_mean * 100, 1)) + "\% (+/- " + str(round((TPR_top_overall_min_mean-TPR_top_overall_min_confidence[0]) * 100, 1)) + "\%)")
        f.write(",")
        f.write(str(round(TPR_top_overall_max_mean * 100, 1)) + "\% (+/- " + str(round((TPR_top_overall_max_mean-TPR_top_overall_max_confidence[0]) * 100, 1)) + "\%)")
        f.write("\n")
