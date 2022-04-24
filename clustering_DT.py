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
from sklearn import preprocessing, neighbors, tree
from sklearn.model_selection import cross_validate
from subprocess import call
# from IPython.display import Image,display
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from itertools import cycle, islice
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from time import sleep
import pydotplus
import collections
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test_case", help='The name of file with test cases ')
parser.add_argument("--clusters", help='The number of clusters', default=2)
args = parser.parse_args()
filename = args.test_case
k = int(args.clusters)

def ClusterFunc(X,k):
    return SpectralClustering(n_clusters=k, random_state=10).fit(X)

df = pd.read_csv("Dataset" + "/" + filename)
df = df.where(pd.notnull(df), "None")
if "/" in filename:
    filename = filename.split("/")[-1]
# remove last row since it can be null row
df = df[:-1]
# make sure to only inclue test cases generated up to 4 hours
df = df[df["timer"] <= 14400]
X = np.concatenate((np.array(df["score"]).reshape(-1,1),np.array(df["AOD"]).reshape(-1,1)),axis=1)
startTime = int(round(time.time() * 1000))
SC = ClusterFunc(X,k)
endTime = int(round(time.time() * 1000))
clust_time = endTime - startTime
y_pred = SC.labels_.astype(int)
# cluster 0 is blue, cluster 1 is orange, cluster 2 is green, cluster 3 is pink, cluster 4 is brown, cluster 5 is pruple!
colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                        '#f781bf', '#a65628', '#984ea3',
                                        '#999999', '#e41a1c', '#dede00']), int(max(y_pred) + 1))))
color_name = ["blue", "orange", "green"]
plt.figure(dpi=150)
plt.scatter(X[:,0],X[:,1], color=colors[y_pred])
plt.xlabel('Accuracy')
plt.ylabel('Average Odds Difference (AOD)')
plt.savefig("Results/" + filename + "_clustered_" + str(k) + ".png", dpi=150)
plt.close()
# plt.show()
df = df.drop(['score', 'AOD', 'TPR', 'FPR', 'timer', 'counter'], axis=1)
df["label"] = y_pred
y = y_pred
X_df = df.drop(['label'],1)
X = np.array(X_df)
XX = []
le = preprocessing.LabelBinarizer()
flag_contains = False
header = []

flag_contain_None = False
for index,col in enumerate(X.T):
    if "None" in X.T[index]:
        flag_contain_None = True
    max_val = -1 * sys.maxsize
    min_val = sys.maxsize
    falg_contain_non_num = False
    Not_None = False
    if flag_contain_None:
        for i, x in enumerate(X.T[index]):
            if(x != "None" and str(x).isnumeric() == False):
                falg_contain_non_num = True
                Not_None = True
                break
            elif (x != "None"):
                Not_None = True
                if(x >= max_val):
                    max_val = x
                if(x <= min_val):
                    min_val = x
        if Not_None:
            for i, x in enumerate(X.T[index]):
                if falg_contain_non_num:
                    X[i,index] = str(x)
                elif X[i,index] == "None":
                    X[i,index] = max_val+1
        flag_contain_None = False
        falg_contain_non_num = False
failed_flag = False
for index, x in enumerate(X.T):
    for e in x:
        if isinstance(e, str):
            flag_contains = True
    if flag_contains:
        try:
            le_fit = le.fit(x)
            arr = le.transform(x)
        except:
            failed_flag = True
            break
        for c in range(len(arr[0])):
            try:
                XX.append(arr[:,c])
                header.append(list(X_df.columns.values)[index] + "==" + str(sorted(set(x))[c]))
            except:
                failed_flag = True
                break
    else:
        header.append(list(X_df.columns.values)[index])
        XX.append(x)

    flag_contains = False
if failed_flag:
    print("failed to convert values of a feature to numerical values")
    exit
XX = np.asarray(XX,dtype='float64')
X = XX.T

accuracy_max = 0.0
precision_max = 0.0
recall_max = 0.0
rTime_max = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

accuracy_avg = 0
precision_avg = 0
recall_avg = 0
clf = None
alpha = 0.02
counter = 0
while True:
    startTime = int(round(time.time() * 1000))
    clf_temp = DecisionTreeClassifier(criterion="gini",splitter='best',max_depth=3, ccp_alpha = alpha)
    clf_temp.fit(X_train,y_train)
    accuracy = clf_temp.score(X_test,y_test)
    y_predict = clf_temp.predict(X_test)
    precision = precision_score(y_test,y_predict,average=None)
    recall = recall_score(y_test,y_predict,average=None)

    endTime = int(round(time.time() * 1000))
    rTime = endTime - startTime
    accuracy_avg = accuracy_avg + accuracy
    precision_avg = precision_avg + precision
    recall_avg = recall_avg + recall
    if(accuracy > accuracy_max):
        accuracy_max = accuracy
        precision_max = precision
        recall_max = recall
        clf = clf_temp
        rTime_max = rTime
    depth_tree = clf.get_depth()
    leaves_tree = clf.get_n_leaves()
    if(depth_tree > 1):
        break
    else:
        alpha = 0.0
    counter += 1
    if(counter > 5):
        break

dir_res = "Results/" + filename +'_tree'
out = dir_res + '.dot'
dot_data = tree.export_graphviz(clf,out_file=None,feature_names=header, filled=True, rounded = True, impurity = False)
graph = pydotplus.graph_from_dot_data(dot_data)
nodes = graph.get_node_list()
edges = graph.get_edge_list()
visited_node = {}
for edge in edges:
    source = edge.get_source()
    dest = edge.get_destination()
    edge.obj_dict['attributes']['headlabel'] = ''
    if 'label' in nodes[int(source)].obj_dict['attributes']:
        get_label_node = nodes[int(source)].obj_dict['attributes']['label']
        if "==" in get_label_node:
            nodes[int(source)].obj_dict['attributes']['label'] = get_label_node.replace("<= 0.5","")
        elif "<=" in get_label_node:
            nodes[int(source)].obj_dict['attributes']['label'] = get_label_node.replace("<=",">")
    if 'label' in nodes[int(dest)].obj_dict['attributes']:
        get_label_node = nodes[int(dest)].obj_dict['attributes']['label']
        if "==" in get_label_node:
            nodes[int(dest)].obj_dict['attributes']['label'] = get_label_node.replace("<= 0.5","")
        elif "<=" in get_label_node:
            nodes[int(dest)].obj_dict['attributes']['label'] = get_label_node.replace("<=",">")
    if int(source) not in visited_node:
        edge.set_label("False")
        visited_node[int(source)] = True
    else:
        edge.set_label("True")
attributes_name = set()
for node in nodes:
    if node.get_name() not in ('node', 'edge', '\"\\n\"'):
        values = clf.tree_.value[int(node.get_name())][0]
        if node.get_attributes()['label'].startswith('\"samples'):
            node.set_fillcolor(colors[np.argmax(values)])
        else:
            if '==' in node.get_attributes()['label']:
                att_name = node.get_attributes()['label'].split('\\n')[0].replace('\"','').replace(" ","")
                attributes_name.add(att_name)
            elif ' > ' in node.get_attributes()['label']:
                att_name = node.get_attributes()['label'].split('\\n')[0].replace('\"','').replace(" ","")
                att_name = att_name.split('>')[0] + '>' + str(round(float(att_name.split('>')[1]),2))
                attributes_name.add(att_name)
            else:
                print("WARNING: There are more cases to consider!!")
                print(node.get_attributes()['label'])
            node.set_fillcolor('whitesmoke')
print_out ='accuracy,precision,recall,total_num_data,total_num_test,tree_computation_time,clust_computation_time\n'
print(print_out)
print(str(accuracy) + "," +  str(precision) + "," + str(recall) + "," + str(len(X)) + "," +  str(len(X_test)) + "," + str(rTime) + "," + str(clust_time))
graph.write_png(dir_res + ".png")
