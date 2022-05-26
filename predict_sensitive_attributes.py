from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
model = "LR"
dataset = "census"
sensitive_param = "sex"
from sklearn.covariance import EmpiricalCovariance
import sklearn.decomposition
import ast
import sys
import numpy as np
import os
# import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef
sys.path.append("./subjects/")
from adf_data.census import census_data
from adf_data.credit import credit_data
from adf_data.bank import bank_data
from adf_data.compas import compas_data

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


@ignore_warnings(category=ConvergenceWarning)
def preduct_sensitive_attributes():
    possible_combos = [("census", 9), ("census", 8), ("credit", 9), ("bank", 1), ("compas", 1), ("compas", 2), ("compas", 3)]
    for dataset, sensitive_param in possible_combos:
        if dataset == "census" and sensitive_param == 9:
            sensitive_name = "gender"
            group_0 = 0  #female
            group_1 = 1  #male
        if dataset == "census" and sensitive_param == 8:
            group_0 = 0
            group_1 = 4
            sensitive_name = "race"
        if dataset == "credit" and sensitive_param == 9:
            group_0 = 0  # male
            group_1 = 1  # female
            sensitive_name = "gender"
        if dataset == "bank" and sensitive_param == 1:  # with 3,5: 0.89; with 2,5: 0.84; with 4,5: 0.05; with 3,4: 0.6
            group_0 = 3
            group_1 = 5
            sensitive_name = "age"
        if dataset == "compas" and sensitive_param == 1:  # sex
            group_0 = 0 # male
            group_1 = 1 # female
            sensitive_name = "gender"
        if dataset == "compas" and sensitive_param == 2:  # age
            group_0 = 0 # under 25
            group_1 = 2 # greater than 45
            sensitive_name = "age"
        if dataset == "compas" and sensitive_param == 3:  # race
            group_0 = 0 # non-Caucasian
            group_1 = 1 # Caucasian
            sensitive_name = "race"
        
        # Pull in the data: 
        data = {"census":census_data, "credit":credit_data, "bank":bank_data, "compas": compas_data}

        X, Y, input_shape, nb_classes = data[dataset]()

        # We ignore anything that is not part of the sensitive attribute. This makes a difference for age. 
        mask = (X[:,sensitive_param-1] == group_0) | (X[:,sensitive_param-1] == group_1)
        X = X[mask]
        Y = Y[mask]


        Y_feature = np.argmax(Y, axis=1)
        Y = X[:,sensitive_param-1]
        X = np.delete(X, sensitive_param-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

        lr = LogisticRegression(max_iter=10000).fit(X_train, y_train)
        dt = DecisionTreeClassifier().fit(X_train, y_train)
        rf = RandomForestClassifier().fit(X_train, y_train)
        sv = LinearSVC(dual=False).fit(X_train, y_train)
        print(f"Dataset: {dataset}; Sensitive feature: {sensitive_name}")
        lr_no = lr.score(X_test,y_test)
        dt_no = dt.score(X_test,y_test)
        rf_no = rf.score(X_test,y_test)
        sv_no = sv.score(X_test,y_test)

        lr_mc_no = matthews_corrcoef(y_test, lr.predict(X_test))
        dt_mc_no = matthews_corrcoef(y_test, dt.predict(X_test))
        rf_mc_no = matthews_corrcoef(y_test, rf.predict(X_test))
        sv_mc_no = matthews_corrcoef(y_test, sv.predict(X_test))

        # Adding original labels
        X = np.insert(X,0, Y_feature, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

        lr = LogisticRegression(max_iter=10000).fit(X_train, y_train)
        dt = DecisionTreeClassifier().fit(X_train, y_train)
        rf = RandomForestClassifier().fit(X_train, y_train)
        sv = LinearSVC(dual=False).fit(X_train, y_train)
        lr_yes = lr.score(X_test,y_test)
        dt_yes = dt.score(X_test,y_test)
        rf_yes = rf.score(X_test,y_test)
        sv_yes = sv.score(X_test,y_test)
        
        lr_mc_yes = matthews_corrcoef(y_test, lr.predict(X_test))
        dt_mc_yes = matthews_corrcoef(y_test, dt.predict(X_test))
        rf_mc_yes = matthews_corrcoef(y_test, rf.predict(X_test))
        sv_mc_yes = matthews_corrcoef(y_test, sv.predict(X_test))

        print(f"LR no label: {lr_no}; LR with label: {lr_yes} || LR matthews no: {lr_mc_no}; LR matthews yes: {lr_mc_yes}")
        print(f"DT no label: {dt_no}; DT with label: {dt_yes} || DT matthews no: {dt_mc_no}; DT matthews yes: {dt_mc_yes}")
        print(f"RF no label: {rf_no}; RF with label: {rf_yes} || RF matthews no: {rf_mc_no}; RF matthews yes: {rf_mc_yes}")
        print(f"SV no label: {sv_no}; SV with label: {sv_yes} || SV matthews no: {sv_mc_no}; SV matthews yes: {sv_mc_yes}")


        assert(len(Y[Y==group_0])/len(Y) + len(Y[Y==group_1])/len(Y) > .999 and len(Y[Y==group_0])/len(Y) + len(Y[Y==group_1])/len(Y)<=1) # Dummy check
        print(f"Baseline (simply predicting the higher occurance in the data): {max(len(Y[Y==group_0])/len(Y), len(Y[Y==group_1])/len(Y))}")
preduct_sensitive_attributes()