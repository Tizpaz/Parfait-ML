import itertools
import time
import xml_parser

import numpy as np
from sklearn.svm import SVC, NuSVC, LinearSVC
import pickle

def SVM(inp, X_train, X_test, y_train, y_test, sensitive_param = None, dataset_name = "", save_model=False):
    print(inp)
    arr, features = xml_parser.xml_parser('SVM_Params.xml',inp)
    print(arr)
    try:
        # domain-specific constraints
        if(arr[8] == 'none'):
            arr[8] = None
        # The combination of penalty='l1' and loss='hinge' is not supported, Parameters: penalty='l1', loss='hinge', dual=False
        if(arr[0] == 'l1' and arr[1] == 'hinge'):
            arr[2] = True
            arr[0] = 'l2'
        if(arr[0] == 'l2' and arr[1] == 'hinge'):
            arr[2] = True
        if(arr[0] == 'l1' and arr[1] == 'squared_hinge'):
            arr[2] = False
        arr[5] = 'ovr'
        arr[9] = 2019
        arr[10] = 0
        clf = LinearSVC(penalty = arr[0], loss = arr[1], dual = arr[2], tol = arr[3], C = arr[4],
                        multi_class = arr[5], fit_intercept = arr[6], intercept_scaling = arr[7],
                        class_weight = arr[8], verbose=arr[10], random_state=arr[9], max_iter=arr[11])
        fitted_clf = clf.fit(X_train, y_train)
        if save_model:
            with open(f"./trained_models/svm_{dataset_name}_{sensitive_param}_{arr[0]}_{arr[1]}_{arr[2]}_{arr[3]}_{arr[4]}_{arr[5]}_{arr[6]}_{arr[7]}\
            _{arr[8]}_{arr[9]}_{arr[10]}_{arr[11]}.pkl", "wb") as file:
                pickle.dump(fitted_clf, file)
        
        score = clf.score(X_test, y_test)
        preds = clf.predict(X_test)
    except ValueError as ve:
        print("here you go------------------------------------")
        print(arr)
        print(ve)
        return False, None, arr, None, None, features
    print(arr)
    return True, clf, arr, score, preds, features
