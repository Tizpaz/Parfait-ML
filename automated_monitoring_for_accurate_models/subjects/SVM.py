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
    write_file = None
    try:
        # domain-specific constraints
        print(arr)
        arr[1] = 'linear'
        if(arr[9] == 'none'):
            arr[9] = None
        arr[6] = True
        arr[14] = 2019
        arr[10] = 0
        if arr[12] != "ovr":
            arr[13] = False
        clf = SVC(C=arr[0], kernel=arr[1], degree=arr[2], gamma=arr[3], coef0=arr[4], shrinking=arr[5], probability=arr[6], tol=arr[7], 
            cache_size=arr[8], class_weight=arr[9], verbose=arr[10], max_iter=arr[11], decision_function_shape=arr[12],
            break_ties=arr[13], random_state=arr[14])
        fitted_clf = clf.fit(X_train, y_train)
        if save_model:
            write_file = f"./trained_models/svm_{dataset_name}_{sensitive_param}_{arr[0]}_{arr[1]}_{arr[2]}_{arr[3]}_{arr[4]}_{arr[5]}_{arr[6]}_{arr[7]}"\
            f"_{arr[8]}_{arr[9]}_{arr[10]}_{arr[11]}_{arr[11]}_{arr[12]}_{arr[13]}_{arr[14]}.pkl"
            with open(write_file, "wb") as file:
                pickle.dump(fitted_clf, file)
        
        score = clf.score(X_test, y_test)
        preds = clf.predict(X_test)
    except ValueError as ve:
        print("here you go------------------------------------")
        print(arr)
        print(ve)
        return False, None, arr, None, None, features, write_file
    print(arr)
    return True, clf, arr, score, preds, features, write_file
