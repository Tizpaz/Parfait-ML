import itertools
import time
import xml_parser

import numpy as np
from sklearn.svm import SVC

def SVM(inp, X_train, X_test, y_train, y_test):
    print(inp)
    arr, features = xml_parser.xml_parser('SVM_Params.xml',inp)
    print(arr)
    try:
        # domain-specific constraints
        if(arr[9] == 'none'):
            arr[9] = None
        # default max_itr is -1
        max_itr = 5000
        if(arr[1] == 'rbf'):
            arr[3] = 'scale'
        #     max_itr = 100
        # break_ties must be False when decision_function_shape is 'ovo'
        if(arr[10] == 'ovo'):
            arr[11] = False
        clf = SVC(C = arr[0], kernel= arr[1], degree= arr[2], gamma = arr[3], coef0 = arr[4],
                shrinking = arr[5], probability = arr[6], tol = arr[7], cache_size = arr[8],
                class_weight = arr[9], verbose=False, max_iter = max_itr, decision_function_shape=arr[10],
                break_ties = arr[11], random_state=2019)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        preds = clf.predict(X_test)
    except ValueError as ve:
        print("here you go------------------------------------")
        print(arr)
        print(ve)
        return False, None, arr, None, None, features
    print(arr)
    return True, clf, arr, score, preds, features
