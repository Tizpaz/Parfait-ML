import itertools
import time
import xml_parser

import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def logistic_regression(inp, X_train, X_test, y_train, y_test, sensitive_param = None, dataset_name = "", save_model=False):
    print(inp)
    arr, features = xml_parser.xml_parser('logistic_regression_Params.xml',inp)
    print(arr)
    try:
        # domain-specific constraints
        if (arr[0] == 'lbfgs' and arr[2] == True):
            arr[2] = False
        if (arr[0] == 'lbfgs' and arr[1] == "l1") or (arr[0] == 'lbfgs' and arr[1] == "elasticnet"):
            arr[1] = "l2"
        if (arr[0] == 'newton-cg' and arr[2] == True):
            arr[2] = False
        if (arr[0] == 'newton-cg' and arr[1] == "l1") or (arr[0] == 'newton-cg' and arr[1] == "elasticnet"):
            arr[1] = "l2"
        if (arr[0] == 'sag' and arr[2] == True):
            arr[2] = False
        if (arr[0] == 'sag' and arr[1] == "l1") or (arr[0] == 'sag' and arr[1] == "elasticnet"):
            arr[1] = "l2"
        if (arr[0] == 'liblinear' and arr[1] == "elasticnet"):
            arr[1] = "l1"
        if (arr[0] == 'lbfgs' and arr[1] == "elasticnet"):
            arr[1] = "l1"
        if (arr[0] == 'saga' and arr[2] == True):
            arr[2] = False
        if (arr[1] == "l1" and arr[2] == True):
            arr[2] = False
        if (arr[0] == 'liblinear' and arr[8] == 'multinomial'):
            arr[8] = 'ovr'
        if (arr[0] == 'liblinear' and arr[1] == 'none'):
            arr[1] = 'l1'
        if(arr[1] != "elasticnet"):
            arr[9] = None
        else:
            arr[9] = np.random.random()
        arr[10] = None
        arr[11] = 2019
        arr[12] = 0
        arr[14] = None

        clf = LogisticRegression(penalty=arr[1], dual = arr[2], tol = arr[3],
        C = arr[4], fit_intercept = arr[5], intercept_scaling = arr[6],
        solver=arr[0], max_iter = arr[7], multi_class=arr[8], l1_ratio = arr[9],
        random_state=arr[11], class_weight = arr[10], verbose = arr[12],
        warm_start = arr[13], n_jobs=arr[14])
        fitted_clf = clf.fit(X_train, y_train)
        if save_model:
            with open(f"./trained_models/logisticRegression_{dataset_name}_{sensitive_param}_{arr[0]}_{arr[1]}_{arr[2]}_{arr[3]}_{arr[4]}_{arr[5]}_{arr[6]}_{arr[7]}\
            _{arr[8]}_{arr[9]}_{arr[10]}_{arr[11]}_{arr[12]}_{arr[13]}_{arr[14]}.pkl", "wb") as file:
                pickle.dump(fitted_clf, file)
        score = clf.score(X_test, y_test)
        preds = clf.predict(X_test)
#        print("here1")
    except ValueError as ve:
#	pass
        print("here you go------------------------------------")
        print(arr)
        print(ve)
        return False, None, arr, None, None, features
    # except KeyError:
    #     # print("here3")
    #     return False
    print(arr)
    return True, clf, arr, score, preds, features
