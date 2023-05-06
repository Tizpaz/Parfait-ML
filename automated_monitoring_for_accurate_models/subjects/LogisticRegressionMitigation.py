import itertools
import time
import xml_parser

import numpy as np
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import GridSearch, ExponentiatedGradient
from fairlearn.reductions import EqualizedOdds

def logistic_regression_mitigation(inp, X_train, X_test, y_train, y_test, sensitive_param):
    print(inp)
    arr, features = xml_parser.xml_parser('logistic_regression_mitigation_Params.xml',inp)
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
        clf = ExponentiatedGradient(LogisticRegression(penalty=arr[1], dual = arr[2], tol = arr[3],
        C = arr[4], fit_intercept = arr[5], intercept_scaling = arr[6],
        solver=arr[0], max_iter = arr[7], multi_class=arr[8], l1_ratio = arr[9], random_state=2019),
        constraints=EqualizedOdds(), eps = arr[10], max_iter = arr[11], eta0 = arr[12],
        run_linprog_step = arr[13])
        clf.fit(X_train, y_train, sensitive_features=X_train[:,sensitive_param-1])
        preds = clf.predict(X_test)
        score = np.sum(y_test == preds)/len(y_test)

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
