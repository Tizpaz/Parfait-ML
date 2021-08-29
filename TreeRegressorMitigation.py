from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time
import xml_parser
from sklearn.datasets import make_regression
import random
from sklearn.metrics import accuracy_score
from fairlearn.reductions import GridSearch, ExponentiatedGradient
from fairlearn.reductions import EqualizedOdds

def trans(x):
    if(x >= 0.5):
        return 1
    else:
        return 0

def TreeRegressMitigation(inp, X_train, X_test, y_train, y_test, sensitive_param = None):
    arr, features = xml_parser.xml_parser('TreeRegressorMitigation_Params.xml',inp)

    # mae is much slower than mse
    if(arr[0] == 'mae'):
        rand = np.random.random()
        if(rand < 0.98):
            arr[0] = 'mse'

    # value for max depth
    if(arr[1] == 'None'):
        arr[1] = None
    else:
        arr[1] = random.randint(5, 20)

    # value for max_features
    if arr[5] == 'None':
        arr[5] = None

    if arr[6] == 'None':
        arr[6] = None

    arr[8] = True

    # if(arr[13] == 'None'):
    arr[13] = None
    # else:
    #     arr[13] = random.randint(int(X_train.shape[0]/4), int(3*X_train.shape[0]/4))

    try:
        random_forest = ExponentiatedGradient(RandomForestRegressor(n_estimators=arr[11], criterion=arr[0],
            max_depth=arr[1], min_samples_split=arr[2], min_samples_leaf=arr[3],
            min_weight_fraction_leaf=arr[4], max_features=arr[5],
            max_leaf_nodes=arr[6], min_impurity_decrease=arr[7],
            bootstrap=arr[8],oob_score=arr[9], warm_start=arr[10], ccp_alpha = arr[12],
            max_samples = arr[13], random_state = 2019), constraints=EqualizedOdds(), eps = arr[14],
            max_iter = arr[15], eta0 = arr[16], run_linprog_step = arr[17])
    except ValueError as VE:
        print("error2: " + str(VE))
        return False, None, arr, None, None, features

    try:
        random_forest.fit(X_train, y_train, sensitive_features=X_train[:,sensitive_param-1])
        x_pred = random_forest.predict(X_test)
        x_pred = list(map(trans, x_pred))
        score = accuracy_score(x_pred, y_test)
        preds = random_forest.predict(X_test)
    except ValueError as VE:
        print("error3: " + str(VE))
        return False, None, arr, None, None, features

    print(arr)
    preds = list(map(trans, preds))
    return True, random_forest, arr, score, preds, features
