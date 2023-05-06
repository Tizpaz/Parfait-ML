from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
import xml_parser
from sklearn.datasets import make_regression
import random
from sklearn.metrics import accuracy_score
import pickle

def trans(x):
    if(x >= 0.5):
        return 1
    else:
        return 0

def TreeRegress(inp, X_train, X_test, y_train, y_test, sensitive_param = None, dataset_name = "", save_model=False):
    arr, features = xml_parser.xml_parser('TreeRegressor_Params.xml',inp)
    write_file = None

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
    arr[7] = 0.0
    arr[8] = True

    # if(arr[13] == 'None'):
    arr[13] = None
    arr[14] = 2019
    arr[15] = 0
    arr[16] = None
    arr[17] = None
    # else:
    #     arr[13] = random.randint(int(X_train.shape[0]/4), int(3*X_train.shape[0]/4))

    try:
        random_forest = RandomForestClassifier(n_estimators=arr[11], criterion=arr[0],
            max_depth=arr[1], min_samples_split=arr[2], min_samples_leaf=arr[3],
            min_weight_fraction_leaf=arr[4], max_features=arr[5],
            max_leaf_nodes=arr[6], min_impurity_decrease=arr[7],
            bootstrap=arr[8],oob_score=arr[9], warm_start=arr[10], ccp_alpha = arr[12],
            max_samples = arr[13], random_state = arr[14], verbose = arr[15], n_jobs = arr[16])
            # min_impurity_split = arr[17]) I switched from RandomForestRegressor to RandomFOrestClassifier, which has "predict_proba". Gonna just remove this "sneakily for now" ;)
    except ValueError as VE:
        print("error2: " + str(VE))
        return False, None, arr, None, None, features

    try:
        fitted_forest = random_forest.fit(X_train, y_train)
        if save_model:
            write_file = f"./trained_models/randomForest_{dataset_name}_{sensitive_param}_{arr[0]}_{arr[1]}_{arr[2]}_{arr[3]}_{arr[4]}_{arr[5]}_{arr[6]}_{arr[7]}"\
            f"_{arr[8]}_{arr[9]}_{arr[10]}_{arr[11]}_{arr[12]}_{arr[13]}_{arr[14]}_{arr[15]}_{arr[16]}_{arr[17]}.pkl"
            with open(write_file, "wb") as file:
                pickle.dump(fitted_forest, file)
        x_pred = random_forest.predict(X_test)
        x_pred = list(map(trans, x_pred))
        score = accuracy_score(x_pred, y_test)
        preds = random_forest.predict(X_test)
    except ValueError as VE:
        print("error3: " + str(VE))
        return False, None, arr, None, None, features, write_file

    print(arr)
    preds = list(map(trans, preds))
    return True, random_forest, arr, score, preds, features, write_file
