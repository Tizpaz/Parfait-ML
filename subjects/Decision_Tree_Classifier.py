import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
import xml_parser
import random
import pickle

def DecisionTree(inp, X_train, X_test, y_train, y_test, sensitive_param = None, dataset_name = "", save_model = False):
    arr, features = xml_parser.xml_parser('Decision_Tree_Classifier_Params.xml',inp)

    if(arr[2] == 'None'):
        arr[2] = None
    else:
        arr[2] = random.randint(5, 20)

    # if np.random.random() < 0.8:
    #     arr[4] = int(arr[4])
    # else:
    #     arr[4] = arr[4]/50.0

    if arr[6] == 'None':
        arr[6] = None

    # if arr[7] == 'None':
    #      arr[7] = None
    # else:
    arr[7] = 2019

    arr[8] = None

    arr[9] = 0.0

    arr[10] = 0.0

    arr[11] = None
    # elif arr[11] == 'weighted':
    #     weight_lst = {}
    #     for class_num in range(2):
    #         weight_lst[class_num] = random.randint(1, 5)
    #     arr[11] = weight_lst

    # X = StandardScaler().fit_transform(X)
    try:
        clf = DecisionTreeClassifier(criterion=arr[0], splitter=arr[1], max_depth=arr[2],
                min_samples_split=arr[3], min_samples_leaf=arr[4], min_weight_fraction_leaf=arr[5],
                max_features=arr[6], random_state=arr[7], max_leaf_nodes=arr[8],
                min_impurity_decrease=arr[9], class_weight=arr[11],
                ccp_alpha = arr[12])
        fitted_clf = clf.fit(X_train, y_train)
        if save_model:
            with open(f"./trained_models/decisionTree_{dataset_name}_{sensitive_param}_{arr[0]}_{arr[1]}_{arr[2]}_{arr[3]}_{arr[4]}_{arr[5]}_{arr[6]}_{arr[7]}\
            _{arr[8]}_{arr[9]}_{arr[11]}_{arr[12]}.pkl", "wb") as file:
                pickle.dump(fitted_clf, file)

        score = clf.score(X_test, y_test)
        preds = clf.predict(X_test)

    except ValueError as ve:
#	pass
        print("here you go------------------------------------")
        print(arr)
        print(ve)
        return False, None, None, None, None, None

    print(arr)
    return True, clf, arr, score, preds, features
