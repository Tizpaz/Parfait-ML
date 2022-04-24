import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
import xml_parser
import random
from fairlearn.reductions import GridSearch, ExponentiatedGradient
from fairlearn.reductions import EqualizedOdds

def DecisionTreeMitigation(inp, X_train, X_test, y_train, y_test, sensitive_param = None):
    arr, features = xml_parser.xml_parser('Decision_Tree_Classifier_Mitigation_Params.xml',inp)

    if(arr[2] == 'None'):
        arr[2] = None
    else:
        if np.random.random() < 0.25:
            arr[2] = 4
        else:
            arr[2] = random.randint(3, 20)

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

    try:
        clf = ExponentiatedGradient(DecisionTreeClassifier(criterion=arr[0], splitter=arr[1], max_depth=arr[2],
                min_samples_split=arr[3], min_samples_leaf=arr[4], min_weight_fraction_leaf=arr[5],
                max_features=arr[6], random_state=arr[7], max_leaf_nodes=arr[8],
                min_impurity_decrease=arr[9], class_weight=arr[11],
                ccp_alpha = arr[12]), constraints=EqualizedOdds(), eps = arr[13], max_iter = arr[14],
                eta0 = arr[15], run_linprog_step = arr[16])
        clf.fit(X_train, y_train, sensitive_features=X_train[:,sensitive_param-1])
        preds = clf.predict(X_test)
        score = np.sum(y_test == preds)/len(y_test)

    except ValueError as ve:
#	pass
        print("here you go------------------------------------")
        print(arr)
        print(ve)
        return False, None, None, None, None, None

    print(arr)
    return True, clf, arr, score, preds, features
