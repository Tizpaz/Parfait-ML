# Implementations based on Flash code: https://github.com/joymallyac/Fairway/blob/master/Multiobjective%20Optimization/optimizer/flash.py
import sys
sys.path.append("./subjects/")
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
if sys.version_info.major==2:
    from Queue import PriorityQueue
else:
    from queue import PriorityQueue
import os
import random,time
import copy
from scipy.stats import randint
import csv
import argparse
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate, true_positive_rate
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference

from adf_utils.config import census, credit, bank, compas
from adf_data.census import census_data
from adf_data.credit import credit_data
from adf_data.bank import bank_data
from adf_data.compas import compas_data
import xml_parser
import xml_parser_domains


from functools import wraps
import errno
import signal

class TimeoutError(Exception):
    pass

def timeout(seconds=10):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError("time_error")

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


def check_for_fairness(X, y_pred, y_true, a, X_new = None, Y_new = None):
    parities = []
    impacts = []
    eq_odds = []
    metric_frames = []
    metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'false positive rate': false_positive_rate,
        'true positive rate': true_positive_rate,
        'selection rate': selection_rate,
        'count': lambda y_true, y_pred: y_true.shape[0]
    }

    parity = demographic_parity_difference(y_true = y_true, y_pred = y_pred, sensitive_features = a)
    di = demographic_parity_ratio(y_true = y_true, y_pred = y_pred, sensitive_features = a)
    eod = equalized_odds_difference(y_true = y_true, y_pred = y_pred, sensitive_features = a)
    metric_frame = MetricFrame(metrics, y_true, y_pred, sensitive_features=a)
    print(metric_frame.by_group)
    parities.append(parity)
    impacts.append(di)
    eq_odds.append(eod)
    metric_frames.append(metric_frame)
    return metric_frame.by_group["accuracy"], metric_frame.by_group["precision"], metric_frame.by_group["recall"], metric_frame.by_group["true positive rate"], metric_frame.by_group["false positive rate"], metric_frame.by_group["count"]

def flash_fair_LSR(X_train, y_train, X_valid, y_valid, biased_col, group_0, group_1, n_obj, start_time, algorithm):  # biased_col can be "sex" or "race", n_obj can be "ABCD" or "AB" or "CD"

    def convert_lsr(index):  # 30 2 2 100
        a = int(index / 400 + 1)
        b = int(index % 400 / 200 + 1)
        c = int(index % 200 / 100 + 1)
        d = int(index % 100 + 10)
        return a, b, c, d

    all_case = set(range(0, 12000))
    modeling_pool = random.sample(all_case, 20)

    List_X = []
    List_Y = []
    List_acc = []
    List_AOD = []
    List_EOD = []

    for i in range(len(modeling_pool)):
        temp = convert_lsr(modeling_pool[i])
        List_X.append(temp)
        if(algorithm == "LogisticRegression"):
            p1 = temp[0]
            if temp[1] == 1:
                p2 = 'l1'
            else:
                p2 = 'l2'
            if temp[2] == 1:
                p3 = 'liblinear'
            else:
                p3 = 'saga'
            p4 = temp[3]
            model = LogisticRegression(C=p1, penalty=p2, solver=p3, max_iter=p4)
        elif(algorithm == "Decision_Tree_Classifier"):
            p1 = temp[0]
            if p1 == 1:
                p1 = p1 + 1
            if temp[1] == 1:
                p2 = 'gini'
            else:
                p2 = 'entropy'
            if temp[2] == 1:
                p3 = 'best'
            else:
                p3 = 'random'
            p4 = temp[3]
            model = DecisionTreeClassifier(criterion=p2, splitter=p3, min_samples_split=p1, min_samples_leaf=p4)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        accuracyR, precisionR, recallR, TPR, FPR, counts = check_for_fairness(X_valid, y_pred, y_valid, X_valid[:,sensitive_param-1])
        EOD = np.abs(TPR[group_0] - TPR[group_1])
        ADO = 0.5 * (EOD + np.abs(FPR[group_0] - FPR[group_1]))
        FAR = float((FPR[group_0] * counts[group_0] + FPR[group_1] * counts[group_1])/(counts[group_0] + counts[group_1]))
        recall = float((recallR[group_0] * counts[group_0] + recallR[group_1] * counts[group_1])/(counts[group_0] + counts[group_1]))
        accuracy = float((accuracyR[group_0] * counts[group_0] + accuracyR[group_1] * counts[group_1])/(counts[group_0] + counts[group_1]))

        candi_four_goal = recall + FAR + ADO + EOD

        if n_obj == "ABCD":
            List_Y.append(candi_four_goal)
            List_acc.append(accuracy)
            List_AOD.append(ADO)
            List_EOD.append(EOD)
        elif n_obj == "AB":
            List_Y.append(two_goal_recall_far)
            List_acc.append(accuracy)
            List_AOD.append(ADO)
            List_EOD.append(EOD)
        elif n_obj == "CD":
            List_Y.append(two_goal_aod_eod)
            List_acc.append(accuracy)
            List_AOD.append(ADO)
            List_EOD.append(EOD)
        else:
            print("Wrong number of objects")

    remain_pool = all_case - set(modeling_pool)
    test_list = []
    for i in list(remain_pool):
        test_list.append(convert_lsr(i))

    upper_model = DecisionTreeRegressor()
    life = 20

    while len(List_X) < 200 and life > 0:
        upper_model.fit(List_X, List_Y)
        candidate = random.sample(test_list, 1)
        test_list.remove(candidate[0])
        candi_pred_value = upper_model.predict(candidate)
        if candi_pred_value < np.median(List_Y):
            List_X.append(candidate[0])
            candi_config = candidate[0]
            if(algorithm == "LogisticRegression"):
                p1 = temp[0]
                if temp[1] == 1:
                    p2 = 'l1'
                else:
                    p2 = 'l2'
                if temp[2] == 1:
                    p3 = 'liblinear'
                else:
                    p3 = 'saga'
                p4 = temp[3]
                model = LogisticRegression(C=p1, penalty=p2, solver=p3, max_iter=p4)
            elif(algorithm == "Decision_Tree_Classifier"):
                p1 = temp[0]
                if p1 == 1:
                    p1 = p1 + 1
                if temp[1] == 1:
                    p2 = 'gini'
                else:
                    p2 = 'entropy'
                if temp[2] == 1:
                    p3 = 'best'
                else:
                    p3 = 'random'
                p4 = temp[3]
                model = DecisionTreeClassifier(criterion=p2, splitter=p3, min_samples_split=p1, min_samples_leaf=p4)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            accuracyR, precisionR, recallR, TPR, FPR, counts = check_for_fairness(X_valid, y_pred, y_valid, X_valid[:,sensitive_param-1])
            EOD = np.abs(TPR[group_0] - TPR[group_1])
            ADO = 0.5 * (EOD + np.abs(FPR[group_0] - FPR[group_1]))
            FAR = float((FPR[group_0] * counts[group_0] + FPR[group_1] * counts[group_1])/(counts[group_0] + counts[group_1]))
            recall = float((recallR[group_0] * counts[group_0] + recallR[group_1] * counts[group_1])/(counts[group_0] + counts[group_1]))
            accuracy = float((accuracyR[group_0] * counts[group_0] + accuracyR[group_1] * counts[group_1])/(counts[group_0] + counts[group_1]))

            candi_four_goal = recall + FAR + ADO + EOD
            if n_obj == "ABCD":
                List_Y.append(candi_four_goal)
                List_acc.append(accuracy)
                List_AOD.append(ADO)
                List_EOD.append(EOD)
            elif n_obj == "AB":
                List_Y.append(candi_two_goal_recall_far)
                List_acc.append(accuracy)
                List_AOD.append(ADO)
                List_EOD.append(EOD)
            elif n_obj == "CD":
                List_Y.append(candi_two_goal_aod_eod)
                List_acc.append(accuracy)
                List_AOD.append(ADO)
                List_EOD.append(EOD)
        else:
            life -= 1

    min_index = int(np.argmin(List_Y))

    end_time = time.time()

    return List_X[min_index], List_Y[min_index], List_acc[min_index], List_AOD[min_index], List_EOD[min_index], end_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help='The name of dataset: census, credit, bank ')
    parser.add_argument("--algorithm", help='The name of algorithm: logistic regression, SVM, Random Forest')
    parser.add_argument("--sensitive_index", help='The index for sensitive feature')
    args = parser.parse_args()

    start_time = time.time()
    dataset = args.dataset
    algorithm = args.algorithm
    # algorithm = LogisticRegression, Decision_Tree_Classifier, TreeRegressor, Discriminant_Analysis

    data = {"census":census_data, "credit":credit_data, "bank":bank_data, "compas": compas_data}
    data_config = {"census":census, "credit":credit, "bank":bank, "compas": compas}

    # census (9 is for sex: 0 (men) vs 1 (female); 8 is for race: 0 (white) vs 4 (black))
    # credit (9 is for sex)
    # bank (1 is for age)
    # compas (1 is for sex: 0 (male) vs 1 (female); 2 is for age: 0 is under 25, 1 is between 25 and 45, and 2 is greater than 45); 2 is for race: Caucasian is 1 and non-Caucasian is 0.
    sensitive_param = int(args.sensitive_index)
    sensitive_name = ""

    group_0 = 0
    group_1 = 1
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

    X, Y, input_shape, nb_classes = data[dataset]()

    Y = np.argmax(Y, axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X, Y, random_state=0)
    n_obj = "ABCD"
    try:
        confing, four_goals, accuracy, AOD, EOD, end_time = flash_fair_LSR(X_train, y_train, X_val, y_val, sensitive_param, group_0, group_1, n_obj, start_time, algorithm)
        print("----------------------------")
        print("accuracy,AOD,EOD,time")
        print(str(accuracy) + "," + str(AOD) + "," + str(EOD) + "," + str(end_time - start_time))
        print(confing)
        print("----------------------------")
        # write to file
        if(algorithm == "LogisticRegression"):
            with open("./Dataset/" + algorithm + "_" + dataset + "_" + sensitive_name + "_" + "SMBO" + "_" + str(int(start_time)) + "_res.csv", 'w') as f:
                f.write("C,penalty,solver,max_iter,score,AOD,EOD,counter,timer\n")
                f.write(str(confing[0]) + "," + str(confing[1]) + "," + str(confing[2]) + "," + str(confing[3]) + "," + str(accuracy) + "," + str(AOD) + "," + str(EOD) + ",0," + str(end_time - start_time))
        if(algorithm == "Decision_Tree_Classifier"):
            with open("./Dataset/" + algorithm + "_" + dataset + "_" + sensitive_name + "_" + "SMBO" + "_" + str(int(start_time)) + "_res.csv", 'w') as f:
                f.write("min_samples_split,criterion,splitter,min_samples_leaf,score,AOD,EOD,counter,timer\n")
                f.write(str(confing[0]) + "," + str(confing[1]) + "," + str(confing[2]) + "," + str(confing[3]) + "," + str(accuracy) + "," + str(AOD) + "," + str(EOD) + ",0," + str(end_time - start_time))
    except TimeoutError as error:
        print("Caght an error!" + str(error))
        print("--- %s seconds ---" % (time.time() - start_time))
