import sys
sys.path.append("./")
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
if sys.version_info.major==2:
    from Queue import PriorityQueue
else:
    from queue import PriorityQueue
import os
import time
import copy
from scipy.stats import randint
import csv
import argparse

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
    print("demographic_parity_difference is " + str(parity))
    print("demographic_parity_ratio is " + str(di))
    print("equalized_odds_difference is " + str(eod))
    print("------------------------------------------")
    return metric_frame.by_group["true positive rate"], metric_frame.by_group["false positive rate"]

@timeout(14400)
def test_cases(dataset, program_name, max_iter, X_train, X_test, y_train, y_test, sensitive_param, group_0, group_1, sensitive_name, start_time):
    num_args = 0
    if(program_name == "LogisticRegression"):
        import LogisticRegression
        input_program = LogisticRegression.logistic_regression
        input_program_tree = 'logistic_regression_Params.xml'
        num_args = 15
    elif(program_name == "Decision_Tree_Classifier"):
        import Decision_Tree_Classifier
        input_program = Decision_Tree_Classifier.DecisionTree
        input_program_tree = 'Decision_Tree_Classifier_Params.xml'
        num_args = 13
    elif(program_name == "TreeRegressor"):
        import TreeRegressor
        input_program = TreeRegressor.TreeRegress
        input_program_tree = 'TreeRegressor_Params.xml'
        num_args = 14
    elif(program_name == "Discriminant_Analysis"):
        import Discriminant_Analysis
        input_program = Discriminant_Analysis.disc_analysis
        input_program_tree = 'Discriminant_Analysis_Params.xml'
        num_args = 9
    elif(program_name == "SVM"):
        import SVM
        input_program = SVM.SVM
        input_program_tree = 'SVM_Params.xml'
        num_args = 9
    elif(program_name == "LogisticRegressionMitigation"):
        import LogisticRegressionMitigation
        input_program = LogisticRegressionMitigation.logistic_regression_mitigation
        input_program_tree = 'logistic_regression_mitigation_Params.xml'
        num_args = 14
    elif(program_name == "Decision_Tree_Classifier_Mitigation"):
        import Decision_Tree_Classifier_Mitigation
        input_program = Decision_Tree_Classifier_Mitigation.DecisionTreeMitigation
        input_program_tree = 'Decision_Tree_Classifier_Mitigation_Params.xml'
        num_args = 17
    elif(program_name == "TreeRegressorMitigation"):
        import TreeRegressorMitigation
        input_program = TreeRegressorMitigation.TreeRegressMitigation
        input_program_tree = 'TreeRegressorMitigation_Params.xml'
        num_args = 18
    elif(program_name == "Discriminant_Analysis_Mitigation"):
        import Discriminant_Analysis_Mitigation
        input_program = Discriminant_Analysis_Mitigation.disc_analysis_mitigation
        input_program_tree = 'Discriminant_Analysis_Mitigation_Params.xml'
        num_args = 13
    elif(program_name == "SVM_Mitigation"):
        import SVM_Mitigation
        input_program = SVM_Mitigation.SVM_Mitigation
        input_program_tree = 'SVM_Mitigation_Params.xml'
        num_args = 13

    arr_min, arr_max, arr_type, arr_default = xml_parser_domains.xml_parser_domains(input_program_tree, num_args)

    promising_inputs_fair1 = []
    promising_inputs_fair2 = []
    promising_metric_fair1 = []
    promising_metric_fair2 = []

    high_diff_1 = 0.0
    high_diff_2 = 0.0
    low_diff_1 = 1.0
    low_diff_2 = 1.0
    default_acc = 0.0
    failed = 0
    highest_acc = 0.0
    highest_acc_inp = None
    AOD_diff = 0.0

    filename = program_name + "_" +  dataset + "_" + sensitive_name + "_random_" + str(int(start_time)) + "_res.csv"

    with open(filename, 'w') as f:
        for counter in range(max_iter):
            inp = []
            # include default value
            if counter == 0:
                for i in range(len(arr_min)):
                    if(arr_type[i] == 'bool'):
                        inp.append(int(arr_default[i]))
                    elif(arr_type[i] == 'int'):
                        inp.append(int(arr_default[i]))
                    elif(arr_type[i] == 'float'):
                        inp.append(float(arr_default[i]))
            else:
                for i in range(len(arr_min)):
                    if(arr_type[i] == 'bool'):
                        inp.append(randint.rvs(0,2))
                    elif(arr_type[i] == 'int'):
                        minVal = int(arr_min[i])
                        maxVal = int(arr_max[i])
                        inp.append(np.random.randint(minVal,maxVal+1))
                    elif(arr_type[i] == 'float'):
                        minVal = float(arr_min[i])
                        maxVal = float(arr_max[i])
                        inp.append(np.random.uniform(minVal,maxVal+0.00001))

            print(inp)

            res, LR, inp_valid, score, preds, features = input_program(inp, X_train, X_test, y_train, y_test, sensitive_param)

            if not res:
                failed += 1
                continue

            if counter == 0:
                features.append("score")
                features.append("AOD")
                features.append("TPR")
                features.append("FPR")
                features.append("counter")
                features.append("timer")
                for i in range(len(features)):
                    if i < len(features) - 1:
                        if features[i] == None:
                            f.write(",")
                        else:
                            f.write("%s," % features[i])
                    else:
                        f.write("%s" % features[i])
                f.write("\n")
                default_acc = score

            if (score < (default_acc - 0.01)):
                continue

            if(score > highest_acc):
                highest_acc = score
                highest_acc_inp = inp_valid

            fair_metric_1, fair_metric_2 = check_for_fairness(X_test, preds, y_test, X_test[:,sensitive_param-1])

            diff_1 = np.abs(fair_metric_1[group_0] - fair_metric_1[group_1])
            diff_2 = np.abs(fair_metric_2[group_0] - fair_metric_2[group_1])

            AOD = (diff_1 + diff_2) * 0.5

            full_inp = inp_valid.copy()
            full_inp.append(score)
            full_inp.append(AOD)
            full_inp.append(diff_1)
            full_inp.append(diff_2)
            full_inp.append(counter)
            full_inp.append(time.time() - start_time)

            for i in range(len(full_inp)):
                if i < len(full_inp) - 1:
                    if full_inp[i] == None:
                        f.write(",")
                    else:
                        f.write("%s," % full_inp[i])
                else:
                    f.write("%s" % full_inp[i])
            f.write("\n")

            if AOD_diff < AOD:
                AOD_diff = AOD

            if high_diff_1 < diff_1:
                promising_inputs_fair1.append(inp_valid)
                promising_metric_fair1.append([diff_1, score])
                high_diff_1 = diff_1

            if high_diff_2 < diff_2:
                promising_inputs_fair2.append(inp_valid)
                promising_metric_fair2.append([diff_2, score])
                high_diff_2 = diff_2

            if low_diff_1 > diff_1:
                low_diff_1 = diff_1

            if low_diff_2 > diff_2:
                low_diff_2 = diff_2

            if counter == 0:
                promising_inputs_fair1.append(inp_valid)
                promising_inputs_fair2.append(inp_valid)
                promising_metric_fair1.append([diff_1, score])
                promising_metric_fair2.append([diff_2, score])
                high_diff_1 = diff_1
                high_diff_2 = diff_2

            print("Metric 1 different is " + str(diff_1))
            print("Metric 2 different is " + str(diff_2))
            print("score is " + str(score))
            print("counter: " + str(counter))
            print("---------------------------------------------------------")

    print("------------------END-----------------------------------")
    print(promising_inputs_fair1[-1])
    print(promising_inputs_fair1[0])
    print(promising_inputs_fair2[-1])
    print(promising_inputs_fair2[0])
    print(promising_metric_fair1[-1])
    print(promising_metric_fair1[0])
    print(promising_metric_fair2[-1])
    print(promising_metric_fair2[0])
    print("Highest AOD differences " + str(AOD_diff))
    print("Lowest fairness (1) differences " + str(low_diff_1))
    print("Lowest fairness (2) differences " + str(low_diff_2))
    print("Failed Test cases: " + str(failed))
    print("Highest accuracy observed: " + str(highest_acc))
    print("Highest accuracy input: " + str(highest_acc_inp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help='The name of dataset: census, credit, bank ')
    parser.add_argument("--algorithm", help='The name of algorithm: logistic regression, SVM, Random Forest')
    parser.add_argument("--sensitive_index", help='The index for sensitive feature')
    parser.add_argument("--max_iter", help='The maximum number of iterations')
    args = parser.parse_args()

    start_time = time.time()
    dataset = args.dataset
    # algorithm = LogisticRegression, Decision_Tree_Classifier, TreeRegressor, Discriminant_Analysis
    algorithm = args.algorithm
    num_iteration =  int(args.max_iter)

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

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

    try:
        test_cases(dataset, algorithm, num_iteration, X_train, X_test, y_train, y_test, sensitive_param, group_0, group_1, sensitive_name, start_time)
    except TimeoutError as error:
        print("Caght an error!" + str(error))
        print("--- %s seconds ---" % (time.time() - start_time))

    print("--- %s seconds ---" % (time.time() - start_time))
