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

from Timeout import timeout

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help='The name of dataset: census, credit, bank ')
parser.add_argument("--algorithm", help='The name of algorithm: logistic regression, SVM, Random Forest')
parser.add_argument("--output", help='The name of output file', required=False)
parser.add_argument("--sensitive_index", help='The index for sensitive feature')
parser.add_argument("--time_out", help='Max. running time', default = 14400, required=False)
parser.add_argument("--max_iter", help='The maximum number of iterations', default = 100000, required=False)
args = parser.parse_args()

class Coverage(object):
    # Trace function
    def traceit(self, frame, event, arg):
        if self.original_trace_function is not None:
            self.original_trace_function(frame, event, arg)

        if event == "line":
            function_name = frame.f_code.co_name
            lineno = frame.f_lineno
            self._trace.append((function_name, lineno))

        return self.traceit

    def __init__(self):
        self._trace = []

    # Start of `with` block
    def __enter__(self):
        self.original_trace_function = sys.gettrace()
        sys.settrace(self.traceit)
        return self

    # End of `with` block
    def __exit__(self, exc_type, exc_value, tb):
        sys.settrace(self.original_trace_function)

    def trace(self):
        """The list of executed lines, as (function_name, line_number) pairs"""
        return self._trace

    def coverage(self):
        """The set of executed lines, as (function_name, line_number) pairs"""
        path_sign = 0
        for func_name, line in self.trace():
            path_sign ^= hash(func_name + str(line))
        return path_sign



class Runner(object):
    # Test outcomes
    PASS = "PASS"
    FAIL = "FAIL"
    UNRESOLVED = "UNRESOLVED"

    def __init__(self):
        """Initialize"""
        pass

    def run(self, inp):
        """Run the runner with the given input"""
        return (inp, Runner.UNRESOLVED)

class PrintRunner(Runner):
    def run(self, inp):
        """Print the given input"""
        print(inp)
        return (inp, Runner.UNRESOLVED)


class Fuzzer(object):
    def __init__(self):
        pass

    def fuzz(self):
        """Return fuzz input"""
        return ""

    # @timeout(5)
    def run(self, runner, inp, X_train, X_test, y_train, y_test):
        """Run `runner` with fuzz input"""
        return runner.run(inp, X_train, X_test, y_train, y_test)


    def runs(self, runner=PrintRunner(), trials=10):
        """Run `runner` with fuzz input, `trials` times"""
        # Note: the list comprehension below does not invoke self.run() for subclasses
        # return [self.run(runner) for i in range(trials)]
        outcomes = []
        for i in range(trials):
            outcomes.append(self.run(runner))
        return outcomes

class FunctionRunner(Runner):
    def __init__(self, function):
        """Initialize.  `function` is a function to be executed"""
        self.function = function

    def run_function(self, inp, X_train, X_test, y_train, y_test):
        res = self.function(inp, X_train, X_test, y_train, y_test)
        return res

    def run(self, inp, X_train, X_test, y_train, y_test):
        try:
            result = self.run_function(inp, X_train, X_test, y_train, y_test)
            outcome = self.PASS
        except Exception as exe:
            print(exe)
            result = None
            outcome = self.FAIL

        return result, outcome

class FunctionCoverageRunner(FunctionRunner):
    def run_function(self, inp, X_train, X_test, y_train, y_test):
        with Coverage() as cov:
            try:
                result = super(FunctionCoverageRunner,self).run_function(inp, X_train, X_test, y_train, y_test)
            except Exception as exc:
                self._coverage = cov.coverage()
                raise exc

        self._coverage = cov.coverage()
        return result

    def coverage(self):
        return self._coverage


def check_for_fairness(X, y_pred, y_true, a, X_new = None, Y_new = None):
    parities = []
    impacts = []
    eq_odds = []
    metric_frames = []
    metrics = {
        'false positive rate': false_positive_rate,
        'true positive rate': true_positive_rate
    }

    metric_frame = MetricFrame(metrics, y_true, y_pred, sensitive_features=a)
    return metric_frame.by_group["true positive rate"], metric_frame.by_group["false positive rate"]


@timeout(int(args.time_out))
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
        num_args = 18
    elif(program_name == "Discriminant_Analysis"):
        import Discriminant_Analysis
        input_program = Discriminant_Analysis.disc_analysis
        input_program_tree = 'Discriminant_Analysis_Params.xml'
        num_args = 9
    elif(program_name == "SVM"):
        import SVM
        input_program = SVM.SVM
        input_program_tree = 'SVM_Params.xml'
        num_args = 12
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

    arr_min, arr_max, arr_type, arr_default = xml_parser_domains.xml_parser_domains(input_program_tree, num_args)

    promising_inputs_fair1 = []
    promising_inputs_fair2 = []
    promising_inputs_AOD = []
    promising_metric_fair1 = []
    promising_metric_fair2 = []
    promising_metric_AOD = []
    observed_coverage = set()
    promising_inputs_coverage = []

    high_diff_1 = 0.0
    high_diff_2 = 0.0
    low_diff_1 = 1.0
    low_diff_2 = 1.0
    default_acc = 0.0
    failed = 0
    highest_acc = 0.0
    highest_acc_inp = None
    AOD_diff = 0.0

    if args.output == None:
        filename = "./Dataset/" + program_name + "_" +  dataset + "_" + sensitive_name + "_coverage_" + str(int(start_time)) + "_res.csv"
    elif args.output == "":
        filename = "./Dataset/" + program_name + "_" +  dataset + "_" + sensitive_name + "_coverage_" + str(int(start_time)) + "_res.csv"
    elif ".csv" in args.output:
        filename = "./Dataset/" + args.output
    else:
        filename = "./Dataset/" + args.output + ".csv"
    
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
                rnd = np.random.random()
                if (rnd < 0.05 and counter > 100) or (rnd < 0.9 and counter < 100):
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
                else:
                    rnd = np.random.random()
                    if rnd < 0.8:
                        if(rnd < 0.6):
                            inp = promising_inputs_AOD[-1]
                        else:
                            rng = range(len(promising_inputs_AOD))
                            sum_indicies = np.sum(rng)
                            index = np.random.choice(a = rng, p=[x/sum_indicies for x in rng])
                            inp = promising_inputs_AOD[index]
                    else:
                        if(rnd < 0.95):
                            inp = promising_inputs_coverage[-1]
                        else:
                            rng = range(len(promising_inputs_coverage))
                            sum_indicies = np.sum(rng)
                            index = np.random.choice(a = rng, p=[x/sum_indicies for x in rng])
                            inp = promising_inputs_coverage[index]
                    index = np.random.randint(0,len(arr_min)-1)
                    if(arr_type[index] == 'bool'):
                        inp[index] = 1 - inp[index]
                    elif(arr_type[index] == 'int'):
                        minVal = int(arr_min[index])
                        maxVal = int(arr_max[index])
                        rnd = np.random.random()
                        if rnd < 0.4:
                            newVal = np.random.randint(minVal,maxVal+1)
                            trail = 0
                            while newVal == inp[index] and trail < 3:
                                newVal = np.random.randint(minVal,maxVal+1)
                                trail += 1
                        elif rnd < 0.7:
                            newVal = inp[index] + 1
                        else:
                            newVal = inp[index] - 1
                        inp[index] = newVal
                    elif(arr_type[index] == 'float'):
                        minVal = float(arr_min[index])
                        maxVal = float(arr_max[index])
                        rnd = np.random.random()
                        if rnd < 0.5:
                            inp[index] = np.random.uniform(minVal,maxVal+0.000001)
                        elif rnd < 0.75:
                            newVal = inp[index] + abs(maxVal-minVal)/100
                        else:
                            newVal = inp[index] - abs(maxVal-minVal)/100

            print(inp)

            inp_program_instr = FunctionCoverageRunner(input_program)
            fuzz = Fuzzer()
            try:
                res, LR, inp_valid, score, preds, features = fuzz.run(inp_program_instr, inp, X_train, X_test, y_train, y_test)[0]
            except TypeError as TE:
                continue

            cov = inp_program_instr.coverage()

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
                promising_inputs_AOD.append(inp)
                promising_metric_AOD.append([AOD, score])
                AOD_diff = AOD

            if high_diff_1 < diff_1:
                promising_inputs_fair1.append(inp)
                promising_metric_fair1.append([diff_1, score])
                high_diff_1 = diff_1

            if high_diff_2 < diff_2:
                promising_inputs_fair2.append(inp)
                promising_metric_fair2.append([diff_2, score])
                high_diff_2 = diff_2

            if low_diff_1 > diff_1:
                low_diff_1 = diff_1

            if low_diff_2 > diff_2:
                low_diff_2 = diff_2

            if cov not in observed_coverage:
                observed_coverage.add(cov)
                promising_inputs_coverage.append(inp)

            if counter == 0:
                promising_inputs_fair1.append(inp)
                promising_inputs_fair2.append(inp)
                promising_inputs_AOD.append(inp)
                promising_inputs_coverage.append(inp)
                promising_metric_fair1.append([diff_1, score])
                promising_metric_fair2.append([diff_2, score])
                promising_metric_AOD.append([AOD, score])
                high_diff_1 = diff_1
                high_diff_2 = diff_2

            print("Highest AOD difference is " + str(AOD_diff))
            print("Highest EOD different is " + str(high_diff_1))
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

    start_time = time.time()
    dataset = args.dataset
    # algorithm = LogisticRegression, Decision_Tree_Classifier, TreeRegressor, Discriminant_Analysis
    algorithm = args.algorithm
    num_iteration =  int(args.max_iter)

    data = {"census":census_data, "credit":credit_data, "bank":bank_data, "compas": compas_data}
    data_config = {"census":census, "credit":credit, "bank":bank, "compas": compas}

    # census (9 is for sex: 0 (men) vs 1 (female); 8 is for race: 0 (white) vs 4 (black))
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
