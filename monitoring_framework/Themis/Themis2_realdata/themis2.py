# Themis 2.0
#
# By: Rico Angell

from __future__ import division

import argparse
import subprocess
from itertools import chain, combinations, product
import math
import random
import scipy.stats as st
import xml.etree.ElementTree as ET
import copy


class Input:
    """
    Class to define an input characteristic to the software.

    Attributes
    ----------
    name : str
        Name of the input.
    values : list
        List of the possible values for this input.

    Methods
    -------
    get_random_input()
        Returns a random element from the values list.
    """
    def __init__(self, name="", values=[], kind="", ub=None, lb=None):
        self.name = name
        self.values = [str(v) for v in values]
        self.kind = kind
        if (ub != None and lb != None):
            self.lb = lb
            self.ub = ub


    def get_random_input(self):
        """
        Return a random value from self.values
        """

        return random.choice(self.values)


    def __str__(self):

        s = "\nInput\n"
        s += "-----\n"
        s += "Name: " + self.name + "\n"
        s += "Values: " + ", ".join(self.values)
        return s
    __repr__ = __str__


class Test:
    """
    Data structure for storing tests.

    Attributes
    ----------
    function : str
        Name of the function to call
    i_fields : list of `Input.name`
        The inputs of interest, i.e. compute the casual discrimination wrt
        these fields.
    threshold : float in [0,1]
        At least this level of discrimination to be considered.
    conf : float in [0, 1]
        The z* confidence level (percentage of normal distribution.
    margin : float in [0, 1]
        The margin of error for the confidence.
    group : bool
        Search for group discrimination if `True`.
    causal : bool
        Search for causal discrimination if `True`.
    """
    def __init__(self, function="", i_fields=[], conf=0.999, margin=0.0001,
                    group=True, causal=True, threshold=0.15):
        self.function = function
        self.i_fields = i_fields
        self.conf = conf
        self.margin = margin
        self.group = group
        self.causal = causal
        self.threshold = threshold

    def __str__(self):
        s = "\n\n"

        # Alters output based on the test that was ran
        if self.function == "discrimination_search":
            print( "Ran discrimination search for: \n")
            if (self.group == True):
                s += "Group Discrimination\n"
            if (self.causal == True):
                s += "Causal Discrimination\n\n"
        elif self.function == "causal_discrimination":
            s += "Calculated causal discrimination for the following sets of inputs: \n (" + ", ".join(self.i_fields) + ") \n\n"
        elif self.function == "group_discrimination":
            s += "Calculated group discrimination for the following sets of inputs: \n (" + ", ".join(self.i_fields) + ") \n\n"

        return s
    __repr__ = __str__


class Themis:
    """
    Compute discrimination for a piece of software.

    Attributes
    ----------

    Methods
    -------
    """
    def __init__(self, S: map, tests: list, xml_fname=""):
        """
        Initialize Themis from xml file.

        Parameters
        ----------
        xml_fname : string
            name of the xml file we want to import settings from.
        """

        if xml_fname != "":
            tree = ET.parse(xml_fname, parser = ET.XMLParser(encoding = "utf-8"))
            root = tree.getroot()

            self.max_samples = int(root.find("max_samples").text)
            self.min_samples = int(root.find("min_samples").text)
            self.rand_seed = int(root.find("seed").text)
            self.software_name = root.find("name").text
            self.command = S
            self._build_input_space(args=root.find("inputs"))
            self._load_tests(tests)
            self._cache = {}
                
        else:
            self._cache = {}
            self.tests = []
            
    def run(self):
        """
        Run Themis tests specified in the configuration file.
        """
##        try:
 

        #key = inputs tuple
        # value = percentage from test execution
        self.group_tests = {}
        self.causal_tests = {}
        self.group_search_results = {}
        self.causal_search_results = {}



        self.raw_test_results = []

        self.group_measurement_results = []
        self.causal_measurement_results = []

        self.simple_discrim_output = ""
        self.detailed_discrim_output = ""

        
        for test in self.tests:
            random.seed(self.rand_seed)
            #print ("--------------------------------------------------")
            if test.function == "causal_discrimination":
                suite, p, self.causal_pairs = self.causal_discrimination(i_fields=test.i_fields,
                                                      conf=test.conf,
                                                      margin=test.margin)
                # store tests for output strings
                causal_key = tuple(test.i_fields)
                self.causal_tests [causal_key] = "{:.1%}".format(p)
                
##                    self.output += str(test)
##                    op = 'Your software discriminates on the above inputs ' + "{:.1%}".format(p) +  ' of the time.'
##                    self.output += op
                
            elif test.function == "group_discrimination":
                suite, p, _ = self.group_discrimination(i_fields=test.i_fields,
                                                     conf=test.conf,
                                                     margin=test.margin)

                # store tests for output strings
                group_key = tuple(test.i_fields)
                self.group_tests [group_key] = "{:.1%}".format(p)

                #save min_group and max_group 
                                   
            elif test.function == "discrimination_search":
                print ("running discrim search")
                print (test.conf)
                print (test.margin)
                print(test.group)
                print(test.causal)
                print(test.threshold)
                
                g, c = self.discrimination_search(threshold=test.threshold,
                                                  conf=test.conf,
                                                  margin=test.margin,
                                                  group=test.group,
                                                  causal=test.causal)

                if g:

                    for key, value in g.items():
                        values = ", ".join(key) + " --> " + "{:.1%}".format(value) + "\n"
                        
                        self.group_search_results[tuple(key)] = "{:.1%}".format(value)
                                                
                if c:
                    for key, value in c.items():
                        values = ", ".join(key) + " --> " + "{:.1%}".format(value) + "\n"

                        self.causal_search_results[tuple(key)] = "{:.1%}".format(value)

        

##        
##        print ("Group Discrimination Tests: \n")
##        for key,value in self.group_tests.items():
##            print ('Input(s): ' + str(key) + '--->' + str(value) + "\n")
##
##        print ("Causal Discrimination Tests: \n")
##        for key, value in self.causal_tests.items():
##            print ('Input(s): ' + str(key) + '--->' + str(value) + "\n")


        
        
            
        
        self.short_output = ""
        self.extended_output = ""

        return self.raw_test_results
            
##        except:
##            print ("Issue in main Themis run")        

    def group_discrimination(self, i_fields=None, conf=0.999, margin=0.0001):
        """
        Compute the group discrimination for characteristics `i_fields`.

        Parameters
        ----------
        i_fields : list of `Input.name`
            The inputs of interest, i.e. compute the casual discrimination wrt
            these fields.
        conf : float in [0, 1]
            The z* confidence level (percentage of normal distribution.
        margin : float in [0, 1]
            The margin of error for the confidence.

        Returns
        -------
        tuple
            * list of dict
                The test suite used to compute group discrimination.
            * float
                The percentage of group discrimination
        """
        assert i_fields != None
##        try:
        min_group_score, max_group_score, test_suite, p = float("inf"), 0, [], 0
        min_group_assign, max_group_assign = "",""
        rand_fields = self._all_other_fields(i_fields)
        for fixed_sub_assign in self._gen_all_sub_inputs(args=i_fields):
            count = 0
            for num_sampled in range(1, self.max_samples):
                assign = self._new_random_sub_input(args=rand_fields)
                assign.update(fixed_sub_assign)
                self._add_assignment(test_suite, assign)
                count += self._get_test_result(assign=assign)

                p, end = self._end_condition(count, num_sampled, conf, margin)
                if end:
                    break
            print (fixed_sub_assign[i_fields[0]] + "--> " + str(p))
            if p < min_group_score:
                min_group_score = p
                min_group_assign = fixed_sub_assign[i_fields[0]]
            if p > max_group_score:
                max_group_score = p
                max_group_assign = fixed_sub_assign[i_fields[0]]
        output = test_suite, (max_group_score - min_group_score), (min_group_score,min_group_assign,max_group_score,max_group_assign)
        self.raw_test_results.append(("group", output))
        return output
##        except:
##            print("Issue in group_discrimination")

    def causal_discrimination(self, i_fields=None, conf=0.999, margin=0.0001):
        """
        Compute the causal discrimination for characteristics `i_fields`.

        Parameters
        ----------
        i_fields : list of `Input.name`
            The inputs of interest, i.e. compute the casual discrimination wrt
            these fields.
        conf : float in [0, 1]
            The z* confidence level (percentage of normal distribution.
        margin : float in [0, 1]
            The margin of error for the confidence.

        Returns
        -------
        tuple
            * list of dict
                The test suite used to compute causal discrimination.
            * float
                The percentage of causal discrimination.
        """
##        try:
        assert i_fields != None
        count, test_suite, p = 0, [], 0
        f_fields = self._all_other_fields(i_fields) # fixed fields
        causal_pairs = []
        
        for num_sampled in range(1, self.max_samples):
            fixed_assign = self._new_random_sub_input(args=f_fields)
            singular_assign = self._new_random_sub_input(args=i_fields)
            assign = self._merge_assignments(fixed_assign, singular_assign)
            causal_assign1 = copy.deepcopy(assign)
            self._add_assignment(test_suite, assign)
            result = self._get_test_result(assign=assign)
            for dyn_sub_assign in self._gen_all_sub_inputs(args=i_fields):
                if dyn_sub_assign == singular_assign:
                    continue
                assign.update(dyn_sub_assign)
                self._add_assignment(test_suite, assign)
                if self._get_test_result(assign=assign) != result:
                    count += 1
                    causal_pairs.append((causal_assign1,copy.deepcopy(assign)))
                    break

            p, end = self._end_condition(count, num_sampled, conf, margin)
            if end:
                break
        output = test_suite, p, causal_pairs
        # print(output)
        self.raw_test_results.append(("causal", output))
        return output
##        except:
##            print("Issue in causal discrimination")

    def discrimination_search(self, threshold=0.2, conf=0.99, margin=0.01,
                              group=False, causal=False):
        """
        Find all minimall subsets of characteristics that discriminate.

        Choose to search by group or causally and set a threshold for
        discrimination.

        Parameters
        ----------
        threshold : float in [0,1]
            At least this level of discrimination to be considered.
        conf : float in [0, 1]
            The z* confidence level (percentage of normal distribution.
        margin : float in [0, 1]
            The margin of error for the confidence.
        group : bool
            Search for group discrimination if `True`.
        causal : bool
            Search for causal discrimination if `True`.

        Returns
        -------
        tuple of dict
            The lists of subsets of the input characteristics that discriminate.
        """
##        try:
        assert group or causal
        group_d_scores, causal_d_scores = {}, {}
        for sub in self._all_relevant_subs(self.input_order):
            if self._supset(list(set(group_d_scores.keys())|
                                 set(causal_d_scores.keys())), sub):
                continue
            if group:

                suite, p, data = self.group_discrimination(i_fields=sub, conf=conf,
                                                   margin=margin)
                
                if p > threshold:
                    group_d_scores[sub] = p
                    self.group_measurement_results.append(MeasurementResult(causal=False, i_fields=sub, p=p, testsuite=suite, data=data)) 
            if causal:
                suite, p, cp = self.causal_discrimination(i_fields=sub, conf=conf,
                                                   margin=margin)
                print (sub)
                print(conf)
                print(margin)
                print (p)
                print (cp)
                if p > threshold:
                    causal_d_scores[sub] = p
                    self.causal_measurement_results.append(MeasurementResult(causal=True, i_fields=sub, p=p, testsuite=suite, data=cp))

        return group_d_scores, causal_d_scores
##        except:
##            print("Issue in trying to search for discrimination")

    def _all_relevant_subs(self, xs):
        return chain.from_iterable(combinations(xs, n) \
                                    for n in range(1, len(xs)))

    def _supset(self, list_of_small, big):
        for small in list_of_small:
            next_subset = False
            for x in small:
                if x not in big:
                    next_subset = True
                    break
            if not next_subset:
                return True

    def _new_random_sub_input(self, args=[]):
        assert args
        return {name : self.inputs[name].get_random_input() for name in args}

    def _gen_all_sub_inputs(self, args=[]):
        assert args
        vals_of_args = [self.inputs[arg].values for arg in args]
        combos = [list(elt) for elt in list(product(*vals_of_args))]
        return ({arg : elt[idx] for idx, arg in enumerate(args)} \
                                                for elt in combos)


    # This is the "interface" with the black box algorithm
    def _get_test_result(self, assign=None):
        assert assign != None
##        try:
        tupled_args = self._tuple(assign)
        if tupled_args in self._cache.keys():
            return self._cache[tupled_args]
        self._cache[tupled_args] = self.command(tupled_args)
        return self._cache[tupled_args]
##        except:
##            print("Issue in getting the results of the tests")

    def _add_assignment(self, test_suite, assign):
        if assign not in test_suite:
            test_suite.append(assign)

    def _all_other_fields(self, i_fields):
        return [f for f in self.input_order if f not in i_fields]


    def _end_condition(self, count, num_sampled, conf, margin):
        p = 0
        if num_sampled > self.min_samples:
            p = count / num_sampled
            error = st.norm.ppf(conf)*math.sqrt((p*(1-p))/num_sampled)
            return p, error < margin
        return p, False

    def _merge_assignments(self, assign1, assign2):
        merged = {}
        merged.update(assign1)
        merged.update(assign2)
        return merged

    def _tuple(self, assign=None):
        assert assign != None
        return tuple(str(assign[name]) for name in self.input_order)

    def _untuple(self, tupled_args=None):
        assert tupled_args != None
        listed_args = list(tupled_args)
        return {name : listed_args[idx] \
                    for idx, name in enumerate(self.input_order)}

    def _build_input_space(self, args=None):
        assert args != None
        self.inputs = {}
        self.input_order = []
        self.input_names = []
        for obj in args.findall("input"):
            name = obj.find("name").text
            self.input_names.append(name)
            values = []
            t = obj.find("type").text
            if t == "categorical":
                values = [elt.text.replace("&lt;", "<").replace("&amp;", "<")
                            for elt in obj.find("values").findall("value")]

                self.inputs[name] = Input(name=name, values=values, kind="categorical")
            elif t == "continuousInt":
                lowerbound = int(obj.find("bounds").find("lowerbound").text)
                upperbound = int(obj.find("bounds").find("upperbound").text)+1
                
                values = range(int(obj.find("bounds").find("lowerbound").text),
                                int(obj.find("bounds").find("upperbound").text)+1)

                

                self.inputs[name] = Input(name=name, values=values, kind="continuousInt", lb = str(lowerbound), ub = str(upperbound))
            else:
                assert False

            self.input_order.append(name)

    def _add_input(self, name=None, kind=None, values=None):
        assert name != None
        assert kind != None
        assert values != None

        try:
            self.inputs
        except AttributeError:
            self.inputs = {}
            self.input_order = []
            self.input_names = []

        local_values = []
                    
        self.input_names.append(name)
        
        if kind == "Categorical":
            i_values = values.split(',')
            self.inputs[name] = Input(name=name, values=i_values, kind="categorical")
        elif kind == "Continuous Int":
            ulb = values.split('-')
            lowerbound = ulb[0]
            upperbound = ulb[1]

            local_values = range(int(lowerbound),int(upperbound)+1)

            self.inputs[name] = Input(name=name, values=local_values, kind="continuousInt", lb = str(lowerbound), ub = str(upperbound))
        else:
            assert False

        self.input_order.append(name)
            

    def _load_tests(self, tests=None):
        assert tests != None
        self.tests = []
        for t in tests:
            test = Test()
            test.function = t["function"]
            if test.function == "causal_discrimination" or \
                test.function == "group_discrimination":
                test.i_fields = t["input_name"]
            if test.function == "discrimination_search":
                test.group = t["group"]
                test.causal = t["causal"]
                test.threshold = t["threshold"]
                
            test.conf = t["conf"]
            test.margin = t["margin"]
            self.tests.append(test)
            
    def _new_test(self, group, causal, name=None, conf=None, margin=None, i_fields=None, threshold=0.20):
        assert name != None
        assert conf != None
        assert margin != None
        assert i_fields != None

##        try:
##            self.tests
##        except AttributeError:
##            self.tests = []

        test = Test()
        test.function = name
        test.conf = conf
        test.margin = margin
        test.group = group
        test.causal = causal
        test.i_fields = i_fields
        test.threshold = threshold

        self.tests.append(test)

class MeasurementResult(object):

    def __init__(self, causal=True, i_fields=None, p=None, testsuite=None, data=None):
        self.causal = causal
        self.i_fields = i_fields
        self.p = p
        self.testsuite = testsuite
        self.data = data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Themis.")
    parser.add_argument("XML_FILE", type=str, nargs=1,
                            help="XML configuration file")
    import loan
    # tests = [{"function": "discrimination_search", "threshold": 0.2, "conf": 0.98, "margin": 0.02, "group": True, "causal": False}]
    tests = [{"function": "causal_discrimination", "threshold": 0.2, "conf": 0.98, "margin": 0.02, "input_name": ["sex"]}]
    t = Themis(loan.loan,tests, xml_fname="settings.xml")
    print(t.run())
