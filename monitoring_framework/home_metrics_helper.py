from sklearn.metrics import accuracy_score
from fairlearn.metrics import true_positive_rate, false_positive_rate
import numpy as np
from Themis.Themis2.themis2 import Themis
from io import BytesIO

def model_accuracy(model, encoded_X, encoded_Y):
    return accuracy_score(encoded_Y, model.predict(encoded_X))








def AOD_score(model, encoded_X_group0, encoded_Y_group0, encoded_X_group1, encoded_Y_group1):
    group0_TPR = true_positive_rate(model.predict(encoded_X_group0), encoded_Y_group0)
    group1_TPR = true_positive_rate(model.predict(encoded_X_group1), encoded_Y_group1)

    group0_FPR = false_positive_rate(model.predict(encoded_X_group0), encoded_Y_group0)
    group1_FPR = false_positive_rate(model.predict(encoded_X_group1), encoded_Y_group1)

    return (np.abs(group1_FPR - group0_FPR) + np.abs(group1_TPR - group0_TPR))/2



def EOD_score(model, encoded_X_group0, encoded_Y_group0, encoded_X_group1, encoded_Y_group1):
    group0_TPR = true_positive_rate(model.predict(encoded_X_group0), encoded_Y_group0)
    group1_TPR = true_positive_rate(model.predict(encoded_X_group1), encoded_Y_group1)

    group0_FPR = false_positive_rate(model.predict(encoded_X_group0), encoded_Y_group0)
    group1_FPR = false_positive_rate(model.predict(encoded_X_group1), encoded_Y_group1)
    return np.maximum(np.abs(group1_FPR - group0_FPR), np.abs(group1_TPR - group0_TPR))

def num_counterfactuals(modle, encoded_X, encoded_y):
    pass # Not implemented for ACM SIGKDD because we do not discuss cunterfactuals.
