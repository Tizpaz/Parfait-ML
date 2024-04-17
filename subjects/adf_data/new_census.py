# adapted from ADF https://github.com/pxzhang94/adf
import numpy as np
import sys, os
sys.path.append('../')
#sys.path.append(os.getcwd()+"../")
print(sys.path)

def new_census_data(dataset):
    """
    Prepare the data of dataset Survey Census Income
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    file_root = "subjects/datasets/" + dataset
    with open(file_root, "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            if (i == 0):
                i += 1
                continue
            # L = map(int, line1[:-1])
            L = [int(i) for i in line1[:-1]]
            X.append(L)
            if int(line1[-1]) == 0:
                Y.append([1, 0])
            else:
                Y.append([0, 1])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, 10)
    nb_classes = 2

    return X, Y, input_shape, nb_classes
