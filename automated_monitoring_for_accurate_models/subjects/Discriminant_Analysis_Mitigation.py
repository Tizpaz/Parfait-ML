from scipy import linalg
import numpy as np
import xml_parser
from sklearn.datasets import make_classification

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from fairlearn.reductions import GridSearch, ExponentiatedGradient
from fairlearn.reductions import EqualizedOdds

# Generate datasets
# def dataset_fixed_cov(n, dim):
#     '''Generate 2 Gaussians samples with the same covariance matrix'''
#     # n, dim = 300, 2
#     np.random.seed(0)
#     # C = np.array([[0., -0.23], [0.83, .23]])
#     # C = np.random.rand()
#     X = np.r_[np.random.randn(n, dim),
#               np.random.randn(n, dim) + 1]
#     y = np.hstack((np.zeros(n), np.ones(n)))
#     return X, y
#
#
# def dataset_cov(n, dim):
#     '''Generate 2 Gaussians samples with different covariance matrices'''
#     # n, dim = 300, 2
#     np.random.seed(0)
#     # C = np.array([[0., -1.], [2.5, .7]]) * 2.
#     X = np.r_[np.random.randn(n, dim),
#               np.random.randn(n, dim) * 0.3]
#     y = np.hstack((np.zeros(n), np.ones(n)))
#     return X, y

def disc_analysis_mitigation(inp, X_train, X_test, y_train, y_test, sensitive_param = None):
    arr, features = xml_parser.xml_parser('Discriminant_Analysis_Mitigation_Params.xml',inp)
    # value for parameter_6
    if(arr[2]=="float"):
        # float does not work for bank dataset
        arr[2] = random.random()
    elif(arr[2]=="auto"):
        arr[2] = "auto"
    else:
        arr[2] = None
    # value for parameter_7
    # if(arr[7]!=None):
    #     val_7 = np.random.dirichlet(np.ones(arr[3]),size=1.0)
    # else:
    arr[3] = None
    # value for parameter_8
    if(arr[4]!='None'):
        arr[4] = np.random.randint(1,arr[3])
    else:
        arr[4] = None

    # Note in sklearn page
    if(arr[1]=='svd' and arr[2] != None):
        arr[2] = None

    if(arr[0]):
        try:
            # Linear Discriminant Analysis
            lda = ExponentiatedGradient(LinearDiscriminantAnalysis(solver=arr[1],
            shrinkage=arr[2],priors=arr[3],n_components=arr[4],
            store_covariance=arr[5], tol=arr[6]),
            constraints=EqualizedOdds(), eps = arr[9], max_iter = arr[10],
            eta0 = arr[11], run_linprog_step = arr[12])
            lda.fit(X_train, y_train, sensitive_features=X_train[:,sensitive_param-1])
            preds = lda.predict(X_test)
            score = np.sum(y_test == preds)/len(y_test)
            return True, lda, arr, score, preds, features
        except ValueError as VE:
            print(VE)
            return False, None, arr, None, None, features
        except TypeError as TE:
            print(TE)
            return False, None, arr, None, None, features
        except np.linalg.LinAlgError as err:
            print(err)
            return False, None, arr, None, None, features
    else:
        try:
            # Quadratic Discriminant Analysis
            qda = ExponentiatedGradient(QuadraticDiscriminantAnalysis(priors=arr[3],
                reg_param=arr[7],store_covariance=[8], tol=arr[6]),
                constraints=EqualizedOdds(), eps = arr[9], max_iter = arr[10],
                eta0 = arr[11], run_linprog_step = arr[12])
            qda.fit(X_train, y_train, sensitive_features=X_train[:,sensitive_param-1])
            preds = qda.predict(X_test)
            score = np.sum(y_test == preds)/len(y_test)
            return True, qda, arr, score, preds, features
            # print("here21")
        except ValueError as VE:
            print(VE)
            return False, None, arr, None, None, features
        except TypeError as TE:
            print(TE)
            return False, None, arr, None, None, features
