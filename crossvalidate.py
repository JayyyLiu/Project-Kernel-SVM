"""
INPUT:	
xTr : dxn input vectors
yTr : 1xn input labels
ktype : (linear, rbf, polynomial)
Cs   : interval of regularization constant that should be tried out
paras: interval of kernel parameters that should be tried out

Output:
bestC: best performing constant C
bestP: best performing kernel parameter
lowest_error: best performing validation error
errors: a matrix where allvalerrs(i,j) is the validation error with parameters Cs(i) and paras(j)

Trains an SVM classifier for all combination of Cs and paras passed and identifies the best setting.
This can be implemented in many ways and will not be tested by the autograder. You should use this
to choose good parameters for the autograder performance test on test data. 
"""

import numpy as np
import math
from trainsvm import trainsvm

def crossvalidate(xTr, yTr, ktype, Cs, paras):
    bestC, bestP, lowest_error = 0, 0, 0
    errors = np.zeros((len(paras),len(Cs)))
    
    # YOUR CODE HERE
    k_fold = 5
    np.random.seed(42)

    index = np.arange(xTr.shape[1])
    np.random.shuffle(index)

    for i in range(len(Cs)):
        for j in range(len(paras)):
            err = 0
            for k in range(k_fold):
                split_index = np.split(index,k_fold)
                test = split_index.pop(k)
                train = np.concatenate(split_index)

                train_X = xTr[:,train]
                train_y = yTr[train]
                test_X = xTr[:,test]
                test_y = yTr[test]

                svm_class = trainsvm(train_X,train_y,Cs[i],ktype,paras[j])
                y_pred = svm_class(test_X)
                err += np.mean(test_y != y_pred)
            errors[i,j] = err / k_fold

    lowest_error = np.min(errors)
    #index_best = np.where(errors == lowest_error)
    index_best = np.argwhere(errors == lowest_error)
    #print(index_best)
    bestC = Cs[index_best[0][0]]
    bestP = paras[index_best[0][1]]
    
    return bestC, bestP, lowest_error, errors


    