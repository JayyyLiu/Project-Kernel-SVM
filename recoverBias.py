"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
alphas  : nx1 vector or alpha values
C : regularization constant

Output:
bias : the scalar hyperplane bias of the kernel SVM specified by alphas

Solves for the hyperplane bias term, which is uniquely specified by the support vectors with alpha values
0<alpha<C
"""

import numpy as np

def recoverBias(K,yTr,alphas,C):
    bias = 0
    
    # YOUR CODE HERE
    sv_index = np.argmin(abs(alphas-C/2))
    bias = 1/yTr[sv_index] - np.dot((alphas*yTr).T,K[:,sv_index])

    
    return bias 
    
