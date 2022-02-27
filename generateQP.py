"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
C : regularization constant

Output:
Q,p,G,h,A,b as defined in qpsolvers.solve_qp

A call of qpsolvers.solve_qp(Q, p, G, h, A, b) should return the optimal nx1 vector of alphas
of the SVM specified by K, yTr, C. Just make these variables np arrays.

"""
import numpy as np

def generateQP(K, yTr, C):
    yTr = yTr.astype(np.double)
    n = yTr.shape[0]
    
    # YOUR CODE HERE
    Q = (K*(np.dot(yTr,yTr.T))).astype(np.double)
    p = -np.ones((n,1),dtype=np.double)
    G = np.vstack((np.eye(n,dtype=np.double),-np.eye(n,dtype=np.double)))
    h = np.vstack((C*np.ones((n,1),dtype=np.double),np.zeros((n,1),dtype=np.double)))
    A = yTr.T
    b = np.array([[0]],dtype=np.double)

            
    return Q, p, G, h, A, b

