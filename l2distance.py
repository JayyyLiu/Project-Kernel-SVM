import numpy as np

"""
function D=l2distance(X,Z)
	
Computes the Euclidean distance matrix. 
Syntax:
D=l2distance(X,Z)
Input:
X: dxn data matrix with n vectors (columns) of dimensionality d
Z: dxm data matrix with m vectors (columns) of dimensionality d

Output:
Matrix D of size nxm 
D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
"""

def l2distance(X,Z):
    d, n = X.shape
    dd, m = Z.shape
    assert d == dd, 'First dimension of X and Z must be equal in input to l2distance'
    
    D = np.zeros((n, m))
    
    # YOUR CODE HERE
    G = np.dot(X.T,Z)
    S = np.dot(np.diag(np.diag(np.dot(X.T,X))),np.ones((n,m)))
    R = (np.dot(np.diag(np.diag(np.dot(Z.T,Z))),np.ones((m,n)))).T
    D = np.sqrt(S - 2*G + R)
    
    return D
