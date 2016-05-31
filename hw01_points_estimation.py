import numpy as np
from numpy import matlib 
from numpy.linalg import *

def PolyVec(x, n):
    'poduce poly vec'
    outvec=[1,]
    for i in range(1, n):
        outvec.append(outvec[i-1]*x)
    return outvec

def PointsEstimate(x, y, n, regularization=0.):
    'Points Estimate using Polynomial'
    assert len(x)==len(y)
    m=len(x)
    #algorithm data
    #allocate memory A is m*n
    A=matlib.zeros((m, n))
    b=matlib.mat(y).T
    for i, xi in enumerate(x):
        A[i]=PolyVec(xi, n)
    #least sqare sum
    ATA=A.T*A
    b=A.T*b
    #add regularization
    for i in range(n):
        ATA[i, i]+=regularization
    return solve(ATA, b)