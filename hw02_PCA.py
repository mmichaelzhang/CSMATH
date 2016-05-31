import numpy as np
from numpy import linalg
from numpy import matlib

def PCA(data_mat, p):
    'reduce the data dimensionality to p, this function will substract the \
    mean from the original data'
    d, N=data_mat.shape
    m=matlib.mean(data_mat, 1)
    data_mat-=m
    if d<N:
        AAT=data_mat*data_mat.T
        w, v=linalg.eigh(AAT)
        return v[:,-p:], m
    else:
        ATA=data_mat.T*data_mat
        w, v=linalg.eigh(ATA)
        return data_mat*v[:, -p:], m