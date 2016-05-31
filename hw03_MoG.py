import math
import numpy as np
from numpy import matlib
from numpy import linalg
import matplotlib.pyplot as plt
from matplotlib import patches

def Covm2Ellipse(muv, covm, nsigma=3, _alpha=0.5, _color='r'):
    assert covm.shape==(2, 2)
    muv=np.array(muv).reshape(-1)
    assert len(muv)==2
    w, v=linalg.eigh(covm)
    #print 'v', v
    l=2*np.sqrt(w)*nsigma
    return patches.Ellipse(np.array(muv).reshape(-1), l[0], l[1], \
    math.degrees(math.atan2(v[1, 0], v[0, 0])), alpha=_alpha, color=_color)

def GetCovm2D(axis1_len, axis2_len, axis1_angle, nsigma=3):
    'axis1_angle measured in degrees'
    assert axis1_len>0 and axis2_len>0
    w=np.array([axis1_len, axis2_len], dtype=np.float32)
    w=(w/nsigma)**2
    v=matlib.zeros((2, 2), dtype=np.float32)
    axis1_rad=math.radians(axis1_angle)
    axis2_rad=math.radians(axis1_angle+90)
    v[:, 0]=matlib.mat((math.cos(axis1_rad), math.sin(axis1_rad))).T
    v[:, 1]=matlib.mat((math.cos(axis2_rad), math.sin(axis2_rad))).T
    #print 'v get2D', v
    return v*matlib.diag(w)*v.T

#v means vector m means matrix
def makeNormalF(muv, covm):
    'Generate a Gaussian function based on the mu and cov'
    assert isinstance(covm, matlib.matrix)
    assert covm.shape[0]==covm.shape[1] 
    muv=np.mat(muv).reshape(covm.shape[0], 1)
    m=covm.shape[0]
    coeff=(np.sqrt((2*np.pi))**m)*np.sqrt(linalg.det(covm))
    coeff=1.0/coeff
    covm_inv=linalg.inv(covm)
    def NormalF(xv):
        'Gaussian function'
        xv=matlib.mat(xv).reshape(m, 1)
        return coeff*np.exp(-0.5*float((xv-muv).T*covm_inv*(xv-muv)))
    return NormalF

#l means list or turple
def makeMoG(pl, muvl, covml ):
    'Generate a Mixture Gaussian Model'
    funcl=[]
    for muv, covm in zip(muvl, covml):
        funcl.append(makeNormalF(muv, covm))
    def MoG(xv):
        'Mixture Gaussian Model'
        sum=0.0
        for p, func in zip(p, funcl):
            sum+=p*func(xv)
        return sum
    return MoG

#use EM algorithm to estimate the parameters of MoG   
def EMForMoG(datam, n, pl, muvl, covml, max_iters=1):
    assert len(pl)==len(muvl) and len(muvl)==len(covml) 
    d=datam.shape[0]
    N=datam.shape[1]
    #start EM posteriorm n*N
    posteriorm=matlib.zeros((n, N), dtype=np.float32)
    for it in xrange(max_iters):
        print 'Iter: '+str(it)
        #create Gaussian kernels
        NormalFl=[makeNormalF(muv, covm) for muv, covm in zip(muvl, covml)]
        #probability
        px=matlib.zeros((1, N), dtype=np.float32)
        posteriorm.fill(0)
        #caculate posteriorm
        for j in xrange(N):
            cur_data=datam[:, j]
            for i in xrange(n):
                #print i, j
                posteriorm[i, j]=pl[i]*NormalFl[i](cur_data)
                px[0, j]+=posteriorm[i, j]
#        print 'px:', px
        posteriorm/=px
        #update parameters
        #soft num n*1
        softnum=matlib.sum(posteriorm, 1)
        print softnum
        softnum_inv=1.0/softnum
        pl=np.array((softnum/N)).reshape(-1).tolist()
        mum=datam*posteriorm.T*matlib.diag(np.array(softnum_inv).reshape(-1))
        muvl=[mum[:, k] for k in range(n)]
        mum=[]#release         
        for k in range(n):
            datam_temp=datam-muvl[k]
            covml[k]=softnum_inv[k, 0]*datam_temp*matlib.diag(np.array(posteriorm[k, :]).reshape(-1))\
            *datam_temp.T
    return pl, muvl, covml
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        