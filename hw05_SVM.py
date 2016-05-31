import numpy as np
from numpy import random
def dftKernel(xiv, xjv):
    'do nothing linear kernel'
    assert isinstance(xiv, np.ndarray) and isinstance(xjv, np.ndarray)
    assert xiv.ndim==1 and xjv.ndim==1
    return np.inner(xiv, xjv)
class SVM:
    def __init__(self, C=1, Kernel=dftKernel, eps=1e-6, tol=0.01, maxiters=5):
        self.C=1
        self.Kernel=Kernel
        self.eps=eps
        self.tol=tol
        self.maxiters=maxiters
#        self.K=np.zeros((self.N, self.N), dtype=np.float32)
#        for i in xrange(self.N):
#            for j in xrange(self.N):
#                K[i, j]=self.targetv[i]*self.targetv[j]\
#                *self.Kernel(self.pointm[:, i], self.pointm[:, j])
    def Train(self, targetv, pointm):
        targetv=np.array(targetv)
        pointm=np.array(pointm)
        assert targetv.ndim==1 and pointm.ndim==2 and\
        len(targetv)==pointm.shape[1]
        self.targetv=targetv
        self.pointm=pointm
         #d means data dimensions N means data number
        self.N=pointm.shape[1]
        self.d=pointm.shape[0]
        #initialization
        self.alpha=np.zeros(self.N, dtype=np.float32)
        self.b=0.0
        self.nonbound=[False]*self.N
        numChanged=0
        examineAll=1
        iters=self.maxiters
        while numChanged>0 or examineAll:
            numChanged=0
            if examineAll:
                for i in xrange(self.N):
                    numChanged+=self.ExamineExmple(i)
            else:
                for i in xrange(self.N):
                    if self.nonbound[i]:
                        numChanged+=self.ExamineExmple(i)
            if examineAll:
                examineAll=0
            elif numChanged==0:
#                raw_input("end one iters")
                if iters!=0:
                    examineAll=1
                    iters=iters-1
#            raw_input("examineAll: "+str(examineAll)+" numChanged: "+str(numChanged))
        return self.alpha, self.b, [i for i in range(self.N) if self.nonbound[i]]
            
    def TakeStep(self, i1, i2):
        if(i1==i2):
            return 0
        alph1=self.alpha[i1]
        alph2=self.alpha[i2]
        y1=self.targetv[i1]
        y2=self.targetv[i2]
        E1=self.E(i1)
        E2=self.E(i2)
        s=y1*y2
        #compute L, H
        L=H=a1=a2=0
        if y1==y2:
            L=max(0, alph2+alph1-self.C)
            H=min(self.C, alph2+alph1)
        else:
            L=max(0, alph2-alph1)
            H=min(self.C, self.C+alph2-alph1)            
        if L==H:
            return 0
        k11=self.Kernel(self.pointm[:, i1], self.pointm[:, i1])
        k12=self.Kernel(self.pointm[:, i1], self.pointm[:, i2])
        k22=self.Kernel(self.pointm[:, i2], self.pointm[:, i2])
        eta=k11+k22-2*k12
        if eta>0:
            a2=alph2+y2*(E1-E2)/eta
            if a2<L:
                a2=L
            elif a2>H:
                a2=H
        else:
            f1=y1*(E1+self.b)-alph1*k11-s*alph2*k12
            f2=y2*(E2+self.b)-s*alph1*k12-alph2*k22
            L1=alph1+s*(alph2-L)
            H1=alph1+s*(alph2-H)
            Lobj=L1*f1+L*f2+0.5*L1*L1*k11+0.5*L*L*k22+s*L*L1*k12
            Hobj=H1*f1+H*f2+0.5*H1*H1*k11+0.5*H*H*k22+s*H*H1*k12
            if Lobj<Hobj-self.eps:
                a2=L
            elif Lobj>Hobj+self.eps:
                a2=H
            else:
                a2=alph2
        if abs(a2-alph2)<self.eps*(a2+alph2+self.eps):
            return 0
        a1=alph1+s*(alph2-a2)
        #update data
        b1=E1+y1*(a1-alph1)*k11+y2*(a2-alph2)*k12+self.b
        b2=E2+y1*(a1-alph1)*k12+y2*(a2-alph2)*k22+self.b
        a1nobound=a1>0 and a1<self.C
        a2nobound=a2>0 and a2<self.C
        if a1nobound:
            self.nonbound[i1]=True
            self.b=b1
        else:
            self.nonbound[i1]=False
            
        if a2nobound:
            self.nonbound[i2]=True
            self.b=b2
        else:
            self.nonbound[i2]=False
        
        if not (a1nobound or a2nobound):
            self.b=(b1+b2)/2
        self.alpha[i1]=a1
        self.alpha[i2]=a2
        return 1
    
    def ExamineExmple(self, i2):
        y2=self.targetv[i2]
        alph2=self.alpha[i2]
        E2=self.E(i2)
        r2=E2*y2
        if (r2<-self.tol and alph2<self.C) or (r2>self.tol and alph2>0):
            nonboundset=[i for i in range(self.N) if self.nonbound[i]]
            num=len(nonboundset)
            if num>=1:
                temp=[abs(self.E(i)-E2) for i in nonboundset if i!=i2]
                if temp==[]:
                    return 0
                i1=np.argmax(temp)
                if self.TakeStep(i1, i2):
                    return 1
            else:
                temp=[i for i in xrange(self.N) if i!=i2]
                if temp==[]:
                    return 0
                i1=random.choice(temp)
                if self.TakeStep(i1, i2):
                    return 1
        return 0
                    
    def u(self, i):
        ui=0
        for j in xrange(self.N):
#            print j
            ui=ui+self.targetv[j]*self.alpha[j]*\
            self.Kernel(self.pointm[:, j], self.pointm[:, i])
        ui=ui-self.b
        return ui
    def E(self, i):
        return self.u(i)-self.targetv[i]
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            