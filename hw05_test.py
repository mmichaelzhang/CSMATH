from numpy import random
import matplotlib.pyplot as plt
from matplotlib import lines
from hw03_MoG import *
from hw05_SVM import *
def makeDecisionF(w, b):
    def DecisionF(x):
        return np.inner(w, x)-b
    return DecisionF
def makeLine2D(w, b, color='b', linestyle='-'):
    'Get a Line2D wx-b=0'
    assert isinstance(w, np.ndarray)
    assert w.ndim==1 and len(w)==2
    if w[0]!=0:
        y=np.array([-100, 100])
        x=(b-w[1]*y)/w[0]
        return lines.Line2D(x, y, color=color, linestyle=linestyle)

fig1=plt.figure(1)
subplot1=fig1.add_subplot(111, aspect='equal')
mum=matlib.mat(random.uniform(-2, 2, (2, 2)))
test_covml=[GetCovm2D(1, 2, 60), GetCovm2D(1.5, 0.5, 90)]
el=[Covm2Ellipse(mum[:, 0], test_covml[0], _color='r'), Covm2Ellipse(mum[:, 1], test_covml[1], _color='g')]
#e1.set_color('r')
#e1.set_alpha(0.5)
subplot1.add_artist(el[0])
subplot1.add_artist(el[1])
subplot1.axis([-4, 4, -4, 4])
subplot1.grid(True)
num1=50
num2=55
datam1=random.multivariate_normal(np.array(mum[:, 0]).reshape(-1), test_covml[0], num1)
datam2=random.multivariate_normal(np.array(mum[:, 1]).reshape(-1), test_covml[1], num2)
datam=np.append(datam1, datam2, 0).T;
subplot1.plot(datam1[:,0], datam1[:, 1], 'ro')
subplot1.plot(datam2[:,0], datam2[:, 1], 'go')
fig1
#test svm
targetv=[1]*num1+[-1]*num2
svm=SVM(maxiters=10, C=0.1)
alpha, b, noboundset=svm.Train(targetv, datam)
print len(noboundset)
boundarypoint1=[i for i in noboundset if targetv[i]==1]
boundarypoint2=[i for i in noboundset if targetv[i]==-1]
x1, y1=datam[:, boundarypoint1]
x2, y2=datam[:, boundarypoint2]
w=np.dot(datam, alpha*targetv)
DecisionF=makeDecisionF(w, b)
out1=[]
out2=[]
for i in range(num1+num2):
    if DecisionF(datam[:, i])>0:
        out1.append(i)
    else:
        out2.append(i)
out1data=datam[:, out1]
out2data=datam[:, out2]
fig2=plt.figure(2)
subplot2=fig2.add_subplot(111)
subplot2.axis([-4, 4, -4, 4])
subplot2.add_line(makeLine2D(w, b, color='y'))
subplot2.add_line(makeLine2D(w, b+1, color='g'))
subplot2.add_line(makeLine2D(w, b-1, color='b'))
subplot2.plot(out1data[0,:], out1data[1, :], 'ro', out2data[0, :], out2data[1, :], 'go')
fig2