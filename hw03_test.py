import math
import numpy as np
from numpy import matlib
from numpy import random
from numpy import linalg
import matplotlib.pyplot as plt
from matplotlib import patches
from hw03_MoG import *

def testresult(datam, pl, muvl, covml):
    'test result'
    fig0=plt.figure(0)
    fig0_subplot=fig0.add_subplot(111, aspect='equal')
    fig0_subplot.grid(True)
    #plot
    fig0_subplot.clear()
#    for i, muv, covm in zip(range(i), muvl, covml):
    e=Covm2Ellipse(muvl[0], covml[0], _color='g')
    fig0_subplot.add_artist(e)
    e=Covm2Ellipse(muvl[1], covml[1], _color='r') 
    fig0_subplot.add_artist(e)
    fig0_subplot.plot(datam[0, :], datam[1, :], 'ob')
    fig0_subplot.axis([-5, 5, -5, 5])
    fig0

fig1=plt.figure(1)
subplot1=fig1.add_subplot(111, aspect='equal')
mum=matlib.mat(random.uniform(-2, 2, (2, 2)))
test_covml=[GetCovm2D(2, 4, 60), GetCovm2D(3, 1, 90)]
el=[Covm2Ellipse(mum[:, 0], test_covml[0], _color='r'), Covm2Ellipse(mum[:, 1], test_covml[1], _color='g')]
#e1.set_color('r')
#e1.set_alpha(0.5)
subplot1.add_artist(el[0])
subplot1.add_artist(el[1])
subplot1.axis([-4, 4, -4, 4])
subplot1.grid(True)
datam1=random.multivariate_normal(np.array(mum[:, 0]).reshape(-1), test_covml[0], 50)
datam2=random.multivariate_normal(np.array(mum[:, 1]).reshape(-1), test_covml[1], 55)
datam=np.append(datam1, datam2, 0).T;
subplot1.plot(datam1[:,0], datam1[:, 1], 'ro')
subplot1.plot(datam2[:,0], datam2[:, 1], 'go')
fig1
#initialization
pl=[0.5, 0.5]
muvl=matlib.mat(random.uniform(-2, 2, (2, 2)))
muvl=[random.uniform(-2, 2, (2, 1)), random.uniform(-2, 2, (2, 1))]
covml=[matlib.identity(2), matlib.identity(2)]


