import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
import hw02_PCA 
#data digit3
digit3s=[]
optdigits=open('optdigits.tra')
for line in optdigits:
    line_data=[int(x) for x in line.split(',')]
    if(line_data[-1]==3):
        digit3s.append(line_data[:-1])

N=len(digit3s)
d=len(digit3s[0])
#allocate data
data_mat=matlib.zeros((d, N), order='F', dtype=np.float32)
for i, e in enumerate(digit3s):
    data_mat[:, i]=np.array(e).reshape(64, 1)
#PCA
B, m=hw02_PCA.PCA(data_mat, 2)
w=B.T*data_mat
x, y=w
plt.plot(x, y, 'ro')