import numpy as np
import numpy.polynomial as poly
import matplotlib.pyplot as plt
import hw01_points_estimation
from hw01_points_estimation import *
x=np.linspace(0, 1, 10)
m=len(x)
y=np.sin(2*np.pi*x)
#add noise
sigma=0.05
mu=0.0
x=x+sigma*np.random.randn(m)+mu
y=y+sigma*np.random.randn(m)+mu
#parameters

w=PointsEstimate(x, y, 4, 0)
rp=poly.Polynomial(np.array(w).reshape(-1))
#plot
plt.figure(0);
rp_x=np.linspace(0, 1, 100)
rp_y=rp(rp_x)
plt.plot(rp_x, rp_y, 'b', x, y, 'or', rp_x, sin(2*np.pi*rp_x), 'g')
plt.show()