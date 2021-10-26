import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.interpolate import CubicSpline, UnivariateSpline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

d=np.zeros((100,4))
d[:,3]=default_rng(0).normal(0,3,100)
d[:,2]=default_rng(1).uniform(0,10,100)
d[:,1]=default_rng(2).uniform(0,10,100)
d[:,0]=10+0.5*d[:,1]+3*d[:,2]+d[:,3]
f=np.zeros((100,4))
for i in range(4):
	f[:,i]=pd.Series(d[:,i]).rolling(window=3,center=True).mean()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(d[:,1],d[:,2],d[:,0])
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()

