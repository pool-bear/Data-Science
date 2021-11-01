'''
Tianyi Lu, UNI:tl3126, E-mail:tl3126@columbia.edu
ACTU PS5841 Data Science Assignment 5
'''
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
from numpy.linalg import inv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

#Part a1
raw=pd.read_csv('~/Documents/data.csv')
x=raw.iloc[:,0:2]
y=raw.iloc[:,2]
reg=lr().fit(x,y)
print("Coefficients:","\n\tb0 =",reg.intercept_,"\n\tb1 =",reg.coef_[0],"\n\tb2 =",reg.coef_[1])

#Part a2
def sse(x,y):
    y_pred=reg.predict(x)
    return np.sum((y-y_pred).T@(y-y_pred))
sigma2_est=sse(x,y)/(x.shape[0]-x.shape[1]-1)
print("Estimated sigma: ",np.sqrt(sigma2_est))

#Part B
x_=np.hstack((np.ones((x.shape[0],1)),x))
def predVar(x1,x2):
    input=np.array([1,x1,x2]).reshape(-1,1)
    return ((input.T)@inv(x_.T@x_)@input*sigma2_est)[0][0]+sigma2_est
print("Prediction variance at (3.14,3.14): ",predVar(3.14,3.14))

#Part C
k=5
split=KFold(n_splits=k)
scores=-cross_val_score(reg,x,y,cv=split,scoring='neg_mean_squared_error')
print("Prediction variance estimated:",np.mean(scores))
print("MSE for test fold associated with K:")
for i in range(k):
    print("\tK="+str(i+1)+": ",scores[i])