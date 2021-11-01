'''
Tianyi Lu, UNI:tl3126, E-mail:tl3126@columbia.edu
ACTU PS5841 Data Science Assignment 5
'''
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
from numpy.linalg import inv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score as cv

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
print("Estimated sigma =",np.sqrt(sigma2_est))

#Part B
x_=np.hstack((np.ones((x.shape[0],1)),x))
def predVar(x1,x2):
    input=np.array([1,x1,x2]).reshape(-1,1)
    return ((input.T)@inv(x_.T@x_)@input*sigma2_est)[0][0]+sigma2_est
print("Predicted variance =",predVar(3.14,3.14))

#Part C
k_start=2 #1-fold is not possible?
k_end=5
mse_kfold=[]
for k in range(k_start,k_end+1):
    split=KFold(n_splits=k,shuffle=True,random_state=3126)
    scores=cv(reg,x,y,cv=split,scoring='neg_mean_squared_error')
    scores=-scores
    mse_kfold.append(np.mean(scores))
print("Prediction variance estimated with 5-fold cross validation:",mse_kfold[5-k_start])
print("MSE for other k:")
for k in range(k_start,k_end):
    print("\t"+str(k)+"-fold: ",mse_kfold[k-k_start])