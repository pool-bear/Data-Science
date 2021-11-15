'''
Tianyi Lu, UNI:tl3126, E-mail:tl3126@columbia.edu
ACTU PS5841 Data Science Assignment 7
'''
from sklearn.linear_model import PoissonRegressor
from statsmodels.api import Poisson
import numpy as np
import pandas as pd

# Load the data
df=pd.read_csv('~/Documents/data.csv')
y=df['accidents'].values.reshape(-1,1)
df['accidents_']=df['accidents']/df['personyears']
y_=df['accidents_'].values.reshape(-1,1)
x=df['age'].values.reshape(-1,1)
x_=np.hstack((np.ones((len(y),1)),df['age'].values.reshape(-1,1)))

# Log likelihood
def log_likelihood(x,y,beta):
    lmb=np.exp(np.log(df['personyears'].values.reshape(-1,1))+beta[0][0]+beta[1][0]*x)
    output=0
    for i in range(lmb.shape[0]):
        log_factorial=0
        for j in range(y[i][0]):
            log_factorial+=np.log(j+1)
        output+=-lmb[i][0]+y[i][0]*np.log(lmb[i][0])-log_factorial
    return output

# Sklearn
reg_sk=PoissonRegressor(alpha=0).fit(x,np.ravel(y_))
beta_sk=np.array([[reg_sk.intercept_],[reg_sk.coef_[0]]])
print("Sklearn results:")
print("b0="+str(beta_sk[0][0])+"\nb1="+str(beta_sk[1][0]))
print("Log likelihood="+str(log_likelihood(x,y,beta_sk))+"\n")

# Statsmodels
reg_sm=Poisson(y,x_,offset=np.log(df['personyears'])).fit(disp=0)
beta_sm=reg_sm.params.reshape(-1,1)
print("Statsmodels results:")
print("b0="+str(beta_sm[0][0])+"\nb1="+str(beta_sm[1][0]))
ll_sm=log_likelihood(x,y,beta_sm)
print("Log likelihood="+str(ll_sm))
reg_nv=Poisson(y,np.ones((len(y),1)),offset=np.log(df['personyears'])).fit(disp=0)
beta_nv=np.vstack((reg_nv.params,np.zeros((1,1))))
ll_nv=log_likelihood(x,y,beta_nv)
print("Pseudo R2="+str(1-ll_sm/ll_nv))