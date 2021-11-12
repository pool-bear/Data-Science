from sklearn.linear_model import PoissonRegressor
from statsmodels.api import Poisson
import numpy as np
import pandas as pd

df=pd.read_csv('~/Documents/data.csv')
y=df['accidents'].values.reshape(-1,1)
df['accidents_']=df['accidents']/df['personyears']
y_=df['accidents_'].values.reshape(-1,1)
x=df['age'].values.reshape(-1,1)
reg_sk=PoissonRegressor(alpha=0).fit(x,np.ravel(y_))
beta=np.array([[reg_sk.intercept_],[reg_sk.coef_[0]]])
print(beta)

def log_likelihood(x,y,beta):
    lmb=np.exp(np.log(df['personyears'].values.reshape(-1,1))+beta[0][0]+beta[1][0]*x)
    output=0
    for i in range(lmb.shape[0]):
        log_factorial=0
        for j in range(y[i][0]):
            log_factorial+=np.log(j+1)
        output+=-lmb[i][0]+y[i][0]*np.log(lmb[i][0])-log_factorial
    return output
print(log_likelihood(x,y,beta))

x_=np.hstack((np.ones((len(y),1)),df['age'].values.reshape(-1,1)))
reg_sm=Poisson(y,x_,offset=np.log(df['personyears'])).fit()
beta=reg_sm.params.reshape(-1,1)
print(reg_sm.summary())
print(beta)
l=log_likelihood(x,y,beta)
print(l)
beta_naive=[[np.log(np.average(y_))],[0]]
l_naive=log_likelihood(x,y,beta_naive)
r2=1-l/l_naive# todo: check this
print(reg_sm.summary())
print(r2)