'''
Tianyi Lu, UNI:tl3126, E-mail:tl3126@columbia.edu
ACTU PS5841 Data Science Assignment 6
'''
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Part A
## Regrression
df=pd.read_csv("~/Documents/Default.csv")
y=df.iloc[:,1]
x=df.iloc[:,2:]
reg=LogisticRegression(random_state=0,solver='lbfgs',multi_class='ovr')
reg.fit(x,y)
beta_reg=np.zeros((x.shape[1]+1,1))
print("a1. Beta estimated:\n\tb0="+str(reg.intercept_[0]))
beta_reg[0]=reg.intercept_[0]
for i in range(reg.coef_.shape[1]):
    print("\tb"+str(i+1)+"="+str(reg.coef_[0][i]))
    beta_reg[i+1]=reg.coef_[0][i]

## Calculate maximum likelihood achieved
n=x.shape[0]
x_=np.copy(x)
x_=np.hstack((np.ones((n,1)),x_))
y_=np.copy(1*(y=="Yes")).reshape(-1,1)
def activation(input):
    return 1/(1+np.exp(-input))
def loss(input,x,y=y_):
    predict=activation(x@input)
    output=np.transpose(1-y)@np.log(1-predict)+np.transpose(y)@np.log(predict)
    return -output[0][0]/n
print("\na2. Maximum likelihood achieved:\n\t"+str(np.exp(-loss(beta_reg,x_))))


# Part B
## Parameters
rate=0.044 # Learning rate, a float
update=200 # Number of iterations, a positive integer
beta=np.array([[0.0],[0.0],[0.0]]).reshape(-1,1) # Initial beta, n+1 floats
d=0.00001 # Step size used in calculation of gradient, a small positive float

## Standardize x
x_std=np.copy(x)
for i in range(x.shape[1]):
    x_std[:,i]=x_std[:,i]-np.mean(x_std[:,i])
    x_std[:,i]=x_std[:,i]/np.std(x_std[:,i])
x_std=np.hstack((np.ones((n,1)),x_std))

## Define functions
def grad(input):
    output=np.zeros((x.shape[1]+1,1))
    input_=np.copy(input)
    for i in range(x.shape[1]+1):
        input_[i,0]+=d
        output[i,0]=(loss(input_,x_std)-loss(input,x_std))/d
        input_[i,0]-=d
    scale=np.sqrt(rate**2/np.sum(output**2))
    output*=scale
    return output
class optimal:
    def __init__(self,beta):
        self.beta=np.copy(beta)
        self.loss=loss(beta,x_std)
    def update(self,beta,score):
        self.beta=np.copy(beta)
        self.loss=score
best=optimal(beta)

## Main
for i in range(update):
    beta-=grad(beta)
    score=loss(beta,x_std)
    if(score<best.loss):
        best.update(beta,score)

## Print results
print("\nb1. Smallest binary cross-entropy achieved:\n\t"+str(best.loss))
print("\nb2. Associated weights:")
for i in range(best.beta.shape[0]):
    print("\tb"+str(i)+"="+str(best.beta[i][0]))

## Print corresponding beta to unstandardized x
beta_unstd=np.copy(best.beta)
for i in range(x.shape[1]):    
    beta_unstd[0,0]=beta_unstd[0,0]-np.mean(x_[:,i+1])*beta_unstd[i+1,0]/np.std(x_[:,i+1])
    beta_unstd[i+1,0]=beta_unstd[i+1,0]/np.std(x_[:,i+1])
print("\nb3. Corresponding beta for non-standardized features:")
for i in range(beta_unstd.shape[0]):
    print("\tb"+str(i)+"="+str(beta_unstd[i][0]))