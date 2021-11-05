import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# Part A
df=pd.read_csv("~/Documents/Default.csv")
y=df.iloc[:,1]
x=df.iloc[:,2:]
reg=LogisticRegression().fit(x,y)
print("Coefficients:\n\tb0="+str(reg.intercept_[0])+"\n\tb1="+str(reg.coef_[0][0])+"\n\tb2="+str(reg.coef_[0][1]))
print("Maximum likelihood achieved: "+str(reg.score(x,y)))

# Part B
# Parameters
rate=25 # Learning rate
update=200
beta=np.array([[0.0],[0.0],[0.0]]).reshape(-1,1) # Initial beta
volume=100
# Prepare x_std and y_ for neural network
n=x.shape[0]
y_=1*(y=="Yes")
x_std=np.copy(x)
for i in range(x.shape[1]):
    x_std[:,i]=x_std[:,i]-np.mean(x_std[:,i])
    x_std[:,i]=x_std[:,i]/np.std(x_std[:,i])
x_std=np.hstack((np.ones((n,1)),x_std))
# Define functions
def activation(input): # Activation function
    return 1/(1+np.exp(-input))
def loss(input,seed): # Loss function
    output=0
    volume_=volume
    rng=np.random.default_rng(seed=seed)
    draw=rng.choice(n,size=volume,replace=False)
    for i in range(volume):
        predict=activation((x_std[draw[i],:]@input)[0])
        if(predict==0 or predict==1):
            volume_-=1
        else:
            output-=(1-y_[draw[i]])*np.log(1-predict)+y_[draw[i]]*np.log(predict)
    return output/volume_
def grad(input,seed): # Gradient of loss at input, input is a beta
	d=0.0001
	output=np.zeros((3,1))
	input_=np.copy(input)
	for i in range(3):
		input_[i,0]+=d
		output[i,0]=(loss(input_,seed)-loss(input,seed))/d
		input_[i,0]-=d
	scale=np.sqrt(rate**2/np.sum(output**2))
	output*=scale
	return output
class optimal:
	def __init__(self,beta):
		self.beta=np.copy(beta)
		self.loss=999
	def update(self,beta,score):
		self.beta=np.copy(beta)
		self.loss=score
best=optimal(beta)
# Start
for i in range(update):
    print("Loading..."+str(int((i+1)/update*100))+"%",end="\r")
    beta-=grad(beta,i)
    score=loss(beta,i)
    if(score<best.loss):
	    best.update(beta,score)
print("",end="\r")
volume=n
print("Loss:"+str(best.loss))
print("Beta:\n\tb0="+str(best.beta[0][0])+"\n\tb1="+str(best.beta[1][0])+"\n\tb2="+str(best.beta[2][0]))
volume=10000
betaa=[[-11.540478115172002],[0.005647107969070656],[0.00002080919845838359]]
print(loss(best.beta,0))
print(loss(betaa,0))
# batch gradient descent with 200 iterations
# loss:0.6148734530177803
# beta:[[-24.77585592]
#  [  3.33483721]
#  [ -0.18927353]]