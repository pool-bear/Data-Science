import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df=pd.read_csv("data.csv",index_col=0,header=0)
y_samp=df.iloc[:,0]
x_samp=df.iloc[:,1:]
r=LinearRegression().fit(x_samp,y_samp)
print("BLUE Beta=(",r.intercept_,r.coef_[0],r.coef_[1],")")
x_samp_=np.hstack((np.ones((df.shape[0],1)),x_samp))
def sqsum(input):
	return (input.reshape(1,-1)@input)[0,0]
def loss(input):
	y_pred=x_samp_@input
	diff=y_samp.values.reshape((df.shape[0],1))-y_pred
	output=sqsum(diff)
	return output
blue=np.array([r.intercept_,r.coef_[0],r.coef_[1]]).reshape((3,-1))
print("SSR=",loss(blue),"\n")

beta=np.array([[0.01],[0.01],[0.01]])
rate=0.01
def grad(input):
	d=0.000001
	output=np.zeros((3,1))
	input_=np.copy(input)
	for i in range(3):
		input_[i,0]+=d
		output[i,0]=(loss(input_)-loss(input))/d
		input_[i,0]-=d
	scale=np.sqrt(rate**2/sqsum(output))
	output*=scale
	return output
iter=10000
best=np.copy(beta)
best_loss=loss(best)
best_no=0
final_no=2000
final=np.zeros(final_no)
for i in range(iter):
	beta-=grad(beta)
	if(loss(beta)<best_loss):
		best_no=i+1
		best=np.copy(beta)
		best_loss=loss(best)
	if(i>=iter-final_no):
		final[i-iter+final_no]=loss(beta)
print("Smallest loss=",best_loss)
print("Learning Rate=",rate)
print("Beta=",best)
print("No. of updates when reaches \"optimal\":",best_no)
plt.plot(np.arange(final_no),final)
plt.xlabel("update number")
plt.ylabel("loss")
plt.title("loss")
plt.show()