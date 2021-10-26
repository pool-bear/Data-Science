'''
Tianyi Lu, UNI:tl3126, E-mail:tl3126@columbia.edu
ACTU PS5841 Data Science Assignment 4
'''
import numpy as np
from numpy.random.mtrand import sample
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Start of Part A
def sqsum(input): # Squared sum of a vector, input is a n*1 vector
	return (input.reshape(1,-1)@input)[0,0]
def loss(input): # Squared Sum of Residuals, input is a beta
	y_pred=x_samp_@input
	diff=y_samp.values.reshape((df.shape[0],1))-y_pred
	output=sqsum(diff)
	return output
df=pd.read_csv("data.csv",index_col=0,header=0)
y_samp=df.loc[:,"y"]
x_samp=df.loc[:,"X1":"X2"]
r=LinearRegression().fit(x_samp,y_samp)
blue=np.array([r.intercept_,r.coef_[0],r.coef_[1]]).reshape((-1,1))
x_samp_=np.hstack((np.ones((df.shape[0],1)),x_samp)) #add ones to first column for calculation

print("BLUE Beta=(",blue.reshape(1,-1),")")
print("SSR=",loss(blue),"\n")

# Start of Part B
# Parameters
rate=0.01 # Learning rate
sensitivity=0.0001 # Sensitivity in determining convergence
beta=np.array([[0.01],[0.01],[0.01]])# Initial beta
# Main
def grad(input): # Gradient of loss at input, input is a beta
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
class best_try: # To record best beta
	def __init__(self):
		self.beta=np.copy(beta)
		self.loss=loss(beta)
		self.num=0
	def update(self,beta,num):
		self.beta=np.copy(beta)
		self.loss=loss(beta)
		self.num=num
class final_records: # To record final 50 betas for plotting
	def __init__(self,size):
		self.size=size
		self.__used=0
		self.content=np.zeros(size)
	def update(self,add):
		if(self.__used<self.size):
			self.content[self.__used]=add
			self.__used+=1
		else:
			self.content[0:self.size-1]=self.content[1:self.size]
			self.content[self.size-1]=add
	def end_diff(self):
		if(self.__used<2):
			return 9999
		return np.abs(self.content[self.__used-1]-self.content[self.__used-2])
best=best_try()
final=final_records(50)
count=0
while(final.end_diff()>sensitivity):
	count+=1
	beta-=grad(beta)
	if(loss(beta)<best.loss):
		best.update(beta,count)
	final.update(loss(beta))
# Report and plot
print("Smallest loss=",best.loss)
print("Learning Rate=",rate)
print("Beta=",best.beta)
print("No. of updates when reaches \"optimal\":",best.num)
plt.plot(np.arange(final.size),final.content)
plt.xlabel("update number")
plt.ylabel("loss")
plt.title("loss")
plt.show()