'''
Tianyi Lu, UNI:tl3126, E-mail:tl3126@columbia.edu
ACTU PS5841 Data Science Assignment 2
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Open file and derive daily return
df=pd.read_csv("SP500_randomized.csv",skiprows=1)
df["Date"]=pd.to_datetime(df["Date"])
df["Close_l"]=np.log(df["Close"])
df=df.sort_values(by="Date")
df["Diff_l"]=df["Close_l"].diff()

# Derive PDF
mean_l=df["Diff_l"][1:].mean()
std_l=df["Diff_l"][1:].std()
df["Rtrn"]=(np.exp(df["Diff_l"])-1)*100 # Return is calculated in percentage and satisfies $Price_t=(Return+100)\%*Price_{t-1}$ (TeX expression)
mean=df["Rtrn"][1:].mean()
std=df["Rtrn"][1:].std()

# Print estimated parameters
print("Mean={0:.4f}%\tStdev={1:.4f}\tVariance={2:.4f}".format(mean,std,std**2))
print("Skewness={0:.4f}\tKurtosis={1:.4f}".format(df["Rtrn"][1:].skew(),df["Rtrn"][1:].kurt()))

# Plot PDF
n,bins,patches=plt.hist(df["Rtrn"][1:],bins=100,density=True,color='blue',rwidth=0.9)
plt.xlabel("daily return")
plt.ylabel("pdf")
plt.title("probablity density function")
plt.show()

# Plot QQ
stats.probplot(df["Rtrn"][1:],dist="norm",fit=False,plot=plt)
plt.plot(np.linspace(-10,10,10),np.linspace(-10,10,10),color="red")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title("")
plt.show()

# Prediction
now=4430
mc_try=100000
now_l=np.log(now)
predict=np.zeros(mc_try)
for path in range(mc_try):
	np.random.seed(path)
	predict_l=now_l
	step=np.random.normal(mean_l,std_l,252)
	for day in range(252):
		predict_l+=step[day]
	predict[path]=np.exp(predict_l)
print("Predicted value={0:.2f}\tStdev={1:.4f}\tVar:{2:.4f}".format(predict.mean(),predict.std(),predict.var()))

# Plot PDF
n,bins,patches=plt.hist(predict,bins=100,density=True,color='blue',rwidth=0.9)
plt.xlabel("Price(T)")
plt.ylabel("pdf")
plt.title("probablity density function")
plt.show()

# Plot QQ
stats.probplot(predict,dist="lognorm",sparams=stats.lognorm.fit(predict),plot=plt)
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.title("")
plt.show()