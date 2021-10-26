'''
Tianyi Lu, UNI:tl3126, E-mail:tl3126@columbia.edu
ACTU PS5841 Data Science Assignment 1
'''

import sys

# Calculation
def annuity(term,rate,mode):
	if(rate==0):
		return term
	fv=((1+rate)**term-1)/rate
	if(mode=="pv"):
		return fv/((1+rate)**term)
	return fv

# Print the table when no argument
def table(mode):
	if(mode=="pv"):
		title="Present"
	else:
		title="Future"
	# Title
	print("{:^92}".format(title+" Value of Annuity Immediate"))
	# First row
	print("  ",end="")
	for rate_ in range(9):
		rate=rate_+1
		print("{:>10}".format("0."+str(rate)+"%"),end="")
	print("")
	# Row [1,24]
	for term_ in range(24):
		term=term_+1
		print("{:>2}".format(str(term)),end="")
		for rate_ in range(9):
			rate=0.01*(rate_+1)
			print("{:>10.4f}".format(annuity(term,rate,mode)),end="")
		print("")
	# Row 30,40,50
	for term_ in range(3,6):
		term=term_*10
		print("{:>2}".format(str(term)),end="")
		for rate_ in range(9):
			rate=0.01*(rate_+1)
			print("{:>10.4f}".format(annuity(term,rate,mode)),end="")
		print("")

# Print output when arguments valid
def arg(term,rate):
	pv=annuity(term,rate,"pv")
	fv=annuity(term,rate,"fv")
	print("{:^40}".format("Annuity Immediate"))
	print("{:>10}{:>10}{:>10}{:>10}".format("term","interest","PV","FV"))
	print("{:>10.0f}{:>10}{:>10.4f}{:>10.4f}".format(term,rate,pv,fv))

# Check number of inputs
if(len(sys.argv)==1):	
	table("pv")
	print("")
	table("fv")
	sys.exit()
elif(len(sys.argv)!=3):
	print("error: incorrect number of inputs")
	sys.exit()

# Verify if arguments valid
try:
	term=float(sys.argv[1])
except:
	term=-1	
try:
	rate=float(sys.argv[2])
except:
	rate=-1
if(term<1 or int(term)!=term or term>100):
	if(rate<=-1 or rate>=1):
		print("input error: term, rate")
	else:
		print("input error: term")
	sys.exit()
elif(rate<=-1 or rate>=1):
	print("input error: rate")
	sys.exit()
else:
	arg(term,rate);
	sys.exit()