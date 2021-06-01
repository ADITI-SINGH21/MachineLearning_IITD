import xlrd
import matplotlib.pyplot as plt
import numpy as np
import math
book = xlrd.open_workbook("q2train.xlsx")
sh = book.sheet_by_index(0)
book1 = xlrd.open_workbook("q2test.xlsx")
sh1 = book1.sheet_by_index(0)
def sigmoid(x):
      return 1 / (1 + math.exp(-x))
y=[]
aptitude=[]
verbal=[]
aptitudep=[]
verbalp=[]
for i in range(1, sh.nrows):
    y.append(sh.cell_value(i,2)) 
    aptitude.append(sh.cell_value(i,0))
    verbal.append(sh.cell_value(i,1))
for i in range(1, sh1.nrows):
   # p.append(sh.cell_value(i,2)) 
    aptitudep.append(sh.cell_value(i,0))
    verbalp.append(sh.cell_value(i,1))
aptitude=np.asarray(aptitude)
verbal=np.asarray(verbal)
y=np.asarray(y)
y=y.transpose()
aptitude=aptitude.transpose()
verbal=verbal.transpose()
verbal1=[]
aptitude1=[]
verbal2=[]
aptitude2=[]
hello=[]
for i in range(0,sh.nrows-1):
    if(y[i]==0):
        aptitude1.append(aptitude[i])
        verbal1.append(verbal[i])
    else:
        aptitude2.append(aptitude[i])
        verbal2.append(verbal[i])
theta00=-6.3
theta11=0.015
theta22=0.08
theta0=0.0
theta1=0.0
theta2=0.0
delta=1.0
alpha=0.04
temp=1.0
while(abs(delta)>pow(10,-5)):
#for j in range(1,10):
    cost=0
    for i in range(0,sh.nrows-1):
        val0=y[i]-sigmoid(theta0+theta1*aptitude[i]+theta2*verbal[i])
        val1=(y[i]-sigmoid(theta0+theta1*aptitude[i]+theta2*verbal[i]))*aptitude[i]
        val2=(y[i]-sigmoid(theta0+theta1*aptitude[i]+theta2*verbal[i]))*verbal[i]
        cost+=-(y[i]*(math.log(sigmoid(theta0+theta1*aptitude[i]+theta2*verbal[i])))+(1-y[i])*math.log(1-math.log(sigmoid(theta0+theta1*aptitude[i]+theta2*verbal[i]))))
        theta0=theta0+alpha*val0
        theta1=theta1+alpha*val1
        theta2=theta2+alpha*val2
        #print(val0,val1,val2)
    #print(theta0,theta1,theta2)
    cost=cost/100
    hello.append(cost)
    #print(hello)
    delta=cost-temp
    temp=cost
print(theta0,theta1,theta2)
plt.ylim(20, 100)
plt.scatter(aptitude1,verbal1,c='red')
plt.scatter(aptitude2,verbal2,c='green')
#plt.plot(aptitude,(0.00499+0.42*aptitude)/(-0.63))
plt.plot(aptitude,(theta0+theta1*aptitude)/(-theta2))
plt.show()
a0=[]
v0=[]
a1=[]
v1=[]
f = open("output.txt", "a")
for i in range(0,30):
    if((sigmoid(theta00+theta11*aptitude[i]+theta22*verbal[i]))>0.5):
        a0.append(aptitude[i])
        v0.append(verbal[i])
        print("1", file=f)
    else:
        a1.append(aptitude[i])
        v1.append(verbal[i])
        print("0", file=f)
plt.scatter(a0,v0,c='green')
plt.scatter(a1,v1,c='red')
plt.show()
f.close()
