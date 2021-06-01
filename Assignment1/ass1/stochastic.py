import xlrd
import matplotlib.pyplot as plt
import numpy as np
import time
book = xlrd.open_workbook("q1.xlsx")
sh = book.sheet_by_index(0)
alpha=0.0001
t0old = 1.0
t1old = 1.0
t0=0.0
t1=0.0
n=0
cost=0.0
delta = 1.0
temp = 1.0
#while(abs(t0-t0old)>pow(10,-5)):
hi=[]
y=[]
x=[]
k=0
for i in range(1, sh.nrows):
    y.append(sh.cell_value(i,1)) 
    x.append(sh.cell_value(i,0))
start=time.time()
while(abs(delta)>pow(10,-8)):
    cost=0.0
    for j in range(1, sh.nrows):
        t0old=t0
        t1old=t1
        #print(t0old, t1old)
  
        val0 = sh.cell_value(j,1)-t0old-t1old*sh.cell_value(j,0)
        val1 =(sh.cell_value(j,1)-t0old-t1old*sh.cell_value(j,0))*sh.cell_value(j,0)
        cost += pow((sh.cell_value(j,1)-t0old-t1old*sh.cell_value(j,0)),2)
        t0=t0old+2*alpha*val0
        t1=t1old+2*alpha*val1
    delta=t1-temp
    hi.append(cost/2)
    temp=t1
     

    #print(t0,t1)
    n+=1
print(t0,t1)

#cost=0.5*cost 
Y=[]
for l in range(0,sh.nrows-1):
    Y.append(x[l]*t1+t0)
plt.plot(x,Y,'r')
plt.scatter(x, y)
plt.show()
end=time.time()
print(end-start)
arr=np.arange(0,n,1)
plt.plot(arr, hi)
plt.show()