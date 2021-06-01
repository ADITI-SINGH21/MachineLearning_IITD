import xlrd
import matplotlib.pyplot as plt
import numpy as np
book = xlrd.open_workbook("q1.xlsx")
sh = book.sheet_by_index(0)
#for i in range(sh.ncols): 
   # print(sh.cell_value(0, i)) 
    
#print(sh.cell_value(0,0))
t0old = 0.0
t1old = 0.0
t0=1.0
t1=1.0
alpha=0.0001
n=1
hello=[ ]
y=[]
x=[]
k=0
for i in range(1, sh.nrows):
    y.append(sh.cell_value(i,1)) 
    x.append(sh.cell_value(i,0))
while(abs(t0-t0old)>pow(10,-5)):
    t0old=t0
    t1old=t1
    val0 = 0.0
    val1 = 0.0
    cost = 0.0
    for i in range(1, sh.nrows):
        val0 += sh.cell_value(i,1)-t0old-t1old*sh.cell_value(i,0)
        val1 +=(sh.cell_value(i,1)-t0old-t1old*sh.cell_value(i,0))*sh.cell_value(i,0)
        cost += pow((sh.cell_value(i,1)-t0old-t1old*sh.cell_value(i,0)),2)
    cost=0.5*cost 
    k+=1
    n+=1
    t0=t0old+2*alpha*val0
    hello.append(cost)
    t1=t1old+2*alpha*val1
arr=np.arange(1,n,1)
#print(hello)
Y=[]
for l in range(0,sh.nrows-1):
    Y.append(x[l]*t1+t0)
print(t0,t1)
#for i in range()
#Y.append(t0+t1*x)
plt.plot(arr,hello)
plt.show()
plt.scatter(x, y)
plt.plot(x, Y,'r')
plt.show()    



    







