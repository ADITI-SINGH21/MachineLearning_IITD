
import xlrd
import matplotlib.pyplot as plt
import numpy as np
book = xlrd.open_workbook("q1.xlsx")
sh = book.sheet_by_index(0)
y=[]
x=[]
for i in range(1, sh.nrows):
    y.append(sh.cell_value(i,1)) 
    x.append(sh.cell_value(i,0))
A=np.ones(sh.nrows-1)
X=np.asarray(x)
X = np.vstack([A, X])
Y=np.asarray(y)
transpose= X.transpose()
multi=np.dot(X,transpose)
print(multi)
multi2=np.dot(X,Y)
print(multi2)
multi3=np.linalg.inv(multi)
multi4=np.dot(multi3,multi2)
print(multi4)

#theta=np.dot(np.linalg.inv(np.dot(X,X.transpose())),X.transpose(),Y)


