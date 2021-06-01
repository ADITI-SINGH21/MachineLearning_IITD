import xlrd
import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.linear_model import Ridge

book = xlrd.open_workbook("q1.xlsx")
sh = book.sheet_by_index(0)
y=[]
X=[]
k=0
for i in range(1, sh.nrows):
    y.append(sh.cell_value(i,1)) 
    X.append(sh.cell_value(i,0))

X = np.array(X).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

print(X.shape)
print(y.shape)

clf = Ridge(alpha=1.0)
clf.fit(X, y)

print(clf.coef_)
print(clf.intercept_)