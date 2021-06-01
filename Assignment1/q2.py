import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
q1 = pd.read_excel('q1.xlsx')
x = np.array(q1['Population in 10,000\'s'])
y = np.array(q1['Profit In Lakhs(Rs)'])
def batchLMS(x,y,alpha):
    final = [0,0,0,0]
    theta_0 = 1.0
    theta_1 = 1.0
    temp = 0.1
    J = []
    cost = 0.0
    n = float(len(y))
    iter = 1
    while(abs(temp-cost)>10**(-7)):
        temp = cost
        y_calc = theta_1*x+theta_0
        cost = sum((y-theta_0-theta_1*x)**2)
        cost*=0.5
        J.append(cost)
        D_1 = (-2/n) * sum(x * (y - y_calc))
        D_0 = (-2/n) * sum((y - y_calc))
        theta_0=theta_0-alpha*D_0
        theta_1=theta_1-alpha*D_1
        iter+=1
    final[0]=theta_0
    final[1]=theta_1
    final[2]=J
    final[3]=iter
    return final
alpha1 = 0.001
arr=batchLMS(x,y,alpha1)
t0 = arr[0]
t1 = arr[1]
print("The calculated Theta values from Batch LMS are:")
print(t0,t1)
y_batch = t1*x+t0
plt.scatter(x,y)
plt.plot(x,y_batch,'r')
plt.title('Batch LMS')
plt.show()
J = arr[2]
num_iter = arr[3]
iterations=np.arange(1,num_iter,1)
plt.plot(iterations,J)
plt.title('Cost vs Iterations for Batch LMS')
plt.show()
def stochasticLMS(x,y,alpha):
    final = [0,0,0,0]
    theta_0 = 1.0
    theta_1 = 1.0
    temp = 0.1
    J = []
    cost = 0.0
    n = float(len(y))
    iter = 1
    while(abs(temp-cost)>10**(-7)):
        temp = cost
        y_calc = theta_1*x+theta_0
        for i in range(int(n)):
            D_0 = (-2/n)*(y[i]-y_calc[i])
            D_1 = (-2/n)*(x[i]*(y[i]-y_calc[i]))
            theta_0-=alpha*D_0
            theta_1-=alpha*D_1  
        cost = sum((y-theta_0-theta_1*x)**2)
        cost*=0.5
        J.append(cost)
        iter+=1
    final[0]=theta_0
    final[1]=theta_1
    final[2]=J
    final[3]=iter
    return final
alpha2 = 0.001
arr2 = stochasticLMS(x,y,alpha2)
t0_s=arr2[0]
t1_s= arr2[1]
print("The calculated Theta values from Stochastic LMS are:")
print(t0_s,t1_s)
y_stochastic = t1_s*x+t0_s
plt.scatter(x,y)
plt.plot(x,y_stochastic,'r')
plt.title('Stochastic LMS')
plt.show()
J_s = arr2[2]
num_iter_s = arr2[3]
iterations2=np.arange(1,num_iter_s,1)
plt.plot(iterations2,J_s)
plt.title('Cost vs Iterations for Stochastic LMS')
plt.show()
def LeastSquare(x,y):
    X = np.vstack([np.ones(len(y)),x])
    X_T = X.transpose()
    prod1 = np.dot(X,X_T)
    prod2 = np.dot(X,y)
    inv = np.linalg.inv(prod1)
    theta = np.dot(inv,prod2)
    return theta
theta_LS = LeastSquare(x,y)
print("The calculated Theta values from Least Square closed form are:")
print(theta_LS)
y_leastsquare = theta_LS[1]*x+theta_LS[0]
plt.scatter(x,y)
plt.plot(x,y_leastsquare,'r')
plt.title('Least Square Closed form')
plt.show()
X_clf = x.reshape(-1,1)
Y_clf = y.reshape(-1,1)
ElasticNetClf = ElasticNet(alpha=0.1)
ElasticNetClf.fit(X_clf,Y_clf)
print("The calculated Theta values from ElasticNet Regression are:")
print(ElasticNetClf.intercept_,ElasticNetClf.coef_)
y_elasticnet = ElasticNetClf.predict(X_clf)
plt.scatter(x,y)
plt.plot(x,y_elasticnet,'r')
plt.title('Elastic Net Regression')
plt.show()
LassoClf = Lasso(alpha=0.1)
LassoClf.fit(X_clf,Y_clf)
print("The calculated Theta values from Lasso Regression are:")
print(LassoClf.intercept_,LassoClf.coef_)
y_lasso = LassoClf.predict(X_clf)
plt.scatter(x,y)
plt.plot(x,y_lasso,'r')
plt.title('Lasso Regression')
plt.show()
RidgeClf = Ridge(alpha=0.1)
RidgeClf.fit(X_clf,Y_clf)
print("The calculated Theta values from Ridge Regression are:")
print(RidgeClf.intercept_,RidgeClf.coef_)
y_ridge = RidgeClf.predict(X_clf)
plt.scatter(x,y)
plt.plot(x,y_ridge,'r')
plt.title('Ridge Regression')
plt.show()