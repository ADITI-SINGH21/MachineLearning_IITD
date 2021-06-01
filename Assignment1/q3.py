import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_excel('q2train.xlsx')
test = pd.read_excel('q2test.xlsx')
x_train = pd.concat([train['Aptitude'],train['Verbal']],axis =1)
y_train = train['Label']
x_l = np.array(x_train)
y_l = np.array(y_train)
norm1 = np.linalg.norm(x_l)
xl_norm = x_l/norm1
def sigmoid(z):
    Gz = 1.0 / (1.0 + np.exp(-z))
    return Gz
def costFunction(x,y,theta):
    h = sigmoid(x.dot(theta))
    J = (-y).T.dot(np.log(h))-((1-y).T.dot(np.log(1-h)))
    J/=len(y)
    return J
def gradFunction(x,y,theta):
    h = sigmoid(x.dot(theta))
    grad = (h-y.reshape((len(y),1))).T.dot(x)
    grad/=len(y)
    return grad
def LogisticRegression(x,y,alpha):
    theta = np.zeros((3,1))
    final = [0,0]
    cost = 0.0
    temp = 1.0
    J =[]
    X = np.hstack((np.ones((x.shape[0],1)),x))
    while(abs(cost-temp)>10**(-6)):
        temp = cost
        cost = costFunction(X,y,theta)
        grad = gradFunction(X,y,theta)
        theta-=alpha*grad.T 
        J.append(cost)
    final[0]=theta
    final[1]=J
    return final
answer=LogisticRegression(xl_norm,y_l,0.01)
print('The calculated theta values for alpha=0.01')
print(answer[0])
