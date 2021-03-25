# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# loading train data
train_data = pd.read_csv('sample_data/mnist_train_small.csv')
x_train = np.array(train_data.iloc[:,1:785])
y_train = np.array(train_data.iloc[:,0]).reshape(-1,1)
m = len(y_train)
ones = np.full((m,1),1)
# adding ones column to x_train
x_train = np.concatenate((ones,x_train),axis = 1)
# axis = 1 for adding ones as column
# converting arrays into matrices
x_train = np.asmatrix(x_train)
y_train = np.asmatrix(y_train)
# determining no of features in "x"
(row,col) = np.shape(x_train)

# initialising parameters
alpha = 0.0001
num_iters = 10000
#creating theta matrix
theta = np.zeros(col).reshape(-1,1)
j = len(theta)

# cost function
def cost_function(x,y,theta):
  squareoferror = np.power(((x*theta)-y),2)  #here x*theta is the predicted value
  return np.sum(squareoferror)/(2*m)  #sum of square errors divided by 2*m

# computing cost for theta=zero
cost = cost_function(x_train,y_train,theta)
print(cost)

# defining gradient descent
def gradientDescent(x, y, theta, alpha, num_iters):
    for i in range(num_iters):
        a = range(j)   
        theta[a] = theta[a] - (alpha/len(x)) * np.sum((np.transpose(x))*(x * theta - y))
        cost = cost_function(x,y,theta)
        return (theta, cost)
        
    

print(gradientDescent(x_train,y_train,theta,alpha, num_iters))

