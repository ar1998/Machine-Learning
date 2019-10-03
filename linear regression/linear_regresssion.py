import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# reading the data
data = pd.read_csv('USA_Housing.csv')
del data['Address']

data = (data - data.mean())/data.std()

X = data.iloc[:, 0:5]
ones = np.ones([X.shape[0], 1])

# converting DataFrame into NUMPY ARRAY
X = np.concatenate((ones, X), axis=1)
y = data.iloc[:, 5:6].values
# initialising theta to zero matrix
theta = np.zeros([1, 6])

def computeCost(X, y, theta):
    summation = np.power(((X @ theta.T) - y), 2)
    return np.sum(summation)/(2*len(X))

# computing gradient descent
def gradientDescent(X, y, theta, iters, alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * ((X @ theta.T) - y), axis=0)
        cost[i] = computeCost(X, y, theta)
        if i % 1000 == 0:
            print(cost[i])
    return theta, cost


# set parameters
alpha = .001
iters = 10000

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

THETA, cost = gradientDescent(X_train, y_train, theta, iters, alpha)
print(THETA)

finalCost = computeCost(X, y, THETA)
print(finalCost)

for i in range(X_test.shape[0]):
    y_predict = X_test @ THETA.T\

fig, axes = plt.subplots()
axes.scatter(y_predict, y_test)
