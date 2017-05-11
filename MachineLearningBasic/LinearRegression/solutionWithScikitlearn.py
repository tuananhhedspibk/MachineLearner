from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt

# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T

# weight (kg)
y = np.array(([[49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]])).T

# Building Xbar
one = np.ones((X.shape[0], 1))	# array with all element is 1
Xbar = np.concatenate((one, X), axis = 1) #Xbar = [one , X]

# Calculating weights of the fitting line
A = np.dot(Xbar.T, Xbar)	# Mutiply matrix (A: 2x2)
b = np.dot(Xbar.T, y)		# b: 2x1
w = np.dot(np.linalg.pinv(A), b)	# w: 2x1 

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept = False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)

# Compare two results
print ("Solution found by scikit-learn : ", regr.coef_)
print ("Solution found by (5): ", w.T)