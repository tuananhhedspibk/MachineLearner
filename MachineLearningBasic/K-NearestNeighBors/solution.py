import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score	# evaluation method

# Load data and display some examples data
# Three class labels: 0, 1, 2
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print "Number of classes: %d" %len(np.unique(iris_y))
print "Number of data points: %d" %len(iris_y)

X0 = iris_X[iris_y == 0, :]						# get all data points with class = 0
print "\nSamples from class 0:\t", X0[:5, :]	# get first 5 points

X1 = iris_X[iris_y == 1, :]						# get all data points with class = 1
print "\nSamples from class 1:\t", X1[:5, :]	# get first 5 points

X2 = iris_X[iris_y == 2, :]						# get all data points with class = 2
print "\nSamples from class 2:\t", X2[:5, :]	# get first 5 points

# Spilit data set to: training set & test set
X_train, X_test, y_train, y_test = train_test_split(
	iris_X, iris_y, test_size = 50)		# test set 's size = 50, training set 's size = 100

print "Training size: %d" %len(y_train)
print "Test size: %d" %len(y_test)

# K = 1
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)	# Use norm = 2 to calculate distance between two data points
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print "Print results for 20 test data points:"
print "Predicted labels: ", y_pred[20:40]
print "Ground truth	   : ", y_test[20:40]

# evaluation method for KNN algorithm
print "Accuracy of 1NN: %.2f %%" %(100 * accuracy_score(y_test, y_pred))

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print "Accuracy of 10NN: %.2f %%" %(100 * accuracy_score(y_test, y_pred))

# Solution with weights
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = "distance")
# weight = 1 / distance

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print "Accuracy of 10NN (1/distance weights): %.2f %%" %(100 * accuracy_score(y_test, y_pred))

def myweight(distances):
	sigma2 = .5	# we can change this value
	return np.exp(-distances ** 2 / sigma2)

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = myweight)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print "Accuracy of 10NN (customized weights): %.2f %%" %(100 * accuracy_score(y_test, y_pred))