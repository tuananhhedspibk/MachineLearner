import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target

print "Number of classes: %d" %len(np.unique(iris_Y))
print "Number of data points: %d" %len(iris_Y)

X0 = iris_X[iris_Y == 0, :]
print "\nSample from class 0:\n", X0[:5, :]

X1 = iris_X[iris_Y == 1, :]
print "\nSample from class 1:\n", X1[:5, :]

X2 = iris_X[iris_Y == 2, :]
print "\nSample from class 2:\n", X2[:5, :]

from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
	iris_X, iris_Y, test_size = 50)

print "Training size: %d" %len(Y_train)
print "Test size: %d" %len(Y_test)

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

print "Print results for 20 test data points:"
print "Predicted labels: ", y_pred[20 : 40]
print "Ground truth: ", Y_test[20 : 40]

# evaluation method
from sklearn.metrics import accuracy_score
print "Accuracy of 10NN: %.2f %%" %(100 * accuracy_score(Y_test, y_pred))

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = 'distance')
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

print "Accuracy of 10NN (1 / distance weights): %.2f %%" %(100 * accuracy_score(Y_test, y_pred))

def myweight(distance):
	sigma2 = .5		# we can change this number
	return np.exp(- distance ** 2 / sigma2)

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = myweight)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

print "Accuracy of 10NN (customized weights): %.2f %%" %(100 * accuracy_score(Y_test, y_pred))