import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score

mntrain = MNIST("./MNIST")
mntrain.load_training()
Xtrain = np.asarray(mntrain.train_images) / 255.0
ytrain = np.asarray(mntrain.train_labels.tolist())

mntest = MNIST("./MNIST")
mntest.load_testing()
Xtest = np.asarray(mntest.test_images) / 255.0
ytest = np.asarray(mntest.test_labels.tolist())

# train
logreg = linear_model.LogisticRegression(C=1e5,
		 solver = "lbfgs", multi_class = "multinomial")
logreg.fit(Xtrain, ytrain)

# test
y_pred = logreg.predict(Xtest)
print "Accuracy: %.2f %%" %(100 * accuracy_score(ytest, y_pred.tolist()))