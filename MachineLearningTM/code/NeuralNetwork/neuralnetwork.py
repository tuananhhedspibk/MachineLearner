from sklearn import datasets
from sklearn.neural_network import MLPClassifier
import data_helper
import random
import numpy as np

def load_train_data(split_percent = 0.9):
	iris = datasets.load_iris()
	feature_data = iris.data
	label_data = iris.target
	shuffle_indices = range(len(feature_data))
	random.shuffle(shuffle_indices)
	shuffled_feature_data = np.array(feature_data)[shuffle_indices]
	shuffled_label_data = np.array(label_data)[shuffle_indices]
	split_index = int(split_percent * len(shuffled_feature_data))
	feature_train = shuffled_feature_data[:split_index]
	label_train = shuffled_label_data[:split_index]
	feature_test = shuffled_feature_data[split_index:]
	label_test = shuffled_label_data[split_index:]
	return feature_train, label_train, feature_test, label_test

def convert_labels(labels):
	new_labels = []
	for label in labels:
		if label == 0:
			new_labels.append([1, 0, 0])
		elif label == 1:
			new_labels.append([0, 1, 0])
		else:
			new_labels.append([0, 0, 1])
	return new_labels

def train(train_data, train_labels):
	mlp = MLPClassifier(solver='sgd', alpha=1e-5,
                    hidden_layer_sizes=[10] , random_state=1)
	mlp.fit(train_data, train_labels)
	return mlp

def test(test_data, test_labels, model):
	predict_labels = model.predict(test_data)
	total = 0
	count = 0
	for id_, label in enumerate(predict_labels):
		if max(label) == max(test_labels[id_]):
			count += 1
		total += 1
	print("accuracy " + str(float(count) / total))

def main():
	feature_train, label_train_, feature_test, label_test_ = load_train_data()
	label_train = convert_labels(label_train_)
	label_test = convert_labels(label_test_)

	feature_train = np.array(feature_train)
	label_train = np.array(label_train)
	feature_test = np.array(feature_test)
	label_test = np.array(label_test)

	mlp = train(feature_train, label_train)
	test(feature_test, label_test, mlp)

if __name__ == "__main__":
	main()