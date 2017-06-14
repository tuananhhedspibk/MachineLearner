import numpy as np
import random
import data_helpers
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import pickle 

def load_train_data(file_path):
	data, labels = data_helpers.load_data(file_path)
	shuffle_indicies = range(len(data))
	random.shuffle(shuffle_indicies)
	shuffle_data = np.array(data)[shuffle_indicies]
	shuffle_labels = np.array(labels)[shuffle_indicies]
	split_index = int(0.8 * len(data))
	data_train = shuffle_data[:split_index]
	labels_train = shuffle_labels[:split_index]
	data_test = shuffle_data[split_index:]
	labels_test = shuffle_labels[split_index:]
	return data_train, labels_train, data_test, labels_test

def train(file_path):
	# gnb = GaussianNB()
	gnb = LogisticRegression()
	train_data, train_labels, _, _ = load_train_data(file_path)
	model = gnb.fit(train_data, train_labels)
	fw = open("model.pkl", "wb")
	pickle.dump(model, fw)
	fw.close()
	return model

def test(file_path):
	_, _, test_data, test_labels = load_train_data(file_path)
	fr = open("model.pkl", "rb")
	model = pickle.load(fr)
	fr.close()
	predict_labels = model.predict(test_data)
	total = 0
	count = 0
	for id_, labels_ in enumerate(predict_labels):
		if labels_ == test_labels[id_]:
			count += 1
		total += 1
	accuracy = float(count) / total
	print(accuracy)

def main():
	train("data.txt")
	test("data.txt")

if __name__ == "__main__":
	main()